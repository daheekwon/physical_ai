"""Frozen Cube final evaluation; development validation never fits rules or probes."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve, write_json, sha256
from zj_wam.cube_readouts import target, predict, capture_errors, effects, future_features
from scripts.analyze_followup_cube import load as audit_bank, choose
from scripts.analyze_followup_cube_probes import captures, signature as probe_signature
from scripts.analyze_followup_response_selection import mean_ci

BASE = resolve('artifacts/followup_v1/cube')
VERIFIERS = ['controller_displacement', 'proprio_contact', 'lewm_global']
DIRECTIONS = ['random0', 'random1', 'lewm_low', 'lewm_high']


def read(path):
    return json.loads(path.read_text())


def banks_and_rules(f):
    """Action-feasibility common support only; no physical-label filtering."""
    rules = {'common': read(BASE/'calibrated_settings.json')['rules'],
             'non_wm_proposals': read(BASE/'proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules']}
    banks = {'common': f, 'non_wm_proposals': f[~f.method.isin(['lewm_low', 'lewm_high'])].copy()}
    d = f[f.method.isin(DIRECTIONS)]
    keys = ['state_id', 'radius', 'sign']
    counts = d.groupby(keys).method.nunique()
    cells = counts[counts == len(DIRECTIONS)].reset_index()[keys]
    matched = d.merge(cells, on=keys, validate='many_to_one')
    shared = f[f.method.isin(['nominal', 'past_pd_optimizer'])]
    spread = matched.groupby(keys).norm.agg(['min', 'max'])
    error = float((spread['max']-spread['min']).max()) if len(spread) else 0.
    assert error < 1e-10
    settings = read(BASE/'matched_proposals/analysis.json')['settings']
    reference = None
    for direction in DIRECTIONS:
        name = 'matched_'+direction
        banks[name] = pd.concat([shared, matched[matched.method == direction]], ignore_index=True)
        rules[name] = settings[direction]['rules']
        sizes = banks[name].groupby('state_id').size()
        if reference is None:
            reference = sizes
        else:
            pd.testing.assert_series_equal(sizes, reference)
    return banks, rules, {'all_states': int(f.state_id.nunique()),
        'states_with_matched_direction': int(matched.state_id.nunique()),
        'matched_radius_sign_cells': len(cells), 'shared_candidates': len(shared),
        'per_matched_bank_candidates': int(reference.sum()), 'matched_norm_max_error': error}


def decompose(g, selected, method, rule):
    rows = []
    score = {'controller_displacement': 'controller_score', 'proprio_contact': 'contact_score', 'lewm_global': 'lewm_score'}[method]
    for state, candidates in g.groupby('state_id'):
        nominal = candidates[candidates.candidate == 0].iloc[0]
        useful = candidates.active & (candidates.object_error <= 1) & (candidates.jerk_gain >= .1)
        accepted = (candidates.active & (candidates.norm <= rule.get('radius', .4)+1e-6)
                    & (candidates[score] <= rule.get('tau', -np.inf)))
        if rule.get('disabled', False):
            accepted[:] = False
        selectable = accepted & (candidates.predicted_jerk_cost < nominal.predicted_jerk_cost-1e-10)
        chosen = selected[selected.state_id == state].iloc[0]
        rows.append({'state_id': state, 'initial_pose_family': nominal.initial_pose_family,
                     'candidate_exists': bool(useful.any()), 'useful_approved': bool((useful & accepted).any()),
                     'useful_rankable': bool((useful & selectable).any()), 'selected_useful': bool(chosen.joint),
                     'selected_unsafe': bool(chosen.active and chosen.object_error > 1),
                     'candidate_coverage': float(accepted[candidates.active].mean()) if candidates.active.any() else 0.,
                     'candidate_risk': float(candidates.loc[accepted, 'violation'].mean()) if accepted.any() else np.nan})
    return pd.DataFrame(rows)


def selection_analysis(f, cfg, dest, validation):
    banks, rules, support = banks_and_rules(f)
    records, decomposition = [], []
    for name, bank in banks.items():
        methods = ['nominal', 'pd_jerk_only', *VERIFIERS, 'oracle'] if name == 'common' else [*VERIFIERS, 'oracle']
        for method in methods:
            rule = rules[name].get(method, {})
            selected = choose(bank, method, rule)
            selected['bank'] = name
            records.append(selected)
            if method in VERIFIERS:
                detail = decompose(bank, selected, method, rule)
                detail['bank'], detail['method'] = name, method
                decomposition.append(detail)
    result = pd.concat(records, ignore_index=True)
    details = pd.concat(decomposition, ignore_index=True)
    summary = []
    for (bank, method), g in result.groupby(['bank', 'method']):
        for phase in ['all', *sorted(g.phase.unique())]:
            x = g if phase == 'all' else g[g.phase == phase]
            active = x[x.active]
            contact = active[active.phase.isin(['grasp', 'transport', 'release'])]
            summary.append({'bank': bank, 'method': method, 'phase': phase, 'states': len(x),
                **{k: mean_ci(x, k) for k in ['joint', 'active', 'jerk_gain', 'drop_relative_to_nominal']},
                'risk': mean_ci(active, 'violation'), 'contact_risk': mean_ci(contact, 'violation')})
    paired = []
    comparisons = [('common', 'lewm_global', 'common', 'controller_displacement'),
                   ('common', 'proprio_contact', 'common', 'controller_displacement')]
    comparisons += [('common', m, 'non_wm_proposals', m) for m in [*VERIFIERS, 'oracle']]
    comparisons += [('matched_'+d, m, 'matched_'+b, m) for d in ['lewm_low', 'lewm_high'] for b in ['random0', 'random1'] for m in [*VERIFIERS, 'oracle']]
    for bank, method, basebank, basemethod in comparisons:
        a = result[(result.bank == bank) & (result.method == method)]
        b = result[(result.bank == basebank) & (result.method == basemethod)]
        x = a.merge(b, on=['state_id', 'initial_pose_family'], suffixes=('', '_base'), validate='one_to_one')
        assert len(x) == f.state_id.nunique()
        x['delta_U'] = x.joint.astype(float)-x.joint_base.astype(float)
        x['delta_gain'] = x.jerk_gain-x.jerk_gain_base
        paired.append({'bank': bank, 'method': method, 'baseline_bank': basebank, 'baseline': basemethod,
                       'U_difference': mean_ci(x, 'delta_U'), 'jerk_gain_difference': mean_ci(x, 'delta_gain')})
    reproduced = 0
    if validation:
        parents = {'common': BASE/'evaluation.csv', 'non_wm_proposals': BASE/'proposal_ablation/evaluation.csv',
                   **{'matched_'+d: BASE/'matched_proposals/evaluation.csv' for d in DIRECTIONS}}
        for name, path in parents.items():
            parent = pd.read_csv(path)
            if name == 'non_wm_proposals': parent = parent[parent.bank == name]
            if name.startswith('matched_'): parent = parent[parent.bank == name.removeprefix('matched_')]
            current = result[result.bank == name]
            x = current.merge(parent, on=['split', 'state_id', 'method'], suffixes=('', '_parent'), validate='one_to_one')
            assert len(x) == len(current)
            np.testing.assert_array_equal(x.candidate, x.candidate_parent)
            np.testing.assert_allclose(x.object_error, x.object_error_parent, atol=1e-7, rtol=0)
            np.testing.assert_allclose(x.jerk_gain, x.jerk_gain_parent, atol=1e-10, rtol=0)
            reproduced += len(x)
    dsummary = []
    for (bank, method), x in details.groupby(['bank', 'method']):
        dsummary.append({'bank': bank, 'method': method, **{k: mean_ci(x.dropna(subset=[k]), k) for k in ['candidate_exists', 'useful_approved', 'useful_rankable', 'selected_useful', 'selected_unsafe', 'candidate_coverage', 'candidate_risk']}})
    result.to_csv(dest/'evaluation.csv', index=False)
    details.to_csv(dest/'selection_decomposition.csv', index=False)
    return {'summary': summary, 'paired': paired, 'support': support, 'decomposition': dsummary,
            'parent_reproduction_rows': reproduced, 'rules': rules}


def probe_analysis(folder, manifest, cfg, dest, partial):
    source = BASE/'probes_v2'
    selection = read(source/'selection.json')
    assert selection['bank_signature'] == manifest['signature'] and selection['probe_signature'] == probe_signature()
    thresholds = read(source/'analysis.json')['resolution_thresholds']
    models = {}
    for name, spec in selection['readouts'].items():
        path = source/(name+'.joblib')
        assert sha256(path) == spec['model_sha256']
        models[name] = joblib.load(path)
    capture_rows, records = [], []
    for representation in ['frame', 'history']:
        x, y, meta = captures(folder, manifest, representation, partial)
        for name, model in models.items():
            if selection['readouts'][name]['representation'] != representation: continue
            r = meta.copy(); r['readout'] = name
            for key, value in capture_errors(predict(model, x), y).items(): r[key] = value
            capture_rows.append(r)
    max_frame_error = 0.
    for path in sorted(folder.glob('e*.npz')):
        meta = read(path.with_suffix('.json'))
        raw = pd.read_csv(path.with_suffix('.csv'))
        with np.load(path) as a:
            endpoint = cfg['horizon']*cfg['substeps']-1
            truth = target(a['candidate_object_pose'][:, endpoint], a['candidate_object_velocity'][:, endpoint], a['candidate_eef_position'][:, endpoint])
            active = np.flatnonzero(raw.norm.to_numpy() > 1e-6)
            wrong = np.roll(np.arange(len(truth)), 1)
            q = a['candidate_object_pose'][0, :, 3:]
            r = Rotation.from_quat(q[:, [1, 2, 3, 0]])
            fd = (r[1:]*r[:-1].inv()).as_rotvec()/cfg['physics_dt']
            v = r[1:].apply(a['candidate_object_velocity'][0, 1:, 3:])
            error = float(np.max(np.abs(fd-v))); assert error < 1e-6
            max_frame_error = max(max_frame_error, error)
            for name, model in models.items():
                spec = selection['readouts'][name]
                real = predict(model, future_features(a['history_embedding'], a['actual_endpoint_latent'], spec['representation']))
                imagined = predict(model, future_features(a['history_embedding'], a['predicted_latent'], spec['representation']))
                rp, tp, rr, tr = effects(real, truth, spec['kind'] != 'ridge_raw')
                ip, _, ir, _ = effects(imagined, truth, spec['kind'] != 'ridge_raw')
                tmag = np.linalg.norm(tp, axis=1)
                for control, dp, dr in [('real_endpoint', rp, rr), ('predicted', ip, ir), ('zero', np.zeros_like(tp), Rotation.identity()), ('wrong_action', ip[wrong], ir[wrong])]:
                    pmag = np.linalg.norm(dp, axis=1)
                    cosine = np.sum(dp*tp, axis=1)/np.maximum(pmag*tmag, 1e-15)
                    frame = pd.DataFrame({'candidate': active, 'position_effect_error_m': np.linalg.norm(dp-tp, axis=1)[active],
                        'rotation_effect_error_rad': (dr*tr.inv()).magnitude()[active], 'true_position_effect_m': tmag[active],
                        'predicted_position_effect_m': pmag[active], 'true_rotation_effect_rad': tr.magnitude()[active],
                        'direction_cosine': cosine[active]})
                    for key, value in [('state_id', meta['state_id']), ('initial_pose_family', meta['episode']), ('phase', meta['phase']), ('readout', name), ('control', control), ('split', folder.name)]: frame[key] = value
                    frame['position_resolved'] = frame.true_position_effect_m > thresholds[name]['position_m']
                    frame['rotation_resolved'] = frame.true_rotation_effect_rad > thresholds[name]['rotation_rad']
                    frame['prediction_has_direction'] = frame.predicted_position_effect_m > 1e-8
                    records.append(frame)
    f = pd.concat(records, ignore_index=True); capture = pd.concat(capture_rows, ignore_index=True)
    capture_summary = []
    for name, g in capture.groupby('readout'):
        capture_summary.append({'readout': name, **{k: mean_ci(g, k) for k in ['position_m', 'rotation_rad', 'rotation_mod_cube_symmetry_rad', 'linear_velocity_m_s', 'angular_velocity_rad_s', 'eef_position_m']}})
    summary = []
    for (name, control), g in f.groupby(['readout', 'control']):
        summary.append({'readout': name, 'control': control, 'rows': len(g),
            **{k: mean_ci(g, k) for k in ['position_effect_error_m', 'rotation_effect_error_rad', 'position_resolved', 'rotation_resolved']},
            'direction_available_when_resolved': mean_ci(g[g.position_resolved], 'prediction_has_direction'),
            'resolved_position_direction': mean_ci(g[g.position_resolved & g.prediction_has_direction], 'direction_cosine')})
    reproduction = 0
    if folder.name != 'test':
        old = pd.read_csv(source/'effects.csv'); old = old[old.split == folder.name]
        merged = f.merge(old, on=['split', 'state_id', 'candidate', 'readout', 'control'], suffixes=('', '_parent'), validate='one_to_one')
        assert len(merged) == len(f)
        for key in ['position_effect_error_m', 'rotation_effect_error_rad', 'true_position_effect_m', 'predicted_position_effect_m', 'direction_cosine']:
            np.testing.assert_allclose(merged[key], merged[key+'_parent'], atol=1e-10, rtol=1e-10)
        reproduction = len(merged)
    f.to_csv(dest/'effects.csv', index=False); capture.to_csv(dest/'capture.csv', index=False)
    return {'capture': capture_summary, 'effects': summary, 'resolution_thresholds': thresholds,
            'effect_rows': len(f), 'capture_rows': len(capture), 'max_velocity_frame_error': max_frame_error,
            'parent_effect_reproduction_rows': reproduction, 'fitted_on_final_data': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['test', 'development', 'calibration'], default='test')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cfg = read(resolve('configs/followup_cube.json'))
    final = args.split == 'test'
    if final:
        assert args.limit is None, 'No partial final result'
        from scripts.run_followup_cube_final import verify, LOCK
        lock = verify(); manifest = lock['manifest']
        complete = read(BASE/'final_execution_complete.json')
        assert complete['lock_sha256'] == sha256(LOCK) and complete['states'] == lock['planned_states']
    else:
        manifest = read(BASE/'manifest.json'); lock = None
    dest = BASE/('final_analysis' if final else 'final_validation_'+args.split)
    dest.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='zj_cube_final_validation_') as temporary:
        folder = BASE/args.split
        if args.limit is not None:
            assert args.limit > 0
            selected = sorted(folder.glob('e*.json'))[:args.limit]
            folder = Path(temporary)/args.split; folder.mkdir()
            for path in selected:
                for suffix in ['.json', '.csv', '.npz']: (folder/path.with_suffix(suffix).name).symlink_to(path.with_suffix(suffix))
        files = sorted(folder.glob('e*.json'))
        identities = []
        for path in files:
            m = read(path); ep = m['episode']; phase = m['phase']
            assert ep in manifest['splits'][args.split] and m['timestep'] == manifest['phases'][str(ep)][phase]
            assert m['split'] == args.split and not m['future_physics_used_for_selection']
            identities.append((ep, phase))
        assert len(set(identities)) == len(files)
        if args.limit is None:
            expected = {(ep, phase) for ep in manifest['splits'][args.split] for phase in manifest['phases'][str(ep)]}
            assert set(identities) == expected
        f, states, audit = audit_bank(folder, cfg, manifest, partial=args.limit is not None)
        states.to_csv(dest/'state_diagnostics.csv', index=False)
        f.to_csv(dest/'audited_candidates.csv', index=False)
        selections = selection_analysis(f, cfg, dest, not final)
        probes = probe_analysis(folder, manifest, cfg, dest, args.limit is not None)
        assert probes['effect_rows'] == int(f.active.sum())*8*4
    audit.update({'split': args.split, 'partial': args.limit is not None, 'unique_episode_phase_pairs': len(identities),
                  'source_sha256': sha256(Path(__file__)), 'selector_reproduction_rows': selections['parent_reproduction_rows'],
                  'probe_reproduction_rows': probes['parent_effect_reproduction_rows'], 'no_fit_or_calibration': True})
    result = {'split': args.split, 'partial': args.limit is not None, 'selection': selections, 'probes': probes,
              'scope': 'Fixed 2mm/fullSO3 2deg dense500Hz object trajectory, native continuous polynomial/gripper fixed. Frozen readouts and development resolution. Matched banks control count/actionnorm, not EEF motion/frequency or compute. Symmetry only in diagnostic readout; physical tolerance unchanged. Family bootstrap, upstreamWM exposure separate.'}
    if final:
        verify(); audit['lock_sha256'] = sha256(LOCK); result['lock_sha256'] = sha256(LOCK)
        primary = selections['paired'][0]
        mainrow = next(x for x in selections['summary'] if x['bank'] == 'common' and x['method'] == 'lewm_global' and x['phase'] == 'all')
        lo = primary['U_difference']['ci95'][0]
        feasible = all(mainrow[k]['mean'] is not None and mainrow[k]['mean'] <= cfg['target_risk'] for k in ['risk', 'contact_risk']) and mainrow['active']['mean'] >= cfg['minimum_active_fraction']
        result['primary_decision'] = {'empirical_feasible': feasible, 'positive_U_evidence': bool(lo > 0),
            'practical_U_evidence': bool(lo >= lock['config']['minimum_practical_U_difference']),
            'supported_practical_advantage': bool(feasible and lo >= lock['config']['minimum_practical_U_difference']),
            'limitation': 'No population safety guarantee, no equivalence from nonsignificance; secondary comparisons exploratory.'}
    write_json(dest/'raw_audit.json', audit); write_json(dest/'analysis.json', result)
    print('COMPLETE', args.split, audit['states'], 'states', audit['candidate_rows'], 'candidates', probes['effect_rows'], 'effectrows', flush=True)


if __name__ == '__main__':
    main()
