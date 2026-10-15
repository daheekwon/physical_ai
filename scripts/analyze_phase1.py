"""Trajectory-clustered statistics, calibration-only allowance, held-out evaluation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from zj_wam.io import resolve, write_json, sha256
from zj_wam.metrics import cluster_bootstrap, preservation_mask


def group_name(name):
    if name.startswith('random_'):
        return 'random'
    if name.startswith('low_mix_'):
        return 'low_mix'
    if name.startswith('low_basis_'):
        return 'low_basis'
    return name


def fit_calibration(frame, cfg, manifest_digest):
    if set(frame.split.unique()) != {'calibration'}:
        raise ValueError('Calibration fitter must receive only calibration records')
    rules = {}
    for phase in ['pooled','contact','near_contact','free_motion']:
        f = frame if phase == 'pooled' else frame[frame.interaction_phase == phase]
        best = {'radius':0.,'predicted_change':0.,'accepted_rows':0,'accepted_states':0,
                'observed_violation_rate':None,'selection_score':0.}
        # Only directions proposed for deployment are used to fit allowance.
        f = f[f.group.isin(['low','low_basis','low_mix'])]
        f = f.sort_values('applied_radius')
        if f.state_id.nunique() >= 10:
            for rho in sorted(f.applied_radius.unique()):
                for tau in np.unique(np.quantile(f.predicted_change, np.linspace(0,1,41))):
                    mask = (f.applied_radius <= rho) & (f.predicted_change <= tau)
                    # Calibrate the deployed maximum connected sampled scale,
                    # not a pool dominated by many harmless tiny corrections.
                    connected = mask.groupby([f.state_id,f.direction,f.sign]).cummin()
                    accepted = f[connected].groupby(['state_id','direction','sign'],sort=False).tail(1)
                    if accepted.state_id.nunique() < 10:
                        continue
                    violation = float((~accepted.preserved).groupby(accepted.state_id).mean().mean())
                    contact = accepted[accepted.interaction_phase=='contact']
                    contact_violation = (float((~contact.preserved).groupby(contact.state_id).mean().mean())
                                         if contact.state_id.nunique()>=10 else None)
                    score = float(accepted.applied_radius.sum())
                    if (violation <= cfg['calibration_max_violation_rate']
                        and (contact_violation is None or contact_violation <= cfg['calibration_max_violation_rate'])
                        and score > best['selection_score']):
                        best = {'radius':float(rho),'predicted_change':float(tau),
                                'accepted_rows':len(accepted),'accepted_states':accepted.state_id.nunique(),
                                'observed_violation_rate':violation,'contact_violation_rate':contact_violation,
                                'selection_score':score}
        rules[phase] = best
    return {'version':manifest_digest,'source_split':'calibration','rules':rules,
            'radius_grid':[float(x*np.sqrt(2*cfg['action_block']*cfg['horizon_blocks'])) for x in cfg['relative_radii']],
            'tolerances':cfg['tolerances'],
            'selection_policy':'maximum connected tested scale per state/direction/sign; contact and pooled calibration violation <= configured limit',
            'note':'Empirical calibration rule. No independent probabilistic safety guarantee; unseen mixtures and intervals are not certified.'}


def analyze(out, fit=False):
    manifest = json.loads((out/'manifest.json').read_text())
    cfg = manifest['config']
    csvs = sorted((out/'states').glob('*.csv'))
    frame = pd.concat([pd.read_csv(p) for p in csvs],ignore_index=True)
    frame.to_csv(out/'all_candidates.csv',index=False)
    expected = manifest['states']
    completed = [json.loads(p.read_text()) for p in (out/'states').glob('*.json')]
    inventory = {split:{'expected':sum(s['split']==split for s in expected),
                        'completed':sum(s['split']==split for s in completed)}
                 for split in cfg['split_episodes']}
    f = frame[frame.status=='ok'].copy()
    f['group'] = f.direction.map(group_name)
    f['preserved'] = preservation_mask(f,cfg['tolerances'])
    write_json(out/'completion.json',inventory)
    if fit:
        if inventory['calibration']['completed'] != inventory['calibration']['expected']:
            raise ValueError('Finish every calibration state before fitting')
        if inventory['test']['completed']:
            raise ValueError('Fit and freeze calibration BEFORE examining test outcomes')
        # Exclude source states that cannot reproduce their recorded object pose
        # within the SAME physical tolerances, before examining any test effects.
        cal = f[f.split=='calibration']
        invalid_replay = []
        for sid in cal.state_id.unique():
            with np.load(out/'states'/f'{sid}.npz') as a:
                p_error = np.linalg.norm(a['recorded_state'][2:4]-a['replayed_state'][2:4])
                angle = a['recorded_state'][4]-a['replayed_state'][4]
                r_error = abs(np.arctan2(np.sin(angle),np.cos(angle)))
            if p_error > cfg['tolerances']['position'] or r_error > cfg['tolerances']['rotation']:
                invalid_replay.append(sid)
        cal = cal[~cal.state_id.isin(invalid_replay)]
        artifact = fit_calibration(cal,cfg,sha256(out/'manifest.json'))
        artifact['excluded_calibration_source_replay_states'] = sorted(invalid_replay)
        artifact['calibration_source_states_used'] = int(cal.state_id.nunique())
        artifact['source_replay_limits'] = {k:cfg['tolerances'][k] for k in ['position','rotation']}
        write_json(out/'calibration.json',artifact)
    rng = np.random.default_rng(cfg['seed'])
    stats, paired, correlations = [], [], []
    eval_split = 'test' if inventory['test']['completed'] else 'development'
    evaluate = f[f.split == eval_split].copy()
    if evaluate.empty:
        evaluate = f[f.split == 'calibration'].copy()
        eval_split = 'calibration'
    for phase in ['all','free_motion','near_contact','contact']:
        subset = evaluate if phase=='all' else evaluate[evaluate.interaction_phase==phase]
        for delta in cfg['relative_radii']:
            part = subset[np.isclose(subset.relative_radius,delta)]
            for group in ['low','random','high','low_basis','low_mix']:
                g = part[part.group==group]
                if g.empty:
                    continue
                # Equal state weighting even with multiple random directions/signs.
                per_state = g.groupby(['episode','state_id'])[['position_error','rotation_error','object_error','preserved']].mean().reset_index()
                rec = {'split':eval_split,'phase':phase,'relative_radius':delta,'group':group,
                       'states':len(per_state),'episodes':per_state.episode.nunique(),
                       'preservation_rate':float(per_state.preserved.mean())}
                for metric in ['position_error','rotation_error','object_error']:
                    rec[f'{metric}_mean']=float(per_state[metric].mean())
                    rec[f'{metric}_median']=float(g[metric].median())
                    rec[f'{metric}_mean_ci']=cluster_bootstrap(per_state[metric],per_state.episode,rng,cfg['bootstrap_samples'])
                stats.append(rec)
            # Average random directions within the SAME state and sign first.
            piv = part[part.group.isin(['low','random','high'])].groupby(
                ['episode','state_id','sign','group']).object_error.mean().unstack('group')
            for other in ['random','high']:
                if not {'low',other}.issubset(piv.columns):
                    continue
                x = piv[['low',other]].dropna()
                difference = x['low']-x[other]
                by_state = difference.groupby(level=['episode','state_id']).mean()
                tie = np.isclose(x['low'],x[other],rtol=1e-7,atol=1e-10)
                paired.append({'split':eval_split,'phase':phase,'relative_radius':delta,'comparison':f'low-{other}',
                               'mean_difference':float(by_state.mean()),
                               'mean_difference_ci':cluster_bootstrap(by_state.values,by_state.index.get_level_values('episode'),rng,cfg['bootstrap_samples']),
                               'strict_win_rate':float(((difference<0)&~tie).mean()),'tie_rate':float(tie.mean()),
                               'win_rate_definition':'paired state/sign versus mean random direction' if other=='random' else 'paired state/sign versus high direction'})
            for pred in ['directional_sensitivity','predicted_change']:
                g=part[part.group.isin(['low','random','high'])]
                if len(g)<3 or g.object_error.nunique()<2 or g[pred].nunique()<2:
                    rho=None
                else:
                    rho=float(spearmanr(g[pred],g.object_error).statistic)
                correlations.append({'phase':phase,'relative_radius':delta,'predictor':pred,'spearman':rho,'rows':len(g)})
    write_json(out/'statistics.json',{'evaluation_split':eval_split,'descriptive':stats,'paired':paired,'correlations':correlations})
    # Sampled finite ranges: first failure stops the connected tested prefix;
    # subsequent safe points are separately counted, not silently discarded.
    ranges=[]
    for keys,g in f.groupby(['split','episode','state_id','direction','sign']):
        g=g.sort_values('applied_radius')
        allowed=0.; failed=False; recovered=0
        for row in g.itertuples():
            if not row.preserved:
                failed=True
            elif failed:
                recovered+=1
            else:
                allowed=max(allowed,row.applied_radius)
        ranges.append(dict(zip(['split','episode','state_id','direction','sign'],keys),
                           tested_connected_radius=float(allowed),safe_points_after_failure=recovered))
    pd.DataFrame(ranges).to_csv(out/'finite_ranges.csv',index=False)
    if (out/'calibration.json').exists() and eval_split=='test':
        calibration=json.loads((out/'calibration.json').read_text())
        selected=[]
        for row in evaluate.itertuples():
            # Pooled rule needs NO ground-truth contact phase at inference.
            rule=calibration['rules']['pooled']
            selected.append(row.applied_radius<=rule['radius'] and row.predicted_change<=rule['predicted_change'])
        evaluate['raw_model_accepted']=selected
        evaluate=evaluate.sort_values('applied_radius')
        evaluate['model_accepted']=evaluate.raw_model_accepted.groupby(
            [evaluate.state_id,evaluate.direction,evaluate.sign]).cummin()
        deployed=evaluate[evaluate.model_accepted].groupby(
            ['state_id','direction','sign'],sort=False).tail(1)
        evaluate['deployed_maximum']=evaluate.index.isin(deployed.index)
        rows=[]
        for phase in ['all','contact','near_contact','free_motion']:
            group=evaluate if phase=='all' else evaluate[evaluate.interaction_phase==phase]
            for method in ['low','low_basis','low_mix','random','high']:
                g=group[group.group==method]
                a=g[g.model_accepted]
                per=a.groupby(['episode','state_id']).preserved.mean().reset_index()
                selected_max = g[g.deployed_maximum]
                max_per = selected_max.groupby(['episode','state_id']).preserved.mean().reset_index()
                rows.append({'phase':phase,'group':method,'candidate_rows':len(g),'accepted_rows':len(a),
                             'accepted_states':a.state_id.nunique(),
                             'deployed_rows':len(selected_max),
                             'deployed_preservation_rate':None if selected_max.empty else float(max_per.preserved.mean()),
                             'deployed_preservation_ci':cluster_bootstrap(max_per.preserved,max_per.episode,rng,cfg['bootstrap_samples']) if len(max_per) else [None,None],
                             'deployed_radius_mean':None if selected_max.empty else float(selected_max.applied_radius.mean()),
                             'preservation_rate':None if a.empty else float(per.preserved.mean()),
                             'preservation_ci':cluster_bootstrap(per.preserved,per.episode,rng,cfg['bootstrap_samples']) if len(per) else [None,None]})
        write_json(out/'heldout_allowance.json',{'rules':calibration['rules'],'evaluation':rows,
                   'inference_rule':'pooled only; no simulator contact information used to accept candidates'})
        evaluate.to_csv(out/'heldout_candidates.csv',index=False)
    plots=out/'plots'; plots.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
    primary=evaluate[np.isclose(evaluate.relative_radius,0.1)]
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,metric,label in zip(axes,['position_error','rotation_error'],['Object position error (px)','Object rotation error (rad)']):
        ax.boxplot([primary[primary.group==g][metric] for g in ['low','random','high']],tick_labels=['low','random','high'],showfliers=False)
        ax.set_ylabel(label)
    fig.suptitle(f'{eval_split}: equal-radius perturbations, relative radius 0.10'); fig.tight_layout(); fig.savefig(plots/'01_direction_comparison.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    for name in ['low','random','high']:
        g=primary[primary.group==name]
        ax.scatter(g.predicted_change,g.object_error,s=9,alpha=.35,label=name)
    ax.set(xlabel='Predicted latent L2 change',ylabel='Executed object error'); ax.legend();fig.tight_layout();fig.savefig(plots/'02_prediction_vs_execution.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,phase in zip(axes,['free_motion','near_contact','contact']):
        g=primary[primary.interaction_phase==phase]
        ax.boxplot([g[g.group==k].object_error for k in ['low','random','high']],tick_labels=['low','random','high'],showfliers=False)
        ax.set_title(f'{phase} (n={g.state_id.nunique()})')
    axes[0].set_ylabel('Executed object error');fig.tight_layout();fig.savefig(plots/'03_contact_comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for phase in ['free_motion','near_contact','contact']:
        picked=next((s for s in completed if s['split']==eval_split and s['interaction_phase']==phase),None)
        if picked:
            ax.semilogy(np.arange(len(picked['eigenvalues'])),np.maximum(picked['eigenvalues'],1e-12),label=f'{phase}: {picked["state_id"]}')
    ax.set(xlabel='Ascending eigenvalue index',ylabel='Eigenvalue of JᵀJ');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(plots/'04_spectrum.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for group in ['low','random','high','low_mix']:
        g=evaluate[evaluate.group==group].groupby('relative_radius').preserved.mean()
        ax.plot(g.index,g.values,'o-',label=group)
    ax.set(xlabel='Relative perturbation radius',ylabel='Observed preservation fraction',ylim=(-.03,1.03));ax.legend();fig.tight_layout();fig.savefig(plots/'05_finite_preservation.png');plt.close(fig)
    primary_pairs=[p for p in paired if p['phase']=='contact' and p['comparison']=='low-random' and np.isclose(p['relative_radius'],0.1)]
    gate='UNCERTAIN'
    if primary_pairs:
        ci=primary_pairs[0]['mean_difference_ci']
        if ci[1] is not None and ci[1]<0: gate='PASS for relative contact effect only; check finite preservation separately'
        elif ci[0] is not None and ci[0]>0: gate='FAIL for the primary low versus random contact hypothesis'
    text=[f'# Phase 1 execution summary',f'\nEvaluation split: {eval_split}.',f'\nCompletion: `{json.dumps(inventory)}`.',f'\nPrimary relative-direction result: **{gate}**.',
          '\nThis decision does not certify safety or establish jerk-reduction freedom.',
          '\n## Paired object response results',
          '\nRandom win rates below compare low with the mean of the three random directions within each state/sign; they are not per-random-draw probabilities.',
          '\n| Phase | Relative radius | Comparison | Mean difference | 95% cluster CI | Win rate | Tie rate |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for p in paired:
        if p['relative_radius'] in cfg['primary_relative_radii']:
            text.append(f'| {p["phase"]} | {p["relative_radius"]} | {p["comparison"]} | {p["mean_difference"]:.6g} | {p["mean_difference_ci"]} | {p["strict_win_rate"]:.3f} | {p["tie_rate"]:.3f} |')
    text += ['\n## Interpretation limits',
             '\nThe model representation includes the pusher and object. Counterfactuals use deterministic prefix replay, and recorded-versus-replayed state errors are saved per state. Task score is ground-truth object polygon coverage; the official seven-state distance (including agent) is separately recorded. Test episodes are held out from our calibration, not necessarily from the original pretrained model training dataset. Finite grid tests do not certify a continuous safe region. No polynomial or jerk optimizer was run.',
             '\nDetailed artifacts: statistics.json, finite_ranges.csv, calibration.json (when fitted), heldout_allowance.json (after test), all_candidates.csv and plots/.']
    (out/'SUMMARY.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'completion':inventory,'result':gate,'rows':len(frame)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/phase1');p.add_argument('--fit-calibration',action='store_true')
    a=p.parse_args();analyze(resolve(a.output),a.fit_calibration)
