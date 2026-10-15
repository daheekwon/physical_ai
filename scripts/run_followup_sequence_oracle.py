"""Finite, privileged full-episode search over multiple polynomial corrections."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve, sha256, write_json
from zj_wam.diagnostics import polynomial_basis, jerk_energy
from zj_wam.followup import optimize_controller
from scripts.run_followup_policy_terminal import replay

CONFIG = 'configs/followup_sequence_oracle.json'
read = lambda p: json.loads(p.read_text())


def metrics(initial, trace, coverage, reference, refcoverage, cfg):
    position = np.linalg.norm(trace[:, 2:4] - reference[:, 2:4], axis=1)
    angle = trace[:, 4] - reference[:, 4]
    rotation = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
    loss = np.maximum(0., refcoverage - coverage)
    error = max(position.max()/cfg['position_tolerance'],
                rotation.max()/cfg['rotation_tolerance'],
                loss.max()/cfg['coverage_tolerance'])
    j0 = jerk_energy(np.vstack([initial[:2], reference[:, :2]]), .01)
    j1 = jerk_energy(np.vstack([initial[:2], trace[:, :2]]), .01)
    return dict(object_error=float(error), position_max=float(position.max()),
                rotation_max=float(rotation.max()), coverage_loss=float(loss.max()),
                jerk_gain=float((j0-j1)/j0), jerk=float(j1), nominal_jerk=float(j0))


def process(seed, smoke=False):
    cfg = read(resolve(CONFIG))
    out = resolve(cfg['output'] + ('_smoke' if smoke else ''))
    done = out / f'{seed}.json'
    manifest_hash = sha256(out/'manifest.json')
    if done.exists():
        assert read(done)['manifest_sha256'] == manifest_hash
        return read(done)
    start_time = time.perf_counter()
    with np.load(resolve(cfg['source'])/f'{seed}_nominal.npz') as z:
        nominal = z['actions'].copy()
        stored_physics, stored_coverage = z['physics'].copy(), z['coverage'].copy()
        states, stored_initial = z['states'].copy(), z['initial_state'].copy()
    # Reuse normalization already recorded by the audited terminal experiment.
    with np.load(resolve(f'artifacts/followup_v1/policy_terminal/{seed}_h05.npz')) as z:
        scale = z['scale'].copy()
    initial, reference, refcoverage = replay(seed, nominal)
    np.testing.assert_array_equal(reference, stored_physics)
    np.testing.assert_array_equal(refcoverage, stored_coverage)
    np.testing.assert_array_equal(initial, stored_initial)
    assert nominal.shape == (cfg['actions'], 2)
    offsets = np.arange(cfg['start'], cfg['actions']-cfg['stride']+1, cfg['stride'])
    basis = polynomial_basis(cfg['block'])
    radius = cfg['relative_radius'] * np.sqrt(2*cfg['block'])
    proposal, solver = [], []
    for offset in offsets:
        block = nominal[offset:offset+cfg['block']]
        suffix = nominal[offset+cfg['block']:offset+cfg['stride']]
        selected, info = optimize_controller(block, suffix, states[offset],
                         reference[offset*10-30:offset*10], scale, radius, basis=basis)
        proposal.append(basis.T @ ((selected-block)/scale).reshape(-1))
        solver.append(bool(info['feasible']))
    proposal = np.asarray(proposal)
    shape = proposal.shape
    rng = np.random.default_rng(np.random.SeedSequence([cfg['random_seed'], seed]))
    rows, coefficients, traces, coverages = [], [], [], []
    cache = {}

    def project(z):
        z = np.asarray(z).reshape(shape).copy()
        z *= np.minimum(1., radius/np.maximum(np.linalg.norm(z, axis=1), 1e-15))[:, None]
        for i, offset in enumerate(offsets):
            delta = (basis @ z[i]).reshape(cfg['block'], 2)*scale
            block = nominal[offset:offset+cfg['block']]
            mask = np.abs(delta)>1e-15
            limits = np.where(delta>0, 1.-block, -1.-block)
            fraction = min(1., np.min(limits[mask]/delta[mask], initial=1.))
            z[i] *= max(0., fraction)
        return z

    def evaluate(z, phase, generation=-1):
        z = project(z)
        key = z.tobytes()
        if key in cache:
            return cache[key]
        action = nominal.copy()
        for i, offset in enumerate(offsets):
            action[offset:offset+cfg['block']] += (basis @ z[i]).reshape(cfg['block'], 2)*scale
        assert np.max(np.abs(action)) <= 1.+1e-12
        ini, physics, cov = replay(seed, action)
        np.testing.assert_array_equal(ini, initial)
        np.testing.assert_array_equal(physics[:cfg['start']*10], reference[:cfg['start']*10])
        m = metrics(initial, physics, cov, reference, refcoverage, cfg)
        norm = float(np.linalg.norm(z))
        row = dict(index=len(rows), phase=phase, generation=generation,
                   norm=norm, active_blocks=int(np.sum(np.linalg.norm(z, axis=1)>1e-6)),
                   useful=bool(norm>1e-6 and m['object_error']<=1 and m['jerk_gain']>=cfg['jerk_gain_threshold']), **m)
        rows.append(row); coefficients.append(z)
        traces.append(physics[:, :5]); coverages.append(cov)
        cache[key] = row['index']
        return row['index']

    def rank(index):
        r = rows[index]
        return (0, -r['jerk_gain']) if r['object_error']<=1 else (1, r['object_error'], -r['jerk_gain'])

    zero = evaluate(np.zeros(shape), 'nominal')
    greedy = zero
    for i in range(len(offsets)):
        choices = [greedy]
        for factor in cfg['greedy_scales']:
            z = coefficients[greedy].copy(); z[i] = proposal[i]*factor
            choices.append(evaluate(z, 'greedy', i))
        greedy = min(choices, key=rank)
    greedy_calls = len(rows)
    # Joint seeds vary multiple chunks at once, including much smaller radii.
    population = [zero, greedy]
    for factor in [.001, .003, .01, .03, .1, .25, .5, 1.]:
        population.append(evaluate(proposal*factor, 'joint_initial'))
    while len(population)<cfg['joint_population']:
        weights = rng.choice([0., .01, .1, .5, 1.], size=(shape[0], 1))
        population.append(evaluate(proposal*weights, 'joint_initial'))
    generations = 1 if smoke else cfg['joint_generations']
    for generation in range(generations):
        ranked = sorted(set(population+[greedy, zero]), key=rank)
        elite = ranked[:min(8, len(ranked))]
        children = []
        for k in range(cfg['joint_population']):
            parent = coefficients[int(rng.choice(elite))]
            if k%4 == 0:
                # All coefficient directions are searchable, not just proposal scales.
                sigma = rng.choice([.001, .003, .01, .03, .1])*radius
                child = parent + rng.normal(size=shape)*sigma
            elif k%4 == 1:
                other = coefficients[int(rng.choice(ranked))]
                child = np.where(rng.random((shape[0], 1))<.5, parent, other)
            elif k%4 == 2:
                child = parent.copy()
                mask = rng.random(shape[0]) < rng.choice([.1, .3, .7])
                child[mask] = proposal[mask]*rng.uniform(0., 1., size=(int(mask.sum()),1))
            else:
                child = parent*rng.uniform(.25, 1.5, size=(shape[0], 1))
            children.append(evaluate(child, 'joint_search', generation))
        population = sorted(set(population+children), key=rank)[:cfg['joint_population']]
    best = min(range(len(rows)), key=rank)
    pd.DataFrame(rows).to_csv(done.with_suffix('.csv'), index=False)
    np.savez_compressed(done.with_suffix('.npz'), initial=initial, nominal=nominal,
                        basis=basis, scale=scale, offsets=offsets, coefficients=coefficients,
                        physics=traces, coverage=coverages)
    result = dict(seed=seed, smoke=smoke, manifest_sha256=manifest_hash,
                  candidates=len(rows), greedy_calls=greedy_calls, greedy_index=greedy,
                  best_index=best, nominal_index=zero, greedy=rows[greedy], best=rows[best],
                  correction_blocks=len(offsets), coefficient_dimension=int(np.prod(shape)),
                  controller_solvers_feasible=solver, seconds=time.perf_counter()-start_time,
                  files={ext:sha256(done.with_suffix(ext)) for ext in ['.csv','.npz']})
    write_json(done, result)
    print(json.dumps({k:result[k] for k in ['seed','candidates','seconds','greedy','best']}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true')
    p.add_argument('--workers', type=int, default=4); args = p.parse_args()
    cfg = read(resolve(CONFIG)); out = resolve(cfg['output']+('_smoke' if args.smoke else ''))
    out.mkdir(parents=True, exist_ok=True)
    inputs = [CONFIG, 'scripts/run_followup_sequence_oracle.py',
              'scripts/run_followup_policy_terminal.py', 'zj_wam/policy_refinement.py',
              'zj_wam/followup.py', 'zj_wam/diagnostics.py']
    for seed in cfg['seeds']:
        inputs.extend([f"{cfg['source']}/{seed}_nominal.npz",
                       f'artifacts/followup_v1/policy_terminal/{seed}_h05.npz'])
    manifest = dict(config=cfg, smoke=args.smoke, inputs={s:sha256(resolve(s)) for s in inputs})
    if (out/'manifest.json').exists(): assert read(out/'manifest.json') == manifest
    else: write_json(out/'manifest.json', manifest)
    seeds = cfg['seeds'][:1] if args.smoke else cfg['seeds']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process, seeds, [args.smoke]*len(seeds)))
    for path, h in manifest['inputs'].items(): assert sha256(resolve(path)) == h, path
    write_json(out/'execution_complete.json', dict(seeds=seeds, states=len(results),
               candidates=sum(r['candidates'] for r in results), manifest_sha256=sha256(out/'manifest.json')))

if __name__ == '__main__': main()
