"""Reproducible Phase 1A/1B counterfactual experiments; resume by state files."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from zj_wam.adapters import ExpertDataset, FrozenLeWM, ReplaySimulator
from zj_wam.io import resolve, sha256, write_json
from zj_wam.metrics import outcome_metrics
from zj_wam.sensitivity import action_jacobian, eigensystem, make_directions, symmetric_bound_radius


def make_manifest(cfg, data, out):
    path = out / 'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['config'] != cfg:
            raise ValueError('Config changed for an existing run; use a new output directory.')
        return saved
    rng = np.random.default_rng(cfg['seed'])
    min_t = (cfg['history_frames'] - 1) * cfg['action_block']
    horizon = cfg['horizon_blocks'] * cfg['action_block']
    eligible = [int(i) for i, n in enumerate(data.dataset.lengths) if n > min_t + horizon + 2]
    # The dataset contains many noisy replays of the same initial placement.
    # Split initial-pose families, then select ONE episode per family.
    starts = data.states[np.asarray(data.dataset.offsets, dtype=int), :5]
    _, family_of = np.unique(starts, axis=0, return_inverse=True)
    families = {}
    for ep in eligible:
        families.setdefault(int(family_of[ep]), []).append(ep)
    family_ids = np.array(sorted(families))
    rng.shuffle(family_ids)
    eligible = [int(rng.choice(families[int(group)])) for group in family_ids]
    excluded = set(cfg.get('excluded_initial_pose_families', []))
    eligible = [ep for ep in eligible if int(family_of[ep]) not in excluded]
    required = sum(cfg['split_episodes'].values())
    if len(eligible) < required:
        raise ValueError(f'Only {len(eligible)} episodes for {required} requested; explicitly revise design before running.')
    states, offset = [], 0
    for split, count in cfg['split_episodes'].items():
        for ep in eligible[offset:offset+count]:
            length = int(data.dataset.lengths[ep])
            # One state from each temporal half; no selection on treatment outcomes.
            intervals = np.array_split(np.arange(min_t, length - horizon - 1), cfg['states_per_episode'])
            for times in intervals:
                t = int(rng.choice(times))
                states.append({'split': split, 'episode': ep, 'timestep': t,
                               'initial_pose_family': int(family_of[ep]),
                               'state_id': f'{split}_e{ep:04d}_t{t:04d}'})
        offset += count
    manifest = {'config': cfg, 'states': states,
                'normalization_mean': data.scaler.mean_.tolist(),
                'normalization_scale': data.scaler.scale_.tolist(),
                'checkpoint_sha256': sha256(resolve(cfg['checkpoint']) / 'weights.pt'),
                'sampling': 'initial-pose-family-disjoint splits; one episode per family, uniform state in each temporal bin; no outcome-based filtering',
                'eligible_initial_pose_families': len(families),
                'representation': '192-dimensional predicted projected CLS embedding, including robot and object; not an object-specific latent',
                'radius_unit': 'relative_radius * sqrt(2 * raw_action_horizon), in standardized action coordinates',
                'task_score': 'object-only target polygon coverage; official seven-state distance and success reported separately'}
    write_json(path, manifest)
    return manifest


def run_state(spec, cfg, data, model, out):
    began = time.perf_counter()
    state_id = spec['state_id']
    state_dir = out / 'states'
    state_dir.mkdir(parents=True, exist_ok=True)
    done = state_dir / f'{state_id}.json'
    if done.exists():
        return json.loads(done.read_text())
    ep, t = spec['episode'], spec['timestep']
    actions, states = data.episode(ep)
    horizon = cfg['action_block'] * cfg['horizon_blocks']
    nominal = actions[t:t+horizon].copy()
    sim = ReplaySimulator(actions, states, t, cfg['seed'] + ep,
                          cfg['action_block'], cfg['history_frames'])
    images, past = sim.restore(capture=True)
    restored = sim.restored_state.copy()
    reference = sim.execute(nominal, restore=False)
    repeat = sim.execute(nominal)
    restore_error = float(np.max(np.abs(reference['states'] - repeat['states'])))
    if restore_error > 1e-9:
        raise AssertionError(f'Non-deterministic restoration at {state_id}: {restore_error}')
    phase = ('contact' if np.any(reference['contacts']) else
             'near_contact' if reference['initial_gap'] <= cfg['near_contact_gap'] else 'free_motion')
    predict, context = model.predictor(images, past, cfg['action_block'], cfg['representation'])
    a0 = torch.from_numpy(model.normalize(nominal).reshape(-1)).to(cfg['device'])
    with torch.no_grad():
        z0 = predict(a0)
        zero_pred_error = float(torch.max(torch.abs(predict(a0 + torch.zeros_like(a0)) - z0)))
    jac = action_jacobian(predict, a0)
    eigenvalues, vectors = eigensystem(jac)
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], ep, t]))
    directions = make_directions(vectors, rng, cfg['random_directions'], cfg['mixture_directions'], cfg['low_rank'])
    directions = {k: v.astype(np.float32) for k, v in directions.items()}
    low = np.tile((-np.ones(2) - data.scaler.mean_) / data.scaler.scale_, horizon)
    high = np.tile((np.ones(2) - data.scaler.mean_) / data.scaler.scale_, horizon)
    max_radius = symmetric_bound_radius(a0.cpu().numpy(), directions.values(), low, high)
    fd_errors = []
    step = cfg['finite_difference_step']
    for name in ['low', 'high', 'random_0']:
        direction = torch.from_numpy(directions[name]).to(cfg['device'])
        with torch.no_grad():
            fd = (predict(a0 + step * direction) - predict(a0 - step * direction)) / (2 * step)
        derivative = jac @ direction
        fd_errors.append(float(torch.linalg.vector_norm(fd - derivative) /
                               torch.linalg.vector_norm(derivative).clamp_min(1e-8)))
    records, trajectories, action_arrays = [], [reference['states']], [nominal]
    for relative in cfg['relative_radii']:
        requested = float(relative * np.sqrt(a0.numel()))
        # Common radius for ALL directions/signs; record every reduction explicitly.
        radius = min(requested, max_radius * 0.999)
        for name, direction_np in directions.items():
            for sign in [-1, 1]:
                shared = {**spec, 'interaction_phase': phase, 'direction': name,
                          'sign': sign, 'relative_radius': relative,
                          'requested_radius': requested, 'applied_radius': radius,
                          'bounds_reduced': bool(radius < requested * (1-1e-6)),
                          'clipped': False, 'seed': cfg['seed'],
                          'nominal_task_success': reference['task_success']}
                if radius <= 1e-7:
                    records.append({**shared, 'status': 'excluded_no_common_bound_radius'})
                    continue
                candidate = a0 + sign * radius * torch.from_numpy(direction_np).to(cfg['device'])
                raw = model.denormalize(candidate.detach().cpu().numpy())
                if np.max(np.abs(raw)) > 1 + 1e-5:
                    raise AssertionError('Action out of bounds')
                actual_delta = float(torch.linalg.vector_norm(candidate - a0))
                if not np.isclose(actual_delta, radius, rtol=2e-4, atol=2e-6):
                    raise AssertionError('Perturbation radius mismatch')
                with torch.no_grad():
                    pred_change = float(torch.linalg.vector_norm(predict(candidate) - z0))
                executed = sim.execute(raw)
                metrics = outcome_metrics(executed, reference)
                jnorm = float(np.linalg.norm(jac.cpu().numpy() @ direction_np))
                records.append({**shared, **metrics, 'status': 'ok',
                                'predicted_change': pred_change,
                                'directional_sensitivity': jnorm,
                                'linear_predicted_change': radius * jnorm,
                                'actual_delta_norm': actual_delta,
                                'raw_delta_norm': float(np.linalg.norm(raw - nominal)),
                                'trajectory_array_index': len(trajectories)})
                trajectories.append(executed['states'])
                action_arrays.append(raw)
    sim.close()
    table_path = state_dir / f'{state_id}.csv'
    pd.DataFrame(records).to_csv(table_path, index=False)
    np.savez_compressed(state_dir / f'{state_id}.npz', nominal_action=nominal,
                        nominal_observation=np.stack(images), past_actions=past,
                        eigenvalues=eigenvalues, eigenvectors=vectors,
                        direction_names=np.array(list(directions)),
                        directions=np.stack(list(directions.values())),
                        jacobian=jac.cpu().numpy(), nominal_prediction=z0.cpu().numpy(),
                        object_trajectories=np.stack(trajectories), actions=np.stack(action_arrays),
                        recorded_state=states[t], replayed_state=restored)
    result = {**spec, 'interaction_phase': phase,
              'zero_simulator_error': restore_error, 'zero_prediction_error': zero_pred_error,
              'finite_difference_relative_errors': fd_errors,
              'recorded_vs_replayed_state_l2': float(np.linalg.norm(states[t] - restored)),
              'eigenvalues': eigenvalues.tolist(), 'rows': len(records),
              'duration_seconds': time.perf_counter() - began}
    write_json(done, result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase1.json')
    parser.add_argument('--split', choices=['development','calibration','test','all'], default='development')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(resolve(args.config).read_text())
    torch.set_num_threads(cfg['torch_threads'])
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    out = resolve(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    data = ExpertDataset(cfg['dataset'])
    manifest = make_manifest(cfg, data, out)
    selected = [s for s in manifest['states'] if args.split == 'all' or s['split'] == args.split]
    if args.split in ['test', 'all'] and not (out/'calibration.json').exists():
        raise ValueError('Freeze calibration.json before held-out test execution')
    if args.split in ['test','all']:
        frozen_hash = sha256(out/'calibration.json')
        selected = [dict(s, calibration_sha256=frozen_hash) if s['split']=='test' else s for s in selected]
    if args.limit:
        selected = selected[:args.limit]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard')
    selected = selected[args.shard_index::args.shard_count]
    model = FrozenLeWM(cfg['checkpoint'], data.scaler, cfg['device'])
    for state in selected:
        run_state(state, cfg, data, model, out)


if __name__ == '__main__':
    main()
