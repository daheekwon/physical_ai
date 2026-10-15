"""Record actual model, data, simulator and derivative checks before experiments."""
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from zj_wam.adapters import ExpertDataset, FrozenLeWM, ReplaySimulator
from zj_wam.io import ROOT, resolve, write_json, sha256
from zj_wam.sensitivity import action_jacobian, eigensystem


def main():
    torch.set_num_threads(4)
    cfg = json.loads(resolve('configs/phase1.json').read_text())
    started = time.perf_counter()
    data = ExpertDataset(cfg['dataset'])
    model = FrozenLeWM(cfg['checkpoint'], data.scaler)
    report = {
        'versions': {p: importlib.metadata.version(p) for p in [
            'torch', 'torchvision', 'stable-worldmodel', 'stable-pretraining',
            'pymunk', 'pygame', 'gymnasium', 'numpy', 'h5py', 'transformers']},
        'lewm_commit': subprocess.check_output(['git','-C',str(ROOT/'external/le-wm'),'rev-parse','HEAD'],text=True).strip(),
        'checkpoint_sha256': sha256(resolve(cfg['checkpoint'])/'weights.pt'),
        'dataset_shape': {k: list(data.dataset.get_col_data(k).shape) for k in ['action','state']},
        'episodes': int(len(data.dataset.lengths)),
        'unique_initial_pose_families': int(len(np.unique(data.states[np.asarray(data.dataset.offsets,dtype=int),:5],axis=0))),
        'action_mean': data.scaler.mean_.tolist(), 'action_scale': data.scaler.scale_.tolist(),
        'action_min': np.nanmin(data.actions,axis=0).tolist(), 'action_max': np.nanmax(data.actions,axis=0).tolist(),
        'parameters': sum(p.numel() for p in model.model.parameters()),
        'frozen_parameters': all(not p.requires_grad for p in model.model.parameters()),
        'checks': [],
    }
    for ep in [0, 1, 2]:
        actions, states = data.episode(ep)
        t = min(30, len(actions)-7)
        sim = ReplaySimulator(actions, states, t, cfg['seed']+ep)
        images, past = sim.restore(capture=True)
        restored = sim.restored_state.copy()
        nominal = actions[t:t+5]
        uninterrupted = sim.execute(nominal, restore=False)
        repeat = sim.execute(nominal)
        predict, context = model.predictor(images,past)
        a0 = torch.from_numpy(model.normalize(nominal).reshape(-1))
        jac = action_jacobian(predict,a0)
        vals, vecs = eigensystem(jac)
        errors = []
        for v in [vecs[:,0],vecs[:,-1]]:
            direction = torch.tensor(v,dtype=a0.dtype)
            eps = cfg['finite_difference_step']
            finite = (predict(a0+eps*direction)-predict(a0-eps*direction))/(2*eps)
            deriv = jac @ direction
            errors.append(float((finite-deriv).norm()/deriv.norm().clamp_min(1e-8)))
        # Compare our cached-context one-step path with the official predict API.
        past_and_current = torch.cat([context['past'],a0.reshape(1,1,10)],1)
        official = model.model.predict(context['embedding'],model.model.action_encoder(past_and_current))[:,-1].reshape(-1)
        comparison = float((official-predict(a0)).abs().max())
        check = {
            'episode': ep,'timestep':t,
            'sim_repeat_max_error':float(np.max(np.abs(uninterrupted['states']-repeat['states']))),
            'zero_prediction_max_error':float((predict(a0)-predict(a0+0)).abs().max()),
            'cached_vs_official_predict_max_error':comparison,
            'action_roundtrip_error':float(np.max(np.abs(model.denormalize(a0.numpy())-nominal))),
            'finite_difference_relative_errors':errors,
            'eigenvalues':vals.tolist(),
            'replay_vs_recorded_state_error':float(np.linalg.norm(restored-states[t])),
            'contact_counts':uninterrupted['contacts'].tolist(),
            'physics_dt':sim.env.dt,'control_hz':sim.env.control_hz,
            'relative_actions':sim.env.relative,'action_scale':sim.env.action_scale,
        }
        report['checks'].append(check)
        print(json.dumps(check),flush=True)
        sim.close()
    report['duration_seconds'] = time.perf_counter()-started
    report['invariants_pass'] = all(
        c['sim_repeat_max_error']<1e-9 and c['zero_prediction_max_error']<1e-8
        and c['cached_vs_official_predict_max_error']<1e-6
        and c['action_roundtrip_error']<1e-6
        and max(c['finite_difference_relative_errors'])<0.05 for c in report['checks'])
    report['official_evaluation_status'] = 'see separate official_eval log; not inferred from unit checks'
    write_json(resolve('artifacts/phase0/checks.json'),report)
    if not report['invariants_pass']:
        raise SystemExit('Phase 0 invariants failed; inspect checks.json')


if __name__=='__main__':
    main()
