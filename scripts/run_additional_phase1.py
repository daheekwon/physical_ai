"""Frozen-protocol proposal diagnostics, stored separately from original Phase 1."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from zj_wam.adapters import ExpertDataset,FrozenLeWM,ReplaySimulator
from zj_wam.diagnostics import DiagnosticSimulator,polynomial_basis,frequency_matched_directions,temporal_properties,jerk_energy
from zj_wam.io import resolve,sha256,write_json
from zj_wam.metrics import outcome_metrics
from zj_wam.sensitivity import action_jacobian,eigensystem,symmetric_bound_radius


def initialize(cfg,data,out):
    path=out/'manifest.json'
    if path.exists():
        saved=json.loads(path.read_text());assert saved['config']==cfg
        return saved
    base=json.loads(resolve(cfg['base_manifest']).read_text())
    metadata={s['state_id']:json.loads((resolve('artifacts/phase1/states')/f"{s['state_id']}.json").read_text()) for s in base['states']}
    excluded=set(json.loads(resolve('artifacts/phase1/calibration.json').read_text())['excluded_calibration_source_replay_states'])
    selected=[];selection_exclusions=[]
    for split,quotas in cfg['oracle_counts'].items():
        used=set()
        for phase,count in quotas.items():
            pool=[s for s in base['states'] if s['split']==split and metadata[s['state_id']]['interaction_phase']==phase]
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],len(selected),count]))
            pool=[pool[i] for i in rng.permutation(len(pool))]
            chosen=[]
            for s in pool:
                ep,t=s['episode'],s['timestep'];actions,_=data.episode(ep)
                reason=None
                if ep in used:continue
                if s['state_id'] in excluded:reason='previously_audited_source_replay_quality'
                elif t+max(cfg['oracle_horizons'])+cfg['suffix_actions']>len(actions):reason='insufficient_future_for_longest_horizon_and_suffix'
                elif np.max(np.abs(actions[t:t+max(cfg['oracle_horizons'])]))>1:reason='nominal_chunk_outside_declared_bounds'
                if reason:
                    selection_exclusions.append({**s,'reason':reason});continue
                chosen.append({**s,'pilot_interaction_phase':phase});used.add(ep)
                if len(chosen)==count:break
            if len(chosen)<count:
                raise ValueError(f'Insufficient eligible {split}/{phase}: {len(chosen)} for {count}; revise protocol explicitly')
            selected+=chosen
    manifest={'config':cfg,'base_manifest_sha256':sha256(resolve(cfg['base_manifest'])),
              'proposal_sha256':sha256(resolve(cfg['proposal'])),
              'original_calibration_sha256':sha256(resolve('artifacts/phase1/calibration.json')),
              'base_config':base['config'],'normalization_mean':base['normalization_mean'],'normalization_scale':base['normalization_scale'],
              'controls_states':[s for s in base['states'] if s['split']=='test'],
              'oracle_states':selected,'oracle_selection_exclusions':selection_exclusions,
              'polynomial_candidate_budget_per_state_horizon':(cfg['oracle_random_directions']+cfg['oracle_low_directions']+1)*2*len(cfg['oracle_radii']),
              'analysis_scope':cfg['notes']}
    write_json(path,manifest)
    return manifest


def bounds(a0,dirs,scaler):
    horizon=a0.numel()//2
    lo=np.tile((-np.ones(2)-scaler.mean_)/scaler.scale_,horizon)
    hi=np.tile((np.ones(2)-scaler.mean_)/scaler.scale_,horizon)
    return symmetric_bound_radius(a0.numpy(),dirs.values(),lo,hi)


def controls(spec,cfg,manifest,data,model,out):
    directory=out/'controls';directory.mkdir(exist_ok=True)
    sid=spec['state_id'];done=directory/f'{sid}.json'
    if done.exists():return
    started=time.perf_counter()
    a=np.load(resolve('artifacts/phase1/states')/f'{sid}.npz')
    original={name:d for name,d in zip(a['direction_names'],a['directions'])}
    directions={k:original[k].astype(float) for k in ['low','high','random_0','random_1','random_2']}
    # Normalize float32 source vectors once; all comparators use identical radius.
    directions={k:v/np.linalg.norm(v) for k,v in directions.items()}
    directions.update({f'coordinate_{i}':v for i,v in enumerate(np.eye(10))})
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],spec['episode'],spec['timestep']]))
    directions.update({f'frequency_{i}':v for i,v in enumerate(frequency_matched_directions(directions['low'],rng,cfg['frequency_draws']))})
    actions,states=data.episode(spec['episode'])
    sim=ReplaySimulator(actions,states,spec['timestep'],manifest['base_config']['seed']+spec['episode'])
    images,past=sim.restore(capture=True);ref=sim.execute(a['nominal_action'],restore=False)
    np.testing.assert_allclose(ref['states'],a['object_trajectories'][0],atol=0,rtol=0)
    phase='contact' if np.any(ref['contacts']) else 'near_contact' if ref['initial_gap']<=10 else 'free_motion'
    predict,_=model.predictor(images,past)
    a0=torch.from_numpy(model.normalize(a['nominal_action']).reshape(-1))
    with torch.no_grad():z0=predict(a0)
    radius_bound=bounds(a0,directions,model.scaler)
    records=[];traces=[ref['states']];raw_actions=[a['nominal_action']]
    for relative in cfg['controls_radii']:
        radius=min(relative*np.sqrt(10),radius_bound*.999)
        assert radius>1e-7
        for name,d in directions.items():
            for sign in [-1,1]:
                candidate=a0+torch.tensor(sign*radius*d,dtype=a0.dtype)
                raw=model.denormalize(candidate.numpy())
                assert np.max(np.abs(raw))<=1+1e-5
                with torch.no_grad():pred=float(torch.linalg.vector_norm(predict(candidate)-z0))
                result=sim.execute(raw)
                deltaq=np.linalg.norm(result['states'][:,:2]-ref['states'][:,:2],axis=1)
                records.append({**spec,'interaction_phase':phase,'direction':name,'sign':sign,
                    'relative_radius':relative,'applied_radius':radius,'predicted_change':pred,
                    'actual_norm':float(torch.linalg.vector_norm(candidate-a0)),
                    'pusher_path_mean':float(deltaq.mean()),'pusher_path_max':float(deltaq.max()),
                    'pusher_final_displacement':float(deltaq[-1]),
                    'nominal_task_success':ref['task_success'],
                    **temporal_properties(d),**outcome_metrics(result,ref),
                    'contact_changed_steps':int(np.sum((result['contacts']>0)!=(ref['contacts']>0))),
                    'trajectory_index':len(traces)})
                traces.append(result['states']);raw_actions.append(raw)
    sim.close()
    pd.DataFrame(records).to_csv(directory/f'{sid}.csv',index=False)
    np.savez_compressed(directory/f'{sid}.npz',trajectories=traces,actions=raw_actions,
                        directions=np.stack(list(directions.values())),direction_names=np.array(list(directions)))
    record={**spec,'rows':len(records),'duration_seconds':time.perf_counter()-started}
    write_json(done,record);print(json.dumps(record),flush=True)


def oracle(spec,h,cfg,manifest,data,model,out):
    directory=out/'oracle';directory.mkdir(exist_ok=True)
    sid=f"{spec['state_id']}_h{h:02d}";done=directory/f'{sid}.json'
    if done.exists():return
    started=time.perf_counter();ep,t=spec['episode'],spec['timestep']
    actions,states=data.episode(ep);nominal=actions[t:t+h];suffix=actions[t+h:t+h+cfg['suffix_actions']]
    sim=DiagnosticSimulator(actions,states,t,manifest['base_config']['seed']+ep)
    images,past=sim.restore(capture=True)
    restored=sim.restored_state.copy()
    ref=sim.execute_diagnostic(nominal,suffix,restore=False)
    repeated=sim.execute_diagnostic(nominal,suffix)
    np.testing.assert_array_equal(ref['physics_trace'],repeated['physics_trace'])
    phase='contact' if np.any(ref['contacts']) else 'near_contact' if ref['initial_gap']<=10 else 'free_motion'
    predict,_=model.predictor(images,past)
    a0=torch.from_numpy(model.normalize(nominal).reshape(-1))
    clock=time.perf_counter();jac=action_jacobian(predict,a0);jac_seconds=time.perf_counter()-clock
    q=polynomial_basis(h);polyjac=jac@torch.tensor(q,dtype=jac.dtype)
    eig,v=eigensystem(polyjac)
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],ep,t,h]))
    dirs={f'poly_low_{i}':q@v[:,i] for i in range(cfg['oracle_low_directions'])}
    for i in range(cfg['oracle_random_directions']):
        direction=rng.normal(size=q.shape[1]);dirs[f'poly_random_{i}']=q@(direction/np.linalg.norm(direction))
    # Analytic descent of raw command roughness, deliberately labelled surrogate.
    scale=np.tile(model.scaler.scale_,h)
    matrix=np.zeros(((h+6)*2,q.shape[1]));matrix[6:6+h*2]=q*scale[:,None]
    design=np.diff(matrix.reshape(h+6,2,-1),n=3,axis=0).reshape(-1,q.shape[1])
    nominal_diff=np.diff(np.concatenate([actions[t-3:t],nominal,suffix]),n=3,axis=0).reshape(-1)
    coefficient=np.linalg.lstsq(design,-nominal_diff,rcond=None)[0]
    direction=q@coefficient
    if np.linalg.norm(direction)<1e-12:direction=q[:,0]
    dirs['command_descent']=direction/np.linalg.norm(direction)
    bound=bounds(a0,dirs,model.scaler)
    with torch.no_grad():
        z0=predict(a0)
        encoded=model.model.encode({'pixels':model.transform(ref['final_image']).unsqueeze(0).unsqueeze(0)})['emb'].reshape(-1)
        observed_prediction_error=float(torch.linalg.vector_norm(z0-encoded))
    fd=[]
    for name in ['poly_low_0','poly_random_0','command_descent']:
        d=torch.tensor(dirs[name],dtype=a0.dtype);step=.05
        with torch.no_grad():finite=(predict(a0+step*d)-predict(a0-step*d))/(2*step)
        derivative=jac@d
        fd.append(float(torch.linalg.vector_norm(finite-derivative)/torch.linalg.vector_norm(derivative).clamp_min(1e-8)))
    records=[];controls_trace=[ref['control_trace']];physics_trace=[ref['physics_trace']];action_arrays=[nominal]
    predicted_vectors=[z0.numpy()];model_forward_seconds=0.
    for relative in cfg['oracle_radii']:
        radius=min(relative*np.sqrt(2*h),bound*.999)
        if radius<=1e-7:raise ValueError(f'No common feasible radius {sid}')
        for name,direction in dirs.items():
            for sign in [-1,1]:
                candidate=a0+torch.tensor(sign*radius*direction,dtype=a0.dtype)
                raw=model.denormalize(candidate.numpy())
                assert np.max(np.abs(raw))<=1+1e-5
                clock=time.perf_counter()
                with torch.no_grad():predicted=predict(candidate)
                model_forward_seconds+=time.perf_counter()-clock
                result=sim.execute_diagnostic(raw,suffix)
                metrics=outcome_metrics(result,ref)
                controlpos=np.linalg.norm(result['control_trace'][:,2:4]-ref['control_trace'][:,2:4],axis=1)
                controlrot=np.abs(np.arctan2(np.sin(result['control_trace'][:,4]-ref['control_trace'][:,4]),np.cos(result['control_trace'][:,4]-ref['control_trace'][:,4])))
                physicalpos=np.linalg.norm(result['physics_trace'][:,2:4]-ref['physics_trace'][:,2:4],axis=1)
                records.append({**spec,'horizon':h,'interaction_phase':phase,'direction':name,'sign':sign,
                    'relative_radius':relative,'applied_radius':radius,'actual_norm':float(torch.linalg.vector_norm(candidate-a0)),
                    'predicted_change':float(torch.linalg.vector_norm(predicted-z0)),
                    'linear_prediction':float(radius*np.linalg.norm(jac.numpy()@direction)),
                    **metrics,'position_with_suffix_max':float(controlpos.max()),'rotation_with_suffix_max':float(controlrot.max()),
                    'physics_object_position_max':float(physicalpos.max()),
                    'contact_changed_steps':int(np.sum((result['contacts']>0)!=(ref['contacts']>0))),
                    'pusher_path_max':float(np.max(np.linalg.norm(result['states'][:,:2]-ref['states'][:,:2],axis=1))),
                    **{k:result[k] for k in ['executed_jerk','physics_jerk','target_jerk','action_roughness']},
                    **{f'nominal_{k}':ref[k] for k in ['executed_jerk','physics_jerk','target_jerk','action_roughness']},
                    'nominal_task_success':ref['task_success'],'trajectory_index':len(controls_trace)})
                controls_trace.append(result['control_trace']);physics_trace.append(result['physics_trace']);action_arrays.append(raw);predicted_vectors.append(predicted.numpy())
    sim.close()
    pd.DataFrame(records).to_csv(directory/f'{sid}.csv',index=False)
    np.savez_compressed(directory/f'{sid}.npz',control_traces=controls_trace,physics_traces=physics_trace,
                        actions=action_arrays,polynomial_basis=q,directions=np.stack(list(dirs.values())),
                        direction_names=np.array(list(dirs)),jacobian=jac.numpy(),eigenvalues=eig,
                        predictions=predicted_vectors,nominal_observed_embedding=encoded.numpy(),
                        recorded_state=states[t],replayed_state=restored)
    record={**spec,'horizon':h,'rows':len(records),'interaction_phase':phase,
            'jacobian_seconds':jac_seconds,'model_forward_seconds':model_forward_seconds,
            'model_candidate_forward_calls':len(records),'finite_difference_relative_errors':fd,
            'nominal_future_latent_prediction_error':observed_prediction_error,
            'repeat_physics_max_error':0.,'duration_seconds':time.perf_counter()-started}
    if spec['split']=='test':record['calibration_sha256']=sha256(out/'oracle_calibration.json')
    write_json(done,record);print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['initialize','controls','oracle'])
    parser.add_argument('--split',default='development',choices=['development','calibration','test'])
    parser.add_argument('--limit',type=int);parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1)
    args=parser.parse_args();cfg=json.loads(resolve('configs/phase1_additional.json').read_text())
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    out=resolve(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    base=json.loads(resolve(cfg['base_manifest']).read_text());data=ExpertDataset(base['config']['dataset'])
    manifest=initialize(cfg,data,out)
    if args.mode=='initialize':print(json.dumps({'oracle_states':manifest['oracle_states'],'exclusions':manifest['oracle_selection_exclusions']},indent=2));return
    if args.mode=='oracle' and args.split=='test' and not (out/'oracle_calibration.json').exists():raise ValueError('Freeze diagnostic calibration before oracle test')
    model=FrozenLeWM(base['config']['checkpoint'],data.scaler)
    selected=manifest['controls_states'] if args.mode=='controls' else [s for s in manifest['oracle_states'] if s['split']==args.split]
    if args.limit:selected=selected[:args.limit]
    for spec in selected[args.shard_index::args.shard_count]:
        if args.mode=='controls':controls(spec,cfg,manifest,data,model,out)
        else:
            for h in cfg['oracle_horizons']:oracle(spec,h,cfg,manifest,data,model,out)

if __name__=='__main__':main()
