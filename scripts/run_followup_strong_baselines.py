"""F9 strong optimization/geometric baselines; no simulator outcome is used to generate."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from scipy.ndimage import gaussian_filter1d
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,dense_metrics,controller_design,controller_rollout,optimize_controller
from zj_wam.diagnostics import polynomial_basis,jerk_energy
from zj_wam.strong_optimizers import quadratic_solve,nonlinear_solve,geometric_solve
from zj_wam.io import resolve,write_json,sha256


def generate(nominal,suffix,root,prefix,past,scale,model,images,jac,cfg,tau,extended=False):
    h=len(nominal);q,base,response,j0,design=controller_design(nominal,suffix,root,prefix,scale);rawmap=q*np.tile(scale,h)[:,None]
    predict,_=model.predictor(images,past);a0=torch.tensor(model.normalize(nominal).reshape(-1),device=model.device)
    penalty=(jac@q).T@(jac@q)/tau**2
    candidates=[]
    for relative in cfg['radii']:
        radius=relative*np.sqrt(2*h)
        controller,ci=optimize_controller(nominal,suffix,root,prefix,scale,radius)
        if not extended:
            candidates.append((controller,{'method':'controller_polynomial','relative_radius':relative,'setting':0.,'info':ci}))
            full=np.concatenate([past[-3:],nominal,suffix])
            for sigma in cfg['filter_sigma']:
                raw=gaussian_filter1d(full,sigma,axis=0,mode='nearest')[3:3+h]
                change=((raw-nominal)/scale).reshape(-1);factor=min(1.,radius/max(np.linalg.norm(change),1e-12));raw=nominal+factor*(raw-nominal)
                candidates.append((raw,{'method':'gaussian_action','relative_radius':relative,'setting':sigma,'info':{'solver_success':True,'seconds':0.}}))
            for weight in cfg['soft_weights']:
                raw,info=quadratic_solve(nominal,rawmap,j0,design,radius,penalty=penalty,weight=weight)
                candidates.append((raw,{'method':'soft_sensitivity','relative_radius':relative,'setting':weight,'info':info}))
        for factor in cfg['nonlinear_tau_multipliers']:
            raw,info=nonlinear_solve(predict,a0,q,nominal,rawmap,j0,design,radius,tau*factor,[np.zeros(q.shape[1]),ci['coefficient']],
                                     budget=cfg['nonlinear_extended_budget'] if extended else cfg['nonlinear_forward_budget'])
            candidates.append((raw,{'method':'nonlinear_wm','relative_radius':relative,'setting':factor,'info':info}))
    if not extended:
        identity=np.eye(2*h);_,fullbase,fullresponse,_,_=controller_design(nominal,suffix,root,prefix,scale,identity);fullmap=identity*np.tile(scale,h)[:,None]
        for method in ['lipo','groove']:
          for family in ['full_command','polynomial_executed']:
            for epsilon in cfg['geometric_bounds_px']:
                radius=cfg['geometric_action_radius']*np.sqrt(2*h)
                if family=='full_command':mapping,b,r,obj=fullmap,fullbase,fullresponse,None
                else:mapping,b,r,obj=rawmap,base,response,(j0,design)
                raw,info=geometric_solve(nominal,mapping,b,r,prefix[[9,19,29],:2],radius,epsilon,method,obj)
                candidates.append((raw,{'method':method+'_'+family,'relative_radius':cfg['geometric_action_radius'],'setting':epsilon,'info':info}))
    return candidates,q


def process(path,cfg,bcfg,data,model,out,extended):
    name='strong_baselines_extended' if extended else 'strong_baselines';dest=out/name;dest.mkdir(exist_ok=True);done=dest/path.name
    if done.exists():return
    old=json.loads(path.read_text());spec={k:old[k] for k in ['state_id','split','episode','timestep','initial_pose_family','horizon']};t=spec['timestep'];h=spec['horizon'];start=time.perf_counter()
    allactions,states=data.episode(spec['episode']);nominal=allactions[t:t+h];suffix=allactions[t+h:t+h+3];scale=data.scaler.scale_
    sim=DenseSimulator(allactions,states,t,cfg['seed']+spec['episode']);images,past=sim.restore(capture=True)
    reference=sim.execute_dense(nominal,suffix,restore=False,capture_endpoints=True);root=reference['root_full_state'];prefix=sim.prefix_physics
    with np.load(out/'interventions'/path.with_suffix('.npz').name) as a:jac=a['jacobian']
    tau=json.loads((out/'selection/selector_rules.json').read_text())['rules']['lewm_global']['tau']
    generated,q=generate(nominal,suffix,root,prefix,past,scale,model,images,jac,bcfg,tau,extended)
    # All candidates are frozen before executing even one modified trajectory.
    actions=[nominal];traces=[reference['physics_trace']];controls=[reference['control_trace']];endpoints=[reference['endpoint_images']]
    coverages=[[reference['coverage_t'],reference['coverage_suffix']]];contacts=[reference['physics_contacts']];rows=[];optimizer=[]
    nominal_predicted=jerk_energy(np.concatenate([prefix[:,:2],controller_rollout(np.concatenate([nominal,suffix]),root[:2],root[5:7])]),.01)
    for raw,meta in generated:
        if np.max(np.abs(raw))>1+1e-6:raise ValueError('Raw action bound failed')
        result=sim.execute_dense(raw,suffix,capture_endpoints=True);change=((raw-nominal)/scale).reshape(-1)
        if np.linalg.norm(change)>meta['relative_radius']*np.sqrt(2*h)+2e-6:raise ValueError('Action norm cap failed')
        prediction=controller_rollout(np.concatenate([raw,suffix]),root[:2],root[5:7]);info=meta.pop('info');optimizer.append(info)
        rows.append({**spec,**meta,'index':len(actions),'actual_norm':float(np.linalg.norm(change)),
                     'polynomial_residual':float(np.linalg.norm(change-q@(q.T@change))),
                     'predicted_jerk':jerk_energy(np.concatenate([prefix[:,:2],prediction]),.01),'nominal_predicted_jerk':nominal_predicted,
                     'nominal_physics_jerk':reference['physics_jerk'],'nominal_contact':bool(np.any(reference['physics_contacts'][:h*10]>0)),
                     'solver_success':info.get('solver_success',True),'optimizer_seconds':info.get('seconds',0.),'forward_calls':info.get('forward_calls',0),'backward_calls':info.get('backward_calls',0),
                     'finite_model_delta':info.get('model_delta',np.nan),'finite_model_threshold':info.get('model_threshold',np.nan),**dense_metrics(result,reference,cfg['tolerances'])})
        actions.append(raw);traces.append(result['physics_trace']);controls.append(result['control_trace']);endpoints.append(result['endpoint_images']);coverages.append([result['coverage_t'],result['coverage_suffix']]);contacts.append(result['physics_contacts'])
    f=pd.DataFrame(rows);f.to_csv(done.with_suffix('.csv'),index=False)
    np.savez_compressed(done.with_suffix('.npz'),actions=actions,physics=traces,control=controls,endpoint_images=endpoints,coverage=coverages,contacts=contacts,history_images=images,past_actions=past,root=root,basis=q)
    write_json(done,{**spec,'candidate_count':len(rows),'source_sha256':sha256(Path(__file__)),'optimizer_sha256':sha256(resolve('zj_wam/strong_optimizers.py')),
                    'config_sha256':sha256(resolve('configs/followup_strong_baselines.json')),'tau_reference':tau,'extended':extended,'seconds':time.perf_counter()-start,'optimizer_details':json.loads(json.dumps(optimizer,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))})
    sim.close();print(done.stem,len(rows),round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration'],default='development');p.add_argument('--limit',type=int);p.add_argument('--extended',action='store_true');args=p.parse_args()
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());bcfg=json.loads(resolve('configs/followup_strong_baselines.json').read_text());out=resolve(cfg['output']);torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    paths=sorted((out/'f1').glob(f'{args.split}*h05.json'))
    if len(paths)!=105:raise ValueError('Incomplete source bank')
    if args.extended:
        if args.split!='development':raise ValueError('Extended convergence study is development-only')
        families=sorted({json.loads(p.read_text())['initial_pose_family'] for p in paths})[:bcfg['extended_development_families']]
        paths=[p for p in paths if json.loads(p.read_text())['initial_pose_family'] in families]
    if args.limit:paths=paths[:args.limit]
    for path in paths:process(path,cfg,bcfg,data,model,out,args.extended)
if __name__=='__main__':main()
