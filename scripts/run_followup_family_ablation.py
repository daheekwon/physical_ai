"""F1 family/actual-boundary and F9 matched-dimension direction-generator ablations."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,dense_metrics,controller_design,controller_rollout,optimize_controller
from zj_wam.diagnostics import polynomial_basis,jerk_energy
from zj_wam.correction_families import interior_basis,spline_basis,terminal_constrained
from zj_wam.sensitivity import action_jacobian
from zj_wam.io import resolve,write_json,sha256


def process(path,cfg,data,model,out):
    dest=out/'family_ablation';dest.mkdir(exist_ok=True);done=dest/path.name
    if done.exists():return
    original=json.loads(path.read_text());spec={k:original[k] for k in ['state_id','split','episode','timestep','initial_pose_family','horizon']};h=spec['horizon'];t=spec['timestep'];start=time.perf_counter()
    a,s=data.episode(spec['episode']);nominal=a[t:t+h];suffix=a[t+h:t+h+3];scale=data.scaler.scale_
    sim=DenseSimulator(a,s,t,cfg['seed']+spec['episode']);images,past=sim.restore(capture=True)
    reference=sim.execute_dense(nominal,suffix,restore=False,capture_endpoints=True);root=reference['root_full_state'];prefix=sim.prefix_physics
    q=polynomial_basis(h);_,_,response,_,_=controller_design(nominal,suffix,root,prefix,scale,q)
    c=response[:h*10].reshape(-1,q.shape[1]);_,cv=np.linalg.eigh(c.T@c)
    cached=out/'interventions'/path.with_suffix('.npz').name
    if cached.exists():
        with np.load(cached) as x:jac=x['jacobian']
    else:
        predict,_=model.predictor(images,past,representation='all_future');a0=torch.tensor(model.normalize(nominal).reshape(-1),device=model.device,requires_grad=True)
        jac=action_jacobian(predict,a0).cpu().numpy()
    local=jac@q;_,wv=np.linalg.eigh(local.T@local)
    bases={'quintic_full':q,'controller_low2':q@cv[:,:2],'lewm_low2':q@wv[:,:2],
           'interior_values':interior_basis(h),'cubic_spline':spline_basis(h)}
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],spec['episode'],t,h,908]))
    for k in range(3):bases[f'random2_{k}']=q@np.linalg.qr(rng.normal(size=(q.shape[1],2)))[0]
    boundary={}
    for name in ['quintic_full','interior_values','cubic_spline']:
        bases[name+'_terminal'],boundary[name]=terminal_constrained(bases[name],nominal,suffix,root,prefix,scale)
    rows=[];actions=[nominal];traces=[reference['physics_trace']];controls=[reference['control_trace']];endpoints=[reference['endpoint_images']]
    coverage=[[reference['coverage_t'],reference['coverage_suffix']]];contacts=[reference['physics_contacts']]
    for name,basis in bases.items():
      for relative in cfg['radii']:
        radius=relative*np.sqrt(2*h)
        if basis.shape[1]:
            raw,info=optimize_controller(nominal,suffix,root,prefix,scale,radius,basis=basis)
        else:raw=nominal.copy();info={'solver_success':True,'iterations':0,'feasible':True}
        result=sim.execute_dense(raw,suffix,capture_endpoints=True);delta=((raw-nominal)/scale).reshape(-1)
        pred=controller_rollout(np.concatenate([raw,suffix]),root[:2],root[5:7]);predjerk=jerk_energy(np.concatenate([prefix[:,:2],pred]),.01)
        e=result['physics'][h*10-1,:2]-reference['physics'][h*10-1,:2]
        v=result['physics'][h*10-1,5:7]-reference['physics'][h*10-1,5:7]
        if name.endswith('_terminal') and max(np.linalg.norm(e),np.linalg.norm(v))>1e-3:raise ValueError('Actual terminal constraint failed')
        rows.append({k:spec[k] for k in ['state_id','split','episode','timestep','initial_pose_family','horizon']}|{
          'method':name,'relative_radius':relative,'index':len(actions),'dimension':basis.shape[1],'actual_norm':float(np.linalg.norm(delta)),
          'predicted_jerk':predjerk,'nominal_predicted_jerk':jerk_energy(np.concatenate([prefix[:,:2],controller_rollout(np.concatenate([nominal,suffix]),root[:2],root[5:7])]),.01),
          'nominal_physics_jerk':reference['physics_jerk'],'nominal_contact':bool(np.any(reference['physics_contacts'][:h*10]>0)),
          'terminal_pusher_position_delta':float(np.linalg.norm(e)),'terminal_pusher_velocity_delta':float(np.linalg.norm(v)),
          'solver_success':info['solver_success'],'solver_iterations':info['iterations'],**dense_metrics(result,reference,cfg['tolerances'])})
        actions.append(raw);traces.append(result['physics_trace']);controls.append(result['control_trace']);endpoints.append(result['endpoint_images']);coverage.append([result['coverage_t'],result['coverage_suffix']]);contacts.append(result['physics_contacts'])
    f=pd.DataFrame(rows);f.to_csv(done.with_suffix('.csv'),index=False)
    np.savez_compressed(done.with_suffix('.npz'),actions=actions,physics=traces,control=controls,endpoint_images=endpoints,coverage=coverage,contacts=contacts,
                        history_images=images,past_actions=past,root=root,basis=q,jacobian=jac,**{'basis_'+k:v for k,v in bases.items()})
    write_json(done,{**spec,'candidate_count':len(f),'dimensions':{k:v.shape[1] for k,v in bases.items()},'duration_seconds':time.perf_counter()-start,
                     'scope':'Candidate-generator/family ablation; raw pass rates and family-specific four-radius oracles are not deployable selection rules.',
                     'source_sha256':sha256(Path(__file__))});sim.close();print(done.stem,len(f),round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--horizon',type=int,choices=[5,10,20],default=5);p.add_argument('--split',choices=['development','calibration'],default='development');p.add_argument('--limit',type=int);args=p.parse_args()
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output']);torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    paths=sorted((out/'f1').glob(f'{args.split}*_h{args.horizon:02d}.json'))
    if len(paths)!=105:raise ValueError('Incomplete baseline bank')
    if args.limit:paths=paths[:args.limit]
    for path in paths:process(path,cfg,data,model,out)
if __name__=='__main__':main()
