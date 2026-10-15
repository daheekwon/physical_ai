"""F2 matched intervention bank, with dense execution and native WM prediction."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,dense_metrics,controller_design
from zj_wam.diagnostics import polynomial_basis,frequency_matched_directions,temporal_properties
from zj_wam.sensitivity import action_jacobian,eigensystem,symmetric_bound_radius
from zj_wam.model_audit import predict_candidates
from zj_wam.io import resolve,write_json,sha256
from scripts.run_followup import initialize,audit


def run_state(spec,cfg,data,model,out,h):
    directory=out/'interventions';directory.mkdir(exist_ok=True)
    key=f"{spec['state_id']}_h{h:02d}";done=directory/f'{key}.json'
    if done.exists():return
    started=time.perf_counter();actions,states=data.episode(spec['episode']);t=spec['timestep']
    nominal=actions[t:t+h];suffix=actions[t+h:t+h+3]
    sim=DenseSimulator(actions,states,t,cfg['seed']+spec['episode'])
    images,past=sim.restore(capture=True)
    reference=sim.execute_dense(nominal,suffix,restore=False,capture_endpoints=True)
    predict,context=model.predictor(images,past)
    a0=torch.as_tensor(model.normalize(nominal).reshape(-1),device=model.device)
    jac=action_jacobian(predict,a0);eig,v=eigensystem(jac)
    q=polynomial_basis(h);_,pv=eigensystem(jac@torch.as_tensor(q,dtype=jac.dtype,device=jac.device))
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],spec['episode'],t,h,202]))
    dirs={'free_low':v[:,0],'free_high':v[:,-1],
          'poly_low_0':q@pv[:,0],'poly_low_1':q@pv[:,1],'poly_high':q@pv[:,-1]}
    for i in range(3):
        d=rng.normal(size=2*h);dirs[f'free_random_{i}']=d/np.linalg.norm(d)
        c=rng.normal(size=2);dirs[f'poly_low_mix_{i}']=q@pv[:,:2]@(c/np.linalg.norm(c))
    dirs.update({f'coordinate_{i}':d for i,d in enumerate(np.eye(2*h))})
    dirs.update({f'frequency_{i}':d for i,d in enumerate(frequency_matched_directions(v[:,0],rng,cfg['frequency_directions']))})
    _,_,response,_,_=controller_design(nominal,suffix,reference['root_full_state'],sim.prefix_physics,data.scaler.scale_,basis=np.eye(2*h))
    controller=response[:h*10].reshape(-1,2*h)
    _,cv=np.linalg.eigh(controller.T@controller)
    for i,angle in enumerate(np.arange(4)*np.pi/4):dirs[f'controller_{i}']=cv[:,:2]@np.array([np.cos(angle),np.sin(angle)])
    dirs={k:d/np.linalg.norm(d) for k,d in dirs.items()}
    a0_np=a0.cpu().numpy();low=np.tile((-1-data.scaler.mean_)/data.scaler.scale_,h);high=np.tile((1-data.scaler.mean_)/data.scaler.scale_,h)
    bound=symmetric_bound_radius(a0_np,dirs.values(),low,high)
    raw_bank=[nominal];metadata=[]
    for relative in cfg['radii']:
        radius=min(relative*np.sqrt(2*h),bound*.999)
        if radius<1e-7:continue
        for name,d in dirs.items():
            for sign in [-1,1]:
                raw=model.denormalize(a0_np+sign*radius*d)
                if np.max(np.abs(raw))>1+1e-6:raise AssertionError('Action bound violation')
                metadata.append({'method':name,'sign':sign,'relative_radius':relative,'applied_radius':radius,
                                 'polynomial':name.startswith('poly_'),**temporal_properties(d)})
                raw_bank.append(raw)
    predictions=predict_candidates(model,context['embedding'],context['past'],raw_bank)
    with torch.no_grad():single=predict(a0).cpu().numpy()
    np.testing.assert_allclose(predictions[0,-1],single,atol=2e-5,rtol=2e-4)
    rows=[];traces=[reference['physics_trace']];controls=[reference['control_trace']]
    images_all=[reference['endpoint_images']];coverages=[[reference['coverage_t'],reference['coverage_suffix']]];contacts=[reference['physics_contacts']]
    for idx,(raw,meta) in enumerate(zip(raw_bank[1:],metadata),1):
        result=sim.execute_dense(raw,suffix,capture_endpoints=True)
        metrics=dense_metrics(result,reference,cfg['tolerances'])
        endpoint_delta=result['control'][h-1,2:5]-reference['control'][h-1,2:5]
        endpoint_delta[2]=np.arctan2(np.sin(endpoint_delta[2]),np.cos(endpoint_delta[2]))
        delta=((raw-nominal)/data.scaler.scale_).reshape(-1)
        nominal_contact=bool(np.any(reference['physics_contacts'][:h*10]>0));modified_contact=bool(np.any(result['physics_contacts'][:h*10]>0))
        rows.append({**spec,'horizon':h,'index':idx,**meta,**metrics,
                     'actual_norm':float(np.linalg.norm(delta)),'predicted_change':float(np.linalg.norm(predictions[idx,-1]-predictions[0,-1])),
                     'linear_change':float(np.linalg.norm(jac.cpu().numpy()@delta)),
                     'nominal_contact':nominal_contact,'modified_contact':modified_contact,
                     'near_miss':not modified_contact and reference['initial_gap']<=cfg['near_contact_gap'],
                     'contact_transition':nominal_contact!=modified_contact,
                     'initial_gap':reference['initial_gap'],
                     'endpoint_dx':float(endpoint_delta[0]),'endpoint_dy':float(endpoint_delta[1]),'endpoint_dtheta':float(endpoint_delta[2]),
                     'nominal_physics_jerk':reference['physics_jerk']})
        traces.append(result['physics_trace']);controls.append(result['control_trace']);images_all.append(result['endpoint_images'])
        coverages.append([result['coverage_t'],result['coverage_suffix']]);contacts.append(result['physics_contacts'])
    pd.DataFrame(rows).to_csv(directory/f'{key}.csv',index=False)
    np.savez_compressed(directory/f'{key}.npz',actions=raw_bank,physics=traces,control=controls,
                        endpoint_images=images_all,coverage=coverages,contacts=contacts,history_images=images,past_actions=past,
                        root=reference['root_full_state'],basis=q,jacobian=jac.cpu().numpy(),predictions=predictions,
                        history_embedding=context['embedding'].cpu().numpy(),directions=np.stack(list(dirs.values())),direction_names=list(dirs))
    record={**spec,'horizon':h,'candidate_count':len(rows),'directions':len(dirs),'nominal_contact':bool(np.any(reference['physics_contacts'][:h*10]>0)),
            'near_miss_count':sum(r['near_miss'] for r in rows),'contact_transition_count':sum(r['contact_transition'] for r in rows),
            'duration_seconds':time.perf_counter()-started,'source_sha256':sha256(Path(__file__))}
    write_json(done,record);sim.close();print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='development',choices=['development','calibration','test']);p.add_argument('--limit',type=int)
    p.add_argument('--horizon',type=int,default=5);args=p.parse_args()
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output'])
    if args.split=='test' and not (out/'protocol/final_selection_lock.json').exists():raise ValueError('Final selection lock required')
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    data=ExpertDataset(cfg['dataset']);manifest=initialize(cfg,data,out)
    model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    states=[s for s in manifest['states'] if s['split']==args.split]
    if args.limit:states=states[:args.limit]
    for spec in states:
        if audit(spec,cfg,data,out)['eligible']:run_state(spec,cfg,data,model,out,args.horizon)

if __name__=='__main__':main()
