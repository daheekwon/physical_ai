"""F5 actual Jacobian refresh frequency with matched rank and finite verifier."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch,joblib
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,dense_metrics,controller_rollout,optimize_controller
from zj_wam.diagnostics import polynomial_basis,jerk_energy
from zj_wam.repeated_refinement import visual_history
from zj_wam.sensitivity import action_jacobian
from zj_wam.model_audit import predict_candidates
from zj_wam.readouts import decode,effects
from zj_wam.selection import select_indices
from zj_wam.io import resolve,write_json,sha256
from scripts.run_followup_repeated_refinement import execute_with_coverage,core_metrics


def bank(nominal,suffix,root,prefix,past,scale,basis,seed,radii):
 rng=np.random.default_rng(seed);directions=rng.normal(size=(8,basis.shape[1]));directions/=np.linalg.norm(directions,axis=1)[:,None]
 actions=[nominal]
 for relative in radii:
  radius=relative*np.sqrt(10)
  for objective in ['executed','raw']:
   raw,_=optimize_controller(nominal,suffix,root,prefix,scale,radius,objective,past[-3:],basis=basis);actions.append(raw)
  for d in directions:
   for sign in [-1,1]:
    raw=nominal+sign*radius*(basis@d).reshape(5,2)*scale
    if np.max(np.abs(raw))<=1+1e-6:actions.append(raw)
 return np.array(actions)


def process(path,cfg,rcfg,acfg,data,model,probe,rule,out):
 spec=json.loads(path.read_text());spec={k:spec[k] for k in ['state_id','split','episode','timestep','initial_pose_family']};dest=out/'relinearization';dest.mkdir(exist_ok=True);done=dest/(spec['state_id']+'.json')
 if done.exists():return
 started=time.perf_counter();original,recorded=data.episode(spec['episode']);t=spec['timestep'];suffix=original[t+20:t+30]
 sim=DenseSimulator(original,recorded,t,cfg['seed']+spec['episode']);reference=execute_with_coverage(sim,original[t:t+20],suffix);sim.close()
 rows=[];local=[];artifact={'nominal_physics':reference['physics_trace'],'nominal_actions':original[t:t+20]};counts={}
 for method,frequency in acfg['refresh_actions'].items():
  delivered=original.copy();basis=None;relinearizations=0;active_chunks=0;method_start=time.perf_counter();bases=[]
  for offset in range(0,20,5):
   current=t+offset;nominal=original[current:current+5];local_suffix=original[current+5:current+8]
   sim=DenseSimulator(delivered,recorded,current,cfg['seed']+spec['episode']);images,past=sim.restore(capture=True)
   observed,_=visual_history(images,sim.context_control,rcfg);root=observed[-1];prefix=sim.prefix_physics.copy()
   predict,context=model.predictor(images,past)
   if basis is None or offset%frequency==0:
    a0=torch.tensor(model.normalize(nominal).reshape(-1),device=model.device,requires_grad=True)
    jac=action_jacobian(predict,a0).cpu().numpy();q=polynomial_basis(5);small=jac@q;_,vectors=np.linalg.eigh(small.T@small);basis=q@vectors[:,:acfg['rank']]
    for i in range(basis.shape[1]):basis[:,i]*=np.sign(basis[np.argmax(np.abs(basis[:,i])),i])
    relinearizations+=1
   bases.append(basis.copy())
   actions=bank(nominal,local_suffix,root,prefix,past,data.scaler.scale_,basis,np.random.SeedSequence([cfg['seed'],spec['episode'],current,5]),cfg['radii'])
   predicted=predict_candidates(model,context['embedding'],context['past'],actions);scores=np.linalg.norm(predicted-predicted[:1],axis=-1).max(1)
   costs=np.array([jerk_energy(np.concatenate([prefix[:,:2],controller_rollout(np.concatenate([a,local_suffix]),root[:2],root[5:7])]),.01) for a in actions])
   norms=np.linalg.norm(((actions-actions[:1])/data.scaler.scale_).reshape(len(actions),-1),axis=1)
   frame=pd.DataFrame({'state_id':['current']*len(actions),'horizon':5,'actual_norm':norms,'predicted_jerk':costs,'nominal_predicted_jerk':costs[0],'lewm_global_score':scores})
   index=max(0,select_indices(frame,'lewm_global',rule)['current']);chosen=actions[index];active_chunks+=int(norms[index]>1e-6)
   predicted_effect=effects(decode(probe,predicted))
   # Counterfactual pair is evaluated only AFTER index and action are fixed.
   nominal_result=sim.execute_dense(nominal,local_suffix)
   chosen_result=sim.execute_dense(chosen,local_suffix)
   true_effect=chosen_result['control'][4,2:5]-nominal_result['control'][4,2:5];true_effect[2]=np.arctan2(np.sin(true_effect[2]),np.cos(true_effect[2]))
   error=predicted_effect[index,0]-true_effect;error[2]=np.arctan2(np.sin(error[2]),np.cos(error[2]))
   contact=(nominal_result['physics_contacts'][:50].reshape(5,10)>0).any(1)
   local.append({**spec,'method':method,'offset':offset,'selected_index':index,'norm':float(norms[index]),'score':float(scores[index]),
       'local_contact':bool(contact.any()),'local_contact_transition':bool(np.any(contact[1:]!=contact[:-1])),
       'position_effect':float(np.linalg.norm(true_effect[:2])),'rotation_effect_deg':float(np.rad2deg(abs(true_effect[2]))),
       'position_effect_error':float(np.linalg.norm(error[:2])),'rotation_effect_error_deg':float(np.rad2deg(abs(error[2]))),
       'local_object_error':dense_metrics(chosen_result,nominal_result,cfg['tolerances'])['object_error']})
   delivered[current:current+5]=chosen;sim.close()
  sim=DenseSimulator(original,recorded,t,cfg['seed']+spec['episode']);result=execute_with_coverage(sim,delivered[t:t+20],suffix);sim.close()
  core=core_metrics(result,reference,cfg['tolerances']);long=dense_metrics(result,reference,cfg['tolerances']);long['object_error']=max(core['object_error'],long['object_error']);long['preserved']=long['object_error']<=1
  nominal_contact=(reference['physics_contacts'][:200].reshape(20,10)>0).any(1)
  rows.append({**spec,'method':method,'horizon':20,'suffix_actions':len(suffix),'nominal_contact':bool(nominal_contact.any()),'nominal_contact_transition':bool(np.any(nominal_contact[1:]!=nominal_contact[:-1])),
      'active_chunks':active_chunks,'relinearizations':relinearizations,'seconds':time.perf_counter()-method_start,
      'actual_norm':float(np.linalg.norm(((delivered[t:t+20]-original[t:t+20])/data.scaler.scale_).reshape(-1))),**core,**{'long_'+k:v for k,v in long.items()}})
  artifact[method+'_actions']=delivered[t:t+20];artifact[method+'_physics']=result['physics_trace'];artifact[method+'_basis']=np.array(bases);counts[method]=relinearizations
 pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);pd.DataFrame(local).to_csv(dest/(spec['state_id']+'_local.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**artifact)
 write_json(done,{**spec,'relinearization_counts':counts,'seconds':time.perf_counter()-started,'source_sha256':sha256(Path(__file__)),'scope':acfg['scope']});print(done.stem,round(time.perf_counter()-started,2),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration'],default='development');p.add_argument('--limit',type=int);args=p.parse_args()
 cfg=json.loads(resolve('configs/followup_v1.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());acfg=json.loads(resolve('configs/followup_relinearization.json').read_text());out=resolve(cfg['output']);torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
 probe=joblib.load(out/'lewm_audit/readouts/projected_cls_ridge.joblib');rule=json.loads((out/'selection/selector_rules.json').read_text())['rules']['lewm_global']
 paths=sorted((out/'f1').glob(f'{args.split}*h05.json'))
 if len(paths)!=105:raise ValueError('Incomplete source states')
 if args.limit:paths=paths[:args.limit]
 for path in paths:process(path,cfg,rcfg,acfg,data,model,probe,rule,out)
if __name__=='__main__':main()
