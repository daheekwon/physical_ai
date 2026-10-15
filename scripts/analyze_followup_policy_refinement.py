"""Closed-loop fixed-time global vs matched-state local preservation, with raw audit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.policy_refinement import absolute_to_relative
from zj_wam.diagnostics import jerk_energy
from scripts.analyze_followup_response_selection import mean_ci
from zj_wam.io import resolve,write_json


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--split',default='development');parser.add_argument('--smoke',action='store_true');parser.add_argument('--partial',action='store_true');args=parser.parse_args()
 cfg=json.loads(resolve('configs/followup_v1.json').read_text());pcfg=json.loads(resolve('configs/followup_policy_refinement.json').read_text());out=resolve(cfg['output'])/('policy_refinement_smoke' if args.smoke else 'policy_refinement')/args.split
 expected=pcfg[args.split+'_seeds'][:1] if args.smoke else pcfg[args.split+'_seeds'];methods=pcfg['methods'];complete=[seed for seed in expected if all((out/f'{seed}_{m}.json').exists() for m in methods)]
 if not args.partial and len(complete)!=len(expected):raise ValueError(f'Incomplete paired seeds {len(complete)}/{len(expected)}')
 if not complete:raise ValueError('No complete paired seeds')
 rows=[];locals=[];maxima={};tol=cfg['tolerances']
 def check(name,x,y,atol=1e-7):
  error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));maxima[name]=max(maxima.get(name,0),error)
  if error>atol:raise ValueError((name,error))
 for seed in complete:
  with np.load(out/f'{seed}_nominal.npz') as a:reference=a['physics'];refcov=a['coverage'];initial=a['initial_state'];refactions=a['actions']
  refj=jerk_energy(np.vstack([initial[:2],reference[:,:2]]),.01)
  for method in methods:
   p=out/f'{seed}_{method}.json';meta=json.loads(p.read_text());local=pd.read_csv(p.with_suffix('.csv'));a=np.load(p.with_suffix('.npz'));actions=a['actions'];trace=a['physics'];coverage=a['coverage'];states=a['states'];n=len(actions)
   assert n==len(refactions) and len(trace)==10*n and len(states)==n+1 and len(meta['decisions'])==(n+7)//8
   check('identical_initial',a['initial_state'],initial);check('control_states_match_trace',states[1:],trace[9::10]);check('raw_action_bound',np.maximum(0,np.abs(actions)-1),0.,1e-6)
   for k,d in enumerate(meta['decisions']):
    t=d['control_step'];assert t==8*k;end=min(t+8,n);eligible=t>=10 and end-t==8;assert d['eligible']==eligible
    assert d['policy_observation_times']==[max(0,t-1),t] and d['policy_noise_seed']==(cfg['seed']+1009*seed+t)%(2**31-1)
    if eligible:assert d['wm_observation_times']==[t-10,t-5,t]
    else:assert not d['wm_observation_times'] and not d['active']
    nominal=a['nominal_relative_chunks'][k,:end-t];compiled,info=absolute_to_relative(a['policy_action_predictions'][k,1:1+end-t],states[t,:2],states[t,5:7]);check('absolute_relative_compilation',nominal,compiled,1e-9);assert info['clipped_actions']==d['clipped_actions']
    if eligible:
     check('unchanged_relative_suffix',actions[t+5:end],nominal[5:]);delta=actions[t:t+5]-nominal[:5]
     if method!='gaussian_action':check('polynomial_endpoint_correction',delta[[0,-1]],0.,1e-6)
    else:check('unrefined_warmup_tail',actions[t:end],nominal)
   for k,r in enumerate(local.itertuples(index=False)):
    t=r.control_step;actual=trace[t*10:(t+8)*10];baseline=a['local_nominal_physics'][k];angle=actual[:,4]-baseline[:,4];pos=np.linalg.norm(actual[:,2:4]-baseline[:,2:4],axis=1).max();rot=np.abs(np.arctan2(np.sin(angle),np.cos(angle))).max();loss=np.maximum(0,a['local_nominal_coverage'][k,[4,7]]-coverage[[t+4,t+7]]).max();error=max(pos/tol['position'],rot/tol['rotation'],loss/tol['coverage_loss'])
    prefix=trace[t*10-30:t*10,:2];j0=jerk_energy(np.concatenate([prefix,baseline[:,:2]]),.01);j1=jerk_energy(np.concatenate([prefix,actual[:,:2]]),.01);gain=(j0-j1)/j0 if j0>1e-8 else 0
    check('local_object_error',error,r.object_error);check('local_jerk_gain',gain,r.jerk_gain)
    if method=='nominal':check('nominal_local_counterfactual',actual,baseline,1e-9)
   local['initial_pose_family']=seed;local['local_joint']=local.active & (local.object_error<=1)&(local.jerk_gain>=.1);local['violation']=local.object_error>1;locals.append(local)
   pos=np.linalg.norm(trace[:,2:4]-reference[:,2:4],axis=1).max();angle=trace[:,4]-reference[:,4];rot=np.abs(np.arctan2(np.sin(angle),np.cos(angle))).max();loss=np.maximum(0,refcov-coverage).max();error=max(pos/tol['position'],rot/tol['rotation'],loss/tol['coverage_loss']);j=jerk_energy(np.vstack([initial[:2],trace[:,:2]]),.01);gain=(refj-j)/refj if refj>1e-8 else 0
   check('physics_jerk_relative',j/max(1,j),meta['physics_jerk']/max(1,j))
   active=meta['active_chunks']>0
   rows.append({'seed':seed,'initial_pose_family':seed,'method':method,'active':active,'object_error':float(error),'position_max':float(pos),'rotation_max_deg':float(np.rad2deg(rot)),'coverage_max_loss':float(loss),'jerk_gain':float(gain),'joint':bool(active and error<=1 and gain>=.1),'any_success':meta['any_success'],'final_success':coverage[-1]>.95,'max_score':meta['max_score'],'active_chunk_fraction':meta['active_chunks']/max(1,meta['eligible_chunks']),'mean_policy_seconds':np.mean(meta['policy_inference_seconds']),'refinement_seconds':meta['refinement_seconds'],'nominal_projection_count':sum(d['clipped_actions'] for d in meta['decisions']),'all_local_preserved':bool((local.object_error<=1).all()),'local_pass_global_fail':bool((local.object_error<=1).all() and error>1)})
 f=pd.DataFrame(rows);local=pd.concat(locals,ignore_index=True);summaries=[];local_summary=[]
 for method,g in f.groupby('method'):
  active=g[g.active].copy();active['violation']=active.object_error>1
  summaries.append({'method':method,'episodes':len(g),**{c:mean_ci(g,c) for c in ['joint','active','jerk_gain','any_success','final_success','max_score','active_chunk_fraction','all_local_preserved','local_pass_global_fail','mean_policy_seconds','refinement_seconds']},'risk':mean_ci(active,'violation')})
 for method,g in local.groupby('method'):
  active=g[g.active];local_summary.append({'method':method,'blocks':len(g),'joint':mean_ci(g,'local_joint'),'active':mean_ci(g,'active'),'risk':mean_ci(active,'violation')})
 f.to_csv(out/'evaluation.csv',index=False);local.to_csv(out/'local_evaluation.csv',index=False)
 write_json(out/'analysis.json',{'scope':'Smoke only; implementation verification, not study performance' if args.smoke else 'F10 policy development/calibration diagnostic with unchanged expert-H5 rules; no risk guarantee under policy shift. Fresh legacy environment seeds. Full300 action simulation, no success-based early exit. Global reference is independent nominal-policy execution with common per-call random seed; local reference from each methods own state.',
 'expected_seeds':len(expected),'completed_paired_seeds':complete,'incomplete_seeds':[x for x in expected if x not in complete],'summary':summaries,'local_summary':local_summary})
 write_json(out/'raw_audit.json',{'paired_seeds':complete,'state_method_rows':len(f),'local_blocks':len(local),'max_absolute_recalculation_error':maxima,'checks':['native absolute-to-relative action compilation','causal observation timestamps','unaltered relative suffix','warmup/tail nominal','polynomial endpoints','raw action bounds','full physics/control alignment','local counterfactual E and jerk','global jerk and metrics','nominal local replay exactness']})
 for r in summaries:print(r['method'],{k:r[k]['mean'] for k in ['joint','risk','any_success','jerk_gain']})
if __name__=='__main__':main()
