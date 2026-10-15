"""Fresh every-branch technical audit of fixed-onset repeated shifts."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.shifted_execution import noisy_history
from zj_wam.followup import dense_metrics
from scripts.repeated_shift_core import simulator,branch,select,clean_robot_proprio,core_metrics
BASE=resolve('artifacts/followup_v1');OUT=BASE/'repeated_shifts_smoke';read=lambda p:json.loads(p.read_text())

def main():
 m=read(OUT/'manifest.json');done=read(OUT/'execution_complete.json');assert done['smoke'] and done['executions']==21 and done['local_decisions']==84 and len(done['nominal_fixture'])==3;assert done['manifest_sha256']==sha256(OUT/'manifest.json')
 for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
 cfg=read(resolve('configs/followup_v1.json'));rcfg=read(resolve('configs/followup_response_models.json'));data=ExpertDataset(cfg['dataset']);torch.set_num_threads(cfg['threads']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);rules=read(BASE/'selection/selector_rules.json')['rules'];strong=read(BASE/'strong_baselines/calibrated_settings.json')['rules'];conditions=read(resolve('configs/followup_shift_evaluation.json'))['conditions'];meta=read(resolve(m['sources'][0]));original,recorded=data.episode(meta['episode']);root=meta['timestep'];seed=cfg['seed']+meta['episode'];suffix=original[root+20:root+30];freshbranches=0;freshselectors=0;cases=0;maxerror={}
 def check(name,x,y,tol=0.):
  x=np.asarray(x,float);y=np.asarray(y,float);np.testing.assert_allclose(x,y,atol=tol,rtol=0,equal_nan=True,err_msg=name);maxerror[name]=max(maxerror.get(name,0.),float(abs(x-y).max()) if x.size else 0.)
 def independent_error(r,n):
  d=r['physics'][:,2:5]-n['physics'][:,2:5];p=np.sqrt(np.sum(d[:,:2]**2,axis=1)).max()/2;angle=np.abs((d[:,2]+np.pi)%(2*np.pi)-np.pi).max()/np.deg2rad(2);cov=max(0,n['coverage_t']-r['coverage_t'],n['coverage_suffix']-r['coverage_suffix'])/.02;return max(p,angle,cov)
 for condition,params in m['conditions'].items():
  ref=branch(original,recorded,root,seed,params,original[root:root+20],suffix);freshbranches+=1
  for method in m['config']['methods']:
   p=OUT/condition/f"{meta['state_id']}__{method}.json";record=read(p);assert record['data_sha256']==sha256(p.with_suffix('.npz')) and record['local_csv_sha256']==sha256(p.with_suffix('.csv'));stored=dict(np.load(p.with_suffix('.npz')));local=pd.read_csv(p.with_suffix('.csv'));delivered=original.copy()
   for offset in range(0,20,5):
    current=root+offset;sim=simulator(delivered,recorded,current,root,seed,params)
    try:
     clean,past=sim.restore(capture=True);images=noisy_history(clean,params,np.random.SeedSequence([cfg['seed'],meta['episode'],current,list(conditions).index(condition)]));proprio=clean_robot_proprio(sim.context_control);get=lambda k:stored[f'block{offset:02d}_{k}'];assert sim.shift_step==max(0,root-10)
     for k,v in [('clean_images',clean),('images',images),('past',past),('proprio',proprio),('prefix',sim.prefix_physics),('root',sim.root_full_state),('nominal',original[current:current+5]),('suffix',original[current+5:current+8])]:check('context_'+k,v,get(k))
     assert not np.any(proprio[:,[2,3,4,7,8,9]])
     before=p.with_name(p.stem+f'.block{offset:02d}.before_physics.npz');bm=read(before.with_suffix('.json'));assert bm['data_sha256']==sha256(before)
     with np.load(before) as a:
      for k in a:check('prephysics_'+k,a[k],get(k))
     expected,info=select(method,model,images,past,proprio,sim.prefix_physics,original[current:current+5],original[current+5:current+8],data.scaler.scale_,np.random.SeedSequence([cfg['seed'],meta['episode'],current,5]),cfg,rcfg,rules,strong);freshselectors+=1;expected=np.asarray(expected,dtype=original.dtype);check('fresh_selected_action',expected,get('applied'))
     delivered[current:current+5]=expected;assert np.max(abs(expected))<=1+1e-6
     nominal=sim.execute_dense(original[current:current+5],original[current+5:current+8]);actual=sim.execute_dense(expected,original[current+5:current+8]);freshbranches+=2
     for label,r in [('nominal',nominal),('actual',actual)]:
      for k in ['physics','control','physics_trace','control_trace','physics_contacts']:check('fresh_local_'+k,r[k],get(label+'_'+k))
     check('local_coverage',[[r['coverage_t'],r['coverage_suffix']] for r in [nominal,actual]],get('coverage'))
     check('independent_local_E',independent_error(actual,nominal),local.iloc[offset//5].object_error,2e-10)
    finally:sim.close()
   result=branch(original,recorded,root,seed,params,delivered[root:root+20],suffix);freshbranches+=1;check('delivered_actions',delivered[root:root+20],stored['actions'])
   for label,r in [('nominal',ref),('modified',result)]:
    for k in ['physics','control','physics_trace','control_trace','physics_contacts','root_full_state']:check('fresh_global_'+k,r[k],stored[label+'_'+k])
    check('global_coverage',[r['coverage_t'],r['coverage_core_suffix'],r['coverage_suffix']],stored[label+'_coverage'])
   core=core_metrics(result,ref,cfg['tolerances']);long=dense_metrics(result,ref,cfg['tolerances']);check('independent_long_E',independent_error(result,ref),long['object_error'],2e-10)
   for k,v in core.items():check('metric_'+k,v,record['summary'][k],1e-8)
   for k,v in long.items():
    if k=='object_error':v=max(v,core['object_error'])
    if k=='preserved':v=max(long['object_error'],core['object_error'])<=1
    check('longmetric_'+k,v,record['summary']['long_'+k],1e-8)
   cases+=1;print('AUDIT REPEATED SHIFT',condition,method,cases,freshbranches,flush=True)
 assert cases==21 and freshselectors==84
 write_json(OUT/'raw_audit.json',{'smoke':True,'executions':cases,'fresh_selectors':freshselectors,'fresh_physical_branches':freshbranches,'max_errors':maxerror,'manifest_sha256':sha256(OUT/'manifest.json'),'execution_complete_sha256':sha256(OUT/'execution_complete.json'),'source_sha256':sha256(Path(__file__)),'scope':'Every smoke history/root/onset/prephysics record, every fresh selection and localnominal/corrected branch, all globalnominal/corrected replay and metrics. Independent pose-based E. Nominal3originalmethod fixtures exact. Technical plumbing only; no full105state performance claim.'})
if __name__=='__main__':
 with threadpool_limits(limits=2,user_api='blas'):main()
