"""All-state fresh physical/selector audit for repeated correction under shifts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.shifted_execution import noisy_history
from zj_wam.followup import dense_metrics
from scripts.repeated_shift_core import simulator,branch,select,clean_robot_proprio,core_metrics
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

HELPERS=['scripts/repeated_shift_core.py','scripts/run_followup_repeated_shifts.py','scripts/run_followup_repeated_refinement.py','zj_wam/shifted_execution.py','zj_wam/followup.py','zj_wam/repeated_refinement.py','zj_wam/locked_geometric.py','zj_wam/selection.py']
def main(smoke):
 OUT=BASE/('repeated_shifts_smoke' if smoke else 'repeated_shifts');m=read(OUT/'manifest.json');done=read(OUT/'execution_complete.json');n=21 if smoke else 5880;assert done['smoke']==smoke and done['executions']==n and done['local_decisions']==4*n;assert done['manifest_sha256']==sha256(OUT/'manifest.json')
 for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
 cfg=read(resolve('configs/followup_v1.json'));rcfg=read(resolve('configs/followup_response_models.json'));data=ExpertDataset(cfg['dataset']);torch.set_num_threads(cfg['threads']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);rules=read(BASE/'selection/selector_rules.json')['rules'];strong=read(BASE/'strong_baselines/calibrated_settings.json')['rules'];conditions=read(resolve('configs/followup_shift_evaluation.json'))['conditions'];freshbranches=0;freshselectors=0;cases=0;maxerror={};rows=[];inputs={};helpers={p:sha256(resolve(p)) for p in HELPERS}
 def check(name,x,y,tol=0.):
  x=np.asarray(x,float);y=np.asarray(y,float);np.testing.assert_allclose(x,y,atol=tol,rtol=0,equal_nan=True,err_msg=name);maxerror[name]=max(maxerror.get(name,0.),float(abs(x-y).max()) if x.size else 0.)
 def independent_error(r,n):
  d=r['physics'][:,2:5]-n['physics'][:,2:5];p=np.sqrt(np.sum(d[:,:2]**2,axis=1)).max()/2;angle=np.abs((d[:,2]+np.pi)%(2*np.pi)-np.pi).max()/np.deg2rad(2);cov=max(0,n['coverage_t']-r['coverage_t'],n['coverage_suffix']-r['coverage_suffix'])/.02;return max(p,angle,cov)

 def independent_jerk(actual,nominal,recorded_gain):
  def cost(trace):
   q=trace[:,:2];d=q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3];return float(np.sum(d*d)/.01**5)
  a,b=cost(actual),cost(nominal);gain=(b-a)/b if b>1e-8 else 0.;check('independent_jerk_gain',gain,recorded_gain,2e-8)
 for name in m['sources']:
  meta=read(resolve(name));original,recorded=data.episode(meta['episode']);root=meta['timestep'];seed=cfg['seed']+meta['episode'];suffix=original[root+20:root+30]
  for condition,params in m['conditions'].items():
   ref=branch(original,recorded,root,seed,params,original[root:root+20],suffix);freshbranches+=1
   for method in m['config']['methods']:
    p=OUT/condition/f"{meta['state_id']}__{method}.json";record=read(p);assert record['manifest_sha256']==sha256(OUT/'manifest.json');assert record['summary']['state_id']==meta['state_id'] and record['summary']['condition']==condition and record['summary']['method']==method;assert record['data_sha256']==sha256(p.with_suffix('.npz')) and record['local_csv_sha256']==sha256(p.with_suffix('.csv'));stored=dict(np.load(p.with_suffix('.npz')));local=pd.read_csv(p.with_suffix('.csv'));delivered=original.copy()
    for offset in range(0,20,5):
     current=root+offset;sim=simulator(delivered,recorded,current,root,seed,params)
     try:
      clean,past=sim.restore(capture=True);images=noisy_history(clean,params,np.random.SeedSequence([cfg['seed'],meta['episode'],current,list(conditions).index(condition)]));proprio=clean_robot_proprio(sim.context_control);get=lambda k:stored[f'block{offset:02d}_{k}'];assert sim.shift_step==max(0,root-10)
      for k,v in [('clean_images',clean),('images',images),('past',past),('proprio',proprio),('prefix',sim.prefix_physics),('root',sim.root_full_state),('nominal',original[current:current+5]),('suffix',original[current+5:current+8])]:check('context_'+k,v,get(k))
      assert not np.any(proprio[:,[2,3,4,7,8,9]])
      before=p.with_name(p.stem+f'.block{offset:02d}.before_physics.npz');bm=read(before.with_suffix('.json'));assert bm['data_sha256']==sha256(before) and bm['manifest_sha256']==sha256(OUT/'manifest.json');assert bm['decision']==record['decisions'][offset//5]
      with np.load(before) as a:
       for k in a:check('prephysics_'+k,a[k],get(k))
      expected,info=select(method,model,images,past,proprio,sim.prefix_physics,original[current:current+5],original[current+5:current+8],data.scaler.scale_,np.random.SeedSequence([cfg['seed'],meta['episode'],current,5]),cfg,rcfg,rules,strong);freshselectors+=1;expected=np.asarray(expected,dtype=original.dtype);check('fresh_selected_action',expected,get('applied'))
      delivered[current:current+5]=expected;assert np.max(abs(expected))<=1+1e-6;norm=float(np.linalg.norm((expected-original[current:current+5])/data.scaler.scale_));check('applied_norm',norm,bm['decision']['applied_norm']);assert bm['decision']['applied_active']==bool(norm>1e-6)
      nominal=sim.execute_dense(original[current:current+5],original[current+5:current+8]);actual=sim.execute_dense(expected,original[current+5:current+8]);freshbranches+=2
      for label,r in [('nominal',nominal),('actual',actual)]:
       for k in ['physics','control','physics_trace','control_trace','physics_contacts']:check('fresh_local_'+k,r[k],get(label+'_'+k))
      check('local_coverage',[[r['coverage_t'],r['coverage_suffix']] for r in [nominal,actual]],get('coverage'))
      check('independent_local_E',independent_error(actual,nominal),local.iloc[offset//5].object_error,2e-10)
      norm=float(np.linalg.norm((expected-original[current:current+5])/data.scaler.scale_));assert bool(local.iloc[offset//5].active)==bool(norm>1e-6)
      independent_jerk(actual['physics_trace'],nominal['physics_trace'],local.iloc[offset//5].jerk_gain)
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
    norm=float(np.linalg.norm((stored['actions']-stored['nominal_actions'])/data.scaler.scale_));row=record['summary'];check('global_applied_norm',norm,row['actual_norm']);assert row['active']==bool(norm>1e-6);assert row['joint']==bool(norm>1e-6 and core['object_error']<=1 and core['jerk_gain']>=.1);assert row['long_joint']==bool(norm>1e-6 and max(core['object_error'],long['object_error'])<=1 and long['jerk_gain']>=.1)
    assert row['nominal_task_success']==ref['nominal_success'] and row['modified_task_success']==result['nominal_success']
    independent_jerk(result['physics_trace'][:260],ref['physics_trace'][:260],core['jerk_gain']);independent_jerk(result['physics_trace'],ref['physics_trace'],long['jerk_gain'])
    rows.append(row);cases+=1
    inputs[str(p.relative_to(resolve('.')))]=sha256(p);inputs[str(p.with_suffix('.npz').relative_to(resolve('.')))]=sha256(p.with_suffix('.npz'));inputs[str(p.with_suffix('.csv').relative_to(resolve('.')))]=sha256(p.with_suffix('.csv'))
    print('AUDIT REPEATED SHIFT',meta['state_id'],condition,method,cases,flush=True)
  write_json(OUT/'audit_progress.json',{'states_done':len({r['state_id'] for r in rows}),'executions':cases,'fresh_selectors':freshselectors,'fresh_physical_branches':freshbranches,'last_state':meta['state_id']})
 assert cases==n and freshselectors==4*n
 exported=pd.read_csv(OUT/'episodes.csv').sort_values(['condition','method','state_id']).reset_index(drop=True);rebuilt=pd.DataFrame(rows).sort_values(['condition','method','state_id']).reset_index(drop=True);assert len(exported)==len(rebuilt)
 for col in rebuilt:
  if col in ['state_id','split','condition','method']:np.testing.assert_array_equal(exported[col],rebuilt[col])
  else:np.testing.assert_allclose(exported[col].to_numpy(float),rebuilt[col].to_numpy(float),atol=1e-7,rtol=1e-10,equal_nan=True,err_msg=col)
 for p,h in {**m['inputs'],**helpers}.items():assert sha256(resolve(p))==h,p
 write_json(OUT/('full_audit_smoke.json' if smoke else 'raw_audit.json'),{'smoke':smoke,'states':len(m['sources']),'executions':cases,'fresh_selectors':freshselectors,'fresh_physical_branches':freshbranches,'max_errors':maxerror,'manifest_sha256':sha256(OUT/'manifest.json'),'execution_complete_sha256':sha256(OUT/'execution_complete.json'),'episodes_csv_sha256':sha256(OUT/'episodes.csv'),'source_sha256':sha256(Path(__file__)),'inputs':inputs,'helpers':helpers,'scope':'Every state/condition/method and all local roots: fresh history/onset/noise, prephysics records, selector actions, localnominal/corrected branches, fullglobal replay. Independent pose E, explicit third-difference jerk gain, applied norms and joint decisions; global CSV reconstruction. Statistical analysis remains separate.'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
 with threadpool_limits(limits=2,user_api='blas'):main(a.smoke)
