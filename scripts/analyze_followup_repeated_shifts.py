"""Family-cluster analysis of audited repeated correction under physical/sensing shifts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_response_selection import mean_ci
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
HELPERS=['scripts/analyze_followup_response_selection.py','scripts/audit_retrained_dinowm_readouts.py','scripts/audit_followup_repeated_shifts.py','configs/followup_repeated_shifts.json']
def main(smoke):
 out=BASE/('repeated_shifts_smoke' if smoke else 'repeated_shifts');audit_name='full_audit_smoke.json' if smoke else 'raw_audit.json';audit=read(out/audit_name);m=read(out/'manifest.json');assert audit['smoke']==smoke and audit['source_sha256']==sha256(resolve('scripts/audit_followup_repeated_shifts.py'));assert audit['episodes_csv_sha256']==sha256(out/'episodes.csv') and audit['manifest_sha256']==sha256(out/'manifest.json')
 for p,h in {**m['inputs'],**audit['inputs'],**audit['helpers']}.items():assert sha256(resolve(p))==h,p
 f=pd.read_csv(out/'episodes.csv');assert len(f)==audit['executions']==(21 if smoke else 5880);assert not f.duplicated(['state_id','condition','method']).any();assert f.initial_pose_family.nunique()==(1 if smoke else 35);assert f.state_id.nunique()==(1 if smoke else 105)
 f['violation']=~f.preserved;f['long_violation']=~f.long_preserved
 np.testing.assert_array_equal(f.joint,f.active&f.preserved&(f.jerk_gain>=.1));np.testing.assert_array_equal(f.long_joint,f.active&f.long_preserved&(f.long_jerk_gain>=.1));assert (f.long_object_error>=f.object_error-1e-12).all()
 local=[]
 for row in f.itertuples():
  p=out/row.condition/f'{row.state_id}__{row.method}.csv';g=pd.read_csv(p);assert len(g)==4 and g.offset.tolist()==[0,5,10,15];g['state_id']=row.state_id;g['initial_pose_family']=row.initial_pose_family;g['condition']=row.condition;g['method']=row.method;g['joint']=g.active&g.preserved&(g.jerk_gain>=.1);g['violation']=~g.preserved;local.append(g)
 local=pd.concat(local,ignore_index=True);assert len(local)==audit['fresh_selectors'];local.to_csv(out/'local_metrics.csv',index=False);errors={}
 def interval(g,key):
  r=mean_ci(g,key)
  if len(g):
   avg,ci=cluster(g[key].to_numpy(),g.initial_pose_family.to_numpy());np.testing.assert_allclose(avg,r['mean'],rtol=0,atol=1e-8);np.testing.assert_allclose(ci,r['ci95'],rtol=0,atol=1e-8);errors[key]=max(errors.get(key,0.),float(abs(avg-r['mean'])),float(np.max(abs(np.asarray(ci)-r['ci95']))))
  return r
 summary=[];paired=[]
 for (condition,method),g in f.groupby(['condition','method']):
  x=local[(local.condition==condition)&(local.method==method)];active=g[g.active]
  result={'condition':condition,'method':method,'states':len(g),'families':g.initial_pose_family.nunique(),**{k:interval(g,k) for k in ['active','joint','long_joint','jerk_gain','long_jerk_gain','object_error','long_object_error','modified_task_success','nominal_task_success','terminal_velocity_delta','terminal_angular_velocity_delta','contact_changed_substeps','refinement_seconds']},'selected_risk':interval(active,'violation'),'long_selected_risk':interval(active,'long_violation'),'local_joint':interval(x,'joint'),'local_active':interval(x,'active'),'local_selected_risk':interval(x[x.active],'violation'),'local_jerk_gain':interval(x,'jerk_gain'),'refinement_seconds_p50':float(g.refinement_seconds.median()),'refinement_seconds_p95':float(g.refinement_seconds.quantile(.95)),'suffix_action_counts':{str(int(k)):int(v) for k,v in g.suffix_actions.value_counts().items()}}
  summary.append(result)
  ctrl=f[(f.condition==condition)&(f.method=='controller_pusher')].set_index('state_id').loc[g.state_id];nom=f[(f.condition=='nominal')&(f.method==method)].set_index('state_id').loc[g.state_id]
  for label,base in [('method_vs_controller',ctrl),('condition_vs_nominal',nom)]:
   for metric in ['joint','long_joint','jerk_gain','long_jerk_gain','modified_task_success']:
    v=g[['initial_pose_family']].copy();v['difference']=g[metric].to_numpy(float)-base[metric].to_numpy(float);paired.append({'type':label,'condition':condition,'method':method,'metric':metric,**interval(v,'difference')})
 for (condition,sid),g in f.groupby(['condition','state_id']):assert g.nominal_task_success.nunique()==1
 write_json(out/'analysis.json',{'smoke':smoke,'states':f.state_id.nunique(),'families':f.initial_pose_family.nunique(),'executions':len(f),'local_decisions':len(local),'summary':summary,'paired':paired,'scope':m['scope'],'interpretation':'Development-only transfer of frozen nominal rules. Fixed expert actions; no policy regeneration. Cluster intervals retain all windows of an initialfamily. Local and initialroot cumulative preservation are separate. Longsuffix keeps core violations. No retuning or guarantee of populationrisk. Fullcommand controls have extra boundary freedom. Timings sum selector calls, exclude sensing/replay/evaluation and are sharedhost, not endtoend delay/deadline validation.'})
 write_json(out/'analysis_audit.json',{'smoke':smoke,'executions':len(f),'local_decisions':len(local),'summary_cells':len(summary),'paired_cells':len(paired),'max_independent_statistic_errors':errors,'raw_audit_sha256':sha256(out/audit_name),'analysis_sha256':sha256(out/'analysis.json'),'episodes_csv_sha256':sha256(out/'episodes.csv'),'local_metrics_sha256':sha256(out/'local_metrics.csv'),'source_sha256':sha256(Path(__file__)),'helpers':{p:sha256(resolve(p)) for p in HELPERS},'scope':'All audited records and local rows reconstructed, success/active/long flags checked, every mean/paired family interval independently recomputed; nominal success consistency across methods. Timing quantiles computed from saved selector times, no new isolated benchmark.'});print('REPEATED SHIFTS ANALYSIS AUDITED',len(f),len(local),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
