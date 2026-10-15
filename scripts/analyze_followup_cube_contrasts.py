"""Paired Cube verifier/probe contrasts and finite-bank diagnostic limits."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json
from scripts.analyze_followup_response_selection import mean_ci


def main():
 out=resolve('artifacts/followup_v1/cube');cfg=json.loads(resolve('configs/followup_cube.json').read_text());f=pd.read_csv(out/'evaluation.csv');verifier=[]
 for split in ['development','calibration']:
  ref=f[(f.split==split)&(f.method=='lewm_global')]
  for other in ['controller_displacement','proprio_contact']:
   x=ref.merge(f[(f.split==split)&(f.method==other)],on='state_id',suffixes=('','_baseline'));x['joint_difference']=x.joint.astype(float)-x.joint_baseline.astype(float);x['violation_difference']=x.violation.astype(float)-x.violation_baseline.astype(float);x['jerk_difference']=x.jerk_gain-x.jerk_gain_baseline
   verifier.append({'split':split,'comparison':'lewm_global minus '+other,'all_methods_active':bool(x.active.all() and x.active_baseline.all()),**{k:mean_ci(x,k) for k in ['joint_difference','violation_difference','jerk_difference']}})
 effects=pd.read_csv(out/'probes_v2/effects.csv');probe=[]
 for (split,name),g in effects.groupby(['split','readout']):
  pred=g[g.control=='predicted']
  for baseline in ['zero','wrong_action','real_endpoint']:
   x=pred.merge(g[g.control==baseline],on=['state_id','candidate'],suffixes=('','_baseline'))
   for k in ['position_effect_error_m','rotation_effect_error_rad']:x[k+'_difference']=x[k]-x[k+'_baseline']
   probe.append({'split':split,'readout':name,'baseline':baseline,'position_error_difference':mean_ci(x,'position_effect_error_m_difference'),'rotation_error_difference':mean_ci(x,'rotation_effect_error_rad_difference')})
 dense=[];direction=[]
 for split in ['development','calibration']:
  frames=[pd.read_csv(p) for p in sorted((out/split).glob('e*.csv'))];bank=pd.concat(frames,ignore_index=True);active=bank[bank.norm>1e-6].copy();active['endpoint_pass']=(active.endpoint_position_m<=cfg['position_tolerance_m'])&(active.endpoint_rotation_rad<=cfg['rotation_tolerance_rad']);active['endpoint_pass_dense_fail']=active.endpoint_pass&(active.object_error>1);dense.append({'split':split,'active_candidates':len(active),'endpoint_pass_dense_fail_count':int(active.endpoint_pass_dense_fail.sum()),'endpoint_pass_dense_fail_fraction':mean_ci(active,'endpoint_pass_dense_fail')})
  # Exactly the same radius, sign and state; require all four prespecified directions feasible.
  records=[]
  for (state,radius,sign),g in bank[bank.method.isin(['lewm_low','lewm_high','random0','random1'])].groupby(['state_id','radius','sign']):
   if set(g.method)!=set(['lewm_low','lewm_high','random0','random1']):continue
   g=g.set_index('method');assert np.max(np.abs(g.norm-radius))<1e-6
   random=g.loc[['random0','random1']]
   records.append({'initial_pose_family':int(g.iloc[0].initial_pose_family),'state_id':state,'radius':radius,'sign':sign,'low_minus_random_E':float(g.loc['lewm_low','object_error']-random.object_error.mean()),'high_minus_random_E':float(g.loc['lewm_high','object_error']-random.object_error.mean()),'low_minus_random_jerk_gain':float(g.loc['lewm_low','jerk_gain']-random.jerk_gain.mean())})
  x=pd.DataFrame(records);direction.append({'split':split,'matched_cells':len(x),'states':x.state_id.nunique(),'families':x.initial_pose_family.nunique(),**{k:mean_ci(x,k) for k in ['low_minus_random_E','high_minus_random_E','low_minus_random_jerk_gain']},'scope':'Sameappliednormalizedactionnorm/state/radius/sign. Only completefeasiblecells; random mean of2directions. ExecutedEEFmotion andfrequency not matched; differences do not establish object-specific WM advantage.'})
 write_json(out/'contrasts.json',{'verifier':verifier,'probe':probe,'endpoint_vs_dense':dense,'matched_action_directions':direction,'oracle_generator_counts':f[(f.split=='development')&(f.method=='oracle')&f.joint].generator_method.value_counts().to_dict(),'scope':'Paired episode-family intervals. Verifier violationdifference is unconditional over allstates; equals appliedrisk difference only when bothalwaysactive. Nativeprobe contrasts conditional on fixedreadout; no newselection or finaltest.'});print('COMPLETE Cube paired contrasts',flush=True)
if __name__=='__main__':main()
