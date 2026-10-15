"""Matched-rank Jacobian refresh; local and accumulated object effects, family intervals."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scripts.analyze_followup_response_selection import mean_ci
from zj_wam.io import resolve,write_json

def main():
 directory=resolve('artifacts/followup_v1/relinearization');frames=[];local=[]
 for split in ['development','calibration']:
  paths=sorted(directory.glob(f'{split}*.json'))
  if len(paths)!=105:raise ValueError(f'{split} {len(paths)}/105')
  frames.extend(pd.read_csv(p.with_suffix('.csv')) for p in paths)
  local.extend(pd.read_csv(p.parent/(p.stem+'_local.csv')) for p in paths)
 f=pd.concat(frames,ignore_index=True);l=pd.concat(local,ignore_index=True)
 assert not f.duplicated(['state_id','method']).any() and len(l)==len(f)*4
 f['active']=f.actual_norm>1e-6;f['joint']=f.active & (f.object_error<=1)&(f.jerk_gain>=.1)
 f['long_joint']=f.active & (f.long_object_error<=1)&(f.long_jerk_gain>=.1)
 l['local_preserved']=l.local_object_error<=1;l['active']=l.norm>1e-6
 resolution=json.loads(resolve('artifacts/followup_v1/lewm_audit/effects/resolution.json').read_text())['thresholds']['projected_cls_ridge']
 l['position_resolved']=l.position_effect>resolution['position'];l['rotation_resolved']=l.rotation_effect_deg>np.rad2deg(resolution['rotation'])
 # Errors relative to zero-effect baseline on the SAME selected physical intervention.
 l['position_error_minus_zero']=l.position_effect_error-l.position_effect
 l['rotation_error_minus_zero']=l.rotation_effect_error_deg-l.rotation_effect_deg
 history=l.groupby(['state_id','method']).local_preserved.all().rename('all_local_preserved').reset_index()
 f=f.merge(history,on=['state_id','method']);f['local_pass_global_fail']=f.all_local_preserved & (f.object_error>1)
 summary=[]
 for (split,method),g in f.groupby(['split','method']):
  for stratum in ['all','contact','transition','full_long_suffix']:
   x=g if stratum=='all' else g[g.nominal_contact] if stratum=='contact' else g[g.nominal_contact_transition] if stratum=='transition' else g[g.suffix_actions==10]
   active=x[x.active].copy();active['violation']=active.object_error>1;active['long_violation']=active.long_object_error>1
   summary.append({'split':split,'method':method,'stratum':stratum,'states':len(x),**{c:mean_ci(x,c) for c in ['joint','long_joint','active','all_local_preserved','local_pass_global_fail','jerk_gain','seconds','relinearizations']},'risk':mean_ci(active,'violation'),'long_risk':mean_ci(active,'long_violation')})
 local_summary=[]
 for (split,method),g in l.groupby(['split','method']):
  for stratum in ['all','contact','transition','active']:
   x=g if stratum=='all' else g[g.local_contact] if stratum=='contact' else g[g.local_contact_transition] if stratum=='transition' else g[g.active]
   local_summary.append({'split':split,'method':method,'stratum':stratum,'blocks':len(x),**{c:mean_ci(x,c) for c in ['local_preserved','position_effect_error','rotation_effect_error_deg','position_error_minus_zero','rotation_error_minus_zero','position_resolved','rotation_resolved']}})
 paired=[]
 for split in ['development','calibration']:
  base=f[(f.split==split)&(f.method=='low_frozen')]
  for method in ['low_refresh10','low_refresh5']:
   x=f[(f.split==split)&(f.method==method)].merge(base,on='state_id',suffixes=('','_base')).copy()
   for metric in ['joint','long_joint','jerk_gain','local_pass_global_fail','seconds']:
    x[metric+'_diff']=x[metric].astype(float)-x[metric+'_base'].astype(float)
   paired.append({'split':split,'method':method,'baseline':'low_frozen',**{m:mean_ci(x,m+'_diff') for m in ['joint','long_joint','jerk_gain','local_pass_global_fail','seconds']}})
 f.to_csv(directory/'evaluation.csv',index=False);l.to_csv(directory/'local_evaluation.csv',index=False)
 write_json(directory/'analysis.json',{'scope':'Development/calibration diagnostic; same rank2 and finite verifier each5 actions, Jacobian recalculation count1/2/4. H5 rule reused, no repeat-risk guarantee. Each method follows its own history; selected actions and hence local intervention effects differ across methods.','resolution':resolution,'summary':summary,'local_summary':local_summary,'paired':paired})
 for r in summary:
  if r['split']=='development' and r['stratum']=='all':print(r['method'],{k:r[k]['mean'] for k in ['joint','risk','long_joint','long_risk','all_local_preserved','local_pass_global_fail']})
if __name__=='__main__':main()
