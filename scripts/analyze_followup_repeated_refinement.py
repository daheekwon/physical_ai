"""Repeated refinement: core preservation, nested long observation and family uncertainty."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from scripts.analyze_followup_response_selection import mean_ci
from zj_wam.io import resolve,write_json
out=resolve('artifacts/followup_v1');directory=out/'repeated_refinement';frames=[]
for split in ['development','calibration']:
 paths=sorted(directory.glob(f'{split}*.json'))
 if len(paths)!=105:raise ValueError(f'{split}: {len(paths)}/105 states')
 frames.extend(pd.read_csv(p.with_suffix('.csv')) for p in paths)
f=pd.concat(frames,ignore_index=True)
# Retain the already declared coverage check at the core suffix endpoint.
# Extending observation must not remove an earlier required check.
f['nested_long_error']=np.maximum(f.object_error,f.long_object_error)
f['active']=f.actual_norm>1e-6
f['joint']=f.active & (f.object_error<=1) & (f.jerk_gain>=.1)
f['long_joint']=f.active & (f.nested_long_error<=1) & (f.long_jerk_gain>=.1)
f['core_joint_later_violated']=f.joint & (f.nested_long_error>1)
f['extra_error']=f.nested_long_error-f.object_error
f.to_csv(directory/'evaluation.csv',index=False)
summary=[]
for (split,method),g in f.groupby(['split','method']):
 for stratum in ['all','contact','transition','full_long_suffix']:
  x=g if stratum=='all' else g[g.nominal_contact] if stratum=='contact' else g[g.nominal_contact_transition] if stratum=='transition' else g[g.suffix_actions==10]
  a=x[x.active].copy();a['violation']=a.object_error>1;a['long_violation']=a.nested_long_error>1
  summary.append({'split':split,'method':method,'stratum':stratum,'states':len(x),
                  **{c:mean_ci(x,c) for c in ['joint','long_joint','active','core_joint_later_violated','jerk_gain','long_jerk_gain','extra_error','refinement_seconds','modified_task_success']},
                  'risk':mean_ci(a,'violation'),'long_risk':mean_ci(a,'long_violation')})
paired=[]
for split in ['development','calibration']:
 for baseline in ['repeat5_predicted_pose','open20_piecewise_vision','repeat5_observed_pose']:
  base=f[(f.split==split)&(f.method==baseline)]
  for method in f.method.unique():
   m=f[(f.split==split)&(f.method==method)].merge(base,on='state_id',suffixes=('','_base')).copy()
   for metric in ['joint','long_joint','jerk_gain','nested_long_error']:
    m[metric+'_diff']=m[metric].astype(float)-m[metric+'_base'].astype(float)
   paired.append({'split':split,'method':method,'baseline':baseline,**{c:mean_ci(m,c+'_diff') for c in ['joint','long_joint','jerk_gain','nested_long_error']}})
write_json(directory/'analysis.json',{'scope':'Fixed expert action source; feedback only changes corrections. H5 rules reused without tuning. Development and calibration diagnostics, not final test or Diffusion Policy evaluation.',
 'long_definition':'E_long=max(E_core,E_at_full_suffix), retaining coverage at both core and long suffix endpoints; G_long uses its own identical paired full time window.',
 'raw_long_score_requiring_core_retention':int((f.object_error>f.long_object_error).sum()),
 'suffix_counts':{str(k):int(v) for k,v in f[f.method==f.method.iloc[0]].suffix_actions.value_counts().items()},'summary':summary,'paired':paired})
for r in summary:
 if r['split']=='development' and r['stratum']=='all':print(r['method'],'U',round(r['joint']['mean'],3),'R',r['risk']['mean'],'longU',round(r['long_joint']['mean'],3),'longR',r['long_risk']['mean'])
