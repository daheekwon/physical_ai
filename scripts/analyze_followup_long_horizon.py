"""Paired endpoint/horizon evidence with family intervals and finite matched wrong actions."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scripts.analyze_followup_response_selection import mean_ci
from zj_wam.readouts import effects,errors
from zj_wam.io import resolve,write_json

out=resolve('artifacts/followup_v1');dest=out/'lewm_audit/long_horizon';frames=[];matching=[]
for h in [5,10,20]:
 paths=sorted(dest.glob(f'development*_h{h:02d}.json'))
 if len(paths)!=105:raise ValueError(f'Incomplete H{h}: {len(paths)}')
 for p in paths:
  f=pd.read_csv(p.with_suffix('.csv'));frames.append(f)
  source=pd.read_csv(out/'f1'/p.with_suffix('.csv').name)
  # Adjacent log-norm cells guarantee matched relative radius and <=5% action-norm ratio.
  source['norm_cell']=np.floor(np.log(np.maximum(source.actual_norm,1e-12))/np.log(1.05)).astype(int)
  index=[];wrong=[]
  for _,g in source.groupby(['relative_radius','norm_cell'],sort=True):
   ids=np.sort(g['index'].to_numpy(int))
   if len(ids)>1:index.extend(ids.tolist());wrong.extend(np.roll(ids,1).tolist())
  index=np.array(index,dtype=int);wrong=np.array(wrong,dtype=int)
  matching.append({'state_id':source.state_id.iloc[0],'horizon':h,'matched':len(index),'total':len(source)})
  with np.load(p.with_suffix('.npz')) as a:
   truth=effects(a['truth'])
   for rep in ['projected_cls','history_projected']:
    predicted=effects(a[rep+'_free_pose']);pe,re=errors(predicted[wrong],truth[index])
    rows=source.set_index('index').loc[index]
    for b in range(h//5):
     frames.append(pd.DataFrame({'state_id':rows.state_id.to_numpy(),'initial_pose_family':rows.initial_pose_family.to_numpy(),'horizon':h,'endpoint_step':5*(b+1),
       'index':index,'method':'wrong_action_norm_matched','representation':rep,'candidate_method':rows.method.to_numpy(),'relative_radius':rows.relative_radius.to_numpy(),
       'nominal_contact':rows.nominal_contact.to_numpy(),'preserved':rows.preserved.to_numpy(),'position_effect_error':pe[:,b],'rotation_effect_error':re[:,b],
       'position_effect':np.linalg.norm(truth[index,b,:2],axis=-1),'rotation_effect':np.abs(truth[index,b,2])}))
f=pd.concat(frames,ignore_index=True);f['rotation_effect_error_deg']=np.rad2deg(f.rotation_effect_error)
columns=['position_effect_error','rotation_effect_error_deg','position_absolute_error','rotation_absolute_error','nominal_position_error','nominal_rotation_error']
groups=['state_id','initial_pose_family','horizon','endpoint_step','method','representation','nominal_contact']
state=f.groupby(groups)[columns].mean().reset_index();state.to_csv(dest/'state_metrics.csv',index=False)
summaries=[]
for (h,endpoint,method,rep),g in state.groupby(['horizon','endpoint_step','method','representation']):
 for stratum in ['all','contact','free']:
  sel=g if stratum=='all' else g[g.nominal_contact==(stratum=='contact')]
  summaries.append({'horizon':int(h),'endpoint_step':int(endpoint),'method':method,'representation':rep,'stratum':stratum,
                    **{c:mean_ci(sel.dropna(subset=[c]),c) for c in columns}})
paired=[]
for (h,endpoint,rep),g in state.groupby(['horizon','endpoint_step','representation']):
 baseline=g[g.method=='free']
 for method in ['refresh_5','refresh_10','real_encoded','repeat_initial_history','zero_effect']:
  m=g[g.method==method].merge(baseline,on=['state_id','initial_pose_family'],suffixes=('','_free'))
  m['position_difference']=m.position_effect_error-m.position_effect_error_free
  m['rotation_difference_deg']=m.rotation_effect_error_deg-m.rotation_effect_error_deg_free
  paired.append({'horizon':int(h),'endpoint_step':int(endpoint),'representation':rep,'method':method,
                 'position_difference':mean_ci(m,'position_difference'),'rotation_difference_deg':mean_ci(m,'rotation_difference_deg')})
write_json(dest/'analysis.json',{'scope':'Development only; teacher forcing/context refresh has executed past observations. Not a deployed open-loop correction guarantee.',
                               'summary':summaries,'paired':paired,'wrong_action_matching':matching,'physical_samples_retained':True})
for r in summaries:
 if r['endpoint_step']==r['horizon'] and r['representation']=='projected_cls' and r['stratum']=='contact':print(r['horizon'],r['method'],round(r['position_effect_error']['mean'],3),round(r['rotation_effect_error_deg']['mean'],3))
