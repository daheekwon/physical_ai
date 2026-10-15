"""F1 family oracles and F9 fixed-verifier generator comparison, no test access."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scripts.analyze_followup_response_selection import mean_ci
from scripts.calibrate_followup_selection import selected_rows
from zj_wam.io import resolve,write_json
out=resolve('artifacts/followup_v1');dest=out/'family_ablation';summary=[];oracles=[];allframes=[]
for split,h in [('development',5),('development',10),('development',20),('calibration',5)]:
 paths=sorted(dest.glob(f'{split}*_h{h:02d}.json'))
 if len(paths)!=105:raise ValueError(f'{split}/H{h}: {len(paths)}/105')
 f=pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in paths],ignore_index=True);allframes.append(f)
 for (method,radius),g in f.groupby(['method','relative_radius']):
  g=g.copy();g['joint']=(g.actual_norm>1e-6)&g.preserved&(g.jerk_gain>=.1)
  for stratum in ['all','contact','free']:
   x=g if stratum=='all' else g[g.nominal_contact==(stratum=='contact')]
   summary.append({'split':split,'horizon':h,'method':method,'relative_radius':float(radius),'stratum':stratum,
                   'dimension':int(g.dimension.iloc[0]),**{c:mean_ci(x,c) for c in ['joint','preserved','jerk_gain','actual_norm','terminal_pusher_position_delta','terminal_pusher_velocity_delta']}})
 for (method,sid),g in f.groupby(['method','state_id']):
  valid=g[(g.actual_norm>1e-6)&g.preserved&(g.jerk_gain>0)];gain=float(valid.jerk_gain.max()) if len(valid) else 0.
  oracles.append({'split':split,'horizon':h,'method':method,'state_id':sid,'initial_pose_family':int(g.initial_pose_family.iloc[0]),'nominal_contact':bool(g.nominal_contact.iloc[0]),'gain':gain,'joint':gain>=.1})
o=pd.DataFrame(oracles);o.to_csv(dest/'oracle_state_metrics.csv',index=False);oracle_summary=[]
for (split,h,method),g in o.groupby(['split','horizon','method']):
 for stratum in ['all','contact','free']:
  x=g if stratum=='all' else g[g.nominal_contact==(stratum=='contact')]
  oracle_summary.append({'split':split,'horizon':int(h),'method':method,'stratum':stratum,'joint':mean_ci(x,'joint'),'gain':mean_ci(x,'gain')})
write_json(dest/'analysis.json',{'scope':'Four-radius family-specific oracle uses future outcomes; not a deployed method or global maximum. Random replicas separate.',
                                'by_radius':summary,'oracle':oracle_summary})
for r in oracle_summary:
 if r['split']=='development' and r['stratum']=='all':print(r['horizon'],r['method'],round(r['joint']['mean'],3))
# Fixed verifier isolates generation. It is not silently recalibrated to each family.
rules=json.loads((out/'selection/selector_rules.json').read_text())['rules'];chosen=[];stats=[]
for split in ['development','calibration']:
 paths=sorted((out/'response_scores/family_ablation').glob(f'{split}*h05.json'))
 if len(paths)!=105:continue
 f=pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in paths],ignore_index=True)
 for generator,g in f.groupby('method'):
  for verifier in ['controller_pusher','lewm_global','lewm_object','vision_contact','vision_adaptive']:
   selected=selected_rows(g.reset_index(drop=True),verifier,rules[verifier]);selected['generator']=generator;selected['split']=split
   selected['joint']=selected.active & selected.preserved & (selected.jerk_gain>=.1);chosen.append(selected)
   for stratum in ['all','contact','free']:
    x=selected if stratum=='all' else selected[selected.nominal_contact==(stratum=='contact')]
    a=x[x.active].copy();a['violation']=~a.preserved
    stats.append({'split':split,'generator':generator,'verifier':verifier,'stratum':stratum,'joint':mean_ci(x,'joint'),'active':mean_ci(x,'active'),'risk':mean_ci(a,'violation'),'mean_gain':mean_ci(x,'jerk_gain')})
if chosen:
 pd.concat(chosen,ignore_index=True).to_csv(dest/'fixed_verifier_choices.csv',index=False)
 write_json(dest/'fixed_verifier_analysis.json',{'scope':'Main F1 selector rule reused unchanged. Generator-distribution risk measured separately, not guaranteed by original calibration. Four candidates each; full family has higher dimension; compare low2 vs controller2 vs each random2 directly.','summary':stats})
