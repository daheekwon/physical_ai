"""Secondary 100-Hz oracle and rate sensitivity; primary 10-Hz endpoint unchanged."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve,write_json
from zj_wam.metrics import cluster_bootstrap
from scripts.analyze_additional_phase1 import nominal_row
out=resolve('artifacts/phase1_additional');cfg=json.loads((out/'manifest.json').read_text())['config']
f=pd.read_csv(out/'oracle_candidates.csv');selected=pd.read_csv(out/'oracle_selected.csv')
more=[]
for (sid,h),g in f[f.split=='test'].groupby(['state_id','horizon']):
    pool=g[g.preserved_path]
    if pool.empty or pool.physics_jerk.min()>=g.nominal_physics_jerk.iloc[0]:r=nominal_row(g)
    else:r=pool.loc[pool.physics_jerk.idxmin()].copy();r['fallback']=False
    r['selection_method']='oracle_path_100hz';more.append(r)
more=pd.DataFrame(more);more.to_csv(out/'oracle_physics_selected.csv',index=False)
all_selected=pd.concat([selected,more],ignore_index=True);rows=[];rng=np.random.default_rng(cfg['seed']+20)
for (h,method),g in all_selected.groupby(['horizon','selection_method']):
    valid10=g.nominal_executed_jerk>1e-8;valid100=g.nominal_physics_jerk>1e-8
    improve10=valid10&(g.jerk_reduction>=.1);improve100=valid100&(g.physics_jerk_reduction>=.1)
    both=g.preserved_path&improve10&improve100
    rows.append({'horizon':int(h),'method':method,'states':len(g),'joint_path_10hz':float((g.preserved_path&improve10).mean()),
                 'joint_path_100hz':float((g.preserved_path&improve100).mean()),'joint_both_rates':float(both.mean()),
                 'joint_100hz_ci':cluster_bootstrap((g.preserved_path&improve100).astype(float),g.episode,rng,2000),
                 'mean_10hz_reduction':float(g.jerk_reduction.mean()),'mean_100hz_reduction':float(g.physics_jerk_reduction.mean())})
write_json(out/'sampling_rate_statistics.json',{'scope':'Secondary sensitivity analysis, same candidate pool; does not alter primary 10-Hz endpoint or calibration','summary':rows})
print(json.dumps(rows,indent=2))
