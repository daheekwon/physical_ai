"""Development F1 summaries; never treat these as held-out final performance."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve,write_json


def interval(frame,column,seed=20261011):
    # One row per state, correlated states sampled together within each family.
    groups=[g[column].to_numpy(dtype=float) for _,g in frame.groupby('initial_pose_family')]
    rng=np.random.default_rng(seed);est=[]
    for _ in range(2000):
        sample=[groups[i] for i in rng.integers(0,len(groups),len(groups))]
        est.append(np.concatenate(sample).mean())
    return np.quantile(est,[.025,.975]).tolist()


def main():
    out=resolve('artifacts/followup_v1');directory=out/'f1'
    done=[json.loads(p.read_text()) for p in directory.glob('*_h[0-9][0-9].json')]
    frames=[]
    for d in done:
        f=pd.read_csv(directory/f"{d['state_id']}_h{d['horizon']:02d}.csv")
        f['active']=f.actual_norm>1e-6
        f['joint']=f.active & (f.object_error<=1) & (f.jerk_gain>=.1) & (f.nominal_physics_jerk>1e-8)
        frames.append(f)
    if not frames:return
    candidates=pd.concat(frames,ignore_index=True)
    summaries=[];oracle=[]
    for (split,h),f in candidates.groupby(['split','horizon']):
        for (method,r),g in f[f.method.str.startswith('optimizer')].groupby(['method','relative_radius']):
            for phase,k in [('all',g),('contact',g[g.nominal_contact]),('no_contact',g[~g.nominal_contact])]:
                if not len(k):continue
                summaries.append({'split':split,'horizon':int(h),'method':method,'radius':float(r),'phase':phase,
                    'states':len(k),'families':k.initial_pose_family.nunique(),
                    'preservation_rate':float(k.preserved.mean()),'joint_rate':float(k.joint.mean()),
                    'joint_family_ci':interval(k,'joint'),'mean_jerk_gain':float(k.jerk_gain.mean()),
                    'solver_success_rate':float(k.solver_success.mean())})
        for sid,g in f.groupby('state_id'):
            ok=g[(g.object_error<=1)&g.active]
            selected=ok.sort_values('physics_jerk').iloc[0] if len(ok) else None
            oracle.append({'split':split,'horizon':int(h),'state_id':sid,'initial_pose_family':int(g.initial_pose_family.iloc[0]),
                           'nominal_contact':bool(g.nominal_contact.iloc[0]),
                           'joint':bool(selected is not None and selected.joint),'gain':max(0.,float(selected.jerk_gain)) if selected is not None else 0.})
    pd.DataFrame(summaries).to_csv(out/'f1/optimizer_summary.csv',index=False)
    o=pd.DataFrame(oracle);o.to_csv(out/'f1/oracle_by_state.csv',index=False)
    os=[]
    for (split,h),g in o.groupby(['split','horizon']):
        for phase,k in [('all',g),('contact',g[g.nominal_contact]),('no_contact',g[~g.nominal_contact])]:
            if len(k):os.append({'split':split,'horizon':int(h),'phase':phase,'states':len(k),'families':k.initial_pose_family.nunique(),
                                'joint':int(k.joint.sum()),'joint_rate':float(k.joint.mean()),'joint_family_ci':interval(k,'joint'),
                                'mean_gain':float(k.gain.mean())})
    result={'scope':'Development only; 100Hz object trajectory with 0.3s suffix and dense executed jerk; not final test',
            'candidate_count':len(candidates),'oracle':os,'optimizers':summaries,
            'max_controller_position_error':float(candidates.controller_position_error.max()),
            'max_polynomial_residual':float(candidates.polynomial_residual.max())}
    write_json(out/'f1/summary.json',result)
    print(json.dumps({'candidate_count':len(candidates),'oracle':os},indent=2))

if __name__=='__main__':main()
