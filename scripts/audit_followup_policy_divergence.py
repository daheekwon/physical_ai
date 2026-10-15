"""Verify common policy execution before intervention and locate cumulative divergence."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json
from scripts.analyze_followup_response_selection import mean_ci


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration','test'],required=True);args=p.parse_args();base=resolve('artifacts/followup_v1');out=base/'policy_refinement'/args.split;a=json.loads((out/'analysis.json').read_text());assert not a['incomplete_seeds'];cfg=json.loads(resolve('configs/followup_v1.json').read_text());pcfg=json.loads(resolve('configs/followup_policy_refinement.json').read_text());rows=[];max_prefix=0.;max_actions=0.
    for seed in a['completed_paired_seeds']:
        with np.load(out/f'{seed}_nominal.npz') as ref:
            nominal=ref['physics'];nominal_actions=ref['actions'];coverage=ref['coverage']
            for method in pcfg['methods']:
                if method=='nominal':continue
                m=json.loads((out/f'{seed}_{method}.json').read_text());active=[d for d in m['decisions'] if d['active']];assert active;first=active[0]['control_step']
                with np.load(out/f'{seed}_{method}.npz') as run:
                    trace=run['physics'];pe=float(np.max(np.abs(trace[:10*first]-nominal[:10*first])));ae=float(np.max(np.abs(run['actions'][:first]-nominal_actions[:first])));max_prefix=max(max_prefix,pe);max_actions=max(max_actions,ae);assert pe<1e-9 and ae<1e-9,(seed,method,first,pe,ae)
                    d=trace[:,2:5]-nominal[:,2:5];pos=np.linalg.norm(d[:,:2],axis=1)/cfg['tolerances']['position'];angle=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2])))/cfg['tolerances']['rotation'];where=np.flatnonzero(np.maximum(pos,angle)>1);where_cov=np.flatnonzero(np.maximum(0,coverage-run['coverage'])>cfg['tolerances']['coverage_loss']);times=[]
                    if len(where):times.append((where[0]+1)*.01)
                    if len(where_cov):times.append((where_cov[0]+1)*.1)
                    crossing=min(times) if times else None
                    if crossing is not None:assert crossing>first*.1
                    local=pd.read_csv(out/f'{seed}_{method}.csv')
                    rows.append({'seed':seed,'initial_pose_family':seed,'method':method,'first_correction_seconds':first*.1,'first_global_violation_seconds':crossing,'seconds_after_first_correction':crossing-first*.1 if crossing is not None else None,'all_local_preserved':bool((local.object_error<=1).all()),'global_violated':crossing is not None})
    f=pd.DataFrame(rows);summary=[]
    for method,g in f.groupby('method'):
        v=g.dropna(subset=['seconds_after_first_correction']);summary.append({'method':method,'episodes':len(g),'global_violated':mean_ci(g,'global_violated'),'all_local_preserved':mean_ci(g,'all_local_preserved'),'mean_seconds_to_first_violation':mean_ci(v,'seconds_after_first_correction'),'median_seconds_to_first_violation':float(v.seconds_after_first_correction.median()) if len(v) else None})
    f.to_csv(out/'divergence.csv',index=False);write_json(out/'divergence_audit.json',{'split':args.split,'rows':len(f),'max_physics_difference_before_first_correction':max_prefix,'max_action_difference_before_first_correction':max_actions,'summary':summary,'scope':'Each method follows exactly same nominal-policy action/physics prefix until its first active correction. First cumulative tolerance crossing from100Hz pose and10Hz coverage checks, with each method own subsequent observations. Time is measured from correction onset; not a causal decomposition of direct effect versus subsequent policy feedback.'});print('PASS',len(f),max_prefix,max_actions,flush=True)
    for r in summary:print(r['method'],'median delay to globalviolation',r['median_seconds_to_first_violation'],flush=True)
if __name__=='__main__':main()
