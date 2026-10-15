"""Local first-chunk preservation versus residual drift under fixed future commands."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci


def main():
    base=resolve('artifacts/followup_v1');out=base/'policy_effect_decomposition';source=base/'policy_refinement/development';m=json.loads((out/'manifest.json').read_text());cfg=m['config'];tol=json.loads(resolve('configs/followup_v1.json').read_text())['tolerances'];audit=json.loads((out/'partial_raw_audit.json').read_text());assert audit['arm_rows']==60
    old=pd.read_csv(out/'partial_evaluation.csv');rows=[];checks=0
    for seed in cfg['development_seeds']:
        with np.load(source/f'{seed}_nominal.npz') as a:ref=a['physics'];refcov=a['coverage'];refactions=a['actions']
        for method in cfg['methods']:
            path=out/f'{seed}_{method}_b_first_fixed.json';meta=json.loads(path.read_text());first=meta['first_correction_step'];end=meta['first_chunk_end']
            with np.load(path.with_suffix('.npz')) as a:
                trace=a['physics'];d=trace[:,2:5]-ref[:,2:5];pos=np.linalg.norm(d[:,:2],axis=1);rot=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2])));loss=np.maximum(0,refcov-a['coverage'])
                local=max(pos[first*10:end*10].max()/tol['position'],rot[first*10:end*10].max()/tol['rotation'],loss[[first+4,end-1]].max()/tol['coverage_loss'])
                global_error=max(pos.max()/tol['position'],rot.max()/tol['rotation'],loss.max()/tol['coverage_loss'])
                row=old[(old.seed==seed)&(old.method==method)&(old.arm=='b_first_fixed')].iloc[0];assert abs(global_error-row.object_error)<1e-8
                prior=pd.read_csv(source/f'{seed}_{method}.csv');original_local=prior[prior.control_step==first].iloc[0];assert abs(local-original_local.object_error)<1e-8
                np.testing.assert_array_equal(a['actions'][end:],refactions[end:]);checks+=1
                pusher=np.linalg.norm(trace[:,:2]-ref[:,:2],axis=1);velocity=np.linalg.norm(trace[:,5:7]-ref[:,5:7],axis=1)
                rows.append({'seed':seed,'initial_pose_family':seed,'method':method,'first_chunk_object_error':local,'first_chunk_preserved':local<=1,'whole_episode_object_error':global_error,'whole_episode_preserved':global_error<=1,'first_pass_episode_fail':bool(local<=1 and global_error>1),'pusher_offset_at_firstchunk_end_px':float(pusher[end*10-1]),'pusher_offset_at_episode_end_px':float(pusher[-1]),'pusher_offset_max_px':float(pusher.max()),'pusher_velocity_difference_at_episode_end_px_s':float(velocity[-1]),'seconds_after_first_correction':row.seconds_after_first_correction})
    f=pd.DataFrame(rows);summary=[]
    for method,g in f.groupby('method'):
        summary.append({'method':method,'families':len(g),**{k:mean_ci(g,k) for k in ['first_chunk_preserved','whole_episode_preserved','first_pass_episode_fail','pusher_offset_at_firstchunk_end_px','pusher_offset_at_episode_end_px','pusher_offset_max_px','pusher_velocity_difference_at_episode_end_px_s']}})
    f.to_csv(out/'residual_diagnostic.csv',index=False);write_json(out/'residual_diagnostic.json',{'summary':summary,'rows':len(f),'parent_first_chunk_exact_checks':checks,'parent_audit_sha256':sha256(out/'partial_raw_audit.json'),'manifest_sha256':sha256(out/'manifest.json'),'source_sha256':sha256(Path(__file__)),'scope':'Four developmentseeds all5methods. Bfixednominalrelativefuturecommands. Different realizedpusherpath remains part of residual effect; not isolated passiveobjectinertia. Local H5+suffix3 same tolerances and exact original firstcorrection local E. Long episode also checkscoverageatallcontroltimes. No Cpolicyfeedback result included.'})
    for r in summary:print(r['method'],'localpass',r['first_chunk_preserved']['mean'],'globalpass',r['whole_episode_preserved']['mean'],'localpass_globalfail',r['first_pass_episode_fail']['mean'],'pusherendpx',r['pusher_offset_at_episode_end_px']['mean'],flush=True)
if __name__=='__main__':main()
