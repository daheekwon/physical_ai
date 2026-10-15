"""Audit nominal Cube policy traces and summarize fixed-target performance."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();out=BASE/('cube_native_policy_smoke' if args.smoke else 'cube_native_policy');m=read(out/'manifest.json');done=read(out/'execution_complete.json');cfg=m['config'];assert done['smoke']==args.smoke and done['manifest_sha256']==sha256(out/'manifest.json')
    for name,h in m['inputs'].items():assert sha256(resolve(name))==h,name
    rows=[];errors={};initials={};validated=0;total_samples=0
    def check(name,x,y,tol=1e-12):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape;err=float(np.max(np.abs(x.astype(float)-y.astype(float))));errors[name]=max(errors.get(name,0.),err);assert err<=tol,(name,err)
    for path in sorted(out.glob('34*.json')):
        meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(path.with_suffix('.npz'))
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
            n=cfg['control_steps'];substeps=round(cfg['control_dt']/cfg['physics_dt']);assert a['actions'].shape==(n,5) and a['actions'].dtype==np.float32 and np.max(abs(a['actions']))<=1
            assert len(a['physics_time'])==n*substeps;total_samples+=len(a['physics_time'])
            for name in a.files:
                assert np.isfinite(a[name]).all(),name
                if name.startswith('control_'):
                    physics='physics_'+name[len('control_'):];check('dense_to_control_'+name,a[physics][substeps-1::substeps],a[name],0.)
            check('physical_clock',a['physics_time'],float(a['initial_time'])+np.arange(1,n*substeps+1)*cfg['physics_dt'],1e-9)
            initial=np.r_[a['initial_qpos'],a['initial_qvel'],a['goal_position'],a['goal_quaternion']]
            if meta['seed'] in initials:check('paired_policy_initial',initial,initials[meta['seed']],0.)
            else:initials[meta['seed']]=initial
            assert np.linalg.norm(a['initial_object_pose'][:2]-a['goal_position'][:2])>=cfg['minimum_start_goal_separation_m']-1e-12
            distance=np.sqrt(np.sum((a['control_object_pose'][:,:3]-a['goal_position'])**2,axis=1));success=distance<=.04;check('success_from_raw_position',success,a['success'],0.);check('final_position',distance[-1],meta['final_position_error_m'],1e-12)
            released=bool(success[-1] and a['control_robot_object_contacts'][-1]==0);assert meta['success_any']==bool(success.any()) and meta['success_final']==bool(success[-1]) and meta['final_released_success']==released
            if meta['seed'] in cfg['validation_seeds']:
                assert len(meta['plain_reproduction_max_error'])>=20 and not any(meta['plain_reproduction_max_error'].values());validated+=1
            dt=cfg['physics_dt'];xyz=a['physics_eef_position'];j=(xyz[3:]-3*xyz[2:-1]+3*xyz[1:-2]-xyz[:-3])/dt**3;energy=float(np.sum(j*j)*dt);joint=np.diff(a['physics_qpos'][:,:6],n=3,axis=0)/dt**3
            rot=Rotation.from_quat(a['physics_eef_quaternion'][:,[1,2,3,0]]);omega=(rot[1:]*rot[:-1].inv()).as_rotvec()/dt;angular=np.diff(omega,n=2,axis=0)/dt**2
            object_r=Rotation.from_quat(a['control_object_pose'][-1,[4,5,6,3]]);goal_r=Rotation.from_quat(a['goal_quaternion'][[1,2,3,0]]);goal_error=(object_r*goal_r.inv()).magnitude();goal_sym=min((object_r*(goal_r*s).inv()).magnitude() for s in Rotation.create_group('O'))
            rows.append({'initial_pose_family':meta['seed'],'policy':meta['policy'],'success_any':bool(success.any()),'success_final':bool(success[-1]),'released_success_final':released,'first_success_seconds':float((np.flatnonzero(success)[0]+1)*cfg['control_dt']) if success.any() else np.nan,'final_position_error_m':float(distance[-1]),'final_goal_rotation_rad':float(goal_error),'final_goal_rotation_symmetry_rad':float(goal_sym),'eef_jerk_energy':energy,'eef_peak_jerk':float(np.linalg.norm(j,axis=1).max()),'joint_jerk_energy':float(np.sum(joint*joint)*dt),'eef_angular_jerk_energy':float(np.sum(angular*angular)*dt),'max_object_height_m':float(a['physics_object_pose'][:,2].max()),'robot_contact_fraction':float((a['physics_robot_object_contacts']>0).mean()),'policy_compute_p95_seconds':float(np.quantile(a['policy_seconds'],.95))})
    assert len(rows)==(2 if args.smoke else 24) and validated==(2 if args.smoke else 4)
    assert len({v.tobytes() for v in initials.values()})==len(initials),'Duplicate initial states'
    frame=pd.DataFrame(rows);summary=[]
    for policy,g in frame.groupby('policy'):
        summary.append({'policy':policy,'families':len(g),'episodes':len(g),**{k:mean_ci(g.dropna(subset=[k]),k) for k in frame.columns if k not in ['initial_pose_family','policy']}})
    frame.to_csv(out/'evaluation.csv',index=False);write_json(out/'analysis.json',{'smoke':args.smoke,'summary':summary,'scope':cfg['scope'],'metric_notes':'Native success tests position within40mm only; report final rotation separately. Released success additionally requires no robot-object contacts at final control. Nominal jerk measured at500Hz over same15s with no prehistory, not a refinement comparison. Bootstrap over independently sampled development scenes, not finaltest.'});write_json(out/'raw_audit.json',{'smoke':args.smoke,'episodes':len(rows),'families':len(initials),'physics_samples':total_samples,'plain_reproduced_episodes':validated,'max_error':errors,'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'source_sha256':sha256(Path(__file__)),'scope':'All initial state uniqueness/policy pairing, dense/control states, times, action bounds, independent task success, measured nominal jerk. Fresh no-instrumentation replay performed by runner for firsttwo scenes and both native policies; remaining episodes raw-audited.'});print('AUDITED native Cube',len(rows),total_samples,validated,flush=True)
if __name__=='__main__':main()
