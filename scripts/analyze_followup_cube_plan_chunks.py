"""Independent causal-input, compiler and physical-trace audit of Cube chunks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,mujoco
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_experiment import fit_response
from scripts.run_followup_cube_native_policy import scene
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def wrap(x):return np.arctan2(np.sin(x),np.cos(x))
def yaw(q):return np.arctan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2]**2+q[...,3]**2))

def independent_compile(targets,root,vel,g,gains,gg,bounds,lo,hi,mode):
    p=root.copy();v=vel.copy();actions=[];positions=[];opening=[]
    for target in targets:
        a=np.clip(2*(np.r_[target[:4]-p,target[4]-g]-lo)/(hi-lo)-1,-1,1).astype(np.float32);actions.append(a)
        d=(a.astype(float)+1)*(hi-lo)*.5+lo;goal=p+d[:4];goal[:3]=np.clip(goal[:3],bounds[0],bounds[1]);goal[3]=wrap(goal[3]);gt=float(np.clip(g+d[4],0,1))
        if mode=='past_pd_h5':
            # Reproduce the declared Euler surrogate without calling the compiler.
            targetp=p+a[:4].astype(float)*np.array([.05,.05,.05,.3]);targetp[:3]=np.clip(targetp[:3],bounds[0],bounds[1]);previous=None
            for _ in range(25):
                error=targetp-p;error[3]=wrap(error[3]);v=v+.002*(gains[:,0]*error-gains[:,1]*v);previous=p.copy();p=p+.002*v
            v=(p-previous)/.002;p[3]=wrap(p[3]);g+=gg*(gt-g)
        else:p=goal;g=gt
        positions.append(p.copy());opening.append(g)
    return np.array(actions),np.array(positions),np.array(opening)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('cube_plan_chunks_smoke' if args.smoke else 'cube_plan_chunks');m=read(out/'manifest.json');cfg=m['config'];native=m['native'];done=read(out/'execution_complete.json');assert done['smoke']==args.smoke and done['manifest_sha256']==sha256(out/'manifest.json')
    for name,h in m['inputs'].items():assert sha256(resolve(name))==h,name
    env,_,initial_info,_,_=scene(cfg['development_seeds'][0],native);model=env._model;grip_id=env._gripper_opening_joint_id;site=env._pinch_site_id;scratch=mujoco.MjData(model);errors={};rows=[];blockrows=[];initials={};totalblocks=0;exact=0
    def check(name,a,b,tol=1e-10):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape);e=float(np.max(abs(a.astype(float)-b.astype(float)))) if a.size else 0.;errors[name]=max(errors.get(name,0.),e);assert e<=tol,(name,e,tol)
    seeds=cfg['development_seeds'][:1] if args.smoke else cfg['development_seeds']
    for seed in seeds:
      for mode in cfg['modes']:
        path=out/f'{seed}_{mode}.json';meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(path.with_suffix('.npz'))
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:a={k:z[k] for k in z.files}
        n=native['control_steps'];dt=native['physics_dt'];sub=round(native['control_dt']/dt);assert a['actions'].shape==(n,5) and a['actions'].dtype==np.float32 and abs(a['actions']).max()<=1
        for k,v in a.items():
            if k.endswith('_residual') and k.startswith('block_000_'):assert np.isnan(v).all()
            else:assert np.isfinite(v).all(),k
            if k.startswith('control_'):check('dense_'+k,a['physics_'+k[8:]][sub-1::sub],v,0.)
        check('clock',a['physics_time'],float(a['initial_time'])+np.arange(1,n*sub+1)*dt,1e-9)
        initial=np.r_[a['initial_qpos'],a['initial_qvel'],a['goal_position'],a['goal_quaternion']]
        if seed in initials:check('paired_initial',initial,initials[seed],0.)
        else:initials[seed]=initial
        with np.load(BASE/'cube_native_policy'/f'{seed}_plan.npz') as original:
            if mode=='native_h1':
                for k in original.files:
                    if k in a:check('native_reproduction_'+k,a[k],original[k],0.)
                exact+=1
            refpose=original['physics_object_pose'].copy()
        # Reconstruct controller's cached site from pre-final-integration qpos;
        # fresh dense EEF records are deliberately a separate measurement.
        roots=[];grippers=np.clip(np.r_[a['initial_qpos'][grip_id],a['control_qpos'][:,grip_id]]/.8,0,1)
        for step in range(n):
            if step==0:
                e,_,inf,_,_=scene(seed,native);root=np.r_[inf['proprio/effector_pos'],inf['proprio/effector_yaw'][0]];e.close()
            else:
                scratch.qpos[:]=a['physics_qpos'][step*sub-2];mujoco.mj_kinematics(model,scratch);mat=scratch.site_xmat[site].reshape(3,3);root=np.r_[scratch.site_xpos[site],np.arctan2(mat[1,0],mat[0,0])]
            roots.append(root)
        roots=np.array(roots);unwrapped=np.unwrap(np.r_[roots[0,3],yaw(a['physics_eef_quaternion'])]);points=np.column_stack([np.r_[a['initial_eef_position'][None],a['physics_eef_position']],unwrapped]);deltas=.5*(a['actions']+1)*(a['action_high']-a['action_low'])+a['action_low'];targets=roots+deltas[:,:4];targets[:,:3]=np.clip(targets[:,:3],a['bounds'][0],a['bounds'][1]);targets[:,3]=roots[:,3]+wrap(targets[:,3]-roots[:,3]);targets[:,3]+=2*np.pi*np.round((unwrapped[1::sub]-targets[:,3])/(2*np.pi));pt=np.repeat(targets,sub,axis=0);gt=np.clip(grippers[:-1]+deltas[:,4],0,1)
        step=0
        for bi in range(meta['blocks']):
            get=lambda k:a[f'block_{bi:03d}_{k}'];h=1 if mode=='native_h1' or step<cfg['warmup_controls'] else cfg['horizon'];h=min(h,n-step);assert int(get('step'))==step and int(get('horizon'))==h and int(get('last_observation_control'))==step
            check('causal_root',get('root'),roots[step],1e-12);check('root_time',get('root_time'),float(a['initial_time'])+step*.05,1e-9);check('root_gripper',get('gripper'),grippers[step],0.)
            count=min(step*sub,cfg['fit_physics_history']);check('past_points',get('past_points'),points[step*sub-count:step*sub+1],1e-11);check('past_targets',get('past_targets'),pt[step*sub-count:step*sub],1e-11);check('gripper_history',get('gripper_history'),grippers[:step+1],0.);check('gripper_targets',get('gripper_targets'),gt[:step],1e-12)
            v=(points[step*sub]-points[step*sub-1])/dt if step else np.zeros(4);check('past_velocity',get('velocity'),v,1e-9)
            if count>=2:
                gains,res=fit_response(get('past_points'),get('past_targets'),dt);check('refit_gains',get('gains'),gains,1e-8);check('refit_residual',get('residual'),res,1e-8)
            else:gains=np.tile(cfg['default_pd'],(4,1));check('default_gains',get('gains'),gains,0.)
            err=gt[:step]-grippers[:step];change=np.diff(grippers[:step+1]);den=err@err;gg=float(np.clip(err@change/den,0,1)) if den>1e-10 else cfg['default_gripper_gain'];check('past_gripper_fit',get('gripper_gain'),gg,1e-12)
            indices=np.minimum(np.floor((float(get('root_time'))+np.arange(h)*.05-float(a['plan_t_init'])+1e-7)/.05).astype(int),len(a['plan'])-1);check('plan_indices',get('plan_indices'),indices,0.);check('known_plan_targets',get('targets'),a['plan'][indices],0.)
            aa,pred,pg=independent_compile(get('targets'),get('root'),get('velocity'),float(get('gripper')),get('gains'),float(get('gripper_gain')),a['bounds'],a['action_low'],a['action_high'],mode);check('independent_actions',aa,get('actions'),0.);check('executed_chunk',get('actions'),a['actions'][step:step+h],0.);check('independent_prediction',pred,get('predicted'),1e-11);check('independent_gripper',pg,get('predicted_gripper'),1e-12)
            pe=np.linalg.norm(pred[:,:3]-a['control_eef_position'][step:step+h],axis=1);ye=abs(wrap(pred[:,3]-yaw(a['control_eef_quaternion'][step:step+h])));ge=abs(pg-grippers[step+1:step+h+1])
            for j in range(h):blockrows.append({'seed':seed,'mode':mode,'step':step,'horizon':h,'offset':j+1,'position_prediction_error_m':float(pe[j]),'yaw_prediction_error_rad':float(ye[j]),'gripper_prediction_error':float(ge[j])})
            step+=h;totalblocks+=1
        assert step==n
        dist=np.linalg.norm(a['control_object_pose'][:,:3]-a['goal_position'],axis=1);success=dist<=.04;check('success',success,a['success'],0.);check('final_distance',dist[-1],meta['final_position_error_m'],1e-12);released=bool(success[-1] and a['control_robot_object_contacts'][-1]==0);assert bool(success.any())==meta['success_any'] and bool(success[-1])==meta['success_final'] and released==meta['released_success_final']
        j=np.diff(a['physics_eef_position'],n=3,axis=0)/dt**3;oe=np.linalg.norm(a['physics_object_pose'][:,:3]-refpose[:,:3],axis=1);rot=Rotation.from_quat(a['physics_object_pose'][:,[4,5,6,3]]);refrot=Rotation.from_quat(refpose[:,[4,5,6,3]]);angle=(rot*refrot.inv()).magnitude()
        rows.append({'seed':seed,'mode':mode,'success_any':bool(success.any()),'success_final':bool(success[-1]),'released_success_final':released,'final_position_error_m':float(dist[-1]),'eef_jerk_energy':float(np.sum(j*j)*dt),'native_object_max_position_m':float(oe.max()),'native_object_max_rotation_rad':float(angle.max()),'native_trajectory_preserved':bool(oe.max()<=.002 and angle.max()<=np.deg2rad(2))})
        print('AUDITED',seed,mode,flush=True)
    env.close();assert len(rows)==done['episodes'] and len(initials)==len(seeds);assert len({v.tobytes() for v in initials.values()})==len(seeds)
    f=pd.DataFrame(rows);b=pd.DataFrame(blockrows);f.to_csv(out/'evaluation.csv',index=False);b.to_csv(out/'prediction_errors.csv',index=False);summary=[]
    for mode,g in f.groupby('mode'):
        summary.append({'mode':mode,'episodes':len(g),**{k:float(g[k].mean()) for k in f.columns if k not in ['seed','mode']}})
    byoffset=b.groupby(['mode','horizon','offset'])[['position_prediction_error_m','yaw_prediction_error_rad','gripper_prediction_error']].mean().reset_index().to_dict('records');write_json(out/'analysis.json',{'smoke':args.smoke,'summary':summary,'prediction_by_offset':byoffset,'scope':cfg['scope'],'notes':'Development nominal-compiler check only. Native success40mm differs from native-path preservation2mm/full2deg. Jerk differences across changed nominal paths are not smoothing gains. Prediction errors compare to fresh dense kinematics, not cached controller observation.'});write_json(out/'raw_audit.json',{'smoke':args.smoke,'episodes':len(rows),'families':len(seeds),'blocks':totalblocks,'physics_samples':len(rows)*n*sub,'native_exact_episodes':exact,'max_error':errors,'source_sha256':sha256(Path(__file__)),'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'scope':'Independent compiler equations, roots from pre-integration qpos, gripper from qpos, all past histories/targets, refit, action assembly, dense/control/time/success and paired initial states.'});print('COMPLETE',len(rows),totalblocks,flush=True)
if __name__=='__main__':main()
