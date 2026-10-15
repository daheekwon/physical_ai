"""Audit fresh Cube calibration policy, causal compiler and every physical sample."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube
from zj_wam.cube_experiment import fit_response
from scripts.run_followup_cube_native_policy import scene
from scripts.analyze_followup_cube_plan_chunks import independent_compile,yaw,wrap
from scripts.audit_followup_dinowm_policy_final import interval
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main(smoke):
    out=BASE/('cube_policy_final_nominal_smoke' if smoke else 'cube_policy_final_nominal');manifest=read(out/'manifest.json');cfg=manifest['compiler'];native=manifest['native'];done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['reserved_test_policy_outcomes_executed']==(not smoke) and done['manifest_sha256']==sha256(out/'manifest.json');errors={};inputs={};rows=[];blocks=0;physics_samples=0
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    def check(name,a,b,tol=1e-10):
        a=np.asarray(a);b=np.asarray(b);np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True,err_msg=name);d=abs(a.astype(float)-b.astype(float));errors[name]=max(errors.get(name,0.),float(np.nanmax(d)) if d.size and np.isfinite(d).any() else 0.)
    if not smoke:
        reg=read(BASE/'cube_policy_calibration_nominal/initial_registry.json');assert reg['expert_initial_count']==10000 and reg['prior_native_initial_count']==12 and len(reg['families'])==88
        for p,h in reg['prior_inputs'].items():assert sha256(resolve(p))==h,p
        assert sum(x['split']=='calibration' for x in reg['families'].values())==24 and all(x['nearest_prior_max_coordinate_difference']>1e-8 for x in reg['families'].values())
    for seed in manifest['seeds']:
        p=out/f'{seed}_nominal.json';meta=read(p);a=dict(np.load(p.with_suffix('.npz')));assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(p.with_suffix('.npz'));n=native['control_steps'];dt=native['physics_dt'];sub=25;assert a['actions'].shape==(n,5) and a['actions'].dtype==np.float32 and np.max(abs(a['actions']))<=1
        env,obs,info,_,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);initial=meter.measure();check('fresh_plan',agent._plan,a['plan'],0)
        for key,value in initial.items():check('fresh_initial_'+key,value,a['initial_'+key],0)
        if not smoke:
            r=reg['families'][str(seed)];assert r['split']=='reserved_test';check('registered_qpos',a['initial_qpos'],r['initial_qpos'],0);check('registered_qvel',a['initial_qvel'],r['initial_qvel'],0);check('registered_goal',a['goal_position'],r['goal_position'],0)
        initialroot=np.r_[info['proprio/effector_pos'],info['proprio/effector_yaw'][0]];model=env._model;grip_id=env._gripper_opening_joint_id;site=env._pinch_site_id;scratch=mujoco.MjData(model);fresh=meter.rollout(a['actions'])
        for key,value in fresh['physics'].items():check('fresh_dense_'+key,value,a['physics_'+key],0)
        for key,value in fresh['control'].items():check('fresh_control_'+key,value,a['control_'+key],0)
        physics_samples+=n*sub;roots=[];grippers=np.clip(np.r_[a['initial_qpos'][grip_id],a['control_qpos'][:,grip_id]]/.8,0,1)
        for step in range(n):
            if step==0:root=initialroot
            else:
                scratch.qpos[:]=a['physics_qpos'][step*sub-2];mujoco.mj_kinematics(model,scratch);mat=scratch.site_xmat[site].reshape(3,3);root=np.r_[scratch.site_xpos[site],np.arctan2(mat[1,0],mat[0,0])]
            roots.append(root)
        roots=np.array(roots);unwrapped=np.unwrap(np.r_[roots[0,3],yaw(a['physics_eef_quaternion'])]);points=np.column_stack([np.r_[a['initial_eef_position'][None],a['physics_eef_position']],unwrapped]);delta=.5*(a['actions']+1)*(a['action_high']-a['action_low'])+a['action_low'];targets=roots+delta[:,:4];targets[:,:3]=np.clip(targets[:,:3],a['bounds'][0],a['bounds'][1]);targets[:,3]=roots[:,3]+wrap(targets[:,3]-roots[:,3]);targets[:,3]+=2*np.pi*np.round((unwrapped[1::sub]-targets[:,3])/(2*np.pi));targetphysics=np.repeat(targets,sub,axis=0);griptargets=np.clip(grippers[:-1]+delta[:,4],0,1);step=0
        for bi in range(meta['blocks']):
            get=lambda key:a[f'block_{bi:03d}_{key}'];h=1 if step<cfg['warmup_controls'] else cfg['horizon'];h=min(h,n-step);assert int(get('step'))==step and int(get('horizon'))==h and int(get('last_observation_control'))==step;check('causal_root',get('root'),roots[step],1e-12);count=min(step*sub,cfg['fit_physics_history']);check('causal_points',get('past_points'),points[step*sub-count:step*sub+1],1e-11);check('causal_targets',get('past_targets'),targetphysics[step*sub-count:step*sub],1e-11);check('causal_gripper_history',get('gripper_history'),grippers[:step+1],0);check('causal_gripper_targets',get('gripper_targets'),griptargets[:step],1e-12);v=(points[step*sub]-points[step*sub-1])/dt if step else np.zeros(4);check('causal_velocity',get('velocity'),v,1e-9)
            if count>=2:
                gain,res=fit_response(get('past_points'),get('past_targets'),dt);check('past_fit',get('gains'),gain,1e-8);check('past_residual',get('residual'),res,1e-8)
            else:check('default_fit',get('gains'),np.tile(cfg['default_pd'],(4,1)),0)
            err=griptargets[:step]-grippers[:step];change=np.diff(grippers[:step+1]);den=err@err;gg=float(np.clip(err@change/den,0,1)) if den>1e-10 else cfg['default_gripper_gain'];check('gripper_fit',get('gripper_gain'),gg,1e-12);indices=np.minimum(np.floor((float(get('root_time'))+np.arange(h)*.05-float(a['plan_t_init'])+1e-7)/.05).astype(int),len(a['plan'])-1);check('known_plan_indices',get('plan_indices'),indices,0);check('known_plan_targets',get('targets'),a['plan'][indices],0);actions,pred,grip=independent_compile(get('targets'),get('root'),get('velocity'),float(get('gripper')),get('gains'),float(get('gripper_gain')),a['bounds'],a['action_low'],a['action_high'],'past_pd_h5');check('independent_actions',actions,get('actions'),0);check('delivered_actions',actions,a['actions'][step:step+h],0);check('independent_prediction',pred,get('predicted'),1e-11);check('independent_gripper',grip,get('predicted_gripper'),1e-12);step+=h;blocks+=1
        assert step==n;check('clock',a['physics_time'],float(a['initial_time'])+.002*np.arange(1,7501),1e-9);dist=np.linalg.norm(a['control_object_pose'][:,:3]-a['goal_position'],axis=1);success=dist<=.04;check('success',success,a['success'],0);assert bool(success.any())==meta['success_any'] and bool(success[-1])==meta['success_final'];check('final_distance',dist[-1],meta['final_position_error_m'],1e-12);released=bool(success[-1] and a['control_robot_object_contacts'][-1]==0);assert released==meta['released_success_final'];d=np.diff(a['physics_eef_position'],n=3,axis=0)/dt**3;rows.append({'seed':seed,'initial_pose_family':seed,'success_any':bool(success.any()),'success_final':bool(success[-1]),'released_success_final':released,'final_position_error_m':float(dist[-1]),'eef_jerk_energy':float((d*d).sum()*dt)});env.close()
        if smoke:
            with np.load(BASE/'cube_plan_chunks'/f'{seed}_past_pd_h5.npz') as old:
                for key,value in a.items():check('old_smoke_'+key,value,old[key],0)
        for ext in ['.json','.npz']:q=p.with_suffix(ext);inputs[str(q.relative_to(resolve('.')))]=sha256(q)
        print('AUDIT NEW CUBE NOMINAL',seed,flush=True)
    f=pd.DataFrame(rows);assert len(f)==done['episodes'];f.to_csv(out/'evaluation.csv',index=False);write_json(out/'analysis.json',{'smoke':smoke,'episodes':len(f),'summary':{key:interval(f,key) for key in ['success_any','success_final','released_success_final','final_position_error_m','eef_jerk_energy']},'scope':manifest['config']['scope']});write_json(out/'raw_audit.json',{'smoke':smoke,'episodes':len(f),'blocks':blocks,'fresh_physics_samples':physics_samples,'fresh_native_plans':len(f),'max_errors':errors,'inputs':inputs,'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'source_sha256':sha256(Path(__file__)),'audit_helpers':{p:sha256(resolve(p)) for p in ['scripts/analyze_followup_cube_plan_chunks.py','scripts/audit_followup_dinowm_policy_final.py']},'scope':'Independent final nominal reference only. Fresh every physical/control sample and initial nativeplan; independent command compilation and causal robot/gripper histories, fit, action/time/success checks. No correction efficacy claim; reserved policy actions executed only after final method lock.'});print('AUDITED NEW CUBE NOMINAL',len(f),blocks,physics_samples,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
