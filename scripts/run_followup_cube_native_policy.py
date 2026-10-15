"""Nominal native Cube closed-loop compatibility before chunked refinement."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def scene(seed,cfg):
    np.random.seed(seed);env=CubeEnv(env_type='single',ob_type='states',mode='data_collection',width=224,height=224,terminate_at_goal=False,visualize_info=False);rng=np.random.default_rng(seed);values={};rejections=0
    for key in ['start_position','start_yaw','goal_position','goal_yaw']:
        space=env.variation_space['cube'][key];values['cube.'+key]=rng.uniform(space.low,space.high).astype(np.float64)
    while np.linalg.norm(values['cube.start_position']-values['cube.goal_position'])<cfg['minimum_start_goal_separation_m']:
        space=env.variation_space['cube']['goal_position'];values['cube.goal_position']=rng.uniform(space.low,space.high).astype(np.float64);rejections+=1;assert rejections<10000
    space=env.variation_space['agent']['ee_start_position'];values['agent.ee_start_position']=rng.uniform(space.low,space.high).astype(np.float64)
    env._np_random=np.random.default_rng(seed);obs,info=env.reset(options={'variation':[],'variation_values':values});assert abs(env._control_timestep-cfg['control_dt'])<1e-12 and abs(env._model.opt.timestep-cfg['physics_dt'])<1e-12
    return env,obs,info,values,rejections


def rollout(seed,kind,cfg,dense):
    env,obs,info,values,rejections=scene(seed,cfg);agent=CubeMarkovOracle(env=env,min_norm=cfg['markov_min_norm']) if kind=='markov' else CubePlanOracle(env=env,segment_dt=cfg['plan_segment_dt'],noise=cfg['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);initial=meter.measure();goal=env._data.mocap_pos[env._cube_target_mocap_ids[0]].copy();goalquat=env._data.mocap_quat[env._cube_target_mocap_ids[0]].copy();controls=[];physics=[];actions=[];success=[];done=[];times=[]
    for step in range(cfg['control_steps']):
        before=time.perf_counter();raw=agent.select_action(obs,info);action=np.asarray(np.clip(raw,-1,1),dtype=np.float32);times.append(time.perf_counter()-before);assert action.shape==(5,) and np.isfinite(action).all()
        if dense:
            result=meter.rollout(action[None],dense=True);physics.append(result['physics']);sample={k:v[-1] for k,v in result['control'].items()};obs=env.compute_observation();info=env.get_step_info()
        else:obs,_,terminated,truncated,info=env.step(action);assert not terminated and not truncated;sample=meter.measure()
        assert not env._reset_next_step;np.testing.assert_array_equal(env._data.mocap_pos[env._cube_target_mocap_ids[0]],goal);np.testing.assert_array_equal(env._data.mocap_quat[env._cube_target_mocap_ids[0]],goalquat)
        controls.append(sample);actions.append(action);success.append(bool(info['success']));done.append(bool(agent.done))
    env.close();arrays={'actions':np.array(actions),'success':np.array(success),'oracle_done':np.array(done),'policy_seconds':np.array(times),'goal_position':goal,'goal_quaternion':goalquat,**{'initial_'+k:np.asarray(v) for k,v in initial.items()},**{'control_'+k:np.array([v[k] for v in controls]) for k in controls[0]}}
    if dense:arrays.update({'physics_'+k:np.concatenate([v[k] for v in physics]) for k in physics[0]})
    distance=np.linalg.norm(arrays['control_object_pose'][:,:3]-goal,axis=1);np.testing.assert_array_equal(arrays['success'],distance<=.04)
    return arrays,{'seed':seed,'policy':kind,'success_any':bool(np.any(success)),'success_final':bool(success[-1]),'first_success_control':int(np.flatnonzero(success)[0]) if np.any(success) else None,'final_position_error_m':float(distance[-1]),'final_robot_contacts':int(arrays['control_robot_object_contacts'][-1]),'final_released_success':bool(success[-1] and arrays['control_robot_object_contacts'][-1]==0),'steps':len(actions),'physics_samples':len(arrays.get('physics_time',[])),'variation_values':{k:v.tolist() for k,v in values.items()},'start_goal_rejected_draws':rejections,'dense':dense}


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();cfg=read(resolve('configs/followup_cube_native_policy.json'));out=BASE/('cube_native_policy_smoke' if args.smoke else 'cube_native_policy');out.mkdir(exist_ok=True)
    import inspect,ogbench.manipspace.oracles.markov.markov_oracle as markov,ogbench.manipspace.oracles.plan.plan_oracle as plan
    paths=[Path(__file__),resolve('configs/followup_cube_native_policy.json'),resolve('zj_wam/cube_execution.py'),Path(inspect.getfile(CubeEnv)),Path(inspect.getfile(CubeMarkovOracle)),Path(inspect.getfile(CubePlanOracle)),Path(inspect.getfile(markov)),Path(inspect.getfile(plan))]
    record={'smoke':args.smoke,'config':cfg,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==record
    else:write_json(out/'manifest.json',record)
    for seed in (cfg['development_seeds'][:1] if args.smoke else cfg['development_seeds']):
        for kind in cfg['policies']:
            dest=out/f'{seed}_{kind}.json'
            if dest.exists():
                old=read(dest);assert old['manifest_sha256']==sha256(out/'manifest.json') and old['data_sha256']==sha256(dest.with_suffix('.npz'));continue
            started=time.perf_counter();arrays,meta=rollout(seed,kind,cfg,True);checks={}
            if seed in cfg['validation_seeds']:
                plain,_=rollout(seed,kind,cfg,False)
                for key,value in plain.items():
                    if key=='policy_seconds':continue
                    np.testing.assert_array_equal(value,arrays[key]);checks[key]=0.
            np.savez_compressed(dest.with_suffix('.npz'),**arrays);write_json(dest,{**meta,'plain_reproduction_max_error':checks,'seconds':time.perf_counter()-started,'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz'))});print(seed,kind,meta['success_any'],meta['success_final'],round(meta['final_position_error_m'],4),'seconds',round(time.perf_counter()-started,2),flush=True)
    for p,h in record['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'episodes':2 if args.smoke else 24,'manifest_sha256':sha256(out/'manifest.json'),'scope':cfg['scope']})
if __name__=='__main__':main()
