"""Validate causalH5 command compilation before Cube policy refinement."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube
from zj_wam.cube_experiment import xyz_yaw
from scripts.run_followup_cube_native_policy import scene
from scripts.cube_plan_chunk_core import fit_past,compile_actions,wrap
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def execute(seed,mode,cfg,native):
    env,obs,info,values,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);initial=meter.measure();goal=env._data.mocap_pos[env._cube_target_mocap_ids[0]].copy();goalquat=env._data.mocap_quat[env._cube_target_mocap_ids[0]].copy();bounds=np.asarray(env._workspace_bounds).copy();physics=[];control=[];allactions=[];success=[];blocks=[];history=[np.r_[initial['eef_position'],info['proprio/effector_yaw'][0]]];past_targets=[];grippers=[float(info['proprio/gripper_opening'][0])];griptargets=[];step=0
    while step<native['control_steps']:
        h=1 if mode=='native_h1' or step<cfg['warmup_controls'] else cfg['horizon'];h=min(h,native['control_steps']-step);root=np.r_[info['proprio/effector_pos'],info['proprio/effector_yaw'][0]];g0=float(info['proprio/gripper_opening'][0]);gains,residual,gg=fit_past(history,past_targets,grippers,griptargets,cfg);p=np.array(history);vel=(p[-1]-p[-2])/.002 if len(p)>1 else np.zeros(4);indices=np.array([min(int((float(info['time'][0])+j*.05-agent._t_init+1e-7)//.05),len(agent._plan)-1) for j in range(h)]);targets=agent._plan[indices].copy()
        actions,predicted,predg=compile_actions(targets,root,vel,g0,gains,gg,bounds,env.action_low,env.action_high,mode);native_first=np.asarray(agent.select_action(obs,info),dtype=np.float32);np.testing.assert_array_equal(actions[0],native_first)
        again,_,_=compile_actions(targets[:1],root,vel,g0,gains,gg,bounds,env.action_low,env.action_high,mode);np.testing.assert_array_equal(again[0],actions[0])
        blocks.append({'step':step,'horizon':h,'root_time':float(info['time'][0]),'plan_indices':indices,'targets':targets,'root':root,'velocity':vel,'gripper':g0,'gains':gains,'residual':residual,'gripper_gain':gg,'predicted':predicted,'predicted_gripper':predg,'actions':actions,'past_points':np.array(history[-cfg['fit_physics_history']-1:]),'past_targets':np.array(past_targets[-cfg['fit_physics_history']:]).reshape(-1,4),'gripper_history':np.array(grippers),'gripper_targets':np.array(griptargets),'last_observation_control':step})
        # The full chunk is fixed before any of its physical outcomes is observed.
        for action in actions:
            before=np.r_[info['proprio/effector_pos'],info['proprio/effector_yaw'][0]];opening=float(info['proprio/gripper_opening'][0]);delta=env.unnormalize_action(action);target=before+delta[:4];target[:3]=np.clip(target[:3],bounds[0],bounds[1]);target[3]=before[3]+wrap(target[3]-before[3]);griptarget=float(np.clip(opening+delta[4],0,1));result=meter.rollout(action[None],dense=True);trace=result['physics'];q=xyz_yaw(trace);offset=2*np.pi*round((history[-1][3]-q[0,3])/(2*np.pi));q[:,3]+=offset;target[3]+=2*np.pi*round((q[0,3]-target[3])/(2*np.pi));history.extend(q);past_targets.extend(np.repeat(target[None],25,axis=0));physics.append(trace);sample={k:v[-1] for k,v in result['control'].items()};control.append(sample);obs=env.compute_observation();info=env.get_step_info();grippers.append(float(info['proprio/gripper_opening'][0]));griptargets.append(griptarget);allactions.append(action);success.append(bool(info['success']));step+=1;np.testing.assert_array_equal(env._data.mocap_pos[env._cube_target_mocap_ids[0]],goal)
    env.close();arrays={'actions':np.array(allactions),'success':np.array(success),'goal_position':goal,'goal_quaternion':goalquat,'plan':agent._plan,'plan_t_init':agent._t_init,'bounds':bounds,'action_low':np.array(env.action_low),'action_high':np.array(env.action_high),**{'initial_'+k:np.asarray(v) for k,v in initial.items()},**{'control_'+k:np.array([v[k] for v in control]) for k in control[0]},**{'physics_'+k:np.concatenate([v[k] for v in physics]) for k in physics[0]}}
    for i,b in enumerate(blocks):
        for k,v in b.items():arrays[f'block_{i:03d}_{k}']=np.asarray(v)
    checks={}
    if mode=='native_h1':
        with np.load(BASE/'cube_native_policy'/f'{seed}_plan.npz') as source:
            for k,v in source.items():
                if k in arrays:np.testing.assert_array_equal(v,arrays[k]);checks[k]=0.
    distance=np.linalg.norm(arrays['control_object_pose'][:,:3]-goal,axis=1);np.testing.assert_array_equal(arrays['success'],distance<=.04)
    return arrays,{'seed':seed,'mode':mode,'blocks':len(blocks),'success_any':bool(np.any(success)),'success_final':bool(success[-1]),'released_success_final':bool(success[-1] and arrays['control_robot_object_contacts'][-1]==0),'final_position_error_m':float(distance[-1]),'native_h1_reproduction':checks,'variation_values':{k:v.tolist() for k,v in values.items()}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();out=BASE/('cube_plan_chunks_smoke' if args.smoke else 'cube_plan_chunks');out.mkdir(exist_ok=True);cfg=read(resolve('configs/followup_cube_plan_chunks.json'));native=read(resolve('configs/followup_cube_native_policy.json'));paths=[resolve(p) for p in ['configs/followup_cube_plan_chunks.json','configs/followup_cube_native_policy.json','scripts/cube_plan_chunk_core.py','scripts/run_followup_cube_plan_chunks.py','scripts/run_followup_cube_native_policy.py','zj_wam/cube_experiment.py','zj_wam/cube_execution.py']]+[BASE/'cube_native_policy/manifest.json',BASE/'cube_native_policy/raw_audit.json']
    for seed in cfg['development_seeds']:paths.extend(BASE/'cube_native_policy'/f'{seed}_plan{ext}' for ext in ['.json','.npz'])
    m={'smoke':args.smoke,'config':cfg,'native':native,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==m
    else:write_json(out/'manifest.json',m)
    for seed in (cfg['development_seeds'][:1] if args.smoke else cfg['development_seeds']):
        for mode in cfg['modes']:
            dest=out/f'{seed}_{mode}.json'
            if dest.exists():assert read(dest)['manifest_sha256']==sha256(out/'manifest.json');continue
            started=time.perf_counter();arrays,meta=execute(seed,mode,cfg,native);np.savez_compressed(dest.with_suffix('.npz'),**arrays);write_json(dest,{**meta,'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'seconds':time.perf_counter()-started});print(seed,mode,meta['success_final'],meta['final_position_error_m'],round(time.perf_counter()-started,2),flush=True)
    for name,h in m['inputs'].items():assert sha256(resolve(name))==h,name
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'episodes':3 if args.smoke else 36,'manifest_sha256':sha256(out/'manifest.json'),'scope':cfg['scope']})
if __name__=='__main__':main()
