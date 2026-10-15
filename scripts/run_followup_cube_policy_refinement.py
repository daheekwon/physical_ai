"""Own-compiler Cube repeated refinement; local branches are evaluation only."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube,snapshot,restore
from zj_wam.cube_experiment import xyz_yaw,metrics
from scripts.run_followup_cube_native_policy import scene
from scripts.cube_plan_chunk_core import fit_past,compile_actions,wrap
from scripts.cube_policy_refinement_core import pipeline
from scripts.run_followup_cube import Model
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def execute(seed,method,cfg,compiler,native,model,mean,scale,rules):
    env,obs,info,values,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);initial=meter.measure();goal=env._data.mocap_pos[env._cube_target_mocap_ids[0]].copy();goalquat=env._data.mocap_quat[env._cube_target_mocap_ids[0]].copy();bounds=np.asarray(env._workspace_bounds).copy();physics=[];control=[];allactions=[];success=[];blocks=[];history=[np.r_[initial['eef_position'],info['proprio/effector_yaw'][0]]];past_targets=[];grippers=[float(info['proprio/gripper_opening'][0])];griptargets=[];images={0:env.render().copy()};step=0;extra={};localrows=[];ccfg=read(resolve('configs/followup_cube.json'));ccfg.update(device=cfg['device'],yaw_length_scale_m=cfg['yaw_length_scale_m']);profile={'bank':'non_wm_proposals','radii':cfg['radii']}
    while step<native['control_steps']:
        h=1 if step<cfg['warmup_controls'] else cfg['horizon'];h=min(h,native['control_steps']-step);root=np.r_[info['proprio/effector_pos'],info['proprio/effector_yaw'][0]];g0=float(info['proprio/gripper_opening'][0]);gains,residual,gg=fit_past(history,past_targets,grippers,griptargets,compiler);hp=np.array(history);vel=(hp[-1]-hp[-2])/.002 if len(hp)>1 else np.zeros(4);width=h+(cfg['suffix'] if h==5 else 0);indices=np.array([min(int((float(info['time'][0])+j*.05-agent._t_init+1e-7)//.05),len(agent._plan)-1) for j in range(width)]);targets=agent._plan[indices].copy();compiled,pred,predg=compile_actions(targets,root,vel,g0,gains,gg,bounds,env.action_low,env.action_high,'past_pd_h5');nominal=compiled[:h].copy();suffix=compiled[h:].copy();np.testing.assert_array_equal(nominal[0],np.asarray(agent.select_action(obs,info),dtype=np.float32));chosen=nominal.copy();selected=0;bank=None;branch_actual=None;bi=len(blocks)
        block={'step':step,'horizon':h,'root_time':float(info['time'][0]),'plan_indices':indices,'targets':targets,'root':root,'velocity':vel,'gripper':g0,'gains':gains,'residual':residual,'gripper_gain':gg,'predicted':pred,'predicted_gripper':predg,'nominal_actions':nominal,'suffix_actions':suffix,'past_points':hp[-compiler['fit_physics_history']-1:],'past_targets':np.array(past_targets[-compiler['fit_physics_history']:]).reshape(-1,4),'gripper_history':np.array(grippers),'gripper_targets':np.array(griptargets),'last_observation_control':step}
        if h==5:
            context={'nominal_actions':nominal,'suffix_actions':suffix,'past_points':hp[-251:],'past_targets':np.array(past_targets[-250:]),'workspace_bounds':bounds,'history_images':np.array([images[step-10],images[step-5],images[step]]),'past_actions':np.array(allactions[-10:]),'root_gripper_contact':float(np.asarray(info['proprio/gripper_contact']).reshape(-1)[0])};block.update(history_images=context['history_images'],past_actions=context['past_actions'],root_gripper_contact=context['root_gripper_contact'],history_control_indices=np.array([step-10,step-5,step]))
            if method!='nominal':
                bank=pipeline(context,{'episode':seed,'timestep':step},ccfg,profile,method,rules[method],model,mean,scale);selected=bank['selected'];chosen=bank['actions'][selected].copy();np.testing.assert_array_equal(chosen[:,4],nominal[:,4]);np.testing.assert_array_equal(bank['actions'][:,:,4],np.broadcast_to(nominal[:,4],bank['actions'].shape[:2]));assert chosen.dtype==np.float32
                for key in ['actions','coefficients','costs','norm','scores','gains','residual','predicted_robot','basis','solver_success','solver_iterations','optimizer_diagnostics']:extra[f'bank_{bi:03d}_{key}']=bank[key]
                extra[f'bank_{bi:03d}_labels']=np.array(bank['labels'],dtype=str);block.update(candidate_count=len(bank['actions']),generation_seconds=bank['seconds'],encoder_calls=bank['encoder_calls'],jacobian_calls=bank['jacobian_calls'],finite_prediction_calls=bank['finite_prediction_calls'])
            else:block.update(candidate_count=1,generation_seconds=0.,encoder_calls=0,jacobian_calls=0,finite_prediction_calls=0)
            # Choice is finalized before either future physical branch is evaluated.
            saved=snapshot(env);prefix={k:np.concatenate([p[k] for p in physics])[-75:].copy() for k in physics[0]};baseline=meter.rollout(np.r_[nominal,suffix])['physics'];restore(env,saved)
            if selected:branch_actual=meter.rollout(np.r_[chosen,suffix])['physics'];restore(env,saved)
            else:branch_actual=baseline
            local=metrics(branch_actual,baseline,prefix,ccfg);localrows.append({'step':step,'block':bi,'selected':selected,'active':bool(selected),'candidate_count':block['candidate_count'],'generation_seconds':block['generation_seconds'],**local})
            for tag,trace in [('nominal',baseline),('modified',branch_actual),('prefix',prefix)]:
                for key,val in trace.items():extra[f'local_{bi:03d}_{tag}_{key}']=val
        block.update(selected=selected,actions=chosen);blocks.append(block);first=len(physics)
        for action in chosen:
            before=np.r_[info['proprio/effector_pos'],info['proprio/effector_yaw'][0]];opening=float(info['proprio/gripper_opening'][0]);delta=env.unnormalize_action(action);target=before+delta[:4];target[:3]=np.clip(target[:3],bounds[0],bounds[1]);target[3]=before[3]+wrap(target[3]-before[3]);griptarget=float(np.clip(opening+delta[4],0,1));result=meter.rollout(action[None]);trace=result['physics'];points=xyz_yaw(trace);offset=2*np.pi*round((history[-1][3]-points[0,3])/(2*np.pi));points[:,3]+=offset;target[3]+=2*np.pi*round((points[0,3]-target[3])/(2*np.pi));history.extend(points);past_targets.extend(np.repeat(target[None],25,axis=0));physics.append(trace);control.append({k:v[-1] for k,v in result['control'].items()});obs=env.compute_observation();info=env.get_step_info();grippers.append(float(info['proprio/gripper_opening'][0]));griptargets.append(griptarget);allactions.append(action);success.append(bool(info['success']));step+=1
            if step%5==0:images[step]=env.render().copy()
            np.testing.assert_array_equal(env._data.mocap_pos[env._cube_target_mocap_ids[0]],goal)
        if branch_actual is not None:
            for key in physics[0]:np.testing.assert_array_equal(np.concatenate([p[key] for p in physics[first:]]),branch_actual[key][:h*25])
    env.close();arrays={'actions':np.array(allactions),'success':np.array(success),'goal_position':goal,'goal_quaternion':goalquat,'plan':agent._plan,'plan_t_init':agent._t_init,'bounds':bounds,'action_low':np.array(env.action_low),'action_high':np.array(env.action_high),'scale':scale,'mean':mean,'history_image_control_indices':np.array(sorted(images)),'history_images':np.array([images[k] for k in sorted(images)]),**{'initial_'+k:np.asarray(v) for k,v in initial.items()},**{'control_'+k:np.array([v[k] for v in control]) for k in control[0]},**{'physics_'+k:np.concatenate([v[k] for v in physics]) for k in physics[0]},**extra}
    for i,b in enumerate(blocks):
        for k,v in b.items():arrays[f'block_{i:03d}_{k}']=np.asarray(v)
    if method=='nominal':
        with np.load(BASE/'cube_plan_chunks'/f'{seed}_past_pd_h5.npz') as z:
            for k in ['actions','success','goal_position','goal_quaternion','physics_qpos','physics_qvel','physics_object_pose','physics_eef_position','physics_time']:np.testing.assert_array_equal(arrays[k],z[k])
    return arrays,{'seed':seed,'method':method,'blocks':len(blocks),'local_blocks':len(localrows),'local_rows':localrows,'success_final':bool(success[-1]),'scope':cfg['scope']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('cube_policy_refinement_smoke' if args.smoke else 'cube_policy_refinement');out.mkdir(exist_ok=True);cfg=read(resolve('configs/followup_cube_policy_refinement.json'));compiler=read(resolve('configs/followup_cube_plan_chunks.json'));native=read(resolve('configs/followup_cube_native_policy.json'));cube=read(resolve('configs/followup_cube.json'));cube.update(device=cfg['device']);paths=[Path(__file__),resolve('configs/followup_cube_policy_refinement.json'),resolve('scripts/cube_policy_refinement_core.py'),resolve('scripts/cube_plan_chunk_core.py'),resolve('scripts/run_followup_cube_native_policy.py'),resolve('scripts/run_followup_cube.py'),resolve('zj_wam/cube_execution.py'),resolve('zj_wam/cube_experiment.py'),BASE/'cube_plan_chunks/raw_audit.json',BASE/'cube_plan_chunks/manifest.json',BASE/'cube/proposal_ablation/analysis.json',BASE/'cube/manifest.json',resolve(cube['checkpoint'])/'weights.pt']
    for seed in cfg['development_seeds']:paths.extend([BASE/'cube_plan_chunks'/f'{seed}_past_pd_h5.json',BASE/'cube_plan_chunks'/f'{seed}_past_pd_h5.npz'])
    manifest={'smoke':args.smoke,'config':cfg,'compiler':compiler,'native':native,'cube':cube,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    torch.set_num_threads(cfg['threads']);m=read(BASE/'cube/manifest.json')['normalization'];mean=np.array(m['mean']);scale=np.array(m['scale']);model=Model(cube,mean,scale);rules=read(BASE/'cube/proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules']
    for seed in cfg['development_seeds'][:1] if args.smoke else cfg['development_seeds']:
        for method in cfg['methods']:
            dest=out/f'{seed}_{method}.json'
            if dest.exists():r=read(dest);assert r['manifest_sha256']==sha256(out/'manifest.json') and r['data_sha256']==sha256(dest.with_suffix('.npz'));continue
            started=time.perf_counter();arrays,meta=execute(seed,method,cfg,compiler,native,model,mean,scale,rules);
            if args.smoke:
                with np.load(BASE/'cube_policy_refinement_initial_smoke_v1'/dest.with_suffix('.npz').name) as old:
                    for key in ['actions','success','physics_qpos','physics_qvel','physics_object_pose','physics_eef_position','physics_time']:np.testing.assert_array_equal(arrays[key],old[key])
                meta['initial_smoke_physics_exact']=True
            np.savez_compressed(dest.with_suffix('.npz'),**arrays);write_json(dest,{**meta,'seconds':time.perf_counter()-started,'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz'))});print(seed,method,meta['success_final'],'local',meta['local_blocks'],'seconds',round(time.perf_counter()-started,2),flush=True)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'episodes':4 if args.smoke else 48,'manifest_sha256':sha256(out/'manifest.json'),'scope':cfg['scope']})
if __name__=='__main__':main()
