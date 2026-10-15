"""Warmup after known model interventions, preserving causal model histories."""
import numpy as np
from zj_wam.cube_experiment import xyz_yaw
from scripts.cube_shift_preflight_core import prepare

def control_rollout(env,meter,actions):
    chunks=[];targets=[]
    for action in actions:
        trace=meter.rollout(np.asarray(action)[None])['physics'];chunks.append(trace);target=np.r_[env._target_effector_pose.translation(),env._target_effector_pose.rotation().compute_yaw_radians()];targets.extend(np.repeat(target[None],25,axis=0))
    return ({k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]} if chunks else {}),np.array(targets).reshape(-1,4)

def points(initial,physics):
    full={key:np.r_[initial[key][None],physics[key]] for key in ['eef_position','eef_quaternion']};return xyz_yaw(full)

def prepare_history(env,saved,original,condition):
    params=prepare(env,saved,original,condition)
    # mj_setConst recomputes XML-overridden visual statistics. Retain the native
    # camera/lighting scale; mass/inertia-derived physical statistics stay updated.
    stats=original['visual_statistics'];env._model.stat.center[:]=stats['center'];env._model.stat.extent=stats['extent'];env._model.stat.meansize=stats['meansize']
    params.update(visual_center=np.asarray(env._model.stat.center).copy(),visual_extent=np.asarray(env._model.stat.extent),visual_meansize=np.asarray(env._model.stat.meansize))
    return params

def warm_history(env,meter,saved,original,condition,actions,pre_points,pre_targets,cfg):
    params=prepare_history(env,saved,original,condition);initial=meter.measure();images=[env.render().copy()];all_trace=[];targets=[]
    for k,action in enumerate(actions):
        trace,command=control_rollout(env,meter,action[None]);all_trace.append(trace);targets.extend(command)
        if k in [4,9]:images.append(env.render().copy())
    physics={key:np.concatenate([x[key] for x in all_trace]) for key in all_trace[0]};p=points(initial,physics);targets=np.asarray(targets);targets[:,3]+=2*np.pi*np.round((p[1:,3]-targets[:,3])/(2*np.pi));length=min(cfg['maximum_fit_physics_steps'],len(pre_targets));assert length>=2 and len(p)==251 and len(targets)==250
    previous=np.array(pre_points[-length-1:],copy=True);oldtargets=np.array(pre_targets[-length:],copy=True);oldtargets[:,3]+=2*np.pi*np.round((previous[1:,3]-oldtargets[:,3])/(2*np.pi));info=env.compute_ob_info()
    context={'history_images':np.array(images),'past_actions':np.array(actions),'past_points':p,'past_targets':targets,'prechange_points':previous,'prechange_targets':oldtargets,'postchange_fit_points':p[-length-1:],'postchange_fit_targets':targets[-length:],'fit_physics_steps':np.asarray(length),'workspace_bounds':np.asarray(env._workspace_bounds),'root_gripper_contact':np.asarray(float(np.asarray(info['proprio/gripper_contact']).reshape(-1)[0])),**{'warmup_'+key:value for key,value in physics.items()},**{'change_root_'+key:np.asarray(value) for key,value in initial.items()},**{'root_'+key:np.asarray(value) for key,value in meter.measure().items()},**{'parameter_'+key:value for key,value in params.items()}}
    return context
