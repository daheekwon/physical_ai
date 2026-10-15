"""Cube execution helpers: exact branch snapshots and non-mutating dense measurement."""
import copy
import numpy as np
import mujoco

SNAPSHOT_ATTRIBUTES=['_reset_next_step','_success','_prev_qpos','_prev_qvel','_prev_ob_info','_target_effector_pose']

def make_env(seed=20261011):
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
    env=CubeEnv(env_type='single',ob_type='states',mode='data_collection',width=224,height=224,terminate_at_goal=False,visualize_info=False)
    # Current CubeEnv.reset omits forwarding its seed argument; set the owned RNG explicitly.
    env._np_random=np.random.default_rng(seed);env.reset()
    return env


def initialize_recorded(env,qpos,qvel,control,time_value,target_pos):
    env._data.ctrl[:]=control;env.set_state(np.asarray(qpos),np.asarray(qvel));env._data.qacc_warmstart[:]=0.;env._data.time=float(time_value)
    env.set_target_pos(0,np.asarray(target_pos));env._reset_next_step=False
    env._prev_qpos=env._data.qpos.copy();env._prev_qvel=env._data.qvel.copy();env.pre_step();env.post_step()


def snapshot(env):
    return {'data':copy.deepcopy(env._data),'ik_data':copy.deepcopy(env._ik._data),'attributes':{k:copy.deepcopy(getattr(env,k)) for k in SNAPSHOT_ATTRIBUTES if hasattr(env,k)},'rng':copy.deepcopy(env.np_random.bit_generator.state),'geom_rgba':env._model.geom_rgba.copy()}


def restore(env,saved):
    env._data=copy.deepcopy(saved['data']);env._ik._data=copy.deepcopy(saved['ik_data'])
    for k,v in saved['attributes'].items():setattr(env,k,copy.deepcopy(v))
    env.np_random.bit_generator.state=copy.deepcopy(saved['rng']);env._model.geom_rgba[:]=saved['geom_rgba']
    # Do not mj_forward: preserve the same derived caches the controller originally sees.


def quat_error(a,b):
    a=np.asarray(a);b=np.asarray(b);a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return 2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=-1)),0,1))


class DenseCube:
    def __init__(self,env):
        self.env=env;self.scratch=mujoco.MjData(env._model);self.cube_geoms=set(env._cube_geom_ids_list[0]);self.object_joint=env._model.joint('object_joint_0').id;self.object_dof=int(env._model.jnt_dofadr[self.object_joint])
    def measure(self):
        env=self.env;self.scratch.qpos[:]=env._data.qpos;mujoco.mj_kinematics(env._model,self.scratch);quat=np.empty(4);mujoco.mju_mat2Quat(quat,self.scratch.site_xmat[env._pinch_site_id]);contacts=0
        for c in env._data.contact[:env._data.ncon]:
            if c.geom1 in self.cube_geoms or c.geom2 in self.cube_geoms:
                other=c.geom2 if c.geom1 in self.cube_geoms else c.geom1
                if 'ur5e' in env._model.body(int(env._model.geom_bodyid[other])).name:contacts+=1
        return {'qpos':env._data.qpos.copy(),'qvel':env._data.qvel.copy(),'object_pose':env._data.joint('object_joint_0').qpos.copy(),'object_velocity':env._data.qvel[self.object_dof:self.object_dof+6].copy(),'eef_position':self.scratch.site_xpos[env._pinch_site_id].copy(),'eef_quaternion':quat,'robot_object_contacts':contacts,'time':float(env._data.time)}
    def rollout(self,actions,dense=True):
        samples=[];control=[];original=mujoco.mj_step
        def step(model,data,nstep=1):
            if data is not self.env._data:return original(model,data,nstep=nstep)
            for _ in range(nstep):original(model,data,nstep=1);samples.append(self.measure())
        if dense:mujoco.mj_step=step
        try:
            for action in actions:self.env.step(action);control.append(self.measure())
        finally:mujoco.mj_step=original
        return {'physics':{k:np.asarray([r[k] for r in samples]) for k in samples[0]} if samples else {},'control':{k:np.asarray([r[k] for r in control]) for k in control[0]}}
