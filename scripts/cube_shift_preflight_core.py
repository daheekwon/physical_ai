"""Model parameter interventions; root physics history remains unchanged."""
import numpy as np,mujoco
from zj_wam.cube_execution import restore

def baseline(env):
    m=env._model;geoms=np.array(env._cube_geom_ids_list[0],int);bodies=np.unique(m.geom_bodyid[geoms]);assert len(bodies)==1
    return {'geoms':geoms,'bodies':bodies,'body_mass':m.body_mass.copy(),'body_inertia':m.body_inertia.copy(),'geom_friction':m.geom_friction.copy()}

def prepare(env,root,original,condition):
    m=env._model;m.body_mass[:]=original['body_mass'];m.body_inertia[:]=original['body_inertia'];m.geom_friction[:]=original['geom_friction'];ids=original['bodies'];gids=original['geoms'];m.body_mass[ids]*=condition['mass_factor'];m.body_inertia[ids]*=condition['mass_factor'];m.geom_friction[gids,0]*=condition['friction_factor'];scratch=mujoco.MjData(m);mujoco.mj_setConst(m,scratch);restore(env,root)
    qpos=env._data.qpos.copy();qvel=env._data.qvel.copy();clock=env._data.time
    if condition['forward']:mujoco.mj_forward(m,env._data)
    np.testing.assert_array_equal(env._data.qpos,qpos);np.testing.assert_array_equal(env._data.qvel,qvel);assert env._data.time==clock
    return {'mass':m.body_mass[ids].copy(),'inertia':m.body_inertia[ids].copy(),'friction':m.geom_friction[gids].copy(),'root_qpos':qpos,'root_qvel':qvel,'root_time':np.asarray(clock)}
