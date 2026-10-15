"""Source adapters for audited Cube policy calibration and development nominals."""
import json
import numpy as np
from zj_wam.io import resolve
from scripts.cube_plan_chunk_core import compile_actions
BASE=resolve('artifacts/followup_v1')
def source_dir(seed):
    return BASE/('cube_policy_calibration_nominal' if 340100<=seed<340124 else 'cube_policy_refinement')
def split(seed):
    assert 340100<=seed<340124 or 340000<=seed<340012
    return 'calibration' if 340100<=seed<340124 else 'development'
def states(seed,cfg):
    with np.load(source_dir(seed)/f'{seed}_nominal.npz') as a:
        steps=np.arange(15,300,5);ix=steps*25-1;z=a['physics_object_pose'][ix,2];contact=a['physics_robot_object_contacts'][ix]>0;z0=float(a['initial_object_pose'][2]);chosen={'approach':15};g=steps[contact&(z<=z0+cfg['grasp_height_above_initial_max_m'])];chosen['grasp']=int(g[0]) if len(g) else None;t=steps[contact&(z>=z0+cfg['transport_height_above_initial_min_m'])];chosen['transport']=int(t[0]) if len(t) else None;r=steps[(~contact)&(z<=z0+cfg['grasp_height_above_initial_max_m'])&(steps>(chosen['transport'] if chosen['transport'] is not None else 300))];chosen['release']=int(r[0]) if len(r) else None
    return chosen

def root_commands(a,bi):
    get=lambda k:a[f'block_{bi:03d}_{k}']
    indices=np.minimum(np.floor((float(get('root_time'))+np.arange(8)*.05-float(a['plan_t_init'])+1e-7)/.05).astype(int),len(a['plan'])-1)
    commands,_,_=compile_actions(a['plan'][indices],get('root'),get('velocity'),float(get('gripper')),get('gains'),float(get('gripper_gain')),a['bounds'],a['action_low'],a['action_high'],'past_pd_h5')
    step=int(get('step'));np.testing.assert_array_equal(commands[:5],a['actions'][step:step+5])
    if f'block_{bi:03d}_nominal_actions' in a:
        np.testing.assert_array_equal(commands[:5],get('nominal_actions'));np.testing.assert_array_equal(commands[5:],get('suffix_actions'))
    return commands
