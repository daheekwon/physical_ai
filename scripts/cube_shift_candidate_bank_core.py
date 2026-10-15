"""Recreate a shifted Cube correction root from audited causal histories."""
import json
import numpy as np
from zj_wam.io import resolve
from zj_wam.cube_execution import DenseCube,snapshot
from scripts.run_followup_cube_native_policy import scene
from scripts.cube_shift_preflight_core import baseline
from scripts.cube_shift_history_core import control_rollout,points,warm_history
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def replay_root(meta,history,condition,native,hcfg):
    seed=meta['seed'];change=meta['change_step'];step=meta['step'];source=dict(np.load(BASE/'cube_policy_refinement'/f'{seed}_nominal.npz'))
    env,obs,info,_,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);np.testing.assert_array_equal(agent._plan,source['plan']);meter=DenseCube(env);initial=meter.measure();original=baseline(env);original['visual_statistics']={k:np.array(getattr(env._model.stat,k),copy=True) for k in ['center','extent','meansize']}
    pre,targets=control_rollout(env,meter,source['actions'][:change]);saved=snapshot(env)
    for k,v in pre.items():np.testing.assert_array_equal(v,source['physics_'+k][:change*25])
    fresh=warm_history(env,meter,saved,original,condition,source['actions'][change:step],points(initial,pre),targets,hcfg)
    for k,v in fresh.items():np.testing.assert_array_equal(v,history[k])
    return env,meter,snapshot(env)

def selector_context(history,compiled):
    return {'nominal_actions':compiled['actions'][:5],'suffix_actions':compiled['actions'][5:],'past_points':history['past_points'],'past_targets':history['past_targets'],'workspace_bounds':history['workspace_bounds'],'history_images':history['history_images'],'past_actions':history['past_actions'],'root_gripper_contact':float(history['root_gripper_contact'])}
