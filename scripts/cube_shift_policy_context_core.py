"""Compile known Cube plan commands from current robot and past observations only."""
import numpy as np
from zj_wam.cube_experiment import fit_response
from scripts.cube_plan_chunk_core import compile_actions

def compile_current(context,plan,plan_t_init,bounds,low,high,compiler):
    # Explicit inputs exclude object state, material factors and future outcomes.
    points=context['past_points'];targets=context['past_targets'];opening=context['gripper_history'];griptargets=context['gripper_targets']
    assert len(points)==len(targets)+1 and len(opening)==len(griptargets)+1
    gains,residual=fit_response(points,targets,.002);velocity=(points[-1]-points[-2])/.002
    error=griptargets-opening[:-1];delta=np.diff(opening);den=float(error@error)
    gg=float(np.clip(error@delta/den,0,1)) if den>1e-10 else compiler['default_gripper_gain']
    indices=np.minimum(np.floor((context['root_time']+np.arange(8)*.05-plan_t_init+1e-7)/.05).astype(int),len(plan)-1)
    actions,pred,grip=compile_actions(plan[indices],context['root_proprio'],velocity,float(opening[-1]),gains,gg,bounds,low,high,'past_pd_h5')
    return {'actions':actions,'plan_indices':indices,'targets':plan[indices],'gains':gains,'residual':residual,'gripper_gain':np.asarray(gg),'velocity':velocity,'predicted_robot_control':pred,'predicted_gripper_control':grip}
