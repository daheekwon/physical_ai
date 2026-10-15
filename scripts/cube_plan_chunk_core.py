"""Causal native-plan command compiler without environment or future-state access."""
import numpy as np
from zj_wam.cube_experiment import fit_response,predict_response


def wrap(x):return np.arctan2(np.sin(x),np.cos(x))


def fit_past(points,targets,gripper_history,gripper_targets,cfg):
    p=np.asarray(points);t=np.asarray(targets);assert len(p)==len(t)+1
    n=min(len(t),cfg['fit_physics_history']);gains=np.tile(cfg['default_pd'],(4,1)).astype(float);residual=np.full(4,np.nan)
    if n>=2:gains,residual=fit_response(p[-n-1:],t[-n:],.002)
    g=np.asarray(gripper_history);gt=np.asarray(gripper_targets);assert len(g)==len(gt)+1
    error=gt-g[:-1];change=np.diff(g);den=float(error@error);gain=float(np.clip(error@change/den,0,1)) if den>1e-10 else cfg['default_gripper_gain']
    return gains,residual,gain


def compile_actions(targets,root,velocity,gripper,gains,gripper_gain,bounds,action_low,action_high,mode):
    """Inputs are known plan targets and current/past robot estimates only."""
    targets=np.asarray(targets);p=np.asarray(root,dtype=float).copy();v=np.asarray(velocity,dtype=float).copy();g=float(gripper);low=np.asarray(action_low);high=np.asarray(action_high);assert targets.ndim==2 and targets.shape[1]==5
    np.testing.assert_allclose((high-low)[:4]/2,[.05,.05,.05,.3],rtol=0,atol=1e-12)
    actions=[];predictions=[];grippers=[]
    for target in targets:
        difference=np.r_[target[:4]-p,target[4]-g];action=np.clip(2*(difference-low)/(high-low)-1,-1,1).astype(np.float32);delta=.5*(action.astype(float)+1)*(high-low)+low;desired=p+delta[:4];desired[:3]=np.clip(desired[:3],bounds[0],bounds[1]);desired[3]=wrap(desired[3]);gg=float(np.clip(g+delta[4],0,1));actions.append(action)
        if mode=='past_pd_h5':
            trace=predict_response(action[None],p,v,gains,bounds,.002,25);v=(trace[-1]-trace[-2])/.002;p=trace[-1].copy();p[3]=wrap(p[3]);g+=gripper_gain*(gg-g)
        else:p=desired;g=gg
        predictions.append(p.copy());grippers.append(g)
    return np.asarray(actions),np.asarray(predictions),np.asarray(grippers)
