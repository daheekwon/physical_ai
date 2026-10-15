"""Observation-driven correction of a fixed expert action sequence."""
import time
import numpy as np,pandas as pd
from scipy.linalg import block_diag
from zj_wam.followup import optimize_controller,controller_rollout
from zj_wam.diagnostics import polynomial_basis,jerk_energy
from zj_wam.response_models import estimate_pose,score_models
from zj_wam.model_audit import predict_candidates
from zj_wam.selection import select_indices


def visual_history(images,proprio,cfg):
    observed=np.zeros((3,10));observed[:,[0,1,5,6]]=np.asarray(proprio)[:,[0,1,5,6]]
    poses=[];valid=True
    for image in images:
        pose,meta=estimate_pose(image,cfg);poses.append(pose);valid=valid and meta['valid']
    observed[:,2:5]=poses
    v=(observed[-1,2:5]-observed[-2,2:5])/.5;v[2]=np.arctan2(np.sin(observed[-1,4]-observed[-2,4]),np.cos(observed[-1,4]-observed[-2,4]))/.5
    observed[-1,7:10]=v
    return observed,valid


def generate_bank(nominal,suffix,root,prefix,past,scale,seed,radii,piecewise=False):
    h=len(nominal);q=block_diag(*[polynomial_basis(5) for _ in range(h//5)]) if piecewise else polynomial_basis(h)
    rng=np.random.default_rng(seed);directions=rng.normal(size=(8,q.shape[1]));directions/=np.linalg.norm(directions,axis=1)[:,None]
    actions=[nominal.copy()];labels=['nominal']
    for relative in radii:
        radius=relative*np.sqrt(2*h)
        for objective in ['executed','raw']:
            raw,info=optimize_controller(nominal,suffix,root,prefix,scale,radius,objective,past[-3:],basis=q)
            actions.append(raw);labels.append('optimizer_'+objective)
        for i,d in enumerate(directions):
            for sign in [-1,1]:
                raw=nominal+sign*radius*(q@d).reshape(h,2)*scale
                if np.max(np.abs(raw))<=1+1e-6:actions.append(raw);labels.append(f'random_{i}_{sign}')
    return np.array(actions),labels


def choose_correction(model,images,past,proprio,prefix,nominal,suffix,scale,seed,cfg,rcfg,rules,method,predicted_pose_history=None,piecewise=False):
    started=time.perf_counter();observed,valid=visual_history(images,proprio,rcfg)
    if predicted_pose_history is not None:
        # This arm observes robot proprioception, but receives no new object image pose.
        observed[:,2:5]=np.asarray(predicted_pose_history)
        observed[:,7:10]=0.
    root=observed[-1].copy();actions,labels=generate_bank(nominal,suffix,root,prefix,past,scale,seed,cfg['radii'],piecewise)
    if method=='lewm_global':
        _,context=model.predictor(images,past);pred=predict_candidates(model,context['embedding'],context['past'],actions)
        score=np.linalg.norm(pred-pred[:1],axis=-1).max(1);predicted_pose=None
    else:
        scores,info=score_models(actions,suffix,root,observed,rcfg)
        name='adaptive' if method=='vision_adaptive' else 'contact';score=scores[name]
        if not valid and predicted_pose_history is None:score[:]=np.inf
        predicted_pose=info['adaptive' if name=='adaptive' else 'fixed'][:,len(nominal)*10-1]
    costs=[]
    for raw in actions:
        q=controller_rollout(np.concatenate([raw,suffix]),root[:2],root[5:7]);costs.append(jerk_energy(np.concatenate([prefix[:,:2],q]),.01))
    norm=np.linalg.norm(((actions-actions[:1])/scale).reshape(len(actions),-1),axis=1)
    frame=pd.DataFrame({'state_id':['live']*len(actions),'horizon':[len(nominal)]*len(actions),'actual_norm':norm,
                        'predicted_jerk':costs,'nominal_predicted_jerk':costs[0],method+'_score':score})
    index=select_indices(frame,method,rules[method])['live'];index=max(0,index)
    pose=predicted_pose[index] if predicted_pose is not None else None
    return actions[index],{'selected_index':index,'selected_generator':labels[index],'active':bool(index>0 and norm[index]>1e-6),
                            'score':float(score[index]) if np.isfinite(score[index]) else None,'norm':float(norm[index]),
                            'predicted_gain':float((costs[0]-costs[index])/max(costs[0],1e-8)),'candidate_count':len(actions)-1,
                            'seconds':time.perf_counter()-started,'visual_valid':bool(valid),'observed_pose':observed[-1,2:5].tolist(),
                            'predicted_endpoint_pose':pose.tolist() if pose is not None else None},observed[:,2:5]
