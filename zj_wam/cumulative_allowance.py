"""Reference-available cumulative-deviation scores; no future execution labels."""
import time
import numpy as np
from zj_wam.repeated_refinement import visual_history,generate_bank
from zj_wam.response_models import score_models,controller_batch,contact_rollout
from zj_wam.diagnostics import jerk_energy
from zj_wam.adaptation_factorial import pick


def deviation_score(poses,reference,eps):
    delta=np.asarray(poses)-np.asarray(reference)
    position=np.linalg.norm(delta[...,:2],axis=-1).max(axis=-1)/eps['position']
    rotation=np.abs(np.arctan2(np.sin(delta[...,2]),np.cos(delta[...,2]))).max(axis=-1)/eps['rotation']
    return np.maximum(position,rotation)


def remaining_margin(root,reference_root,eps):
    error=deviation_score(np.asarray(root[2:5])[None,None],np.asarray(reference_root[2:5])[None,None],eps)[0]
    return float(max(0,1-error)),float(error)


def choose(images,proprio,past,prefix,reference_images,reference_proprio,nominal,suffix,scale,seed,cfg,rcfg,base_rule,tier,method,multiplier):
    started=time.perf_counter()
    if tier=='vision':
        history,valid=visual_history(images,proprio,rcfg);reference,reference_valid=visual_history(reference_images,reference_proprio,rcfg)
    elif tier=='state':history=np.asarray(proprio).copy();reference=np.asarray(reference_proprio).copy();valid=reference_valid=True
    else:raise ValueError(tier)
    root=history[-1];reference_root=reference[-1];margin,error=remaining_margin(root,reference_root,cfg['tolerances']);factor=multiplier*(margin if method=='remaining_margin' else 1.)
    radius=base_rule['relative_radius']*factor;tau=base_rule['tau']*factor
    candidates,labels=generate_bank(nominal,suffix,root,prefix,past,scale,seed,cfg['radii']);scores,info=score_models(candidates,suffix,root,history,rcfg)
    refq=controller_batch(np.concatenate([nominal,suffix])[None],reference_root);refpose,_=contact_rollout(refq,reference_root,rcfg,info['past']['gains'])
    global_score=deviation_score(info['adaptive'],refpose,cfg['tolerances']);local_score=scores['adaptive'];selected_score=global_score if method=='reference_forecast' else local_score
    costs=np.array([jerk_energy(np.concatenate([prefix[:,:2],q]),.01) for q in info['pusher']]);norms=np.linalg.norm(((candidates-candidates[:1])/scale).reshape(len(candidates),-1),axis=1)
    usable=valid and (reference_valid if method in ['remaining_margin','reference_forecast'] else True)
    index=pick(costs,norms,selected_score,usable,tau,radius);gain=(costs[0]-costs[index])/max(costs[0],1e-8)
    decision={'tier':tier,'method':method,'multiplier':float(multiplier),'index':int(index),'active':bool(norms[index]>1e-6),'observed_reference_error':error,'remaining_margin':margin,'effective_radius':float(radius),'effective_tau':float(tau),'predicted_gain':float(gain),'selected_local_score':float(local_score[index]),'selected_reference_score':float(global_score[index]),'nominal_reference_score':float(global_score[0]),'norm':float(norms[index]),'vision_valid':bool(valid),'reference_valid':bool(reference_valid),'seconds':time.perf_counter()-started,'predicted_nominal_violation':bool(global_score[0]>tau),'selected_generator':labels[index]}
    archive={'actions':candidates,'costs':costs,'norms':norms,'local_scores':local_score,'reference_scores':global_score,'history':history,'reference_history':reference,'predicted_poses':info['adaptive'],'predicted_reference':refpose,'past_gains':np.array(info['past']['gains'])}
    return candidates[index],decision,archive
