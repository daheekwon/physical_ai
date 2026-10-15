"""Past-only allowance and actual staged-computation adaptation in one polynomial bank."""
import time
import numpy as np
from zj_wam.diagnostics import polynomial_basis,jerk_energy
from zj_wam.followup import optimize_controller
from zj_wam.response_models import controller_batch,contact_rollout,fit_past_models,response_score
from zj_wam.repeated_refinement import visual_history


def past_holdout_error(history,past,rcfg):
    # Fit interval [-10,-5], predict [-5,0], then compare to now; no future input.
    gains=fit_past_models(history[:2],rcfg)['gains']
    pusher=controller_batch(np.asarray(past[-5:])[None],history[1])
    poses,_=contact_rollout(pusher,history[1],rcfg,gains)
    delta=poses[0,-1]-history[2,2:5]
    components=np.array([np.linalg.norm(delta[:2])/2.,abs(np.arctan2(np.sin(delta[2]),np.cos(delta[2])))/np.deg2rad(2),np.linalg.norm(pusher[0,-1]-history[2,:2])/2.])
    return float(components.max()),components


def pick(costs,norms,scores,valid,tau,radius,horizon=5):
    if not valid:return 0
    allowed=np.flatnonzero(np.isfinite(scores)&(scores<=tau)&(norms>1e-6)&(norms<=radius*np.sqrt(2*horizon)+1e-7))
    if not len(allowed):return 0
    index=int(allowed[np.lexsort((norms[allowed],costs[allowed]))[0]])
    return index if costs[0]>1e-8 and (costs[0]-costs[index])/costs[0]>1e-8 else 0


def make_stage(nominal,suffix,root,prefix,past,scale,seed,radii,stage):
    q=polynomial_basis(len(nominal));actions=[];labels=[];solver=[]
    if stage==1:
        actions=[nominal.copy()];labels=['nominal']
    rng=np.random.default_rng(seed);directions=rng.normal(size=(8,q.shape[1]));directions/=np.linalg.norm(directions,axis=1)[:,None]
    for relative in radii:
        radius=relative*np.sqrt(nominal.size)
        objective='executed' if stage==1 else 'raw'
        raw,info=optimize_controller(nominal,suffix,root,prefix,scale,radius,objective,past[-3:],basis=q)
        actions.append(raw);labels.append(f'{relative:g}:optimizer_{objective}');solver.append({'success':info['solver_success'],'iterations':info['iterations']})
        if stage==2:
            for i,d in enumerate(directions):
                for sign in [-1,1]:
                    raw=nominal+sign*radius*(q@d).reshape(nominal.shape)*scale
                    if np.max(np.abs(raw))<=1+1e-6:actions.append(raw);labels.append(f'{relative:g}:random_{i}_{sign}')
    return np.asarray(actions),labels,solver


def score(actions,nominal,suffix,history,prefix,scale,rcfg):
    # nominal always at index zero; permits independent scoring of only new candidates.
    future=np.concatenate([actions,np.broadcast_to(suffix,(len(actions),*suffix.shape))],axis=1)
    pusher=controller_batch(future,history[-1])
    gains=fit_past_models(history,rcfg)['gains'];poses,_=contact_rollout(pusher,history[-1],rcfg,gains)
    costs=np.array([jerk_energy(np.concatenate([prefix[:,:2],q]),.01) for q in pusher])
    norms=np.linalg.norm(((actions-nominal)/scale).reshape(len(actions),-1),axis=1)
    return costs,norms,response_score(poses)


def decide(images,proprio,past,prefix,nominal,suffix,scale,seed,radii,rcfg,acfg,rule,method,alpha=0.,constant_radius=None):
    started=time.perf_counter();timing={};observed,valid=visual_history(images,proprio,rcfg);timing['vision']=time.perf_counter()-started
    allowance=method in ['allowance','both'];dynamic=method in ['compute','both']
    residual=0.;components=np.zeros(3);ts=time.perf_counter()
    if allowance and valid:residual,components=past_holdout_error(observed,past,rcfg)
    radius=rule['relative_radius']/(1+alpha*min(residual,acfg['residual_cap'])) if allowance else rule['relative_radius']
    if method=='constant_shrink':radius=float(constant_radius)
    timing['allowance']=time.perf_counter()-ts
    # Invalid image pose: all image-based rules fall back without unnecessary optimization.
    if not valid or not rule['enabled']:
        return nominal.copy(),{'method':method,'index':0,'active':False,'vision_valid':bool(valid),'radius':float(radius),'alpha':float(alpha),'past_error':None,'past_error_components':None,'predicted_gain':0.,'actual_norm':0.,'score':None,'stages':0,'early_stop':False,'scored_candidates':0,'solver_calls':0,'solver_failures':0,'seconds':time.perf_counter()-started,'timing':timing,'labels':['nominal']},{'actions':nominal[None].copy(),'costs':np.array([0.]),'norms':np.array([0.]),'scores':np.array([np.inf]),'observed':observed}
    ts=time.perf_counter();actions,labels,solvers=make_stage(nominal,suffix,observed[-1],prefix,past,scale,seed,radii,1);timing['generation']=time.perf_counter()-ts
    ts=time.perf_counter();costs,norms,scores=score(actions,nominal,suffix,observed,prefix,scale,rcfg);timing['scoring']=time.perf_counter()-ts
    ts=time.perf_counter();index=pick(costs,norms,scores,valid,rule['tau'],radius);gain=(costs[0]-costs[index])/max(costs[0],1e-8);upper=(costs[0]-costs.min())/max(costs[0],1e-8)
    early=bool(dynamic and index>0 and gain>=acfg['early_stop_gain'] and gain>=acfg['early_stop_fraction']*upper)
    timing['selection']=time.perf_counter()-ts;scored=len(actions);stages=1
    if method!='cheap_only' and not early:
        ts=time.perf_counter();extra,names,more=make_stage(nominal,suffix,observed[-1],prefix,past,scale,seed,radii,2);timing['generation']+=time.perf_counter()-ts
        ts=time.perf_counter();c,n,s=score(np.concatenate([nominal[None],extra]),nominal,suffix,observed,prefix,scale,rcfg);timing['scoring']+=time.perf_counter()-ts
        actions=np.concatenate([actions,extra]);labels+=names;solvers+=more;costs=np.r_[costs,c[1:]];norms=np.r_[norms,n[1:]];scores=np.r_[scores,s[1:]];scored+=len(extra)+1;stages=2
        ts=time.perf_counter();index=pick(costs,norms,scores,valid,rule['tau'],radius);timing['selection']+=time.perf_counter()-ts
    gain=(costs[0]-costs[index])/max(costs[0],1e-8)
    decision={'method':method,'index':int(index),'active':bool(norms[index]>1e-6),'vision_valid':bool(valid),'radius':float(radius),'alpha':float(alpha),'past_error':float(residual) if allowance else None,'past_error_components':components.tolist() if allowance else None,'predicted_gain':float(gain),'actual_norm':float(norms[index]),'score':float(scores[index]),'stages':stages,'early_stop':early,'scored_candidates':scored,'solver_calls':len(solvers),'solver_failures':sum(not x['success'] for x in solvers),'seconds':time.perf_counter()-started,'timing':timing,'labels':labels}
    return actions[index].copy(),decision,{'actions':actions,'costs':costs,'norms':norms,'scores':scores,'observed':observed}
