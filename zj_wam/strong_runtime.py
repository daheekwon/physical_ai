"""Actual uncached strong/spectral refinement paths for computation comparison."""
import numpy as np,pandas as pd,torch
from zj_wam.locked_geometric import refine as geometric_refine
from zj_wam.followup import controller_design,controller_rollout,optimize_controller
from zj_wam.diagnostics import jerk_energy
from zj_wam.sensitivity import action_jacobian
from zj_wam.strong_optimizers import quadratic_solve,nonlinear_solve
from zj_wam.selection import select_indices


def refine(method,budget,model,images,past,root,prefix,nominal,suffix,scale,cfg,rules,strong):
    robot=np.zeros_like(root);robot[[0,1,5,6]]=root[[0,1,5,6]]
    history=np.zeros_like(prefix);history[:,:2]=prefix[:,:2]
    if method=='gaussian_action' or method.startswith(('lipo_','groove_')):
        a,info=geometric_refine(method,nominal,suffix,robot,history,past,scale,strong[method])
        return np.concatenate([a.astype(float),suffix]) if info['active'] else np.concatenate([nominal,suffix]),info
    q,base,response,j0,design=controller_design(nominal,suffix,robot,history,scale)
    predict,_=model.predictor(images,past);a0=torch.tensor(model.normalize(nominal).reshape(-1),device=model.device)
    tau=rules['lewm_global']['tau'];rawmap=q*np.tile(scale,len(nominal))[:,None]
    info={};before=jerk_energy(np.concatenate([history[:,:2],base]),.01)
    if method.endswith('_low2_lewm'):
        if method=='lewm_low2_lewm':
            jac=action_jacobian(predict,a0).cpu().numpy();local=jac@q
        else:local=response[:len(nominal)*10].reshape(-1,q.shape[1])
        _,vectors=np.linalg.eigh(local.T@local);basis=q@vectors[:,:2]
        candidates=[nominal.astype(float)];solver=[]
        for relative in cfg['radii']:
            a,detail=optimize_controller(nominal,suffix,robot,history,scale,relative*np.sqrt(2*len(nominal)),basis=basis);candidates.append(a);solver.append(detail)
        with torch.no_grad():
            preds=[predict(torch.tensor(model.normalize(a).reshape(-1),device=model.device)).cpu().numpy() for a in candidates]
        scores=np.linalg.norm(np.asarray(preds)-preds[0],axis=1)
        costs=np.array([jerk_energy(np.concatenate([history[:,:2],controller_rollout(np.concatenate([a,suffix]),robot[:2],robot[5:7])]),.01) for a in candidates])
        norms=np.linalg.norm(((np.asarray(candidates)-nominal)/scale).reshape(len(candidates),-1),axis=1)
        frame=pd.DataFrame({'state_id':'current','horizon':len(nominal),'actual_norm':norms,'predicted_jerk':costs,'nominal_predicted_jerk':before,'lewm_global_score':scores})
        index=max(0,select_indices(frame,'lewm_global',rules['lewm_global'])['current']);a=candidates[index]
        info={'solver_success':all(x['solver_success'] for x in solver),'candidate_count':len(candidates),'selected_index':index,'finite_model_delta':float(scores[index]),'jacobian_computed':method=='lewm_low2_lewm'}
    else:
        rule=strong[method];radius=rule['radius']*np.sqrt(2*len(nominal))
        if method=='soft_sensitivity':
            jac=action_jacobian(predict,a0).cpu().numpy();penalty=(jac@q).T@(jac@q)/tau**2
            a,info=quadratic_solve(nominal,rawmap,j0,design,radius,penalty=penalty,weight=rule['setting'])
        elif method=='nonlinear_wm':
            _,ci=optimize_controller(nominal,suffix,robot,history,scale,radius)
            a,info=nonlinear_solve(predict,a0,q,nominal,rawmap,j0,design,radius,tau*rule['setting'],[np.zeros(q.shape[1]),ci['coefficient']],budget=budget)
        else:raise ValueError(method)
    after=jerk_energy(np.concatenate([history[:,:2],controller_rollout(np.concatenate([a,suffix]),robot[:2],robot[5:7])]),.01)
    norm=float(np.linalg.norm((a-nominal)/scale));gain=(before-after)/before if before>1e-8 else 0.
    active=norm>1e-6 and gain>1e-8
    if not method.endswith('_low2_lewm'):active=active and strong[method]['enabled']
    if not active:a=nominal.astype(float);norm=0.;gain=0.;after=before
    info.update({'active':bool(active),'actual_norm':norm,'predicted_gain':gain,'predicted_nominal_jerk':before,'predicted_selected_jerk':after})
    assert np.max(np.abs(a))<=1+1e-6
    return np.concatenate([a,suffix]) if active else np.concatenate([nominal,suffix]),info
