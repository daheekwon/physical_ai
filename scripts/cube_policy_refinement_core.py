"""Causal Cube policy candidates; quantize applied actions before scoring."""
import time
import numpy as np
import torch
from scipy.optimize import minimize,LinearConstraint
from zj_wam.cube_experiment import basis,fit_response,predict_response,predicted_cost


def pipeline(context,meta,cfg,profile,method,rule,model,mean,scale):
    assert cfg['device']=='cpu';started=time.perf_counter()
    nominal=context['nominal_actions'];suffix=context['suffix_actions'];points=context['past_points'];bounds=context['workspace_bounds'];q=basis()
    gains,residual=fit_response(points,context['past_targets'],cfg['physics_dt']);velocity=(points[-1]-points[-2])/cfg['physics_dt'];prefix=points[-75:];rawmap=scale[np.arange(25)%5,None]*q
    def response(a):return predict_response(np.concatenate([a,suffix]),points[-1],velocity,gains,bounds,cfg['physics_dt'],cfg['substeps'])
    reference=response(nominal);j0=predicted_cost(reference,prefix,cfg['physics_dt'],cfg['yaw_length_scale_m']);normalizer=max(j0,1e-12)
    def cost(z):return predicted_cost(response(nominal+(rawmap@z).reshape(5,5)),prefix,cfg['physics_dt'],cfg['yaw_length_scale_m'])/normalizer
    uses_wm=profile['bank']=='common' or method=='lewm_global';predict=None;encode_calls=0;jac_calls=0;prediction_calls=0
    x0=((nominal-mean)/scale).reshape(-1)
    if uses_wm:
        _,predict=model.predictor(context['history_images'],context['past_actions']);encode_calls=1
    dirs={}
    if profile['bank']=='common':
        tx=torch.tensor(x0,dtype=torch.float32,device=cfg['device']);jac=torch.autograd.functional.jacobian(predict,tx,vectorize=True).detach().cpu().numpy().astype(float);jac_calls=1
        _,_,vt=np.linalg.svd(jac@q,full_matrices=False);dirs.update(lewm_low=vt[-1],lewm_high=vt[0])
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],meta['episode'],meta['timestep']]))
    for k in range(2):
        v=rng.normal(size=q.shape[1]);dirs['random'+str(k)]=v/np.linalg.norm(v)
    actions=[nominal.copy()];labels=[('nominal',0.,0)];coefficients=[np.zeros(q.shape[1])];solver_success=[True];solver_iterations=[0];optimizer_diagnostics=[]
    for radius in profile['radii']:
        for name,d in dirs.items():
            for sign in [-1,1]:
                z=sign*radius*d;a=nominal+(rawmap@z).reshape(5,5)
                if np.max(np.abs(a))<=1+1e-9:actions.append(a);coefficients.append(z);labels.append((name,radius,sign));solver_success.append(True);solver_iterations.append(0)
        constraints=[LinearConstraint(rawmap,-1-nominal.reshape(-1),1-nominal.reshape(-1)),{'type':'ineq','fun':lambda z,r=radius:r*r-z@z}]
        before=len(actions);opt=minimize(cost,np.zeros(q.shape[1]),method='SLSQP',constraints=constraints,options={'maxiter':80,'ftol':1e-9});z=opt.x;a=nominal+(rawmap@z).reshape(5,5)
        if np.max(np.abs(a))<=1+1e-7 and np.linalg.norm(z)<=radius+1e-6 and cost(z)<=1+1e-8:
            actions.append(a);coefficients.append(z);labels.append(('past_pd_optimizer',radius,0));solver_success.append(bool(opt.success));solver_iterations.append(int(opt.nit))
        optimizer_diagnostics.append([radius,int(opt.success),int(opt.nit),float(opt.fun),int(len(actions)>before)])
    actions=np.asarray(actions,dtype=np.float32);pred=np.asarray([response(a) for a in actions]);costs=np.array([predicted_cost(p,prefix,cfg['physics_dt'],cfg['yaw_length_scale_m']) for p in pred]);norm=np.linalg.norm(((actions-nominal)/scale).reshape(len(actions),-1),axis=1)
    delta=pred-pred[:1];position=np.linalg.norm(delta[:,:,:3],axis=-1);angle=np.abs(np.arctan2(np.sin(delta[:,:,3]),np.cos(delta[:,:,3])))
    contact=bool(float(context['root_gripper_contact'])>.1 or np.any(np.r_[nominal[:,4],suffix[:,4]]>.1))
    if method=='controller_displacement':scores=np.maximum(position.max(1)/cfg['position_tolerance_m'],angle.max(1)/cfg['rotation_tolerance_rad'])
    elif method=='proprio_contact':scores=np.maximum((position+np.sqrt(3)*.02*angle).max(1)/cfg['position_tolerance_m'],angle.max(1)/cfg['rotation_tolerance_rad']) if contact else np.zeros(len(actions))
    elif method=='lewm_global':
        latents=[]
        with torch.no_grad():
            for a in actions:
                tx=torch.tensor(x0+((a-nominal)/scale).reshape(-1),dtype=torch.float32,device=cfg['device']);latents.append(predict(tx).cpu().numpy());prediction_calls+=1
        latent=np.asarray(latents);scores=np.linalg.norm(latent-latent[0],axis=1)
    else:raise ValueError(method)
    allowed=(norm>1e-6)&(norm<=rule.get('radius',.4)+1e-6)&(scores<=rule.get('tau',-np.inf))&(costs<costs[0]-1e-10)
    if rule.get('disabled',False):allowed[:]=False
    ids=np.flatnonzero(allowed);selected=int(ids[np.lexsort((ids,costs[ids]))[0]]) if len(ids) else 0
    elapsed=time.perf_counter()-started
    return {'actions':actions,'labels':labels,'coefficients':np.asarray(coefficients),'costs':costs,'norm':norm,'scores':scores,'selected':selected,'seconds':elapsed,'encoder_calls':encode_calls,'jacobian_calls':jac_calls,'finite_prediction_calls':prediction_calls,'gains':gains,'residual':residual,'predicted_robot':pred,'basis':q,'nominal_contact_proxy':contact,'solver_success':np.array(solver_success),'solver_iterations':np.array(solver_iterations),'optimizer_diagnostics':np.array(optimizer_diagnostics)}
