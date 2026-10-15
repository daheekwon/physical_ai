"""Fresh CPU refinement from causal observation buffers, without future physics."""
import time
import numpy as np
import torch
from scipy.optimize import minimize,LinearConstraint
from zj_wam.followup import optimize_controller,controller_rollout
from zj_wam.diagnostics import polynomial_basis,jerk_energy
from zj_wam.cube_experiment import basis,fit_response,predict_response,predicted_cost
from zj_wam.repeated_refinement import visual_history
from zj_wam.response_models import score_models
from zj_wam.model_audit import predict_candidates


def generate(env,c,cfg):
    nominal=c['nominal'];suffix=c['suffix'];scale=c['scale'];actions=[nominal.copy()];labels=[('nominal',0.,0)];solver=[]
    if env=='pusht':
        q=polynomial_basis(5);rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],c['episode'],c['timestep'],5]));dirs=rng.normal(size=(8,q.shape[1]));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
        for relative in [.05,.1,.2]:
            radius=relative*np.sqrt(10)
            for objective in ['executed','raw']:
                a,info=optimize_controller(nominal,suffix,c['root'],c['prefix'],scale,radius,objective,c['past'][-3:]);solver.append({'radius':relative,'objective':objective,'success':info['solver_success'],'iterations':info['iterations'],'accepted':info['feasible']})
                if np.max(abs(a))<=1+1e-6:actions.append(a);labels.append(('optimizer_'+objective,relative,0))
            for i,d in enumerate(dirs):
                for sign in [-1,1]:
                    a=nominal+sign*radius*(q@d).reshape(5,2)*scale
                    if np.max(abs(a))<=1+1e-6:actions.append(a);labels.append(('random_'+str(i),relative,sign))
        def response(a):return controller_rollout(np.r_[a,suffix],c['root'][:2],c['root'][5:7])
        def cost(p):return jerk_energy(np.r_[c['prefix'][:,:2],p],.01)
    else:
        q=basis();gains,residual=fit_response(c['points'],c['targets'],.002);velocity=(c['points'][-1]-c['points'][-2])/.002;rawmap=scale[np.arange(25)%5,None]*q
        def response(a):return predict_response(np.r_[a,suffix],c['points'][-1],velocity,gains,c['bounds'],.002,25)
        def cost(p):return predicted_cost(p,c['points'][-75:],.002,.05)
        c0=cost(response(nominal));rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],c['episode'],c['timestep']]));dirs=rng.normal(size=(2,q.shape[1]));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
        for radius in [.025,.05,.1,.2]:
            for i,d in enumerate(dirs):
                for sign in [-1,1]:
                    a=nominal+(rawmap@(sign*radius*d)).reshape(5,5)
                    if np.max(abs(a))<=1+1e-9:actions.append(a);labels.append(('random'+str(i),radius,sign))
            def objective(z):return cost(response(nominal+(rawmap@z).reshape(5,5)))/max(c0,1e-12)
            constraints=[LinearConstraint(rawmap,-1-nominal.reshape(-1),1-nominal.reshape(-1)),{'type':'ineq','fun':lambda z,r=radius:r*r-z@z}];result=minimize(objective,np.zeros(q.shape[1]),method='SLSQP',constraints=constraints,options={'maxiter':80,'ftol':1e-9});z=result.x;a=nominal+(rawmap@z).reshape(5,5);valid=bool(np.max(abs(a))<=1+1e-7 and np.linalg.norm(z)<=radius+1e-6 and objective(z)<=1+1e-8);solver.append({'radius':radius,'objective':'past_pd','success':bool(result.success),'iterations':int(result.nit),'accepted':valid})
            if valid:actions.append(a);labels.append(('past_pd_optimizer',radius,0))
    actions=np.asarray(actions);robot=np.array([response(a) for a in actions]);costs=np.array([cost(p) for p in robot]);norms=np.linalg.norm(((actions-actions[:1])/scale).reshape(len(actions),-1),axis=1)
    return actions,labels,robot,costs,norms,solver


def run(env,c,cfg,method,rule,lewm=None,dino=None,response_cfg=None):
    start=time.perf_counter();actions,labels,robot,costs,norms,solver=generate(env,c,cfg);generation=time.perf_counter()-start;encode_seconds=0.;prediction_seconds=0.;encode_calls=0;prediction_calls=0;score_start=time.perf_counter()
    if method in ['controller','contact']:
        assert lewm is None and dino is None
        if env=='pusht':
            if method=='controller':scores=np.linalg.norm(robot-robot[:1],axis=2).max(1)
            else:
                observed,valid=visual_history(c['images'],c['proprio'],response_cfg);values,_=score_models(actions,c['suffix'],observed[-1],observed,response_cfg);scores=values['contact'] if valid else np.full(len(actions),np.inf)
        else:
            delta=robot-robot[:1];dp=np.linalg.norm(delta[:,:,:3],axis=2);dr=abs(np.arctan2(np.sin(delta[:,:,3]),np.cos(delta[:,:,3])));scores=np.maximum(dp.max(1)/.002,dr.max(1)/np.deg2rad(2))
            if method=='contact':
                contact=c['gripper_contact']>.1 or np.any(np.r_[c['nominal'][:,4],c['suffix'][:,4]]>.1);scores=np.maximum((dp+np.sqrt(3)*.02*dr).max(1)/.002,dr.max(1)/np.deg2rad(2)) if contact else np.zeros(len(actions))
    elif method=='lewm_global':
        assert lewm is not None and dino is None;clock=time.perf_counter();_,context=lewm.predictor(c['images'],c['past']);encode_seconds=time.perf_counter()-clock;encode_calls=1;clock=time.perf_counter()
        with torch.no_grad():
            if env=='pusht':latents=predict_candidates(lewm,context['embedding'],context['past'],actions,batch_size=64)[:,0];prediction_calls=int(np.ceil(len(actions)/64))
            else:
                latents=np.array([context(torch.tensor(((a-c['mean'])/c['scale']).reshape(-1),dtype=torch.float32,device='cpu')).numpy() for a in actions]);prediction_calls=len(actions)
        prediction_seconds=time.perf_counter()-clock;scores=np.linalg.norm(latents-latents[:1],axis=1)
    else:
        assert method in ['dino_patch_rms','dino_patch_mean'] and dino is not None and lewm is None;clock=time.perf_counter();encoded=dino.encode(c['images']);encode_seconds=time.perf_counter()-clock;encode_calls=1;clock=time.perf_counter()
        with torch.no_grad():latents=np.array([dino.predict(encoded,c['past'],a)[0].numpy() for a in actions],dtype=np.float64)
        prediction_seconds=time.perf_counter()-clock;prediction_calls=len(actions);delta=latents-latents[:1];scores=np.sqrt(np.mean(np.sum(delta**2,axis=2),axis=1)) if method=='dino_patch_rms' else np.linalg.norm(delta.mean(1),axis=1)
    score_seconds=time.perf_counter()-score_start;selection_start=time.perf_counter();gain=(costs[0]-costs)/max(costs[0],1e-12);allowed=(norms>1e-6)&(gain>1e-8)&(scores<=rule['tau']) if rule['enabled'] else np.zeros(len(actions),bool);ids=np.flatnonzero(allowed);index=int(ids[np.lexsort((ids,norms[ids],costs[ids]))[0]]) if len(ids) else 0;selection_seconds=time.perf_counter()-selection_start;total=time.perf_counter()-start
    return {'actions':actions,'labels':labels,'costs':costs,'norms':norms,'scores':scores,'selected':index,'solver':solver,'generation_seconds':generation,'score_seconds':score_seconds,'encode_seconds':encode_seconds,'prediction_seconds':prediction_seconds,'selection_seconds':selection_seconds,'total_seconds':total,'encode_calls':encode_calls,'prediction_calls':prediction_calls,'jacobian_calls':0}
