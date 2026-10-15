"""Constrained smoothing baselines; selection sees model predictions only."""
import time
import numpy as np
import torch
from scipy.optimize import LinearConstraint,minimize


def quadratic_solve(nominal,raw_map,j0,design,radius,extra_matrix=None,extra_bound=None,penalty=None,weight=0.,initial=None):
    d=raw_map.shape[1];normalizer=max(float(j0@j0),1.)
    penalty=np.zeros((d,d)) if penalty is None else penalty
    def value(z):
        e=j0+design@z
        return float(e@e/normalizer+weight*(z@penalty@z))
    def gradient(z):return 2*design.T@(j0+design@z)/normalizer+2*weight*penalty@z
    constraints=[LinearConstraint(raw_map,-1-nominal.reshape(-1),1-nominal.reshape(-1)),
                 {'type':'ineq','fun':lambda z:radius**2-z@z,'jac':lambda z:-2*z}]
    if extra_matrix is not None:constraints.append(LinearConstraint(extra_matrix,-np.asarray(extra_bound),np.asarray(extra_bound)))
    start=time.perf_counter();result=minimize(value,np.zeros(d) if initial is None else initial,jac=gradient,method='SLSQP',constraints=constraints,options={'maxiter':200,'ftol':1e-10})
    z=result.x;raw=nominal+(raw_map@z).reshape(nominal.shape)
    valid=np.linalg.norm(z)<=radius+1e-6 and np.max(np.abs(raw))<=1+1e-6
    if extra_matrix is not None:valid=valid and np.max(np.abs(extra_matrix@z)-extra_bound)<=1e-5
    # Feasible incumbents are retained even if SLSQP does not certify convergence.
    if not valid or value(z)>value(np.zeros(d))+1e-8:z=np.zeros(d);raw=nominal.copy()
    return raw,{'coefficient':z,'solver_success':bool(result.success),'feasible':bool(valid),'iterations':int(result.nit),
                'message':str(result.message),'seconds':time.perf_counter()-start,'objective_ratio':value(z),'function_evaluations':int(result.nfev)}


class EvaluationBudget(Exception):pass


def nonlinear_solve(predict,normalized_nominal,basis,nominal,raw_map,j0,design,radius,tau,starts,budget=64):
    """Two starts share a fixed forward-evaluation budget; keep best model-feasible iterate."""
    a0=normalized_nominal.detach();q=torch.as_tensor(basis,dtype=a0.dtype,device=a0.device)
    with torch.no_grad():reference=predict(a0).detach()
    norm=max(float(j0@j0),1.);d=basis.shape[1];cache={};forward=1;backward=0;best=np.zeros(d);best_cost=float(j0@j0/norm)
    def objective(z):e=j0+design@z;return float(e@e/norm)
    def gradient(z):return 2*design.T@(j0+design@z)/norm
    status=[];start=time.perf_counter();start_limit=budget
    def evaluate(z):
        nonlocal forward,backward,best,best_cost
        key=np.asarray(z,dtype=np.float64).tobytes()
        if key in cache:return cache[key]
        if forward>=start_limit:raise EvaluationBudget()
        parameter=torch.tensor(z,dtype=a0.dtype,device=a0.device,requires_grad=True)
        difference=predict(a0+q@parameter)-reference;metric=difference.square().sum()
        grad=torch.autograd.grad(metric,parameter)[0];forward+=1;backward+=1
        value=float(metric.detach().cpu());derivative=grad.detach().cpu().numpy().astype(float)
        cache[key]=(value,derivative)
        raw=nominal+(raw_map@z).reshape(nominal.shape)
        feasible=np.isfinite(value) and value<=tau*tau*(1+1e-5)+1e-12 and np.linalg.norm(z)<=radius+1e-6 and np.max(np.abs(raw))<=1+1e-6
        cost=objective(z)
        if feasible and cost<best_cost:best=np.asarray(z).copy();best_cost=cost
        return value,derivative
    constraints=[LinearConstraint(raw_map,-1-nominal.reshape(-1),1-nominal.reshape(-1)),
                 {'type':'ineq','fun':lambda z:radius**2-z@z,'jac':lambda z:-2*z},
                 {'type':'ineq','fun':lambda z:1-evaluate(z)[0]/tau**2,'jac':lambda z:-evaluate(z)[1]/tau**2}]
    for i,initial in enumerate(starts):
        start_limit=min(budget,1+(budget-1)*(i+1)//len(starts))
        try:
            result=minimize(objective,np.asarray(initial),jac=gradient,method='SLSQP',constraints=constraints,options={'maxiter':200,'ftol':1e-9})
            evaluate(result.x);status.append({'success':bool(result.success),'iterations':int(result.nit),'message':str(result.message)})
        except EvaluationBudget:status.append({'success':False,'message':'prespecified forward-evaluation budget reached'})
    raw=nominal+(raw_map@best).reshape(nominal.shape)
    final_metric=cache.get(np.asarray(best,dtype=np.float64).tobytes(),(0.,None))[0]
    return raw,{'coefficient':best,'solver_success':any(s['success'] for s in status),'starts':status,'forward_calls':forward,'backward_calls':backward,
                'budget':budget,'model_delta':float(np.sqrt(final_metric)),'model_threshold':tau,'objective_ratio':best_cost,'seconds':time.perf_counter()-start}


def geometric_solve(nominal,raw_map,position_base,position_response,past_positions,radius,epsilon,method='lipo',executed_objective=None):
    """2D position-path adapters. Same 10Hz controller; no invented chunk overlap."""
    h=len(nominal);d=raw_map.shape[1]
    path=position_base[:h*10][9::10];response=position_response[:h*10][9::10]
    full=np.concatenate([past_positions[-3:],path]);extended=np.concatenate([np.zeros((3,2,d)),response])
    j0=np.diff(full,n=3,axis=0).reshape(-1);design=np.diff(extended,n=3,axis=0).reshape(-1,d)
    # GROOVE's tiny magnitude term applies to delivered displacement commands.
    if method=='groove':
        velocities=np.diff(np.concatenate([past_positions[-1:],path]),axis=0)
        derivative=np.diff(np.concatenate([np.zeros((1,2,d)),response]),axis=0)
        j0=np.r_[j0,np.sqrt(5e-9)*velocities.reshape(-1)];design=np.r_[design,np.sqrt(5e-9)*derivative.reshape(-1,d)]
    if executed_objective is not None:j0,design=executed_objective
    regions=[(np.eye(2),np.array([epsilon,epsilon]))]
    if method=='groove':
        for angle in [0.,np.pi/2,np.pi/4,-np.pi/4]:
            c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
            regions.append((rotation,np.array([epsilon/3,epsilon*3])))
    candidates=[];metadata=[]
    for rotation,width in regions:
        matrix=np.einsum('ji,tjk->tik',rotation,response).reshape(-1,d);bound=np.tile(width,h)
        raw,info=quadratic_solve(nominal,raw_map,j0,design,radius,matrix,bound)
        z=info['coefficient'];correction=np.einsum('tid,d->ti',response,z);changed=full.copy();changed[3:]+=correction
        B=float(np.sum(np.abs(correction)));P=float(np.linalg.norm(np.diff(changed,n=3,axis=0),axis=1).mean())
        info.update({'deviation':B,'jerk_score':P});candidates.append(raw);metadata.append(info)
    reference=metadata[0];eligible=[0]
    if method=='groove' and reference['feasible']:
        eligible += [i for i,m in enumerate(metadata[1:],1) if m['feasible'] and m['deviation']<=1.5*max(reference['deviation'],1e-12) and m['jerk_score']<reference['jerk_score']]
    chosen=min(eligible,key=lambda i:(metadata[i]['jerk_score'],i))
    return candidates[chosen],{'selected_region':chosen,'regions':metadata,'seconds':sum(m['seconds'] for m in metadata),
                              'solver_success':metadata[chosen]['solver_success'],'coefficient':metadata[chosen]['coefficient']}
