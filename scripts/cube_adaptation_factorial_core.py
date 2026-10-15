"""Actual staged Cube generation and past-only allowance in a fixed contact verifier."""
import time
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from zj_wam.cube_experiment import basis,fit_response,predict_response,predicted_cost

def past_error(context):
    p=context['past_points'];targets=context['past_targets'];assert len(p)==251 and len(targets)==250
    gains,_=fit_response(p[:126],targets[:125],.002);state=p[125].copy();velocity=(p[125]-p[124])/.002;pred=[]
    for target in targets[125:]:
        e=target-state;e[3]=np.arctan2(np.sin(e[3]),np.cos(e[3]));velocity+=.002*(gains[:,0]*e-gains[:,1]*velocity);state+=.002*velocity;pred.append(state.copy())
    pred=np.array(pred);d=pred-p[126:];components=np.array([np.linalg.norm(d[:,:3],axis=1).max()/.002,np.abs(np.arctan2(np.sin(d[:,3]),np.cos(d[:,3]))).max()/np.deg2rad(2)])
    return float(components.max()),components,pred,gains

def decide(context,meta,cfg,acfg,rule,mean,scale,mode,alpha=0.,constant_radius=None):
    assert mode in acfg['methods'];start=time.perf_counter();times={};ts=start;allowance=mode in ['allowance','both'];dynamic=mode in ['compute','both'];residual=0.;components=np.zeros(2)
    if allowance:residual,components,_,_=past_error(context)
    radius=acfg['base_radius']/(1+alpha*min(residual,acfg['residual_cap'])) if allowance else acfg['base_radius']
    if mode=='constant_shrink':radius=float(constant_radius)
    times['allowance']=time.perf_counter()-ts;ts=time.perf_counter();nominal=context['nominal_actions'];suffix=context['suffix_actions'];p=context['past_points'];bounds=context['workspace_bounds'];q=basis();gains,res=fit_response(p,context['past_targets'],cfg['physics_dt']);velocity=(p[-1]-p[-2])/cfg['physics_dt'];prefix=p[-75:];rawmap=scale[np.arange(25)%5,None]*q
    def response(a):return predict_response(np.r_[a,suffix],p[-1],velocity,gains,bounds,cfg['physics_dt'],cfg['substeps'])
    reference=response(nominal);j0=predicted_cost(reference,prefix,cfg['physics_dt'],0.);normalizer=max(j0,1e-12)
    def objective(z):return predicted_cost(response(nominal+(rawmap@z).reshape(5,5)),prefix,cfg['physics_dt'],0.)/normalizer
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],meta['episode'],meta['timestep']]));directions=[]
    for _ in range(2):v=rng.normal(size=q.shape[1]);directions.append(v/np.linalg.norm(v))
    contact=bool(context['root_gripper_contact']>.1 or np.any(np.r_[nominal[:,4],suffix[:,4]]>.1));entries=[('nominal',0.,0,np.zeros(q.shape[1]),nominal.astype(np.float32),True,0)];solvers=[];times['setup']=time.perf_counter()-ts;ts=time.perf_counter()
    for r in cfg['radii']:
        constraints=[LinearConstraint(rawmap,-1-nominal.reshape(-1),1-nominal.reshape(-1)),{'type':'ineq','fun':lambda z,r=r:r*r-z@z}];opt=minimize(objective,np.zeros(q.shape[1]),method='SLSQP',constraints=constraints,options={'maxiter':80,'ftol':1e-9});z=opt.x;a=nominal+(rawmap@z).reshape(5,5);ok=np.max(abs(a))<=1+1e-7 and np.linalg.norm(z)<=r+1e-6 and objective(z)<=1+1e-8;solvers.append([r,int(opt.success),int(opt.nit),float(opt.fun),int(ok)])
        if ok:entries.append(('past_pd_optimizer',r,0,z,a.astype(np.float32),bool(opt.success),int(opt.nit)))
    times['generation']=time.perf_counter()-ts
    def score(entries):
        a=np.array([x[4] for x in entries],dtype=np.float32);robot=np.array([response(x) for x in a]);cost=np.array([predicted_cost(x,prefix,.002,0.) for x in robot]);norm=np.linalg.norm(((a-nominal)/scale).reshape(len(a),-1),axis=1);d=robot-reference;dp=np.linalg.norm(d[:,:,:3],axis=-1);dr=abs(np.arctan2(np.sin(d[:,:,3]),np.cos(d[:,:,3])));scores=np.maximum((dp+np.sqrt(3)*.02*dr).max(1)/.002,dr.max(1)/np.deg2rad(2)) if contact else np.zeros(len(a));return a,robot,cost,norm,scores
    def order(entry):return (-1.,-1) if entry[0]=='nominal' else (entry[1],4 if entry[0]=='past_pd_optimizer' else 2*int(entry[0][-1])+(entry[2]>0))
    def pick(entries,cost,norm,scores):
        if not rule['enabled']:return 0
        eligible=np.flatnonzero((norm>acfg['minimum_norm'])&(norm<=radius+1e-6)&(scores<=rule['tau'])&((j0-cost)/normalizer>acfg['minimum_predicted_gain']))
        return int(min(eligible,key=lambda i:(cost[i],norm[i],order(entries[i])))) if len(eligible) else 0
    ts=time.perf_counter();a,robot,cost,norm,scores=score(entries);times['scoring']=time.perf_counter()-ts;ts=time.perf_counter();chosen=pick(entries,cost,norm,scores);gain=(cost[0]-cost[chosen])/max(cost[0],1e-12);upper=(cost[0]-cost.min())/max(cost[0],1e-12);early=bool(dynamic and chosen>0 and gain>=acfg['early_stop_gain'] and gain>=acfg['early_stop_fraction']*upper);times['selection']=time.perf_counter()-ts;scored=len(a);stages=1
    if mode!='cheap_only' and not early:
        ts=time.perf_counter();extra=[]
        for r in cfg['radii']:
            for k,v in enumerate(directions):
                for sign in [-1,1]:
                    z=sign*r*v;x=nominal+(rawmap@z).reshape(5,5)
                    if np.max(abs(x))<=1+1e-9:extra.append(('random'+str(k),r,sign,z,x.astype(np.float32),True,0))
        times['generation']+=time.perf_counter()-ts
        if extra:
            ts=time.perf_counter();aa,rr,cc,nn,ss=score(extra);times['scoring']+=time.perf_counter()-ts;entries+=extra;a=np.r_[a,aa];robot=np.r_[robot,rr];cost=np.r_[cost,cc];norm=np.r_[norm,nn];scores=np.r_[scores,ss];scored+=len(extra)
        stages=2
    ts=time.perf_counter();ids=sorted(range(len(entries)),key=lambda i:order(entries[i]));entries=[entries[i] for i in ids];a,robot,cost,norm,scores=[x[ids] for x in [a,robot,cost,norm,scores]];chosen=pick(entries,cost,norm,scores);times['selection']+=time.perf_counter()-ts
    bank={'actions':a,'predicted_robot':robot,'costs':cost,'norm':norm,'scores':scores,'labels':np.array([x[:3] for x in entries],dtype=str),'coefficients':np.array([x[3] for x in entries]),'solver_success':np.array([x[5] for x in entries]),'solver_iterations':np.array([x[6] for x in entries]),'optimizer_diagnostics':np.array(solvers),'basis':q}
    return bank,{'mode':mode,'selected':chosen,'alpha':float(alpha),'radius':float(radius),'past_error':float(residual) if allowance else None,'past_error_components':components.tolist() if allowance else None,'stages':stages,'early_stop':early,'scored_candidates':scored,'solver_calls':len(solvers),'solver_iterations':int(sum(x[2] for x in solvers)),'predicted_gain':float((cost[0]-cost[chosen])/max(cost[0],1e-12)),'seconds':time.perf_counter()-start,'timing':times}
