"""Fixed-subspace utility diagnosis; only current/past robot inputs and frozen J."""
import numpy as np
from zj_wam.followup import controller_design,optimize_controller

def generate(nominal,suffix,root,prefix,scale,jac,cfg,episode,timestep):
    q,base,response,j0,design=controller_design(nominal,suffix,root,prefix,scale);cost0=float(j0@j0*.01);gradient=2*design.T@j0*.01;relative=gradient/max(cost0,1.);norm=np.linalg.norm(relative);valid=bool(norm>cfg['gradient_relative_floor']);k=cfg['dimension'];d=q.shape[1];robot=response[:len(nominal)*10].reshape(-1,d)
    ce,cv=np.linalg.eigh(robot.T@robot);wm=jac.astype(float)@q;we,wv=np.linalg.eigh(wm.T@wm);rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],episode,timestep]));random=np.linalg.qr(rng.normal(size=(d,k)))[0]
    if valid:
        columns=[relative/norm]
        for v in wv.T:
            r=v-sum(np.dot(u,v)*u for u in columns)
            if np.linalg.norm(r)>1e-10:columns.append(r/np.linalg.norm(r))
            if len(columns)==k:break
        augmented=np.column_stack(columns)
    else:augmented=wv[:,:k].copy()
    bases={'controller_low':cv[:,:k],'lewm_low':wv[:,:k],'random':random,'lewm_plus_gradient':augmented,'full':np.eye(d)};directions={};etas={}
    for name,b in bases.items():
        np.testing.assert_allclose(b.T@b,np.eye(b.shape[1]),atol=1e-12);v=-(b@(b.T@relative));directions[name]=v/np.linalg.norm(v) if np.linalg.norm(v)>1e-14 else b[:,0];etas[name]=float(np.linalg.norm(b.T@relative)**2/norm**2) if valid else None
    bound=np.inf
    for v in directions.values():
        raw=(q@v).reshape(len(nominal),2)*scale;take=abs(raw)>1e-14
        if take.any():bound=min(bound,float(np.min((1-abs(nominal.astype(float)[take]))/abs(raw[take]))))
    bank=[];records=[];coeff=[]
    for rr in cfg['relative_radii']:
        radius=max(0.,min(rr*np.sqrt(nominal.size),.999*bound))
        for name,b in bases.items():
            for sign in [-1,1]:
                z=sign*radius*directions[name];raw=nominal+(q@z).reshape(nominal.shape)*scale;bank.append(raw);coeff.append(z);records.append({'method':name,'kind':'signed','sign':sign,'relative_radius':rr,'radius':radius,'solver_success':True,'solver_iterations':0})
            raw,info=optimize_controller(nominal,suffix,root,prefix,scale,radius,basis=q@b);z=b@info['coefficient'];bank.append(raw);coeff.append(z);records.append({'method':name,'kind':'optimizer','sign':0,'relative_radius':rr,'radius':radius,'solver_success':info['solver_success'],'solver_iterations':info['iterations']})
    coefficients=np.array(coeff);pred=base[None]+np.einsum('tcd,nd->ntc',response,coefficients);costs=np.sum((j0[None]+coefficients@design.T)**2,axis=1)*.01;scores=np.linalg.norm(pred-base[None],axis=-1).max(1)
    return {'actions':bank,'records':records,'coefficients':coefficients,'bases':bases,'directions':directions,'eta':etas,'gradient_valid':valid,'gradient':gradient,'relative_gradient':relative,'basis':q,'base':base,'response':response,'design':design,'j0':j0,'cost0':cost0,'costs':costs,'scores':scores,'predicted':pred,'controller_eigenvalues':ce,'lewm_eigenvalues':we,'common_bound':bound}
