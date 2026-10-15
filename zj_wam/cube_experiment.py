"""Cube polynomial candidates and past-only Cartesian response surrogate."""
import numpy as np
from scipy.optimize import lsq_linear, minimize, LinearConstraint
from scipy.spatial.transform import Rotation
from zj_wam.cube_execution import quat_error
from zj_wam.diagnostics import jerk_energy


def basis(h=5):
    t=np.linspace(0,1,h);e=t*t*(1-t)**2
    raw=np.kron(np.column_stack([e,e*(2*t-1)]),np.eye(4))
    q,s,_=np.linalg.svd(raw,full_matrices=False);q=q[:,s>s.max()*1e-10]
    out=np.zeros((h,5,q.shape[1]));out[:,:4]=q.reshape(h,4,-1)
    return out.reshape(h*5,-1)


def yaw(quaternions):
    q=np.asarray(quaternions)
    # wxyz, yaw from matrix first column; no quaternion sign dependence.
    return np.arctan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2]**2+q[...,3]**2))


def xyz_yaw(physics):
    return np.column_stack([physics['eef_position'],np.unwrap(yaw(physics['eef_quaternion']))])


def fit_response(points,targets,dt):
    """Effective diagonal PD from past measured EEF and commanded target only."""
    p=np.asarray(points);v=np.diff(p,axis=0)/dt;acc=np.diff(v,axis=0)/dt;target=np.asarray(targets)[1:]
    gain=[];residual=[]
    for k in range(4):
        error=target[:,k]-p[1:-1,k]
        if k==3:error=np.arctan2(np.sin(error),np.cos(error))
        x=np.column_stack([error,-v[:-1,k]])
        # Fixed weak Tikhonov prior stabilizes unexcited axes, no future fitting.
        prior=np.array([100.,20.]);ridge=1e-6
        sol=lsq_linear(np.r_[x,np.sqrt(ridge)*np.eye(2)],np.r_[acc[:,k],np.sqrt(ridge)*prior],bounds=([0.,0.],[10000.,300.]))
        gain.append(sol.x);residual.append(np.sqrt(np.mean((x@sol.x-acc[:,k])**2)))
    return np.asarray(gain),np.asarray(residual)


def predict_response(actions,root,velocity,gains,bounds,dt=.002,substeps=25):
    p=np.asarray(root).copy();v=np.asarray(velocity).copy();result=[]
    for a in actions:
        target=p+np.asarray(a[:4])*np.array([.05,.05,.05,.3]);target[:3]=np.clip(target[:3],bounds[0],bounds[1])
        for _ in range(substeps):
            error=target-p;error[3]=np.arctan2(np.sin(error[3]),np.cos(error[3]));v+=dt*(gains[:,0]*error-gains[:,1]*v);p+=dt*v;result.append(p.copy())
    return np.asarray(result)


def predicted_cost(trace,prefix,dt,rotation_scale):
    allp=np.concatenate([prefix,trace]);weighted=allp.copy();weighted[:,3]*=rotation_scale
    return jerk_energy(weighted,dt)


def generate_candidates(nominal,suffix,scale,q,predict_latent,x0,prefix,root,velocity,gains,bounds,cfg,rng):
    import torch,time
    started=time.perf_counter();rawmap=scale[np.arange(25)%5,None]*q
    def response(a):return predict_response(np.concatenate([a,suffix]),root,velocity,gains,bounds,cfg['physics_dt'],cfg['substeps'])
    reference=response(nominal);j0=predicted_cost(reference,prefix,cfg['physics_dt'],cfg['yaw_length_scale_m']);normalizer=max(j0,1e-12)
    def cost(z):return predicted_cost(response(nominal+(rawmap@z).reshape(5,5)),prefix,cfg['physics_dt'],cfg['yaw_length_scale_m'])/normalizer
    tx=torch.tensor(x0,dtype=torch.float32,device=cfg['device'])
    latent0=predict_latent(tx).detach();jac=torch.autograd.functional.jacobian(predict_latent,tx,vectorize=True).detach().cpu().numpy().astype(float)
    _,s,vt=np.linalg.svd(jac@q,full_matrices=False)
    dirs={'lewm_low':vt[-1],'lewm_high':vt[0]}
    for k in range(2):
        v=rng.normal(size=q.shape[1]);dirs['random'+str(k)]=v/np.linalg.norm(v)
    actions=[nominal.copy()];metadata=[{'method':'nominal','radius':0.,'sign':0,'solver_success':True}]
    coefficients=[np.zeros(q.shape[1])];rejected=[]
    for radius in cfg['radii']:
        for name,d in dirs.items():
            for sign in [-1,1]:
                z=sign*radius*d;a=nominal+(rawmap@z).reshape(5,5)
                if np.max(np.abs(a))<=1+1e-9:actions.append(a);coefficients.append(z);metadata.append({'method':name,'radius':radius,'sign':sign,'solver_success':True})
                else:rejected.append({'method':name,'radius':radius,'sign':sign,'reason':'action_bound','raw_max_abs':float(np.abs(a).max())})
        constraints=[LinearConstraint(rawmap,-1-nominal.reshape(-1),1-nominal.reshape(-1)),{'type':'ineq','fun':lambda z,r=radius:r*r-z@z}]
        opt=minimize(cost,np.zeros(q.shape[1]),method='SLSQP',constraints=constraints,options={'maxiter':80,'ftol':1e-9})
        z=opt.x;a=nominal+(rawmap@z).reshape(5,5)
        if np.max(np.abs(a))<=1+1e-7 and np.linalg.norm(z)<=radius+1e-6 and cost(z)<=1+1e-8:
            actions.append(a);coefficients.append(z);metadata.append({'method':'past_pd_optimizer','radius':radius,'sign':0,'solver_success':bool(opt.success)})
        else:rejected.append({'method':'past_pd_optimizer','radius':radius,'sign':0,'reason':'infeasible_or_worse_than_nominal','raw_max_abs':float(np.abs(a).max()),'norm':float(np.linalg.norm(z))})
    actions=np.asarray(actions);predicted=np.asarray([response(a) for a in actions]);costs=np.asarray([predicted_cost(p,prefix,cfg['physics_dt'],cfg['yaw_length_scale_m']) for p in predicted]);latents=[]
    with torch.no_grad():
        for a in actions:
            txa=torch.tensor(x0+((a-nominal)/scale).reshape(-1),dtype=torch.float32,device=cfg['device']);latents.append(predict_latent(txa).cpu().numpy())
    latents=np.asarray(latents);scores=np.linalg.norm(latents-latents[0],axis=1)
    return {'actions':actions,'coefficients':np.asarray(coefficients),'metadata':metadata,'predicted':predicted,'costs':costs,'latent':latents,'lewm_scores':scores,'jacobian':jac,'singular_values':s,'seconds':time.perf_counter()-started,'rejected':rejected}


def angular_jerk(quats,dt):
    q=np.asarray(quats);rot=Rotation.from_quat(q[:,[1,2,3,0]])
    # World-frame incremental angular velocity; derivative frame fixed across time.
    velocity=(rot[1:]*rot[:-1].inv()).as_rotvec()/dt
    return float(np.sum((np.diff(velocity,n=2,axis=0)/dt**2)**2)*dt)


def metrics(actual,nominal,prefix,cfg):
    pos=np.linalg.norm(actual['object_pose'][:,:3]-nominal['object_pose'][:,:3],axis=1)
    angle=quat_error(actual['object_pose'][:,3:],nominal['object_pose'][:,3:]);h=cfg['horizon']*cfg['substeps']
    j0=jerk_energy(np.r_[prefix['eef_position'],nominal['eef_position']],cfg['physics_dt']);j=jerk_energy(np.r_[prefix['eef_position'],actual['eef_position']],cfg['physics_dt'])
    angular0=angular_jerk(np.r_[prefix['eef_quaternion'],nominal['eef_quaternion']],cfg['physics_dt']);angular=angular_jerk(np.r_[prefix['eef_quaternion'],actual['eef_quaternion']],cfg['physics_dt'])
    joint0=jerk_energy(np.r_[prefix['qpos'][:,:6],nominal['qpos'][:,:6]],cfg['physics_dt']);joint=jerk_energy(np.r_[prefix['qpos'][:,:6],actual['qpos'][:,:6]],cfg['physics_dt'])
    return {'object_error':max(float(pos.max()/cfg['position_tolerance_m']),float(angle.max()/cfg['rotation_tolerance_rad'])),'object_position_max_m':float(pos.max()),'object_rotation_max_rad':float(angle.max()),'endpoint_position_m':float(pos[h-1]),'endpoint_rotation_rad':float(angle[h-1]),'eef_jerk':j,'nominal_eef_jerk':j0,'jerk_gain':float((j0-j)/j0) if j0>1e-12 else 0.,'angular_jerk':angular,'nominal_angular_jerk':angular0,'joint_jerk':joint,'nominal_joint_jerk':joint0,'contact_mismatch_fraction':float(np.mean((actual['robot_object_contacts']>0)!=(nominal['robot_object_contacts']>0))),'drop_relative_to_nominal':bool(np.any((nominal['object_pose'][:,2]>.05)&(actual['object_pose'][:,2]<.025))),'terminal_velocity_difference':float(np.linalg.norm(actual['object_velocity'][-1,:3]-nominal['object_velocity'][-1,:3]))}
