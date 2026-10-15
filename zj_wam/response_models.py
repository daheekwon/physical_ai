"""Simple PushT object-response models with explicit observation and time boundaries."""
import numpy as np


RECTS=np.array([[-60.,60.,0.,30.],[-15.,15.,30.,120.]])


def rotate(points,angle):
    c=np.cos(angle);s=np.sin(angle)
    return np.stack([c*points[...,0]-s*points[...,1],s*points[...,0]+c*points[...,1]],axis=-1)


def point_geometry(points,pose):
    """Signed circle-center distance and outward normal for the known T shape."""
    local=rotate(points-pose[...,:2],-pose[...,2])
    distances=[];normals=[]
    for x0,x1,y0,y1 in RECTS:
        closest=np.stack([np.clip(local[...,0],x0,x1),np.clip(local[...,1],y0,y1)],-1)
        delta=local-closest;length=np.linalg.norm(delta,axis=-1)
        inside=(local[...,0]>=x0)&(local[...,0]<=x1)&(local[...,1]>=y0)&(local[...,1]<=y1)
        edges=np.stack([local[...,0]-x0,x1-local[...,0],local[...,1]-y0,y1-local[...,1]],-1)
        edge=np.argmin(edges,axis=-1);face=np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]])[edge]
        normal=np.where(inside[...,None],face,delta/np.maximum(length[...,None],1e-12))
        distances.append(np.where(inside,-np.min(edges,axis=-1),length));normals.append(normal)
    ds=np.stack(distances,-1);which=np.argmin(ds,axis=-1)
    ns=np.stack(normals,-2);normal=np.take_along_axis(ns,which[...,None,None],axis=-2)[...,0,:]
    return np.min(ds,axis=-1),rotate(normal,pose[...,2])


def contact_increment(pose,pusher,pusher_delta,cfg):
    distance,outward=point_geometry(pusher,pose)
    penetration=np.maximum(0,cfg['pusher_radius']-distance)
    inward=-outward
    point=pusher-outward*distance[...,None]
    cog=pose[...,:2]+rotate(np.broadcast_to(cfg['object_cog'],pose[...,:2].shape),pose[...,2])
    lever=point-cog;cross=lever[...,0]*inward[...,1]-lever[...,1]*inward[...,0]
    normal_motion=penetration/(1+cross**2/cfg['gyration_squared'])
    dc=inward*normal_motion[...,None];da=cross/cfg['gyration_squared']*normal_motion
    tangent=pusher_delta-np.sum(pusher_delta*inward,axis=-1)[...,None]*inward
    tnorm=np.linalg.norm(tangent,axis=-1)
    amount=np.minimum(cfg['tangential_gain']*tnorm,cfg['friction_cap']*penetration)
    friction=tangent/np.maximum(tnorm[...,None],1e-12)*amount[...,None]
    dc+=friction
    da+=(lever[...,0]*friction[...,1]-lever[...,1]*friction[...,0])/cfg['gyration_squared']
    return dc,da,penetration>0


def controller_batch(actions,root,kp=100.,kv=20.,dt=.01,substeps=10):
    q=np.broadcast_to(root[:2],(len(actions),2)).copy().astype(float)
    v=np.broadcast_to(root[5:7],q.shape).copy().astype(float);trace=[]
    for i in range(actions.shape[1]):
        target=q+100.*actions[:,i]
        for _ in range(substeps):
            v+=(kp*(target-q)-kv*v)*dt;q+=v*dt;trace.append(q.copy())
    return np.stack(trace,axis=1)


def contact_rollout(pusher,root,cfg,gains=(1.,1.)):
    pose=np.broadcast_to(root[2:5],(len(pusher),3)).copy().astype(float)
    previous=np.broadcast_to(root[:2],(len(pusher),2));trace=[];contacts=[]
    coglocal=np.broadcast_to(cfg['object_cog'],pose[:,:2].shape)
    for t in range(pusher.shape[1]):
        dc,da,contact=contact_increment(pose,pusher[:,t],pusher[:,t]-previous,cfg)
        center=pose[:,:2]+rotate(coglocal,pose[:,2])
        pose[:,2]+=gains[1]*da;center+=gains[0]*dc
        pose[:,:2]=center-rotate(coglocal,pose[:,2]);previous=pusher[:,t]
        trace.append(pose.copy());contacts.append(contact)
    return np.stack(trace,1),np.stack(contacts,1)


def fit_past_models(history,cfg):
    """history ends at current time; consumes no candidate or future execution."""
    pose=history[:,2:5];cog=pose[:,:2]+rotate(np.broadcast_to(cfg['object_cog'],pose[:,:2].shape),pose[:,2])
    dx=np.diff(history[:,:2],axis=0);dy=np.diff(cog,axis=0)
    da=np.arctan2(np.sin(np.diff(pose[:,2])),np.cos(np.diff(pose[:,2])))
    rg=np.sqrt(cfg['gyration_squared']);target=np.column_stack([dy,da*rg])
    gram=dx.T@dx;ridge=max(1e-6,cfg['local_ridge_fraction']*np.trace(gram));inverse=np.linalg.inv(gram+ridge*np.eye(2))
    mapping=inverse@dx.T@target
    residual=target-dx@mapping
    noise=max(cfg['local_noise_floor_px'],float(np.sqrt(np.mean(residual**2))))
    eigen=np.linalg.eigvalsh(gram)
    excitation=float(eigen[0]/max(eigen[-1],1e-9))
    pred_dc,pred_da,contact=contact_increment(pose[:-1],history[1:,:2],dx,cfg)
    alpha=[]
    for prediction,truth in [(pred_dc,dy),(pred_da*rg,da*rg)]:
        strength=float(np.sum(prediction**2));prior=cfg['adaptive_ridge']
        gain=(float(np.sum(prediction*truth))+prior)/(strength+prior)
        alpha.append(float(np.clip(gain,*cfg['adaptive_gain_bounds'])))
    return {'mapping':mapping,'inverse':inverse,'noise':noise,'excitation':excitation,'gains':alpha,
            'observed_intervals':len(dx),'observed_contact_intervals':int(contact.sum()),'ridge':ridge}


def response_score(poses):
    delta=poses-poses[0:1]
    p=np.linalg.norm(delta[...,:2],axis=-1).max(1)/2.
    r=np.abs(np.arctan2(np.sin(delta[...,2]),np.cos(delta[...,2]))).max(1)/np.deg2rad(2)
    return np.maximum(p,r)


def score_models(actions,suffix,root,history,cfg):
    future=np.concatenate([actions,np.broadcast_to(suffix,(len(actions),*suffix.shape))],axis=1)
    pusher=controller_batch(future,root)
    fixed,contacts=contact_rollout(pusher,root,cfg)
    past=fit_past_models(history,cfg);adaptive,_=contact_rollout(pusher,root,cfg,past['gains'])
    dq=pusher-root[:2];change=dq@past['mapping'];rg=np.sqrt(cfg['gyration_squared'])
    local=np.broadcast_to(root[2:5],(*pusher.shape[:2],3)).copy()
    local[:,:,2]+=change[:,:,2]/rg
    center=root[2:4]+rotate(np.array(cfg['object_cog']),root[4])+change[:,:,:2]
    local[:,:,:2]=center-rotate(np.broadcast_to(cfg['object_cog'],pusher.shape),local[:,:,2])
    correction=pusher-pusher[0:1]
    uncertainty=past['noise']*np.sqrt(np.maximum(0,np.einsum('nti,ij,ntj->nt',correction,past['inverse'],correction))).max(1)/2
    scores={'pusher':np.linalg.norm(correction,axis=-1).max(1),
            'contact':response_score(fixed),'local':response_score(local)+uncertainty,
            'adaptive':response_score(adaptive)}
    # An unexcited local model must not infer safety from zero fitted response.
    scores['local']=np.maximum(scores['local'],(1-min(1.,past['excitation']/.1))*scores['contact'])
    return scores,{'fixed':fixed,'adaptive':adaptive,'local':local,'pusher':pusher,'contacts':contacts,'past':past}


def estimate_pose(image,cfg):
    """Known-color, known-shape visual moment estimator; no simulator state input."""
    rgb=np.asarray(image,dtype=float);colors=np.asarray(cfg['vision_colors'])
    distance=np.min(np.linalg.norm(rgb[:,:,None,:]-colors,axis=-1),axis=-1)
    mask=distance<=cfg['vision_color_tolerance'];yy,xx=np.nonzero(mask)
    if len(xx)<cfg['vision_minimum_pixels']:
        return np.array([0.,0.,0.]),{'valid':False,'pixels':len(xx),'area_ratio':0.}
    points=np.column_stack([(xx+.5)*512/image.shape[1]-.5,(yy+.5)*512/image.shape[0]-.5])
    center=points.mean(0);relative=points-center;cov=relative.T@relative/len(points)
    _,vectors=np.linalg.eigh(cov);axis=vectors[:,-1];projection=relative@axis
    if np.mean(projection**3)<0:axis=-axis
    angle=float(np.arctan2(axis[1],axis[0])-np.pi/2)
    origin=center-rotate(np.asarray(cfg['vision_centroid_local']),angle)
    return np.r_[origin,angle],{'valid':True,'pixels':len(xx),'area_ratio':float(len(xx)*(512/image.shape[1])**2/cfg['vision_expected_area'])}
