"""Supervised diagnostic readouts for frozen Cube latent features."""
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def target(pose,velocity,eef):
    p=np.asarray(pose);rotation=Rotation.from_quat(p[..., [4,5,6,3]]);r=rotation.as_matrix().reshape(*p.shape[:-1],9)
    velocity=np.asarray(velocity).copy();velocity[...,3:]=rotation.apply(velocity[...,3:])
    return np.concatenate([p[...,:3],r,velocity,np.asarray(eef)],axis=-1)


def rotations(y):
    m=np.asarray(y)[...,3:12].reshape(-1,3,3);u,_,vt=np.linalg.svd(m);sign=np.linalg.det(u@vt);u[:,:,-1]*=sign[:,None]
    return Rotation.from_matrix(u@vt)


def symmetry_rotation_error(pred,true):
    a,b=rotations(pred),rotations(true)
    return np.min(np.stack([(a*(b*s).inv()).magnitude() for s in Rotation.create_group('O')]),axis=0)


def capture_errors(pred,true):
    return {'position_m':np.linalg.norm(pred[:,:3]-true[:,:3],axis=1),'rotation_rad':(rotations(pred)*rotations(true).inv()).magnitude(),'rotation_mod_cube_symmetry_rad':symmetry_rotation_error(pred,true),'linear_velocity_m_s':np.linalg.norm(pred[:,12:15]-true[:,12:15],axis=1),'angular_velocity_rad_s':np.linalg.norm(pred[:,15:18]-true[:,15:18],axis=1),'eef_position_m':np.linalg.norm(pred[:,18:21]-true[:,18:21],axis=1)}


def fit(x,y,kind,alpha,seed):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    xs=StandardScaler().fit(x);ys=StandardScaler().fit(y)
    model=Ridge(alpha=alpha) if kind=='ridge' else MLPRegressor(hidden_layer_sizes=(64,),activation='tanh',solver='adam',alpha=.01,max_iter=500,early_stopping=True,random_state=seed,batch_size=64,n_iter_no_change=30)
    model.fit(xs.transform(x),ys.transform(y));return {'x_scaler':xs,'y_scaler':ys,'model':model,'kind':kind,'alpha':alpha,'training_rows':len(x)}


def predict(model,x):return model['y_scaler'].inverse_transform(model['model'].predict(model['x_scaler'].transform(np.asarray(x,dtype=np.float64))))


def effects(decoded,truth,align_cube_symmetry=False):
    """Both are candidate matrices with nominal first; SO(3) effects in world frame."""
    dp=decoded[:,:3]-decoded[:1,:3];tp=truth[:,:3]-truth[:1,:3]
    dr=rotations(decoded);tr=rotations(truth)
    if align_cube_symmetry:dr=align_rotations(dr,dr[0])
    dr=dr*dr[0].inv();tr=tr*tr[0].inv()
    return dp,tp,dr,tr


def align_rotations(rotations_in,reference):
    candidates=np.stack([(rotations_in*s).as_matrix() for s in Rotation.create_group('O')],axis=1)
    ref=reference.as_matrix();scores=np.einsum('...ij,ij->...',candidates,ref)
    chosen=[]
    for matrices,values in zip(candidates,scores):
        tied=np.flatnonzero(values>=values.max()-1e-10)
        index=min(tied,key=lambda i:tuple(np.round(matrices[i].reshape(-1),10)))
        chosen.append(matrices[index])
    return Rotation.from_matrix(np.asarray(chosen))


def canonical_target(y):
    result=np.asarray(y).copy();r=align_rotations(rotations(result),Rotation.identity());result[:,3:12]=r.as_matrix().reshape(-1,9);return result


def current_features(history,representation):
    h=np.asarray(history);assert h.shape==(3,192)
    return h[-1].copy() if representation=='frame' else h.reshape(-1).copy()


def future_features(history,endpoints,representation):
    h=np.asarray(history);e=np.asarray(endpoints);assert h.shape==(3,192) and e.ndim==2 and e.shape[1]==192
    if representation=='frame':return e.copy()
    return np.concatenate([np.broadcast_to(h[-2:],(len(e),2,192)),e[:,None]],axis=1).reshape(len(e),576)
