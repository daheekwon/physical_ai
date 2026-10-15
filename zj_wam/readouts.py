"""Physical readout utilities with periodic orientation and fixed training scalers."""
import numpy as np


def decode(readout,features):
    flat=np.asarray(features).reshape(-1,features.shape[-1])
    raw=readout['output_scaler'].inverse_transform(readout['estimator'].predict(readout['input_scaler'].transform(flat)))
    pose=np.column_stack([raw[:,:2],np.arctan2(raw[:,2],raw[:,3])])
    return pose.reshape(*features.shape[:-1],3)


def effects(pose):
    delta=np.asarray(pose)-pose[0:1]
    delta=delta.copy();delta[...,2]=np.arctan2(np.sin(delta[...,2]),np.cos(delta[...,2]))
    return delta


def errors(predicted,true):
    e=np.asarray(predicted)-true
    return np.linalg.norm(e[...,:2],axis=-1),np.abs(np.arctan2(np.sin(e[...,2]),np.cos(e[...,2])))


def direction_cosine(predicted,true):
    p=np.asarray(predicted)[...,:2];t=np.asarray(true)[...,:2]
    return np.sum(p*t,axis=-1)/np.maximum(np.linalg.norm(p,axis=-1)*np.linalg.norm(t,axis=-1),1e-12)


def history_features(context, future):
    """Native current context plus each real/predicted block, in matching physical order."""
    n,blocks,dim=future.shape
    context=np.broadcast_to(np.asarray(context).reshape(1,3,dim),(n,3,dim))
    full=np.concatenate([context,future],axis=1)
    return np.stack([full[:,i+1:i+4].reshape(n,3*dim) for i in range(blocks)],axis=1)
