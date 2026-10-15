"""Frozen readout effects and physically defined controls for DINO token banks."""
import numpy as np
from scipy.spatial.transform import Rotation
from zj_wam.cube_readouts import rotations,align_rotations
from scripts.retrained_dinowm_readout_core import predict


def feature(tokens,history,rep,q,current=False):
    z=np.asarray(tokens);h=np.asarray(history);n=len(z)
    if 'spatial' in rep:
        g=z.reshape(n,14,14,384);x=np.concatenate([g[:,i:i+7,j:j+7].mean((1,2)) for i in [0,7] for j in [0,7]],axis=1)
    elif 'history' in rep:
        hm=h.mean(1);xm=z.mean(1)
        x=np.concatenate([np.broadcast_to(hm[:-1] if current else hm[-2:],(n,2,384)),xm[:,None]],axis=1).reshape(n,-1)
    else:x=z.mean(1)
    x=x.astype(np.float64)
    if 'projected' in rep:x=(x.reshape(n,-1,384)@q).reshape(n,-1)
    return x


def wrap(x):return np.arctan2(np.sin(x),np.cos(x))


def pose_effect(env,decoded,symmetry=False):
    pos=decoded[:,:2] if env=='pusht' else decoded[:,:3];dp=pos-pos[:1]
    if env=='pusht':
        angle=np.arctan2(decoded[:,2],decoded[:,3]);dr=wrap(angle-angle[0])[:,None]
        return dp,dr,pos,angle
    absolute=rotations(decoded)
    if symmetry:absolute=align_rotations(absolute,absolute[0])
    delta=absolute*absolute[0].inv()
    return dp,delta.as_rotvec(),pos,absolute.as_quat()


def true_pose(env,parent):
    if env=='pusht':
        p=parent['physics'][:,79,2:5];decoded=np.c_[p[:,:2],np.sin(p[:,2]),np.cos(p[:,2])];return pose_effect(env,decoded)
    p=parent['candidate_object_pose'][:,124];r=Rotation.from_quat(p[:,[4,5,6,3]]);decoded=np.c_[p[:,:3],r.as_matrix().reshape(-1,9)];return pose_effect(env,decoded)


def effect_metrics(env,pred,truth):
    dp,dr,absolute_pos,absolute_rot=pred;tp,tr,true_pos,true_rot=truth
    if env=='pusht':re=np.abs(wrap(dr[:,0]-tr[:,0]));ae=np.abs(wrap(absolute_rot-true_rot))
    else:
        re=(Rotation.from_rotvec(dr)*Rotation.from_rotvec(tr).inv()).magnitude();ae=(Rotation.from_quat(absolute_rot)*Rotation.from_quat(true_rot).inv()).magnitude()
    def cosine(x,y):
        den=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1);return np.divide(np.sum(x*y,axis=1),den,out=np.full(len(x),np.nan),where=den>1e-12)
    return {'position_effect_error':np.linalg.norm(dp-tp,axis=1),'rotation_effect_error':re,'position_absolute_error':np.linalg.norm(absolute_pos-true_pos,axis=1),'rotation_absolute_error':ae,'position_cosine':cosine(dp,tp),'rotation_cosine':cosine(dr,tr),'predicted_position_magnitude':np.linalg.norm(dp,axis=1),'predicted_rotation_magnitude':np.linalg.norm(dr,axis=1)}


def wrong_indices(actions,scale,cfg):
    d=((np.asarray(actions)-np.asarray(actions)[:1])/np.asarray(scale)).reshape(len(actions),-1);norm=np.linalg.norm(d,axis=1);index=np.full(len(d),-1,dtype=int);cosine=np.full(len(d),np.nan);ratio=np.full(len(d),np.nan)
    for i in range(1,len(d)):
        if norm[i]<=1e-6:continue
        ratio_i=norm/norm[i];den=norm*norm[i];cos=np.divide(d@d[i],den,out=np.ones(len(d)),where=den>1e-12)
        valid=(norm>1e-6)&(np.arange(len(d))!=i)&(ratio_i>=cfg['wrong_action_norm_ratio'][0])&(ratio_i<=cfg['wrong_action_norm_ratio'][1])&(cos<=cfg['wrong_action_max_cosine'])
        ids=np.flatnonzero(valid)
        if len(ids):j=ids[np.argmin(cos[ids])];index[i]=j;cosine[i]=cos[j];ratio[i]=ratio_i[j]
    return index,cosine,ratio


def decode_all(env,rep,kind,model,arrays,q,parent,cfg,scale):
    real=predict(model,feature(arrays['actual'],arrays['history'],rep,q));imagined=predict(model,feature(arrays['predicted'],arrays['history'],rep,q));current=predict(model,feature(arrays['history'][-1:],arrays['history'],rep,q,current=True));sym=kind=='symmetry'
    truth=true_pose(env,parent);re=pose_effect(env,real,sym);pe=pose_effect(env,imagined,sym);n=len(real);ce=pose_effect(env,np.repeat(current,n,axis=0),sym);zero=(np.zeros_like(re[0]),np.zeros_like(re[1]),np.repeat(re[2][:1],n,axis=0),np.repeat(re[3][:1],n,axis=0))
    wrong,cos,ratio=wrong_indices(parent['actions'],scale,cfg);safe=np.maximum(wrong,0);we=tuple(x[safe] for x in pe)
    methods={'real_encoded':effect_metrics(env,re,truth),'predicted':effect_metrics(env,pe,truth),'zero_effect':effect_metrics(env,zero,truth),'persistence':effect_metrics(env,ce,truth),'wrong_action':effect_metrics(env,we,truth)}
    for values in methods['wrong_action'].values():values[wrong<0]=np.nan
    return methods,{'decoded_real':real,'decoded_predicted':imagined,'decoded_current':current,'true_dp':truth[0],'true_dr':truth[1],'wrong_index':wrong,'wrong_cosine':cos,'wrong_norm_ratio':ratio}
