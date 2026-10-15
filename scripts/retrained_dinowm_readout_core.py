"""Array-only diagnostic probe fitting and matched frozen-feature loading."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from zj_wam.io import resolve
from zj_wam.cube_readouts import capture_errors,current_features,future_features,target,canonical_target
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def projection(cfg):
    q=np.linalg.qr(np.random.default_rng(cfg['projection']['seed']).normal(size=(384,192)))[0]
    np.testing.assert_allclose(q.T@q,np.eye(192),rtol=0,atol=1e-12)
    return q


def load(env,split,cfg,smoke=False):
    capture=BASE/('dinowm_capture_smoke' if smoke else 'dinowm_capture');m=read(capture/'manifest.json');paths=m['states'][env][split];values={};ys=[];families=[];keys=[]
    for name in paths:
        src=resolve(name);dest=capture/env/src.name if env=='pusht' else capture/env/split/src.name
        with np.load(dest.with_suffix('.npz')) as a:
            truth=a['targets'].copy();n=len(truth);ys.append(truth);families.extend(a['family'].tolist());keys.extend([src.stem+'_'+str(i) for i in range(n)])
            for rep in cfg['dino_representations']:values.setdefault('dino_'+rep,[]).append(a[rep].copy())
        with np.load(src.with_suffix('.npz')) as a:
            if env=='pusht':
                np.testing.assert_array_equal(truth,a['targets'])
                for native,key in [('projected_cls','frame'),('spatial_quadrants','spatial'),('history_projected','history')]:values.setdefault('lewm_'+key,[]).append(a[native].copy())
            else:
                history=a['history_embedding'];end=a['future_embedding'][None] if split=='probe_train' else a['actual_endpoint_latent'][:1]
                for rep in ['frame','history']:values.setdefault('lewm_'+rep,[]).append(np.stack([current_features(history,rep),future_features(history,end,rep)[0]]))
                if split=='probe_train':expected=np.stack([target(a[k+'_object_pose'][None],a[k+'_object_velocity'][None],a[k+'_eef_position'][None])[0] for k in ['current','future']])
                else:expected=np.stack([target(a['root_object_pose'][None],a['root_velocity'][None],a['root_eef_position'][None])[0],target(a['candidate_object_pose'][0,124][None],a['candidate_object_velocity'][0,124][None],a['candidate_eef_position'][0,124][None])[0]])
                np.testing.assert_array_equal(truth,expected)
    x={k:np.concatenate(v).astype(np.float64) for k,v in values.items()};q=projection(cfg)
    for rep in cfg['dino_representations']:
        v=x['dino_'+rep];x['dino_projected_'+rep]=(v.reshape(len(v),-1,384)@q).reshape(len(v),-1)
    return {'x':x,'y':np.concatenate(ys).astype(np.float64),'families':np.array(families),'keys':np.array(keys)}


def targets(env,rep,y,kind):
    if env=='pusht':return y[:,:9 if 'history' in rep else 6]
    return canonical_target(y) if kind=='symmetry' else y.copy()


def errors(env,pred,true):
    if env=='ogb-cube':
        e=capture_errors(pred,true);e['joint_pose']=(e['position_m']<=.002)&(e['rotation_rad']<=np.deg2rad(2));e['joint_pose_symmetry']=(e['position_m']<=.002)&(e['rotation_mod_cube_symmetry_rad']<=np.deg2rad(2));return e
    d=pred[:,:2]-true[:,:2];a=np.arctan2(pred[:,2],pred[:,3])-np.arctan2(true[:,2],true[:,3]);angle=np.abs(np.arctan2(np.sin(a),np.cos(a)));pos=np.linalg.norm(d,axis=1)
    return {'position_px':pos,'rotation_rad':angle,'pusher_px':np.linalg.norm(pred[:,4:6]-true[:,4:6],axis=1),'joint_pose':(pos<=2)&(angle<=np.deg2rad(2))}


def score(env,pred,true,kind):
    e=errors(env,pred,true);position=e['position_px']/2 if env=='pusht' else e['position_m']/.002
    angle=e['rotation_mod_cube_symmetry_rad'] if kind=='symmetry' else e['rotation_rad']
    return float(np.mean(position+angle/np.deg2rad(2)))


def predict(state,x):
    x=(np.asarray(x,dtype=np.float64)-state['x_mean'])/state['x_scale']
    if 'coef' in state:z=x@state['coef'].T+state['intercept']
    else:z=np.tanh(x@state['w0']+state['b0'])@state['w1']+state['b1']
    return z*state['y_scale']+state['y_mean']


def fit(x,y,kind,alpha,env,seed):
    xs=StandardScaler().fit(x);ys=StandardScaler().fit(y);xt=xs.transform(x);yt=ys.transform(y)
    if kind=='mlp':
        common=dict(hidden_layer_sizes=(64,),activation='tanh',solver='adam',alpha=.01,random_state=seed)
        estimator=MLPRegressor(**common,batch_size=128,max_iter=400,early_stopping=False) if env=='pusht' else MLPRegressor(**common,batch_size=64,max_iter=500,early_stopping=True,n_iter_no_change=30)
    else:estimator=Ridge(alpha=alpha)
    estimator.fit(xt,yt);state={'x_mean':xs.mean_,'x_scale':xs.scale_,'y_mean':ys.mean_,'y_scale':ys.scale_}
    if kind=='mlp':state.update(w0=estimator.coefs_[0],b0=estimator.intercepts_[0],w1=estimator.coefs_[1],b1=estimator.intercepts_[1])
    else:state.update(coef=estimator.coef_,intercept=estimator.intercept_)
    expected=ys.inverse_transform(estimator.predict(xt[:5]));actual=predict(state,x[:5]);np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
    return state,{'iterations':int(getattr(estimator,'n_iter_',0) or 0),'training_rows':len(x),'input_dimension':x.shape[1],'output_dimension':y.shape[1]}
