"""Aligned native LeWM and DINO features for repeated supervised probes."""
import json
from pathlib import Path
import numpy as np
from zj_wam.io import resolve,sha256
from scripts.retrained_dinowm_effect_readout_core import feature,pose_effect,true_pose,effect_metrics,wrong_indices
from scripts.retrained_dinowm_readout_core import predict
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def catalog():
    out=BASE/'probe_stability';done=read(out/'fit_complete.json');models={};entries={};paths=[]
    for name,h in done['record_hashes'].items():
        p=out/name;assert sha256(p)==h;r=read(p);paths.append(p);env=r['environment'];models.setdefault(env,{})
        for learner in r['learners']:
            replica=r['group']['name']+'--'+learner;key=r['representation']+'__'+r['target']+'__'+replica;path=p.parent/(r['representation']+'__'+learner+'.npz');assert sha256(path)==r['files'][path.name]
            with np.load(path,allow_pickle=False) as a:models[env][key]=dict(a)
            entries[env+'/'+key]={'environment':env,'representation':r['representation'],'target':r['target'],'replica':replica,'group':r['group'],'learner':learner,'path':str(path.relative_to(resolve('.'))),'sha256':sha256(path)};paths.append(path)
    assert sum(map(len,models.values()))==180
    return models,entries,paths

def features(rep,tokens,lewm,q):
    if rep.startswith('dino_'):
        return {'real':feature(tokens['actual'],tokens['history'],rep,q),'predicted':feature(tokens['predicted'],tokens['history'],rep,q),'current':feature(tokens['history'][-1:],tokens['history'],rep,q,current=True)}
    h=lewm['history'];real=lewm['actual'];pred=lewm['predicted']
    if rep=='lewm_frame':return {'real':real,'predicted':pred,'current':h[-1:]}
    assert rep=='lewm_history';n=len(real);prior=np.broadcast_to(h[-2:].reshape(1,-1),(n,h[-2:].size));return {'real':np.c_[prior,real],'predicted':np.c_[prior,pred],'current':h.reshape(1,-1)}

def decode_features(env,kind,model,x,parent,cfg,scale):
    real=predict(model,x['real']);imagined=predict(model,x['predicted']);current=predict(model,x['current']);sym=kind=='symmetry';truth=true_pose(env,parent);re=pose_effect(env,real,sym);pe=pose_effect(env,imagined,sym);n=len(real);ce=pose_effect(env,np.repeat(current,n,axis=0),sym);zero=(np.zeros_like(re[0]),np.zeros_like(re[1]),np.repeat(re[2][:1],n,axis=0),np.repeat(re[3][:1],n,axis=0));wrong,cos,ratio=wrong_indices(parent['actions'],scale,cfg);we=tuple(v[np.maximum(wrong,0)] for v in pe)
    methods={'real_encoded':effect_metrics(env,re,truth),'predicted':effect_metrics(env,pe,truth),'zero_effect':effect_metrics(env,zero,truth),'persistence':effect_metrics(env,ce,truth),'wrong_action':effect_metrics(env,we,truth)}
    for v in methods['wrong_action'].values():v[wrong<0]=np.nan
    return methods,{'decoded_real':real,'decoded_predicted':imagined,'decoded_current':current,'true_dp':truth[0],'true_dr':truth[1],'wrong_index':wrong,'wrong_cosine':cos,'wrong_norm_ratio':ratio}
