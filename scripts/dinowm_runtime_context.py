"""Load causal inputs independently from future outcome evaluation."""
import json
import numpy as np
import pandas as pd


def load_context(env,src,data,mean,scale):
    meta=json.loads(src.read_text());frame=pd.read_csv(src.with_suffix('.csv'))
    with np.load(src.with_suffix('.npz'),allow_pickle=False) as archive:a=dict(archive)
    c={'episode':meta['episode'],'timestep':meta['timestep'],'images':a['history_images'],'past':a['past_actions'],'scale':scale,'mean':mean}
    if env=='pusht':
        original,recorded=data.episode(meta['episode']);t=meta['timestep'];root=np.zeros(10);root[[0,1,5,6]]=a['root'][[0,1,5,6]];prefix=np.zeros_like(a['physics'][0,:30]);prefix[:,[0,1,5,6]]=a['physics'][0,:30,[0,1,5,6]].T;proprio=np.zeros((3,10));proprio[:,:7]=recorded[[t-10,t-5,t]];proprio[-1,:2]=root[:2];proprio[-1,5:7]=root[5:7];proprio[:,[2,3,4,7,8,9]]=0
        c.update(nominal=original[t:t+5],suffix=original[t+5:t+8],root=root,prefix=prefix,proprio=proprio)
        lookup={('nominal',0.,0):0}
        for r in frame.itertuples():
            name=r.method;sign=0
            if '_sign' in name:name,sg=name.split('_sign');sign=int(sg)
            lookup[(name,float(r.relative_radius),sign)]=int(r.index)
    else:
        c.update(nominal=a['nominal_actions'],suffix=a['suffix_actions'],points=a['past_points'],targets=a['past_targets'],bounds=a['workspace_bounds'],gripper_contact=float(a['root_gripper_contact']))
        lookup={(r.method,float(r.radius),int(r.sign)):int(r.candidate) for r in frame.itertuples()}
    return meta,c,lookup,a
