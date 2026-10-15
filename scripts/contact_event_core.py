"""Prespecified event windows, matched controls and frozen readout calibration."""
import json
from pathlib import Path
import numpy as np
from zj_wam.io import resolve,sha256
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def motion(parent,t):
    q=parent['states'][t:t+6,:2];return float(np.sqrt(np.mean(np.sum((q-q[:1])**2,axis=1))))

def windows(seed,parent,contacts,cfg):
    b=np.any(np.asarray(contacts).reshape(-1,10)>0,axis=1);events=[];missing=[];n=len(b);k=cfg['stable_controls_each_side'];offsets=cfg['window_offsets']
    for kind in cfg['event_types']:
        target=kind=='onset';eligible=[]
        for e in range(k,n-k+1):
            if not (np.all(b[e-k:e]==(not target)) and np.all(b[e:e+k]==target)):continue
            if any(e+offset<10 or e+offset+8>n for offset in offsets.values()):continue
            eligible.append(e)
        if not eligible:missing.append({'seed':seed,'event_type':kind,'reason':'No2controlstableevent with all3causalH5+3windows'});continue
        e=eligible[0];event_id=f'{seed}_{kind}_{e:03d}';cases=[{'role':role,'step':e+offset} for role,offset in offsets.items()];target_motion=motion(parent,e-2)
        for role in cfg['stable_controls']:
            contact=role=='stable_contact';choices=[]
            for t in range(10,n-8+1):
                if not np.all(b[t:t+8]==contact):continue
                if not (t+8<=e-5 or t>=e+9):continue
                m=motion(parent,t);choices.append((abs(np.log((m+1e-6)/(target_motion+1e-6))),abs(t-(e-2)),t,m))
            if choices:
                match=min(choices);cases.append({'role':role,'step':match[2],'motion_log_mismatch':float(match[0])})
            else:missing.append({'seed':seed,'event_type':kind,'role':role,'reason':'No disjoint8controlstable matching window'})
        for case in cases:
            t=case['step'];micro=(np.asarray(contacts)[10*t:10*(t+8)]>0).astype(int);case.update(seed=seed,event_id=event_id,event_type=kind,event_control=e,event_first_physics_seconds=(10*e+1)*.01,state_id=f'{seed}_t{t:03d}',nominal_motion_rms=motion(parent,t),control_contact_fraction=float(b[t:t+8].mean()),physics_contact_fraction=float(micro.mean()),physics_contact_edges=int(np.count_nonzero(np.diff(micro))));events.append(case)
    return events,missing

def resolution(cfg):
    folder=BASE/'output_replacement_lewm';done=read(folder/'pipeline_complete.json');audit=read(folder/'raw_audit.json');assert done['raw_audit_sha256']==sha256(folder/'raw_audit.json');paths=sorted((folder/'pusht/calibration').glob('*.json'));assert len(paths)==105;values={rep:{'position':[],'rotation':[],'weights':[]} for rep in cfg['representations']};inputs={}
    for p in paths:
        meta=read(p)
        for name,h in meta['files'].items():assert sha256(p.parent/name)==h,name
        src=resolve(meta['source']);arrays=dict(np.load(p.with_suffix('.npz')));raw=dict(np.load(src.with_suffix('.npz')));truth=raw['physics'][:,79,2:5];norm=np.linalg.norm((raw['actions']-raw['actions'][:1]).reshape(len(truth),-1),axis=1);ids=np.flatnonzero(norm>1e-12);assert len(ids)>0
        for rep in cfg['representations']:
            y=arrays[rep+'__raw__real'];dpos=(y[:,:2]-y[:1,:2])-(truth[:,:2]-truth[:1,:2]);theta=np.arctan2(y[:,2],y[:,3]);dr=(theta-theta[0])-(truth[:,2]-truth[0,2]);pe=np.linalg.norm(dpos,axis=1)[ids];re=abs(np.arctan2(np.sin(dr),np.cos(dr)))[ids];values[rep]['position'].extend(pe.tolist());values[rep]['rotation'].extend(re.tolist());values[rep]['weights'].extend([1/len(ids)]*len(ids))
        for q in [p,p.with_suffix('.npz'),src,src.with_suffix('.npz')]:inputs[str(q.relative_to(resolve('.')))]=sha256(q)
    thresholds={}
    for rep,v in values.items():
        w=np.array(v['weights']);thresholds[rep]={}
        for axis in ['position','rotation']:
            x=np.array(v[axis]);order=np.argsort(x,kind='stable');total=np.cumsum(w[order]);idx=np.flatnonzero(total>=cfg['resolution_quantile']*total[-1])[0];thresholds[rep][axis]=max(1e-9,float(x[order[idx]]))
    return {'states':105,'thresholds':thresholds,'inputs':inputs,'scope':cfg['resolution_source']}
