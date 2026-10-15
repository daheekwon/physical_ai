"""Recompute F2 preservation/jerk and retain contact timing from saved dense traces."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json

out=resolve('artifacts/followup_v1');cfg=json.loads(resolve('configs/followup_v1.json').read_text());scale=np.array(json.loads((out/'protocol/manifest.json').read_text())['normalization_scale'])
counts={};rows=[];maxerrors={'object_error':0.,'jerk':0.,'action_norm':0.};states=0
for p in sorted((out/'interventions').glob('*_h05.json')):
    if not p.stem.startswith(('development_','calibration_')):continue
    f=pd.read_csv(p.with_suffix('.csv'))
    with np.load(p.with_suffix('.npz')) as a:
        physical=a['physics'];control=a['control'];cover=a['coverage'];actions=a['actions'];contacts=a['contacts']
    pos=np.linalg.norm(physical[:,30:,2:4]-physical[0:1,30:,2:4],axis=-1).max(1)
    delta=physical[:,30:,4]-physical[0:1,30:,4];rot=np.abs(np.arctan2(np.sin(delta),np.cos(delta))).max(1)
    loss=np.maximum(0,cover[0:1]-cover).max(1)
    errors=np.maximum.reduce([pos/2,rot/np.deg2rad(2),loss/.02])
    jerk=(np.diff(physical[:,:,:2],n=3,axis=1)**2).sum((1,2))/.01**5
    norms=np.linalg.norm(((actions-actions[0:1])/scale).reshape(len(actions),-1),axis=1)
    for label,actual,stored in [('object_error',errors[1:],f.object_error),('jerk',jerk[1:],f.physics_jerk),('action_norm',norms[1:],f.actual_norm)]:
        np.testing.assert_allclose(actual,stored,rtol=1e-9,atol=1e-7)
        maxerrors[label]=max(maxerrors[label],float(np.max(np.abs(actual-stored))))
    assert np.array_equal(errors[1:]<=1,f.preserved.to_numpy())
    horizon=int(f.horizon.iloc[0]);end=3+horizon-1
    finalpos=np.linalg.norm(control[:,end,2:4]-control[0:1,end,2:4],axis=-1)
    dr=control[:,end,4]-control[0,end,4];finalrot=np.abs(np.arctan2(np.sin(dr),np.cos(dr)))
    finalok=(finalpos<=2)&(finalrot<=np.deg2rad(2))&(np.maximum(0,cover[0,0]-cover[:,0])<=.02)
    nominal=contacts[0,:horizon*10]>0;non=np.flatnonzero(nominal)
    for i in range(1,len(actions)):
        c=contacts[i,:horizon*10]>0;on=np.flatnonzero(c)
        rows.append({'state_id':f.state_id.iloc[0],'split':f.split.iloc[0],'initial_pose_family':int(f.initial_pose_family.iloc[0]),'index':i,
                     'first_contact_seconds':float((on[0]+1)*.01) if len(on) else -1.,
                     'nominal_first_contact_seconds':float((non[0]+1)*.01) if len(non) else -1.,
                     'contact_duration_seconds':float(c.sum()*.01),'contact_transitions':int(np.sum(c[1:]!=c[:-1])),
                     'contact_changed_substeps':int(np.sum(c!=nominal)),
                     'final_preserved':bool(finalok[i]),'dense_preserved':bool(errors[i]<=1),
                     'final_safe_dense_unsafe':bool(finalok[i] and errors[i]>1)})
    split=f.split.iloc[0];counts[split]=counts.get(split,0)+len(f);states+=1
pd.DataFrame(rows).to_csv(out/'interventions/contact_and_endpoint_audit.csv',index=False)
write_json(out/'interventions/raw_audit.json',{'states':states,'candidates':counts,'max_recomputed_difference':maxerrors,
           'final_safe_dense_unsafe':sum(r['final_safe_dense_unsafe'] for r in rows),
           'zero_contact_is_explicit':True,'physics_dt':.01,'suffix_actions':3})
print(json.dumps({'states':states,'candidates':counts,'final_safe_dense_unsafe':sum(r['final_safe_dense_unsafe'] for r in rows)}))
