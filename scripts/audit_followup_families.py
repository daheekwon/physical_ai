"""Independent artifact checks for family/oracle claims."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.adapters import ExpertDataset
from zj_wam.io import resolve,write_json
out=resolve('artifacts/followup_v1');data=ExpertDataset('data/datasets/pusht_expert_train.h5');maxima={k:0. for k in ['object_error','jerk_gain','norm','basis_residual','terminal_position','terminal_velocity','basis_orthonormality']};count=0;states=0
for p in sorted((out/'family_ablation').glob('*h*.json')):
 with np.load(p.with_suffix('.npz')) as a:
  f=pd.read_csv(p.with_suffix('.csv'));phys=a['physics'];cov=a['coverage'];actions=a['actions'];h=actions.shape[1];idx=f['index'].to_numpy(int)
  dp=np.linalg.norm(phys[idx,30:,2:4]-phys[0:1,30:,2:4],axis=-1).max(1)/2
  da=phys[idx,30:,4]-phys[0:1,30:,4];dr=np.abs(np.arctan2(np.sin(da),np.cos(da))).max(1)/np.deg2rad(2)
  loss=np.maximum(0,cov[0:1]-cov[idx]).max(1)/.02;error=np.maximum.reduce([dp,dr,loss])
  jerk=np.sum(np.diff(phys[:,:,:2],n=3,axis=1)**2,axis=(1,2))/.01**5;gain=(jerk[0]-jerk[idx])/jerk[0]
  delta=((actions[idx]-actions[:1])/data.scaler.scale_).reshape(len(f),-1)
  maxima['object_error']=max(maxima['object_error'],float(np.max(np.abs(error-f.object_error))))
  maxima['jerk_gain']=max(maxima['jerk_gain'],float(np.max(np.abs(gain-f.jerk_gain))))
  maxima['norm']=max(maxima['norm'],float(np.max(np.abs(np.linalg.norm(delta,axis=1)-f.actual_norm))))
  for name,g in f.groupby('method'):
   basis=a['basis_'+name];positions=g.index.to_numpy();residual=delta[positions]-(delta[positions]@basis)@basis.T
   maxima['basis_residual']=max(maxima['basis_residual'],float(np.linalg.norm(residual,axis=1).max()))
   maxima['basis_orthonormality']=max(maxima['basis_orthonormality'],float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1])),initial=0)))
   if name.endswith('_terminal'):
    ids=g['index'].to_numpy(int);end=30+h*10-1
    maxima['terminal_position']=max(maxima['terminal_position'],float(np.linalg.norm(phys[ids,end,:2]-phys[0,end,:2],axis=-1).max()))
    maxima['terminal_velocity']=max(maxima['terminal_velocity'],float(np.linalg.norm(phys[ids,end,5:7]-phys[0,end,5:7],axis=-1).max()))
  count+=len(f);states+=1
assert states==420
assert maxima['object_error']<1e-10 and maxima['jerk_gain']<1e-10 and maxima['norm']<1e-10
assert maxima['basis_residual']<1e-8 and maxima['terminal_position']<1e-3 and maxima['terminal_velocity']<1e-3
write_json(out/'family_ablation/raw_audit.json',{'states':states,'candidates':count,'maximum_error':maxima});print(states,count,maxima)
