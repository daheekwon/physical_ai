"""Independent saved-feature/pose/physical-label reconstruction of nuisance contrasts."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
out=resolve('artifacts/followup_v1/visual_nuisance');cfg=json.loads(resolve('configs/followup_visual_nuisance.json').read_text());maxima={};count=0
for path in sorted(out.glob('*_f*_e*_t*_h05.json')):
    m=json.loads(path.read_text());assert sha256(Path(m['source']))==m['source_sha256'];assert sha256(Path(m['source']).with_suffix('.npz'))==m['source_npz_sha256']
    frame=pd.read_csv(path.with_suffix('.csv'))
    with np.load(path.with_suffix('.npz')) as a:
        lookup={int(x):i for i,x in enumerate(a['indices'])};true=a['truth']
        for r in frame.itertuples(index=False):
            j=cfg['appearances'].index(r.appearance);i=lookup[r.index];z=a['features_'+r.representation];p=a[f'pose_{r.representation}_{r.readout}']
            if r.contrast=='appearance_only_nominal':dz=z[j,0]-z[0,0];dp=p[j,0]-p[0,0];dt=np.zeros(3)
            elif r.contrast=='physical_pair':dz=z[j,i]-z[j,0];dp=p[j,i]-p[j,0];dt=true[i]-true[0]
            else:assert r.contrast=='combined';dz=z[j,i]-z[0,0];dp=p[j,i]-p[0,0];dt=true[i]-true[0]
            angular=abs(np.arctan2(np.sin(dp[2]-dt[2]),np.cos(dp[2]-dt[2])));position=np.linalg.norm(dp[:2]-dt[:2]);rotation=np.rad2deg(angular)
            values={'latent_distance':np.linalg.norm(dz),'position_error':position,'rotation_error_deg':rotation,'true_position_effect':np.linalg.norm(dt[:2]),'true_rotation_effect_deg':np.rad2deg(abs(np.arctan2(np.sin(dt[2]),np.cos(dt[2]))))}
            for k,v in values.items():
                error=abs(float(v)-getattr(r,k));maxima[k]=max(maxima.get(k,0.),error);assert error<(1e-4 if 'rotation' in k else 1e-5),(k,error)
            assert bool(r.within_2px_2deg)==bool(position<=2 and angular<=np.deg2rad(2));count+=1
previous=json.loads((out/'raw_audit.json').read_text());previous.update({'reconstructed_rows':count,'feature_pose_truth_recalculation_max_error':maxima,'audit_precision':'Float32 probe-angle wrapping versus independent wrap is checked at1e-4degree; other metric errors1e-5. Both much below declared physical tolerances.','auditor_sha256':sha256(Path(__file__))});write_json(out/'raw_audit.json',previous);print(previous)
