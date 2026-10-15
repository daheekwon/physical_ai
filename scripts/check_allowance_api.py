"""Check the public allowance API against actual held-out model decisions."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from zj_wam.adapters import FrozenLeWM
from zj_wam.allowance import estimate_action_allowance
from zj_wam.io import resolve,write_json

out=resolve('artifacts/phase1')
manifest=json.loads((out/'manifest.json').read_text())
calibration=json.loads((out/'calibration.json').read_text())
cfg=manifest['config']
torch.set_num_threads(4)
scaler=StandardScaler()
scaler.mean_=np.array(manifest['normalization_mean']);scaler.scale_=np.array(manifest['normalization_scale'])
scaler.var_=scaler.scale_**2;scaler.n_features_in_=2
model=FrozenLeWM(cfg['checkpoint'],scaler)
frame=pd.read_csv(out/'heldout_candidates.csv')
selected=sorted(frame[frame.interaction_phase=='contact'].state_id.unique())[:3]
results=[]
for sid in selected:
    arrays=np.load(out/'states'/f'{sid}.npz')
    predict,_=model.predictor(arrays['nominal_observation'],arrays['past_actions'])
    action=torch.from_numpy(model.normalize(arrays['nominal_action']).reshape(-1))
    rows=frame[frame.state_id==sid]
    grid=sorted(rows.applied_radius.unique().tolist())
    valid=lambda a: bool(np.max(np.abs(model.denormalize(a.detach().numpy())))<=1+1e-6)
    result=estimate_action_allowance(predict,action,calibration,valid_action=valid,radius_grid=grid)
    expected=np.zeros_like(result.allowed_correction_scale)
    for i,key in enumerate(['low','low_basis_1','low_basis_2']):
        for j,sign in enumerate([-1,1]):
            choices=rows[(rows.direction==key)&(rows.sign==sign)&rows.deployed_maximum]
            if len(choices): expected[i,j]=choices.applied_radius.max()
    np.testing.assert_allclose(result.allowed_correction_scale,expected,atol=1e-7)
    results.append({'state_id':sid,'allowed_signed_scales':result.allowed_correction_scale.tolist(),
                    'matches_heldout_decisions':True,'diagnostics':result.diagnostics})
write_json(out/'allowance_api_check.json',results)
print(json.dumps(results,indent=2))
