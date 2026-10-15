"""Audit post hoc controller-only and sampling-rate sensitivity diagnostics."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve,write_json
from scripts.controller_only_diagnostic import controller_response
out=resolve('artifacts/phase1_additional');m=json.loads((out/'manifest.json').read_text())
f=pd.read_csv(out/'controller_candidates.csv');assert len(f)==120*24
assert set(f.state_id)=={s['state_id'] for s in m['controls_states']}
scale=np.array(m['normalization_scale']);r=controller_response(5,scale);maxerror=0.
for sid,g in f.groupby('state_id'):
    a=dict(np.load(out/'controller'/f'{sid}.npz'))
    old=pd.read_csv(out/'controls'/f'{sid}.csv')
    for row in g.itertuples():
        assert np.isclose(row.applied_radius,old[old.relative_radius==row.relative_radius].applied_radius.iloc[0])
        actual=((a['actions'][row.trajectory_index]-a['actions'][0])/scale).reshape(-1)
        assert np.isclose(np.linalg.norm(actual),row.applied_radius,rtol=2e-4,atol=3e-6)
        assert np.max(np.abs(a['actions'][row.trajectory_index]))<=1+1e-5
        trajectory=a['trajectories'][row.trajectory_index];nominal=a['trajectories'][0]
        prediction=r@actual;execution=(trajectory[:,:2]-nominal[:,:2]).reshape(-1)
        error=float(np.max(np.abs(prediction-execution)));maxerror=max(error,maxerror)
        assert error<1e-3
        assert np.isclose(np.linalg.norm(trajectory[-1,2:4]-nominal[-1,2:4]),row.position_error,atol=1e-8)
raw=pd.read_csv(out/'oracle_candidates.csv');chosen=pd.read_csv(out/'oracle_physics_selected.csv')
assert len(chosen)==24*3
for row in chosen.itertuples():
    g=raw[(raw.state_id==row.state_id)&(raw.horizon==row.horizon)]
    feasible=g[g.preserved_path]
    best=min(g.nominal_physics_jerk.iloc[0],feasible.physics_jerk.min()) if len(feasible) else g.nominal_physics_jerk.iloc[0]
    assert np.isclose(row.physics_jerk,best)
write_json(out/'extensions_audit.json',{'controller_executions_verified':len(f),'analytic_pusher_response_max_error':maxerror,
    'controller_candidates_equal_norm_with_primary_controls':True,'physics_oracle_selections_verified':len(chosen),
    'scope':'Post hoc exploratory controller baseline and secondary sampling-rate analysis; no change to original or additional calibration'})
print('Extension audit passed',maxerror)
