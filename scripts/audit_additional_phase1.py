"""Verify new raw records, matched spectra, polynomial membership and selection."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.fft import dct
from zj_wam.io import resolve,sha256,write_json
from zj_wam.diagnostics import jerk_energy
from zj_wam.metrics import rotation_error
from scripts.analyze_additional_phase1 import prepare,select_model,nominal_row

cfg=json.loads(resolve('configs/phase1_additional.json').read_text());out=resolve(cfg['output'])
m=json.loads((out/'manifest.json').read_text());assert m['config']==cfg
assert sha256(resolve(cfg['proposal']))==m['proposal_sha256']
assert sha256(resolve('artifacts/phase1/calibration.json'))==m['original_calibration_sha256']
for path,digest in json.loads(resolve('artifacts/phase1/source_checksums.json').read_text()).items():assert sha256(resolve(path))==digest,path
scale=np.array(m['normalization_scale']);audit={};traces_checked=0
controls=pd.read_csv(out/'controls_candidates.csv')
assert len(controls)==len(m['controls_states'])*108
assert set(controls.state_id)=={s['state_id'] for s in m['controls_states']}
assert np.allclose(controls.applied_radius,controls.actual_norm,rtol=2e-4,atol=2e-6)
assert controls.groupby(['state_id','relative_radius']).applied_radius.nunique().max()==1
for sid,g in controls.groupby('state_id'):
    a=dict(np.load(out/'controls'/f'{sid}.npz'));d=dict(zip(a['direction_names'],a['directions']))
    power=lambda x:np.sum(dct(x.reshape(-1,2),axis=0,norm='ortho')**2,axis=1)
    for key,value in d.items():
        assert np.isclose(np.linalg.norm(value),1,atol=1e-6)
        if key.startswith('frequency_'):np.testing.assert_allclose(power(value),power(d['low']),atol=1e-12)
    for r in g.itertuples():
        tr=a['trajectories'][r.trajectory_index];ref=a['trajectories'][0]
        assert np.isclose(np.linalg.norm(tr[-1,2:4]-ref[-1,2:4]),r.position_error,atol=1e-8)
        assert np.max(np.abs(a['actions'][r.trajectory_index]))<=1+1e-5
        actual=((a['actions'][r.trajectory_index]-a['actions'][0])/scale).reshape(-1)
        assert np.isclose(np.linalg.norm(actual),r.applied_radius,rtol=2e-4,atol=3e-6)
        traces_checked+=1
oracle=prepare(pd.read_csv(out/'oracle_candidates.csv'),cfg)
expected=len(m['oracle_states'])*3*m['polynomial_candidate_budget_per_state_horizon'];assert len(oracle)==expected
assert set(oracle.state_id)=={s['state_id'] for s in m['oracle_states']}
cal_hash=sha256(out/'oracle_calibration.json');assert (out/'oracle_calibration_frozen.sha256').read_text().strip()==cal_hash
metadata=[]
for (sid,h),g in oracle.groupby(['state_id','horizon']):
    stem=f'{sid}_h{h:02d}';meta=json.loads((out/'oracle'/f'{stem}.json').read_text());metadata.append(meta)
    if meta['split']=='test':assert meta['calibration_sha256']==cal_hash
    assert meta['repeat_physics_max_error']==0
    a=dict(np.load(out/'oracle'/f'{stem}.npz'));q=a['polynomial_basis']
    assert q.shape==(2*h,4)
    for r in g.itertuples():
        k=r.trajectory_index;tr=a['control_traces'][k];ref=a['control_traces'][0]
        assert tr.shape==(h+6,7)
        assert a['physics_traces'][k].shape==((h+3)*10+3,7)
        actual=((a['actions'][k]-a['actions'][0])/scale).reshape(-1)
        np.testing.assert_allclose(q@q.T@actual,actual,atol=2e-6)
        np.testing.assert_allclose(actual[[0,1,-2,-1]],0,atol=2e-6)
        assert np.isclose(np.linalg.norm(actual),r.applied_radius,rtol=2e-4,atol=3e-6)
        assert np.isclose(jerk_energy(tr[:,:2],.1),r.executed_jerk)
        assert np.isclose(jerk_energy(a['physics_traces'][k,:,:2],.01),r.physics_jerk)
        assert np.isclose(np.linalg.norm(tr[h+2,2:4]-ref[h+2,2:4]),r.position_error,atol=1e-8)
        assert np.isclose(np.max(np.linalg.norm(tr[:,2:4]-ref[:,2:4],axis=1)),r.position_with_suffix_max,atol=1e-8)
        assert np.isclose(np.linalg.norm(a['predictions'][k]-a['predictions'][0]),r.predicted_change,rtol=2e-5,atol=1e-7)
        assert np.max(np.abs(a['actions'][k]))<=1+1e-5
        traces_checked+=1
selected=pd.read_csv(out/'oracle_selected.csv');cal=json.loads((out/'oracle_calibration.json').read_text())
for (sid,h),g in oracle[oracle.split=='test'].groupby(['state_id','horizon']):
    expected=select_model(g,cal['rules'][str(h)])
    stored=selected[(selected.state_id==sid)&(selected.horizon==h)&(selected.selection_method=='model_selected')].iloc[0]
    assert stored.trajectory_index==expected.trajectory_index
    for method,flag in [('oracle_final','preserved_final'),('oracle_path','preserved_path')]:
        available=g[g[flag]]
        optimum=min(g.nominal_executed_jerk.iloc[0],available.executed_jerk.min()) if len(available) else g.nominal_executed_jerk.iloc[0]
        found=selected[(selected.state_id==sid)&(selected.horizon==h)&(selected.selection_method==method)].iloc[0]
        assert np.isclose(found.executed_jerk,optimum)
# Family membership is disjoint across all three splits.
groups={split:{s['initial_pose_family'] for s in m['oracle_states'] if s['split']==split} for split in ['development','calibration','test']}
assert not(groups['development']&groups['calibration'] or groups['development']&groups['test'] or groups['calibration']&groups['test'])
summary={'controls_states':controls.state_id.nunique(),'controls_executions':len(controls),'oracle_states':oracle.state_id.nunique(),'oracle_executions':len(oracle),
         'all_raw_trajectories_checked':traces_checked,'frequency_spectra_match':True,'equal_norms_and_bounds':True,
         'polynomial_membership_endpoints_verified':True,'boundary_control_and_physics_jerk_verified':True,
         'oracle_optima_match_candidate_pool':True,'model_selection_uses_locked_rule':True,
         'original_sources_and_calibration_unchanged':True,'proposal_unchanged':True,'split_families_disjoint':True,
         'max_directional_fd_relative_error':np.array([r['finite_difference_relative_errors'] for r in metadata]).max(0).tolist(),
         'scope':'Finite candidate search, exploratory existing test families; no global optimum, external generalization or full Phase 2 claim.'}
write_json(out/'audit.json',summary);print(json.dumps(summary,indent=2))
