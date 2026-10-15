"""Audit raw experiment outputs and produce additional diagnostic figures."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zj_wam.io import resolve,sha256,write_json
from zj_wam.metrics import preservation_mask,rotation_error

out=resolve('artifacts/phase1')
manifest=json.loads((out/'manifest.json').read_text())
cfg=manifest['config']
expected=manifest['states']
metadata=[json.loads(p.read_text()) for p in (out/'states').glob('*.json')]
assert {r['state_id'] for r in metadata}=={r['state_id'] for r in expected}
assert len(metadata)==len(expected)
groups={k:{r['initial_pose_family'] for r in expected if r['split']==k} for k in cfg['split_episodes']}
assert not(groups['development']&groups['calibration'] or groups['development']&groups['test'] or groups['calibration']&groups['test'])
assert not groups['test'] & set(cfg.get('excluded_initial_pose_families', []))
cal_hash=sha256(out/'calibration.json')
assert all(r['calibration_sha256']==cal_hash for r in metadata if r['split']=='test')
assert all(r['zero_simulator_error']==0 and r['zero_prediction_error']==0 for r in metadata)
frame=pd.read_csv(out/'all_candidates.csv')
assert len(frame)==len(expected)*120
valid=frame[frame.status=='ok'].copy()
assert not valid.clipped.any()
assert np.allclose(valid.actual_delta_norm,valid.applied_radius,rtol=2e-4,atol=2e-6)
assert valid.groupby(['state_id','relative_radius']).applied_radius.nunique().max()==1
assert len(frame.groupby(['state_id','relative_radius','direction','sign']))==len(frame)
quality=[]
for spec in expected:
    name=spec['state_id']
    with np.load(out/'states'/f'{name}.npz') as a:
        assert np.allclose(np.linalg.norm(a['directions'],axis=1),1,atol=1e-6)
        assert np.all(np.diff(a['eigenvalues'])>=-1e-12)
        assert np.allclose(a['eigenvectors'].T@a['eigenvectors'],np.eye(10),atol=1e-7)
        block=valid[valid.state_id==name]
        trajectories=a['object_trajectories']
        ref=trajectories[0]
        for row in block.itertuples():
            k=int(row.trajectory_array_index)
            candidate=trajectories[k]
            assert np.isclose(np.linalg.norm(candidate[-1,2:4]-ref[-1,2:4]),row.position_error,atol=1e-9)
            assert np.isclose(rotation_error(candidate[-1,4],ref[-1,4]),row.rotation_error,atol=1e-9)
            assert np.max(np.abs(a['actions'][k]))<=1+1e-5
        quality.append({**spec,
                        'replay_object_position_error':float(np.linalg.norm(a['recorded_state'][2:4]-a['replayed_state'][2:4])),
                        'replay_object_rotation_error':float(rotation_error(a['recorded_state'][4],a['replayed_state'][4]))})
quality=pd.DataFrame(quality);quality.to_csv(out/'replay_quality.csv',index=False)
fd=np.array([r['finite_difference_relative_errors'] for r in metadata])
audit={'complete':True,'states':len(metadata),'candidate_rows':len(frame),'valid_rows':len(valid),
       'excluded_rows':len(frame)-len(valid),'bounds_reduced_rows':int(valid.bounds_reduced.sum()),
       'bounds_reduced_state_ids':sorted(valid[valid.bounds_reduced].state_id.unique().tolist()),
       'all_zero_perturbation_and_restore_checks_pass':True,
       'unit_directions_equal_norm_and_action_bounds_pass':True,
       'stored_trajectories_match_metrics':True,'split_families_disjoint':True,
       'calibration_sha256':cal_hash,
       'checkpoint_unchanged':sha256(resolve(cfg['checkpoint'])/'weights.pt')==manifest['checkpoint_sha256'],
       'finite_difference_max_relative_error_by_low_high_random':fd.max(0).tolist(),
       'finite_difference_median_relative_error_by_low_high_random':np.median(fd,axis=0).tolist(),
       'replay_object_position_max':float(quality.replay_object_position_error.max()),
       'replay_object_position_median':float(quality.replay_object_position_error.median()),
       'replay_object_rotation_max':float(quality.replay_object_rotation_error.max()),
       'phase_counts':{split:pd.Series([r['interaction_phase'] for r in metadata if r['split']==split]).value_counts().to_dict() for split in cfg['split_episodes']}}
audit['previous_test_families_excluded'] = True
calibration=json.loads((out/'calibration.json').read_text())
audit['calibration_source_states_used']=calibration['calibration_source_states_used']
audit['calibration_source_quality_exclusions']=calibration['excluded_calibration_source_replay_states']
audit['replay_quality_by_split']={}
for split,g in quality.groupby('split'):
    audit['replay_quality_by_split'][split]={'position_max':float(g.replay_object_position_error.max()),
                                            'position_median':float(g.replay_object_position_error.median()),
                                            'rotation_max':float(g.replay_object_rotation_error.max())}
used=quality[(quality.split=='calibration') & ~quality.state_id.isin(calibration['excluded_calibration_source_replay_states'])]
assert used.replay_object_position_error.max()<=cfg['tolerances']['position']
assert used.replay_object_rotation_error.max()<=cfg['tolerances']['rotation']
test_quality=quality[quality.split=='test']
assert test_quality.replay_object_position_error.max()<=cfg['tolerances']['position']
assert test_quality.replay_object_rotation_error.max()<=cfg['tolerances']['rotation']
write_json(out/'audit.json',audit)
heldout=pd.read_csv(out/'heldout_candidates.csv')
# Descriptive sensitivity analysis only: the deployed calibration is not changed.
tolerance_rows=[]
for scale in [.5,1.,2.]:
    tolerance={k:v*scale for k,v in cfg['tolerances'].items()}
    for phase in ['all','contact','free_motion','near_contact']:
        f=heldout if phase=='all' else heldout[heldout.interaction_phase==phase]
        mask=preservation_mask(f,tolerance)
        for (radius,method),g in f.assign(preserved_scaled=mask).groupby(['relative_radius','group']):
            tolerance_rows.append({'tolerance_scale':scale,'phase':phase,'relative_radius':radius,'group':method,'preservation_rate':float(g.preserved_scaled.mean())})
pd.DataFrame(tolerance_rows).to_csv(out/'tolerance_sensitivity.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for ax,radius in zip(axes,[.1,.8]):
    for method in ['low','random','high','low_mix']:
        f=pd.DataFrame(tolerance_rows)
        f=f[(f.phase=='contact')&(f.relative_radius==radius)&(f.group==method)]
        ax.plot(f.tolerance_scale,f.preservation_rate,'o-',label=method)
    ax.set(xlabel='Tolerance multiplier',ylabel='Contact preservation fraction',title=f'Relative radius {radius}',ylim=(-.03,1.03))
axes[1].legend();fig.tight_layout();fig.savefig(out/'plots/06_tolerance_sensitivity.png',dpi=140);plt.close(fig)
failures=heldout[(heldout.group.isin(['low','low_basis','low_mix']))&heldout.deployed_maximum&~heldout.preserved].copy()
failures.sort_values('object_error',ascending=False).to_csv(out/'deployed_failure_cases.csv',index=False)
# First three contact states by ID, selected without looking at effect magnitude.
selected=sorted(heldout[heldout.interaction_phase=='contact'].state_id.unique())[:3]
fig,axes=plt.subplots(1,len(selected),figsize=(5*len(selected),4),squeeze=False)
for ax,sid in zip(axes[0],selected):
    a=np.load(out/'states'/f'{sid}.npz');tr=a['object_trajectories']
    ax.plot(tr[0,:,2],tr[0,:,3],'ko-',label='nominal',linewidth=2)
    for direction,color in [('low','tab:blue'),('high','tab:red')]:
        row=heldout[(heldout.state_id==sid)&(heldout.direction==direction)&(heldout.sign==1)&np.isclose(heldout.relative_radius,.4)].iloc[0]
        t=tr[int(row.trajectory_array_index)]
        ax.plot(t[:,2],t[:,3],'o--',color=color,label=direction,alpha=.8)
    ax.set_title(sid,fontsize=8);ax.set(xlabel='Object x (px)',ylabel='Object y (px)');ax.invert_yaxis();ax.axis('equal');ax.legend()
fig.suptitle('First three held-out contact states; + perturbation, relative radius 0.40')
fig.tight_layout();fig.savefig(out/'plots/07_object_trajectories.png',dpi=140);plt.close(fig)
print(json.dumps(audit,indent=2),flush=True)
