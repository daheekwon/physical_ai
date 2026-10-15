"""Calibrate resolution from real encoded pairs, then report propagation separately."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve,write_json,sha256


def load(out,split):
    folder=out/'lewm_audit/effects';paths=sorted(folder.glob(f'{split}_*_h05.json'))
    if len(paths)!=105 and split!='test':raise ValueError(f'Need all 105 {split} states, got {len(paths)}')
    return pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in paths],ignore_index=True)


def family_interval(values,families):
    groups=pd.DataFrame({'v':np.asarray(values),'f':np.asarray(families)}).groupby('f').v.agg(['sum','count']).to_numpy()
    if not len(groups):return None
    rng=np.random.default_rng(20261011)
    sampled=groups[rng.integers(0,len(groups),(2000,len(groups)))].sum(1)
    boot=sampled[:,0]/sampled[:,1]
    return np.quantile(boot,[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-resolution',action='store_true');parser.add_argument('--split',default='development');args=parser.parse_args()
    out=resolve('artifacts/followup_v1');directory=out/'lewm_audit/effects';cfg=json.loads(resolve('configs/followup_effect_audit.json').read_text())
    threshold_path=directory/'resolution.json'
    if args.fit_resolution:
        if threshold_path.exists():raise ValueError('Resolution already frozen')
        if list(directory.glob('test_*.json')):raise ValueError('Cannot fit resolution after test effects')
        cal=load(out,'calibration');real=cal[cal.method=='real_encoded'];thresholds={}
        for (rep,kind),g in real.groupby(['representation','readout']):
            thresholds[rep+'_'+kind]={'position':max(1e-6,float(g.position_effect_error.quantile(cfg['resolution_error_quantile']))*cfg['resolution_multiplier']),
                                     'rotation':max(1e-8,float(g.rotation_effect_error.quantile(cfg['resolution_error_quantile']))*cfg['resolution_multiplier']),
                                     'states':g.state_id.nunique(),'families':g.initial_pose_family.nunique()}
        write_json(threshold_path,{'config':cfg,'thresholds':thresholds,'config_sha256':sha256(resolve('configs/followup_effect_audit.json')),
                    'interpretation':'Diagnostic effect resolution only, never a correction acceptance threshold'})
    if not threshold_path.exists():raise ValueError('Fit calibration resolution first')
    resolution=json.loads(threshold_path.read_text());data=load(out,args.split);rows=[];bins=[]
    for (rep,kind),g in data.groupby(['representation','readout']):
        threshold=resolution['thresholds'][rep+'_'+kind]
        for method,v in g.groupby('method'):
            for phase,frame in [('all',v),('contact',v[v.nominal_contact])]:
                if not len(frame):continue
                qualified_pos=frame.position_effect>threshold['position'];qualified_rot=frame.rotation_effect>threshold['rotation']
                point={'representation':rep,'readout':kind,'method':method,'phase':phase,'n':len(frame),
                       'states':frame.state_id.nunique(),'families':frame.initial_pose_family.nunique(),
                       'position_error_mean':float(frame.position_effect_error.mean()),
                       'rotation_error_mean_deg':float(np.rad2deg(frame.rotation_effect_error.mean())),
                       'position_error_ci':family_interval(frame.position_effect_error,frame.initial_pose_family),
                       'rotation_error_ci_deg':family_interval(np.rad2deg(frame.rotation_effect_error),frame.initial_pose_family),
                       'position_absolute_mae':float(frame.position_absolute_error.mean()),
                       'rotation_absolute_mae_deg':float(np.rad2deg(frame.rotation_absolute_error.mean())),
                       'position_resolvable_fraction':float(qualified_pos.mean()),'rotation_resolvable_fraction':float(qualified_rot.mean()),
                       'position_direction_agreement':float((frame.loc[qualified_pos,'position_cosine']>=cfg['direction_cosine_threshold']).mean()) if qualified_pos.any() else None,
                       'rotation_sign_agreement':float(frame.loc[qualified_rot,'rotation_sign_correct'].mean()) if qualified_rot.any() else None,
                       'qualified_position_error':float(frame.loc[qualified_pos,'position_effect_error'].mean()) if qualified_pos.any() else None,
                       'qualified_rotation_error_deg':float(np.rad2deg(frame.loc[qualified_rot,'rotation_effect_error'].mean())) if qualified_rot.any() else None,
                       'physical_preservation_all_candidates':float(frame.preserved.mean())}
                rows.append(point)
                for axis,edges in [('position',cfg['position_bins_px']),('rotation',np.deg2rad(cfg['rotation_bins_deg']))]:
                    binned=pd.cut(frame[axis+'_effect'],edges,include_lowest=True,right=False)
                    for label,k in frame.groupby(binned,observed=True):
                        bins.append({'representation':rep,'readout':kind,'method':method,'phase':phase,'axis':axis,'bin':str(label),
                                     'n':len(k),'effect_mean':float(k[axis+'_effect'].mean()),'error_mean':float(k[axis+'_effect_error'].mean()),
                                     'predicted_magnitude_mean':float(k['predicted_'+axis+'_magnitude'].mean())})
    comparisons=[]
    for (rep,kind),g in data.groupby(['representation','readout']):
        predicted=g[g.method=='predicted']
        if not len(predicted):continue
        keys=['state_id','index','endpoint_step']
        for control in ['zero_effect','wrong_action','real_encoded']:
            paired=predicted.merge(g[g.method==control][keys+['position_effect_error','rotation_effect_error']],on=keys,suffixes=('','_control'),validate='one_to_one')
            for phase,p in [('all',paired),('contact',paired[paired.nominal_contact])]:
                pe=p.position_effect_error-p.position_effect_error_control;re=np.rad2deg(p.rotation_effect_error-p.rotation_effect_error_control)
                comparisons.append({'representation':rep,'readout':kind,'control':control,'phase':phase,'n':len(p),
                                    'position_predicted_minus_control':float(pe.mean()),'position_family_ci':family_interval(pe,p.initial_pose_family),
                                    'rotation_predicted_minus_control_deg':float(re.mean()),'rotation_family_ci_deg':family_interval(re,p.initial_pose_family)})
    pd.DataFrame(rows).to_csv(directory/f'{args.split}_summary.csv',index=False);pd.DataFrame(bins).to_csv(directory/f'{args.split}_effect_bins.csv',index=False)
    write_json(directory/f'{args.split}_summary.json',{'scope':f'{args.split}; thresholds from calibration; final test remains separately locked',
               'resolution_sha256':sha256(threshold_path),'summaries':rows,'paired_controls':comparisons,
               'all_physical_cases_retained':True,
               'persistence_note':'Persistence uses same current readout; zero-effect absolute anchor is the measured nominal encoded endpoint, so zero-effect is an effect baseline, not a deployable absolute predictor.'})
    print(json.dumps({'rows':len(rows),'paired_controls':len(comparisons),'resolution':resolution['thresholds']},indent=2))

if __name__=='__main__':main()
