"""Lock each strong baseline's declared global setting on calibration, then evaluate development."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scripts.analyze_followup_response_selection import mean_ci
from scripts.calibrate_followup_selection import selection_metrics
from zj_wam.io import resolve,write_json,sha256

OUT=resolve('artifacts/followup_v1')

def load(split,extended=False):
    bank='strong_baselines_extended' if extended else 'strong_baselines'
    paths=sorted((OUT/bank).glob(f'{split}*h05.json'));expected=36 if extended else 105
    if len(paths)!=expected:raise ValueError(f'{bank}/{split} incomplete {len(paths)}/{expected}')
    return pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in paths],ignore_index=True)


def rows_for_setting(frame,method,setting=None,radius=None,disabled=False):
    f=frame[frame.method==method]
    if disabled:
        f=f.groupby('state_id',sort=True).first().reset_index()
    else:f=f[np.isclose(f.setting,setting)&np.isclose(f.relative_radius,radius)].copy()
    if f.state_id.duplicated().any():raise ValueError('Ambiguous setting')
    nominal=f.nominal_predicted_jerk.to_numpy();predicted=np.divide(nominal-f.predicted_jerk,nominal,out=np.zeros(len(f)),where=nominal>1e-8)
    active=(f.actual_norm.to_numpy()>1e-6)&(predicted>1e-8)&(not disabled)
    return pd.DataFrame({'state_id':f.state_id.to_numpy(),'initial_pose_family':f.initial_pose_family.to_numpy(),'nominal_contact':f.nominal_contact.to_numpy(),
                         'method':method,'active':active,'index':np.where(active,f['index'],0),'predicted_gain':np.where(active,predicted,0.),
                         'jerk_gain':np.where(active,f.jerk_gain,0.),'object_error':np.where(active,f.object_error,0.),'preserved':np.where(active,f.preserved,True),
                         'norm':np.where(active,f.actual_norm,0.),'optimizer_seconds':f.optimizer_seconds.to_numpy(),
                         'forward_calls':f.forward_calls.to_numpy(),'backward_calls':f.backward_calls.to_numpy()})


def main():
    cfg=json.loads(resolve('configs/followup_selection.json').read_text());directory=OUT/'strong_baselines';cal=load('calibration');dev=load('development')
    path=directory/'calibrated_settings.json'
    if not path.exists():
        if (OUT/'protocol/final_selection_lock.json').exists():raise ValueError('Cannot tune after test unlock')
        rules={};curves=[]
        for method,f in cal.groupby('method'):
            best=None;best_value=None
            for (setting,radius),_ in f.groupby(['setting','relative_radius']):
                chosen=rows_for_setting(cal,method,setting,radius);metrics=selection_metrics(chosen,cfg)
                valid=metrics['active_coverage']>=cfg['minimum_active_coverage'] and metrics['risk'] is not None and metrics['risk']<=cfg['target_violation'] and metrics['contact_risk'] is not None and metrics['contact_risk']<=cfg['target_violation']
                curves.append({'method':method,'setting':float(setting),'radius':float(radius),**metrics,'valid':valid})
                value=(metrics['predicted_useful_coverage'],metrics['mean_predicted_gain'])
                if valid and (best_value is None or value>best_value):best={'enabled':True,'setting':float(setting),'radius':float(radius),'calibration':metrics};best_value=value
            rules[method]=best or {'enabled':False,'reason':'No declared global setting met calibration risk/coverage constraints'}
        pd.DataFrame(curves).to_csv(directory/'calibration_curves.csv',index=False)
        write_json(path,{'rules':rules,'scope':'Calibration empirical constraints; no safety guarantee. Fixed global settings, per-state optimizer remains active.','source_sha256':sha256(Path(__file__))})
    else:rules=json.loads(path.read_text())['rules']
    choices=[];summary=[]
    for split,frame in [('development',dev),('calibration',cal)]:
        for method,r in rules.items():
            chosen=rows_for_setting(frame,method,r.get('setting'),r.get('radius'),not r['enabled']);chosen['split']=split
            chosen['joint']=chosen.active & chosen.preserved & (chosen.jerk_gain>=.1);choices.append(chosen)
            for stratum in ['all','contact','free']:
                g=chosen if stratum=='all' else chosen[chosen.nominal_contact==(stratum=='contact')]
                active=g[g.active].copy();active['violation']=~active.preserved
                summary.append({'split':split,'method':method,'stratum':stratum,'joint':mean_ci(g,'joint'),'active':mean_ci(g,'active'),'risk':mean_ci(active,'violation'),'gain':mean_ci(g,'jerk_gain'),
                                'solver_seconds':mean_ci(g,'optimizer_seconds'),'native_forward':mean_ci(g,'forward_calls'),'native_backward':mean_ci(g,'backward_calls')})
    chosen=pd.concat(choices,ignore_index=True);chosen.to_csv(directory/'selected.csv',index=False)
    common=pd.read_csv(OUT/'selection/selected_evaluation.csv');paired=[]
    for split in ['development','calibration']:
      for baseline in ['controller_pusher','lewm_global','vision_contact','vision_adaptive']:
       base=common[(common.split==split)&(common.rule_source=='selector')&(common.method==baseline)]
       for method in rules:
        g=chosen[(chosen.split==split)&(chosen.method==method)].merge(base,on='state_id',suffixes=('','_base'))
        g['joint_difference']=g.joint.astype(float)-g.joint_base.astype(float);g['gain_difference']=g.jerk_gain-g.jerk_gain_base
        paired.append({'split':split,'method':method,'baseline':baseline,'joint_difference':mean_ci(g,'joint_difference'),'gain_difference':mean_ci(g,'gain_difference')})
    write_json(directory/'analysis.json',{'scope':'Development diagnostic and calibration resubstitution. Geometry/filter use larger action families; polynomial comparators identified explicitly. Wall times are solver-only under shared GPUs, not end-to-end latency or a compute-matched claim.','summary':summary,'paired':paired})
    for r in summary:
        if r['split']=='development' and r['stratum']=='all':print(r['method'],'U',r['joint']['mean'],'R',r['risk']['mean'])
    extended_paths=list((OUT/'strong_baselines_extended').glob('development*h05.json'))
    if len(extended_paths)==36:
        extended=load('development',True);small=dev[dev.method=='nonlinear_wm'];joined=extended.merge(small,on=['state_id','setting','relative_radius'],suffixes=('','_64'))
        joined['predicted_gain_extra']=(joined.predicted_jerk_64-joined.predicted_jerk)/joined.nominal_predicted_jerk
        joined['actual_gain_extra']=joined.jerk_gain-joined.jerk_gain_64
        joined['joint_delta']=((joined.actual_norm>1e-6)&joined.preserved&(joined.jerk_gain>=.1)).astype(float)-((joined.actual_norm_64>1e-6)&joined.preserved_64&(joined.jerk_gain_64>=.1)).astype(float)
        state=joined.groupby(['state_id','initial_pose_family'])[['predicted_gain_extra','actual_gain_extra','joint_delta']].mean().reset_index()
        write_json(directory/'convergence_analysis.json',{'scope':'Same 12 development families, same 16 parameter settings; 256 vs64 native forward budget, two starts. Family averages over settings.','rows':len(joined),'states':len(state),**{c:mean_ci(state,c) for c in ['predicted_gain_extra','actual_gain_extra','joint_delta']}})
if __name__=='__main__':main()
