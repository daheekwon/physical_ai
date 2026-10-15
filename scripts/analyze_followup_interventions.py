"""F2 controls, response strata, and outcome-blind matching of actual pusher motion."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from zj_wam.io import resolve,write_json,sha256


def bootstrap_pairs(frame, column, seed=20261011):
    if len(frame)==0:return None
    values=[g[column].to_numpy(float) for _,g in frame.groupby('initial_pose_family')]
    rng=np.random.default_rng(seed)
    means=[np.concatenate([values[i] for i in rng.integers(0,len(values),len(values))]).mean() for _ in range(2000)]
    return np.quantile(means,[.025,.975]).tolist()


def match_pusher(low,other,cfg):
    if not len(low) or not len(other):return []
    a=low[['pusher_rms','pusher_max','frequency_centroid']].to_numpy()
    b=other[['pusher_rms','pusher_max','frequency_centroid']].to_numpy()
    size=np.abs(a[:,None,:2]-b[None,:,:2])/np.maximum(np.maximum(a[:,None,:2],b[None,:,:2]),cfg['minimum_pusher_magnitude'])
    frequency=np.abs(a[:,None,2]-b[None,:,2])
    valid=(size.max(-1)<=cfg['pusher_relative_caliper'])&(frequency<=cfg['frequency_centroid_caliper'])
    cost=size.sum(-1)+frequency
    ii,jj=linear_sum_assignment(np.where(valid,cost,1e6))
    return [(low.iloc[i],other.iloc[j],float(size[i,j].max()),float(frequency[i,j])) for i,j in zip(ii,jj) if valid[i,j]]


def main():
    out=resolve('artifacts/followup_v1');bank=out/'interventions';cfg=json.loads(resolve('configs/followup_f2_analysis.json').read_text())
    paths=sorted(bank.glob('development_*_h05.json'))
    frames=[pd.read_csv(p.with_suffix('.csv')) for p in paths]
    f=pd.concat(frames,ignore_index=True)
    pos=np.hypot(f.endpoint_dx,f.endpoint_dy)/2;rot=np.abs(f.endpoint_dtheta)/np.deg2rad(2)
    f['effect_type']=np.where((pos+rot)<1e-6,'unmoved',np.where(pos>=2*rot,'translation',np.where(rot>=2*pos,'rotation','mixed')))
    comparisons=[];matches=[]
    for sid,g in f.groupby('state_id'):
        low=g[g.method=='free_low']
        for comparator,prefix in [('frequency','frequency_'),('controller','controller_'),('random','free_random_')]:
            other=g[g.method.str.startswith(prefix)]
            for (relative,sign),l in low.groupby(['relative_radius','sign']):
                a=l.iloc[0];b=other[(other.relative_radius==relative)&(other.sign==sign)]
                comparisons.append({'state_id':sid,'initial_pose_family':a.initial_pose_family,'nominal_contact':a.nominal_contact,
                    'comparator':comparator,'relative_radius':relative,'sign':sign,'low_minus_object_error':float(a.object_error-b.object_error.mean()),
                    'low_minus_position':float(a.position_max-b.position_max.mean()),'low_minus_pusher_rms':float(a.pusher_rms-b.pusher_rms.mean())})
            for a,b,size,freq in match_pusher(low,other,cfg):
                matches.append({'state_id':sid,'initial_pose_family':a.initial_pose_family,'nominal_contact':a.nominal_contact,
                    'comparator':comparator,'low_index':int(a['index']),'other_index':int(b['index']),
                    'size_relative_mismatch':size,'centroid_mismatch':freq,
                    'low_minus_object_error':float(a.object_error-b.object_error),'low_minus_position':float(a.position_max-b.position_max),
                    'low_norm':a.actual_norm,'other_norm':b.actual_norm})
    c=pd.DataFrame(comparisons);m=pd.DataFrame(matches)
    c.to_csv(out/'interventions/equal_action_pairs.csv',index=False);m.to_csv(out/'interventions/matched_pusher_pairs.csv',index=False)
    summaries=[]
    for kind,table in [('equal_action',c),('matched_pusher',m)]:
        for comparator,g in table.groupby('comparator'):
            for phase,k in [('all',g),('contact',g[g.nominal_contact])]:
                if not len(k):continue
                # Equal-action has four radii; do not let its pooled score replace per-radius reporting.
                for radius,v in ([('all',k)] if kind=='matched_pusher' else list(k.groupby('relative_radius'))):
                    summaries.append({'matching':kind,'comparator':comparator,'phase':phase,'radius':radius,
                        'pairs':len(v),'states':v.state_id.nunique(),'families':v.initial_pose_family.nunique(),
                        'low_minus_object_error':float(v.low_minus_object_error.mean()),'family_ci':bootstrap_pairs(v,'low_minus_object_error')})
    f.groupby(['method','relative_radius','nominal_contact','effect_type']).agg(n=('index','size'),preservation=('preserved','mean'),error=('object_error','mean')).reset_index().to_csv(out/'interventions/strata.csv',index=False)
    write_json(out/'interventions/analysis.json',{'scope':'Development only, 100Hz whole-window+suffix effects',
        'analysis_config':cfg,'analysis_config_sha256':sha256(resolve('configs/followup_f2_analysis.json')),
        'states':f.state_id.nunique(),'families':f.initial_pose_family.nunique(),'candidates':len(f),
        'nominal_contact_states':f[f.nominal_contact].state_id.nunique(),
        'contact_change_candidates':int(f.contact_transition.sum()),'effect_counts':f.effect_type.value_counts().to_dict(),
        'eligible_low_pairs':int((f.method=='free_low').sum()),'matched_pairs':m.comparator.value_counts().to_dict(),
        'comparisons':summaries})
    print(json.dumps({'states':f.state_id.nunique(),'candidates':len(f),'matched_pairs':m.comparator.value_counts().to_dict()}))

if __name__=='__main__':main()
