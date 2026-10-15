"""F2 pusher-RMS matching through controller amplitude normalization, no outcome selection."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import DenseSimulator,controller_rollout,dense_metrics
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_interventions import bootstrap_pairs


def main():
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output']);dest=out/'matched_amplitudes';dest.mkdir(exist_ok=True)
    design={'low_relative_radius':.1,'maximum_comparator_norm':.4*np.sqrt(10),'match':'whole-window 100Hz pusher RMS, computed from known controller only',
            'comparators':['controller_0','controller_1','controller_2','controller_3','frequency_0','frequency_1','frequency_2'],
            'signs':[-1,1],'bounds':'No clipping; report unmatched if target RMS needs out-of-bounds action or excessive norm'}
    if (dest/'design.json').exists():assert json.loads((dest/'design.json').read_text())==design
    else:write_json(dest/'design.json',design)
    data=ExpertDataset(cfg['dataset']);m=json.loads((out/'protocol/manifest.json').read_text());allrows=[]
    for spec in [s for s in m['states'] if s['split']=='development']:
      key=spec['state_id'];done=dest/f'{key}.json'
      if not done.exists():
        started=time.perf_counter();base=out/'interventions'/f'{key}_h05.npz'
        if not base.with_suffix('.json').exists():continue
        with np.load(base) as z:
            directions=dict(zip(z['direction_names'].tolist(),z['directions']))
        frame=pd.read_csv(base.with_suffix('.csv'));a,s=data.episode(spec['episode']);t=spec['timestep'];nom=a[t:t+5];suffix=a[t+5:t+8]
        sim=DenseSimulator(a,s,t,cfg['seed']+spec['episode']);ref=sim.execute_dense(nom,suffix);root=ref['root_full_state']
        origin=controller_rollout(np.concatenate([nom,suffix]),root[:2],root[5:7]);rows=[];traces=[ref['physics_trace']];raws=[nom]
        for sign in design['signs']:
          low=frame[(frame.method=='free_low')&(frame.relative_radius==.1)&(frame.sign==sign)].iloc[0]
          target=float(low.pusher_rms)
          for name in design['comparators']:
            unit=(sign*directions[name]).reshape(5,2)*data.scaler.scale_
            moved=controller_rollout(np.concatenate([nom+unit,suffix]),root[:2],root[5:7])
            response_rms=float(np.sqrt(np.mean(np.sum((moved-origin)**2,axis=1))))
            norm=target/max(response_rms,1e-12);raw=nom+norm*unit
            row={**spec,'comparator':name,'sign':sign,'low_index':int(low['index']),
                 'low_norm':float(low.actual_norm),'comparator_norm':float(norm),'target_pusher_rms':target,
                 'low_object_error':float(low.object_error),'low_position_max':float(low.position_max),
                 'nominal_contact':bool(low.nominal_contact)}
            if norm>design['maximum_comparator_norm'] or np.max(np.abs(raw))>1+1e-7:
                rows.append({**row,'matched':False,'reason':'norm_or_action_bound'});continue
            result=sim.execute_dense(raw,suffix);metrics=dense_metrics(result,ref,cfg['tolerances'])
            mismatch=abs(metrics['pusher_rms']-target)/max(target,1e-8)
            if mismatch>1e-3:raise AssertionError(('Amplitude matching mismatch',key,name,mismatch))
            rows.append({**row,'matched':True,'reason':'ok','index':len(traces),'relative_rms_error':mismatch,
                         'peak_ratio':metrics['pusher_max']/max(float(low.pusher_max),1e-8),
                         'frequency_centroid':float(frame[frame.method==name].frequency_centroid.iloc[0]),
                         'low_frequency_centroid':float(low.frequency_centroid),
                         'low_minus_object_error':float(low.object_error-metrics['object_error']),
                         'low_minus_position':float(low.position_max-metrics['position_max']),**metrics})
            traces.append(result['physics_trace']);raws.append(raw)
        pd.DataFrame(rows).to_csv(dest/f'{key}.csv',index=False)
        np.savez_compressed(dest/f'{key}.npz',physics=traces,actions=raws)
        write_json(done,{**spec,'rows':len(rows),'matched':sum(r['matched'] for r in rows),'duration_seconds':time.perf_counter()-started,'source_sha256':sha256(Path(__file__))})
        sim.close();print(key,flush=True)
      allrows.append(pd.read_csv(dest/f'{key}.csv'))
    f=pd.concat(allrows,ignore_index=True);f.to_csv(dest/'candidates.csv',index=False)
    f['group']=f.comparator.str.split('_').str[0];summaries=[]
    for name,g in f.groupby('group'):
      for phase,k in [('all',g),('contact',g[g.nominal_contact])]:
       good=k[k.matched]
       for strict,v in [('rms_only',good),('rms_peak_frequency',good[(good.peak_ratio.between(.9,1.1))&((good.frequency_centroid-good.low_frequency_centroid).abs()<=.25)])]:
        if not len(v):continue
        summaries.append({'group':name,'phase':phase,'matching':strict,'attempts':len(k),'matches':len(v),
                          'families':v.initial_pose_family.nunique(),'low_minus_object_error':float(v.low_minus_object_error.mean()),
                          'ci':bootstrap_pairs(v,'low_minus_object_error'),'mean_norm_ratio':float((v.comparator_norm/v.low_norm).mean())})
    write_json(dest/'summary.json',{'design':design,'states':f.state_id.nunique(),'attempts':len(f),'matched':int(f.matched.sum()),'comparisons':summaries})
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
