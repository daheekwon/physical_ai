"""Separate training repetition ranges from per-model evaluation uncertainty."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('probe_stability_effects_smoke' if args.smoke else 'probe_stability_effects');audit=read(out/'raw_audit.json');a=read(out/'analysis.json');protocol=read(out/'protocol.json');resolution=read(out/'resolution.json');assert audit['smoke']==args.smoke and audit['states']==(4 if args.smoke else 402) and audit['analysis_sha256']==sha256(out/'analysis.json');assert len(protocol['models'])==180
    def meta(r):
        model=protocol['models'][r['environment']+'/'+r['representation']+'__'+r['target']+'__'+r['readout']];group=model['group'];learner=model['learner'];baselearner='mlp64' if learner.startswith('mlp64_seed_') else learner
        return {'environment':r['environment'],'split':r.get('split','calibration'),'representation':r['representation'],'target':r['target'],'readout':r['readout'],'group':group['name'],'subset_seed':group['subset_seed'],'fraction':group['fraction'],'learner':baselearner,'initialization':int(learner.rsplit('_',1)[1]) if learner.startswith('mlp64_seed_') else None}
    rows=[]
    for r in a['summary']:
        if r['phase']!='all':continue
        common=meta(r)|{'method':r['method']}
        for name,value in r.items():
            if isinstance(value,dict) and 'mean' in value:rows.append(common|{'metric':name,'value':value['mean'],'ci_low':value['ci95'][0] if value['ci95'] else None,'ci_high':value['ci95'][1] if value['ci95'] else None,'evaluation_families':value['families']})
    for r in resolution['thresholds']:
        common=meta(r)|{'method':'measurement_resolution'}
        for axis in ['position','rotation']:rows.append(common|{'metric':axis,'value':r[axis],'ci_low':None,'ci_high':None,'evaluation_families':r['families']})
    for r in a['paired_controls']:
        common=meta(r)|{'method':'predicted_minus_'+r['control']}
        for axis in ['position','rotation']:
            v=r[axis+'_difference'];rows.append(common|{'metric':axis+'_effect_error','value':v['mean'],'ci_low':v['ci95'][0] if v['ci95'] else None,'ci_high':v['ci95'][1] if v['ci95'] else None,'evaluation_families':v['families']})
    frame=pd.DataFrame(rows);frame.to_csv(out/'replicate_metrics.csv',index=False);variation=[];keys=['environment','split','representation','target','fraction','learner','method','metric']
    groups=[('training_subset',frame[frame.fraction<1]),('mlp_initialization',frame[(frame.fraction==1)&(frame.learner=='mlp64')])]
    for axis,part in groups:
        for key,g in part.groupby(keys,dropna=False):
            assert len(g)==3,(key,len(g));good=g.value.dropna();entry=dict(zip(keys,key))|{'variation_axis':axis,'replicates':3,'available':len(good),'mean':float(good.mean()) if len(good) else None,'min':float(good.min()) if len(good) else None,'max':float(good.max()) if len(good) else None}
            if key[-2].startswith('predicted_minus_'):entry.update(negative_mean_count=int((g.value<0).sum()),negative_upper_ci_count=int((g.ci_high<0).sum()))
            variation.append(entry)
    # Matched192-dimensional frame/mean comparison; each paired seed/init stays paired.
    paired=[];matched=frame[(frame.representation.isin(['lewm_frame','dino_projected_patch_mean']))&(frame.method.isin(['real_encoded','predicted']))&(frame.metric.isin(['position_effect_error','rotation_effect_error']))]
    join=['environment','split','target','fraction','learner','group','readout','method','metric']
    left=matched[matched.representation=='lewm_frame'];right=matched[matched.representation=='dino_projected_patch_mean'];merge=left.merge(right,on=join,suffixes=('_lewm','_dino'),validate='one_to_one')
    for key,g in merge.groupby(['environment','split','fraction','learner','method','metric']):
        diff=g.value_lewm-g.value_dino;paired.append(dict(zip(['environment','split','fraction','learner','method','metric'],key))|{'repetitions':len(g),'lewm_lower_mean_error_count':int((diff<0).sum()),'dino_lower_mean_error_count':int((diff>0).sum()),'mean_difference':float(diff.mean()),'min_difference':float(diff.min()),'max_difference':float(diff.max()),'note':'Training-repetition sign counts only; not paired family significance or independent repetition evidence.'})
    warnings=[]
    for name in read(BASE/'probe_stability/fit_complete.json')['record_hashes']:
        r=read(BASE/'probe_stability'/name);warnings.extend({'record':name,**w} for w in r['warnings'])
    write_json(out/'reproducibility.json',{'smoke':args.smoke,'states':audit['states'],'models':180,'variation':variation,'matched_192dim_rank_counts':paired,'training_warnings':warnings,'raw_audit_sha256':sha256(out/'raw_audit.json'),'analysis_sha256':sha256(out/'analysis.json'),'resolution_sha256':sha256(out/'resolution.json'),'catalog_sha256':sha256(out/'protocol.json'),'source_sha256':sha256(Path(__file__)),'scope':'Ranges across three overlapping subsets or three initializations, distinct from individual evaluation-family confidence intervals. No representation or seed selection. Absolute capture separate; intervention errors common candidate distribution. Direction accuracies have probe-specific resolution support and must not be compared as if identical subsets. Independent raw audit precedes this derived summary.'});print('SUMMARIZED reproducibility',len(variation),len(paired),flush=True)
if __name__=='__main__':main()
