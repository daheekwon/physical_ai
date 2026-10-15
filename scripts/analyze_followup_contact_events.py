"""Frozen event-conditioned capture/effect audit and common-bank selection metrics."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.probe_stability_effect_core import features,decode_features
from scripts.output_replacement_core import physical
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('contact_events_smoke' if smoke else 'contact_events');protocol=read(out/'protocol.json');audit=read(out/'raw_audit.json');assert audit['protocol_sha256']==sha256(out/'protocol.json') and audit['states']==len(protocol['states']);cfg=protocol['config'];resolution=read(BASE/'contact_events_catalogue/resolution.json')['thresholds'];models={rep:dict(np.load(BASE/'dinowm_readouts/pusht'/f'{rep}__raw__ridge.npz')) for rep in cfg['representations']};effects=[];state_effects=[];selected=[];physical_rows=[];captures=[];paired=[];inputs={}
    for spec in protocol['states']:
        p=out/(spec['state_id']+'.json');m=read(p);a=dict(np.load(p.with_suffix('.npz')));truth=physical('pusht',a);cases=[r for r in protocol['cases'] if r['state_id']==spec['state_id']];active=a['norms']>cfg['minimum_norm'];ids=np.flatnonzero(active);n=len(ids);assert n>0;save={};perrep={};true_end=a['physics'][:,79,2:5];root=a['physics'][0,29,2:5]
        for rep,model in models.items():
            x=features(rep,None,{k:a[k] for k in ['history','actual','predicted']},None);methods,decoded=decode_features('pusht','raw',model,x,a,cfg,a['input_scale']);save.update({rep+'__'+k:v for k,v in decoded.items()});perrep[rep]=methods;tp=np.linalg.norm(decoded['true_dp'],axis=1);tr=abs(decoded['true_dr'][:,0]);current=decoded['decoded_current'][0];cap={'current_position_error':float(np.linalg.norm(current[:2]-root[:2])),'current_rotation_error':float(abs(np.arctan2(np.sin(np.arctan2(current[2],current[3])-root[2]),np.cos(np.arctan2(current[2],current[3])-root[2]))))}
            for case in cases:
                common={k:case[k] for k in ['event_id','event_type','role','state_id']};common.update(initial_pose_family=spec['seed'],representation=rep)
                captures.append({**common,**cap})
                for method,values in methods.items():
                    f=pd.DataFrame({**common,'method':method,'candidate':ids,'true_position_magnitude':tp[ids],'true_rotation_magnitude':tr[ids],'physical_E':truth['dense_full'][ids],'jerk_gain':truth['jerk_gain'][ids],**{k:v[ids] for k,v in values.items()}});effects.append(f);good=np.isfinite(f.position_effect_error)&np.isfinite(f.rotation_effect_error);g=f[good];row={**common,'method':method,'active_candidates':n,'valid_candidates':len(g),'control_support':len(g)/n}
                    for axis in ['position','rotation']:
                        resolved=g['true_'+axis+'_magnitude']>resolution[rep][axis];finite=np.isfinite(g[axis+'_cosine']);direction=resolved&finite
                        row[axis+'_effect_error']=float(g[axis+'_effect_error'].mean()) if len(g) else np.nan;row[axis+'_absolute_error']=float(g[axis+'_absolute_error'].mean()) if len(g) else np.nan;row[axis+'_resolvable_fraction']=float(resolved.mean()) if len(g) else np.nan;row[axis+'_direction_support']=float(direction.mean()) if len(g) else np.nan;row[axis+'_direction_agreement']=float((g.loc[direction,axis+'_cosine']>=0).mean()) if direction.any() else np.nan
                    state_effects.append(row)
                for control in ['real_encoded','zero_effect','persistence','wrong_action']:
                    mask=active & np.isfinite(methods[control]['position_effect_error']) & np.isfinite(methods[control]['rotation_effect_error']);paired.append({**common,'control':control,'matched_candidates':int(mask.sum()),**{axis+'_difference':float((methods['predicted'][axis+'_effect_error'][mask]-methods[control][axis+'_effect_error'][mask]).mean()) if mask.any() else np.nan for axis in ['position','rotation']}})
        np.savez_compressed(p.with_suffix('.decoded.npz'),**save)
        for case in cases:
            common={k:case[k] for k in ['event_id','event_type','role','state_id']};common['initial_pose_family']=spec['seed'];contact=a['contacts']>0;both=contact.any(1)&contact[0].any();first=np.array([(np.flatnonzero(c)[0]+1)*.01 if c.any() else np.nan for c in contact]);offset=abs(first-first[0]);q=a['physics'][:,30:,:2]-a['physics'][:1,30:,:2];has_useful=bool(np.any(active&(truth['dense_full']<=1)&(truth['jerk_gain']>=.1)));physical_rows.append({**common,'active_candidates':n,'preserved_fraction':float((truth['dense_full'][ids]<=1).mean()),'useful_fraction':float(((truth['dense_full'][ids]<=1)&(truth['jerk_gain'][ids]>=.1)).mean()),'has_useful_candidate':has_useful,'contact_mask_disagreement':float((contact[ids]!=contact[:1]).mean()),'contact_existence_flip_fraction':float((contact[ids].any(1)!=contact[0].any()).mean()),'first_contact_shift_mean_seconds':float(np.nanmean(offset[ids])) if np.isfinite(offset[ids]).any() else np.nan,'first_contact_shift_support':float(both[ids].mean()),'pusher_delta_rms':float(np.sqrt(np.mean(np.sum(q[ids]*q[ids],axis=2),axis=1)).mean())})
            for method in cfg['methods']:
                i=m['selected_before_physics'][method];selected.append({**common,'method':method,'candidate':i,'active':bool(active[i]),'object_error':truth['dense_full'][i],'violation':bool(truth['dense_full'][i]>1),'jerk_gain':truth['jerk_gain'][i],'joint':bool(active[i] and truth['dense_full'][i]<=1 and truth['jerk_gain'][i]>=.1)})
        for ext in ['.json','.npz','.csv','.decision.json','.decoded.npz']:inputs[str(p.with_suffix(ext).relative_to(resolve('.')))]=sha256(p.with_suffix(ext))
    frames={'effects':pd.concat(effects,ignore_index=True),'state_effects':pd.DataFrame(state_effects),'selected':pd.DataFrame(selected),'physical_states':pd.DataFrame(physical_rows),'capture_states':pd.DataFrame(captures),'paired_effects':pd.DataFrame(paired)}
    for name,f in frames.items():f.to_csv(out/(name+('.csv.gz' if name=='effects' else '.csv')),index=False)
    result={'smoke':smoke,'states':len(protocol['states']),'cases':len(protocol['cases']),'families':len({r['seed'] for r in protocol['states']}),'candidates':audit['candidates'],'effect_rows':len(frames['effects']),'selection_rows':len(frames['selected']),'resolution':resolution,'scope':cfg['scope']}
    definitions={'selection_summary':('selected',['event_type','role','method'],['active','joint','jerk_gain']),'physical_summary':('physical_states',['event_type','role'],['has_useful_candidate','preserved_fraction','useful_fraction','contact_mask_disagreement','contact_existence_flip_fraction','first_contact_shift_mean_seconds','first_contact_shift_support','pusher_delta_rms']),'capture_summary':('capture_states',['event_type','role','representation'],['current_position_error','current_rotation_error']),'effect_summary':('state_effects',['event_type','role','representation','method'],['control_support']+[axis+'_'+metric for axis in ['position','rotation'] for metric in ['effect_error','absolute_error','resolvable_fraction','direction_support','direction_agreement']]),'paired_effect_summary':('paired_effects',['event_type','role','representation','control'],['position_difference','rotation_difference'])}
    for name,(table,keys,metrics) in definitions.items():
        result[name]=[]
        for group,g in frames[table].groupby(keys,sort=True):
            row={**dict(zip(keys,group)),'cases':len(g),**{metric:mean_ci(g.dropna(subset=[metric]),metric) for metric in metrics}}
            if table=='selected':row['selected_risk']=mean_ci(g[g.active],'violation')
            result[name].append(row)
    phase_pairs=[]
    for rep in cfg['representations']:
        for method in ['real_encoded','predicted']:
            f=frames['state_effects'];f=f[(f.representation==rep)&(f.method==method)]
            for reference in ['pre','post','stable_free','stable_contact']:
                g=f[f.role=='transition'].merge(f[f.role==reference],on=['event_id','event_type','initial_pose_family'],suffixes=('','_reference'),validate='one_to_one')
                for event_type,h in g.groupby('event_type'):
                    for metric in ['position_effect_error','rotation_effect_error']:
                        h=h.copy();h['delta']=h[metric]-h[metric+'_reference'];phase_pairs.append({'event_type':event_type,'representation':rep,'method':method,'reference':reference,'metric':metric,**mean_ci(h.dropna(subset=['delta']),'delta')})
    result['paired_phase_errors']=phase_pairs;write_json(out/'analysis.json',result);write_json(out/'analysis_complete.json',{'smoke':smoke,'states':result['states'],'inputs':inputs,'source_sha256':sha256(Path(__file__)),'analysis_sha256':sha256(out/'analysis.json'),'raw_audit_sha256':sha256(out/'raw_audit.json'),'helpers':{p:sha256(resolve(p)) for p in ['scripts/probe_stability_effect_core.py','scripts/retrained_dinowm_effect_readout_core.py','scripts/retrained_dinowm_readout_core.py','scripts/analyze_followup_response_selection.py']}});print('ANALYZED CONTACT EVENTS',result['states'],result['effect_rows'],result['selection_rows'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
