"""Small supervised state gates vs candidate-conditioned future score.

Only existing development/calibration banks; separate gate fitting families.
"""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.output_replacement_core import fit,selections
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');OUT=BASE/'state_gate';read=lambda p:json.loads(p.read_text())

def main():
    cfg=read(resolve('configs/followup_state_gate.json'));OUT.mkdir(exist_ok=True)
    source=read(BASE/'lewm_common_effect_features/manifest.json');families=sorted({read(resolve(p))['initial_pose_family'] for p in source['states']['development']});rng=np.random.default_rng(cfg['seed']);order=rng.permutation(families);train=set(map(int,order[:cfg['training_families']]));diagnostic=set(map(int,order[cfg['training_families']:]));assert len(train)==24 and len(diagnostic)==11
    # Fixed projection drawn independently of all state features and labels.
    projection=np.linalg.qr(np.random.default_rng(cfg['seed']+1).normal(size=(192,cfg['projection_dimension'])))[0]
    inputs={};rows=[];state_rows=[];xs=[];xp=[];targets=[]
    for name in ['configs/followup_state_gate.json','scripts/run_followup_state_gate.py','scripts/output_replacement_core.py','scripts/analyze_followup_response_selection.py']:
        inputs[name]=sha256(resolve(name))
    for name in ['lewm_common_effect_features/pipeline_complete.json','lewm_common_effect_features/raw_audit.json','lewm_common_effect_features/manifest.json','response_scores/jerk_input_audit.json']:
        p=BASE/name;inputs[str(p.relative_to(resolve('.')))]=sha256(p)
    for split,names in source['states'].items():
        for name in names:
            src=resolve(name);meta=read(src);csv=BASE/'response_scores/f1'/src.with_suffix('.csv').name;features=BASE/'lewm_common_effect_features'/split/src.with_suffix('.npz').name;fm=read(features.with_suffix('.json'));assert fm['data_sha256']==sha256(features)
            for p in [src,csv,features,features.with_suffix('.json')]:inputs[str(p.relative_to(resolve('.')))]=sha256(p)
            f=pd.read_csv(csv);assert set(f.method.str.split('_').str[0])<= {'optimizer','random'}
            f=f[(f.relative_radius<=cfg['maximum_relative_radius']+1e-12)&(f.actual_norm>cfg['minimum_norm'])].copy()
            with np.load(features,allow_pickle=False) as a:current=a['history'][-1].astype(float);nominal=a['predicted'][0].astype(float)
            role='calibration' if split=='calibration' else 'gate_fit' if meta['initial_pose_family'] in train else 'diagnostic'
            sid=meta['state_id'];phase='contact' if bool(f.nominal_contact.iloc[0]) else 'free';ratio=f.object_error.to_numpy()/np.maximum(f.controller_pusher_score.to_numpy(),cfg['controller_floor']);target=np.log(max(cfg['sensitivity_floor'],np.quantile(ratio,cfg['sensitivity_quantile'])))
            i=len(state_rows);state_rows.append({'state_id':sid,'initial_pose_family':meta['initial_pose_family'],'source_split':split,'role':role,'phase':phase,'rows':len(f),'training_target_log_sensitivity':float(target) if role=='gate_fit' else None});xs.append(current@projection);xp.append(nominal@projection);targets.append(target)
            rows.append(pd.DataFrame({'state_id':sid,'initial_pose_family':meta['initial_pose_family'],'phase':phase,'role':role,'state_index':i,'candidate':f['index'].to_numpy(int),'norm':f.actual_norm.to_numpy(),'relative_radius':f.relative_radius.to_numpy(),'predicted_cost':f.predicted_jerk.to_numpy(),'predicted_gain':(f.nominal_predicted_jerk-f.predicted_jerk).to_numpy()/np.maximum(f.nominal_predicted_jerk.to_numpy(),1e-12),'dense_full':f.object_error.to_numpy(),'jerk_gain':f.jerk_gain.to_numpy(),'controller_only':f.controller_pusher_score.to_numpy(),'candidate_future':f.lewm_global_score.to_numpy()}))
    states=pd.DataFrame(state_rows);frame=pd.concat(rows,ignore_index=True);x={'current_state_gate':np.array(xs),'nominal_future_gate':np.array(xp)};y=np.array(targets);mask=(states.role=='gate_fit').to_numpy();calf=set(states.loc[states.role=='calibration','initial_pose_family']);assert not calf&train and not calf&diagnostic
    manifest={'config':cfg,'inputs':inputs,'gate_fit_families':sorted(train),'diagnostic_families':sorted(diagnostic),'calibration_families':sorted(calf),'state_ids':states.state_id.tolist(),'scope':cfg['scope']}
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==manifest
    else:write_json(OUT/'protocol.json',manifest)
    arrays={'projection':projection,'target_train':y[mask],'training_mask':mask};models={}
    for method,features in x.items():
        mu=features[mask].mean(0);sd=features[mask].std(0);sd=np.where(sd<1e-12,1.,sd);z=(features-mu)/sd;ym=float(y[mask].mean());coef=np.linalg.solve(z[mask].T@z[mask]+cfg['ridge_alpha']*np.eye(z.shape[1]),z[mask].T@(y[mask]-ym));pred=z@coef+ym;sensitivity=np.exp(np.clip(pred,-20,20));frame[method]=frame.controller_only.to_numpy()*sensitivity[frame.state_index.to_numpy(int)];states[method+'_sensitivity']=sensitivity
        for k,v in {'features':features,'mean':mu,'scale':sd,'coef':coef,'intercept':np.array(ym),'predicted_log':pred}.items():arrays[method+'__'+k]=v
        models[method]={'features':cfg['projection_dimension'],'ridge_alpha':cfg['ridge_alpha'],'training_states':int(mask.sum()),'training_families':len(train),'labels':'95th percentile dense object error / controller score per training state, then log','fit_rmse':float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))}
    np.savez_compressed(OUT/'models.npz',**arrays);states.to_csv(OUT/'states.csv',index=False);frame.to_csv(OUT/'candidates.csv',index=False);chosen=[];rules={};curves=[]
    for method in cfg['methods']:
        rule,c=fit(frame[frame.role=='calibration'],method,cfg,'pusht');rules[method]=rule;curves.extend({'method':method,**r} for r in c)
        for role in ['gate_fit','calibration','diagnostic']:
            selected=selections(frame[frame.role==role],method,rule,cfg);selected['method']=method;selected['role']=role;chosen.append(selected)
    chosen=pd.concat(chosen,ignore_index=True);chosen.to_csv(OUT/'selected.csv',index=False);pd.DataFrame(curves).to_csv(OUT/'calibration_curves.csv',index=False);write_json(OUT/'rules.json',rules);summary=[];paired=[]
    for (role,method),f in chosen.groupby(['role','method']):
        for phase in ['all','contact','free']:
            g=f if phase=='all' else f[f.phase==phase];a=g[g.active]
            summary.append({'role':role,'method':method,'phase':phase,'states':len(g),**{k:mean_ci(g,k) for k in ['active','joint','jerk_gain','predicted_gain','candidate_coverage']},'selected_risk':mean_ci(a,'violation')})
    for role,g in chosen.groupby('role'):
        for baseline in ['controller_only','current_state_gate','nominal_future_gate']:
            a=g[g.method=='candidate_future'];b=g[g.method==baseline];v=a.merge(b,on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for metric in ['joint','jerk_gain','active']:
                v['difference']=v[metric].astype(float)-v[metric+'_base'].astype(float);paired.append({'role':role,'method':'candidate_future','baseline':baseline,'metric':metric,**mean_ci(v,'difference')})
    write_json(OUT/'analysis.json',{'states':len(states),'candidate_rows':len(frame),'models':models,'summary':summary,'paired':paired,'scope':cfg['scope'],'interpretation':'Gate-fit and calibration are resubstitution. Diagnostic families excluded from gate fitting and threshold calibration but already exposed to earlier research; not independent finaltest. Future physical labels only used for gate fit/calibration/evaluation. No full inference latency measured.'})
    for name,digest in manifest['inputs'].items():assert sha256(resolve(name))==digest,name
    write_json(OUT/'execution_complete.json',{'states':len(states),'candidate_rows':len(frame),'protocol_sha256':sha256(OUT/'protocol.json'),'files':{n:sha256(OUT/n) for n in ['models.npz','states.csv','candidates.csv','selected.csv','calibration_curves.csv','rules.json','analysis.json']},'source_sha256':sha256(Path(__file__)),'unix':time.time(),'scope':'Independent audit required before reporting.'});print('STATE GATE EXECUTION COMPLETE',len(states),len(frame),flush=True)

if __name__=='__main__':main()
