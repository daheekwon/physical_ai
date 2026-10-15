"""Audit raw physics, causal features, independent ridge fit and gate selection."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset
from scripts.analyze_followup_final_expert import Coverage
from scripts.audit_followup_output_replacement import selected,pd_positions
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');OUT=BASE/'state_gate';read=lambda p:json.loads(p.read_text())

def main():
    protocol=read(OUT/'protocol.json');cfg=protocol['config'];done=read(OUT/'execution_complete.json');assert done['protocol_sha256']==sha256(OUT/'protocol.json') and done['states']==210
    for name,digest in protocol['inputs'].items():assert sha256(resolve(name))==digest,name
    for name,digest in done['files'].items():assert sha256(OUT/name)==digest,name
    errors={}
    def check(name,a,b,tol=1e-8,rtol=0.):
        x=np.asarray(a,float);y=np.asarray(b,float);assert x.shape==y.shape,(name,x.shape,y.shape);assert np.array_equal(np.isnan(x),np.isnan(y)),(name,x,y);np.testing.assert_allclose(x,y,atol=tol,rtol=rtol,equal_nan=True,err_msg=name);v=abs(x-y);errors[name]=max(errors.get(name,0.),float(np.nanmax(v)) if v.size and np.isfinite(v).any() else 0.)
    state=pd.read_csv(OUT/'states.csv');frame=pd.read_csv(OUT/'candidates.csv');output=pd.read_csv(OUT/'selected.csv');curves=pd.read_csv(OUT/'calibration_curves.csv');rules=read(OUT/'rules.json');analysis=read(OUT/'analysis.json')
    with np.load(OUT/'models.npz',allow_pickle=False) as z:arrays=dict(z)
    maincfg=read(resolve('configs/followup_v1.json'));data=ExpertDataset(maincfg['dataset']);source=read(BASE/'lewm_common_effect_features/manifest.json');rawinputs={};families=sorted(state.loc[state.source_split=='development','initial_pose_family'].unique());order=np.random.default_rng(cfg['seed']).permutation(families);train=set(map(int,order[:24]));diag=set(map(int,order[24:]));cal=set(state.loc[state.source_split=='calibration','initial_pose_family']);assert sorted(train)==protocol['gate_fit_families'] and sorted(diag)==protocol['diagnostic_families'] and sorted(cal)==protocol['calibration_families'];assert not train&diag and not train&cal and not diag&cal
    q=np.linalg.qr(np.random.default_rng(cfg['seed']+1).normal(size=(192,8)))[0];check('fixed_projection',q,arrays['projection'],0);check('orthonormal',q.T@q,np.eye(8),1e-12);targets=[];actual_features={k:[] for k in ['current_state_gate','nominal_future_gate']}
    paths={read(resolve(name))['state_id']:resolve(name) for names in source['states'].values() for name in names}
    for i,s in state.iterrows():
        p=paths[s.state_id];expected='calibration' if s.source_split=='calibration' else 'gate_fit' if s.initial_pose_family in train else 'diagnostic';assert s.role==expected
        meta=read(p);npz=p.with_suffix('.npz');assert sha256(npz)==source['inputs'][str(npz.relative_to(resolve('.')))];rawinputs[str(npz.relative_to(resolve('.')))]=sha256(npz)
        with np.load(npz,allow_pickle=False) as z:a=dict(z)
        features=BASE/'lewm_common_effect_features'/s.source_split/p.with_suffix('.npz').name
        with np.load(features,allow_pickle=False) as z:fv=dict(z)
        actual_features['current_state_gate'].append(np.asarray(fv['history'][-1],float)@q);actual_features['nominal_future_gate'].append(np.asarray(fv['predicted'][0],float)@q)
        f=frame[frame.state_id==s.state_id];assert len(f)==s.rows and (f.state_index==i).all();ids=f.candidate.to_numpy(int);assert (f.relative_radius<=.2+1e-12).all()
        original=pd.read_csv(BASE/'response_scores/f1'/p.with_suffix('.csv').name);original=original[(original.relative_radius<=.2+1e-12)&(original.actual_norm>cfg['minimum_norm'])];np.testing.assert_array_equal(ids,original['index']);assert set(original.method.str.split('_').str[0])<= {'optimizer','random'}
        check('applied_norm',np.linalg.norm(((a['actions'][ids]-a['actions'][:1])/data.scaler.scale_).reshape(len(ids),-1),axis=1),f.norm,1e-9)
        physics=a['physics'];delta=physics[:,30:,2:5]-physics[:1,30:,2:5];position=np.linalg.norm(delta[:,:,:2],axis=2).max(1)/2;angle=np.abs((delta[:,:,2]+np.pi)%(2*np.pi)-np.pi).max(1)/np.deg2rad(2);cov=Coverage(data,meta,maincfg);coverage=np.array([[cov(trace[79]),cov(trace[-1])] for trace in physics]);check('independent_polygon_coverage',coverage,a['coverage'],1e-10);obj=np.maximum.reduce([position,angle,np.maximum(0,coverage[:1]-coverage).max(1)/.02]);jerk=np.sum(np.diff(physics[:,:,:2],n=3,axis=1)**2,axis=(1,2))/.01**5;gain=(jerk[0]-jerk)/jerk[0] if jerk[0]>1e-8 else np.zeros(len(jerk));check('dense_object_error',obj[ids],f.dense_full,1e-9);check('executed_jerk_gain',gain[ids],f.jerk_gain,1e-10)
        ep,_=data.episode(meta['episode']);suffix=ep[meta['timestep']+5:meta['timestep']+8];pred=np.array([pd_positions(np.r_[act,suffix],a['root']) for act in a['actions']]);ctrl=np.linalg.norm(pred-pred[:1],axis=2).max(1);cost=np.sum(np.diff(np.concatenate([np.broadcast_to(physics[0,:30,:2],(len(pred),30,2)),pred],axis=1),n=3,axis=1)**2,axis=(1,2))/.01**5;check('controller_score',ctrl[ids],f.controller_only,1e-9);check('causal_predicted_cost',cost[ids],f.predicted_cost,1e-5,1e-10);check('causal_predicted_gain',(cost[0]-cost[ids])/max(cost[0],1e-12),f.predicted_gain,1e-9)
        latent=fv['predicted'];score=np.linalg.norm(latent-latent[:1],axis=1);check('native_candidate_score_CPU_GPU_parity',score[ids],f.candidate_future,2e-5,1e-6)
        # Verify exports first, then restore original arithmetic for exact threshold membership.
        # CSV round trips are not an interchangeable source for <= at a fitted quantile.
        ix=f.index
        source_values={'norm':original.actual_norm.to_numpy(),'predicted_cost':original.predicted_jerk.to_numpy(),'predicted_gain':(original.nominal_predicted_jerk-original.predicted_jerk).to_numpy()/np.maximum(original.nominal_predicted_jerk.to_numpy(),1e-12),'dense_full':original.object_error.to_numpy(),'jerk_gain':original.jerk_gain.to_numpy(),'controller_only':original.controller_pusher_score.to_numpy(),'candidate_future':original.lewm_global_score.to_numpy()}
        for column,values in source_values.items():
            check('source_export_'+column,values,frame.loc[ix,column],1e-5 if column=='predicted_cost' else 1e-10)
            frame.loc[ix,column]=values
        if s.role=='gate_fit':
            target=np.log(max(cfg['sensitivity_floor'],np.quantile(obj[ids]/np.maximum(ctrl[ids],cfg['controller_floor']),.95)));targets.append(target);check('training_label',target,s.training_target_log_sensitivity,1e-8)
        else:assert pd.isna(s.training_target_log_sensitivity)
    mask=(state.role=='gate_fit').to_numpy();np.testing.assert_array_equal(mask,arrays['training_mask']);assert int(mask.sum())==72;check('all_training_labels',targets,arrays['target_train'],1e-8)
    for method,feat in actual_features.items():
        x=np.array(feat);check('raw_features',x,arrays[method+'__features'],1e-12);scaler=StandardScaler().fit(x[mask]);model=Ridge(alpha=10.).fit(scaler.transform(x[mask]),np.array(targets));log=model.predict(scaler.transform(x));check('refit_scaler',scaler.mean_,arrays[method+'__mean'],1e-12);check('refit_scale',scaler.scale_,arrays[method+'__scale'],1e-12);check('refit_coefficients',model.coef_,arrays[method+'__coef'],1e-8);check('refit_intercept',model.intercept_,arrays[method+'__intercept'],1e-8);check('refit_inference',log,arrays[method+'__predicted_log'],1e-8);sensitivity=np.exp(np.clip(log,-20,20));check('state_sensitivity',sensitivity,state[method+'_sensitivity'],1e-8);check('gate_score',frame.controller_only.to_numpy()*sensitivity[frame.state_index.to_numpy(int)],frame[method],1e-8)
    # Fresh inference from the independently checked original fit, preserving its exact
    # floating-point score values at quantile cutoffs. The separate sklearn refit above
    # establishes fit correctness, but its tiny numerical differences must not redefine choices.
    for method in actual_features:
        x=np.array(actual_features[method]);log=(x-arrays[method+'__mean'])/arrays[method+'__scale']@arrays[method+'__coef']+arrays[method+'__intercept'];sensitivity=np.exp(np.clip(log,-20,20))
        frame[method]=frame.controller_only.to_numpy()*sensitivity[frame.state_index.to_numpy(int)]
    reconstructed=[];trials=0
    for method in cfg['methods']:
        calframe=frame[frame.role=='calibration'];grid=np.unique(np.r_[0.,1.,np.quantile(calframe[method],np.linspace(0,1,41))]);recorded=curves[curves.method==method].reset_index(drop=True);check('threshold_grid',grid,recorded.tau,1e-10);best=None;value=None
        for i,tau in enumerate(grid):
            r=selected(calframe,method,{'enabled':True,'tau':float(tau)},cfg);act=r[r.active];contact=act[act.phase=='contact'];risk=float(act.violation.mean()) if len(act) else None;crisk=float(contact.violation.mean()) if len(contact) else None;active=float(r.active.mean());v=(float((r.active&(r.predicted_gain>=.1)).mean()),float(r.predicted_gain.mean()));valid=active>=.25 and risk is not None and risk<=.1 and crisk is not None and crisk<=.1;saved=recorded.iloc[i]
            for key,x in [('risk',risk),('contact_risk',crisk),('active_fraction',active),('predicted_useful',v[0]),('mean_predicted_gain',v[1])]:check('calibration_'+key,np.nan if x is None else x,saved[key],1e-10)
            assert bool(saved.valid)==valid
            if valid and (value is None or v>value):best={'enabled':True,'tau':float(tau)};value=v
            trials+=1
        rule=best or {'enabled':False,'tau':None};assert rule['enabled']==rules[method]['enabled']
        if rule['enabled']:check('chosen_threshold',rule['tau'],rules[method]['tau'],1e-10)
        for role in ['gate_fit','calibration','diagnostic']:
            r=selected(frame[frame.role==role],method,rule,cfg);r['method']=method;r['role']=role;reconstructed.append(r)
    rebuilt=pd.concat(reconstructed,ignore_index=True);keys=['role','method','state_id'];rebuilt=rebuilt.sort_values(keys).reset_index(drop=True);output=output.sort_values(keys).reset_index(drop=True);assert len(rebuilt)==len(output)==840
    for col in rebuilt:
        if col in keys+['phase']:np.testing.assert_array_equal(rebuilt[col],output[col])
        else:check('selected_'+col,rebuilt[col],output[col],1e-10)
    for r in analysis['summary']:
        g=rebuilt[(rebuilt.role==r['role'])&(rebuilt.method==r['method'])];g=g if r['phase']=='all' else g[g.phase==r['phase']];assert len(g)==r['states']
        for metric in ['active','joint','jerk_gain','predicted_gain','candidate_coverage','selected_risk']:
            v=g[g.active] if metric=='selected_risk' else g;column='violation' if metric=='selected_risk' else metric
            if not len(v):assert r[metric]['mean'] is None;continue
            avg,ci=cluster(v[column].to_numpy(),v.initial_pose_family.to_numpy());check('summary_mean',avg,r[metric]['mean'],1e-10);check('summary_ci',ci,r[metric]['ci95'],1e-10)
    for r in analysis['paired']:
        g=rebuilt[rebuilt.role==r['role']];a=g[g.method==r['method']].set_index('state_id');b=g[g.method==r['baseline']].set_index('state_id').loc[a.index];d=a[r['metric']].astype(float)-b[r['metric']].astype(float);avg,ci=cluster(d.to_numpy(),a.initial_pose_family.to_numpy());check('paired_mean',avg,r['mean'],1e-10);check('paired_ci',ci,r['ci95'],1e-10)
    write_json(OUT/'raw_audit.json',{'states':len(state),'gate_fit_states':int(mask.sum()),'gate_fit_families':len(train),'calibration_families':len(cal),'diagnostic_families':len(diag),'candidate_rows':len(frame),'selected_rows':len(rebuilt),'calibration_trials':trials,'max_errors':errors,'source_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(OUT/'protocol.json'),'analysis_sha256':sha256(OUT/'analysis.json'),'raw_inputs':rawinputs,'audit_helpers':{p:sha256(resolve(p)) for p in ['scripts/analyze_followup_final_expert.py','scripts/audit_followup_output_replacement.py','scripts/audit_retrained_dinowm_readouts.py']},'scope':'Independent rawphysics/coverage/norm/controllercost, native score parity, train-only features/labels, sklearn ridge refit, all calibration thresholds/selection and family intervals. Existing diagnostic data not fresh finaltest. No end-to-end runtime.'});print('AUDITED STATE GATE',len(state),len(frame),trials,len(rebuilt),flush=True)

if __name__=='__main__':main()
