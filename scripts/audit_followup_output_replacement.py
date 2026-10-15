"""Independent feature, physical-score, calibration and selection reproduction."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset
from scripts.analyze_followup_final_expert import Coverage
from scripts.audit_retrained_dinowm_readouts import infer,cluster
from scripts.audit_retrained_dinowm_effect_readouts import features as token_features,pose,delta
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def selected(frame,level,rule,cfg):
    rows=[]
    for sid,g in frame.groupby('state_id',sort=True):
        active=g[g.norm>cfg['minimum_norm']];accepted=active[active[level]<=rule['tau']] if rule['enabled'] else active.iloc[:0];eligible=accepted[accepted.predicted_gain>cfg['minimum_predicted_gain']]
        r=eligible.sort_values(['predicted_cost','norm','candidate'],kind='stable').iloc[0] if len(eligible) else None
        rows.append({'state_id':sid,'initial_pose_family':int(g.initial_pose_family.iloc[0]),'phase':g.phase.iloc[0],'candidate':int(r.candidate) if r is not None else 0,'active':r is not None,'predicted_gain':float(r.predicted_gain) if r is not None else 0.,'jerk_gain':float(r.jerk_gain) if r is not None else 0.,'object_error':float(r.dense_full) if r is not None else 0.,'violation':bool(r is not None and r.dense_full>1),'joint':bool(r is not None and r.dense_full<=1 and r.jerk_gain>=cfg['meaningful_gain']),'candidate_coverage':len(accepted)/len(active) if len(active) else 0.,'candidate_violation_mass':float((accepted.dense_full>1).sum()/len(active)) if len(active) else 0.,'approved_count':len(accepted),'active_candidates':len(active)})
    return pd.DataFrame(rows)

def pd_positions(actions,root):
    pos=np.array(root[:2],float);vel=np.array(root[5:7],float);points=[]
    for a in actions:
        goal=pos+100*a
        for _ in range(10):vel+=.01*(100*(goal-pos)-20*vel);pos+=.01*vel;points.append(pos.copy())
    return np.array(points)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['lewm','dino']);ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('output_replacement_'+args.family+('_smoke' if args.smoke else ''));protocol=read(out/'protocol.json');cfg=protocol['config'];done=read(out/'analysis_complete.json');assert done['family']==args.family and done['smoke']==args.smoke and done['analysis_sha256']==sha256(out/'analysis.json') and done['rules_sha256']==sha256(out/'rules.json')
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    errors={};data=ExpertDataset(read(resolve('configs/followup_v1.json'))['dataset']);maincfg=read(resolve('configs/followup_v1.json'));projection=np.load(BASE/'dinowm_readouts/projection.npy');cube_scale=np.array(read(BASE/'cube/manifest.json')['normalization']['scale']);records=0;decoded_count=0;frames=[]
    def check(name,x,y,tol=1e-8,rtol=0.):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);assert x.shape==y.shape,(name,x.shape,y.shape);assert np.array_equal(np.isnan(x),np.isnan(y)),name;finite=np.isfinite(x)&np.isfinite(y);err=float(abs(x[finite]-y[finite]).max()) if finite.any() else 0.;errors[name]=max(errors.get(name,0.),err);assert np.all(abs(x[finite]-y[finite])<=tol+rtol*abs(y[finite])),(name,err)
    for mp in sorted(out.glob('*/*/*.json')):
        meta=read(mp);src=resolve(meta['source']);env=meta['environment'];old=pd.read_csv(src.with_suffix('.csv'));frame=pd.read_csv(mp.with_suffix('.csv'));frames.append(frame);records+=1;assert len(frame)==meta['rows']
        assert meta['protocol_sha256']==sha256(out/'protocol.json') and meta['feature_sha256']==sha256(resolve(meta['feature_source']))
        for p,h in meta['files'].items():assert sha256(mp.parent/p)==h,p
        with np.load(src.with_suffix('.npz'),allow_pickle=False) as z:a=dict(z)
        with np.load(resolve(meta['feature_source']),allow_pickle=False) as z:fv=dict(z)
        with np.load(mp.with_suffix('.npz'),allow_pickle=False) as z:saved=dict(z)
        ids=old['index' if env=='pusht' else 'candidate'].to_numpy(int)
        if env=='pusht':
            p=a['physics'];d=p[:,30:,2:5]-p[:1,30:,2:5];pos=np.sqrt(np.sum(d[:,:,:2]**2,axis=2))/2;rot=abs((d[:,:,2]+np.pi)%(2*np.pi)-np.pi)/np.deg2rad(2);dense=np.maximum(pos.max(1),rot.max(1));endpoint=np.maximum(pos[:,49],rot[:,49]);cov=Coverage(data,read(src),maincfg);coverage=np.array([[cov(x[79]),cov(x[-1])] for x in p]);check('independent_coverage',coverage,a['coverage'],1e-10);full=np.maximum(dense,np.maximum(0,coverage[:1]-coverage).max(1)/.02);points=p[:,:,:2];dt=.01
            sm=read(src);aa,_=data.episode(sm['episode']);suffix=aa[sm['timestep']+5:sm['timestep']+8];pred=np.array([pd_positions(np.r_[action,suffix],a['root']) for action in a['actions']]);prefix=p[:,:30,:2];predpoints=np.concatenate([prefix,pred],axis=1);predcost=np.sum(np.diff(predpoints,n=3,axis=1)**2,axis=(1,2))/.01**5;norm=np.linalg.norm(((a['actions']-a['actions'][:1])/data.scaler.scale_).reshape(len(p),-1),axis=1)
        else:
            p=a['candidate_object_pose'];pos=np.sqrt(np.sum((p[:,:,:3]-p[:1,:,:3])**2,axis=-1))/.002;q=p[:,:,3:]/np.linalg.norm(p[:,:,3:],axis=-1,keepdims=True);dots=np.clip(abs(np.einsum('ntk,tk->nt',q,q[0])),0,1);rot=2*np.arctan2(np.sqrt(np.maximum(0,1-dots*dots)),dots)/np.deg2rad(2);dense=np.maximum(pos.max(1),rot.max(1));endpoint=np.maximum(pos[:,124],rot[:,124]);full=dense;dt=.002;points=np.concatenate([np.broadcast_to(a['prefix_eef_position'],(len(p),75,3)),a['candidate_eef_position']],axis=1)
            # Reconstruct all predicted costs independently from stored causal surrogate traces.
            pp=np.concatenate([np.broadcast_to(a['past_points'][-75:],(len(p),75,4)),a['predicted_robot']],axis=1).copy();pp[:,:,3]*=.05;predcost=np.sum(np.diff(pp,n=3,axis=1)**2,axis=(1,2))/.002**5;norm=np.linalg.norm(((a['actions']-a['actions'][:1])/cube_scale).reshape(len(p),-1),axis=1)
        jerk=np.sum((points[:,3:]-3*points[:,2:-1]+3*points[:,1:-2]-points[:,:-3])**2,axis=(1,2))/dt**5;gain=(jerk[0]-jerk)/jerk[0] if jerk[0]>1e-8 else np.zeros(len(jerk));assert np.all(endpoint<=dense+2e-5) and np.all(dense<=full+2e-5)
        check('parent_physical_E',full[ids],old.object_error,2e-5);check('parent_actual_gain',gain[ids],old.jerk_gain,2e-8)
        for (rep,kind),f in frame.groupby(['representation','target'],sort=True):
            key=rep+'__'+kind;np.testing.assert_array_equal(f.candidate,ids)
            with np.load(BASE/'dinowm_readouts'/env/(key+'__ridge.npz'),allow_pickle=False) as z:model=dict(z)
            for level,field in [('real_encoded','real'),('predicted','predicted')]:
                if args.family=='dino':x=token_features(fv['actual' if field=='real' else 'predicted'],fv['history'],rep,projection)
                else:
                    h=fv['history'] if env=='pusht' else fv['history_embedding'];x=fv['actual' if field=='real' else 'predicted'] if env=='pusht' else fv['actual_endpoint_latent' if field=='real' else 'predicted_latent'];x=x.astype(float)
                    if rep=='lewm_history':x=np.concatenate([np.tile(h[1:].reshape(1,-1),(len(x),1)),x],axis=1)
                decoded=infer(model,x);check('independent_decoded',decoded,saved[key+'__'+field],1e-8);decoded_count+=1;dp,dr=delta(env,pose(env,decoded,kind));score=np.maximum(np.linalg.norm(dp,axis=1)/(2 if env=='pusht' else .002),np.linalg.norm(dr,axis=1)/np.deg2rad(2));check('independent_'+level,score[ids],f[level],2e-6)
            for name,raw in [('true_endpoint',endpoint),('dense_pose',dense),('dense_full',full),('actual_jerk',jerk),('jerk_gain',gain),('norm',norm),('predicted_cost',predcost)]:check(name,raw[ids],f[name],2e-5 if name not in ['predicted_cost','actual_jerk'] else 1e-2,1e-8)
            check('nominal_predicted_cost',np.repeat(predcost[0],len(f)),f.nominal_predicted_cost,1e-2,1e-8);check('expected_gain',(predcost[0]-predcost[ids])/max(predcost[0],1e-12),f.predicted_gain,1e-8)
        print('AUDITED RAW',env,src.stem,flush=True)
    assert records==(4 if args.smoke else 402)
    allrows=pd.concat(frames,ignore_index=True);rules=read(out/'rules.json')['rules'];curves=pd.read_csv(out/'calibration_curves.csv');output=pd.read_csv(out/'selected.csv');rulemap={(r['environment'],r['representation'],r['target'],r['mode'],r['level']):r['rule'] for r in rules};reconstructed=[];trial_count=0
    for key,f in allrows.groupby(['environment','representation','target'],sort=True):
        env,rep,kind=key;cal=f[f.split=='calibration'];best_rules={}
        for level in cfg['levels']:
            finite=cal[(cal.norm>cfg['minimum_norm'])&np.isfinite(cal[level])][level].to_numpy();grid=sorted(set([0.,1.,*np.quantile(finite,np.linspace(0,1,cfg['threshold_quantiles']))])) if len(finite) else [0.,1.];table=curves[(curves.environment==env)&(curves.representation==rep)&(curves.target==kind)&(curves.level==level)];check('threshold_grid',grid,table.tau.to_numpy(),1e-10);best=None;best_value=None
            for tau,record in zip(grid,table.itertuples()):
                v=selected(cal,level,{'enabled':True,'tau':tau},cfg);active=v[v.active];contact=active[active.phase=='contact'];risk=float(active.violation.mean()) if len(active) else None;crisk=float(contact.violation.mean()) if len(contact) else None;useful=float((v.active&(v.predicted_gain>=cfg['meaningful_gain'])).mean());avg=float(v.predicted_gain.mean());valid=len(active)/len(v)>=cfg['minimum_active_fraction'] and risk is not None and risk<=cfg['target_risk']
                if env=='pusht' and cfg['enforce_pusht_contact_risk']:valid=valid and crisk is not None and crisk<=cfg['target_risk']
                for name,value in [('active_fraction',float(v.active.mean())),('risk',risk),('contact_risk',crisk),('predicted_useful',useful),('mean_predicted_gain',avg)]:check('calibration_'+name,np.nan if value is None else value,getattr(record,name),1e-12)
                assert valid==record.valid;trial_count+=1
                if valid and (best_value is None or (useful,avg)>best_value):best={'enabled':True,'tau':tau};best_value=(useful,avg)
            best_rules[level]=best or {'enabled':False,'tau':None};expected=rulemap[(*key,'separately_calibrated',level)];assert expected['enabled']==best_rules[level]['enabled']
            if expected['enabled']:check('best_threshold',expected['tau'],best_rules[level]['tau'],1e-10)
        for mode in cfg['modes']:
            for level in cfg['levels']:
                rule={'enabled':True,'tau':1.} if mode=='unit_budget' else best_rules['predicted'] if mode=='predicted_rule_transfer' else best_rules[level];savedrule=rulemap[(*key,mode,level)];assert savedrule['enabled']==rule['enabled']
                if rule['enabled']:check('rule_transfer',rule['tau'],savedrule['tau'],1e-10)
                for split in ['calibration','development']:
                    r=selected(f[f.split==split],level,rule,cfg)
                    for k,v in dict(environment=env,representation=rep,target=kind,mode=mode,level=level,split=split).items():r[k]=v
                    if mode=='unit_budget' and level=='dense_full':assert not r.violation.any()
                    reconstructed.append(r)
        print('AUDITED CALIBRATION',key,flush=True)
    raw=pd.concat(reconstructed,ignore_index=True);keys=['environment','representation','target','mode','level','split','state_id'];raw=raw.sort_values(keys).reset_index(drop=True);stored=output.sort_values(keys).reset_index(drop=True)
    assert len(raw)==len(stored)
    for name in raw.columns:
        if name in keys or name=='phase':np.testing.assert_array_equal(raw[name],stored[name])
        else:check('selection_'+name,raw[name],stored[name],1e-10)
    analysis=read(out/'analysis.json')
    for r in analysis['summary']:
        v=raw
        for k in ['environment','representation','target','mode','level','split']:v=v[v[k]==r[k]]
        if r['phase']!='all':v=v[v.phase==r['phase']]
        assert len(v)==r['states'] and int(v.active.sum())==r['active_states']
        for metric in ['active','joint','jerk_gain','predicted_gain','candidate_coverage','selected_risk']:
            z=v[v.active] if metric=='selected_risk' else v;col='violation' if metric=='selected_risk' else metric
            if not len(z):assert r[metric]['mean'] is None;continue
            avg,ci=cluster(z[col].to_numpy(),z.initial_pose_family.to_numpy());check('summary_mean',avg,r[metric]['mean'],1e-10);check('summary_ci',ci,r[metric]['ci95'],1e-10)
        candidate_risk=v.candidate_violation_mass.sum()/v.candidate_coverage.sum() if v.candidate_coverage.sum()>0 else np.nan;check('candidate_risk',candidate_risk,np.nan if r['candidate_risk'] is None else r['candidate_risk'],1e-10)
    for r in analysis['paired_adjacent_levels']:
        v=raw
        for k in ['environment','representation','target','mode','split']:v=v[v[k]==r[k]]
        a=v[v.level==r['from']].set_index('state_id');b=v[v.level==r['to']].set_index('state_id').loc[a.index];diff=b[r['metric']].astype(float)-a[r['metric']].astype(float);avg,ci=cluster(diff.to_numpy(),a.initial_pose_family.to_numpy());check('paired_mean',avg,r['mean'],1e-10);check('paired_ci',ci,r['ci95'],1e-10)
    write_json(out/'raw_audit.json',{'family':args.family,'smoke':args.smoke,'states':records,'candidate_readout_rows':len(allrows),'decoded_arrays':decoded_count,'calibration_trials':trial_count,'selected_rows':len(raw),'max_errors':errors,'source_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'rules_sha256':sha256(out/'rules.json'),'scope':'Allrawfeature/readoutinference, independent effect geometry anddensephysicalpose, PushTrawpolygoncoverage, actualjerk, causalexpectedcosts, everycalibrationthreshold/choice, allselectedidentities, familyuncertainty. Cubecausalresponse traces inheritedfromauditedparent. Diagnosticfuturelevels only.'});print('COMPLETE AUDIT',records,len(raw),trial_count,flush=True)
if __name__=='__main__':main()
