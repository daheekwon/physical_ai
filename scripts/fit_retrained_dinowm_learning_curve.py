"""Prespecified nested family learning curves; full-data readouts reused unchanged."""
import json,math,sys,time,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from zj_wam.io import resolve,sha256,write_json
from scripts.retrained_dinowm_readout_core import load,targets,fit,predict,errors,score
from scripts.audit_retrained_dinowm_readouts import infer,metric,cluster
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');PARENT=BASE/'dinowm_readouts';OUT=BASE/'dinowm_learning_curve';read=lambda p:json.loads(p.read_text())


def main():
    OUT.mkdir(exist_ok=True);cfg=read(PARENT/'protocol.json')['config'];paths=[Path(__file__),resolve('scripts/retrained_dinowm_readout_core.py'),resolve('scripts/audit_retrained_dinowm_readouts.py'),PARENT/'protocol.json',PARENT/'data_lock.json',PARENT/'selection.json',PARENT/'analysis.json',PARENT/'raw_audit.json',PARENT/'pipeline_complete.json']
    protocol={'fractions':[.25,.5,1.],'subset_seed':20261010,'config':cfg,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'scope':'Nested random training families only. Same all17 features, targets, four learners and dev alpha search; Cube raw ridge and symmetry fourlearners. 100% results reused exactly from audited parent. Calibration evaluates all; no calibration selection. Labels used by probes explicit; no encoder/predictor updates. Not finaltest.'}
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:write_json(OUT/'protocol.json',protocol)
    parent=read(PARENT/'analysis.json');metrics=[];curves=[];maxerr={};fitcount=0;warningrows=[];splitcounts={}
    def check(name,x,y,tol=1e-7):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape;err=float(abs(x.astype(float)-y.astype(float)).max());maxerr[name]=max(maxerr.get(name,0.),err);assert err<=tol,(name,err)
    for env in cfg['tasks']:
        data={s:load(env,s,cfg) for s in cfg['splits']};train=data['probe_train'];families=np.unique(train['families']);order=np.random.default_rng(protocol['subset_seed']).permutation(families);splitcounts[env]=len(families);previous=set()
        for fraction in [.25,.5]:
            selected=order[:math.ceil(len(order)*fraction)];assert previous.issubset(set(selected));previous=set(selected);mask=np.isin(train['families'],selected);folder=OUT/env/f'fraction_{fraction:.2f}';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'subset.npz',families=selected,mask=mask,sample_ids=train['keys'][mask]);shuffle=np.random.default_rng(cfg['seed']).permutation(int(mask.sum()))
            for rep,xall in train['x'].items():
                for kind in (['raw'] if env=='pusht' else ['raw','symmetry']):
                    x=xall[mask];y=targets(env,rep,train['y'],'raw' if kind=='raw' else 'symmetry')[mask];key=rep+'__'+kind;path=folder/(key+'.json')
                    if path.exists():
                        r=read(path);assert r['protocol_sha256']==sha256(OUT/'protocol.json')
                        for name,h in r['files'].items():assert sha256(folder/name)==h
                        metrics.extend(r['metrics']);curves.extend(r['curves']);fitcount+=r['learners'];warningrows.extend(r['warnings']);continue
                    start=time.perf_counter();states={};trials=[];trialstates={};warn=[]
                    for alpha in cfg['alphas'][env]:
                        state,_=fit(x,y,'ridge',alpha,env,cfg['seed']);value=score(env,predict(state,data['development']['x'][rep]),targets(env,rep,data['development']['y'],'raw'),kind);trials.append({'fraction':fraction,'environment':env,'representation':rep,'target':kind,'alpha':alpha,'development_score':value});trialstates[alpha]=state
                    best=min(trials,key=lambda t:t['development_score']);alpha=best['alpha'];states['ridge']=trialstates[alpha]
                    if env=='pusht' or kind=='symmetry':
                        for learner,labels in [('label_shuffle',y[shuffle]),('mlp64',y)]:
                            with warnings.catch_warnings(record=True) as observed:
                                warnings.simplefilter('always');states[learner],_=fit(x,labels,'mlp' if learner=='mlp64' else 'ridge',alpha,env,cfg['seed'])
                            warn.extend({'task':env,'fraction':fraction,'representation':rep,'learner':learner,'warning':str(w.message)} for w in observed)
                        states['mean']={'x_mean':np.zeros(x.shape[1]),'x_scale':np.ones(x.shape[1]),'y_mean':y.mean(0),'y_scale':np.ones(y.shape[1]),'coef':np.zeros((y.shape[1],x.shape[1])),'intercept':np.zeros(y.shape[1])}
                    rows=[];files=[]
                    for learner,state in states.items():
                        fitcount+=1;name=key+'__'+learner;mp=folder/(name+'.npz');np.savez_compressed(mp,**state);files.append(mp)
                        with np.load(mp,allow_pickle=False) as raw:restored=dict(raw)
                        if learner in ['ridge','label_shuffle']:
                            xm=x.mean(0);xs=x.std(0);xs[xs==0]=1.;labels=y[shuffle] if learner=='label_shuffle' else y;ym=labels.mean(0);ys=labels.std(0);ys[ys==0]=1.;reference=Ridge(alpha=alpha).fit((x-xm)/xs,(labels-ym)/ys);check('independent_ridge_coef',state['coef'],reference.coef_,1e-7)
                        for split in ['development','calibration']:
                            d=data[split];truth=targets(env,rep,d['y'],'raw');pred=predict(state,d['x'][rep]);check('saved_inference',pred,infer(restored,d['x'][rep]),1e-8);e=errors(env,pred,truth);independent=metric(env,pred,truth)
                            for k,v in e.items():check('physical_'+k,v,independent[k])
                            frame=pd.DataFrame({'initial_pose_family':d['families'],**e});row={'fraction':fraction,'training_families':len(selected),'training_rows':int(mask.sum()),'environment':env,'split':split,'representation':rep,'target':kind,'readout':learner,'alpha':alpha if learner in ['ridge','label_shuffle'] else None,'rows':len(truth),'input_dimension':x.shape[1],**{k:mean_ci(frame,k) for k in e}}
                            for k,v in independent.items():
                                avg,ci=cluster(v,d['families']);check('family_mean',avg,row[k]['mean']);check('family_ci',ci,row[k]['ci95'])
                            rows.append(row);ep=folder/(name+'__'+split+'.npz');np.savez_compressed(ep,predicted=pred,truth=truth,families=d['families'],sample_ids=d['keys'],**e);files.append(ep)
                    record={'fraction':fraction,'environment':env,'representation':rep,'target':kind,'selected_alpha':alpha,'training_families':len(selected),'learners':len(states),'metrics':rows,'curves':trials,'warnings':warn,'files':{p.name:sha256(p) for p in files},'protocol_sha256':sha256(OUT/'protocol.json'),'seconds':time.perf_counter()-start};write_json(path,record);metrics.extend(rows);curves.extend(trials);warningrows.extend(warn);print('FIT',env,fraction,key,round(time.perf_counter()-start,2),flush=True)
    for row in parent['metrics']:
        env=row['environment'];metrics.append({**row,'fraction':1.,'training_families':splitcounts[env],'training_rows':2000 if env=='pusht' else 512,'reused_parent':True})
    for name,h in protocol['inputs'].items():assert sha256(resolve(name))==h,name
    assert fitcount==152 and len(metrics)==456 and len(curves)==250,(fitcount,len(metrics),len(curves))
    write_json(OUT/'analysis.json',{'metrics':metrics,'curves':curves,'warnings':warningrows,'scope':protocol['scope']})
    write_json(OUT/'raw_audit.json',{'new_learners':fitcount,'metric_rows':len(metrics),'new_alpha_trials':len(curves),'max_error':maxerr,'protocol_sha256':sha256(OUT/'protocol.json'),'analysis_sha256':sha256(OUT/'analysis.json'),'scope':'Nested family subsets, training-only refit, independent ridge coefficients at selected alpha, saved-array inference, independent rotation geometry and family intervals. Full100% rows from separately audited parent. MLP optimizer not independently retrained.'})
    print('COMPLETE learning curve',fitcount,len(metrics),maxerr,flush=True)
if __name__=='__main__':main()
