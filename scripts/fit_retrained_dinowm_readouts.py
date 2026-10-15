"""Fit matched readouts after all capture data pass audit; array-only models."""
import argparse,json,os,sys,time,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.retrained_dinowm_readout_core import load,projection,targets,errors,score,fit,predict
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_readouts';read=lambda p:json.loads(p.read_text())


def protocol():
    OUT.mkdir(exist_ok=True);paths=[resolve(p) for p in ['configs/followup_dinowm_readouts.json','scripts/retrained_dinowm_readout_core.py','scripts/fit_retrained_dinowm_readouts.py','zj_wam/cube_readouts.py','scripts/analyze_followup_response_selection.py']]+[BASE/'dinowm_capture/pipeline_lock.json',BASE/'dinowm_readout_smoke.json']
    m={'config':read(paths[0]),'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==m
    else:write_json(path,m)
    return m


def fit_all(m):
    complete=read(BASE/'dinowm_capture/pipeline_complete.json');assert complete['labels']==4296
    assert complete['raw_audit_sha256']==sha256(BASE/'dinowm_capture/raw_audit.json')
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    cfg=m['config'];q=projection(cfg);np.save(OUT/'projection.npy',q)
    paths=[BASE/'dinowm_capture/manifest.json',BASE/'dinowm_capture/raw_audit.json',BASE/'dinowm_capture/pipeline_complete.json']+list((BASE/'dinowm_capture/pusht').glob('*.npz'))+list((BASE/'dinowm_capture/ogb-cube').glob('*/*.npz'))
    data_lock={'protocol_sha256':sha256(OUT/'protocol.json'),'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    p=OUT/'data_lock.json'
    if p.exists():assert read(p)==data_lock
    else:write_json(p,data_lock)
    selections={};metrics=[];curves=[]
    for env in cfg['tasks']:
        all_data={split:load(env,split,cfg) for split in cfg['splits']};train=all_data['probe_train'];dev=all_data['development'];cal=all_data['calibration']
        for x,y in [('probe_train','development'),('probe_train','calibration'),('development','calibration')]:assert not set(all_data[x]['families'])&set(all_data[y]['families'])
        folder=OUT/env;folder.mkdir(exist_ok=True);shuffle=np.random.default_rng(cfg['seed']).permutation(len(train['y']));np.save(folder/'shuffle.npy',shuffle)
        for rep in sorted(train['x']):
            for target_kind in (['raw'] if env=='pusht' else ['raw','symmetry']):
                key=rep+'__'+target_kind;record_path=folder/(key+'.json')
                if record_path.exists():
                    r=read(record_path);assert r['data_lock_sha256']==sha256(OUT/'data_lock.json')
                    for name,h in r['files'].items():assert sha256(folder/name)==h
                    selections[env+'/'+key]=r;metrics.extend(r['metrics']);curves.extend(r['curves']);continue
                start=time.perf_counter();x=train['x'][rep];y=targets(env,rep,train['y'],target_kind);dy=targets(env,rep,dev['y'],'raw');trials=[];models={};infos={};warnings_seen=[]
                for alpha in cfg['alphas'][env]:
                    with warnings.catch_warnings(record=True) as observed:
                        warnings.simplefilter('always');state,info=fit(x,y,'ridge',alpha,env,cfg['seed'])
                    warnings_seen += [str(w.message) for w in observed];value=score(env,predict(state,dev['x'][rep]),dy,target_kind)
                    trials.append({'environment':env,'representation':rep,'target':target_kind,'alpha':alpha,'development_score':value});models[alpha]=state;infos[alpha]=info
                best=min(trials,key=lambda r:r['development_score']);alpha=best['alpha'];states={'ridge':models[alpha]};info_map={'ridge':infos[alpha]}
                if env=='pusht' or target_kind=='symmetry':
                    for kind,labels in [('label_shuffle',y[shuffle]),('mlp64',y)]:
                        with warnings.catch_warnings(record=True) as observed:
                            warnings.simplefilter('always');states[kind],info_map[kind]=fit(x,labels,'mlp' if kind=='mlp64' else 'ridge',alpha,env,cfg['seed'])
                        warnings_seen += [str(w.message) for w in observed]
                    states['mean']={'x_mean':np.zeros(x.shape[1]),'x_scale':np.ones(x.shape[1]),'y_mean':y.mean(0),'y_scale':np.ones(y.shape[1]),'coef':np.zeros((y.shape[1],x.shape[1])),'intercept':np.zeros(y.shape[1])};info_map['mean']={'training_rows':len(x),'input_dimension':x.shape[1],'output_dimension':y.shape[1]}
                rows=[];files=[]
                for kind,state in states.items():
                    name=key+'__'+kind;model_path=folder/(name+'.npz');np.savez_compressed(model_path,**state);files.append(model_path)
                    for split in ['development','calibration']:
                        data=all_data[split];truth=targets(env,rep,data['y'],'raw');prediction=predict(state,data['x'][rep]);e=errors(env,prediction,truth)
                        frame=pd.DataFrame({'initial_pose_family':data['families'],'sample_id':data['keys'],**e});record={'environment':env,'split':split,'representation':rep,'target':target_kind,'readout':kind,'alpha':alpha if kind in ['ridge','label_shuffle'] else None,'rows':len(frame),'input_dimension':x.shape[1],**{k:mean_ci(frame,k) for k in e}}
                        rows.append(record);saved=folder/(name+'__'+split+'.npz');np.savez_compressed(saved,predicted=prediction,truth=truth,families=data['families'],sample_ids=data['keys'],**e);files.append(saved)
                record={'environment':env,'representation':rep,'target':target_kind,'selected_alpha':alpha,'development_score':best['development_score'],'learner_info':info_map,'warnings':warnings_seen,'metrics':rows,'curves':trials,'files':{p.name:sha256(p) for p in files},'data_lock_sha256':sha256(OUT/'data_lock.json'),'seconds':time.perf_counter()-start}
                write_json(record_path,record);selections[env+'/'+key]=record;metrics.extend(rows);curves.extend(trials);print('FIT',env,key,alpha,round(time.perf_counter()-start,2),flush=True)
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(OUT/'selection.json',{'readouts':{k:{field:v[field] for field in ['environment','representation','target','selected_alpha','development_score','learner_info','warnings','files']} for k,v in selections.items()},'protocol_sha256':sha256(OUT/'protocol.json'),'data_lock_sha256':sha256(OUT/'data_lock.json'),'scope':cfg['scope']})
    write_json(OUT/'analysis.json',{'metrics':metrics,'curves':curves,'scope':cfg['scope'],'selection':cfg['selection']})
    write_json(OUT/'fit_complete.json',{'configurations':len(selections),'metric_rows':len(metrics),'selection_sha256':sha256(OUT/'selection.json'),'analysis_sha256':sha256(OUT/'analysis.json'),'scope':'Fit/evaluation complete, independent audit still required.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int);args=p.parse_args();m=protocol();start=time.monotonic()
    if args.dependency_pid:
        while not (BASE/'dinowm_capture/pipeline_complete.json').exists():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Capture pipeline ended before complete audit; inspect failure')
            if time.monotonic()-start>7200:raise RuntimeError('Observation window elapsed; revalidate parent')
            time.sleep(5)
    fit_all(m)
if __name__=='__main__':main()
