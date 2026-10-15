"""Train/read exact reused probes for plan4.7B; distinct subset and init axes."""
import argparse,json,math,sys,time,warnings,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json
from scripts.retrained_dinowm_readout_core import load,targets,fit,predict,score,errors
BASE=resolve('artifacts/followup_v1');PARENT=BASE/'dinowm_readouts';CURVE=BASE/'dinowm_learning_curve';read=lambda p:json.loads(p.read_text())

def groups(cfg):
    out=[{'name':f'subset_{seed}_f{int(frac*100):03d}','subset_seed':seed,'fraction':frac} for seed in cfg['subset_seeds'] for frac in cfg['fractions'] if frac<1]
    out.append({'name':'full','subset_seed':None,'fraction':1.});return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_probe_stability.json'));parentcfg=read(PARENT/'protocol.json')['config'];out=BASE/('probe_stability_smoke' if args.smoke else 'probe_stability');out.mkdir(exist_ok=True)
    paths=[resolve(p) for p in ['configs/followup_probe_stability.json','scripts/fit_followup_probe_stability.py','scripts/retrained_dinowm_readout_core.py','zj_wam/cube_readouts.py']]+[PARENT/p for p in ['protocol.json','data_lock.json','selection.json','raw_audit.json','pipeline_complete.json']]+[CURVE/p for p in ['protocol.json','analysis.json','raw_audit.json']]
    protocol={'smoke':args.smoke,'config':cfg,'parent_config':parentcfg,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    allrecords=[]
    for env,reps in cfg['representations'].items():
        data={split:load(env,split,parentcfg) for split in ['probe_train','development','calibration']};train=data['probe_train'];families=np.unique(train['families']);kind=cfg['targets'][env]
        for group in groups(cfg):
            if args.smoke and group['name'] not in ['subset_20261011_f025','full']:continue
            fraction=group['fraction'];seed=group['subset_seed'];selected=families if seed is None else np.random.default_rng(seed).permutation(families)[:math.ceil(len(families)*fraction)];mask=np.isin(train['families'],selected);folder=out/env/group['name'];folder.mkdir(parents=True,exist_ok=True);subsetfile=folder/'subset.npz';subset={'families':selected,'mask':mask,'sample_ids':train['keys'][mask]}
            if subsetfile.exists():
                with np.load(subsetfile) as old:
                    for k,v in subset.items():np.testing.assert_array_equal(old[k],v)
            else:np.savez_compressed(subsetfile,**subset)
            for rep in reps[:1] if args.smoke else reps:
                path=folder/(rep+'.json')
                if path.exists():
                    old=read(path);assert old['protocol_sha256']==sha256(out/'protocol.json')
                    for name,h in old['files'].items():assert sha256(folder/name)==h
                    allrecords.append(old);continue
                started=time.perf_counter();x=train['x'][rep][mask];y=targets(env,rep,train['y'],kind)[mask];key=rep+'__'+kind;parentrecord=read(PARENT/env/(key+'.json'));states={};trialstates={};trials=[];sources={};warn=[];info={}
                reusefolder=CURVE/env/f'fraction_{fraction:.2f}' if seed==parentcfg['seed'] and fraction<1 else PARENT/env if fraction==1 else None
                if reusefolder:
                    rr=read(reusefolder/(key+'.json'));alpha=rr['selected_alpha']
                    if fraction<1:
                        with np.load(reusefolder/'subset.npz') as z:
                            for k,v in subset.items():np.testing.assert_array_equal(z[k],v)
                        trials=[r for r in rr['curves'] if r['representation']==rep and r['target']==kind]
                    else:trials=rr['trials'] if 'trials' in rr else []
                    for learner in cfg['learners']:
                        source=reusefolder/(key+'__'+learner+'.npz');assert rr['files'][source.name]==sha256(source)
                        with np.load(source,allow_pickle=False) as raw:states[learner]=dict(raw)
                        sources[learner]={'path':str(source.relative_to(resolve('.'))),'sha256':sha256(source)};info[learner]={'reused':True}
                else:
                    for alpha in parentcfg['alphas'][env]:
                        state,_=fit(x,y,'ridge',alpha,env,parentcfg['seed']);value=score(env,predict(state,data['development']['x'][rep]),targets(env,rep,data['development']['y'],'raw'),kind);trials.append({'alpha':alpha,'development_score':value});trialstates[alpha]=state
                    alpha=min(trials,key=lambda t:t['development_score'])['alpha'];states['ridge']=trialstates[alpha];info['ridge']={'reused':False};shuffle=np.random.default_rng(parentcfg['seed']).permutation(len(y))
                    for learner,labels in [('label_shuffle',y[shuffle]),('mlp64',y)]:
                        with warnings.catch_warnings(record=True) as ws:
                            warnings.simplefilter('always');states[learner],info[learner]=fit(x,labels,'mlp' if learner=='mlp64' else 'ridge',alpha,env,cfg['fixed_subset_mlp_seed'])
                        warn.extend({'learner':learner,'warning':str(w.message)} for w in ws)
                    states['mean']={'x_mean':np.zeros(x.shape[1]),'x_scale':np.ones(x.shape[1]),'y_mean':y.mean(0),'y_scale':np.ones(y.shape[1]),'coef':np.zeros((y.shape[1],x.shape[1])),'intercept':np.zeros(y.shape[1])};info['mean']={'reused':False}
                if fraction==1:
                    states['mlp64_seed_'+str(parentcfg['seed'])]=states.pop('mlp64');sources['mlp64_seed_'+str(parentcfg['seed'])]=sources.pop('mlp64');info['mlp64_seed_'+str(parentcfg['seed'])]=info.pop('mlp64')
                    for init in cfg['full_mlp_seeds']:
                        if init==parentcfg['seed']:continue
                        learner='mlp64_seed_'+str(init)
                        with warnings.catch_warnings(record=True) as ws:
                            warnings.simplefilter('always');states[learner],info[learner]=fit(x,y,'mlp',alpha,env,init)
                        warn.extend({'learner':learner,'warning':str(w.message)} for w in ws)
                files={};metricrows=[]
                for learner,state in states.items():
                    mp=folder/(rep+'__'+learner+'.npz');np.savez_compressed(mp,**state);files[mp.name]=sha256(mp)
                    for split in ['development','calibration']:
                        d=data[split];truth=targets(env,rep,d['y'],'raw');prediction=predict(state,d['x'][rep]);e=errors(env,prediction,truth);ep=folder/(rep+'__'+learner+'__'+split+'.npz');np.savez_compressed(ep,predicted=prediction,truth=truth,families=d['families'],sample_ids=d['keys'],**e);files[ep.name]=sha256(ep);metricrows.append({'environment':env,'group':group['name'],'subset_seed':seed,'fraction':fraction,'representation':rep,'target':kind,'learner':learner,'split':split,'training_families':len(selected),'training_rows':len(x),'input_dimension':x.shape[1],'metrics':{k:float(v.mean()) for k,v in e.items()}})
                record={'environment':env,'group':group,'representation':rep,'target':kind,'selected_alpha':alpha,'trials':trials,'learners':list(states),'learner_info':info,'reused_sources':sources,'files':files,'metrics':metricrows,'warnings':warn,'subset_sha256':sha256(subsetfile),'protocol_sha256':sha256(out/'protocol.json'),'seconds':time.perf_counter()-started};write_json(path,record);allrecords.append(record);print('FIT',env,group['name'],rep,len(states),round(time.perf_counter()-started,2),flush=True)
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    n=sum(len(r['learners']) for r in allrecords);assert n==(20 if args.smoke else 180),(n,args.smoke)
    write_json(out/'fit_complete.json',{'smoke':args.smoke,'groups':len(allrecords),'models':n,'new_models':sum(len(r['learners'])-len(r['reused_sources']) for r in allrecords),'protocol_sha256':sha256(out/'protocol.json'),'record_hashes':{str((out/r['environment']/r['group']['name']/(r['representation']+'.json')).relative_to(out)):sha256(out/r['environment']/r['group']['name']/(r['representation']+'.json')) for r in allrecords},'scope':'Capture fitting only; independent audit and real/pred effect robustness remain required.'});print('COMPLETE FIT',n,flush=True)
if __name__=='__main__':main()
