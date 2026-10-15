"""Common-bank actual-output replacement diagnosis; no fresh test or physics."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset
from scripts.probe_stability_effect_core import features
from scripts.retrained_dinowm_readout_core import predict
from scripts.analyze_followup_direction_utility import pd_trace
from scripts.output_replacement_core import decoded_score,physical,selections,fit,summarize
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def prepare(family,smoke):
    cfg=read(resolve('configs/followup_output_replacement.json'));out=BASE/('output_replacement_'+family+('_smoke' if smoke else ''));out.mkdir(exist_ok=True);source=read(BASE/'dinowm_interventions/manifest.json');capture=BASE/('dinowm_interventions_smoke' if smoke else 'dinowm_interventions');lf=BASE/'lewm_common_effect_features';models={};files={}
    for env,kinds in cfg['targets'].items():
        models[env]={}
        for rep in cfg['representations'][family]:
            for kind in kinds:
                key=rep+'__'+kind;p=BASE/'dinowm_readouts'/env/(key+'__ridge.npz');files[str(p.relative_to(resolve('.')))]=sha256(p)
                with np.load(p,allow_pickle=False) as z:models[env][key]=dict(z)
    paths=[resolve('configs/followup_output_replacement.json'),resolve('scripts/output_replacement_core.py'),Path(__file__),resolve('scripts/probe_stability_effect_core.py'),resolve('scripts/retrained_dinowm_readout_core.py'),resolve('scripts/retrained_dinowm_effect_readout_core.py'),resolve('scripts/analyze_followup_direction_utility.py'),resolve('scripts/analyze_followup_response_selection.py'),BASE/'dinowm_readouts/raw_audit.json',BASE/'dinowm_readouts/projection.npy',BASE/'dinowm_interventions/manifest.json',BASE/'cube/raw_audit.json',BASE/'cube/calibration/raw_audit.json']
    paths += [lf/'manifest.json',lf/'raw_audit.json',lf/'pipeline_complete.json'] if family=='lewm' else [capture/'manifest.json']
    states={env:{split:names[:1] if smoke else names for split,names in splits.items()} for env,splits in source['states'].items()};files.update({str(p.relative_to(resolve('.'))):sha256(p) for p in paths})
    for splits in states.values():
        for names in splits.values():
            for name in names:
                for ext in ['.json','.csv','.npz']:
                    p=resolve(name).with_suffix(ext);key=str(p.relative_to(resolve('.')));assert sha256(p)==source['inputs'][key];files[key]=source['inputs'][key]
    record={'family':family,'smoke':smoke,'config':cfg,'states':states,'inputs':files,'scope':cfg['scope']}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==record
    else:write_json(out/'protocol.json',record)
    return out,capture,lf,cfg,models,record

def process(env,split,src,out,capture,lf,cfg,models,q,data,family):
    folder=out/env/split;folder.mkdir(parents=True,exist_ok=True);dest=folder/src.name;featurepath=capture/env/split/src.with_suffix('.npz').name if family=='dino' else lf/split/src.with_suffix('.npz').name if env=='pusht' else src.with_suffix('.npz');featuremeta=read(featurepath.with_suffix('.json'))
    if family=='dino' or env=='pusht':assert featuremeta['data_sha256']==sha256(featurepath)
    if dest.exists():
        old=read(dest);assert old['protocol_sha256']==sha256(out/'protocol.json') and old['feature_sha256']==sha256(featurepath)
        for p,h in old['files'].items():assert sha256(folder/p)==h
        return
    started=time.perf_counter();meta=read(src);old=pd.read_csv(src.with_suffix('.csv'))
    with np.load(src.with_suffix('.npz'),allow_pickle=False) as z:parent=dict(z)
    with np.load(featurepath,allow_pickle=False) as z:fv=dict(z)
    if family=='dino':tokens=fv;lewm=None
    else:
        tokens=None;lewm={k:fv[k] for k in ['history','actual','predicted']} if env=='pusht' else {'history':fv['history_embedding'],'actual':fv['actual_endpoint_latent'],'predicted':fv['predicted_latent']}
    ids=old['index' if env=='pusht' else 'candidate'].to_numpy(int);norm=old['actual_norm' if env=='pusht' else 'norm'].to_numpy();truth=physical(env,parent);costs=old['predicted_jerk' if env=='pusht' else 'predicted_jerk_cost'].to_numpy()
    if env=='pusht':
        aa,_=data.episode(meta['episode']);t=meta['timestep'];base=pd_trace(aa[t:t+8],parent['root']);nominalcost=float(np.sum(np.diff(np.r_[parent['physics'][0,:30,:2],base],n=3,axis=0)**2)/.01**5);phase='contact' if old.nominal_contact.iloc[0] else 'free'
    else:
        pts=np.r_[parent['past_points'][-75:],parent['predicted_robot'][0]].copy();pts[:,3]*=.05;nominalcost=float(np.sum(np.diff(pts,n=3,axis=0)**2)/.002**5);phase=meta['phase']
    np.testing.assert_allclose(truth['dense_full'][ids],old.object_error,atol=2e-5,rtol=1e-9);np.testing.assert_allclose(truth['jerk_gain'][ids],old.jerk_gain,atol=1e-8,rtol=1e-8)
    common={'environment':env,'split':split,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'phase':phase,'candidate':ids,'norm':norm,'predicted_cost':costs,'nominal_predicted_cost':nominalcost,'predicted_gain':(nominalcost-costs)/max(nominalcost,1e-12),**{k:v[ids] for k,v in truth.items()}};frames=[];arrays={};cache={}
    for key,model in models.items():
        rep,kind=key.split('__')
        if rep not in cache:cache[rep]=features(rep,tokens,lewm,q)
        real=predict(model,cache[rep]['real']);pred=predict(model,cache[rep]['predicted']);assert len(real)==len(parent['actions']);arrays[key+'__real']=real;arrays[key+'__predicted']=pred
        frames.append(pd.DataFrame({**common,'representation':rep,'target':kind,'predicted':decoded_score(env,pred,kind)[ids],'real_encoded':decoded_score(env,real,kind)[ids]}))
    frame=pd.concat(frames,ignore_index=True);csv=dest.with_suffix('.csv');npz=dest.with_suffix('.npz');frame.to_csv(csv,index=False);np.savez_compressed(npz,**arrays);write_json(dest,{'source':str(src.relative_to(resolve('.'))),'feature_source':str(featurepath.relative_to(resolve('.'))),'environment':env,'split':split,'state_id':meta['state_id'],'rows':len(frame),'readouts':len(models),'candidate_rows':len(old),'feature_sha256':sha256(featurepath),'protocol_sha256':sha256(out/'protocol.json'),'files':{p.name:sha256(p) for p in [csv,npz]},'seconds':time.perf_counter()-started});print('SCORED',family,env,split,src.stem,len(frame),flush=True)

def aggregate(out,cfg):
    frame=pd.concat([pd.read_csv(p) for p in sorted(out.glob('*/*/*.csv'))],ignore_index=True);rules=[];curves=[];rows=[]
    for (env,rep,kind),f in frame.groupby(['environment','representation','target'],sort=True):
        cal=f[f.split=='calibration'];fitted={}
        for level in cfg['levels']:
            fitted[level],c=fit(cal,level,cfg,env);curves.extend([{'environment':env,'representation':rep,'target':kind,'level':level,**r} for r in c])
        for mode in cfg['modes']:
            for level in cfg['levels']:
                rule={'enabled':True,'tau':1.} if mode=='unit_budget' else fitted['predicted'] if mode=='predicted_rule_transfer' else fitted[level];common={'environment':env,'representation':rep,'target':kind,'mode':mode,'level':level};rules.append({**common,'rule':rule})
                for split in ['calibration','development']:
                    r=selections(f[f.split==split],level,rule,cfg)
                    for k,v in {**common,'split':split}.items():r[k]=v
                    rows.append(r)
        print('CALIBRATED',env,rep,kind,flush=True)
    selected=pd.concat(rows,ignore_index=True);selected.to_csv(out/'selected.csv',index=False);pd.DataFrame(curves).to_csv(out/'calibration_curves.csv',index=False);write_json(out/'rules.json',{'rules':rules,'scope':cfg['scope']});analysis=summarize(selected);analysis.update(family=read(out/'protocol.json')['family'],smoke=read(out/'protocol.json')['smoke'],candidate_readout_rows=len(frame),selected_rows=len(selected));write_json(out/'analysis.json',analysis)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['lewm','dino'],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--dependency-pid',type=int);args=ap.parse_args();out,capture,lf,cfg,models,protocol=prepare(args.family,args.smoke);data=ExpertDataset(read(resolve('configs/followup_v1.json'))['dataset']);q=np.load(BASE/'dinowm_readouts/projection.npy');pending=[(env,split,resolve(p)) for env,splits in protocol['states'].items() for split,paths in splits.items() for p in paths];start=time.monotonic()
    while pending:
        remaining=[]
        for env,split,src in pending:
            if args.family=='dino' and not (capture/env/split/src.name).exists():remaining.append((env,split,src));continue
            process(env,split,src,out,capture,lf,cfg,models[env],q,data,args.family)
        pending=remaining
        if pending:
            assert args.dependency_pid;os.kill(args.dependency_pid,0)
            if time.monotonic()-start>21600:raise RuntimeError('Observationwindow elapsed; inspect live dependency')
            time.sleep(5)
    if args.family=='dino':
        while not (capture/'raw_audit.json').exists():assert args.dependency_pid;os.kill(args.dependency_pid,0);time.sleep(5)
        assert read(capture/'raw_audit.json')['states']==(4 if args.smoke else 402)
    for name,h in protocol['inputs'].items():assert sha256(resolve(name))==h,name
    aggregate(out,cfg);write_json(out/'analysis_complete.json',{'family':args.family,'smoke':args.smoke,'states':4 if args.smoke else 402,'analysis_sha256':sha256(out/'analysis.json'),'rules_sha256':sha256(out/'rules.json'),'protocol_sha256':sha256(out/'protocol.json'),'scope':'Independent raw+calibration audit required before reporting.'});print('COMPLETE ANALYSIS',args.family,flush=True)
if __name__=='__main__':main()
