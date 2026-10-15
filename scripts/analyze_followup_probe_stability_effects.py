"""Stream same-candidate effect measurements for180 fixed repeated probes."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.probe_stability_effect_core import catalog,features,decode_features
from scripts.analyze_retrained_dinowm_effect_readouts import summarize
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def prepare(smoke):
    out=BASE/('probe_stability_effects_smoke' if smoke else 'probe_stability_effects');out.mkdir(exist_ok=True);capture=BASE/('dinowm_interventions_smoke' if smoke else 'dinowm_interventions');lewm=BASE/('lewm_common_effect_features_smoke' if smoke else 'lewm_common_effect_features');cfg=read(resolve('configs/followup_probe_stability_effects.json'));m=read(capture/'manifest.json');models,entries,paths=catalog();paths += [resolve(p) for p in ['configs/followup_probe_stability_effects.json','scripts/probe_stability_effect_core.py','scripts/analyze_followup_probe_stability_effects.py','scripts/retrained_dinowm_effect_readout_core.py','scripts/retrained_dinowm_readout_core.py','scripts/analyze_retrained_dinowm_effect_readouts.py']]+[BASE/'probe_stability/capture_raw_audit.json',BASE/'probe_stability/capture_pipeline_complete.json',BASE/'probe_stability/fit_complete.json',BASE/'dinowm_readouts/projection.npy',capture/'manifest.json',lewm/'manifest.json']
    for env in ['pusht','ogb-cube']:paths.append(resolve('data/checkpoints/dinowm-retrained-fafraob')/env/'swm_h5_norm_stats.json')
    protocol={'smoke':smoke,'config':cfg,'models':entries,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'scope':cfg['scope']}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    return out,capture,lewm,cfg,m,models,protocol

def process(env,split,src,capture,lewm,out,cfg,models,q):
    folder=out/env/split;folder.mkdir(parents=True,exist_ok=True);dest=folder/src.name;tp=capture/env/split/src.with_suffix('.npz').name;tm=read(tp.with_suffix('.json'));assert tm['data_sha256']==sha256(tp) and tm['manifest_sha256']==sha256(capture/'manifest.json');lp=lewm/split/src.with_suffix('.npz').name if env=='pusht' else src.with_suffix('.npz')
    if env=='pusht':lm=read(lp.with_suffix('.json'));assert lm['data_sha256']==sha256(lp) and lm['manifest_sha256']==sha256(lewm/'manifest.json')
    if dest.exists():
        r=read(dest);assert r['protocol_sha256']==sha256(out/'protocol.json') and r['tokens_sha256']==sha256(tp) and r['lewm_sha256']==sha256(lp)
        for name,h in r['files'].items():assert sha256(folder/name)==h
        return
    started=time.perf_counter();source=read(src);old=pd.read_csv(src.with_suffix('.csv'));ids=old['index' if env=='pusht' else 'candidate'].to_numpy(int);norm=old['actual_norm' if env=='pusht' else 'norm'].to_numpy();scale=np.array(read(resolve('data/checkpoints/dinowm-retrained-fafraob')/env/'swm_h5_norm_stats.json')['action_std']);arrays={};frames=[]
    with np.load(tp) as z:tokens=dict(z)
    with np.load(src.with_suffix('.npz')) as a:parent={k:a[k] for k in (['physics','actions'] if env=='pusht' else ['candidate_object_pose','actions'])}
    with np.load(lp) as a:lf={k:a[k] for k in ['history','actual','predicted']} if env=='pusht' else {'history':a['history_embedding'],'actual':a['actual_endpoint_latent'],'predicted':a['predicted_latent']}
    assert tokens['actual'].shape[0]==lf['actual'].shape[0]==len(parent['actions']);cache={}
    for key,model in models[env].items():
        rep,kind,replica=key.split('__')
        if rep not in cache:cache[rep]=features(rep,tokens,lf,q)
        methods,saved=decode_features(env,kind,model,cache[rep],parent,cfg,scale);arrays.update({key+'__'+k:v for k,v in saved.items()});posmag=np.linalg.norm(saved['true_dp'],axis=1);rotmag=np.linalg.norm(saved['true_dr'],axis=1)
        for method,values in methods.items():
            common={'environment':env,'split':split,'state_id':source['state_id'],'initial_pose_family':source['initial_pose_family'],'representation':rep,'target':kind,'readout':replica,'method':method,'candidate':ids,'active':norm>1e-6,'physical_E':old.object_error.to_numpy(),'jerk_gain':old.jerk_gain.to_numpy(),'true_position_magnitude':posmag[ids],'true_rotation_magnitude':rotmag[ids],'wrong_index':saved['wrong_index'][ids],'wrong_norm_ratio':saved['wrong_norm_ratio'][ids],'wrong_cosine':saved['wrong_cosine'][ids],'phase':np.where(old.nominal_contact,'contact','free') if env=='pusht' else source['phase']};frames.append(pd.DataFrame({**common,**{k:v[ids] for k,v in values.items()}}))
    f=pd.concat(frames,ignore_index=True);csv=dest.with_suffix('.csv.gz');npz=dest.with_suffix('.npz');f.to_csv(csv,index=False);np.savez_compressed(npz,**arrays);write_json(dest,{'environment':env,'split':split,'state_id':source['state_id'],'source':str(src.relative_to(resolve('.'))),'rows':len(f),'candidate_rows':len(old),'readouts':len(models[env]),'protocol_sha256':sha256(out/'protocol.json'),'tokens_sha256':sha256(tp),'lewm_sha256':sha256(lp),'files':{p.name:sha256(p) for p in [csv,npz]},'seconds':time.perf_counter()-started});print('ANALYZED',env,split,src.stem,len(f),round(time.perf_counter()-started,2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--dino-pid',type=int);ap.add_argument('--lewm-pid',type=int);args=ap.parse_args();out,capture,lewm,cfg,m,models,protocol=prepare(args.smoke);q=np.load(BASE/'dinowm_readouts/projection.npy');pending=[(env,split,resolve(name)) for env,splits in m['states'].items() for split,names in splits.items() for name in names];started=time.monotonic()
    while pending:
        remaining=[]
        for env,split,src in pending:
            if not (capture/env/split/src.name).exists():remaining.append((env,split,src));continue
            if env=='pusht' and not (lewm/split/src.name).exists():remaining.append((env,split,src));continue
            process(env,split,src,capture,lewm,out,cfg,models,q)
        pending=remaining
        if pending:
            assert args.dino_pid and args.lewm_pid,'Missing dependency data; supply live parents'
            for env,split,src in pending:
                if not (capture/env/split/src.name).exists():os.kill(args.dino_pid,0)
                if env=='pusht' and not (lewm/split/src.name).exists():os.kill(args.lewm_pid,0)
            if time.monotonic()-started>21600:raise RuntimeError('Observation window elapsed; inspect live dependencies')
            time.sleep(5)
    for root,pid,n in [(capture,args.dino_pid,4 if args.smoke else 402),(lewm,args.lewm_pid,2 if args.smoke else 210)]:
        while not (root/'raw_audit.json').exists():assert pid;os.kill(pid,0);time.sleep(5)
        audit=read(root/'raw_audit.json');assert audit['states']==n and audit['manifest_sha256']==sha256(root/'manifest.json')
    for name,h in protocol['inputs'].items():assert sha256(resolve(name))==h,name
    summarize(out,cfg);write_json(out/'analysis_complete.json',{'smoke':args.smoke,'states':4 if args.smoke else 402,'models':180,'analysis_sha256':sha256(out/'analysis.json'),'resolution_sha256':sha256(out/'resolution.json'),'dino_audit_sha256':sha256(capture/'raw_audit.json'),'lewm_audit_sha256':sha256(lewm/'raw_audit.json'),'scope':'Effect analysis complete; independent audit required, B not complete until then.'});print('COMPLETE ANALYSIS',flush=True)
if __name__=='__main__':main()
