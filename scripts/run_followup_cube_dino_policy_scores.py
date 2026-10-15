"""DINO scores for the audited Cube policy bank; no new physical outcomes."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from scripts.model_runtime_imports import load_dino
BASE=resolve('artifacts/followup_v1')
PARENT=BASE/'cube_policy_calibration_bank'
read=lambda p:json.loads(p.read_text())
SOURCES=['scripts/run_followup_cube_dino_policy_scores.py','scripts/model_runtime_imports.py','scripts/retrained_dinowm_core.py','configs/followup_cube_dino_policy.json']

def verify(inputs):
    for name,h in inputs.items():assert sha256(resolve(name))==h,name

def inputs_only(path):
    # Explicit allowlist: do not load candidate physical outcomes into prediction context.
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
        return {k:a[k].copy() for k in ['actions','input_history_images','input_past_actions']}

def native(model,encoded,past,action):
    actions=np.concatenate([past,action]).astype(np.float32)
    normalized=(torch.tensor(actions)-torch.tensor(model.stats['action_mean'],dtype=torch.float32))/torch.tensor(model.stats['action_std'],dtype=torch.float32)
    with torch.no_grad():return model.model.rollout_last_visual(encoded,normalized.reshape(1,-1,25))[0].numpy()

def fixture(model):
    src=BASE/'cube/development/e00108_approach_t022.npz'
    old=BASE/'dinowm_interventions/ogb-cube/development'/src.name
    with np.load(src,allow_pickle=False) as a,np.load(old,allow_pickle=False) as z:
        encoded=model.encode(a['history_images']);np.testing.assert_array_equal(encoded[0].numpy(),z['history'])
        for i in [0,len(a['actions'])-1]:
            pred=native(model,encoded,a['past_actions'],a['actions'][i]);np.testing.assert_array_equal(pred,z['predicted'][i])
    return {'source':str(src.relative_to(resolve('.'))),'source_sha256':sha256(src),'previous_prediction_sha256':sha256(old),'history_exact':True,'native_predictions_exact':2}

def prepare(smoke):
    cfg=read(resolve(SOURCES[-1]));ba=read(PARENT/'raw_audit.json');gate=read(PARENT/'pipeline_complete.json')
    assert gate['raw_audit_sha256']==sha256(PARENT/'raw_audit.json') and ba['states']==138
    paths=sorted(PARENT.glob('*_t*.json'))
    if smoke:paths=[p for p in paths if read(p)['seed'] in cfg['smoke_seeds']]
    assert len(paths)==(8 if smoke else 138)
    out=BASE/('cube_dino_policy_scores_smoke' if smoke else 'cube_dino_policy_scores');out.mkdir(exist_ok=True)
    files=[resolve(p) for p in SOURCES]+[PARENT/'manifest.json',PARENT/'raw_audit.json',PARENT/'pipeline_complete.json',BASE/'cube_policy_selection/rules.json']
    files+=sorted((BASE/'protocol/dinowm_retrained_inventory').glob('*.json'))
    files+=[p.with_suffix(ext) for p in paths for ext in ['.json','.npz','.csv']]
    m={'smoke':smoke,'config':cfg,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==m
    else:write_json(out/'manifest.json',m)
    return out,m

def collect(smoke):
    out,m=prepare(smoke);model=load_dino('ogb-cube');f=fixture(model);write_json(out/'native_fixture.json',f);total=0
    for number,name in enumerate(m['sources'],1):
        src=resolve(name);dest=out/src.name;parent=read(src)
        if dest.exists():
            r=read(dest);assert r['manifest_sha256']==sha256(out/'manifest.json') and r['data_sha256']==sha256(dest.with_suffix('.npz'));total+=r['candidates'];continue
        t=time.perf_counter();a=inputs_only(src);assert a['actions'].shape[1:]==(5,5) and a['input_past_actions'].shape==(10,5) and a['input_history_images'].shape==(3,224,224,3)
        assert sha256(src.with_suffix('.npz'))==parent['data_sha256']
        encoded=model.encode(a['input_history_images'])
        with torch.no_grad():pred=np.array([model.predict(encoded,a['input_past_actions'],action)[0].numpy() for action in a['actions']])
        assert pred.shape==(len(a['actions']),196,384) and np.isfinite(pred).all()
        diff=pred.astype(float)-pred[:1].astype(float);rms=np.sqrt(np.mean(np.sum(diff**2,axis=2),axis=1));mean=np.linalg.norm(diff.mean(1),axis=1)
        np.savez_compressed(dest.with_suffix('.npz'),predicted=pred,history=encoded[0].numpy(),dino_patch_rms=rms,dino_patch_mean=mean)
        write_json(dest,{'source':name,'split':parent['split'],'seed':parent['seed'],'phase':parent['phase'],'candidates':len(pred),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'seconds':time.perf_counter()-t})
        total+=len(pred);print('CUBE DINO SCORES',number,src.stem,len(pred),round(time.perf_counter()-t,2),flush=True)
    verify(m['inputs']);write_json(out/'execution_complete.json',{'states':len(m['sources']),'candidates':total,'smoke':smoke,'manifest_sha256':sha256(out/'manifest.json'),'fixture_sha256':sha256(out/'native_fixture.json')})

def audit(smoke):
    out=BASE/('cube_dino_policy_scores_smoke' if smoke else 'cube_dino_policy_scores');m=read(out/'manifest.json');done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['manifest_sha256']==sha256(out/'manifest.json');verify(m['inputs']);model=load_dino('ogb-cube');fixture(model);total=0;fresh=0;maxerr=0.
    for number,name in enumerate(m['sources'],1):
        src=resolve(name);meta=read(out/src.name);dest=out/src.with_suffix('.npz').name
        assert meta['source']==name and meta['data_sha256']==sha256(dest) and meta['manifest_sha256']==sha256(out/'manifest.json')
        a=inputs_only(src)
        with np.load(dest,allow_pickle=False) as z:
            pred=z['predicted'].astype(float);assert pred.shape==(len(a['actions']),196,384) and np.isfinite(pred).all();d=pred-pred[0]
            rms=np.linalg.norm(d.reshape(len(d),-1),axis=1)/np.sqrt(196);mean=np.linalg.norm(pred.mean(1)-pred[0].mean(0),axis=1)
            np.testing.assert_allclose(rms,z['dino_patch_rms'],rtol=0,atol=1e-10);np.testing.assert_allclose(mean,z['dino_patch_mean'],rtol=0,atol=1e-10)
            encoded=model.encode(a['input_history_images']);np.testing.assert_array_equal(encoded[0].numpy(),z['history'])
            ids=range(len(pred)) if smoke else sorted({0,len(pred)-1})
            for i in ids:
                expected=native(model,encoded,a['input_past_actions'],a['actions'][i]);err=float(abs(expected-pred[i]).max());maxerr=max(maxerr,err);np.testing.assert_array_equal(expected,pred[i]);fresh+=1
            total+=len(pred)
        print('AUDIT CUBE DINO',number,src.stem,fresh,flush=True)
    assert total==done['candidates'];verify(m['inputs']);write_json(out/'raw_audit.json',{'smoke':smoke,'states':len(m['sources']),'candidates':total,'fresh_history_forwards':len(m['sources']),'fresh_native_candidate_forwards':fresh,'max_prediction_error':maxerr,'manifest_sha256':sha256(out/'manifest.json'),'execution_complete_sha256':sha256(out/'execution_complete.json'),'source_sha256':sha256(Path(__file__)),'scope':'All native token shapes/finiteness and two distance metrics independently checked; fresh encoder all states and native normalization+predictor every smoke candidate, first/last every full state. Exact prior expert fixture. No new physics; audited parent bank retained. Statistical calibration/selection remains separate.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','audit']);p.add_argument('--smoke',action='store_true');args=p.parse_args();cfg=read(resolve(SOURCES[-1]));torch.set_num_threads(cfg['torch_threads'])
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):(collect if args.stage=='collect' else audit)(args.smoke)
