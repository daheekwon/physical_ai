"""Four state all-candidate native single-versus-batch parity diagnostic."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits,threadpool_info
from zj_wam.io import resolve,sha256,write_json
from scripts.model_runtime_imports import load_dino
from scripts.dinowm_batch_runtime import predict_batch
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_batching_smoke';read=lambda p:json.loads(p.read_text())

def scores(z):
    delta=z.astype(np.float64)-z[:1].astype(np.float64)
    return {'dino_patch_rms':np.sqrt(np.mean(np.sum(delta**2,axis=2),axis=1)),'dino_patch_mean':np.linalg.norm(delta.mean(1),axis=1)}

def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    manifest=read(BASE/'dinowm_interventions_smoke/manifest.json');checks=[]
    paths=['scripts/dinowm_batch_runtime.py','scripts/audit_followup_dinowm_batching.py','scripts/model_runtime_imports.py','scripts/retrained_dinowm_core.py'];inputs={p:sha256(resolve(p)) for p in paths}
    for env,splits in manifest['states'].items():
        dino=load_dino(env)
        for split,names in splits.items():
            src=resolve(names[0]);frame=pd.read_csv(src.with_suffix('.csv'));a=dict(np.load(src.with_suffix('.npz')))
            if env=='pusht':ids=[0]+frame[(frame.relative_radius<=.2)&(~frame.method.str.startswith('nominal'))]['index'].tolist()
            else:ids=frame[(frame.method=='nominal')|((frame.radius<=.2)&frame.method.isin(['random0','random1','past_pd_optimizer']))].candidate.tolist()
            ids=np.array(sorted(set(ids)),int);actions=a['actions'][ids];encoded=dino.encode(a['history_images']);t=time.perf_counter()
            with torch.no_grad():single=np.array([dino.predict(encoded,a['past_actions'],v)[0].numpy() for v in actions])
            single_time=time.perf_counter()-t;ss=scores(single);raw={'candidate_ids':ids,'single':single}
            for size in [4,8]:
                t=time.perf_counter();batch=predict_batch(dino,encoded,a['past_actions'],actions,size);elapsed=time.perf_counter()-t;bs=scores(batch);error=float(abs(batch-single).max());r={'environment':env,'split':split,'state_id':src.stem,'batch_size':size,'candidates':len(ids),'max_token_error':error,'max_score_error':{k:float(abs(bs[k]-ss[k]).max()) for k in ss},'single_seconds':single_time,'batch_seconds':elapsed};checks.append(r);raw['batch_'+str(size)]=batch;print(r,flush=True);assert error<=3e-5,r
            np.savez_compressed(OUT/(env+'__'+src.stem+'.npz'),**raw)
            write_json(OUT/(env+'__'+src.stem+'.json'),{'input_sha256':sha256(src.with_suffix('.npz')),'checks':checks[-2:]})
        del dino
    for p,h in inputs.items():assert sha256(resolve(p))==h
    write_json(OUT/'complete.json',{'states':4,'checks':checks,'inputs':inputs,'threadpools':threadpool_info(),'scope':'Technical native single versus batch inference parity; first-call times are not a randomized warmed benchmark. Cached source candidates, no fresh physics. Small numeric score differences may change calibration membership; any batching deployment needs its own actual scores and calibration.'})

if __name__=='__main__':
    with threadpool_limits(limits=2,user_api='blas'):main()
