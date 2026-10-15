"""Align real/predicted LeWM features to the existing common F1 action bank."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import FrozenLeWM
from zj_wam.model_audit import predict_candidates
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
@torch.no_grad()
def encode(model,images,batch):
    rows=[]
    for i in range(0,len(images),batch):
        px=torch.stack([model.transform(im) for im in images[i:i+batch]])[None];rows.append(model.model.encode({'pixels':px})['emb'][0].cpu().numpy())
    return np.concatenate(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_lewm_common_effect_features.json'));maincfg=read(resolve('configs/followup_v1.json'));out=BASE/('lewm_common_effect_features_smoke' if args.smoke else 'lewm_common_effect_features');out.mkdir(exist_ok=True);oldmanifest=read(BASE/'protocol/manifest.json');torch.set_num_threads(cfg['threads']);paths={s:sorted((BASE/'f1').glob(s+'*_h05.json')) for s in cfg['splits']};assert all(len(v)==105 for v in paths.values())
    inputs=[resolve(p) for p in ['configs/followup_lewm_common_effect_features.json','scripts/collect_followup_lewm_common_effect_features.py','zj_wam/adapters.py','zj_wam/model_audit.py','external/stable-worldmodel/stable_worldmodel/wm/utils.py']]+[resolve(maincfg['checkpoint'])/'weights.pt',BASE/'protocol/manifest.json',BASE/'dinowm_interventions/manifest.json'];sources=[]
    for group in paths.values():
        for path in group:
            sources += [path,path.with_suffix('.npz'),BASE/'response_scores/f1'/path.with_suffix('.npz').name,BASE/'interventions'/path.with_suffix('.npz').name,BASE/'lewm_audit/effects'/path.with_suffix('.npz').name]
    manifest={'smoke':args.smoke,'config':cfg,'states':{s:[str(p.relative_to(resolve('.'))) for p in (group[:1] if args.smoke else group)] for s,group in paths.items()},'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in inputs+sources}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    scaler=StandardScaler();scaler.mean_=np.array(oldmanifest['normalization_mean']);scaler.scale_=np.array(oldmanifest['normalization_scale']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=2;model=FrozenLeWM(maincfg['checkpoint'],scaler,'cpu');count=0;total=0
    for split,names in manifest['states'].items():
        folder=out/split;folder.mkdir(exist_ok=True)
        for name in names:
            src=resolve(name);dest=folder/src.name;meta=read(src)
            if dest.exists():
                rec=read(dest);assert rec['manifest_sha256']==sha256(out/'manifest.json') and rec['data_sha256']==sha256(dest.with_suffix('.npz'));count+=1;total+=rec['candidates'];continue
            started=time.perf_counter()
            with np.load(src.with_suffix('.npz')) as a:actions=a['actions'];past=a['past_actions'];history=a['history_images'];images=a['endpoint_images'][:,0]
            with np.load(BASE/'interventions'/src.with_suffix('.npz').name) as a:np.testing.assert_array_equal(history,a['history_images']);np.testing.assert_array_equal(past,a['past_actions']);np.testing.assert_array_equal(images[0],a['endpoint_images'][0,0]);np.testing.assert_array_equal(actions[0],a['actions'][0]);oldhistory=a['history_embedding'][0].copy()
            _,context=model.predictor(history,past);hist=context['embedding'][0].cpu().numpy();actual=encode(model,images,cfg['batch_size']);pred=predict_candidates(model,context['embedding'],context['past'],actions,cfg['batch_size'])[:,0];checks={}
            with np.load(BASE/'response_scores/f1'/src.with_suffix('.npz').name) as a:oldpred=a['predictions'][:,0].copy()
            with np.load(BASE/'lewm_audit/effects'/src.with_suffix('.npz').name) as a:oldactual=a['real_projected_cls'][0,0].copy()
            for key,new,old in [('history',hist,oldhistory),('predicted_all',pred,oldpred),('real_nominal',actual[0],oldactual)]:
                assert new.shape==old.shape;np.testing.assert_allclose(new,old,rtol=cfg['relative_parity_tolerance'],atol=cfg['absolute_parity_tolerance']);checks[key]=float(abs(new-old).max())
            one=encode(model,images[[0,-1]],1);np.testing.assert_allclose(one,actual[[0,-1]],rtol=cfg['relative_parity_tolerance'],atol=cfg['absolute_parity_tolerance']);checks['single_vs_batch_encoder']=float(abs(one-actual[[0,-1]]).max());assert pred.shape==actual.shape==(len(actions),192) and hist.shape==(3,192)
            np.savez_compressed(dest.with_suffix('.npz'),history=hist,actual=actual,predicted=pred);write_json(dest,{'source':name,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'split':split,'candidates':len(actions),'device':'cpu','dtype':'float32','parity_max_error':checks,'data_sha256':sha256(dest.with_suffix('.npz')),'manifest_sha256':sha256(out/'manifest.json'),'seconds':time.perf_counter()-started});count+=1;total+=len(actions);print(split,src.stem,len(actions),checks,round(time.perf_counter()-started,2),flush=True)
    for name,h in manifest['inputs'].items():assert sha256(resolve(name))==h,name
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'states':count,'candidates_including_nominal':total,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Feature collection complete; independent audit required.'})
if __name__=='__main__':main()
