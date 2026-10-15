"""Independent feature assembly and native single-action parity for common bank."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import FrozenLeWM
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('lewm_common_effect_features_smoke' if args.smoke else 'lewm_common_effect_features');m=read(out/'manifest.json');cfg=m['config'];done=read(out/'execution_complete.json');assert done['smoke']==args.smoke and done['manifest_sha256']==sha256(out/'manifest.json');parent=read(BASE/'protocol/manifest.json');scaler=StandardScaler();scaler.mean_=np.array(parent['normalization_mean']);scaler.scale_=np.array(parent['normalization_scale']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=2;torch.set_num_threads(cfg['threads']);model=FrozenLeWM(parent['config']['checkpoint'],scaler,'cpu');errors={};states=0;candidates=0;fresh=0
    def check(name,a,b):
        assert a.shape==b.shape;np.testing.assert_allclose(a,b,atol=cfg['absolute_parity_tolerance'],rtol=cfg['relative_parity_tolerance']);errors[name]=max(errors.get(name,0.),float(abs(a-b).max()))
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    for split,names in m['states'].items():
        for name in names:
            src=resolve(name);path=out/split/src.name;meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(path.with_suffix('.npz'))
            with np.load(src.with_suffix('.npz')) as a:actions=a['actions'];images=a['endpoint_images'][:,0];history=a['history_images'];past=a['past_actions'];root=a['root'];physical=a['physics']
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:actual=z['actual'];pred=z['predicted'];hist=z['history']
            assert actual.shape==pred.shape==(len(actions),192) and hist.shape==(3,192) and np.isfinite(actual).all() and np.isfinite(pred).all();assert meta['candidates']==len(actions);candidates+=len(actions);states+=1
            # Direct encoder/projector path, separate from model.encode collection.
            encoded=[];combined=np.r_[history,images]
            with torch.no_grad():
                for j in range(0,len(combined),7):
                    px=torch.stack([model.transform(im) for im in combined[j:j+7]]);hidden=model.model.encoder(px,interpolate_pos_encoding=True).last_hidden_state[:,0];encoded.append(model.model.projector(hidden).cpu().numpy())
            encoded=np.concatenate(encoded);check('fresh_direct_history',hist,encoded[:3]);check('fresh_direct_actual_all',actual,encoded[3:]);predict,_=model.predictor(history,past)
            with torch.no_grad():
                for idx in [0,len(actions)-1]:
                    x=torch.tensor(model.normalize(actions[idx]).reshape(-1),device='cpu');value=predict(x).cpu().numpy();check('fresh_single_native_prediction',pred[idx],value);fresh+=1
            with np.load(BASE/'response_scores/f1'/src.with_suffix('.npz').name) as z:check('original_gpu_predictions',pred,z['predictions'][:,0])
            with np.load(BASE/'interventions'/src.with_suffix('.npz').name) as z:
                np.testing.assert_array_equal(z['actions'][0],actions[0]);np.testing.assert_array_equal(z['physics'][0],physical[0]);np.testing.assert_array_equal(z['history_images'],history);np.testing.assert_array_equal(z['endpoint_images'][0,0],images[0]);check('original_gpu_history',hist,z['history_embedding'][0])
            with np.load(BASE/'lewm_audit/effects'/src.with_suffix('.npz').name) as z:check('original_gpu_nominal_real',actual[0],z['real_projected_cls'][0,0])
            print('AUDITED',split,src.stem,len(actions),flush=True)
    assert states==done['states']==(2 if args.smoke else 210) and candidates==done['candidates_including_nominal']
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'states':states,'candidates_including_nominal':candidates,'fresh_native_single_predictions':fresh,'all_actual_endpoints_freshly_encoded':True,'max_error':errors,'manifest_sha256':sha256(out/'manifest.json'),'source_sha256':sha256(Path(__file__)),'scope':'Full raw endpoint/history encoding via independent direct encoder/projector, two fresh single native predictions per state, all originalGPU prediction parity, same nominal images/actions/physics with olddifferentbank. FP32 device/batch numerical parity, not bitwise identity.'});print('COMPLETE AUDIT',states,candidates,flush=True)
if __name__=='__main__':main()
