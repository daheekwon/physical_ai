"""Audit saved tokens, paired physical outcomes, and fresh native wrapper samples."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.io import resolve,sha256,write_json
from scripts.retrained_dinowm_core import FrozenRetrainedDinoWM
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();out=BASE/('dinowm_interventions_smoke' if args.smoke else 'dinowm_interventions');m=read(out/'manifest.json');done=read(out/'execution_complete.json');assert done['smoke']==args.smoke and done['manifest_sha256']==sha256(out/'manifest.json')
    for name,h in m['inputs'].items():assert sha256(resolve(name))==h,name
    tol=read(resolve('configs/followup_v1.json'))['tolerances'];cube=read(resolve('configs/followup_cube.json'));count=0;candidates=0;rows=0;errors={};fresh=[];torch.set_num_threads(2)
    def check(name,x,y,tolerance):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape
        err=float(np.max(np.abs(x.astype(float)-y.astype(float))));errors[name]=max(errors.get(name,0.),err);assert err<=tolerance,(name,err)
    for env,splits in m['states'].items():
        model=FrozenRetrainedDinoWM(env)
        for split,paths in splits.items():
            for number,name in enumerate(paths):
                src=resolve(name);dest=out/env/split/src.name;meta=read(dest);assert meta['source']==name and meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(dest.with_suffix('.npz'));old=pd.read_csv(src.with_suffix('.csv'))
                with np.load(src.with_suffix('.npz'),allow_pickle=False) as a,np.load(dest.with_suffix('.npz'),allow_pickle=False) as saved:
                    n=len(a['actions']);assert saved['predicted'].shape==saved['actual'].shape==(n,196,384) and saved['history'].shape==(3,196,384)
                    assert all(np.isfinite(saved[k]).all() for k in ['predicted','actual','history']);candidates+=n;rows+=len(old);count+=1
                    if env=='pusht':
                        d=a['physics'][:,30:,2:5]-a['physics'][:1,30:,2:5];position=np.linalg.norm(d[:,:,:2],axis=-1)/tol['position'];angle=np.abs((d[:,:,2]+np.pi)%(2*np.pi)-np.pi)/tol['rotation'];coverage=np.maximum(0,a['coverage'][:1]-a['coverage'])/tol['coverage_loss'];E=np.maximum(np.maximum(position.max(1),angle.max(1)),coverage.max(1));ids=old['index'].to_numpy(int)
                    else:
                        pose=a['candidate_object_pose'];position=np.linalg.norm(pose[:,:,:3]-pose[:1,:,:3],axis=-1)/cube['position_tolerance_m'];q=pose[:,:,3:]/np.linalg.norm(pose[:,:,3:],axis=-1,keepdims=True);dots=np.clip(np.abs(np.sum(q*q[:1],axis=-1)),0,1);angle=2*np.arctan2(np.sqrt(np.maximum(0,1-dots*dots)),dots)/cube['rotation_tolerance_rad'];E=np.maximum(position.max(1),angle.max(1));ids=old.candidate.to_numpy(int)
                    check(env+'_physical_E',E[ids],old.object_error.to_numpy(),2e-5)
                    if number==0:
                        encoded=model.encode(a['history_images']);check(env+'_fresh_history',encoded[0].cpu().numpy(),saved['history'],0.)
                        for index in [0,n-1]:
                            acts=np.concatenate([a['past_actions'],a['actions'][index]]).astype(np.float32);normalized=(torch.tensor(acts)-torch.tensor(model.stats['action_mean'],dtype=torch.float32))/torch.tensor(model.stats['action_std'],dtype=torch.float32)
                            with torch.no_grad():
                                pred=model.model.rollout_last_visual(encoded,normalized.reshape(1,-1,acts.shape[1]*5))[0].cpu().numpy()
                                im=a['endpoint_images'][index,0] if env=='pusht' else a['endpoint_images'][index];pixels=model.model.img_transform(im)[None,None];actual=model.model.encode_visual(pixels)[0,0].cpu().numpy()
                            check(env+'_fresh_native_prediction',pred,saved['predicted'][index],0.);check(env+'_single_vs_batched_encoder',actual,saved['actual'][index],2e-5)
                        fresh.append({'environment':env,'split':split,'state':src.stem,'candidate_indices':[0,n-1]})
        del model
    assert count==(4 if args.smoke else 402) and candidates==done['candidates_including_nominal']
    for name,h in m['inputs'].items():assert sha256(resolve(name))==h,name
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'states':count,'candidates_including_nominal':candidates,'parent_csv_rows':rows,'max_error':errors,'fresh_forward_checks':fresh,'manifest_sha256':sha256(out/'manifest.json'),'source_sha256':sha256(Path(__file__)),'scope':'All file hashes/shapes/finiteness and physical E from original raw arrays. Fresh history plus nominal/last candidate native predictor and individually encoded actual endpoints in first state per task/split; all candidate forwards not repeated.'})
    print('AUDITED',count,candidates,errors,flush=True)
if __name__=='__main__':main()
