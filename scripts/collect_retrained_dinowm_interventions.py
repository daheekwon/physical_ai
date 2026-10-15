"""Frozen full-token intervention collection for the common smoothing candidate banks."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from zj_wam.io import resolve,sha256,write_json
from scripts.retrained_dinowm_core import FrozenRetrainedDinoWM
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def prepare(smoke):
    cfg=read(resolve('configs/followup_dinowm_interventions.json'));out=BASE/('dinowm_interventions_smoke' if smoke else 'dinowm_interventions');out.mkdir(exist_ok=True);states={}
    for env in cfg['tasks']:
        states[env]={}
        for split in cfg['splits']:
            folder=BASE/('f1' if env=='pusht' else 'cube/'+split);paths=sorted(folder.glob(split+'*_h05.json' if env=='pusht' else 'e*.json'));assert len(paths)==(105 if env=='pusht' else 96)
            states[env][split]=[str(p.relative_to(resolve('.'))) for p in (paths[:1] if smoke else paths)]
    paths=[resolve(p) for p in ['configs/followup_dinowm_interventions.json','configs/followup_v1.json','configs/followup_cube.json','scripts/collect_retrained_dinowm_interventions.py','scripts/retrained_dinowm_core.py']]+[BASE/'dinowm_readouts/selection.json',BASE/'dinowm_readouts/raw_audit.json',BASE/'dinowm_readouts/pipeline_complete.json']
    for splits in states.values():
        for group in splits.values():
            for name in group:paths.extend(resolve(name).with_suffix(ext) for ext in ['.json','.csv','.npz'])
    record={'smoke':smoke,'config':cfg,'states':states,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==record
    else:write_json(out/'manifest.json',record)
    return out,record


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();out,m=prepare(args.smoke);cfg=m['config'];torch.set_num_threads(cfg['torch_threads']);count=0;total=0
    for env,splits in m['states'].items():
        model=FrozenRetrainedDinoWM(env)
        for split,paths in splits.items():
            folder=out/env/split;folder.mkdir(parents=True,exist_ok=True)
            for name in paths:
                src=resolve(name);dest=folder/src.name
                if dest.exists():
                    meta=read(dest);assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(dest.with_suffix('.npz'));count+=1;total+=meta['candidates'];continue
                start=time.perf_counter();parent=read(src)
                with np.load(src.with_suffix('.npz'),allow_pickle=False) as a:actions=a['actions'].copy();past=a['past_actions'].copy();history=a['history_images'].copy();images=a['endpoint_images'].copy()
                if env=='pusht':images=images[:,0]
                assert len(actions)==len(images);encoded=model.encode(history);predicted=[];actual=[]
                with torch.no_grad():
                    for a in actions:predicted.append(model.predict(encoded,past,a)[0].cpu().numpy())
                    for i in range(0,len(images),cfg['image_batch_size']):
                        pixels=torch.stack([model.model.img_transform(im) for im in images[i:i+cfg['image_batch_size']]])[None]
                        actual.extend(model.model.encode_visual(pixels)[0].cpu().numpy())
                predicted=np.stack(predicted);actual=np.stack(actual);h=encoded[0].cpu().numpy();assert predicted.shape==actual.shape==(len(actions),196,384) and h.shape==(3,196,384)
                assert all(np.isfinite(a).all() for a in [predicted,actual,h]);np.savez_compressed(dest.with_suffix('.npz'),predicted=predicted,actual=actual,history=h)
                write_json(dest,{'source':name,'environment':env,'split':split,'state_id':parent['state_id'],'initial_pose_family':parent['initial_pose_family'],'candidates':len(actions),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'seconds':time.perf_counter()-start});count+=1;total+=len(actions);print(env,split,src.stem,len(actions),round(time.perf_counter()-start,2),flush=True)
        del model
    for name,h in m['inputs'].items():assert sha256(resolve(name))==h,name
    assert count==(4 if args.smoke else 402)
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'states':count,'candidates_including_nominal':total,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Collection complete; independent forward/physical audit and readout analysis required.'})
if __name__=='__main__':main()
