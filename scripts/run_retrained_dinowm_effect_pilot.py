"""Score already executed physical interventions with pinned DINO-WM."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.io import resolve,sha256,write_json
from scripts.retrained_dinowm_core import FrozenRetrainedDinoWM
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_retrained_effect_pilot';read=lambda p:json.loads(p.read_text())


def prepare():
    cfg=read(resolve('configs/followup_dinowm_effect_pilot.json'));OUT.mkdir(exist_ok=True);selected={}
    for env in cfg['tasks']:
        folder=BASE/('family_ablation' if env=='pusht' else 'cube/development');paths=sorted(folder.glob('development*_h05.json' if env=='pusht' else 'e*.json'))
        groups={}
        for p in paths:groups.setdefault(str(read(p)['initial_pose_family']),[]).append(p)
        families=list(groups)[:cfg['families_per_task']];selected[env]=[str(p.relative_to(resolve('.'))) for family in families for p in groups[family]]
        assert len(selected[env])==cfg['families_per_task']*(3 if env=='pusht' else 4)
    paths=[resolve(p) for p in ['configs/followup_dinowm_effect_pilot.json','configs/followup_v1.json','configs/followup_cube.json','scripts/analyze_followup_response_selection.py','scripts/retrained_dinowm_core.py','scripts/run_retrained_dinowm_effect_pilot.py','scripts/analyze_retrained_dinowm_effect_pilot.py']]+[BASE/'dinowm_retrained_preflight/audit.json']
    paths+=list((BASE/'protocol/dinowm_retrained_inventory').glob('*.json'))
    for group in selected.values():
        for name in group:paths += [resolve(name).with_suffix(ext) for ext in ['.json','.csv','.npz']]
    m={'config':cfg,'states':selected,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==m
    else:write_json(OUT/'manifest.json',m)
    return cfg,m


def main():
    cfg,m=prepare();torch.set_num_threads(cfg['torch_threads'])
    for env,paths in m['states'].items():
        model=FrozenRetrainedDinoWM(env);folder=OUT/env;folder.mkdir(exist_ok=True)
        for name in paths:
            path=resolve(name);dest=folder/path.name
            if dest.exists():assert read(dest)['manifest_sha256']==sha256(OUT/'manifest.json');continue
            start=time.perf_counter();meta=read(path)
            with np.load(path.with_suffix('.npz')) as a:
                actions=a['actions'].copy();history=a['history_images'].copy();past=a['past_actions'].copy();endpoints=a['endpoint_images'].copy()
            if env=='pusht':endpoints=endpoints[:,0]
            encoded=model.encode(history);predicted=[];actual=[]
            with torch.no_grad():
                for action,frame in zip(actions,endpoints):
                    predicted.append(model.predict(encoded,past,action)[0].cpu().numpy())
                    pixels=model.model.img_transform(frame)[None,None];actual.append(model.model.encode_visual(pixels)[0,0].cpu().numpy())
            predicted=np.asarray(predicted);actual=np.asarray(actual);assert predicted.shape==actual.shape==(len(actions),196,384)
            np.savez_compressed(dest.with_suffix('.npz'),predicted=predicted,actual=actual)
            write_json(dest,{'environment':env,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'source':name,'candidates_including_nominal':len(actions),'manifest_sha256':sha256(OUT/'manifest.json'),'seconds':time.perf_counter()-start})
            print(env,meta['state_id'],len(actions),'seconds',round(time.perf_counter()-start,2),flush=True)
        del model
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(OUT/'execution_complete.json',{'states':sum(map(len,m['states'].values())),'manifest_sha256':sha256(OUT/'manifest.json')})
if __name__=='__main__':main()
