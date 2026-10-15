"""Collect frozen DINO image features on existing probe splits and labels."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import DenseSimulator
from zj_wam.cube_readouts import target
from scripts.retrained_dinowm_core import FrozenRetrainedDinoWM
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def features(model,images,batch):
    latent=[]
    with torch.no_grad():
        for i in range(0,len(images),batch):
            x=torch.stack([model.model.img_transform(im) for im in images[i:i+batch]])[None]
            latent.append(model.model.encode_visual(x)[0].cpu().numpy())
    z=np.concatenate(latent);assert z.shape[1:]==(196,384);grid=z.reshape(len(z),14,14,384)
    quadrants=np.concatenate([grid[:,i:i+7,j:j+7].mean((1,2)) for i in [0,7] for j in [0,7]],axis=1)
    return {'patch_mean':z.mean(1),'spatial_quadrants':quadrants}


def prepare(smoke):
    cfg=read(resolve('configs/followup_dinowm_capture.json'));main=read(resolve('configs/followup_v1.json'));out=BASE/('dinowm_capture_smoke' if smoke else 'dinowm_capture');out.mkdir(exist_ok=True);selected={}
    for env in cfg['tasks']:
        selected[env]={}
        for split in cfg['splits']:
            folder=BASE/('lewm_audit/representations' if env=='pusht' else 'cube/'+split)
            paths=sorted(folder.glob(split+'_*.json' if env=='pusht' else 'e*.json'));assert len(paths)==({'probe_train':100,'development':35,'calibration':35}[split] if env=='pusht' else {'probe_train':256,'development':96,'calibration':96}[split])
            selected[env][split]=[str(p.relative_to(resolve('.'))) for p in (paths[:1] if smoke else paths)]
    paths=[Path(__file__),resolve('configs/followup_dinowm_capture.json'),resolve('configs/followup_v1.json'),resolve('scripts/retrained_dinowm_core.py'),resolve('zj_wam/cube_readouts.py'),BASE/'dinowm_retrained_preflight/audit.json']
    for env,splits in selected.items():
        for group in splits.values():
            for name in group:paths += [resolve(name),resolve(name).with_suffix('.npz')]
    m={'smoke':smoke,'config':cfg,'states':selected,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==m
    else:write_json(out/'manifest.json',m)
    return cfg,main,out,m


def pusht(path,cfg,main,out,model,data):
    spec=read(path);dest=out/'pusht'/path.name;dest.parent.mkdir(exist_ok=True)
    if dest.exists():assert read(dest)['manifest_sha256']==sha256(out/'manifest.json');return
    start=time.perf_counter()
    with np.load(path.with_suffix('.npz')) as old:times=old['timesteps'].copy();truth=old['full_state'].copy();targets=old['targets'].copy()
    wanted=sorted({int(t-k) for t in times for k in [10,5,0]});a,s=data.episode(spec['episode']);sim=DenseSimulator(a,s,0,main['seed']+spec['episode']);sim.restore();images=[];states=[]
    for t in range(max(wanted)+1):
        if t in wanted:images.append(sim.env.render().copy());states.append(sim.full_observation())
        if t<max(wanted):sim.env.step(a[t])
    sim.close();lookup={t:i for i,t in enumerate(wanted)};ids=[lookup[int(t)] for t in times];history=np.array([[lookup[int(t-k)] for k in [10,5,0]] for t in times]);full=np.asarray(states)[ids];np.testing.assert_array_equal(full,truth)
    f=features(model,images,cfg['batch_frames']);arrays={k:v[ids] for k,v in f.items()};arrays['history_patch_mean']=f['patch_mean'][history].reshape(len(times),-1)
    np.savez_compressed(dest.with_suffix('.npz'),**arrays,targets=targets,full_state=full,timesteps=times,family=np.repeat(spec['initial_pose_family'],len(times)))
    write_json(dest,{'environment':'pusht','split':spec['split'],'initial_pose_family':spec['initial_pose_family'],'source':str(path.relative_to(resolve('.'))),'labels':len(times),'encoded_frames':len(images),'physical_labels_exact':True,'manifest_sha256':sha256(out/'manifest.json'),'seconds':time.perf_counter()-start});print('pusht',path.stem,round(time.perf_counter()-start,2),flush=True)


def cube(path,split,cfg,out,model):
    meta=read(path);folder=out/'ogb-cube'/split;folder.mkdir(parents=True,exist_ok=True);dest=folder/path.name
    if dest.exists():assert read(dest)['manifest_sha256']==sha256(out/'manifest.json');return
    start=time.perf_counter()
    with np.load(path.with_suffix('.npz')) as a:
        if split=='probe_train':
            history=a['images'];future=a['future_image'];labels=np.stack([target(a[k+'_object_pose'][None],a[k+'_object_velocity'][None],a[k+'_eef_position'][None])[0] for k in ['current','future']])
        else:
            history=a['history_images'];future=a['endpoint_images'][0];labels=np.stack([target(a['root_object_pose'][None],a['root_velocity'][None],a['root_eef_position'][None])[0],target(a['candidate_object_pose'][0,124][None],a['candidate_object_velocity'][0,124][None],a['candidate_eef_position'][0,124][None])[0]])
    f=features(model,list(history)+[future],cfg['batch_frames']);arrays={k:v[[2,3]] for k,v in f.items()};arrays['history_patch_mean']=f['patch_mean'][[[0,1,2],[1,2,3]]].reshape(2,-1)
    np.savez_compressed(dest.with_suffix('.npz'),**arrays,targets=labels,family=np.repeat(meta['initial_pose_family'],2))
    write_json(dest,{'environment':'ogb-cube','split':split,'initial_pose_family':meta['initial_pose_family'],'source':str(path.relative_to(resolve('.'))),'labels':2,'encoded_frames':4,'source_labels_reused':True,'manifest_sha256':sha256(out/'manifest.json'),'seconds':time.perf_counter()-start});print('cube',split,path.stem,round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();cfg,main,out,m=prepare(args.smoke);torch.set_num_threads(cfg['torch_threads']);total=0
    for env,splits in m['states'].items():
        model=FrozenRetrainedDinoWM(env);data=ExpertDataset(main['dataset']) if env=='pusht' else None
        for split,group in splits.items():
            for name in group:
                if env=='pusht':pusht(resolve(name),cfg,main,out,model,data)
                else:cube(resolve(name),split,cfg,out,model)
                total+=1
        del model
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'source_records':total,'manifest_sha256':sha256(out/'manifest.json')})
if __name__=='__main__':main()
