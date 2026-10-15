"""Check capture label alignment, pooling identities, and a fresh encoder sample."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from zj_wam.io import resolve,write_json,sha256
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import DenseSimulator
from scripts.retrained_dinowm_core import FrozenRetrainedDinoWM
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();out=BASE/('dinowm_capture_smoke' if args.smoke else 'dinowm_capture');m=read(out/'manifest.json');done=read(out/'execution_complete.json');assert done['smoke']==args.smoke and done['manifest_sha256']==sha256(out/'manifest.json')
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    errors={};records=0;labels=0;families={};first={};roundoff_failures=[];roundoff_rechecks=[]
    def check(name,x,y,tol=1e-7):
        err=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),err);assert err<=tol,(name,err)
    for env,splits in m['states'].items():
        families[env]={}
        for split,paths in splits.items():
            group=set()
            for name in paths:
                src=resolve(name);path=out/env/src.name if env=='pusht' else out/env/split/src.name;meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['source']==name
                with np.load(path.with_suffix('.npz')) as a,np.load(src.with_suffix('.npz')) as parent:
                    n=len(a['targets']);records+=1;labels+=n;group.update(a['family'].tolist());first.setdefault(env,(src,path,split))
                    for key,width in [('patch_mean',384),('spatial_quadrants',1536),('history_patch_mean',1152)]:assert a[key].shape==(n,width) and np.isfinite(a[key]).all()
                    check('history_last_is_current',a['history_patch_mean'][:,-384:],a['patch_mean'],0.)
                    qmean=a['spatial_quadrants'].reshape(n,4,384).mean(1);reference=a['patch_mean']
                    errors['equal_quadrants_average_absolute']=max(errors.get('equal_quadrants_average_absolute',0.),float(np.max(np.abs(qmean-reference))))
                    # Equivalent FP32 reductions use different accumulation orders.
                    # Use elementwise relative+absolute tolerance, not a fixed
                    # absolute threshold independent of token magnitude.
                    ratio=float(np.max(np.abs(qmean-reference)/(1e-6+1e-6*np.abs(reference))))
                    errors['quadrant_roundoff_normalized']=max(errors.get('quadrant_roundoff_normalized',0.),ratio)
                    if ratio>1:roundoff_failures.append((env,src,path))
                    if env=='pusht':
                        check('parent_targets',a['targets'],parent['targets'],0.);check('parent_full_state',a['full_state'],parent['full_state'],0.);check('parent_timestamps',a['timesteps'],parent['timesteps'],0.)
                        assert meta['physical_labels_exact']
                    else:
                        # Position and EEF targets are simple direct measurements;
                        # the SO3 target remains the frozen prior target function.
                        if split=='probe_train':pos=np.stack([parent['current_object_pose'][:3],parent['future_object_pose'][:3]])
                        else:pos=np.stack([parent['root_object_pose'][:3],parent['candidate_object_pose'][0,124,:3]])
                        check('cube_position_targets',a['targets'][:,:3],pos,0.)
                    assert np.isfinite(a['targets']).all()
            families[env][split]=sorted(group)
        keys=list(splits)
        for i,x in enumerate(keys):
            for y in keys[i+1:]:assert not set(families[env][x])&set(families[env][y]),(env,x,y)
    assert records==(6 if args.smoke else 618)
    torch.set_num_threads(2);main=read(resolve('configs/followup_v1.json'));data=ExpertDataset(main['dataset'])
    for env,(src,path,split) in first.items():
        model=FrozenRetrainedDinoWM(env)
        if env=='pusht':
            meta=read(src)
            with np.load(src.with_suffix('.npz')) as a:t=int(a['timesteps'][0])
            actions,states=data.episode(meta['episode']);sim=DenseSimulator(actions,states,t,main['seed']+meta['episode']);images,_=sim.restore(capture=True);sim.close()
        else:
            with np.load(src.with_suffix('.npz')) as a:images=a['images']
        with torch.no_grad():
            # Direct encoder forward, separate from batched capture wrapper.
            pixels=model.pixels(images);x=model.model.wm.encoder_transform(pixels[0]);tokens=model.model.wm.encoder(x).cpu().numpy();mean=tokens.mean(1)
        with np.load(path.with_suffix('.npz')) as a:
            check(env+'_fresh_current_forward',mean[-1],a['patch_mean'][0],2e-5);check(env+'_fresh_history_forward',mean.reshape(-1),a['history_patch_mean'][0],2e-5)
        for failed_env,failed_src,failed_path in roundoff_failures:
            if failed_env!=env:continue
            assert env=='pusht','Unanticipated failing environment requires its own reproduction'
            meta=read(failed_src)
            with np.load(failed_path.with_suffix('.npz')) as a:times=a['timesteps'].copy()
            wanted=sorted({int(t-k) for t in times for k in [10,5,0]});actions,states=data.episode(meta['episode']);sim=DenseSimulator(actions,states,0,main['seed']+meta['episode']);sim.restore();images=[]
            for t in range(max(wanted)+1):
                if t in wanted:images.append(sim.env.render().copy())
                if t<max(wanted):sim.env.step(actions[t])
            sim.close();lookup={t:i for i,t in enumerate(wanted)};ids=[lookup[int(t)] for t in times];tokens=[]
            with torch.no_grad():
                for offset in range(0,len(images),8):
                    pixels=torch.stack([model.model.img_transform(im) for im in images[offset:offset+8]])[None]
                    tokens.append(model.model.encode_visual(pixels)[0].cpu().numpy())
            z=np.concatenate(tokens)[ids];grid=z.reshape(len(z),14,14,384);means=z.mean(1);quadrants=np.concatenate([grid[:,i:i+7,j:j+7].mean((1,2)) for i in [0,7] for j in [0,7]],axis=1)
            with np.load(failed_path.with_suffix('.npz')) as a:
                check('failed_records_fresh_mean',means,a['patch_mean'],0.);check('failed_records_fresh_quadrants',quadrants,a['spatial_quadrants'],0.)
            exact=z.astype(np.float64).mean(1);doublegrid=z.astype(np.float64).reshape(len(z),14,14,384);doublequad=np.stack([doublegrid[:,i:i+7,j:j+7].mean((1,2)) for i in [0,7] for j in [0,7]],axis=1).mean(1)
            check('failed_records_float64_identity',exact,doublequad,1e-12)
            # Standard floating point summation/division bound using observed
            # absolute tokens; unlike an output-relative tolerance it handles cancellation.
            u=2.**-24;gamma=lambda n:n*u/(1-n*u);bound=(gamma(197)+gamma(50)+gamma(4)+gamma(50)*gamma(4))*np.abs(z.astype(float)).mean(1)+np.finfo(np.float32).tiny
            discrepancy=np.abs(means.astype(float)-quadrants.reshape(len(z),4,384).mean(1).astype(float));ratio=float(np.max(discrepancy/bound));assert ratio<=1,ratio
            errors['failed_records_roundoff_bound_ratio']=max(errors.get('failed_records_roundoff_bound_ratio',0.),ratio)
            roundoff_rechecks.append({'path':str(failed_path.relative_to(resolve('.'))),'labels':len(z),'fresh_mean_and_quadrants_exact':True,'maximum_roundoff_bound_ratio':ratio})
            print('RECHECKED',failed_path.stem,ratio,flush=True)
        del model
    assert len(roundoff_rechecks)==len(roundoff_failures)
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'source_records':records,'labels':labels,'families':{env:{s:len(v) for s,v in groups.items()} for env,groups in families.items()},'max_error':errors,'manifest_sha256':sha256(out/'manifest.json'),'source_sha256':sha256(Path(__file__)),'roundoff_rechecks':roundoff_rechecks,'original_auditor_sha256':sha256(resolve('scripts/audit_retrained_dinowm_capture.py')),'scope':'All saved labels/dimensions/finiteness/splitdisjointness/pooling identities; original FP32 pooling tolerance failures re-encoded exactly and checked with tokenwise roundoff bound; firsttrainfamily pertask freshdirectencoder replay vs batchedfeatures. PushT allphysical labels matched parent duringcollection. No readout fitting or accuracyclaim.'})
    print('AUDITED DINO capture',records,'records',labels,'labels',flush=True)
if __name__=='__main__':main()
