"""F7 matched input-tier response scoring; future ground truth is joined only after scoring."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch,joblib
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.response_models import score_models,estimate_pose,point_geometry,response_score
from zj_wam.model_audit import predict_candidates
from zj_wam.readouts import decode,effects,history_features
from zj_wam.io import resolve,write_json,sha256


def process(path,cfg,rcfg,data,model,readouts,latent_scale,out):
    bank=path.parent.name;dest=out/'response_scores'/bank;dest.mkdir(parents=True,exist_ok=True);key=path.stem
    done=dest/f'{key}.json'
    if done.exists():return
    clock=time.perf_counter()
    # Read only current/past observations and candidate inputs for scores.
    with np.load(path) as a:
        actions=a['actions'];root=a['root'].copy();prefix=a['physics'][0,:30].copy()
        images=a['history_images'];past=a['past_actions']
    spec=json.loads(path.with_suffix('.json').read_text());ep,t=spec['episode'],spec['timestep'];h=actions.shape[1]
    episode_actions,recorded=data.episode(ep);suffix=episode_actions[t+h:t+h+3]
    started=time.perf_counter();gt,gtinfo=score_models(actions,suffix,root,prefix,rcfg);gt_seconds=time.perf_counter()-started
    # The robot's recorded proprioceptive history is available before correction;
    # current pusher state uses exact live root. No future object truth is consumed.
    estimated=[];vision_meta=[]
    for image in images:
        pose,meta=estimate_pose(image,rcfg);estimated.append(pose);vision_meta.append(meta)
    observed=np.zeros((3,10));observed[:,:7]=recorded[[t-10,t-5,t]];observed[-1,:2]=root[:2];observed[-1,5:7]=root[5:7]
    observed[:,2:5]=estimated
    velocity=(np.asarray(estimated)[-1]-estimated[-2])/.5
    velocity[2]=np.arctan2(np.sin(estimated[-1][2]-estimated[-2][2]),np.cos(estimated[-1][2]-estimated[-2][2]))/.5
    observed[-1,7:10]=velocity;vroot=observed[-1].copy()
    started=time.perf_counter();vision,vinfo=score_models(actions,suffix,vroot,observed,rcfg);vision_seconds=time.perf_counter()-started
    valid=all(x['valid'] for x in vision_meta)
    _,context=model.predictor(images,past);started=time.perf_counter()
    predictions=predict_candidates(model,context['embedding'],context['past'],actions)
    torch.cuda.synchronize(model.device) if model.device.startswith('cuda') else None
    wm_seconds=time.perf_counter()-started
    delta=predictions-predictions[0:1]
    scores={'action_norm':np.linalg.norm(((actions-actions[0:1])/data.scaler.scale_).reshape(len(actions),-1),axis=1),
            'controller_pusher':gt['pusher'],'lewm_global':np.linalg.norm(delta,axis=-1).max(1),
            'lewm_whitened':np.linalg.norm(delta/latent_scale,axis=-1).max(1)}
    predicted_pose=decode(readouts['projected_cls'],predictions)
    history=history_features(context['embedding'].cpu().numpy(),predictions)
    predicted_history_pose=decode(readouts['history_projected'],history)
    scores['lewm_object']=response_score(predicted_pose);scores['lewm_history_object']=response_score(predicted_history_pose)
    for tier,values,info in [('state',gt,gtinfo),('vision',vision,vinfo)]:
        for name in ['contact','local','adaptive']:
            scores[tier+'_'+name]=values[name] if tier=='state' or valid else np.full(len(actions),np.inf)
        endpoint=info['fixed'][:,h*10-1:h*10]
        scores[tier+'_contact_endpoint']=response_score(endpoint) if tier=='state' or valid else np.full(len(actions),np.inf)
    # Outcomes below are evaluation-only and were not supplied to score_models.
    source=pd.read_csv(path.with_suffix('.csv'));indices=source['index'].to_numpy(int)
    for name,values in scores.items():source[name+'_score']=values[indices]
    for tier,info in [('state',gtinfo),('vision',vinfo)]:
        source[tier+'_excitation']=info['past']['excitation']
        source[tier+'_translation_gain']=info['past']['gains'][0];source[tier+'_rotation_gain']=info['past']['gains'][1]
    source['vision_valid']=valid
    gap,_=point_geometry(vroot[None,:2],vroot[None,2:5]);source['observed_gap']=float(gap[0]-rcfg['pusher_radius'])
    q=np.concatenate([np.broadcast_to(prefix[None,:,:2],(len(actions),30,2)),gtinfo['pusher']],axis=1)
    cost=np.sum(np.diff(q,n=3,axis=1)**2,axis=(1,2))/.01**5
    source['predicted_jerk']=cost[indices]
    source['nominal_predicted_jerk']=cost[0]
    source.to_csv(dest/f'{key}.csv',index=False)
    np.savez_compressed(dest/f'{key}.npz',predictions=predictions,predicted_pose=predicted_pose,predicted_history_pose=predicted_history_pose,
                        state_fixed=gtinfo['fixed'],state_adaptive=gtinfo['adaptive'],state_local=gtinfo['local'],
                        vision_fixed=vinfo['fixed'],vision_adaptive=vinfo['adaptive'],vision_local=vinfo['local'],
                        vision_pose_history=np.array(estimated),observed_history=observed)
    vision_error=np.asarray(estimated[-1])-root[2:5];vision_error[2]=np.arctan2(np.sin(vision_error[2]),np.cos(vision_error[2]))
    write_json(done,{'state_id':spec['state_id'],'split':spec['split'],'initial_pose_family':spec['initial_pose_family'],'horizon':h,'bank':bank,
                    'rows':len(source),'vision_valid':valid,'vision_position_error':float(np.linalg.norm(vision_error[:2])),
                    'vision_rotation_error_deg':float(np.rad2deg(abs(vision_error[2]))),'vision_metadata':vision_meta,
                    'seconds':{'state_models':gt_seconds,'vision_models':vision_seconds,'wm_batch_predict':wm_seconds,'total':time.perf_counter()-clock},
                    'input_tiers':{'state':rcfg['state_tier'],'vision':rcfg['vision_tier']},
                    'source_sha256':sha256(Path(__file__)),'response_model_sha256':sha256(resolve('zj_wam/response_models.py'))})
    print(json.dumps({'state':key,'bank':bank,'rows':len(source),'seconds':time.perf_counter()-clock}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='development',choices=['development','calibration','test']);p.add_argument('--bank',default='interventions',choices=['interventions','f1','family_ablation']);p.add_argument('--limit',type=int)
    args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());out=resolve(cfg['output'])
    if args.split=='test' and not (out/'protocol/final_selection_lock.json').exists():raise ValueError('Test lock required')
    torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    readouts={name:joblib.load(out/f'lewm_audit/readouts/{name}_ridge.joblib') for name in ['projected_cls','history_projected']}
    path=out/'response_scores/latent_scale.npy';path.parent.mkdir(exist_ok=True)
    if path.exists():latent_scale=np.load(path)
    else:
        latents=np.concatenate([np.load(p)['projected_cls'] for p in sorted((out/'lewm_audit/representations').glob('probe_train_*.npz'))])
        latent_scale=np.maximum(latents.std(0),1e-3);np.save(path,latent_scale)
    paths=sorted(p for p in (out/args.bank).glob(f'{args.split}_*_h05.npz') if p.with_suffix('.json').exists())
    expected=45 if args.split=='test' else 105
    if not args.limit and len(paths)!=expected:raise ValueError(f'Incomplete source bank: {len(paths)}/{expected}')
    if args.limit:paths=paths[:args.limit]
    for path in paths:
        if path.with_suffix('.json').exists():process(path,cfg,rcfg,data,model,readouts,latent_scale,out)

if __name__=='__main__':main()
