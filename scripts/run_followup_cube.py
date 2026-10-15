"""Cube native expert bank with frozen LeWM, explicit family splits and dense traces."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py,hdf5plugin,numpy as np,torch,pandas as pd
from torchvision.transforms import v2 as T
from zj_wam.io import resolve,write_json,sha256
from zj_wam.cube_execution import make_env,initialize_recorded,snapshot,restore,DenseCube
from zj_wam.cube_experiment import basis,xyz_yaw,fit_response,generate_candidates,metrics


def signature():
    files=['configs/followup_cube.json','zj_wam/cube_execution.py','zj_wam/cube_experiment.py','scripts/run_followup_cube.py']
    return {p:sha256(resolve(p)) for p in files}


def phases(f,offset,n,cfg):
    contact=f['proprio_gripper_contact'][offset:offset+n,0]>.1;z=f['privileged_block_0_pos'][offset:offset+n,2]
    possible=np.flatnonzero(contact & (np.arange(n)>=10)&(np.arange(n)<n-15))
    if not len(possible):raise ValueError('No eligible source grasp phase; never silently exclude based on corrections')
    onset=int(possible[0]);endpoints=np.flatnonzero((~contact)&(np.arange(n)>onset+5))
    release=int(endpoints[0]) if len(endpoints) else n-9
    transport=onset+int(np.argmax(z[onset:release+1]));last=n-cfg['horizon']-cfg['suffix']-1
    return {k:int(np.clip(t,10,last)) for k,t in {'approach':onset-5,'grasp':onset+2,'transport':transport,'release':release-2}.items()}


def prepare(cfg,out):
    p=out/'manifest.json';sig=signature()
    if p.exists():
        m=json.loads(p.read_text());assert m['signature']==sig,'Source changed: explicitly version or archive prior output';return m
    dataset=json.loads(resolve('artifacts/followup_v1/cube_preflight/dataset_manifest.json').read_text());archive=json.loads(resolve('artifacts/followup_v1/protocol/cube_archive_audit.json').read_text());assert archive['official_lfs_hash_match']
    rng=np.random.default_rng(cfg['seed']);eligible=np.setdiff1d(np.arange(dataset['episodes']),cfg['excluded_pilot_episodes']);rng.shuffle(eligible);splits={};cursor=0
    for split,count in cfg['split_episodes'].items():splits[split]=sorted(eligible[cursor:cursor+count].tolist());cursor+=count
    with h5py.File(resolve(cfg['dataset']),'r') as f:
        offsets=f['ep_offset'][:];initial=np.c_[f['qpos'][offsets],f['qvel'][offsets]];seen={}
        for split,episodes in splits.items():
            for ep in episodes:
                family=tuple(np.round(initial[ep],8));assert family not in seen,(ep,seen.get(family));seen[family]=split
        phase_map={str(ep):phases(f,int(offsets[ep]),int(f['ep_len'][ep]),cfg) for split,eps in splits.items() if split!='test' for ep in eps}
    m={'signature':sig,'config':cfg,'splits':splits,'phases':phase_map,'normalization':{'mean':dataset['action_mean'],'scale':dataset['action_scale']},'checkpoint_sha256':sha256(resolve(cfg['checkpoint'])/'weights.pt'),'dataset_archive_sha256':archive['compressed_sha256'],'family_definition':'Unique initial qpos/qvel rounded1e-8; sameepisode phases clustered for uncertainty. WM pretraining exposure separate. Pilot0..5 excluded. Test phases/results not inspected.','created_unix':time.time()};write_json(p,m);return m


class Model:
    def __init__(self,cfg,mean,scale):
        from stable_worldmodel.wm.utils import load_pretrained
        self.cfg=cfg;self.mean=mean;self.scale=scale;self.model=load_pretrained(str(resolve(cfg['checkpoint']))).to(cfg['device']).eval().requires_grad_(False)
        self.transform=T.Compose([T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225]),T.Resize(224)])
    def encode(self,images):
        with torch.no_grad():
            frames=torch.stack([self.transform(im) for im in images])[None].to(self.cfg['device']);return self.model.encode({'pixels':frames})['emb'].detach()
    def predictor(self,images,past):
        embedding=self.encode(images);old=torch.tensor(((past-self.mean)/self.scale).astype(np.float32).reshape(1,2,25),device=self.cfg['device'])
        def predict(x):
            action=torch.cat([old,x.reshape(1,1,25)],1);return self.model.predict(embedding,self.model.action_encoder(action))[:,-1].reshape(-1)
        return embedding,predict


def state_bank(env,dense,root,history,images,past,nominal,suffix,cfg,model,out,key,meta,sig):
    done=out/(key+'.json')
    if done.exists():assert json.loads(done.read_text())['signature']==sig;return
    start=time.perf_counter();restore(env,root);q=basis();mean=model.mean;scale=model.scale;points=xyz_yaw(history);targets=history['targets'];gains,residual=fit_response(points[-251:],targets[-250:],cfg['physics_dt']);velocity=(points[-1]-points[-2])/cfg['physics_dt'];prefix={k:v[-75:] for k,v in history.items() if k!='targets'};prefixpoints=points[-75:];embedding,predict=model.predictor(images,past);x0=((nominal-mean)/scale).reshape(-1)
    bank=generate_candidates(nominal,suffix,scale,q,predict,x0,prefixpoints,points[-1],velocity,gains,env._workspace_bounds,cfg,np.random.default_rng(np.random.SeedSequence([cfg['seed'],meta['episode'],meta['timestep']])))
    raw={};outputs=[];endimages=[];startpose=dense.measure();root_gripper_contact=float(np.asarray(env.compute_ob_info()['proprio/gripper_contact']).reshape(-1)[0]);nominal_trace=None;endpoint_measures=[]
    for i,a in enumerate(bank['actions']):
        restore(env,root);first=dense.rollout(a);endimages.append(env.render().copy());endpoint_measures.append(dense.measure());last=dense.rollout(suffix);trace={k:np.r_[first['physics'][k],last['physics'][k]] for k in first['physics']}
        if i==0:nominal_trace=trace
        row={**meta,**bank['metadata'][i],'candidate':i,**metrics(trace,nominal_trace,prefix,cfg),'norm':float(np.linalg.norm((a-nominal)/scale)),'lewm_score':float(bank['lewm_scores'][i]),'predicted_jerk_cost':float(bank['costs'][i])};outputs.append(row)
        for k,v in trace.items():raw.setdefault('candidate_'+k,[]).append(v)
    for k in raw:raw[k]=np.asarray(raw[k])
    actual_embeddings=model.encode(np.stack(endimages)).cpu().numpy()[0]
    raw.update({'actions':bank['actions'],'nominal_actions':nominal,'suffix_actions':suffix,'polynomial_basis':q,'coefficients':bank['coefficients'],'history_images':images,'history_embedding':embedding.cpu().numpy()[0],'past_actions':past,'past_points':points[-251:],'past_targets':targets[-250:],'response_gains':gains,'response_fit_rmse':residual,'predicted_robot':bank['predicted'],'predicted_latent':bank['latent'],'actual_endpoint_latent':actual_embeddings,'endpoint_images':np.stack(endimages),'jacobian':bank['jacobian'],'singular_values':bank['singular_values'],'root_object_pose':startpose['object_pose'],'root_eef_position':startpose['eef_position'],'root_velocity':startpose['object_velocity'],'root_gripper_contact':root_gripper_contact,'workspace_bounds':env._workspace_bounds})
    for k,v in prefix.items():raw['prefix_'+k]=v
    pd.DataFrame(outputs).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**raw);write_json(done,{'signature':sig,**meta,'candidates':len(outputs),'rejected_candidates':bank['rejected'],'generation_seconds':bank['seconds'],'seconds':time.perf_counter()-start,'phase_is_source_defined':True,'native_history_steps':[meta['timestep']-10,meta['timestep']-5,meta['timestep']],'gripper_actions_unchanged':bool(np.array_equal(bank['actions'][:,:,4],np.broadcast_to(nominal[:,4],bank['actions'][:,:,4].shape))),'future_physics_used_for_selection':False});print(key,len(outputs),'seconds',round(time.perf_counter()-start,2),flush=True)


def run_episode(f,ep,split,cfg,manifest,model,out,limit_states=None):
    offset=int(f['ep_offset'][ep]);n=int(f['ep_len'][ep]);a=f['action'][offset:offset+n-1];selected=manifest['phases'][str(ep)] if split!='pilot' else phases(f,offset,n,cfg)
    if limit_states:selected=dict(list(selected.items())[:limit_states])
    if all((out/f'e{ep:05d}_{phase}_t{t:03d}.json').exists() for phase,t in selected.items()):return
    env=make_env(cfg['seed']+ep);initialize_recorded(env,f['qpos'][offset],f['qvel'][offset],f['control'][offset],f['time'][offset,0],f['privileged_target_block_pos'][offset]);dense=DenseCube(env);initial=dense.measure();allphys={k:[v] for k,v in initial.items()};targets=[];images={};snapshots={};needed=set(t-d for t in selected.values() for d in [10,5,0]);images[0]=env.render().copy()
    for t in range(max(selected.values())+1):
        if t in needed:images[t]=env.render().copy()
        if t in selected.values():snapshots[t]=snapshot(env)
        if t==max(selected.values()):break
        result=dense.rollout(a[t:t+1])['physics']
        for k,v in result.items():allphys[k].extend(v)
        target=np.r_[env._target_effector_pose.translation(),env._target_effector_pose.rotation().compute_yaw_radians()];targets.extend([target.copy()]*cfg['substeps'])
    allphys={k:np.asarray(v) for k,v in allphys.items()};targets=np.asarray(targets)
    for phase,t in selected.items():
        key=f'e{ep:05d}_{phase}_t{t:03d}';done=out/(key+'.json');meta={'episode':ep,'initial_pose_family':ep,'split':split,'phase':phase,'timestep':t,'state_id':key};history={k:v[:t*25+1] for k,v in allphys.items()};history['targets']=targets[:t*25];frames=np.stack([images[t-d] for d in [10,5,0]])
        if split=='probe_train':
            if done.exists():continue
            restore(env,snapshots[t]);current=dense.measure();embedding=model.encode(frames).cpu().numpy()[0];dense.rollout(a[t:t+5]);future=dense.measure();futureimage=env.render().copy();latent=model.encode([futureimage]).cpu().numpy()[0,0];data={'current_embedding':embedding[-1],'future_embedding':latent,'history_embedding':embedding,'images':frames,'future_image':futureimage}
            for k in ['object_pose','object_velocity','eef_position']:data['current_'+k]=current[k];data['future_'+k]=future[k]
            np.savez_compressed(done.with_suffix('.npz'),**data);write_json(done,{**meta,'signature':manifest['signature']});print(key,'probe_train',flush=True)
        else:state_bank(env,dense,snapshots[t],history,frames,a[t-10:t],a[t:t+5],a[t+5:t+8],cfg,model,out,key,meta,manifest['signature'])
    env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['pilot','probe_train','development','calibration'],required=True);p.add_argument('--limit',type=int);p.add_argument('--limit-states',type=int);args=p.parse_args();torch.set_num_threads(4);cfg=json.loads(resolve('configs/followup_cube.json').read_text());out=resolve(cfg['output']);out.mkdir(exist_ok=True);manifest=prepare(cfg,out);episodes=cfg['excluded_pilot_episodes'] if args.split=='pilot' else manifest['splits'][args.split]
    if args.limit:episodes=episodes[:args.limit]
    folder=out/args.split;folder.mkdir(exist_ok=True);mean=np.asarray(manifest['normalization']['mean']);scale=np.asarray(manifest['normalization']['scale']);model=Model(cfg,mean,scale)
    with h5py.File(resolve(cfg['dataset']),'r') as f:
        for ep in episodes:run_episode(f,ep,args.split,cfg,manifest,model,folder,args.limit_states)
if __name__=='__main__':main()
