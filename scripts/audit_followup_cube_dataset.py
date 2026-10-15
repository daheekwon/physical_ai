"""Official Cube dataset statistics, expert replay and contact-rich branch restoration."""
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py,hdf5plugin,numpy as np,torch
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as T
from zj_wam.cube_execution import make_env,initialize_recorded,snapshot,restore,DenseCube,quat_error
from zj_wam.io import resolve,write_json,sha256


def main():
    started=time.perf_counter();base=resolve('artifacts/followup_v1');archive=json.loads((base/'protocol/cube_archive_audit.json').read_text());assert archive['official_lfs_hash_match'];path=resolve('data/datasets/lewm-cube/cube_single_expert.h5');assert path.stat().st_size==archive['members'][0]['bytes'];out=base/'cube_preflight';out.mkdir(exist_ok=True)
    with h5py.File(path,'r') as f:
        lengths=f['ep_len'][:];offsets=f['ep_offset'][:];actions=f['action'][:];finite=np.isfinite(actions).all(1);scaler=StandardScaler().fit(actions[finite]);initial=f['qpos'][offsets];initialv=f['qvel'][offsets];family=np.unique(np.round(np.concatenate([initial,initialv],axis=1),8),axis=0)
        schema={k:{'shape':list(f[k].shape),'dtype':str(f[k].dtype)} for k in f};stats={'dataset':str(path),'archive_sha256':archive['compressed_sha256'],'episodes':len(lengths),'rows':len(actions),'unique_initial_qpos_qvel_rounded_1e8':len(family),'episode_length_range':[int(lengths.min()),int(lengths.max())],'finite_action_rows':int(finite.sum()),'nonfinite_action_rows':int((~finite).sum()),'action_mean':scaler.mean_.tolist(),'action_scale':scaler.scale_.tolist(),'action_abs_max':float(np.abs(actions[finite]).max()),'schema':schema,'probe_episodes':[0,1,2,3,4,5],'exposure':'Episodes0..5usedfor interface/replay development; exclude their initial families from downstreamtest. Model pretraining exposure separate. Full dataset normalization matches official evaluator.'};write_json(out/'dataset_manifest.json',stats)
        rows=[];branches=[];model_inputs=None
        for episode in stats['probe_episodes']:
            offset=int(offsets[episode]);n=int(lengths[episode]);a=actions[offset:offset+n-1];qpos=f['qpos'][offset:offset+n];qvel=f['qvel'][offset:offset+n];control=f['control'][offset:offset+n];target=f['privileged_target_block_pos'][offset];time0=float(f['time'][offset,0]);eef=f['proprio_effector_pos'][offset:offset+n];cube=f['privileged_block_0_pos'][offset:offset+n];quat=f['privileged_block_0_quat'][offset:offset+n];contact=f['proprio_gripper_contact'][offset:offset+n,0]
            env=make_env(20261011+episode);initialize_recorded(env,qpos[0],qvel[0],control[0],time0,target);dense=DenseCube(env);trace=[dense.measure()];images={0:env.render().copy()};cached_eef=[env._data.site_xpos[env._pinch_site_id].copy()];saved={};branch_candidates={'approach':10,'max_height':int(np.argmax(cube[:n-8,2])),'max_gripper_contact':int(np.argmax(contact[:n-8]))}
            # Keep enough observed history and5future+3suffix; labels select physical phases, not correction outcomes.
            times=sorted(set(max(10,min(n-9,t)) for t in branch_candidates.values()))
            for t in range(len(a)):
                if t in times:saved[t]=snapshot(env)
                env.step(a[t]);trace.append(dense.measure());cached_eef.append(env._data.site_xpos[env._pinch_site_id].copy())
                if t+1 in [5,10,15,20,50,100,200]:images[t+1]=env.render().copy()
            actualq=np.array([r['qpos'] for r in trace]);actualv=np.array([r['qvel'] for r in trace]);actualpose=np.array([r['object_pose'] for r in trace]);acteef=np.array(cached_eef);pixels=f['pixels'][offset+np.array(sorted(images))];rendered=np.stack([images[t] for t in sorted(images)]);position=np.linalg.norm(actualpose[:,:3]-cube,axis=1);rotation=quat_error(actualpose[:,3:],quat)
            row={'episode':episode,'actions':len(a),'qpos_max_error':float(np.abs(actualq-qpos).max()),'qvel_max_error':float(np.abs(actualv-qvel).max()),'object_position_max_error_m':float(position.max()),'object_rotation_max_error_deg':float(np.rad2deg(rotation.max())),'eef_cached_max_error_m':float(np.linalg.norm(acteef-eef,axis=1).max()),'same_replay_image_mae_255':float(np.abs(rendered.astype(float)-pixels.astype(float)).mean()),'robot_object_contact_control_steps':int(sum(r['robot_object_contacts']>0 for r in trace)),'dataset_object_height_max_m':float(cube[:,2].max()),'actual_object_height_max_m':float(actualpose[:,2].max())};rows.append(row)
            np.savez_compressed(out/f'episode_{episode}.npz',qpos=actualq,qvel=actualv,object_pose=actualpose,cached_eef=acteef,source_qpos=qpos,source_qvel=qvel,source_object_position=cube,source_object_quaternion=quat,source_images=pixels,replay_images=rendered,image_steps=sorted(images),actions=a)
            for t in times:
                sequence=a[t:t+8];restore(env,saved[t]);first=dense.rollout(sequence,dense=True);restore(env,saved[t]);second=dense.rollout(sequence,dense=True);restore(env,saved[t]);native=dense.rollout(sequence,dense=False)
                repeat=max(float(np.abs(first['physics'][k]-second['physics'][k]).max()) for k in first['physics']);instrument=max(float(np.abs(first['control'][k]-native['control'][k]).max()) for k in first['control']);assert repeat<1e-10 and instrument<1e-10,(episode,t,repeat,instrument)
                branch={'episode':episode,'timestep':t,'repeat_max_error':repeat,'dense_instrumentation_control_max_error':instrument,'physics_samples':len(first['physics']['time']),'robot_object_contact_substeps':int(np.count_nonzero(first['physics']['robot_object_contacts']))};branches.append(branch)
                np.savez_compressed(out/f'branch_e{episode}_t{t}.npz',**first['physics'],actions=sequence)
            if model_inputs is None:model_inputs=(np.stack([images[k] for k in [0,5,10]]),a[:10],a[10:15])
            env.close();print(row,flush=True)
    # Generic5D native predictor, with official full-dataset normalization and real replay history.
    from stable_worldmodel.wm.utils import load_pretrained
    torch.set_num_threads(4);model=load_pretrained(str(resolve('data/checkpoints/lewm-cube'))).to('cuda:2').eval().requires_grad_(False);transform=T.Compose([T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225]),T.Resize(224)]);images,past,future=model_inputs;frames=torch.stack([transform(im) for im in images])[None].to('cuda:2')
    with torch.no_grad():embedding=model.encode({'pixels':frames})['emb'].detach()
    old=torch.from_numpy(scaler.transform(past).astype(np.float32).reshape(1,2,25)).to('cuda:2');x=torch.tensor(scaler.transform(future).astype(np.float32).reshape(-1),device='cuda:2',requires_grad=True)
    def predict(z):return model.predict(embedding,model.action_encoder(torch.cat([old,z.reshape(1,1,25)],1)))[:,-1].reshape(-1)
    y=predict(x);grad=torch.autograd.grad(y.square().sum(),x)[0];assert torch.isfinite(y).all() and torch.isfinite(grad).all() and grad.abs().max()>0
    write_json(out/'audit.json',{'archive_verified':True,'dataset_manifest':'dataset_manifest.json','replay':rows,'branches':branches,'model':{'checkpoint_sha256':sha256(resolve('data/checkpoints/lewm-cube/weights.pt')),'image_steps':[0,5,10],'past_action_steps':[0,10],'future_action_steps':[10,15],'latent_shape':list(embedding.shape),'action_gradient_max':float(grad.abs().max()),'all_parameters_frozen':all(not p.requires_grad for p in model.parameters())},'physics_dt':.002,'control_dt':.05,'scope':'Six development episodes full200action replay; exact full-MjData/IK/cache restoration and dense-instrumentation comparison at selected approach/contact/height states. No Cube smoothing, tolerance calibration, independenttest or broad modelfidelity claim yet. Reset seed omission handled by explicit RNG; initialsource qpos/qvel/control/time andtarget used.','seconds':time.perf_counter()-started,'source_sha256':sha256(Path(__file__)),'execution_module_sha256':sha256(resolve('zj_wam/cube_execution.py'))})
    print('COMPLETE Cube dataset/interface audit',len(rows),len(branches),flush=True)
if __name__=='__main__':main()
