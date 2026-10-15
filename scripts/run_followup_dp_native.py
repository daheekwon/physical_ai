"""Frozen official Diffusion Policy in its native PushT environment, compatibility baseline."""
import argparse, collections, hashlib, importlib.metadata, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dill, hydra, numpy as np, torch
from omegaconf import OmegaConf
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from zj_wam.io import resolve,write_json,sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=3);p.add_argument('--device',default='cuda:2');args=p.parse_args()
 torch.set_num_threads(4);out=resolve('artifacts/followup_v1/dp_native');out.mkdir(exist_ok=True)
 ckpt=resolve('data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt')
 payload=torch.load(ckpt,map_location='cpu',pickle_module=dill,weights_only=False);cfg=payload['cfg']
 policy=hydra.utils.instantiate(cfg.policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload
 policy=policy.to(args.device).eval().requires_grad_(False)
 assert policy.num_inference_steps==100 and policy.n_obs_steps==2 and policy.n_action_steps==8
 manifest=out/'manifest.json'
 if not manifest.exists():
  write_json(manifest,{'checkpoint_sha256':sha256(ckpt),'checkpoint_url':'https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0500-test_mean_score=0.884.ckpt',
    'official_git_commit':subprocess.check_output(['git','-C',str(resolve('external/diffusion-policy')),'rev-parse','HEAD'],text=True).strip(),
    'versions':{name:importlib.metadata.version(name) for name in ['torch','diffusers','robomimic','pymunk','gym']},'source_sha256':sha256(Path(__file__)),
    'policy_config':OmegaConf.to_container(cfg.policy,resolve=True),'weights':'ema_model, strict state_dict, eval and no gradients',
    'scope':'Native legacy PushT compatibility evaluation. New dependency versions; not exact original benchmark reproduction and not SWM adapter validation. No correction or training. Official benchmark seeds overlap original reported evaluation; not fresh held-out research test.',
    'execution':{'max_actions':300,'inference_steps':100,'action_chunk':8,'history':2,'base_environment_seed':100000,'torch_seed':'20261011 + environment_seed'},
    'reference_versions':{'torch':'1.12.1','diffusers':'0.11.1','pymunk':'6.2.1'}})
 for seed in range(100000,100000+args.episodes):
  done=out/f'seed_{seed}.json'
  if done.exists():continue
  env=PushTImageEnv(legacy=True);env.seed(seed);obs=env.reset();history=collections.deque([obs,obs],maxlen=2)
  torch.manual_seed(20261011+seed);policy.reset();actions=[];scores=[];states=[];physics=[];images=[];timings=[];chunks=[];start=time.perf_counter();success=False
  def state():return np.r_[env.agent.position,env.block.position,env.block.angle,env.agent.velocity,env.block.velocity,env.block.angular_velocity]
  initial=state();original=env.space.step
  def step(dt):original(dt);physics.append(state())
  env.space.step=step
  while len(actions)<300 and not success:
   batch={k:torch.from_numpy(np.stack([o[k] for o in history])[None]).to(args.device,dtype=torch.float32) for k in obs}
   torch.cuda.synchronize();tic=time.perf_counter()
   with torch.no_grad():prediction=policy.predict_action(batch)
   torch.cuda.synchronize();timings.append(time.perf_counter()-tic);chunk=prediction['action'][0].cpu().numpy();chunks.append(prediction['action_pred'][0].cpu().numpy())
   assert chunk.shape==(8,2) and np.isfinite(chunk).all()
   for a in chunk:
    obs,reward,success,info=env.step(a);history.append(obs);actions.append(a);scores.append(float(reward));states.append(state());images.append((obs['image']*255).round().astype(np.uint8))
    if success or len(actions)==300:break
  env.space.step=original;env.close()
  np.savez_compressed(done.with_suffix('.npz'),actions=actions,states=states,physics=physics,images=images,rewards=scores,action_predictions=chunks,initial_state=initial)
  write_json(done,{'seed':seed,'actions':len(actions),'success':bool(success),'max_score':max(scores),'last_score':scores[-1],'policy_calls':len(timings),'inference_seconds':timings,'seconds':time.perf_counter()-start,'source_sha256':sha256(Path(__file__))})
  print(seed,'score',round(max(scores),4),'success',success,'actions',len(actions),flush=True)
 results=[json.loads(p.read_text()) for p in sorted(out.glob('seed_*.json'))]
 write_json(out/'analysis.json',{'episodes':len(results),'success_rate':float(np.mean([x['success'] for x in results])),'mean_max_score':float(np.mean([x['max_score'] for x in results])),'scope':'Compatibility baseline only; small sample unless all 50 official seeds are evaluated. No smoothing applied.'})
if __name__=='__main__':main()
