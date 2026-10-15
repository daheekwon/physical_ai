"""Official Cube asset/interface preflight; no expert fidelity or smoothing claim."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np,torch
from torchvision.transforms import v2 as T
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from stable_worldmodel.wm.utils import load_pretrained
from zj_wam.io import resolve,write_json,sha256

def main():
 torch.set_num_threads(4);env=CubeEnv(env_type='single',ob_type='states',width=224,height=224,terminate_at_goal=False,visualize_info=False);obs,info=env.reset(seed=20261011);frames=[env.render().copy()]
 zero=np.zeros(5)
 for t in range(10):
  env.step(zero)
  if t in [4,9]:frames.append(env.render().copy())
 data_snapshot=copy.deepcopy(env._data)
 snapshot_type=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(env._model,snapshot_type));mujoco.mj_getState(env._model,env._data,state,snapshot_type)
 attributes={k:copy.deepcopy(getattr(env,k)) for k in ['_reset_next_step','_success','_step_count'] if hasattr(env,k)}
 rng=np.random.default_rng(20261011);actions=rng.uniform(-.1,.1,(5,5));actions[:,4]=0
 def rollout():
  result=[]
  for a in actions:env.step(a);result.append(np.r_[env._data.qpos,env._data.qvel,env._data.site_xpos[env._pinch_site_id]])
  return np.array(result)
 first=rollout();mujoco.mj_setState(env._model,env._data,state,snapshot_type);mujoco.mj_forward(env._model,env._data)
 for k,v in attributes.items():setattr(env,k,copy.deepcopy(v))
 second=rollout();repeat_error=float(np.abs(first-second).max())
 # Preserve MuJoCo's derived caches too, without an extra forward call.
 env._data=copy.deepcopy(data_snapshot)
 for k,v in attributes.items():setattr(env,k,copy.deepcopy(v))
 third=rollout();full_copy_error=float(np.abs(first-third).max())
 assert full_copy_error<1e-10
 model=load_pretrained(str(resolve('data/checkpoints/lewm-cube'))).eval().requires_grad_(False)
 transform=T.Compose([T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225]),T.Resize(224)])
 pixels=torch.stack([transform(im) for im in frames])[None]
 with torch.no_grad():emb=model.encode({'pixels':pixels})['emb']
 # Normalized zero is solely an interface probe; dataset normalization not yet validated.
 normalized=torch.zeros((1,3,25),requires_grad=True);action_emb=model.action_encoder(normalized);prediction=model.predict(emb,action_emb);gradient=torch.autograd.grad(prediction[:,-1].square().sum(),normalized)[0]
 assert torch.isfinite(prediction).all() and torch.isfinite(gradient).all() and gradient.abs().max()>0
 config=json.loads(resolve('data/checkpoints/lewm-cube/config.json').read_text());out=resolve('artifacts/followup_v1/protocol/cube_interface_audit.json')
 write_json(out,{'checkpoint_repo':'https://huggingface.co/quentinll/lewm-cube','revision':'b0747c5002e86d2ce8f3cd8178004b97524c587d','checkpoint_sha256':sha256(resolve('data/checkpoints/lewm-cube/weights.pt')),'config_sha256':sha256(resolve('data/checkpoints/lewm-cube/config.json')),'parameters':sum(p.numel() for p in model.parameters()),'all_weights_frozen':all(not p.requires_grad for p in model.parameters()),'action_shape':list(env.action_space.shape),'native_physics_dt':env._model.opt.timestep,'physics_steps_per_action':env._n_steps,'control_dt':env._model.opt.timestep*env._n_steps,'action_semantics':'normalized relative EEFxyz,yaw,gripper opening increment; gripper held fixed in initial smoothing','history_frames':len(frames),'image_shape':list(frames[0].shape),'latent_shape':list(emb.shape),'prediction_shape':list(prediction.shape),'action_gradient_max':float(gradient.abs().max()),'integration_restore_repeat_max_error':repeat_error,'full_mjdata_copy_repeat_max_error':full_copy_error,'restore_interpretation':'Integration-state plus mj_forward changes the next controller input in this low-motion case; full MjData cache copy is compared separately. Contact-rich expert restoration remains unverified.','restored_python_attributes':list(attributes),'scope':'Model strict load and nonzero input derivative, live Cube render/step, one low-motion integration-state replay. No dataset normalization, expert replay, contact/grasp restoration or smoothing validated yet.','source_sha256':sha256(Path(__file__))})
 env.close();print(out,{'integration_and_forward_error':repeat_error,'full_mjdata_copy_error':full_copy_error},flush=True)
if __name__=='__main__':main()
