"""Check DP native rollout transfer to SWM PushT: state, renderer and relative actions."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from stable_worldmodel.envs.pusht.env import PushT
from zj_wam.io import resolve,write_json,sha256

def state(env):return np.r_[env.agent.position,env.block.position,env.block.angle,env.agent.velocity,env.block.velocity,env.block.angular_velocity]
def assign(env,x):
 env.agent.position=x[:2].tolist();env.agent.velocity=x[5:7].tolist();env.block.angle=float(x[4]);env.block.position=x[2:4].tolist();env.block.velocity=x[7:9].tolist();env.block.angular_velocity=float(x[9]);env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block)
def main():
 directory=resolve('artifacts/followup_v1/dp_native');out=resolve('artifacts/followup_v1/dp_transfer');out.mkdir(exist_ok=True);rows=[]
 for path in sorted(directory.glob('seed_*.npz')):
  a=np.load(path);env=PushT(resolution=96,relative=True,render_action=False);env.reset(seed=20261011);env._setup();env._set_goal_state(np.array([256,400,256,256,np.pi/4,0,0]));env.goal_pose=np.array([256,256,np.pi/4]);assign(env,a['initial_state'])
  renderer=PushT(resolution=96,relative=True,render_action=False);renderer.reset(seed=20261011);renderer._setup();renderer.goal_pose=np.array([256,256,np.pi/4])
  original=env.space.step;physics=[]
  def step(dt):original(dt);physics.append(state(env))
  env.space.step=step;raw=[];render_errors=[];states=[]
  for index,target in enumerate(a['actions']):
   action=(target-np.asarray(env.agent.position))/100;raw.append(action);env.step(action);states.append(state(env));actual=state(env)
   # Renderer comparison uses exactly the stored native pose, independent of dynamics drift.
   assign(renderer,a['states'][index]);img=renderer.render();reference=np.moveaxis(a['images'][index],0,-1);render_errors.append(float(np.abs(img.astype(float)-reference.astype(float)).mean()))
  env.space.step=original;env.close();renderer.close();raw=np.asarray(raw);trace=np.asarray(physics);reference=a['physics'];delta=trace-reference;rot=np.arctan2(np.sin(delta[:,4]),np.cos(delta[:,4]))
  row={'seed':int(path.stem.split('_')[1]),'actions':len(raw),'relative_action_abs_max':float(np.abs(raw).max()),'actions_exceeding_bound':int((np.abs(raw)>1+1e-6).any(1).sum()),'pusher_max_difference_px':float(np.linalg.norm(delta[:,:2],axis=1).max()),'object_max_difference_px':float(np.linalg.norm(delta[:,2:4],axis=1).max()),'object_max_rotation_difference_deg':float(np.rad2deg(np.abs(rot).max())),'same_pose_image_mae_255':float(np.mean(render_errors)),'same_pose_image_max_mae_255':max(render_errors)};rows.append(row)
  np.savez_compressed(out/path.name,relative_actions=raw,physics=trace,states=states,same_pose_image_mae=render_errors)
  print(row,flush=True)
 write_json(out/'audit.json',{'scope':'Same post-reset native initial full state, canonical goal, same absolute target sequence mapped online to relative commands. Native Pymunk6 versus SWM Pymunk7; render at96 without marker. No action clipping in this diagnostic. Same-pose renderer comparison separates image difference from dynamics drift.','rows':rows,'source_sha256':sha256(Path(__file__))})
if __name__=='__main__':main()
