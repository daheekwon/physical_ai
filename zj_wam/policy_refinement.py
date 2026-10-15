"""PushT policy interface and causal live refinement; no future object inputs."""
import time
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon
from shapely.ops import unary_union
from stable_worldmodel.envs.pusht.env import PushT
from zj_wam.followup import controller_rollout
from zj_wam.diagnostics import jerk_energy
from zj_wam.repeated_refinement import choose_correction,generate_bank
from zj_wam.selection import select_indices


def full_state(env):
 return np.r_[env.agent.position,env.block.position,env.block.angle,env.agent.velocity,env.block.velocity,env.block.angular_velocity]


def make_scene(seed):
 """Official native legacy random initial state, SWM physics and canonical goal."""
 env=PushT(resolution=224,relative=True,render_action=False)
 env.reset(seed=int(seed));env._setup();rng=np.random.RandomState(seed)
 x=np.array([rng.randint(50,450),rng.randint(50,450),rng.randint(100,400),rng.randint(100,400),rng.randn()*2*np.pi-np.pi])
 env.agent.position=x[:2].tolist();env.agent.velocity=(0.,0.)
 # Native legacy order: setting angle after position rotates around CoG.
 env.block.angle=0.
 env.block.position=x[2:4].tolist();env.block.angle=float(x[4]);env.block.velocity=(0.,0.);env.block.angular_velocity=0.
 env.space.step(env.dt)
 env.goal_pose=np.array([256.,256.,np.pi/4]);env._set_goal_state(np.array([256.,400.,256.,256.,np.pi/4,0.,0.]))
 return env


def coverage(env):
 goal=env._get_goal_pose_body(env.goal_pose)
 def poly(body):return unary_union([Polygon([body.local_to_world(v) for v in s.get_vertices()]) for s in env.block.shapes])
 target=poly(goal);return float(poly(env.block).intersection(target).area/target.area)


def observe(env):
 wm=env.render().copy();old=env.render_size
 try:
  env.render_size=96;policy=np.moveaxis(env.render().astype(np.float32)/255.,-1,0)
 finally:env.render_size=old
 return wm,{'image':policy,'agent_pos':np.asarray(env.agent.position,dtype=np.float32).copy()}


def absolute_to_relative(targets,position,velocity,bound=1.):
 """Compile a policy chunk once using known PD; all methods share this interface."""
 q=np.asarray(position,dtype=float).copy();v=np.asarray(velocity,dtype=float).copy();raw=[];clipped=[];realized=[]
 for target in np.asarray(targets):
  unconstrained=(target-q)/100.;action=np.clip(unconstrained,-bound,bound);raw.append(action);clipped.append(bool(np.any(np.abs(unconstrained)>bound)))
  applied=q+100*action;realized.append(applied.copy())
  for _ in range(10):v+=(100*(applied-q)-20*v)*.01;q+=v*.01
 return np.asarray(raw),{'clipped_actions':sum(clipped),'max_target_projection_px':float(np.linalg.norm(np.asarray(realized)-targets,axis=1).max())}


def context_at(t,images,states,delivered,physics):
 if t<10 or len(images)!=t+1 or len(states)!=t+1 or len(delivered)!=t or len(physics)!=t*10:raise ValueError('Causal history alignment failed')
 ids=[t-10,t-5,t];proprio=np.zeros((3,10));proprio[:,[0,1,5,6]]=np.asarray(states)[ids][:,[0,1,5,6]]
 return np.asarray(images)[ids],np.asarray(delivered)[t-10:t],proprio,np.asarray(physics)[-30:].copy(),ids


def refine(method,model,images,past,proprio,prefix,nominal,scale,seed,cfg,rcfg,rules,strong_rule):
 start=time.perf_counter();block=nominal[:5];suffix=nominal[5:8];root=np.asarray(proprio)[-1]
 if method in ['vision_contact','vision_adaptive','lewm_global']:
  chosen,info,_=choose_correction(model,images,past,proprio,prefix,block,suffix,scale,seed,cfg,rcfg,rules,method)
 elif method=='controller_pusher':
  candidates,labels=generate_bank(block,suffix,root,prefix,past,scale,seed,cfg['radii']);pred=np.array([controller_rollout(np.concatenate([a,suffix]),root[:2],root[5:7]) for a in candidates])
  costs=np.array([jerk_energy(np.concatenate([prefix[:,:2],q]),.01) for q in pred]);norms=np.linalg.norm(((candidates-candidates[:1])/scale).reshape(len(candidates),-1),axis=1)
  score=np.linalg.norm(pred-pred[:1],axis=-1).max(1);f=pd.DataFrame({'state_id':'live','horizon':5,'actual_norm':norms,'predicted_jerk':costs,'nominal_predicted_jerk':costs[0],'controller_pusher_score':score})
  index=max(0,select_indices(f,method,rules[method])['live']);chosen=candidates[index];info={'selected_index':index,'selected_generator':labels[index],'active':bool(norms[index]>1e-6),'norm':float(norms[index]),'score':float(score[index]),'candidate_count':len(candidates)-1,'predicted_gain':float((costs[0]-costs[index])/max(costs[0],1e-8))}
 elif method=='gaussian_action':
  assert strong_rule['enabled'];filt=gaussian_filter1d(np.concatenate([past[-3:],block,suffix]),strong_rule['setting'],axis=0,mode='nearest')[3:8];delta=(filt-block)/scale;factor=min(1.,strong_rule['radius']*np.sqrt(10)/max(np.linalg.norm(delta),1e-12));chosen=block+factor*(filt-block)
  costs=[jerk_energy(np.concatenate([prefix[:,:2],controller_rollout(np.concatenate([x,suffix]),root[:2],root[5:7])]),.01) for x in [block,chosen]]
  gain=(costs[0]-costs[1])/max(costs[0],1e-8)
  if gain<=1e-8:chosen=block.copy()
  info={'active':bool(np.linalg.norm((chosen-block)/scale)>1e-6),'norm':float(np.linalg.norm((chosen-block)/scale)),'candidate_count':1,'predicted_gain':float(max(0,gain))}
 else:raise ValueError(method)
 result=nominal.copy();result[:5]=chosen
 assert np.max(np.abs(result))<=1+1e-6
 info['seconds']=time.perf_counter()-start
 return result,info


def evaluate_local(seed,delivered,nominal,actual,prefix,tolerances):
 """Evaluation only, called after selected actions execute; clone by prefix replay."""
 env=make_scene(seed)
 for a in delivered:env.step(a)
 trace=[];cov=[];original=env.space.step
 def step(dt):original(dt);trace.append(full_state(env))
 env.space.step=step
 for a in nominal:env.step(a);cov.append(coverage(env))
 env.space.step=original;env.close();trace=np.asarray(trace)
 position=np.linalg.norm(actual['physics'][:,2:4]-trace[:,2:4],axis=1).max();theta=actual['physics'][:,4]-trace[:,4];rotation=np.abs(np.arctan2(np.sin(theta),np.cos(theta))).max()
 losses=np.maximum(0,np.asarray(cov)[[4,7]]-np.asarray(actual['coverage'])[[4,7]])
 error=max(position/tolerances['position'],rotation/tolerances['rotation'],losses.max()/tolerances['coverage_loss'])
 j0=jerk_energy(np.concatenate([prefix[:,:2],trace[:,:2]]),.01);j1=jerk_energy(np.concatenate([prefix[:,:2],actual['physics'][:,:2]]),.01)
 return {'object_error':float(error),'jerk_gain':float((j0-j1)/j0 if j0>1e-8 else 0.),'position_max':float(position),'rotation_max':float(rotation),'coverage_loss':float(losses.max())},trace,np.asarray(cov)
