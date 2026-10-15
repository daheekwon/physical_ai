"""Frozen DP closed-loop corrections in SWM, unchanged policy8/control/physics timing."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dill,hydra,numpy as np,pandas as pd,torch
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.policy_refinement import make_scene,full_state,observe,coverage,absolute_to_relative,context_at,refine,evaluate_local
from zj_wam.diagnostics import jerk_energy
from zj_wam.io import resolve,write_json,sha256


def run_episode(seed,method,pcfg,cfg,rcfg,policy,model,scale,rules,strong,out,max_actions):
 done=out/f'{seed}_{method}.json'
 if done.exists():return
 start=time.perf_counter();env=make_scene(seed);initial=full_state(env);im,obs=observe(env);images=[im];obs_history=[obs];states=[initial];physics=[];delivered=[];cov=[];logs=[];locals=[];localnom=[];localcov=[];nominals=[];predictions=[];policy_times=[];render_times=[];first_success=None;policy.reset()
 original=env.space.step
 def step(dt):original(dt);physics.append(full_state(env))
 env.space.step=step
 while len(delivered)<max_actions:
  t=len(delivered);idx=[max(0,t-1),t]
  batch={k:torch.from_numpy(np.stack([obs_history[i][k] for i in idx])[None]).to(cfg['device'],dtype=torch.float32) for k in obs}
  noise_seed=(cfg['seed']+1009*seed+t)%(2**31-1);torch.manual_seed(noise_seed);torch.cuda.synchronize(cfg['device']);tic=time.perf_counter()
  with torch.no_grad():prediction=policy.predict_action(batch)
  torch.cuda.synchronize(cfg['device']);policy_times.append(time.perf_counter()-tic);targets=prediction['action'][0].cpu().numpy()[:min(8,max_actions-t)];predictions.append(prediction['action_pred'][0].cpu().numpy())
  nominal,adapter=absolute_to_relative(targets,states[-1][:2],states[-1][5:7]);chosen=nominal.copy();info={'active':False,'norm':0.,'seconds':0.,'predicted_gain':0.};eligible=t>=10 and len(nominal)==8;history_ids=[]
  if eligible:
   wmimages,past,proprio,prefix,history_ids=context_at(t,images,states,delivered,physics)
   if method!='nominal':chosen,info=refine(method,model,wmimages,past,proprio,prefix,nominal,scale,np.random.SeedSequence([cfg['seed'],seed,t]),cfg,rcfg,rules,strong)
  prior=np.array(delivered).reshape(-1,2).copy();begin=len(physics);covstart=len(cov)
  for a in chosen:
   env.step(a);delivered.append(a.copy());states.append(full_state(env));cov.append(coverage(env));tic=time.perf_counter();im,obs=observe(env);render_times.append(time.perf_counter()-tic);images.append(im);obs_history.append(obs)
   if cov[-1]>.95 and first_success is None:first_success=len(delivered)
  if eligible:
   actual={'physics':np.array(physics[begin:]),'coverage':cov[covstart:]};metric,trace,coverage_trace=evaluate_local(seed,prior,nominal,actual,prefix,cfg['tolerances'])
   locals.append({'seed':seed,'method':method,'control_step':t,'active':info['active'],**metric});localnom.append(trace);localcov.append(coverage_trace)
  nominals.append(nominal);logs.append({'control_step':t,'policy_noise_seed':noise_seed,'policy_observation_times':idx,'wm_observation_times':history_ids,'eligible':eligible,**adapter,**info})
 env.space.step=original;env.close();physics=np.asarray(physics)
 # Padded nominal chunks only in audit artifact; valid length is in decision logs and delivered count.
 padded=np.full((len(nominals),8,2),np.nan)
 for i,n in enumerate(nominals):padded[i,:len(n)]=n
 np.savez_compressed(done.with_suffix('.npz'),initial_state=initial,states=states,physics=physics,actions=delivered,coverage=cov,nominal_relative_chunks=padded,policy_action_predictions=predictions,local_nominal_physics=localnom,local_nominal_coverage=localcov,images=np.asarray(images,dtype=np.uint8))
 pd.DataFrame(locals).to_csv(done.with_suffix('.csv'),index=False)
 write_json(done,{'seed':seed,'method':method,'actions':len(delivered),'active_chunks':sum(x['active'] for x in logs),'eligible_chunks':sum(x['eligible'] for x in logs),'first_success_action':first_success,'any_success':first_success is not None,'final_coverage':cov[-1],'max_coverage':max(cov),'max_score':min(1.,max(cov)/.95),'physics_jerk':jerk_energy(np.vstack([initial[:2],physics[:,:2]]),.01),'policy_calls':len(logs),'policy_inference_seconds':policy_times,'render_seconds':render_times,'refinement_seconds':sum(x['seconds'] for x in logs),'seconds':time.perf_counter()-start,'decisions':logs,'source_sha256':sha256(Path(__file__)),'module_sha256':sha256(resolve('zj_wam/policy_refinement.py'))})
 print(seed,method,'success',first_success,'active',sum(x['active'] for x in logs),'seconds',round(time.perf_counter()-start,2),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration','test'],default='development');p.add_argument('--limit',type=int);p.add_argument('--methods',nargs='+');p.add_argument('--smoke',action='store_true');args=p.parse_args()
 cfg=json.loads(resolve('configs/followup_v1.json').read_text());pcfg=json.loads(resolve('configs/followup_policy_refinement.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());base=resolve(cfg['output'])
 if args.split=='test' and not (base/'protocol/final_selection_lock.json').exists():raise ValueError('Finaltest locked')
 out=base/('policy_refinement_smoke' if args.smoke else 'policy_refinement')/args.split;out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(cfg['threads']);payload=torch.load(resolve('data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt'),map_location='cpu',pickle_module=dill,weights_only=False);policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload;policy.to(cfg['device']).eval().requires_grad_(False)
 assert policy.num_inference_steps==100 and policy.n_action_steps==8 and policy.n_obs_steps==2
 data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);rules=json.loads((base/'selection/selector_rules.json').read_text())['rules'];strong=json.loads((base/'strong_baselines/calibrated_settings.json').read_text())['rules']['gaussian_action']
 manifest={'config':pcfg,'main_config_sha256':sha256(resolve('configs/followup_v1.json')),'source_sha256':sha256(Path(__file__)),'module_sha256':sha256(resolve('zj_wam/policy_refinement.py')),'checkpoint_sha256':json.loads((base/'dp_native/manifest.json').read_text())['checkpoint_sha256'],'smoke':args.smoke,'max_actions':40 if args.smoke else pcfg['max_actions']}
 target=out/'manifest.json'
 if target.exists():
  old=json.loads(target.read_text())
  if old!=manifest:raise ValueError('Manifest changed: use a new experiment version or audit before resuming')
 else:write_json(target,manifest)
 seeds=pcfg[args.split+'_seeds'][:args.limit] if args.limit else pcfg[args.split+'_seeds'];methods=args.methods or pcfg['methods']
 assert set(methods)<=set(pcfg['methods'])
 for seed in seeds:
  for method in methods:run_episode(seed,method,pcfg,cfg,rcfg,policy,model,data.scaler.scale_,rules,strong,out,manifest['max_actions'])
if __name__=='__main__':main()
