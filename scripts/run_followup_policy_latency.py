"""Physical-clock latency replay with own observations and common target hold."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from zj_wam.policy_refinement import make_scene,full_state,observe,absolute_to_relative,context_at,refine,evaluate_local,coverage
from scripts.policy_latency_core import draw_delay,holding_action

BASE=resolve('artifacts/followup_v1')


def read(p):return json.loads(p.read_text())


def prepare(smoke):
    cfg=read(resolve('configs/followup_policy_latency.json'));out=BASE/('policy_latency_smoke' if smoke else 'policy_latency');out.mkdir(exist_ok=True);source=BASE/'policy_refinement'/('development' if smoke else 'calibration');audit=read(source/'raw_audit.json');assert audit['state_method_rows']==72 and audit['local_blocks']==2520
    samples={'policy':[],'acquisition':[],'refinement':{m:[] for m in cfg['methods']}}
    paths=sorted(source.glob('21*.json'));assert len(paths)==72
    for path in paths:
        meta=read(path);method=meta['method']
        if method=='nominal':samples['policy']+=meta['policy_inference_seconds'];samples['acquisition']+=meta['render_seconds']
        samples['refinement'][method]+=[d['seconds'] for d in meta['decisions'] if d['eligible']]
    assert all(len(v)>0 for v in samples['refinement'].values());files=[resolve(p) for p in ['configs/followup_policy_latency.json','scripts/policy_latency_core.py','scripts/run_followup_policy_latency.py','scripts/analyze_followup_policy_latency.py','scripts/analyze_followup_policy_final.py','configs/followup_v1.json','configs/followup_response_models.json','zj_wam/policy_refinement.py','zj_wam/repeated_refinement.py','zj_wam/followup.py','zj_wam/selection.py']]+[BASE/'selection/selector_rules.json',BASE/'strong_baselines/calibrated_settings.json',source/'raw_audit.json']+paths
    manifest={'config':cfg,'smoke':smoke,'timing_split':source.name,'samples':samples,'input_sha256':{str(p.relative_to(resolve('.'))):sha256(p) for p in files}}
    p=out/'manifest.json'
    if p.exists():assert read(p)==manifest,'Protocol changed: preserve old results and version explicitly'
    else:write_json(p,manifest)
    return cfg,out,samples


def run_episode(seed,method,profile,cfg,main,rcfg,policy,model,scale,rules,strong,samples,out,steps):
    path=out/f'{seed}_{method}_{profile}.json'
    if path.exists():return
    import torch
    start=time.perf_counter();env=make_scene(seed);initial=full_state(env);states=[initial];images=[];observations=[];actions=[];physics=[];cov=[];roles=[];call_ids=[];offsets=[];targets=[];logs=[];nominals=[];planned=[];policy_predictions=[];chunks=[]
    im,obs=observe(env);images.append(im);observations.append(obs);target=initial[:2].copy();policy.reset();original=env.space.step
    def physics_step(dt):original(dt);physics.append(full_state(env))
    env.space.step=physics_step
    def advance(action,role,call,offset):
        nonlocal target
        previous=states[-1][:2].copy()
        if role=='chunk':target=previous+100*np.asarray(action)
        env.step(action);actions.append(np.asarray(action).copy());roles.append(int(role=='chunk'));call_ids.append(call);offsets.append(offset);targets.append(target.copy());states.append(full_state(env));cov.append(coverage(env));im,ob=observe(env);images.append(im);observations.append(ob)
    while len(actions)<steps:
        t=len(actions);k=len(logs);ids=[max(0,t-1),t];batch={key:torch.from_numpy(np.stack([observations[i][key] for i in ids])[None]).to(main['device'],dtype=torch.float32) for key in observations[-1]};noise=(main['seed']+1009*seed+8*k)%(2**31-1);torch.manual_seed(noise);tic=time.perf_counter()
        with torch.no_grad():result=policy.predict_action(batch)
        pred=result['action_pred'][0].cpu().numpy();absolute=result['action'][0].cpu().numpy()[:min(8,steps-t)];nominal,adapter=absolute_to_relative(absolute,states[-1][:2],states[-1][5:7]);chosen=nominal.copy();eligible=t>=10 and len(nominal)==8;info={'active':False,'seconds':0.,'norm':0.,'predicted_gain':0.};history=[]
        if eligible:
            ims,past,proprio,prefix,history=context_at(t,images,states,actions,physics)
            if method!='nominal':chosen,info=refine(method,model,ims,past,proprio,prefix,nominal,scale,np.random.SeedSequence([main['seed'],seed,t]),main,rcfg,rules,strong)
        wall=time.perf_counter()-tic;delay=draw_delay(samples,seed,k,method,cfg['methods'],profile,eligible,main['seed']);ready=t+delay['ticks'];prior_target=target.copy()
        while len(actions)<min(ready,steps):advance(holding_action(target,states[-1][:2]),'hold',k,-1)
        dispatch=len(actions);count=min(len(chosen),steps-dispatch);prefix=np.asarray(physics[-30:]).copy() if len(physics)>=30 else np.empty((0,10))
        for index,action in enumerate(chosen[:count]):advance(action,'chunk',k,index)
        if eligible and count==8:chunks.append({'call_index':k,'request_step':t,'dispatch_step':dispatch,'active':info['active'],'nominal':nominal.copy(),'prefix':prefix})
        row={'call_index':k,'request_step':t,'ready_step':ready,'dispatch_step':dispatch if count else None,'delivered_count':count,'observation_times':ids,'policy_noise_seed':noise,'eligible':eligible,'wm_observation_times':history,'delay':delay,'wall_plan_seconds':wall,'held_target_at_request':prior_target.tolist(),**adapter,**info};logs.append(row);nominals.append(nominal);planned.append(chosen);policy_predictions.append(pred)
    env.space.step=original;env.close();physics=np.asarray(physics);actions=np.asarray(actions);cov=np.asarray(cov);local_rows=[];local_physics=[];local_coverage=[]
    for chunk in chunks:
        d=chunk['dispatch_step'];actual={'physics':physics[d*10:(d+8)*10],'coverage':cov[d:d+8]};metric,trace,ct=evaluate_local(seed,actions[:d],chunk['nominal'],actual,chunk['prefix'],main['tolerances']);local_rows.append({key:chunk[key] for key in ['call_index','request_step','dispatch_step','active']}|metric);local_physics.append(trace);local_coverage.append(ct)
    pad=lambda xs:np.array([np.pad(x,((0,8-len(x)),(0,0)),constant_values=np.nan) for x in xs])
    np.savez_compressed(path.with_suffix('.npz'),initial_state=initial,states=states,physics=physics,actions=actions,coverage=cov,roles=roles,call_ids=call_ids,offsets=offsets,hold_targets=targets,nominal_chunks=pad(nominals),planned_chunks=pad(planned),policy_predictions=policy_predictions,local_nominal_physics=np.asarray(local_physics),local_nominal_coverage=np.asarray(local_coverage))
    pd.DataFrame(local_rows,columns=['call_index','request_step','dispatch_step','active','object_error','jerk_gain','position_max','rotation_max','coverage_loss']).to_csv(path.with_suffix('.csv'),index=False)
    write_json(path,{'seed':seed,'method':method,'profile':profile,'steps':steps,'calls':logs,'eligible_complete_chunks':len(chunks),'hold_steps':int((np.asarray(roles)==0).sum()),'seconds':time.perf_counter()-start,'manifest_sha256':sha256(out/'manifest.json')});print(seed,method,profile,'holds',int((np.asarray(roles)==0).sum()),'calls',len(logs),'seconds',round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--split',choices=['development','test'],default='development');p.add_argument('--methods',nargs='+');p.add_argument('--profiles',nargs='+');a=p.parse_args();cfg,out,samples=prepare(a.smoke)
    if a.split=='test':raise ValueError('Latencytest requires separate freeze after complete development validation')
    import torch,dill,hydra
    from zj_wam.adapters import ExpertDataset,FrozenLeWM
    main=read(resolve('configs/followup_v1.json'));rcfg=read(resolve('configs/followup_response_models.json'));torch.set_num_threads(main['threads']);payload=torch.load(resolve('data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt'),map_location='cpu',pickle_module=dill,weights_only=False);policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload;policy.to(main['device']).eval().requires_grad_(False)
    assert policy.num_inference_steps==100 and policy.n_action_steps==8 and policy.n_obs_steps==2
    data=ExpertDataset(main['dataset']);model=FrozenLeWM(main['checkpoint'],data.scaler,main['device']);rules=read(BASE/'selection/selector_rules.json')['rules'];strong=read(BASE/'strong_baselines/calibrated_settings.json')['rules']['gaussian_action'];methods=a.methods or cfg['methods'];profiles=a.profiles or cfg['profiles'];assert set(methods)<=set(cfg['methods']) and set(profiles)<=set(cfg['profiles'])
    for seed in cfg['development_seeds'][:1] if a.smoke else cfg['development_seeds']:
        for method in methods:
            for profile in profiles:run_episode(seed,method,profile,cfg,main,rcfg,policy,model,data.scaler.scale_,rules,strong,samples,out,120 if a.smoke else cfg['max_actions'])
if __name__=='__main__':main()
