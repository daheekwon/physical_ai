"""Four-arm first-correction/feedback decomposition with unchanged frozen policy."""
import argparse,json,sys,time,os,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,write_json,sha256
from zj_wam.policy_refinement import make_scene,full_state,observe,coverage,absolute_to_relative

BASE=resolve('artifacts/followup_v1');OUT=BASE/'policy_effect_decomposition';SOURCE=BASE/'policy_refinement/development'


def read(p):return json.loads(p.read_text())


def config():return read(resolve('configs/followup_policy_effect_decomposition.json'))


def prepare():
    cfg=config();OUT.mkdir(exist_ok=True)
    audit=read(SOURCE/'raw_audit.json');assert len(audit['paired_seeds'])==12
    files=[resolve(p) for p in ['configs/followup_policy_effect_decomposition.json','configs/followup_policy_refinement.json','configs/followup_v1.json','scripts/run_followup_policy_effect_decomposition.py','scripts/analyze_followup_policy_effect_decomposition.py','zj_wam/policy_refinement.py','zj_wam/diagnostics.py','external/diffusion-policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py','external/diffusion-policy/diffusion_policy/policy/base_image_policy.py']]
    files += [SOURCE/'raw_audit.json',SOURCE/'manifest.json']
    for seed in cfg['development_seeds']:
        for method in ['nominal',*cfg['methods']]:
            files += [SOURCE/f'{seed}_{method}{ext}' for ext in ['.json','.npz']]
    inputs={str(p.relative_to(resolve('.'))):sha256(p) for p in files}
    manifest={'config':cfg,'input_sha256':inputs,'checkpoint_sha256':sha256(resolve('data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt'))}
    p=OUT/'manifest.json'
    if p.exists():assert read(p)==manifest,'Version changed; preserve evidence and create a new protocol'
    else:write_json(p,manifest)
    return cfg


def source_arrays(seed,method):
    with np.load(SOURCE/f'{seed}_{method}.npz') as a:return {k:a[k].copy() for k in ['initial_state','states','physics','actions','coverage']}


def execute(seed,method,arm,policy=None):
    path=OUT/f'{seed}_{method}_{arm}.json'
    if path.exists():return
    cfg=config();main=read(resolve('configs/followup_v1.json'));start=time.perf_counter();nominal=source_arrays(seed,'nominal');original=source_arrays(seed,method)
    decisions=read(SOURCE/f'{seed}_{method}.json')['decisions'];active=[d for d in decisions if d['active']];assert active
    first=active[0]['control_step'];end=first+8;n=cfg['max_actions'];assert n==len(nominal['actions'])==len(original['actions'])
    np.testing.assert_array_equal(original['actions'][:first],nominal['actions'][:first]);np.testing.assert_array_equal(original['physics'][:10*first],nominal['physics'][:10*first])
    intended=nominal['actions'].copy()
    if arm!='a_nominal':intended[first:end]=original['actions'][first:end]
    env=make_scene(seed);initial=full_state(env);np.testing.assert_array_equal(initial,nominal['initial_state']);trace=[];states=[initial];cov=[];actions=[];obs_history=[];predictions=[];calls=[]
    oldstep=env.space.step
    def step(dt):oldstep(dt);trace.append(full_state(env))
    env.space.step=step
    if arm=='c_first_replan':
        import torch
        assert policy is not None;policy.reset();_,obs=observe(env);obs_history.append(obs)
    while len(actions)<n:
        t=len(actions)
        if arm=='c_first_replan' and t>=end:
            assert t%8==0
            ids=[t-1,t];batch={k:torch.from_numpy(np.stack([obs_history[i][k] for i in ids])[None]).to(main['device'],dtype=torch.float32) for k in obs_history[-1]}
            noise=(main['seed']+1009*seed+t)%(2**31-1);torch.manual_seed(noise)
            with torch.no_grad():result=policy.predict_action(batch)
            prediction=result['action_pred'][0].cpu().numpy();targets=result['action'][0].cpu().numpy()[:min(8,n-t)]
            chosen,adapter=absolute_to_relative(targets,states[-1][:2],states[-1][5:7]);predictions.append(prediction)
            # At first post-correction call, history exactly matches the old repeated run.
            if t==end:
                with np.load(SOURCE/f'{seed}_{method}.npz') as old:np.testing.assert_allclose(prediction,old['policy_action_predictions'][t//8],atol=1e-5,rtol=0)
            calls.append({'control_step':t,'policy_noise_seed':noise,'observation_times':ids,**adapter})
        else:chosen=intended[t:min(t+8,n)]
        for action in chosen:
            env.step(action);actions.append(action.copy());states.append(full_state(env));cov.append(coverage(env))
            if arm=='c_first_replan':_,obs=observe(env);obs_history.append(obs)
    env.space.step=oldstep;env.close();trace=np.asarray(trace);actions=np.asarray(actions);states=np.asarray(states);cov=np.asarray(cov)
    if arm=='a_nominal':
        np.testing.assert_array_equal(actions,nominal['actions']);np.testing.assert_array_equal(trace,nominal['physics']);np.testing.assert_array_equal(cov,nominal['coverage'])
    else:
        np.testing.assert_array_equal(actions[:end],original['actions'][:end]);np.testing.assert_array_equal(trace[:end*10],original['physics'][:end*10]);np.testing.assert_array_equal(cov[:end],original['coverage'][:end])
        if arm=='b_first_fixed':np.testing.assert_array_equal(actions[end:],nominal['actions'][end:])
    np.savez_compressed(path.with_suffix('.npz'),initial_state=initial,states=states,physics=trace,actions=actions,coverage=cov,policy_predictions=np.asarray(predictions))
    write_json(path,{'seed':seed,'method':method,'arm':arm,'first_correction_step':first,'first_chunk_end':end,'actions':n,'calls':calls,'seconds':time.perf_counter()-start,'manifest_sha256':sha256(OUT/'manifest.json'),'scope':cfg['scope']})
    print(seed,method,arm,'seconds',round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['open-loop','replan','continue-when-ready']);p.add_argument('--dependency-pid',type=int);a=p.parse_args();cfg=prepare()
    if a.mode=='open-loop':
        for seed in cfg['development_seeds']:
            execute(seed,cfg['methods'][0],'a_nominal')
            for method in cfg['methods']:execute(seed,method,'b_first_fixed')
        return
    if a.mode=='continue-when-ready':
        assert a.dependency_pid;start=time.monotonic();gate=BASE/'final_expert/runtime_strong.done.json';print('WAITING for PushT timing gate',flush=True)
        while not gate.exists():
            try:os.kill(a.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Parent ended before timing gate; inspect actual failure')
            if time.monotonic()-start>14400:raise RuntimeError('Observation window ended; revalidate parent before continuing')
            time.sleep(10)
        from scripts.run_followup_final_expert import verify_lock
        verify_lock();assert read(gate)['lock_sha256']==sha256(BASE/'protocol/final_selection_lock.json')
    assert len(list(OUT.glob('*_b_first_fixed.json')))==len(cfg['development_seeds'])*len(cfg['methods'])
    import torch,dill,hydra
    maincfg=read(resolve('configs/followup_v1.json'));torch.set_num_threads(maincfg['threads'])
    payload=torch.load(resolve('data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt'),map_location='cpu',pickle_module=dill,weights_only=False)
    policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload
    policy.to(maincfg['device']).eval().requires_grad_(False);assert policy.num_inference_steps==100 and policy.n_action_steps==8 and policy.n_obs_steps==2
    for seed in cfg['development_seeds']:
        for method in cfg['methods']:execute(seed,method,'c_first_replan',policy)
    prepare();subprocess.run([sys.executable,str(resolve('scripts/analyze_followup_policy_effect_decomposition.py'))],check=True)
if __name__=='__main__':main()
