"""Separate immutable protocol for independent physical-clock latency test."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json

BASE=resolve('artifacts/followup_v1');ROOT=BASE/'policy_latency';LOCK=ROOT/'final_protocol.json'
read=lambda p:json.loads(p.read_text())


def development_ready():
    if not (ROOT/'development_complete.json').exists():return False
    complete=read(ROOT/'development_complete.json');audit=read(ROOT/'raw_audit.json')
    assert complete['episodes']==72 and not audit['smoke'] and audit['episodes']==72
    assert complete['audit_sha256']==sha256(ROOT/'raw_audit.json') and complete['normalized_analysis_sha256']==sha256(ROOT/'normalized_analysis.json') and complete['lock_sha256']==sha256(ROOT/'pipeline_lock.json')
    return True


def verify():
    lock=read(LOCK)
    for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,('Latency final input changed',p)
    return lock


def freeze():
    if LOCK.exists():return verify()
    assert development_ready()
    validation=read(ROOT/'final_validation/raw_audit.json');assert validation['validation_only'] and not validation['smoke'] and validation['episodes']==72
    assert validation['source_sha256']==sha256(resolve('scripts/analyze_followup_policy_latency_final.py'))
    assert validation['manifest_sha256']==sha256(ROOT/'manifest.json')
    for k,v in validation['max_error'].items():
        if k.startswith('parent_reproduction_'):assert v<=1e-9
    parent=read(ROOT/'manifest.json');cfg=parent['config'];final=read(resolve('configs/followup_policy_latency_final.json'));assert final['test_seeds']==cfg['test_seeds'] and len(final['test_seeds'])==final['families']==12
    assert not list((ROOT/'test').glob('213*.json')) and not list((ROOT/'test').glob('213*.npz')),'Latency final outcomes exposed before separate lock'
    from zj_wam.policy_refinement import make_scene,full_state
    original=read(resolve('configs/followup_policy_refinement.json'));oldseeds=original['development_seeds']+original['calibration_seeds']+original['test_seeds'];assert not set(oldseeds)&set(final['test_seeds'])
    previous=[]
    for seed in oldseeds:
        env=make_scene(seed);previous.append(full_state(env));env.close()
    initial={}
    for seed in final['test_seeds']:
        env=make_scene(seed);state=full_state(env);env.close()
        assert not any(np.allclose(state,x,atol=1e-8,rtol=0) for x in previous)
        previous.append(state);initial[str(seed)]=state.tolist()
    paths=[resolve(p) for p in ['configs/followup_policy_latency_final.json','configs/followup_policy_refinement.json','scripts/run_followup_policy_latency_final.py','scripts/analyze_followup_policy_latency_final.py']]
    paths += [ROOT/p for p in ['manifest.json','pipeline_lock.json','development_complete.json','raw_audit.json','normalized_analysis.json','normalized_evaluation.csv','final_validation/raw_audit.json','final_validation/analysis.json']]
    inputs=dict(read(ROOT/'pipeline_lock.json')['input_sha256']);inputs.update({str(p.relative_to(resolve('.'))):sha256(p) for p in paths})
    for p,h in inputs.items():assert sha256(resolve(p))==h,p
    record={'final_config':final,'config':cfg,'samples':parent['samples'],'inputs':inputs,'test_initial_states':initial,'planned_episodes':216,'created_unix':time.time(),'scope':'Only test initial states inspected before lock, no test predictions/actions/outcomes. Original calibrated selector and complete calibration timing distribution unchanged. Full latency development reproduced by final analyzer before test.'}
    with LOCK.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print('LOCKED latency test 12families216episodes',flush=True);return record


def run():
    lock=verify();cfg=lock['config'];out=ROOT/'test';out.mkdir(exist_ok=True)
    manifest={'config':cfg,'smoke':False,'timing_split':'calibration','samples':lock['samples'],'input_sha256':{**lock['inputs'],str(LOCK.relative_to(resolve('.'))):sha256(LOCK)}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    import torch,dill,hydra
    from zj_wam.adapters import ExpertDataset,FrozenLeWM
    from scripts.run_followup_policy_latency import run_episode
    main=read(resolve('configs/followup_v1.json'));response=read(resolve('configs/followup_response_models.json'));torch.set_num_threads(main['threads'])
    payload=torch.load(resolve('data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt'),map_location='cpu',pickle_module=dill,weights_only=False);policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload;policy.to(main['device']).eval().requires_grad_(False)
    assert policy.num_inference_steps==100 and policy.n_action_steps==8 and policy.n_obs_steps==2
    data=ExpertDataset(main['dataset']);model=FrozenLeWM(main['checkpoint'],data.scaler,main['device']);rules=read(BASE/'selection/selector_rules.json')['rules'];strong=read(BASE/'strong_baselines/calibrated_settings.json')['rules']['gaussian_action']
    for seed in cfg['test_seeds']:
        for method in cfg['methods']:
            for profile in cfg['profiles']:
                run_episode(seed,method,profile,cfg,main,response,policy,model,data.scaler.scale_,rules,strong,lock['samples'],out,cfg['max_actions'])
    verify();assert len(list(out.glob('213*.json')))==216
    write_json(ROOT/'final_execution_complete.json',{'episodes':216,'lock_sha256':sha256(LOCK),'unix':time.time()})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','continue-when-ready']);p.add_argument('--dependency-pid',type=int);args=p.parse_args()
    if args.mode=='freeze':freeze();return
    if args.mode=='continue-when-ready':
        assert args.dependency_pid;start=time.monotonic();smoke=read(BASE/'policy_latency_smoke/final_validation/raw_audit.json');assert smoke['validation_only'] and smoke['episodes']==6 and smoke['source_sha256']==sha256(resolve('scripts/analyze_followup_policy_latency_final.py'))
        print('WAITING for full latency development72 audit; no test lock or outcomes yet',flush=True)
        while not development_ready():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Latency development parent ended before completed audit; inspect logs')
            if time.monotonic()-start>21600:raise RuntimeError('Observation window elapsed; revalidate parent')
            time.sleep(10)
        with (ROOT/'final_validation.log').open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/analyze_followup_policy_latency_final.py')),'--validate-source','development'],stdout=log,stderr=subprocess.STDOUT,check=True)
        freeze()
    run()
    with (ROOT/'final_analysis.log').open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/analyze_followup_policy_latency_final.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE latency final execution and raw audit; report required',flush=True)

if __name__=='__main__':main()
