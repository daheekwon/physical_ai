"""Separate final lock and completion gates for frozen-rule policy transfer."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,write_json,sha256

BASE=resolve('artifacts/followup_v1');OUT=BASE/'policy_refinement';LOCK=OUT/'final_protocol.json'


def read(p):return json.loads(p.read_text())


def complete_calibration():
    for split in ['development','calibration']:
        folder=OUT/split
        if not (folder/'raw_audit.json').exists() or not (folder/'analysis.json').exists():return False
        audit=read(folder/'raw_audit.json');analysis=read(folder/'analysis.json')
        if analysis['incomplete_seeds'] or len(audit['paired_seeds'])!=12 or audit['state_method_rows']!=72 or audit['local_blocks']!=2520:return False
        if len(list(folder.glob('21*.json')))!=72:return False
    return True


def inputs():
    files=[resolve(p) for p in ['configs/followup_policy_final.json','configs/followup_policy_refinement.json','configs/followup_v1.json','configs/followup_response_models.json','scripts/run_followup_policy_final.py','scripts/analyze_followup_policy_final.py','scripts/run_followup_policy_refinement.py','scripts/analyze_followup_policy_refinement.py','scripts/audit_followup_policy_divergence.py','scripts/analyze_followup_response_selection.py','external/diffusion-policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py','external/diffusion-policy/diffusion_policy/policy/base_image_policy.py']]
    files+=list(resolve('zj_wam').glob('*.py'))
    files+=[BASE/p for p in ['selection/selector_rules.json','strong_baselines/calibrated_settings.json','response_scores/latent_scale.npy','dp_native/manifest.json']]
    for split in ['development','calibration']:
        files += [OUT/split/name for name in ['manifest.json','analysis.json','raw_audit.json','evaluation.csv','local_evaluation.csv']]
    files += [OUT/'final_validation_development/raw_audit.json',OUT/'final_validation_development/analysis.json']
    return {str(p.relative_to(resolve('.'))):sha256(p) for p in files}


def verify():
    m=read(LOCK)
    for path,h in m['inputs'].items():assert sha256(resolve(path))==h,('Policyfinalinputchanged',path)
    for path,h in m['checkpoints'].items():assert sha256(resolve(path))==h
    return m


def freeze():
    if LOCK.exists():return verify()
    assert complete_calibration(),'Both old splits require full raw audit'
    validation=read(OUT/'final_validation_development/raw_audit.json');assert validation['episodes']==72 and validation['local_blocks']==2520 and validation['validation_only']
    from scripts.run_followup_final_expert import verify_lock
    verify_lock()
    assert not list((OUT/'test').glob('212*.npz')) and not list((OUT/'test').glob('212*.json')),'Policytest already exposed without policy lock'
    cfg=read(resolve('configs/followup_policy_refinement.json'));fcfg=read(resolve('configs/followup_policy_final.json'))
    seeds=cfg['test_seeds'];assert len(seeds)==12 and not set(seeds)&set(cfg['development_seeds']+cfg['calibration_seeds'])
    from zj_wam.policy_refinement import make_scene,full_state
    oldstates=[]
    for split in ['development','calibration']:
        for seed in cfg[split+'_seeds']:
            with np.load(OUT/split/f'{seed}_nominal.npz') as a:oldstates.append(a['initial_state'])
    newstates=[]
    for seed in seeds:
        env=make_scene(seed);state=full_state(env);env.close()
        assert not any(np.allclose(state,x,rtol=0,atol=1e-8) for x in oldstates+newstates)
        newstates.append(state)
    checkpoints=['data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt',str(Path(read(resolve('configs/followup_v1.json'))['checkpoint'])/'weights.pt')]
    record={'config':fcfg,'original_policy_config':cfg,'inputs':inputs(),'checkpoints':{p:sha256(resolve(p)) for p in checkpoints},'test_initial_states':dict(zip(map(str,seeds),[x.tolist() for x in newstates])),'states_inspected':'Initialstatesonly; no testpolicy predictions/interventions/outcomes beforelock','planned_episodes':72,'families':12,'created_unix':time.time()}
    with LOCK.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print('POLICY FINAL LOCKED 12families72episodes',flush=True);return record


def alive(pid,label):
    try:os.kill(pid,0)
    except ProcessLookupError:raise RuntimeError(label+' dependency ended before gate; inspect actual failure, no automatic restart')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','continue-when-ready']);p.add_argument('--dependency-pid',type=int);p.add_argument('--expert-pid',type=int);a=p.parse_args()
    if a.mode=='freeze':freeze();return
    if a.mode=='continue-when-ready':
        assert a.dependency_pid and a.expert_pid;start=time.monotonic();gate=BASE/'final_expert/runtime_strong.done.json';print('WAITING for fullpolicycal audit and PushTtiming',flush=True)
        while not complete_calibration() or not gate.exists():
            if not complete_calibration():alive(a.dependency_pid,'Policy calibration')
            if not gate.exists():alive(a.expert_pid,'PushT timing')
            if time.monotonic()-start>21600:raise RuntimeError('Sixhour observation window elapsed; revalidate parents')
            time.sleep(10)
        from scripts.run_followup_final_expert import verify_lock
        verify_lock();assert read(gate)['lock_sha256']==sha256(BASE/'protocol/final_selection_lock.json');freeze()
    verify()
    for script,args in [('run_followup_policy_refinement.py',['--split','test']),('analyze_followup_policy_refinement.py',['--split','test']),('analyze_followup_policy_final.py',[])]:
        verify();path=OUT/('final_'+script.removesuffix('.py')+'.log')
        with path.open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script)),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    verify();write_json(OUT/'final_execution_complete.json',{'lock_sha256':sha256(LOCK),'episodes':72,'unix':time.time()})
if __name__=='__main__':main()
