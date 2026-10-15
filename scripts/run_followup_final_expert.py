"""Preflight, freeze and execute the declared expert test without tuning."""
import argparse,json,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,write_json,sha256

BASE=resolve('artifacts/followup_v1')
CONFIG='configs/followup_final_expert.json'
LOCK=BASE/'protocol/final_selection_lock.json'
SCRIPTS=['run_followup.py','run_followup_interventions.py','run_followup_probes.py','run_followup_effects.py','run_followup_response_scores.py','run_followup_family_ablation.py','run_followup_strong_baselines.py','run_followup_runtime.py','run_followup_strong_runtime.py','run_followup_shift_evaluation.py','run_followup_geometric_shifts.py','run_followup_repeated_refinement.py','run_followup_final_expert.py','analyze_followup_final_expert.py','report_followup_final_expert.py','calibrate_followup_selection.py','calibrate_followup_strong_baselines.py','analyze_followup_response_selection.py']
CONFIGS=['followup_v1','followup_response_models','followup_effect_audit','followup_selection','followup_strong_baselines','followup_runtime','followup_strong_runtime','followup_shift_evaluation','followup_repeated_refinement','followup_final_expert']
ASSETS=['protocol/manifest.json','protocol/exposure.json','selection/bank_rules.json','selection/selector_rules.json','strong_baselines/calibrated_settings.json','runtime/calibrated_settings.json','strong_runtime/calibrated_budgets.json','lewm_audit/readouts/selection.json','lewm_audit/effects/resolution.json','response_scores/latent_scale.npy']


def load(p):return json.loads(p.read_text())

def fingerprints():
    paths=[resolve('scripts/'+p) for p in SCRIPTS]+[resolve('configs/'+p+'.json') for p in CONFIGS]+list(resolve('zj_wam').glob('*.py'))+[BASE/p for p in ASSETS]+list((BASE/'lewm_audit/readouts').glob('*.joblib'))
    return {str(p.relative_to(resolve('.'))):sha256(p) for p in sorted(paths)}


def verify_lock():
    assert LOCK.exists(),'Final lock required';lock=load(LOCK)
    for p,h in lock['fingerprints'].items():assert sha256(resolve(p))==h,('Frozen input changed',p)
    return lock


def preflight():
    cfg=load(resolve(CONFIG));manifest=load(BASE/'protocol/manifest.json');states=[s for s in manifest['states'] if s['split']=='test'];families=manifest['families'];exposure=load(BASE/'protocol/exposure.json');missing=[]
    assert len(states)==cfg['states'] and len(families['test'])==cfg['families'];assert set(families['test']).isdisjoint(exposure['exposed_families'])
    for split in ['probe_train','development','calibration']:assert set(families['test']).isdisjoint(families[split])
    required=['runtime/raw_audit.json','strong_runtime/raw_audit.json','strong_runtime/analysis.json','strong_runtime/calibrated_budgets.json','selection/evaluation_summary.json','strong_baselines/raw_audit.json','geometric_shifts/raw_audit.json']
    for name in required+ASSETS:
        if not (BASE/name).exists():missing.append(name)
    if (BASE/'strong_runtime/raw_audit.json').exists():
        a=load(BASE/'strong_runtime/raw_audit.json')
        if a.get('partial') or a.get('states')!={'calibration':105,'development':105} or a.get('rows')!=2310 or a.get('timing_repetitions')!=6930:missing.append('strong_runtime full210state audited completion')
    if (BASE/'runtime/raw_audit.json').exists():
        a=load(BASE/'runtime/raw_audit.json');assert all(a[s]['states']==105 for s in ['calibration','development'])
    sources=[resolve('scripts/'+p) for p in SCRIPTS]
    missing += [str(p) for p in sources if not p.exists()]
    if not LOCK.exists():
        for folder in ['f1','interventions','family_ablation','strong_baselines','lewm_audit/effects','runtime/test','strong_runtime/test']:
            assert not list((BASE/folder).glob('test*.npz')),('Test already exposed before lock',folder)
    missing=sorted(set(missing))
    return {'ready':not missing,'missing':missing,'states':states,'families':families['test'],'scope':cfg['scope']}


def run_stage(stage):
    lock=verify_lock();cfg=load(resolve('configs/followup_v1.json'));fcfg=load(resolve(CONFIG));out=BASE/'final_expert';out.mkdir(exist_ok=True);log=out/(stage+'.log');done=out/(stage+'.done.json')
    if done.exists():assert load(done)['lock_sha256']==sha256(LOCK);return
    start=time.time();commands={
      'f1_h5':['run_followup.py','f1','--split','test'],
      'interventions':['run_followup_interventions.py','--split','test'],
      'probe_capture':['run_followup_probes.py','collect','--split','test'],
      'effects':['run_followup_effects.py','--split','test'],
      'scores_interventions':['run_followup_response_scores.py','--split','test','--bank','interventions'],
      'scores_f1':['run_followup_response_scores.py','--split','test','--bank','f1'],
      'scores_family':['run_followup_response_scores.py','--split','test','--bank','family_ablation'],
      'shifts':['run_followup_shift_evaluation.py','--split','test'],
      'analyze':['analyze_followup_final_expert.py','--split','test'],
      'report':['report_followup_final_expert.py']}
    if stage in commands:
        command=[sys.executable,str(resolve('scripts/'+commands[stage][0])),*commands[stage][1:]]
        with log.open('a') as handle:subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True,cwd=resolve('.'))
    elif stage=='horizon_f1':
        with log.open('a') as handle:
            for h in fcfg['diagnostic_horizons']:subprocess.run([sys.executable,str(resolve('scripts/run_followup.py')),'f1','--split','test','--horizon',str(h)],stdout=handle,stderr=subprocess.STDOUT,check=True,cwd=resolve('.'))
    else:
        # Existing process functions consume supplied state specs; their CLI105-state limits remain unchanged.
        import torch,contextlib
        from zj_wam.adapters import ExpertDataset,FrozenLeWM
        torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=None if stage=='shifts_geometric' else FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);paths=sorted((BASE/'f1').glob('test*h05.json'));assert len(paths)==45
        rules=load(BASE/'selection/selector_rules.json')['rules'];strong=load(BASE/'strong_baselines/calibrated_settings.json')['rules'];rcfg=load(resolve('configs/followup_response_models.json'))
        with log.open('a') as handle,contextlib.redirect_stdout(handle),contextlib.redirect_stderr(handle):
            if stage=='family':
                from scripts.run_followup_family_ablation import process
                for path in paths:process(path,cfg,data,model,BASE)
            elif stage=='strong':
                from scripts.run_followup_strong_baselines import process
                bcfg=load(resolve('configs/followup_strong_baselines.json'))
                for path in paths:process(path,cfg,bcfg,data,model,BASE,False)
            elif stage=='runtime_common':
                from scripts.run_followup_runtime import run,signature
                tcfg=load(resolve('configs/followup_runtime.json'));dest=BASE/'runtime/test';dest.mkdir(exist_ok=True);sig=signature();assert sig==load(BASE/'runtime/manifest.json')['signature']
                for i,path in enumerate(paths):run(path,cfg,rcfg,tcfg,data,model,rules,strong['gaussian_action'],dest,sig,i==0)
            elif stage=='runtime_strong':
                from scripts.run_followup_strong_runtime import process,signature
                tcfg=load(resolve('configs/followup_strong_runtime.json'));dest=BASE/'strong_runtime/test';dest.mkdir(exist_ok=True);sig=signature();assert sig==load(BASE/'strong_runtime/manifest.json')['signature']
                for i,path in enumerate(paths):process(path,cfg,tcfg,data,model,rules,strong,dest,sig,i==0)
            elif stage=='shifts_geometric':
                from scripts.run_followup_geometric_shifts import process,signature
                scfg=load(resolve('configs/followup_shift_evaluation.json'));sig=signature();assert sig==load(BASE/'geometric_shifts/manifest.json')['signature']
                for path in paths:process(path,cfg,scfg,data,strong,BASE/'geometric_shifts',sig)
            elif stage=='repeated':
                from scripts.run_followup_repeated_refinement import process
                scfg=load(resolve('configs/followup_repeated_refinement.json'))
                for path in paths:process(path,cfg,rcfg,scfg,data,model,rules,BASE)
            else:raise ValueError(stage)
    verify_lock();write_json(done,{'stage':stage,'lock_sha256':sha256(LOCK),'started_unix':start,'finished_unix':time.time(),'exit':'completed; raw audit is a separate required step'});print('COMPLETE',stage,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','freeze','run','continue-when-ready']);p.add_argument('--stage');p.add_argument('--dependency-pid',type=int);args=p.parse_args()
    if args.mode=='preflight':print(json.dumps(preflight(),indent=2));return
    if args.mode=='continue-when-ready':
        import os
        assert args.dependency_pid is not None
        started=time.monotonic();previous=None
        while True:
            result=preflight()
            if result['ready']:break
            if result['missing']!=previous:print('WAITING',result['missing'],flush=True);previous=result['missing']
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError(('Dependency terminal before required audit',result['missing']))
            if time.monotonic()-started>7200:raise RuntimeError('Two-hour observation window elapsed; revalidate dependency, do not restart it automatically')
            time.sleep(10)
        subprocess.run([sys.executable,__file__,'freeze'],check=True)
        subprocess.run([sys.executable,__file__,'run'],check=True)
        return
    if args.mode=='freeze':
        if LOCK.exists():verify_lock();print('Already locked');return
        result=preflight();assert result['ready'],result['missing'];cfg=load(resolve(CONFIG))
        record={'version':1,'scope':'PushT expert primary/declared secondary only; policy and Cube final protocols separate','config':cfg,'test_states':result['states'],'test_families':result['families'],'fingerprints':fingerprints(),'created_unix':time.time(),'policy_test_authorized':False,'cube_test_authorized':False}
        with LOCK.open('x') as handle:json.dump(record,handle,indent=2);handle.write('\n')
        print('Frozen',LOCK);return
    verify_lock();stages=load(resolve(CONFIG))['stages']
    if args.stage:assert args.stage in stages;stages=[args.stage]
    for stage in stages:run_stage(stage)
if __name__=='__main__':main()
