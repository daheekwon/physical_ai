"""Freeze audited smoke; wait for existing nominal audit, then full108 bank/audit."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_policy_bank';read=lambda p:json.loads(p.read_text())

def main(parent_pid):
    smoke=BASE/'dinowm_policy_bank_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['states']==3 and a['candidates']==165 and a['fresh_native_DINO_predictions']==165 and a['fresh_physical_branches']==6;assert a['source_sha256']==sha256(resolve('scripts/audit_followup_dinowm_policy_bank.py'))
    for p,h in {**read(smoke/'manifest.json')['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    files=['configs/followup_dinowm_policy_bank.json','scripts/run_followup_dinowm_policy_bank.py','scripts/audit_followup_dinowm_policy_bank.py','scripts/run_followup_dinowm_policy_bank_pipeline.py'];OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in files},'smoke_audit_sha256':sha256(smoke/'raw_audit.json'),'nominal_dependency_pid':parent_pid,'scope':'Same3fixedwindows on all36freshpolicydevelopment/calibration families; collect/audit108states. No calibration/test/repeated conclusions yet.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    gate=BASE/'dinowm_policy_nominal/pipeline_complete.json'
    while not gate.exists():
        try:os.kill(parent_pid,0)
        except ProcessLookupError:raise RuntimeError('Nominal parent ended before audited completion; inspect original process, no restart')
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':'waiting_nominal_audit','status':'running','verified_dependency_pid':parent_pid,'updated_unix':time.time()});time.sleep(10)
    assert read(gate)['raw_audit_sha256']==sha256(BASE/'dinowm_policy_nominal/raw_audit.json')
    for stage in ['run','audit']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve('scripts/'+stage+'_followup_dinowm_policy_bank.py'))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'failed','returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    a=read(OUT/'raw_audit.json');assert a['states']==108;write_json(OUT/'pipeline_complete.json',{'states':a['states'],'candidates':a['candidates'],'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time(),'scope':'Bank collection/audit complete; selection calibration and independent test pending.'});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--nominal-pid',type=int,required=True);args=ap.parse_args();main(args.nominal_pid)
