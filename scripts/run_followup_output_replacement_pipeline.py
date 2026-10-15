"""Freeze replacement controls, stream each model family, independently audit."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['lewm','dino']);ap.add_argument('--dependency-pid',type=int);args=ap.parse_args();out=BASE/('output_replacement_'+args.family);out.mkdir(exist_ok=True)
    paths=[resolve(p) for p in ['configs/followup_output_replacement.json','scripts/output_replacement_core.py','scripts/analyze_followup_output_replacement.py','scripts/audit_followup_output_replacement.py','scripts/run_followup_output_replacement_pipeline.py','scripts/audit_retrained_dinowm_readouts.py','scripts/audit_retrained_dinowm_effect_readouts.py','scripts/analyze_followup_final_expert.py','scripts/audit_followup_streaming_and_strong.py']]
    for family in ['lewm','dino']:
        p=BASE/('output_replacement_'+family+'_smoke/raw_audit.json');a=read(p);assert a['smoke'] and a['states']==4 and a['selected_rows']==180 and a['decoded_arrays']==24 and a['source_sha256']==sha256(resolve('scripts/audit_followup_output_replacement.py'));paths.append(p)
    lock={'family':args.family,'states':402,'readout_configs':6,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'scope':'Plan4.6/F5 actual-output replacement, development/calibration only; nofinaltest, no newphysics. Future-information levels diagnostic.'}
    if (out/'pipeline_lock.json').exists():assert read(out/'pipeline_lock.json')==lock
    else:write_json(out/'pipeline_lock.json',lock)
    for script in ['analyze_followup_output_replacement.py','audit_followup_output_replacement.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        extra=['--family',args.family]
        if script.startswith('analyze') and args.dependency_pid:extra+=['--dependency-pid',str(args.dependency_pid)]
        with (out/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script)),*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',args.family,script,flush=True)
    a=read(out/'raw_audit.json');assert a['states']==402 and a['selected_rows']==17820
    write_json(out/'pipeline_complete.json',{'family':args.family,'states':402,'selected_rows':17820,'raw_audit_sha256':sha256(out/'raw_audit.json'),'lock_sha256':sha256(out/'pipeline_lock.json'),'unix':time.time()})
if __name__=='__main__':main()
