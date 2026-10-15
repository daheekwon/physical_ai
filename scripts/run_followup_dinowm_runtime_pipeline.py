"""Frozen sequential smoke/audit/full/audit/report; fail without restarting."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
from scripts.run_followup_dinowm_runtime import SOURCES
BASE=resolve('artifacts/followup_v1');OUT=BASE/'protocol/dinowm_runtime_pipeline';read=lambda p:json.loads(p.read_text())

def main():
    OUT.mkdir(exist_ok=True);paths=SOURCES+['scripts/audit_followup_dinowm_runtime.py','scripts/lewm_runtime_reference.py','scripts/report_followup_dinowm_runtime.py','scripts/run_followup_dinowm_runtime_pipeline.py','scripts/output_replacement_core.py','scripts/audit_retrained_dinowm_readouts.py','scripts/analyze_followup_response_selection.py'];record={'inputs':{p:sha256(resolve(p)) for p in paths},'scope':'Four-state randomized warmed smoke, audit, full402 fresh runtime, audit and report. Abort on any error; never mark completion on timeout.'}
    if (OUT/'lock.json').exists():assert read(OUT/'lock.json')==record
    else:write_json(OUT/'lock.json',record)
    jobs=[('collect_smoke','scripts/run_followup_dinowm_runtime.py',['--smoke']),('audit_smoke','scripts/audit_followup_dinowm_runtime.py',['--smoke']),('collect_full','scripts/run_followup_dinowm_runtime.py',[]),('audit_full','scripts/audit_followup_dinowm_runtime.py',[]),('report','scripts/report_followup_dinowm_runtime.py',[])]
    for name,script,options in jobs:
        for p,h in record['inputs'].items():assert sha256(resolve(p))==h,p
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':name,'status':'running','updated_unix':time.time()})
        with (OUT/(name+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script)),*options],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            write_json(OUT/'status.json',{'pid':os.getpid(),'stage':name,'status':'failed','returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((name,r.returncode))
        print('COMPLETE',name,flush=True)
    out=BASE/'dinowm_runtime';write_json(out/'pipeline_complete.json',{'states':402,'raw_audit_sha256':sha256(out/'raw_audit.json'),'report_sha256':sha256(out/'report_evidence.json'),'lock_sha256':sha256(OUT/'lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})

if __name__=='__main__':main()
