"""Freeze validated native-DINO selection, run full402, audit and report."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_selection';read=lambda p:json.loads(p.read_text())

def main():
    smoke=read(BASE/'dinowm_selection_smoke/raw_audit.json');assert smoke['smoke'] and smoke['states']==4 and smoke['selected_rows']==20 and smoke['candidate_rows']==148;assert smoke['source_sha256']==sha256(resolve('scripts/audit_followup_dinowm_selection.py'));collection=read(BASE/'dinowm_interventions/pipeline_complete.json');assert collection['states']==402 and collection['raw_audit_sha256']==sha256(BASE/'dinowm_interventions/raw_audit.json')
    paths=[resolve(p) for p in ['configs/followup_dinowm_selection.json','scripts/run_followup_dinowm_selection.py','scripts/audit_followup_dinowm_selection.py','scripts/report_followup_dinowm_selection.py','scripts/run_followup_dinowm_selection_pipeline.py']]+[BASE/'dinowm_selection_smoke/raw_audit.json',BASE/'dinowm_interventions/pipeline_complete.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'states':402,'scope':'Common nonWMcandidate verifier development/calibration only. No cached runtime or finaltest claim.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script in ['run_followup_dinowm_selection.py','audit_followup_dinowm_selection.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    a=read(OUT/'raw_audit.json');assert a['states']==402 and a['selected_rows']==2010;write_json(OUT/'pipeline_complete.json',{'states':402,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
    subprocess.run([sys.executable,str(resolve('scripts/report_followup_dinowm_selection.py'))],check=True)

if __name__=='__main__':main()
