"""Freeze validated plan4.7C and run all105 development states."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'gradient_controls';read=lambda p:json.loads(p.read_text())
def main():
    smoke=read(BASE/'gradient_controls_smoke/raw_audit.json');assert smoke['smoke'] and smoke['states']==1 and smoke['candidates']==72 and smoke['source_sha256']==sha256(resolve('scripts/analyze_followup_gradient_controls.py'))
    paths=[resolve(p) for p in ['configs/followup_gradient_controls.json','scripts/gradient_controls_core.py','scripts/run_followup_gradient_controls.py','scripts/analyze_followup_gradient_controls.py','scripts/run_followup_gradient_controls_pipeline.py']]+[BASE/'gradient_controls_smoke/raw_audit.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'states':105,'families':35,'planned_candidates':7560,'scope':'AdditionalC development diagnosis only. No newtest or recalibration.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script in ['run_followup_gradient_controls.py','analyze_followup_gradient_controls.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    a=read(OUT/'raw_audit.json');assert a['states']==105 and a['candidates']==7560
    write_json(OUT/'pipeline_complete.json',{'states':105,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
if __name__=='__main__':main()
