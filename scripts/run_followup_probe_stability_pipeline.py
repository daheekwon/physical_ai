"""Freeze successful two-environment capture validation and run all probe repetitions."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'probe_stability';read=lambda p:json.loads(p.read_text())
def main():
    a=read(BASE/'probe_stability_smoke/capture_raw_audit.json');assert a['smoke'] and a['models']==20 and a['evaluations']==40 and a['source_sha256']==sha256(resolve('scripts/audit_followup_probe_stability.py'))
    files=[resolve(p) for p in ['configs/followup_probe_stability.json','scripts/fit_followup_probe_stability.py','scripts/audit_followup_probe_stability.py','scripts/run_followup_probe_stability_pipeline.py']]+[BASE/'probe_stability_smoke/capture_raw_audit.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'models':180,'scope':'Capture repetitions only. AdditionalB completion also requires real/predicted-effect stability with these exact models.'};OUT.mkdir(exist_ok=True)
    if (OUT/'capture_pipeline_lock.json').exists():assert read(OUT/'capture_pipeline_lock.json')==lock
    else:write_json(OUT/'capture_pipeline_lock.json',lock)
    for script in ['fit_followup_probe_stability.py','audit_followup_probe_stability.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    audit=read(OUT/'capture_raw_audit.json');assert audit['models']==180 and audit['reused_models']==72
    write_json(OUT/'capture_pipeline_complete.json',{'models':180,'raw_audit_sha256':sha256(OUT/'capture_raw_audit.json'),'lock_sha256':sha256(OUT/'capture_pipeline_lock.json'),'unix':time.time(),'scope':'Capture complete, effect stability pending.'})
if __name__=='__main__':main()
