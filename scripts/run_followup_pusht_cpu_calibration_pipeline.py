"""Freeze nativeCPU calibration smoke, refresh210oldstates and independently audit."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'pusht_cpu_calibration';read=lambda p:json.loads(p.read_text())

def main():
    smoke=BASE/'pusht_cpu_calibration_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['states']==2 and a['nativeCPU_fresh_predictions']==110 and a['selections']==10;assert a['source_sha256']==sha256(resolve('scripts/audit_followup_pusht_cpu_calibration.py'))
    for p,h in read(smoke/'protocol.json')['inputs'].items():assert sha256(resolve(p))==h,p
    files=['scripts/run_followup_pusht_cpu_calibration.py','scripts/audit_followup_pusht_cpu_calibration.py','scripts/run_followup_pusht_cpu_calibration_pipeline.py'];OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in files},'smoke_audit_sha256':sha256(smoke/'raw_audit.json'),'scope':'Refresh/audit existing210oldexpert states only. No21reserved newexpert outcomes. CalibratednativeCPUrules needed before freshfinal.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage in ['run','audit']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve('scripts/'+stage+'_followup_pusht_cpu_calibration.py'))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'status':'failed','stage':stage,'returncode':r.returncode,'unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    a=read(OUT/'raw_audit.json');assert a['states']==210 and a['selections']==1050;write_json(OUT/'pipeline_complete.json',{'states':210,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'rules_sha256':sha256(OUT/'rules.json'),'unix':time.time(),'scope':'NativeCPU oldexpert calibration only; originalexpert21familyfinal notrun.'});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
