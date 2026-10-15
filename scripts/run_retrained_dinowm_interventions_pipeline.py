"""Freeze four-state validation, collect full common banks, then audit them."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_interventions';read=lambda p:json.loads(p.read_text())


def main():
    smoke=read(BASE/'dinowm_interventions_smoke/raw_audit.json');assert smoke['smoke'] and smoke['states']==4 and smoke['candidates_including_nominal']==238
    for env in ['pusht','ogb-cube']:assert smoke['max_error'][env+'_fresh_native_prediction']==0 and smoke['max_error'][env+'_single_vs_batched_encoder']==0
    assert smoke['source_sha256']==sha256(resolve('scripts/audit_retrained_dinowm_interventions.py'))
    paths=[resolve(p) for p in ['configs/followup_dinowm_interventions.json','scripts/collect_retrained_dinowm_interventions.py','scripts/audit_retrained_dinowm_interventions.py','scripts/run_retrained_dinowm_interventions_pipeline.py']]+[BASE/'dinowm_interventions_smoke/raw_audit.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'states':402,'scope':'Complete development/calibration banks; same physical traces. No finaltest or selection calibration in this pipeline.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script in ['collect_retrained_dinowm_interventions.py','audit_retrained_dinowm_interventions.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    a=read(OUT/'raw_audit.json');assert a['states']==402
    write_json(OUT/'pipeline_complete.json',{'states':402,'candidates':a['candidates_including_nominal'],'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
if __name__=='__main__':main()
