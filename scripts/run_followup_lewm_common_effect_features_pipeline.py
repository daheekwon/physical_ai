"""Freeze aligned common-bank LeWM features and validate all210 states."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'lewm_common_effect_features';read=lambda p:json.loads(p.read_text())
def main():
    a=read(BASE/'lewm_common_effect_features_smoke/raw_audit.json');assert a['smoke'] and a['states']==2 and a['candidates_including_nominal']==146 and a['all_actual_endpoints_freshly_encoded'];assert a['source_sha256']==sha256(resolve('scripts/audit_followup_lewm_common_effect_features.py'))
    files=[resolve(p) for p in ['configs/followup_lewm_common_effect_features.json','scripts/collect_followup_lewm_common_effect_features.py','scripts/audit_followup_lewm_common_effect_features.py','scripts/run_followup_lewm_common_effect_features_pipeline.py']]+[BASE/'lewm_common_effect_features_smoke/raw_audit.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'states':210,'scope':'CPU LeWM features on sameF1H5 commonbank asDINO; no newphysicalexecutions, no finaltest. Validate CPU/GPUbatch numerical errors explicitly.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script in ['collect_followup_lewm_common_effect_features.py','audit_followup_lewm_common_effect_features.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    a=read(OUT/'raw_audit.json');assert a['states']==210
    write_json(OUT/'pipeline_complete.json',{'states':210,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
if __name__=='__main__':main()
