"""Freeze completed-smoke evidence, then collect and audit all probe splits."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,write_json,sha256
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_capture';read=lambda p:json.loads(p.read_text())


def main():
    audit=read(BASE/'dinowm_capture_smoke/raw_audit.json');assert audit['smoke'] and audit['source_records']==6 and audit['labels']==66
    assert audit['source_sha256']==sha256(resolve('scripts/audit_retrained_dinowm_capture.py'))
    for env in ['pusht','ogb-cube']:assert audit['max_error'][env+'_fresh_current_forward']==0 and audit['max_error'][env+'_fresh_history_forward']==0
    paths=[resolve(p) for p in ['configs/followup_dinowm_capture.json','scripts/collect_retrained_dinowm_capture.py','scripts/audit_retrained_dinowm_capture.py','scripts/run_retrained_dinowm_capture_pipeline.py','scripts/retrained_dinowm_core.py']]+[BASE/'dinowm_capture_smoke/raw_audit.json']
    record={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'planned_source_records':618,'planned_labels':4296,'scope':'Samepriorprobe families andphysical labels, no supervised fitting orfinaltest inthis pipeline.'}
    OUT.mkdir(exist_ok=True);lock=OUT/'pipeline_lock.json'
    if lock.exists():assert read(lock)==record
    else:
        with lock.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    for script in ['collect_retrained_dinowm_capture.py','audit_retrained_dinowm_capture.py']:
        for p,h in record['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    audit=read(OUT/'raw_audit.json');assert audit['source_records']==618 and audit['labels']==4296
    write_json(OUT/'pipeline_complete.json',{'source_records':618,'labels':4296,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(lock),'unix':time.time()})
if __name__=='__main__':main()
