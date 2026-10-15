"""Resume after independently reproduced floating point audit failure; preserve originals."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,write_json,sha256
BASE=resolve('artifacts/followup_v1');CAP=BASE/'dinowm_capture';OUT=BASE/'dinowm_readouts';read=lambda p:json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-pid',type=int,required=True);args=p.parse_args()
    paths=[resolve('scripts/audit_retrained_dinowm_capture_v2.py'),resolve('scripts/audit_retrained_dinowm_capture.py'),resolve('scripts/resume_retrained_dinowm_readouts.py'),resolve('scripts/audit_retrained_dinowm_readouts.py'),CAP/'pipeline_lock.json',CAP/'manifest.json',CAP/'execution_complete.json',CAP/'audit_retrained_dinowm_capture.log',OUT/'protocol.json']
    record={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'reason':'Original full audit failed FP32 pooling output-relative check on 7 records/8 rows. Re-encode every affected record; require exact saved mean/quadrant reproduction plus float64 identity and raw-token summation error bound. Do not edit data or old pipeline/auditor. Original dependency-gated fit and audit exited before training.'}
    lock=CAP/'audit_recovery_protocol.json'
    if lock.exists():assert read(lock)==record
    else:write_json(lock,record)
    start=time.monotonic()
    while not (CAP/'raw_audit.json').exists():
        os.kill(args.audit_pid,0)
        if time.monotonic()-start>1800:raise RuntimeError('Observation window elapsed, inspect live audit')
        time.sleep(3)
    audit=read(CAP/'raw_audit.json');assert audit['labels']==4296 and audit['source_records']==618 and audit['source_sha256']==sha256(resolve('scripts/audit_retrained_dinowm_capture_v2.py'))
    assert len(audit['roundoff_rechecks'])==7 and all(r['fresh_mean_and_quadrants_exact'] for r in audit['roundoff_rechecks'])
    for name,h in record['inputs'].items():assert sha256(resolve(name))==h,name
    write_json(CAP/'pipeline_complete.json',{'source_records':618,'labels':4296,'raw_audit_sha256':sha256(CAP/'raw_audit.json'),'lock_sha256':sha256(CAP/'pipeline_lock.json'),'recovery_protocol_sha256':sha256(lock),'unix':time.time(),'scope':'Collection original protocol; independent v2 audit replaces failed pooling tolerance audit. Original pipeline did not complete unchanged.'})
    print('CAPTURE RECOVERY VERIFIED',flush=True)
    for script in ['fit_retrained_dinowm_readouts.py','audit_retrained_dinowm_readouts.py']:
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    write_json(OUT/'pipeline_complete.json',{'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'fit_complete_sha256':sha256(OUT/'fit_complete.json'),'recovery_protocol_sha256':sha256(lock),'unix':time.time()})
if __name__=='__main__':main()
