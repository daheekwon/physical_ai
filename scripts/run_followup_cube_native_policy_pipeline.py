"""Run all prespecified nominal Cube scenes after exact smoke reproduction."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_native_policy';read=lambda p:json.loads(p.read_text())


def main():
    smoke=read(BASE/'cube_native_policy_smoke/raw_audit.json');assert smoke['smoke'] and smoke['episodes']==2 and smoke['plain_reproduced_episodes']==2 and smoke['source_sha256']==sha256(resolve('scripts/analyze_followup_cube_native_policy.py'))
    paths=[resolve(p) for p in ['configs/followup_cube_native_policy.json','scripts/run_followup_cube_native_policy.py','scripts/analyze_followup_cube_native_policy.py','scripts/run_followup_cube_native_policy_pipeline.py']]+[BASE/'cube_native_policy_smoke/raw_audit.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'episodes':24,'families':12,'scope':'Native privileged fixed-target policy baseline only; chunk generation and smoothing adaptation remain separate.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script in ['run_followup_cube_native_policy.py','analyze_followup_cube_native_policy.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    write_json(OUT/'pipeline_complete.json',{'episodes':24,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
if __name__=='__main__':main()
