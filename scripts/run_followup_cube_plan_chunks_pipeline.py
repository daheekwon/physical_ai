"""Freeze validated causal compiler and audit all12 development scenes."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_plan_chunks';read=lambda p:json.loads(p.read_text())
def main():
    smoke=read(BASE/'cube_plan_chunks_smoke/raw_audit.json');assert smoke['smoke'] and smoke['episodes']==3 and smoke['native_exact_episodes']==1 and smoke['source_sha256']==sha256(resolve('scripts/analyze_followup_cube_plan_chunks.py'))
    paths=[resolve(p) for p in ['configs/followup_cube_plan_chunks.json','scripts/cube_plan_chunk_core.py','scripts/run_followup_cube_plan_chunks.py','scripts/analyze_followup_cube_plan_chunks.py','scripts/run_followup_cube_plan_chunks_pipeline.py']]+[BASE/'cube_plan_chunks_smoke/raw_audit.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'episodes':36,'families':12,'scope':'Nominal causal H5 compiler compatibility only; no smoothing or finaltest.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script in ['run_followup_cube_plan_chunks.py','analyze_followup_cube_plan_chunks.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    a=read(OUT/'raw_audit.json');assert a['episodes']==36 and a['native_exact_episodes']==12
    write_json(OUT/'pipeline_complete.json',{'episodes':36,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
if __name__=='__main__':main()
