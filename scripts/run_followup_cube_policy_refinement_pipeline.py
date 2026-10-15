"""Freeze validated Cube repeated refinement and audit all48 development episodes."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_policy_refinement';read=lambda p:json.loads(p.read_text())
def main():
    smoke=BASE/'cube_policy_refinement_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['episodes']==4 and a['local_blocks']==228 and a['optimizer_attempts']==855 and a['exact_nominal_episodes']==1 and a['source_sha256']==sha256(resolve('scripts/audit_followup_cube_policy_refinement.py'))
    paths=[resolve(p) for p in ['configs/followup_cube_policy_refinement.json','scripts/cube_policy_refinement_core.py','scripts/run_followup_cube_policy_refinement.py','scripts/audit_followup_cube_policy_refinement.py','scripts/run_followup_cube_policy_refinement_pipeline.py','scripts/analyze_followup_cube_plan_chunks.py','scripts/analyze_followup_response_selection.py','scripts/report_followup_cube_policy_refinement.py']]+[smoke/'raw_audit.json',smoke/'manifest.json',smoke/'execution_complete.json',BASE/'cube_plan_chunks/raw_audit.json']
    for p in smoke.glob('340*.json'):
        r=read(p);assert r['initial_smoke_physics_exact'] and r['data_sha256']==sha256(p.with_suffix('.npz'));paths += [p,p.with_suffix('.npz')]
    archive=BASE/'cube_policy_refinement_initial_smoke_v1';ar=read(archive/'archive_reason.json')
    for p,r in ar['sources'].items():assert sha256(resolve(r['copy']))==r['sha256'];paths.append(resolve(r['copy']))
    for p in archive.glob('340*.json'):assert read(p)['data_sha256']==sha256(p.with_suffix('.npz'));paths += [p,p.with_suffix('.npz')]
    lock={'states':12,'episodes':48,'local_blocks':2736,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'scope':'Whole15s repeatedrefinement, ownpastPDH5baseline, nonWMgenerator, threeverifiers, transferredexpertthresholds. Developmentonly; policycalibration/final/adaptation/latency separate.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script in ['run_followup_cube_policy_refinement.py','audit_followup_cube_policy_refinement.py']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    a=read(OUT/'raw_audit.json');assert a['episodes']==48 and a['local_blocks']==2736 and a['families']==12
    write_json(OUT/'pipeline_complete.json',{'episodes':48,'families':12,'local_blocks':2736,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
    with (OUT/'report_followup_cube_policy_refinement.log').open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/report_followup_cube_policy_refinement.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE cube policy refinement report',flush=True)
if __name__=='__main__':main()
