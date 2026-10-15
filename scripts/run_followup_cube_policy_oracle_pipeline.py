"""Freeze audited Cube oracle smoke and run full collection/audit/report."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_policy_oracle';read=lambda p:json.loads(p.read_text())

def main():
    smoke=BASE/'cube_policy_oracle_smoke';a=read(smoke/'raw_audit.json');assert a['states']==4 and a['candidates']==84 and a['fresh_physical_branches_first_family']==8;assert a['analysis_sha256']==sha256(smoke/'analysis.json');assert a['source_sha256']==sha256(resolve('scripts/audit_followup_cube_policy_oracle.py'))
    for p,h in read(smoke/'manifest.json')['inputs'].items():assert sha256(resolve(p))==h,p
    paths=['configs/followup_cube_policy_oracle.json','scripts/run_followup_cube_policy_oracle.py','scripts/audit_followup_cube_policy_oracle.py','scripts/report_followup_cube_policy_oracle.py','scripts/run_followup_cube_policy_oracle_pipeline.py','scripts/cube_policy_refinement_core.py','scripts/run_followup_cube_native_policy.py','scripts/run_followup_cube.py','scripts/audit_followup_cube_policy_refinement.py','zj_wam/cube_execution.py','zj_wam/cube_experiment.py'];OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in paths},'smoke_audit_sha256':sha256(smoke/'raw_audit.json'),'scope':'Same preset12 nominal families and4phase bank; no expanded candidatefamily or posthoc threshold tuning.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage in ['run','audit','report']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        script='scripts/'+stage+'_followup_cube_policy_oracle.py';write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'failed','returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    a=read(OUT/'raw_audit.json');write_json(OUT/'pipeline_complete.json',{'states':a['states'],'candidates':a['candidates'],'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_sha256':sha256(OUT/'report_evidence.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})

if __name__=='__main__':main()
