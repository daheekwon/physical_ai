"""Watch the existing audited pipelines and publish each report once ready."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'protocol/effect_reports';read=lambda p:json.loads(p.read_text())

def main():
    OUT.mkdir(exist_ok=True)
    jobs=[('dino_effects','dinowm_effect_readouts',3488890,['scripts/report_followup_effect_diagnostics.py','--kind','dino']),('probe_effects','probe_stability_effects',3528792,['scripts/report_followup_effect_diagnostics.py','--kind','stability']),('dino_replacement','output_replacement_dino',3542715,['scripts/report_followup_output_replacement.py','--family','dino'])]
    lock={'jobs':jobs,'inputs':{p:sha256(resolve(p)) for p in ['scripts/run_followup_effect_reports.py','scripts/report_followup_effect_diagnostics.py','scripts/report_followup_output_replacement.py']},'scope':'No collection restart, no partial publication. Existing processes must complete independent audits first.'}
    if (OUT/'lock.json').exists():assert read(OUT/'lock.json')==json.loads(json.dumps(lock))
    else:write_json(OUT/'lock.json',lock)
    pending=jobs.copy();completed=[];failed=[]
    while pending:
        remaining=[]
        for name,folder,pid,cmd in pending:
            if (BASE/folder/'pipeline_complete.json').exists():
                for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
                with (OUT/(name+'.log')).open('a') as log:r=subprocess.run([sys.executable,*cmd],stdout=log,stderr=subprocess.STDOUT)
                if r.returncode:failed.append({'job':name,'reason':'report verification failed','returncode':r.returncode})
                else:completed.append(name)
                print('REPORT',name,r.returncode,flush=True)
            else:
                try:os.kill(pid,0)
                except ProcessLookupError:failed.append({'job':name,'reason':'dependency ended before pipeline completion; inspect logs, no restart'})
                else:remaining.append((name,folder,pid,cmd))
        pending=remaining;write_json(OUT/'status.json',{'pid':os.getpid(),'pending':[x[0] for x in pending],'completed':completed,'failed':failed,'status':'running' if pending else 'failed' if failed else 'complete','updated_unix':time.time()})
        if pending:time.sleep(10)
    if failed:raise RuntimeError(failed)

if __name__=='__main__':main()
