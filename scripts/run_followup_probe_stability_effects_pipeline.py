"""Freeze validated repeated-probe effects and stream all common402 states."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'probe_stability_effects';read=lambda p:json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dino-pid',type=int,required=True);ap.add_argument('--lewm-pid',type=int,required=True);args=ap.parse_args();smoke=read(BASE/'probe_stability_effects_smoke/raw_audit.json');repro=read(BASE/'probe_stability_effects_smoke/reproducibility.json');assert smoke['smoke'] and smoke['states']==4 and smoke['candidate_method_rows_including_inactive']==114000 and smoke['decoded_array_checks']==1080;assert smoke['source_sha256']==sha256(resolve('scripts/audit_followup_probe_stability_effects.py'));assert repro['raw_audit_sha256']==sha256(BASE/'probe_stability_effects_smoke/raw_audit.json') and repro['models']==180 and repro['source_sha256']==sha256(resolve('scripts/summarize_followup_probe_effect_stability.py'))
    paths=[resolve(p) for p in ['configs/followup_probe_stability_effects.json','scripts/probe_stability_effect_core.py','scripts/analyze_followup_probe_stability_effects.py','scripts/audit_followup_probe_stability_effects.py','scripts/summarize_followup_probe_effect_stability.py','scripts/run_followup_probe_stability_effects_pipeline.py']]+[BASE/'probe_stability_effects_smoke/raw_audit.json',BASE/'probe_stability_effects_smoke/reproducibility.json',BASE/'probe_stability/capture_pipeline_complete.json',BASE/'dinowm_interventions/pipeline_lock.json',BASE/'lewm_common_effect_features/pipeline_lock.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'states':402,'models':180,'scope':'Stream completed and hashverified commonbank feature records; aggregate afterbothcollectionaudits, independentlyaudit allraw effects, thentraining-repetition summary. No finaltestorcorrectionselection.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script,extra in [('analyze_followup_probe_stability_effects.py',['--dino-pid',str(args.dino_pid),'--lewm-pid',str(args.lewm_pid)]),('audit_followup_probe_stability_effects.py',[]),('summarize_followup_probe_effect_stability.py',[])]:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script)),*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    a=read(OUT/'raw_audit.json');r=read(OUT/'reproducibility.json');assert a['states']==r['states']==402 and r['models']==180
    write_json(OUT/'pipeline_complete.json',{'states':402,'models':180,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'reproducibility_sha256':sha256(OUT/'reproducibility.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
if __name__=='__main__':main()
