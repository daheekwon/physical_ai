"""Freeze smoke validation and stream full F4/F5 diagnostics, followed by audit."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_effect_readouts';read=lambda p:json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection-pid',type=int,required=True);args=p.parse_args();smoke=read(BASE/'dinowm_effect_readouts_smoke/raw_audit.json');assert smoke['smoke'] and smoke['states']==4 and smoke['decoded_array_checks']==180 and smoke['source_sha256']==sha256(resolve('scripts/audit_retrained_dinowm_effect_readouts.py'))
    paths=[resolve(p) for p in ['configs/followup_dinowm_effect_readouts.json','scripts/retrained_dinowm_effect_readout_core.py','scripts/analyze_retrained_dinowm_effect_readouts.py','scripts/audit_retrained_dinowm_effect_readouts.py','scripts/run_retrained_dinowm_effect_readouts_pipeline.py']]+[BASE/'dinowm_effect_readouts_smoke/raw_audit.json',BASE/'dinowm_effect_readouts_smoke/cube_geometry_controls.json',BASE/'dinowm_interventions/pipeline_lock.json'];lock={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'scope':'Full402 states, no finaltest. Stream completed token records; aggregate only after collection raw audit. Independent complete readout audit required.'};OUT.mkdir(exist_ok=True)
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for script,extra in [('analyze_retrained_dinowm_effect_readouts.py',['--dependency-pid',str(args.collection_pid)]),('audit_retrained_dinowm_effect_readouts.py',[])]:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:subprocess.run([sys.executable,str(resolve('scripts/'+script)),*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    a=read(OUT/'raw_audit.json');assert a['states']==402
    write_json(OUT/'pipeline_complete.json',{'states':402,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})
if __name__=='__main__':main()
