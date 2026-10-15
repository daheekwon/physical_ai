"""Gate complete development latency replay on audited calibration timings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,write_json,sha256
from scripts.run_followup_policy_final import complete_calibration

BASE=resolve('artifacts/followup_v1')
OUT=BASE/'policy_latency'

def read(path):
    return json.loads(path.read_text())

def check_smoke():
    folder=BASE/'policy_latency_smoke'
    m=read(folder/'manifest.json');a=read(folder/'raw_audit.json');s=read(folder/'normalized_analysis.json')
    assert a['smoke'] and a['episodes']==6 and a['local_blocks']==34
    assert a['manifest_sha256']==sha256(folder/'manifest.json')
    assert a['max_error']['zero_reproduces_original_physics']==0
    assert a['max_error']['nominal_two_delay_profiles_exact']==0
    assert s['smoke'] and s['old_vs_normalized_active_rows_changed']==0
    for path,h in m['input_sha256'].items():assert sha256(resolve(path))==h,path
    for name,h in s['inputs'].items():assert sha256(folder/name)==h,name


def freeze():
    check_smoke();assert complete_calibration()
    OUT.mkdir(exist_ok=True)
    files=[resolve(p) for p in ['configs/followup_policy_latency.json','scripts/policy_latency_core.py','scripts/run_followup_policy_latency.py','scripts/analyze_followup_policy_latency.py','scripts/summarize_followup_policy_latency.py','scripts/run_followup_policy_latency_pipeline.py','scripts/run_followup_policy_final.py','scripts/analyze_followup_policy_final.py','configs/followup_v1.json','configs/followup_response_models.json','scripts/analyze_followup_response_selection.py']]
    files+=list(resolve('zj_wam').glob('*.py'))
    files += [BASE/p for p in ['selection/selector_rules.json','strong_baselines/calibrated_settings.json','response_scores/latent_scale.npy','policy_latency_smoke/raw_audit.json','policy_latency_smoke/normalized_analysis.json','policy_refinement/calibration/raw_audit.json','policy_refinement/calibration/analysis.json']]
    files+=list((BASE/'policy_refinement/calibration').glob('21*.json'))
    weights=[resolve('data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt'),resolve('data/checkpoints/lewm-pusht/weights.pt')]
    record={'input_sha256':{str(p.relative_to(resolve('.'))):sha256(p) for p in files+weights},'config':read(resolve('configs/followup_policy_latency.json')),'episodes':72,'scope':'4 development seeds x6 methods x3 timing profiles; 300 physical ticks each. Complete calibration timings, frozen original selection rules. Final latency test remains separately locked after validation.'}
    path=OUT/'pipeline_lock.json'
    if path.exists():assert read(path)==record,'Latency pipeline inputs changed'
    else:
        assert not list(OUT.glob('21*.json')),'Exposed development outputs before pipeline lock'
        with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    return record


def verify(record):
    for path,h in record['input_sha256'].items():assert sha256(resolve(path))==h,path


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int,required=True);a=p.parse_args();check_smoke();start=time.monotonic()
    print('WAITING for complete 72-episode calibration audit; smoke verified',flush=True)
    while not complete_calibration():
        try:os.kill(a.dependency_pid,0)
        except ProcessLookupError:raise RuntimeError('Calibration parent ended before audit gate. Inspect logs; no automatic restart.')
        if time.monotonic()-start>21600:raise RuntimeError('Observation window elapsed; revalidate live dependency')
        time.sleep(10)
    record=freeze();print('FROZEN latency development 72 episodes',flush=True)
    for script in ['run_followup_policy_latency.py','analyze_followup_policy_latency.py','summarize_followup_policy_latency.py']:
        verify(record)
        with (OUT/(script.removesuffix('.py')+'.log')).open('a') as log:
            subprocess.run([sys.executable,str(resolve('scripts/'+script))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',script,flush=True)
    verify(record);audit=read(OUT/'raw_audit.json');assert audit['episodes']==72 and not audit['smoke']
    write_json(OUT/'development_complete.json',{'episodes':72,'audit_sha256':sha256(OUT/'raw_audit.json'),'normalized_analysis_sha256':sha256(OUT/'normalized_analysis.json'),'lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()})

if __name__=='__main__':main()
