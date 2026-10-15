"""Collect reserved Cube policy references only after complete method/rule lock."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json
from scripts.run_followup_cube_plan_chunks import execute
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main(smoke):
    cfg=read(resolve('configs/followup_cube_policy_final.json'));compiler=read(resolve('configs/followup_cube_plan_chunks.json'));native=read(resolve('configs/followup_cube_native_policy.json'));out=BASE/('cube_policy_final_nominal_smoke' if smoke else 'cube_policy_final_nominal');out.mkdir(exist_ok=True);seeds=[cfg['smoke_seed']] if smoke else list(range(cfg['test_seed_start'],cfg['test_seed_start']+cfg['test_families']));reg=read(BASE/'cube_policy_calibration_nominal/initial_registry.json');files=[resolve(p) for p in ['configs/followup_cube_policy_final.json','configs/followup_cube_plan_chunks.json','configs/followup_cube_native_policy.json','scripts/run_followup_cube_policy_final_nominal.py','scripts/audit_followup_cube_policy_final_nominal.py','scripts/run_followup_cube_plan_chunks.py','scripts/cube_plan_chunk_core.py','zj_wam/cube_execution.py','zj_wam/cube_experiment.py']]+[BASE/'cube_policy_calibration_nominal/initial_registry.json',BASE/'cube_policy_selection/raw_audit.json',BASE/'cube_policy_selection/pipeline_complete.json']
    final=None
    if not smoke:
        fp=BASE/'cube_policy_final/final_protocol.json';final=read(fp);assert final['config']==cfg
        for p,h in final['inputs'].items():assert sha256(resolve(p))==h,p
        files.append(fp)
    else:files.extend(BASE/'cube_plan_chunks'/f'{cfg["smoke_seed"]}_past_pd_h5{ext}' for ext in ['.json','.npz'])
    manifest={'smoke':smoke,'config':cfg,'compiler':compiler,'native':native,'seeds':seeds,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    for seed in seeds:
        if final:
            for p,h in final['inputs'].items():assert sha256(resolve(p))==h,p
        dest=out/f'{seed}_nominal.json'
        if dest.exists():m=read(dest);assert m['manifest_sha256']==sha256(out/'manifest.json') and m['data_sha256']==sha256(dest.with_suffix('.npz'));continue
        start=time.perf_counter();arrays,meta=execute(seed,cfg['mode'],compiler,native)
        if smoke:
            with np.load(BASE/'cube_plan_chunks'/f'{seed}_past_pd_h5.npz') as old:
                for k,v in arrays.items():np.testing.assert_array_equal(v,old[k])
        else:
            r=reg['families'][str(seed)];assert r['split']=='reserved_test';np.testing.assert_array_equal(arrays['initial_qpos'],r['initial_qpos']);np.testing.assert_array_equal(arrays['initial_qvel'],r['initial_qvel']);np.testing.assert_array_equal(arrays['goal_position'],r['goal_position'])
        np.savez_compressed(dest.with_suffix('.npz'),**arrays);write_json(dest,{**meta,'smoke':smoke,'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'seconds':time.perf_counter()-start});print('FINAL CUBE NOMINAL',seed,meta['success_final'],flush=True)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':smoke,'episodes':len(seeds),'manifest_sha256':sha256(out/'manifest.json'),'reserved_test_policy_outcomes_executed':not smoke,'scope':'Reference collection only; independent audit required before finalcorrection evaluation.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
