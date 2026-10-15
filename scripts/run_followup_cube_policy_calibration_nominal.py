"""Collect new Cube calibration nominals with the unchanged audited H5 compiler."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py,hdf5plugin,numpy as np
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube
from scripts.run_followup_cube_native_policy import scene
from scripts.run_followup_cube_plan_chunks import execute
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_policy_calibration_nominal.json','scripts/run_followup_cube_policy_calibration_nominal.py','configs/followup_cube_plan_chunks.json','configs/followup_cube_native_policy.json','scripts/run_followup_cube_plan_chunks.py','scripts/cube_plan_chunk_core.py','scripts/run_followup_cube_native_policy.py','zj_wam/cube_execution.py','zj_wam/cube_experiment.py']
def main(smoke):
    cfg=read(resolve(SOURCES[0]));compiler=read(resolve('configs/followup_cube_plan_chunks.json'));native=read(resolve('configs/followup_cube_native_policy.json'));out=BASE/('cube_policy_calibration_nominal_smoke' if smoke else 'cube_policy_calibration_nominal');out.mkdir(exist_ok=True);files=[resolve(p) for p in SOURCES]+[BASE/'cube_plan_chunks/manifest.json',BASE/'cube_plan_chunks/raw_audit.json',BASE/'cube_preflight/dataset_manifest.json',BASE/'cube/manifest.json'];assert read(BASE/'cube_plan_chunks/raw_audit.json')['episodes']==36;seeds=[cfg['smoke_seed']] if smoke else list(range(cfg['calibration_seed_start'],cfg['calibration_seed_start']+cfg['calibration_families']));registry=None
    if not smoke:
        registry_path=out/'initial_registry.json'
        if registry_path.exists():registry=read(registry_path)
        else:
            dataset=read(resolve('configs/followup_cube.json'))['dataset']
            with h5py.File(resolve(dataset),'r') as f:offsets=f['ep_offset'][:];expert=np.c_[f['qpos'][offsets],f['qvel'][offsets]]
            prior=[];prior_inputs={}
            for seed in compiler['development_seeds']:
                p=BASE/'cube_plan_chunks'/f'{seed}_past_pd_h5.npz'
                with np.load(p) as a:prior.append(np.r_[a['initial_qpos'],a['initial_qvel']])
                prior_inputs[str(p.relative_to(resolve('.')))]=sha256(p)
            existing=np.r_[expert,np.array(prior)];entries={};seen=[];allseeds=seeds+list(range(cfg['reserved_test_seed_start'],cfg['reserved_test_seed_start']+cfg['reserved_test_families']))
            for seed in allseeds:
                env,_,_,_,_=scene(seed,native);initial=DenseCube(env).measure();goal=env._data.mocap_pos[env._cube_target_mocap_ids[0]].copy();q=np.r_[initial['qpos'],initial['qvel']];env.close();distance=np.max(abs(existing-q),axis=1);assert distance.min()>1e-8,(seed,'prior family overlap')
                if seen:assert np.max(abs(np.array(seen)-q),axis=1).min()>1e-8,(seed,'new family overlap')
                seen.append(q);entries[str(seed)]={'split':'calibration' if seed in seeds else 'reserved_test','initial_qpos':initial['qpos'].tolist(),'initial_qvel':initial['qvel'].tolist(),'initial_time':float(initial['time']),'goal_position':goal.tolist(),'nearest_prior_max_coordinate_difference':float(distance.min())}
            registry={'families':entries,'expert_initial_count':len(expert),'prior_native_initial_count':len(prior),'prior_inputs':prior_inputs,'dataset_manifest_sha256':sha256(BASE/'cube_preflight/dataset_manifest.json'),'definition':'Initialqpos/qvel differ in maxcoordinate>1e-8; no policy actions/outcomes inspected for reservedtest; pretraining exposure separate.'};write_json(registry_path,registry)
        files.append(registry_path)
    if smoke:files.extend(BASE/'cube_plan_chunks'/f'{cfg["smoke_seed"]}_past_pd_h5{ext}' for ext in ['.json','.npz'])
    manifest={'smoke':smoke,'config':cfg,'compiler':compiler,'native':native,'seeds':seeds,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    for seed in seeds:
        dest=out/f'{seed}_nominal.json'
        if dest.exists():m=read(dest);assert m['manifest_sha256']==sha256(out/'manifest.json') and m['data_sha256']==sha256(dest.with_suffix('.npz'));continue
        started=time.perf_counter();arrays,meta=execute(seed,cfg['mode'],compiler,native)
        if smoke:
            with np.load(BASE/'cube_plan_chunks'/f'{seed}_past_pd_h5.npz') as old:
                for key,value in arrays.items():np.testing.assert_array_equal(value,old[key])
        else:
            reg=registry['families'][str(seed)];np.testing.assert_array_equal(arrays['initial_qpos'],reg['initial_qpos']);np.testing.assert_array_equal(arrays['initial_qvel'],reg['initial_qvel']);np.testing.assert_array_equal(arrays['goal_position'],reg['goal_position'])
        np.savez_compressed(dest.with_suffix('.npz'),**arrays);write_json(dest,{**meta,'smoke':smoke,'original_smoke_arrays_exact':bool(smoke),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'seconds':time.perf_counter()-started});print('NEW CUBE CAL NOMINAL',seed,meta['success_final'],flush=True)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':smoke,'episodes':len(seeds),'manifest_sha256':sha256(out/'manifest.json'),'reserved_test_policy_outcomes_executed':False,'scope':'Nominal collection only; independent compiler/physical audit needed before correction calibration.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
