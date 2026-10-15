"""Check effective Cube mass/friction changes before adaptation claims."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube,snapshot,restore
from scripts.run_followup_cube_native_policy import scene
from scripts.cube_shift_preflight_core import baseline,prepare
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');SRC=BASE/'cube_policy_oracle';PARENT=BASE/'cube_policy_refinement';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_shift_preflight.json','scripts/run_followup_cube_shift_preflight.py','scripts/cube_shift_preflight_core.py','zj_wam/cube_execution.py','scripts/run_followup_cube_native_policy.py']

def main(smoke):
    cfg=read(resolve(SOURCES[0]));out=BASE/('cube_shift_preflight_smoke' if smoke else 'cube_shift_preflight');out.mkdir(exist_ok=True);parent=read(PARENT/'manifest.json');native=parent['native'];audit=read(SRC/'raw_audit.json');assert audit['states']==48 and read(SRC/'pipeline_complete.json')['raw_audit_sha256']==sha256(SRC/'raw_audit.json');allpaths=sorted(SRC.glob('*_t*.json'));paths=[p for p in allpaths if read(p)['seed']==340000] if smoke else allpaths;files=[resolve(p) for p in SOURCES]+[SRC/'manifest.json',SRC/'raw_audit.json',SRC/'pipeline_complete.json',PARENT/'manifest.json'];seeds=sorted({read(p)['seed'] for p in paths})
    for p in paths:files.extend([p,p.with_suffix('.npz')])
    for seed in seeds:files.extend([PARENT/f'{seed}_nominal.json',PARENT/f'{seed}_nominal.npz'])
    manifest={'smoke':smoke,'config':cfg,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    states=0;branches=0
    for seed in seeds:
        src=dict(np.load(PARENT/f'{seed}_nominal.npz'));env,obs,info,_,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);original=baseline(env);current=0
        for p in sorted([p for p in paths if read(p)['seed']==seed],key=lambda p:read(p)['step']):
            meta=read(p);t=meta['step'];raw=dict(np.load(p.with_suffix('.npz')))
            if t>current:
                trace=meter.rollout(src['actions'][current:t])['physics']
                for key,value in trace.items():np.testing.assert_array_equal(value,src['physics_'+key][current*25:t*25])
                current=t
            root=snapshot(env);actions=np.r_[raw['actions'][0],raw['input_suffix_actions']];arrays={'actions':actions,'body_ids':original['bodies'],'geom_ids':original['geoms'],'baseline_mass':original['body_mass'],'baseline_inertia':original['body_inertia'],'baseline_friction':original['geom_friction']}
            for condition in cfg['conditions']:
                name=condition['name']
                for repeat in range(cfg['repeats']):
                    params=prepare(env,root,original,condition);trace=meter.rollout(actions)['physics'];tag=f'{name}_r{repeat}_'
                    arrays.update({tag+k:v for k,v in trace.items()});arrays.update({tag+'parameter_'+k:v for k,v in params.items()});branches+=1
                    if name=='nominal':
                        for key,value in trace.items():np.testing.assert_array_equal(value,raw['candidate_'+key][0])
                    if repeat:
                        for key,value in trace.items():np.testing.assert_array_equal(value,arrays[name+'_r0_'+key])
            prepare(env,root,original,cfg['conditions'][0]);restore(env,root);dest=out/p.name;np.savez_compressed(dest.with_suffix('.npz'),**arrays);write_json(dest,{'seed':seed,'phase':meta['phase'],'step':t,'source':str(p.relative_to(resolve('.'))),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz'))});states+=1;print('CUBE SHIFT',seed,meta['phase'],t,'branches',branches,flush=True)
        env.close()
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':smoke,'states':states,'families':len(seeds),'branches':branches,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Physical condition-validity collection; no adaptation/smoothing performance claim. Independent audit required.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
