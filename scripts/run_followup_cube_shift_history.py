"""Collect causal Cube histories after shifted-physics warmup; no future fitting."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube,snapshot,restore
from scripts.run_followup_cube_native_policy import scene
from scripts.cube_shift_preflight_core import baseline,prepare
from scripts.cube_shift_history_core import control_rollout,points,warm_history,prepare_history
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');PARENT=BASE/'cube_policy_refinement';SRC=BASE/'cube_policy_oracle';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_shift_history.json','scripts/cube_shift_history_core.py','scripts/run_followup_cube_shift_history.py','scripts/cube_shift_preflight_core.py','zj_wam/cube_execution.py','zj_wam/cube_experiment.py','scripts/run_followup_cube_native_policy.py']
def main(smoke):
    cfg=read(resolve(SOURCES[0]));conditions=read(resolve(cfg['condition_source']))['conditions'];preflight=BASE/'cube_shift_preflight';assert read(preflight/'pipeline_complete.json')['raw_audit_sha256']==sha256(preflight/'raw_audit.json');paths=sorted(SRC.glob('*_t*.json'));paths=[p for p in paths if read(p)['seed']==340000] if smoke else paths;out=BASE/('cube_shift_history_smoke' if smoke else 'cube_shift_history');out.mkdir(exist_ok=True);native=read(PARENT/'manifest.json')['native'];files=[resolve(p) for p in SOURCES]+[resolve(cfg['condition_source']),preflight/'raw_audit.json',preflight/'pipeline_complete.json',preflight/'sham_comparison.json',PARENT/'manifest.json',SRC/'raw_audit.json'];seeds=sorted({read(p)['seed'] for p in paths})
    for p in paths:files.extend([p,p.with_suffix('.npz')])
    for seed in seeds:files.extend([PARENT/f'{seed}_nominal.json',PARENT/f'{seed}_nominal.npz'])
    manifest={'smoke':smoke,'config':cfg,'conditions':conditions,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    count=0
    for seed in seeds:
        source=dict(np.load(PARENT/f'{seed}_nominal.npz'));env,obs,info,_,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);initial=meter.measure();original=baseline(env);original['visual_statistics']={k:np.array(getattr(env._model.stat,k),copy=True) for k in ['center','extent','meansize']};current=0;prefix=[];targets=[]
        for p in sorted([p for p in paths if read(p)['seed']==seed],key=lambda p:read(p)['step']):
            meta=read(p);t=meta['step'];change=t-cfg['warmup_actions'];assert change>=1
            if change>current:
                trace,commands=control_rollout(env,meter,source['actions'][current:change]);prefix.append(trace);targets.extend(commands)
                for key,value in trace.items():np.testing.assert_array_equal(value,source['physics_'+key][current*25:change*25])
                current=change
            physics={key:np.concatenate([x[key] for x in prefix]) for key in prefix[0]};prepoints=points(initial,physics);saved=snapshot(env);parentraw=dict(np.load(p.with_suffix('.npz')))
            for condition in conditions:
                context=warm_history(env,meter,saved,original,condition,source['actions'][change:t],prepoints,np.array(targets),cfg);context['nominal_actions']=source['actions'][t:t+5];context['suffix_actions']=source['actions'][t+5:t+8];context['history_control_indices']=np.array([change,t-5,t]);folder=out/condition['name'];folder.mkdir(exist_ok=True);dest=folder/p.name;context['condition_factors']=np.array([condition['mass_factor'],condition['friction_factor']]);context['condition_change_control']=np.asarray(change)
                if condition['name']=='nominal':
                    for key in ['qpos','qvel','object_pose','object_velocity','eef_position','eef_quaternion','robot_object_contacts','time']:np.testing.assert_array_equal(context['warmup_'+key],source['physics_'+key][change*25:t*25])
                    np.testing.assert_array_equal(context['history_images'],parentraw['input_history_images']);np.testing.assert_allclose(context['past_points'][:,:3],parentraw['input_past_points'][:,:3],atol=0,rtol=0)
                # Future trace is measured only after all causal history fields are fixed.
                future=meter.rollout(source['actions'][t:t+cfg['future_actions']])['physics'];context.update({'nominal_future_'+key:value for key,value in future.items()});np.savez_compressed(dest.with_suffix('.npz'),**context);write_json(dest,{'seed':seed,'phase':meta['phase'],'step':t,'change_step':change,'condition':condition['name'],'source':str(p.relative_to(resolve('.'))),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz'))});count+=1;print('CUBE SHIFT HISTORY',seed,meta['phase'],condition['name'],count,flush=True)
            prepare_history(env,saved,original,conditions[0]);restore(env,saved)
        env.close()
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    assert count==len(paths)*len(conditions);write_json(out/'execution_complete.json',{'smoke':smoke,'states':len(paths),'families':len(seeds),'condition_contexts':count,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Causal changedhistory preparation only. Independent audit required before refinement/adaptation experiments.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
