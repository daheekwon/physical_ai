"""All-candidate physical branches from fixed nominal policy roots."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube,snapshot,restore
from zj_wam.cube_experiment import metrics
from scripts.run_followup_cube_native_policy import scene
from scripts.cube_policy_refinement_core import pipeline
from scripts.run_followup_cube import Model
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');SRC=BASE/'cube_policy_refinement';read=lambda p:json.loads(p.read_text())


def states(seed,cfg):
    with np.load(SRC/f'{seed}_nominal.npz') as a:
        steps=np.arange(15,300,5);ix=steps*25-1;z=a['physics_object_pose'][ix,2];contact=a['physics_robot_object_contacts'][ix]>0;z0=float(a['initial_object_pose'][2]);chosen={'approach':15};g=steps[contact&(z<=z0+cfg['grasp_height_above_initial_max_m'])];chosen['grasp']=int(g[0]) if len(g) else None;t=steps[contact&(z>=z0+cfg['transport_height_above_initial_min_m'])];chosen['transport']=int(t[0]) if len(t) else None;r=steps[(~contact)&(z<=z0+cfg['grasp_height_above_initial_max_m'])&(steps>(chosen['transport'] or 300))];chosen['release']=int(r[0]) if len(r) else None
    return chosen


def main(smoke):
    cfg=read(resolve('configs/followup_cube_policy_oracle.json'));out=BASE/('cube_policy_oracle_smoke' if smoke else 'cube_policy_oracle');out.mkdir(exist_ok=True);parent=read(SRC/'manifest.json');native=parent['native'];ccfg={**parent['cube'],'device':'cpu','yaw_length_scale_m':0.};torch.set_num_threads(cfg['torch_threads']);seeds=cfg['seeds'][:1] if smoke else cfg['seeds'];phase_map={str(s):states(s,cfg) for s in seeds};files=[resolve(p) for p in ['configs/followup_cube_policy_oracle.json','scripts/run_followup_cube_policy_oracle.py','scripts/cube_policy_refinement_core.py','scripts/run_followup_cube_native_policy.py','scripts/run_followup_cube.py','zj_wam/cube_execution.py','zj_wam/cube_experiment.py']]+[SRC/'manifest.json',SRC/'raw_audit.json',SRC/'pipeline_complete.json',BASE/'cube/proposal_ablation/analysis.json',resolve(ccfg['checkpoint'])/'weights.pt'];files += [SRC/f'{s}_nominal{ext}' for s in seeds for ext in ['.json','.npz']];manifest={'smoke':smoke,'config':cfg,'phases':phase_map,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    norm=read(BASE/'cube/manifest.json')['normalization'];mean=np.array(norm['mean']);scale=np.array(norm['scale']);model=Model(ccfg,mean,scale);rules=read(BASE/'cube/proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules'];nstate=0;ncandidate=0
    for seed in seeds:
        source=SRC/f'{seed}_nominal.npz';meta=read(source.with_suffix('.json'));assert meta['data_sha256']==sha256(source)
        with np.load(source) as z:a=dict(z)
        env,obs,info,_,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);np.testing.assert_array_equal(agent._plan,a['plan']);meter=DenseCube(env);initial=meter.measure()
        for k in initial:np.testing.assert_array_equal(initial[k],a['initial_'+k])
        bystep={v:k for k,v in phase_map[str(seed)].items() if v is not None};assert len(bystep)==sum(x is not None for x in phase_map[str(seed)].values());block_lookup={int(a[f'block_{i:03d}_step']):i for i in range(meta['blocks'])};images={0:env.render().copy()};current=0
        for step,phase in sorted(bystep.items()):
            if step>current:
                for t in range(current,step):
                    trace=meter.rollout(a['actions'][t:t+1])['physics']
                    for k,v in trace.items():np.testing.assert_array_equal(v,a['physics_'+k][t*25:(t+1)*25])
                    if (t+1)%5==0:images[t+1]=env.render().copy()
                current=step
            dest=out/f'{seed}_{phase}_t{step:03d}.json';bi=block_lookup[step];get=lambda k:a[f'block_{bi:03d}_{k}'];root=snapshot(env)
            if dest.exists():record=read(dest);assert record['manifest_sha256']==sha256(out/'manifest.json');nstate+=1;ncandidate+=record['candidates'];continue
            frames=np.array([images[step-10],images[step-5],images[step]]);np.testing.assert_array_equal(frames,get('history_images'));context={'nominal_actions':get('nominal_actions'),'suffix_actions':get('suffix_actions'),'past_points':get('past_points')[-251:],'past_targets':get('past_targets')[-250:],'workspace_bounds':a['bounds'],'history_images':frames,'past_actions':a['actions'][step-10:step],'root_gripper_contact':float(get('root_gripper_contact'))};prefix={k:a['physics_'+k][step*25-75:step*25] for k in initial};raw={};banks={};started=time.perf_counter()
            for method in cfg['methods']:
                banks[method]=pipeline(context,{'episode':seed,'timestep':step},ccfg,{'bank':'non_wm_proposals','radii':cfg['radii']},method,rules[method],model if method=='lewm_global' else None,mean,scale)
                if method!=cfg['methods'][0]:
                    for k in ['actions','coefficients','costs','norm','predicted_robot']:np.testing.assert_array_equal(banks[method][k],banks[cfg['methods'][0]][k])
                raw[method+'_scores']=banks[method]['scores']
            bank=banks[cfg['methods'][0]];rows=[];nominal=None
            for i,action in enumerate(bank['actions']):
                restore(env,root);trace=meter.rollout(np.r_[action,context['suffix_actions']])['physics']
                if i==0:
                    nominal=trace
                    for k,v in trace.items():np.testing.assert_array_equal(v,a[f'local_{bi:03d}_nominal_{k}'])
                for k,v in trace.items():raw.setdefault('candidate_'+k,[]).append(v)
                row={'seed':seed,'initial_pose_family':seed,'phase':phase,'step':step,'candidate':i,'generator':bank['labels'][i][0],'radius':bank['labels'][i][1],'sign':bank['labels'][i][2],'norm':float(bank['norm'][i]),'predicted_jerk_cost':float(bank['costs'][i]),'predicted_gain':float((bank['costs'][0]-bank['costs'][i])/max(bank['costs'][0],1e-12)),**metrics(trace,nominal,prefix,ccfg)};rows.append(row)
            restore(env,root);raw={k:np.asarray(v) for k,v in raw.items()};raw.update({k:bank[k] for k in ['actions','coefficients','costs','norm','predicted_robot','basis','solver_success','solver_iterations','optimizer_diagnostics']});raw['labels']=np.array(bank['labels'],dtype=str);raw.update({'input_'+k:np.asarray(v) for k,v in context.items()});raw.update({'prefix_'+k:v for k,v in prefix.items()});raw.update(mean=mean,scale=scale);np.savez_compressed(dest.with_suffix('.npz'),**raw);pd.DataFrame(rows).to_csv(dest.with_suffix('.csv'),index=False);write_json(dest,{'seed':seed,'phase':phase,'step':step,'source_block':bi,'candidates':len(rows),'selected':{m:int(banks[m]['selected']) for m in cfg['methods']},'nominal_physics_exact':True,'prefix_physics_exact':True,'causal_images_exact':True,'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'seconds':time.perf_counter()-started});nstate+=1;ncandidate+=len(rows);print('ORACLE',seed,phase,step,len(rows),round(time.perf_counter()-started,2),flush=True)
        env.close()
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    assert nstate==sum(v is not None for phases in phase_map.values() for v in phases.values());write_json(out/'execution_complete.json',{'states':nstate,'candidates':ncandidate,'smoke':smoke,'manifest_sha256':sha256(out/'manifest.json'),'scope':cfg['scope']})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_cube_policy_oracle.json'))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(args.smoke)
