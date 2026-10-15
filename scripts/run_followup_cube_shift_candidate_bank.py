"""Execute common polynomial candidates at shifted current-observation policy roots."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import restore
from zj_wam.cube_experiment import metrics
from scripts.run_followup_cube import Model
from scripts.cube_policy_refinement_core import pipeline
from scripts.cube_shift_candidate_bank_core import replay_root,selector_context
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_shift_candidate_bank.json','scripts/run_followup_cube_shift_candidate_bank.py','scripts/audit_followup_cube_shift_candidate_bank.py','scripts/cube_shift_candidate_bank_core.py','scripts/cube_policy_refinement_core.py','scripts/cube_shift_history_core.py','scripts/cube_shift_preflight_core.py','scripts/run_followup_cube.py','scripts/run_followup_cube_native_policy.py','zj_wam/cube_execution.py','zj_wam/cube_experiment.py','scripts/audit_followup_cube_shift_history.py']
def main(smoke):
    cfg=read(resolve(SOURCES[0]));hist=BASE/'cube_shift_history';commands=BASE/'cube_shift_policy_context';ha=read(hist/'raw_audit.json');ca=read(commands/'raw_audit.json');assert ha['condition_contexts']==ca['contexts']==336
    for p,h in {**read(commands/'manifest.json')['inputs'],**ca['inputs']}.items():assert sha256(resolve(p))==h,p
    out=BASE/('cube_shift_candidate_bank_smoke' if smoke else 'cube_shift_candidate_bank');out.mkdir(exist_ok=True);hm=read(hist/'manifest.json');conditions={x['name']:x for x in hm['conditions']};parent=read(BASE/'cube_policy_refinement/manifest.json');native=parent['native'];ccfg={**parent['cube'],'device':'cpu','yaw_length_scale_m':0.};torch.set_num_threads(cfg['torch_threads']);paths=sorted(commands.glob('*/*_t*.json'));paths=[p for p in paths if read(p)['seed']==cfg['smoke_seed']] if smoke else paths
    files=[resolve(p) for p in SOURCES]+[commands/'manifest.json',commands/'raw_audit.json',hist/'manifest.json',hist/'raw_audit.json',hist/'pipeline_complete.json',BASE/'cube_policy_refinement/manifest.json',BASE/'cube/manifest.json',resolve(ccfg['checkpoint'])/'weights.pt']
    for p in paths:files.extend([p,p.with_suffix('.npz'),resolve(read(p)['history_source']),resolve(read(p)['history_source']).with_suffix('.npz')])
    manifest={'smoke':smoke,'config':cfg,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    normalization=read(BASE/'cube/manifest.json')['normalization'];mean=np.array(normalization['mean']);scale=np.array(normalization['scale']);model=Model(ccfg,mean,scale);count=0;candidates=0
    for path in paths:
        meta=read(path);hp=resolve(meta['history_source']);hmeta=read(hp);history=dict(np.load(hp.with_suffix('.npz')));compiled=dict(np.load(path.with_suffix('.npz')));context=selector_context(history,compiled);dest=out/meta['condition']/path.name;dest.parent.mkdir(exist_ok=True)
        if dest.exists():m=read(dest);assert m['manifest_sha256']==sha256(out/'manifest.json') and m['data_sha256']==sha256(dest.with_suffix('.npz'));count+=1;candidates+=m['candidates'];continue
        started=time.perf_counter();env,meter,root=replay_root(hmeta,history,conditions[meta['condition']],native,hm['config']);info=env.get_step_info();np.testing.assert_allclose(np.r_[info['proprio/effector_pos'],info['proprio/effector_yaw'][0]],compiled['input_root_proprio'],atol=1e-12,rtol=0);np.testing.assert_array_equal(info['proprio/gripper_opening'][0],compiled['input_gripper_history'][-1]);banks={};raw={}
        for method in cfg['methods']:
            banks[method]=pipeline(context,{'episode':meta['seed'],'timestep':meta['step']},ccfg,{'bank':'non_wm_proposals','radii':cfg['radii']},method,{'disabled':True},model if method=='lewm_global' else None,mean,scale)
            if len(banks)>1:
                for k in ['actions','coefficients','costs','norm','predicted_robot']:np.testing.assert_array_equal(banks[method][k],banks[cfg['methods'][0]][k])
            raw[method+'_scores']=banks[method]['scores']
        bank=banks[cfg['methods'][0]];raw.update({k:bank[k] for k in ['actions','coefficients','costs','norm','predicted_robot','basis','solver_success','solver_iterations','optimizer_diagnostics']});raw['labels']=np.array(bank['labels'],dtype=str);raw.update({'input_'+k:np.asarray(v) for k,v in context.items()});raw.update(mean=mean,scale=scale)
        # Persist the full candidate actions/predictions before any future physics.
        pre=dest.with_name(dest.stem+'.before_physics.npz');np.savez_compressed(pre,**raw);prehash=sha256(pre);prefix={k[7:]:v[-75:] for k,v in history.items() if k.startswith('warmup_')};raw.update({'prefix_'+k:v for k,v in prefix.items()});rows=[];nominal=None
        for i,actions in enumerate(bank['actions']):
            restore(env,root);trace=meter.rollout(np.r_[actions,context['suffix_actions']])['physics']
            if i==0:nominal=trace
            for k,v in trace.items():raw.setdefault('candidate_'+k,[]).append(v)
            rows.append({'state_id':path.stem,'seed':meta['seed'],'initial_pose_family':meta['seed'],'phase':meta['phase'],'condition':meta['condition'],'candidate':i,'norm':float(bank['norm'][i]),'predicted_cost':float(bank['costs'][i]),'predicted_gain':float((bank['costs'][0]-bank['costs'][i])/max(bank['costs'][0],1e-12)),**metrics(trace,nominal,prefix,ccfg)})
        env.close();raw={k:np.asarray(v) for k,v in raw.items()};np.savez_compressed(dest.with_suffix('.npz'),**raw);pd.DataFrame(rows).to_csv(dest.with_suffix('.csv'),index=False);write_json(dest,{**meta,'candidates':len(rows),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'before_physics_sha256':prehash,'seconds':time.perf_counter()-started});count+=1;candidates+=len(rows);print('SHIFT CANDIDATES',meta['seed'],meta['phase'],meta['condition'],len(rows),count,flush=True)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    assert count==len(paths);write_json(out/'execution_complete.json',{'smoke':smoke,'contexts':count,'candidates':candidates,'manifest_sha256':sha256(out/'manifest.json'),'scope':cfg['scope']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=read(resolve(SOURCES[0]))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(a.smoke)
