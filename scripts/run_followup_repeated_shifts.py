"""Repeated correction under fixed-onset shifts, with post-choice local/global evaluation."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import dense_metrics
from scripts.repeated_shift_core import branch,run_method,core_metrics
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['scripts/run_followup_repeated_shifts.py','scripts/repeated_shift_core.py','scripts/run_followup_repeated_refinement.py','configs/followup_repeated_shifts.json','configs/followup_shift_evaluation.json','configs/followup_v1.json','configs/followup_response_models.json','zj_wam/shifted_execution.py','zj_wam/repeated_refinement.py','zj_wam/followup.py','zj_wam/adapters.py','zj_wam/selection.py','zj_wam/response_models.py','zj_wam/model_audit.py','zj_wam/locked_geometric.py','zj_wam/strong_optimizers.py','zj_wam/diagnostics.py']
def serial(x):return json.loads(json.dumps(x,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))
def main(smoke):
    cfg=read(resolve('configs/followup_v1.json'));rcfg=read(resolve('configs/followup_response_models.json'));scfg=read(resolve('configs/followup_repeated_shifts.json'));conditions=read(resolve(scfg['conditions_source']))['conditions'];out=BASE/('repeated_shifts_smoke' if smoke else 'repeated_shifts');out.mkdir(exist_ok=True);paths=sorted((BASE/'f1').glob('development*h05.json'));assert len(paths)==105
    if smoke:paths=paths[:1];conditions={k:v for k,v in conditions.items() if k in ['nominal','friction_02','rgb_noise_15']}
    rules=read(BASE/'selection/selector_rules.json')['rules'];strong=read(BASE/'strong_baselines/calibrated_settings.json')['rules'];files=[resolve(p) for p in SOURCES]+[BASE/'selection/selector_rules.json',BASE/'strong_baselines/calibrated_settings.json',resolve(cfg['checkpoint'])/'weights.pt',*paths];manifest={'smoke':smoke,'config':scfg,'conditions':conditions,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':scfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    data=ExpertDataset(cfg['dataset']);torch.set_num_threads(cfg['threads']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);count=0;allrows=[];fixture=[]
    for path in paths:
        meta=read(path);original,recorded=data.episode(meta['episode']);root=meta['timestep'];suffix=original[root+20:root+30];seed=cfg['seed']+meta['episode']
        for condition,params in conditions.items():
            condition_index=list(read(resolve(scfg['conditions_source']))['conditions']).index(condition);folder=out/condition;folder.mkdir(exist_ok=True);reference=branch(original,recorded,root,seed,params,original[root:root+20],suffix)
            for method in scfg['methods']:
                dest=folder/f"{meta['state_id']}__{method}.json"
                if dest.exists():
                    r=read(dest);assert r['manifest_sha256']==sha256(out/'manifest.json') and r['data_sha256']==sha256(dest.with_suffix('.npz'));allrows.append(r['summary']);count+=1;continue
                started=time.perf_counter()
                def save(offset,info,arrays):
                    target=folder/f'{dest.stem}.block{offset:02d}.before_physics.npz';np.savez_compressed(target,**arrays);write_json(target.with_suffix('.json'),{'decision':serial(info),'data_sha256':sha256(target),'manifest_sha256':sha256(out/'manifest.json')})
                actions,result,logs,local,arrays=run_method(original,recorded,meta,params,condition_index,method,model,data,cfg,rcfg,rules,strong,save)
                core=core_metrics(result,reference,cfg['tolerances']);long=dense_metrics(result,reference,cfg['tolerances']);norm=float(np.linalg.norm((actions-original[root:root+20])/data.scaler.scale_));row={k:meta[k] for k in ['state_id','split','episode','timestep','initial_pose_family']};row.update(condition=condition,method=method,actual_norm=norm,active=norm>1e-6,joint=bool(norm>1e-6 and core['object_error']<=1 and core['jerk_gain']>=.1),long_joint=bool(norm>1e-6 and max(long['object_error'],core['object_error'])<=1 and long['jerk_gain']>=.1),suffix_actions=len(suffix),nominal_task_success=reference['nominal_success'],modified_task_success=result['nominal_success'],refinement_seconds=sum(x['seconds'] for x in logs),**core,**{'long_'+k:v for k,v in long.items()});row['long_object_error']=max(long['object_error'],core['object_error']);row['long_preserved']=row['long_object_error']<=1
                arrays.update(actions=actions,nominal_actions=original[root:root+20],suffix=suffix)
                for label,r in [('nominal',reference),('modified',result)]:
                    for k in ['physics','control','physics_trace','control_trace','physics_contacts','root_full_state']:arrays[label+'_'+k]=r[k]
                    arrays[label+'_coverage']=np.array([r['coverage_t'],r['coverage_core_suffix'],r['coverage_suffix']])
                np.savez_compressed(dest.with_suffix('.npz'),**arrays);pd.DataFrame(local).to_csv(dest.with_suffix('.csv'),index=False)
                write_json(dest,{'summary':serial(row),'decisions':serial(logs),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'local_csv_sha256':sha256(dest.with_suffix('.csv')),'seconds':time.perf_counter()-started})
                count+=1;allrows.append(row);print('REPEATED SHIFT',condition,meta['state_id'],method,count,round(time.perf_counter()-started,2),flush=True)
                if smoke and condition=='nominal' and method in ['vision_contact','vision_adaptive','lewm_global']:
                    alias={'vision_contact':'repeat5_observed_pose','vision_adaptive':'repeat5_online_gains','lewm_global':'repeat5_lewm'}[method]
                    with np.load(BASE/'repeated_refinement'/(meta['state_id']+'.npz'),allow_pickle=False) as old:
                        np.testing.assert_array_equal(actions,old[alias+'_actions']);np.testing.assert_array_equal(result['physics_trace'],old[alias+'_physics'])
                    fixture.append({'method':method,'nominal_actions_and_all_physics_exact':True})
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    assert count==len(paths)*len(conditions)*len(scfg['methods']);pd.DataFrame(allrows).to_csv(out/'episodes.csv',index=False);write_json(out/'execution_complete.json',{'smoke':smoke,'states':len(paths),'conditions':len(conditions),'methods':len(scfg['methods']),'executions':count,'local_decisions':count*4,'manifest_sha256':sha256(out/'manifest.json'),'nominal_fixture':fixture,'scope':'Selection and execution collection only. Independent raw metrics, onset consistency, noisyhistory and fresh replay audit required before statistical conclusions.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):main(a.smoke)
