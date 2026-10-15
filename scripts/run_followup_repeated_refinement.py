"""F5 horizon/observation-refresh and F10 repeated expert refinement pilot."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,dense_metrics
from zj_wam.diagnostics import jerk_energy
from zj_wam.repeated_refinement import choose_correction
from zj_wam.io import resolve,write_json,sha256


def core_metrics(result,reference,tolerances,h=20,suffix=3):
    n=(h+suffix)*10
    def truncate(r):
        x=dict(r);x['physics']=r['physics'][:n];x['control']=r['control'][:h+suffix]
        x['physics_trace']=r['physics_trace'][:30+n];x['control_trace']=r['control_trace'][:3+h+suffix]
        x['physics_contacts']=r['physics_contacts'][:n];x['physics_jerk']=jerk_energy(x['physics_trace'][:,:2],.01);x['control_jerk']=jerk_energy(x['control_trace'][:,:2],.1)
        # Coverage at core suffix end is captured separately by the runner.
        x['coverage_suffix']=r['coverage_core_suffix'];return x
    return dense_metrics(truncate(result),truncate(reference),tolerances)


def execute_with_coverage(sim,raw,suffix):
    # Dense execution already captures T and full suffix. Capture the core end
    # through an independent identical replay so no interpolation is introduced.
    full=sim.execute_dense(raw,suffix,capture_endpoints=True)
    if len(suffix)==3:full['coverage_core_suffix']=full['coverage_suffix']
    else:full['coverage_core_suffix']=sim.execute_dense(raw,suffix[:3])['coverage_suffix']
    return full


def process(path,cfg,rcfg,scfg,data,model,rules,out):
    meta=json.loads(path.read_text());spec={k:meta[k] for k in ['state_id','split','episode','timestep','initial_pose_family']};key=spec['state_id'];dest=out/'repeated_refinement';dest.mkdir(exist_ok=True);done=dest/f'{key}.json'
    if done.exists():return
    start=time.perf_counter();original,recorded=data.episode(spec['episode']);t=spec['timestep'];h=scfg['total_actions'];suffix=original[t+h:t+h+scfg['long_suffix_max_actions']]
    if len(suffix)<3:raise ValueError('Original eligible state lacks core suffix')
    sim=DenseSimulator(original,recorded,t,cfg['seed']+spec['episode']);reference=execute_with_coverage(sim,original[t:t+h],suffix);sim.close()
    contact=(reference['physics_contacts'][:h*10].reshape(h,10)>0).any(1);transition=bool(np.any(contact[1:]!=contact[:-1]))
    rows=[];artifacts={'nominal_physics':reference['physics_trace'],'nominal_control':reference['control_trace'],'nominal_actions':original[t:t+h]};decisions={}
    for name in scfg['methods']:
        delivered=original.copy();steps=[0] if name.startswith(('open20','short5')) else list(range(0,h,5));logs=[];predicted_history=None
        for offset in steps:
            block=h if name.startswith('open20') else 5;current=t+offset
            sim=DenseSimulator(delivered,recorded,current,cfg['seed']+spec['episode']);images,past=sim.restore(capture=True)
            nominal=original[current:current+block];local_suffix=original[current+block:current+block+3]
            method='lewm_global' if name=='repeat5_lewm' else 'vision_adaptive' if name=='repeat5_online_gains' else 'vision_contact'
            chosen,info,pose_history=choose_correction(model,images,past,sim.context_control,sim.prefix_physics,nominal,local_suffix,data.scaler.scale_,
                np.random.SeedSequence([cfg['seed'],spec['episode'],current,block]),cfg,rcfg,rules,method,
                predicted_pose_history=predicted_history if name=='repeat5_predicted_pose' else None,piecewise=name=='open20_piecewise_vision')
            delivered[current:current+block]=chosen;info['offset']=offset;info['block_actions']=block;logs.append(info)
            if name=='repeat5_predicted_pose':predicted_history=np.concatenate([pose_history[1:],np.array(info['predicted_endpoint_pose'])[None]],axis=0)
            sim.close()
        # Replay after decisions, then use physical outcomes only for evaluation.
        sim=DenseSimulator(original,recorded,t,cfg['seed']+spec['episode']);result=execute_with_coverage(sim,delivered[t:t+h],suffix);sim.close()
        core=core_metrics(result,reference,cfg['tolerances']);long=dense_metrics(result,reference,cfg['tolerances']);norm=np.linalg.norm(((delivered[t:t+h]-original[t:t+h])/data.scaler.scale_).reshape(-1))
        rows.append({**spec,'method':name,'horizon':h,'suffix_actions':len(suffix),'nominal_contact':bool(contact.any()),'nominal_contact_transition':transition,
                     'actual_norm':float(norm),'active_chunks':sum(x['active'] for x in logs),'replans':len(logs),'refinement_seconds':sum(x['seconds'] for x in logs),
                     'nominal_task_success':reference['nominal_success'],'modified_task_success':result['nominal_success'],**core,
                     **{'long_'+k:v for k,v in long.items()}})
        decisions[name]=logs;artifacts[name+'_actions']=delivered[t:t+h];artifacts[name+'_physics']=result['physics_trace'];artifacts[name+'_control']=result['control_trace']
    pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**artifacts)
    write_json(done,{**spec,'horizon':h,'suffix_actions':len(suffix),'seconds':time.perf_counter()-start,'decisions':decisions,
                    'source_sha256':sha256(Path(__file__)),'selector_sha256':sha256(resolve('zj_wam/repeated_refinement.py')),
                    'scope':'Fixed expert nominal actions; observation-driven refinement only. No claim of frozen Diffusion Policy closed-loop or Jacobian relinearization.'})
    print(key,round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration'],default='development');p.add_argument('--limit',type=int);args=p.parse_args()
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());scfg=json.loads(resolve('configs/followup_repeated_refinement.json').read_text());out=resolve(cfg['output']);torch.set_num_threads(cfg['threads'])
    data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);rules=json.loads((out/'selection/selector_rules.json').read_text())['rules']
    paths=sorted((out/'f1').glob(f'{args.split}*h05.json'))
    if len(paths)!=105:raise ValueError('Incomplete source states')
    if args.limit:paths=paths[:args.limit]
    for path in paths:process(path,cfg,rcfg,scfg,data,model,rules,out)
if __name__=='__main__':main()
