"""Run follow-up experiments with immutable splits and resumable per-state output."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import DenseSimulator, dense_metrics, controller_rollout, optimize_controller
from zj_wam.diagnostics import polynomial_basis, jerk_energy
from zj_wam.io import resolve, write_json, sha256


def exposures(family_of, output):
    exposed=set(); evidence=[]
    for p in sorted(resolve('artifacts').rglob('*.json')):
        if output in p.parents or 'setup' in p.parts:continue
        try:record=json.loads(p.read_text())
        except (ValueError,OSError):continue
        found=set()
        def walk(value):
            if isinstance(value,dict):
                for k,v in value.items():
                    if k in ('episode','episode_id') and isinstance(v,int) and 0<=v<len(family_of):
                        found.add(int(family_of[v]))
                    if k=='initial_pose_family' and isinstance(v,int):found.add(v)
                    walk(v)
            elif isinstance(value,list):
                for v in value:walk(v)
        walk(record)
        if found:
            evidence.append({'path':str(p.relative_to(resolve('.'))),'sha256':sha256(p),'families':sorted(found)})
            exposed.update(found)
    return exposed,evidence


def initialize(cfg,data,out):
    path=out/'protocol/manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        if m['config']!=cfg:raise ValueError('Frozen config changed; create a new protocol version')
        return m
    starts=data.states[np.asarray(data.dataset.offsets,dtype=int),:5]
    _,family=np.unique(starts,axis=0,return_inverse=True)
    exposed,evidence=exposures(family,out)
    rng=np.random.default_rng(cfg['seed'])
    old=np.array(sorted(exposed));rng.shuffle(old)
    unseen=sorted(set(int(x) for x in family)-exposed)
    if len(unseen)!=cfg['family_counts']['test']:raise ValueError('Unexpected untouched family count')
    assignment={};offset=0
    for split,count in cfg['family_counts'].items():
        if split=='test':assignment[split]=unseen
        else:
            assignment[split]=old[offset:offset+count].tolist();offset+=count
    if offset!=len(old):raise ValueError('Unassigned or oversubscribed exposed families')
    states=[];episodes=[];excluded=[]
    future=max(cfg['horizons'])+cfg['suffix_actions']
    for split,families in assignment.items():
        for fam in families:
            choices=rng.permutation(np.where(family==fam)[0])
            chosen=None
            for ep in choices:
                a,_=data.episode(int(ep))
                ts=np.array([t for t in range(10,len(a)-future+1)
                             if np.isfinite(a[t:t+future]).all() and np.max(np.abs(a[t:t+future]))<=1])
                if len(ts)>=cfg['states_per_family']:
                    chosen=(int(ep),ts);break
                excluded.append({'episode':int(ep),'family':fam,'reason':'insufficient bounded future windows'})
            if chosen is None:raise ValueError(f'No eligible episode in family {fam}')
            ep,ts=chosen;episodes.append({'split':split,'episode':ep,'initial_pose_family':fam})
            for values in np.array_split(ts,cfg['states_per_family']):
                t=int(rng.choice(values))
                states.append({'split':split,'episode':ep,'initial_pose_family':fam,
                               'timestep':t,'state_id':f'{split}_f{fam:03d}_e{ep:05d}_t{t:04d}'})
    write_json(out/'protocol/exposure.json',{'exposed_families':sorted(exposed),'unexposed_families':unseen,'evidence':evidence})
    source_paths=['docs/ZJ_WAM_FOLLOWUP_EXPERIMENT_PLAN.md','docs/ZJ_WAM_Research_Proposal_Updated_EN.tex',
                  'zj_wam/adapters.py','zj_wam/followup.py','scripts/run_followup.py',
                  'external/stable-worldmodel/stable_worldmodel/envs/pusht/env.py']
    m={'config':cfg,'families':assignment,'episodes':episodes,'states':states,'selection_exclusions':excluded,
       'normalization_mean':data.scaler.mean_.tolist(),'normalization_scale':data.scaler.scale_.tolist(),
       'checkpoint_sha256':sha256(resolve(cfg['checkpoint'])/'weights.pt'),
       'source_sha256':{p:sha256(resolve(p)) for p in source_paths},
       'created_unix':time.time(),'test_scope':cfg['scope'],
       'upstream_exposure':'Full dataset used by official normalization and possibly pretrained WM; independence is downstream experimental selection only.'}
    write_json(path,m)
    write_json(out/'protocol/progress.json',{'objective':'Execute and report all F0-F11 in follow-up plan',
        'steps':{f'F{i}':{'status':'in_progress' if i==0 else 'pending'} for i in range(12)}})
    return m


def audit(spec,cfg,data,out):
    p=out/'protocol/states'/f"{spec['state_id']}.json"
    if p.exists():return json.loads(p.read_text())
    started=time.perf_counter();a,s=data.episode(spec['episode']);t=spec['timestep'];h=max(cfg['horizons'])
    sim=DenseSimulator(a,s,t,cfg['seed']+spec['episode'])
    sim.restore();root=sim.root_full_state.copy()
    ref=sim.execute_dense(a[t:t+h],a[t+h:t+h+3],restore=False)
    repeated=sim.execute_dense(a[t:t+h],a[t+h:t+h+3])
    repeat=float(np.max(np.abs(ref['physics_trace']-repeated['physics_trace'])))
    pred=controller_rollout(a[t:t+h+3],root[:2],root[5:7])
    controller_error=float(np.max(np.abs(pred-ref['physics'][:,:2])))
    # Dataset states label the state before action t; include the corrected chunk and suffix.
    recorded=s[t:t+h+4]
    actual=np.concatenate([root[None,:7],ref['control'][:,:7]])
    n=min(len(recorded),len(actual))
    pos=float(np.linalg.norm(actual[:n,2:4]-recorded[:n,2:4],axis=1).max())
    angle=float(np.abs(np.arctan2(np.sin(actual[:n,4]-recorded[:n,4]),np.cos(actual[:n,4]-recorded[:n,4]))).max())
    eligible=pos<=cfg['source_replay_position_limit'] and angle<=cfg['source_replay_rotation_limit']
    if repeat!=0 or controller_error>1e-3:raise AssertionError((spec,repeat,controller_error))
    record={**spec,'eligible':eligible,'source_position_max':pos,'source_rotation_max':angle,
            'repeat_full_state_max':repeat,'controller_position_max':controller_error,
            'nominal_contact_substeps':int(np.sum(ref['physics_contacts']>0)),
            'physics_samples':len(ref['physics']),'root_object_velocity':root[7:].tolist(),
            'duration_seconds':time.perf_counter()-started}
    write_json(p,record);sim.close();print(json.dumps(record),flush=True)
    return record


def candidate_bank(spec,cfg,data,out,h):
    directory=out/'f1';directory.mkdir(exist_ok=True)
    key=f"{spec['state_id']}_h{h:02d}";done=directory/f'{key}.json'
    if done.exists():return
    started=time.perf_counter();a,s=data.episode(spec['episode']);t=spec['timestep']
    nominal=a[t:t+h];suffix=a[t+h:t+h+3]
    sim=DenseSimulator(a,s,t,cfg['seed']+spec['episode']);images,past=sim.restore(capture=True)
    reference=sim.execute_dense(nominal,suffix,restore=False,capture_endpoints=True)
    root=reference['root_full_state'];prefix=sim.prefix_physics.copy()
    scale=data.scaler.scale_;q=polynomial_basis(h)
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],spec['episode'],t,h]))
    directions=rng.normal(size=(cfg['random_directions'],q.shape[1]));directions/=np.linalg.norm(directions,axis=1)[:,None]
    rows=[];actions=[nominal];traces=[reference['physics_trace']];controls=[reference['control_trace']]
    images_all=[reference['endpoint_images']];coverage=[[reference['coverage_t'],reference['coverage_suffix']]]
    contact=[reference['physics_contacts']]
    def execute(raw,method,radius,extra=None):
        if np.max(np.abs(raw))>1+1e-6:return
        result=sim.execute_dense(raw,suffix,capture_endpoints=True)
        predicted=controller_rollout(np.concatenate([raw,suffix]),root[:2],root[5:7])
        predicted_jerk=jerk_energy(np.concatenate([prefix[:,:2],predicted]),.01)
        delta=((raw-nominal)/scale).reshape(-1)
        metrics=dense_metrics(result,reference,cfg['tolerances'])
        row={**spec,'horizon':h,'method':method,'relative_radius':float(radius),'index':len(actions),
             'actual_norm':float(np.linalg.norm(delta)),
             'polynomial_residual':float(np.linalg.norm(delta-q@(q.T@delta))),
             'predicted_jerk':predicted_jerk,
             'controller_position_error':float(np.max(np.abs(predicted-result['physics'][:,:2]))),
             'nominal_physics_jerk':reference['physics_jerk'],
             'nominal_contact':bool(np.any(reference['physics_contacts'][:h*10]>0)),
             'initial_gap':reference['initial_gap'],**metrics,**(extra or {})}
        rows.append(row);actions.append(raw);traces.append(result['physics_trace']);controls.append(result['control_trace'])
        images_all.append(result['endpoint_images']);coverage.append([result['coverage_t'],result['coverage_suffix']]);contact.append(result['physics_contacts'])
    for relative in cfg['radii']:
        radius=relative*np.sqrt(2*h)
        for objective in ['executed','raw']:
            raw,info=optimize_controller(nominal,suffix,root,prefix,scale,radius,objective,a[t-3:t])
            execute(raw,'optimizer_'+objective,relative,{'solver_success':info['solver_success'],'solver_iterations':info['iterations']})
        for i,d in enumerate(directions):
            for sign in [-1,1]:
                delta=sign*radius*(q@d).reshape(h,2)*scale
                execute(nominal+delta,f'random_{i}_sign{sign}',relative)
    frame=pd.DataFrame(rows)
    if frame.controller_position_error.max()>1e-3:raise AssertionError('Controller does not predict actual pusher')
    if frame.polynomial_residual.max()>1e-5:raise AssertionError('Correction left polynomial family')
    frame.to_csv(directory/f'{key}.csv',index=False)
    np.savez_compressed(directory/f'{key}.npz',actions=np.asarray(actions),physics=np.asarray(traces),control=np.asarray(controls),
                        endpoint_images=np.asarray(images_all),coverage=np.asarray(coverage),contacts=np.asarray(contact),
                        history_images=np.asarray(images),past_actions=past,root=root,basis=q)
    feasible=frame[(frame.object_error<=1)&(frame.actual_norm>1e-6)]
    useful=feasible[feasible.jerk_gain>=cfg['meaningful_jerk_reduction']]
    record={**spec,'horizon':h,'candidate_count':len(frame),'oracle_useful':len(useful)>0,
            'best_feasible_gain':float(max(0,feasible.jerk_gain.max())) if len(feasible) else 0.,
            'controller_max_error':float(frame.controller_position_error.max()),
            'duration_seconds':time.perf_counter()-started}
    write_json(done,record);sim.close();print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','audit','f1'])
    p.add_argument('--split',default='development');p.add_argument('--limit',type=int)
    p.add_argument('--horizon',type=int,default=5)
    args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text())
    torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);out=resolve(cfg['output'])
    manifest=initialize(cfg,data,out)
    if args.mode=='initialize':
        print(json.dumps({'families':{k:len(v) for k,v in manifest['families'].items()},'states':len(manifest['states'])}));return
    states=[s for s in manifest['states'] if (args.split=='all' or s['split']==args.split) and s['split']!='probe_train']
    if args.limit:states=states[:args.limit]
    if args.mode=='f1' and any(s['split']=='test' for s in states) and not (out/'protocol/final_selection_lock.json').exists():
        raise ValueError('Test interventions require a final selection lock')
    for spec in states:
        record=audit(spec,cfg,data,out)
        if args.mode=='f1' and record['eligible']:candidate_bank(spec,cfg,data,out,args.horizon)
    if args.mode=='audit':
        records=[json.loads(f.read_text()) for f in (out/'protocol/states').glob('*.json')]
        write_json(out/'protocol/audit_summary.json',{'audited_states':len(records),
                   'eligible_states':sum(r['eligible'] for r in records),
                   'excluded':[r for r in records if not r['eligible']],
                   'repeat_max':max(r['repeat_full_state_max'] for r in records),
                   'controller_max':max(r['controller_position_max'] for r in records)})

if __name__=='__main__':main()
