"""Post hoc controller-only baseline; no object state or world-model sensitivity."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from zj_wam.adapters import ExpertDataset,FrozenLeWM,ReplaySimulator
from zj_wam.io import resolve,write_json
from zj_wam.metrics import outcome_metrics,cluster_bootstrap


def controller_response(h,scale,kp=100.,kv=20.,dt=.01,substeps=10,action_scale=100.):
    response=[]
    for d in np.eye(2*h):
        q=np.zeros(2);v=np.zeros(2);trace=[]
        for a in d.reshape(h,2)*scale:
            target=q+action_scale*a
            for _ in range(substeps):
                v+=(kp*(target-q)-kv*v)*dt
                q+=v*dt
            trace.append(q.copy())
        response.append(np.asarray(trace).reshape(-1))
    return np.stack(response,axis=1)


def main():
    out=resolve('artifacts/phase1_additional');m=json.loads((out/'manifest.json').read_text());cfg=m['config']
    data=ExpertDataset(m['base_config']['dataset']);torch.set_num_threads(4)
    model=FrozenLeWM(m['base_config']['checkpoint'],data.scaler)
    response=controller_response(5,data.scaler.scale_);_,v=np.linalg.eigh(response.T@response);b=v[:,:2]
    dirs={f'controller_{i}':b@np.array([np.cos(a),np.sin(a)]) for i,a in enumerate(np.arange(4)*np.pi/4)}
    directory=out/'controller';directory.mkdir(exist_ok=True)
    all_rows=[];diagnostics=[]
    for spec in m['controls_states']:
        sid=spec['state_id'];done=directory/f'{sid}.json'
        if not done.exists():
            began=time.perf_counter();base=pd.read_csv(out/'controls'/f'{sid}.csv')
            original=dict(np.load(out/'controls'/f'{sid}.npz'))
            a=dict(np.load(resolve('artifacts/phase1/states')/f'{sid}.npz'))
            # Validate analytic controller against all signed coordinate executions.
            coordinate=base[(base.relative_radius==.1)&base.direction.str.startswith('coordinate_')]
            errors=[]
            for r in coordinate.itertuples():
                j=int(r.direction.split('_')[1]);actual=(original['trajectories'][r.trajectory_index,:,:2]-original['trajectories'][0,:,:2]).reshape(-1)
                actual_action_delta=((original['actions'][r.trajectory_index]-original['actions'][0])/data.scaler.scale_).reshape(-1)
                expected=response@actual_action_delta
                errors.append(float(np.max(np.abs(actual-expected))))
            # Float32 actions enter NumPy/Vec2d target arithmetic; allow <0.001 px
            # absolute roundoff, far below the 2 px object tolerance.
            assert max(errors)<1e-3, (sid,max(errors))
            actions,states=data.episode(spec['episode'])
            sim=ReplaySimulator(actions,states,spec['timestep'],m['base_config']['seed']+spec['episode'])
            ref=sim.execute(a['nominal_action']);predict,_=model.predictor(a['nominal_observation'],a['past_actions'])
            a0=torch.from_numpy(model.normalize(a['nominal_action']).reshape(-1))
            with torch.no_grad():z0=predict(a0)
            rows=[];traces=[ref['states']];candidates=[a['nominal_action']]
            for relative,radius in base.groupby('relative_radius').applied_radius.first().items():
                for name,d in dirs.items():
                    for sign in [-1,1]:
                        candidate=a0+torch.tensor(sign*radius*d,dtype=a0.dtype);raw=model.denormalize(candidate.numpy())
                        assert np.max(np.abs(raw))<=1+1e-5
                        result=sim.execute(raw)
                        with torch.no_grad():pred=float(torch.linalg.vector_norm(predict(candidate)-z0))
                        q=np.linalg.norm(result['states'][:,:2]-ref['states'][:,:2],axis=1)
                        rows.append({**spec,'interaction_phase':base.interaction_phase.iloc[0],'direction':name,'sign':sign,'relative_radius':relative,'applied_radius':radius,'predicted_change':pred,'pusher_path_mean':float(q.mean()),**outcome_metrics(result,ref),'trajectory_index':len(traces)})
                        traces.append(result['states']);candidates.append(raw)
            sim.close()
            pd.DataFrame(rows).to_csv(directory/f'{sid}.csv',index=False)
            np.savez_compressed(directory/f'{sid}.npz',trajectories=traces,actions=candidates,directions=np.stack(list(dirs.values())),response=response)
            low=a['directions'][list(a['direction_names']).index('low')]
            record={**spec,'controller_response_max_error':max(errors),'world_low_projection_into_controller_low_two':float(np.linalg.norm(b.T@low)**2),'duration_seconds':time.perf_counter()-began,'rows':len(rows)}
            write_json(done,record);print(json.dumps(record),flush=True)
        all_rows.append(pd.read_csv(directory/f'{sid}.csv'));diagnostics.append(json.loads(done.read_text()))
    f=pd.concat(all_rows,ignore_index=True);f.to_csv(out/'controller_candidates.csv',index=False)
    g=f[(f.interaction_phase=='contact')&(f.relative_radius==.1)]
    prior=pd.read_csv(out/'controls_candidates.csv');prior=prior[(prior.interaction_phase=='contact')&(prior.relative_radius==.1)]
    low=prior[prior.method=='low'].groupby(['episode','state_id']).object_error.mean()
    generic=g.groupby(['episode','state_id']).object_error.mean();diff=low-generic
    rng=np.random.default_rng(cfg['seed']+919)
    stats={'design':'Exploratory post hoc control; fixed official PD dynamics only, four fixed orientations in lowest two response directions; same applied norm as existing matched controls',
           'candidate_count':len(f),'contact_position_mean':float(g.position_error.mean()),'contact_pusher_path_mean':float(g.pusher_path_mean.mean()),
           'low_minus_controller_object_error':float(diff.mean()),'cluster_ci':cluster_bootstrap(diff.values,diff.index.get_level_values('episode'),rng,2000),
           'controller_response_max_error':max(r['controller_response_max_error'] for r in diagnostics),
           'mean_world_low_projection_into_controller_low_two':float(np.mean([r['world_low_projection_into_controller_low_two'] for r in diagnostics]))}
    write_json(out/'controller_statistics.json',stats);print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
