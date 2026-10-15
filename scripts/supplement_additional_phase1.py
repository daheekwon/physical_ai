"""Reproduce nominal-episode quality, frequency CIs and jerk-gradient diagnostics."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve,write_json
from zj_wam.metrics import rotation_error,cluster_bootstrap


def nominal_quality(out):
    from zj_wam.adapters import ExpertDataset,ReplaySimulator
    manifest=json.loads(resolve('artifacts/phase1/manifest.json').read_text())
    data=ExpertDataset(manifest['config']['dataset']);rows=[]
    for split in ['development','calibration','test']:
        for ep in sorted({s['episode'] for s in manifest['states'] if s['split']==split}):
            actions,states=data.episode(ep)
            sim=ReplaySimulator(actions,states,len(actions)-1,manifest['config']['seed']+ep)
            actual=sim.restore();success,distance=sim.env.eval_state(sim.goal_state,actual)
            rows.append({'split':split,'episode':ep,'nominal_episode_success':bool(success),
                         'official_state_distance':float(distance),'object_coverage':sim.coverage(),
                         'final_rotation_error':float(rotation_error(actual[4],states[-1,4]))})
            sim.close()
    pd.DataFrame(rows).to_csv(out/'nominal_episode_quality.csv',index=False)


def statistics(out):
    f=pd.concat([pd.read_csv(p) for p in (out/'controls').glob('*.csv')])
    f['method']=f.direction.str.replace(r'_\d+$','',regex=True)
    g=f[(f.interaction_phase=='contact')&(f.relative_radius==.1)]
    rng=np.random.default_rng(20261010);result={}
    for metric in ['position_error','rotation_error','object_error','pusher_path_mean']:
        piv=g.groupby(['episode','state_id','sign','method'])[metric].mean().unstack('method')
        diff=(piv.low-piv.frequency).groupby(level=['episode','state_id']).mean()
        result[metric]={'low_minus_frequency_mean':float(diff.mean()),'cluster_ci':cluster_bootstrap(diff.values,diff.index.get_level_values('episode'),rng,2000)}
    write_json(out/'frequency_primary_ci.json',result)
    oracle=pd.read_csv(out/'oracle_candidates.csv');rows=[]
    for (sid,h),g in oracle.groupby(['state_id','horizon']):
        a=np.load(out/'oracle'/f'{sid}_h{h:02d}.npz');q=a['polynomial_basis']
        directions=dict(zip(a['direction_names'],a['directions']))
        small=g[g.relative_radius==g.relative_radius.min()]
        design=[];jerk_derivative=[];command_derivative=[]
        for name,v in small.groupby('direction'):
            minus=v[v.sign==-1].iloc[0];plus=v[v.sign==1].iloc[0]
            design.append(q.T@directions[name])
            jerk_derivative.append((plus.executed_jerk-minus.executed_jerk)/(2*plus.applied_radius))
            command_derivative.append((plus.action_roughness-minus.action_roughness)/(2*plus.applied_radius))
        design=np.array(design);j=np.array(jerk_derivative);c=np.array(command_derivative)
        gj=np.linalg.lstsq(design,j,rcond=None)[0];gc=np.linalg.lstsq(design,c,rcond=None)[0]
        low=np.stack([q.T@directions[f'poly_low_{i}'] for i in range(2)],axis=1)
        rows.append({'state_id':sid,'split':g.split.iloc[0],'episode':int(g.episode.iloc[0]),'horizon':int(h),
                     'executed_gradient_norm':float(np.linalg.norm(gj)),
                     'low_two_gradient_energy_fraction':float(np.linalg.norm(low.T@gj)**2/max(np.linalg.norm(gj)**2,1e-30)),
                     'command_executed_gradient_cosine':float(gj@gc/max(np.linalg.norm(gj)*np.linalg.norm(gc),1e-30)),
                     'finite_difference_fit_relative_residual':float(np.linalg.norm(design@gj-j)/max(np.linalg.norm(j),1e-30))})
    pd.DataFrame(rows).to_csv(out/'objective_gradient_alignment.csv',index=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nominal-quality',action='store_true');args=p.parse_args()
    out=resolve('artifacts/phase1_additional')
    if args.nominal_quality:nominal_quality(out)
    else:statistics(out)
