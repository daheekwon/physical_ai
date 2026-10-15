"""Independent physical and action audit for first-correction feedback branches."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,pymunk
from shapely.geometry import Polygon
from shapely.ops import unary_union
from zj_wam.io import resolve,write_json,sha256
from zj_wam.policy_refinement import make_scene,absolute_to_relative
from scripts.analyze_followup_response_selection import mean_ci

BASE=resolve('artifacts/followup_v1');OUT=BASE/'policy_effect_decomposition';SOURCE=BASE/'policy_refinement/development'


def read(p):return json.loads(p.read_text())


def arrays(p):
    with np.load(p) as a:return {k:a[k].copy() for k in a.files if k in ['initial_state','states','physics','actions','coverage','policy_predictions']}


def energy(q):return float(np.sum((np.diff(q,n=3,axis=0)/.01**3)**2)*.01)


class Coverage:
    def __init__(self,seed):
        env=make_scene(seed);self.vertices=[list(s.get_vertices()) for s in env.block.shapes]
        self.body=pymunk.Body(body_type=pymunk.Body.KINEMATIC);self.body.center_of_gravity=env.block.center_of_gravity
        goal=env._get_goal_pose_body(env.goal_pose);self.target=self.polygon(goal);env.close()
    def polygon(self,body):return unary_union([Polygon([body.local_to_world(v) for v in vs]) for vs in self.vertices])
    def __call__(self,pose):
        self.body.angle=float(pose[4]);self.body.position=pose[2:4].tolist()
        return self.polygon(self.body).intersection(self.target).area/self.target.area


def metrics(x,reference,tol,first,end):
    trace=x['physics'];ref=reference['physics'];d=trace[:,2:5]-ref[:,2:5]
    pos=np.linalg.norm(d[:,:2],axis=1);angle=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2])))
    loss=np.maximum(0,reference['coverage']-x['coverage']);dense=np.maximum(pos/tol['position'],angle/tol['rotation']);cv=loss/tol['coverage_loss']
    E=float(max(dense.max(),cv.max()));cross=np.flatnonzero(dense>1);crosscov=np.flatnonzero(cv>1)
    times=[]
    if len(cross):times.append((cross[0]+1)*.01)
    if len(crosscov):times.append((crosscov[0]+1)*.1)
    crossing=min(times) if times else None
    j=energy(np.vstack([x['initial_state'][:2],trace[:,:2]]));j0=energy(np.vstack([reference['initial_state'][:2],ref[:,:2]]));gain=(j0-j)/j0 if j0>1e-8 else 0.
    result={'object_error':E,'preserved':E<=1,'position_max':float(pos.max()),'rotation_max_deg':float(np.rad2deg(angle.max())),
        'coverage_max_loss':float(loss.max()),'jerk_gain':gain,'first_violation_seconds':crossing,
        'seconds_after_first_correction':crossing-first*.1 if crossing is not None else None,
        'any_success':bool((x['coverage']>.95).any()),'final_success':bool(x['coverage'][-1]>.95),
        'end_linear_velocity_difference':float(np.linalg.norm(trace[-1,7:9]-ref[-1,7:9])),
        'end_angular_velocity_difference':float(abs(trace[-1,9]-ref[-1,9]))}
    for label,t in [('first_chunk_end',end),('next_chunk_end',end+8),('second_next_chunk_end',end+16),('episode_end',len(x['actions']))]:
        idx=min(t,len(x['actions']))*10-1
        result[label+'_pose_error']=float(dense[idx])
        result[label+'_linear_velocity_difference']=float(np.linalg.norm(trace[idx,7:9]-ref[idx,7:9]))
        result[label+'_angular_velocity_difference']=float(abs(trace[idx,9]-ref[idx,9]))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    manifest=read(OUT/'manifest.json');cfg=manifest['config'];main=read(resolve('configs/followup_v1.json'));rows=[];errors={};predchecks=0;rawcount=0;available_c=0
    for path,h in manifest['input_sha256'].items():assert sha256(resolve(path))==h,path
    def check(name,x,y,tol=1e-9):
        error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),error);assert error<=tol,(name,error)
    for seed in cfg['development_seeds']:
        reference=arrays(SOURCE/f'{seed}_nominal.npz');covfun=Coverage(seed)
        amethod=cfg['methods'][0];apath=OUT/f'{seed}_{amethod}_a_nominal.json'
        ameta=read(apath);assert ameta['manifest_sha256']==sha256(OUT/'manifest.json');nominal=arrays(apath.with_suffix('.npz'))
        for k in ['initial_state','states','physics','actions','coverage']:check('nominal_full_replay_'+k,nominal[k],reference[k],0.)
        for method in cfg['methods']:
            original=arrays(SOURCE/f'{seed}_{method}.npz');original_meta=read(SOURCE/f'{seed}_{method}.json');first=next(d['control_step'] for d in original_meta['decisions'] if d['active']);end=first+8
            bpath=OUT/f'{seed}_{method}_b_first_fixed.json';bmeta=read(bpath);assert bmeta['manifest_sha256']==sha256(OUT/'manifest.json');b=arrays(bpath.with_suffix('.npz'))
            branches={'a_nominal':nominal,'b_first_fixed':b,'d_repeated':original};cpath=OUT/f'{seed}_{method}_c_first_replan.json'
            if cpath.exists():
                cm=read(cpath);assert cm['manifest_sha256']==sha256(OUT/'manifest.json');c=arrays(cpath.with_suffix('.npz'));branches['c_first_replan']=c;available_c+=1
                calls=cm['calls'];assert [d['control_step'] for d in calls]==list(range(end,cfg['max_actions'],8));assert len(c['policy_predictions'])==len(calls)
                for i,d in enumerate(calls):
                    t=d['control_step'];n=min(8,cfg['max_actions']-t);assert d['observation_times']==[t-1,t] and d['policy_noise_seed']==(main['seed']+1009*seed+t)%(2**31-1)
                    compiled,info=absolute_to_relative(c['policy_predictions'][i,1:1+n],c['states'][t,:2],c['states'][t,5:7]);check('replan_delivers_uncorrected_compilation',compiled,c['actions'][t:t+n]);assert info['clipped_actions']==d['clipped_actions']
                with np.load(SOURCE/f'{seed}_{method}.npz') as old:check('first_postcorrection_policy_prediction',c['policy_predictions'][0],old['policy_action_predictions'][end//8],1e-5)
                predchecks+=1
            elif not a.partial:raise ValueError(('Missing C branch',seed,method))
            check('fixed_future_commands',b['actions'][end:],reference['actions'][end:],0.)
            for arm,x in branches.items():
                assert len(x['actions'])==cfg['max_actions'] and len(x['physics'])==10*cfg['max_actions'] and len(x['states'])==cfg['max_actions']+1
                check('initial',x['initial_state'],reference['initial_state'],0.);check('control_dense_alignment',x['states'][1:],x['physics'][9::10],0.);check('action_bounds',np.maximum(0,np.abs(x['actions'])-1),0.,1e-6)
                check('precorrection_actions',x['actions'][:first],reference['actions'][:first],0.);check('precorrection_physics',x['physics'][:first*10],reference['physics'][:first*10],0.)
                if arm!='a_nominal':
                    check('common_first_corrected_actions',x['actions'][:end],original['actions'][:end],0.);check('common_first_corrected_physics',x['physics'][:end*10],original['physics'][:end*10],0.)
                recomputed=np.array([covfun(state) for state in x['physics'][9::10]])
                check('independent_polygon_coverage',recomputed,x['coverage'],1e-10);rawcount+=1
                row={'seed':seed,'initial_pose_family':seed,'method':method,'arm':arm,'first_correction_step':first,**metrics(x,reference,main['tolerances'],first,end)}
                row['joint']=arm!='a_nominal' and row['preserved'] and row['jerk_gain']>=.1;rows.append(row)
    f=pd.DataFrame(rows);summary=[];paired=[]
    for (method,arm),g in f.groupby(['method','arm']):
        summary.append({'method':method,'arm':arm,'families':len(g),**{k:mean_ci(g,k) for k in ['preserved','joint','jerk_gain','object_error','position_max','rotation_max_deg','any_success','end_linear_velocity_difference','end_angular_velocity_difference']}})
    for method in cfg['methods']:
        for later,earlier in [('b_first_fixed','a_nominal'),('c_first_replan','b_first_fixed'),('d_repeated','c_first_replan')]:
            x=f[(f.method==method)&(f.arm==later)].merge(f[(f.method==method)&(f.arm==earlier)],on=['seed','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            if not len(x):continue
            for key in ['preserved','jerk_gain','object_error']:x[key+'_difference']=x[key].astype(float)-x[key+'_base'].astype(float)
            paired.append({'method':method,'later':later,'earlier':earlier,'families':len(x),**{k:mean_ci(x,k) for k in ['preserved_difference','jerk_gain_difference','object_error_difference']}})
    tag='partial_' if a.partial else '';f.to_csv(OUT/(tag+'evaluation.csv'),index=False)
    write_json(OUT/(tag+'analysis.json'),{'partial':a.partial,'new_replan_branches':available_c,'summary':summary,'paired':paired,'scope':cfg['scope']+' Exploratory four-family diagnosis. A-B/B-C/C-D contrasts are conditional interventions with nonlinear interactions, not additive independent contributions. Same relative future commands in B, not identical pusher trajectory. No independent population conclusion from four seeds.'})
    write_json(OUT/(tag+'raw_audit.json'),{'partial':a.partial,'manifest_sha256':sha256(OUT/'manifest.json'),'arm_rows':len(f),'raw_trace_checks_including_reused':rawcount,'first_replan_prediction_checks':predchecks,'max_error':errors,'checks':'Complete nominal replay exact, unchanged prefix/corrected firstchunk, Bfixedfuturecommands, Cownobservations/noise/recompileduncorrectedcommands, independentposepolygoncoverage, densecontrolalignment/fullglobalmetrics.'})
    print('AUDITED',len(f),'arm rows','replan branches',available_c,flush=True)
    for r in summary:print(r['method'],r['arm'],'preserved',r['preserved']['mean'],'E',r['object_error']['mean'],'gain',r['jerk_gain']['mean'],flush=True)
if __name__=='__main__':main()
