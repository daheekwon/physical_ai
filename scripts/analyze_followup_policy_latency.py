"""Audit actual-clock waiting, stored-command dispatch, and dual global references."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from zj_wam.policy_refinement import absolute_to_relative
from scripts.policy_latency_core import draw_delay,holding_action,delay_ticks
from scripts.analyze_followup_policy_final import Coverage,energy
from scripts.analyze_followup_response_selection import mean_ci

BASE=resolve('artifacts/followup_v1')


def read(p):return json.loads(p.read_text())


def arrays(p):
    with np.load(p) as a:return {k:a[k].copy() for k in a.files}


def compare(trace,cov,initial,ref,refcov,tol):
    d=trace[:,2:5]-ref[:,2:5];pos=np.linalg.norm(d[:,:2],axis=1).max();angle=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2]))).max();loss=np.maximum(0,refcov-cov).max();E=max(pos/tol['position'],angle/tol['rotation'],loss/tol['coverage_loss']);j0=energy(np.vstack([initial[:2],ref[:,:2]]));j=energy(np.vstack([initial[:2],trace[:,:2]]));gain=(j0-j)/j0 if j0>1e-8 else 0.
    return {'object_error':float(E),'preserved':E<=1,'jerk_gain':gain,'position_max':float(pos),'rotation_max_deg':float(np.rad2deg(angle)),'coverage_loss':float(loss)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--methods',nargs='+');args=p.parse_args();out=BASE/('policy_latency_smoke' if args.smoke else 'policy_latency');manifest=read(out/'manifest.json');cfg=manifest['config'];maincfg=read(resolve('configs/followup_v1.json'));tol=maincfg['tolerances'];methods=args.methods or cfg['methods'];seeds=cfg['development_seeds'][:1] if args.smoke else cfg['development_seeds'];errors={};rows=[];locals_all=[];calls_all=[];local_count=0
    if not args.smoke:assert methods==cfg['methods'],'Full development must include allmethods'
    for path,h in manifest['input_sha256'].items():assert sha256(resolve(path))==h,path
    def check(name,x,y,atol=1e-8):
        e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),e);assert e<=atol,(name,e)
    for seed in seeds:
        zero=arrays(out/f'{seed}_nominal_zero.npz');covfun=Coverage(seed)
        for method in methods:
            for profile in cfg['profiles']:
                path=out/f'{seed}_{method}_{profile}.json';meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json');a=arrays(path.with_suffix('.npz'));ref=arrays(out/f'{seed}_nominal_{profile}.npz');n=meta['steps'];assert n==(120 if args.smoke else cfg['max_actions']) and len(a['physics'])==10*n and len(a['states'])==n+1
                check('same_initial',a['initial_state'],zero['initial_state'],0.);check('dense_control',a['states'][1:],a['physics'][9::10],0.);check('raw_bounds',np.maximum(0,np.abs(a['actions'])-1),0.,1e-6)
                check('independent_coverage',[covfun(x) for x in a['physics'][9::10]],a['coverage'],1e-10)
                cursor=0;active=False;eligible_complete=0
                for k,d in enumerate(meta['calls']):
                    t=d['request_step'];assert t==cursor and d['call_index']==k and d['observation_times']==[max(0,t-1),t] and d['policy_noise_seed']==(maincfg['seed']+1009*seed+8*k)%(2**31-1)
                    expected=draw_delay(manifest['samples'],seed,k,method,cfg['methods'],profile,d['eligible'],maincfg['seed']);assert d['delay']==expected;ready=t+delay_ticks(expected['total']);assert d['ready_step']==ready
                    width=min(8,n-t);eligible=t>=10 and width==8;assert eligible==d['eligible'] and d['wm_observation_times']==([t-10,t-5,t] if eligible else [])
                    compiled,info=absolute_to_relative(a['policy_predictions'][k,1:1+width],a['states'][t,:2],a['states'][t,5:7]);check('request_compilation',compiled,a['nominal_chunks'][k,:width],1e-9)
                    target=np.asarray(d['held_target_at_request']);count=d['delivered_count'];dispatch=min(ready,n);assert count==min(width,n-dispatch) and d['dispatch_step']==(dispatch if count else None)
                    for u in range(t,dispatch):
                        assert a['roles'][u]==0 and a['call_ids'][u]==k and a['offsets'][u]==-1
                        check('absolute_target_hold',a['actions'][u],holding_action(target,a['states'][u,:2]),1e-12);check('hold_target_constant',a['hold_targets'][u],target,0.)
                    if count:
                        actual=a['actions'][dispatch:dispatch+count];check('stored_stale_commands_dispatched',actual,a['planned_chunks'][k,:count],0.)
                        check('chunk_target',a['hold_targets'][dispatch:dispatch+count],a['states'][dispatch:dispatch+count,:2]+100*actual,1e-10)
                        assert (a['roles'][dispatch:dispatch+count]==1).all() and (a['call_ids'][dispatch:dispatch+count]==k).all();np.testing.assert_array_equal(a['offsets'][dispatch:dispatch+count],np.arange(count))
                        applied=np.linalg.norm(actual-a['nominal_chunks'][k,:count])>1e-6;active|=applied
                    if eligible and count==8:eligible_complete+=1
                    cursor=dispatch+count;calls_all.append({'seed':seed,'initial_pose_family':seed,'method':method,'profile':profile,'call_index':k,'request_step':t,'ready_step':ready,'applied_actions':count,'sampled_total_seconds':expected['total'],'injected_ticks':expected['ticks'],'wall_plan_seconds':d['wall_plan_seconds']})
                assert cursor==n and eligible_complete==meta['eligible_complete_chunks'];local=pd.read_csv(path.with_suffix('.csv'));assert len(local)==eligible_complete
                for i,r in enumerate(local.itertuples()):
                    dispatch=r.dispatch_step;actual=a['physics'][dispatch*10:(dispatch+8)*10];truth=a['local_nominal_physics'][i];d=actual[:,2:5]-truth[:,2:5];pos=np.linalg.norm(d[:,:2],axis=1).max();rot=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2]))).max();ct=np.array([covfun(x) for x in truth[[49,79]]]);check('local_coverage',ct,a['local_nominal_coverage'][i,[4,7]],1e-10);loss=np.maximum(0,ct-a['coverage'][[dispatch+4,dispatch+7]]).max();E=max(pos/tol['position'],rot/tol['rotation'],loss/tol['coverage_loss']);check('local_object_E',E,r.object_error)
                    prefix=a['physics'][dispatch*10-30:dispatch*10,:2];j0=energy(np.r_[prefix,truth[:,:2]]);gain=(j0-energy(np.r_[prefix,actual[:,:2]]))/j0 if j0>1e-8 else 0.;check('local_gain',gain,r.jerk_gain);local_count+=1
                local['seed']=seed;local['initial_pose_family']=seed;local['method']=method;local['profile']=profile;locals_all.append(local)
                if profile=='zero':
                    assert meta['hold_steps']==0
                    with np.load(BASE/'policy_refinement/development'/f'{seed}_{method}.npz') as old:
                        check('zero_reproduces_original_actions',a['actions'],old['actions'][:n],1e-8);check('zero_reproduces_original_physics',a['physics'],old['physics'][:n*10],1e-7)
                for reference,other in [('same_profile',ref),('zero_nominal',zero)]:
                    metric=compare(a['physics'],a['coverage'],a['initial_state'],other['physics'],other['coverage'],tol);rows.append({'seed':seed,'initial_pose_family':seed,'method':method,'profile':profile,'reference':reference,'active':active,**metric,'joint':bool(active and metric['preserved'] and metric['jerk_gain']>=.1),'any_success':bool((a['coverage']>.95).any()),'final_success':bool(a['coverage'][-1]>.95),'hold_fraction':meta['hold_steps']/n,'policy_calls':len(meta['calls'])})
        only=arrays(out/f'{seed}_nominal_policy_only.npz');full=arrays(out/f'{seed}_nominal_full.npz')
        check('nominal_two_delay_profiles_exact',only['physics'],full['physics'],0.);check('nominal_delay_actions_exact',only['actions'],full['actions'],0.)
    f=pd.DataFrame(rows);summary=[]
    for (method,profile,reference),g in f.groupby(['method','profile','reference']):
        active=g[g.active].copy();active['violation']=~active.preserved;summary.append({'method':method,'profile':profile,'reference':reference,'episodes':len(g),**{k:mean_ci(g,k) for k in ['joint','preserved','jerk_gain','any_success','final_success','hold_fraction','policy_calls']},'active_risk':mean_ci(active,'violation')})
    f.to_csv(out/'evaluation.csv',index=False);pd.concat(locals_all,ignore_index=True).to_csv(out/'local_evaluation.csv',index=False);pd.DataFrame(calls_all).to_csv(out/'calls.csv',index=False)
    write_json(out/'analysis.json',{'smoke':args.smoke,'summary':summary,'scope':cfg['scope']+' Timing replay profiles frozen before execution. Smoke120steps is implementation check only, not study performance. Global sameprofile and original zero references both retained.'})
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'episodes':len(seeds)*len(methods)*3,'local_blocks':local_count,'max_error':errors,'source_sha256':sha256(Path(__file__)),'manifest_sha256':sha256(out/'manifest.json'),'checks':'Physicaltime includesholds; storedrequestcommands deliveredlatewithoutreoptimization; causalhistory/noise; unchangedprefixreferenceinitial; bound/coverage/localcounterfactual; zeroexactparentreproduction; nominal policyonly/fullprofilesexact.'})
    print('AUDITED',len(seeds)*len(methods)*3,'episodes',local_count,'localblocks',flush=True)
if __name__=='__main__':main()
