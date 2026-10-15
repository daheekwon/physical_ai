"""Raw outcome/time audit and calibration-selected common latency caps."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json
from scripts.analyze_followup_response_selection import mean_ci


def load(folder,cfg,tcfg,signature,partial=False):
    paths=sorted(folder.glob('*_f*_e*_t*.json'))
    if not partial:assert len(paths)==105,(folder,len(paths))
    frames=[];timings=[];stages=[];errors={}
    def check(name,x,y,tol=1e-7):
        e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),e);assert e<=tol,(name,e)
    for path in paths:
        meta=json.loads(path.read_text());assert meta['signature']==signature;f=pd.read_csv(path.with_suffix('.csv'))
        with np.load(path.with_suffix('.npz')) as a:
            reference=a['nominal_physics'];eps=cfg['tolerances'];j0=np.sum((np.diff(reference[:,:2],n=3,axis=0)/.01**3)**2)*.01
            for row in f.itertuples():
                actual=a[row.variant+'_physics'];action=a[row.variant+'_actions'];check('fixed_prefix',actual[:30],reference[:30],1e-9);check('fixed_suffix',action[5:],a['nominal_actions'][5:]);check('raw_action_bounds',np.maximum(0,np.abs(action)-1),0.,1e-6)
                if row.method!='gaussian_action':check('polynomial_endpoints',action[[0,4]],a['nominal_actions'][[0,4]],1e-7)
                delta=actual[30:,2:5]-reference[30:,2:5];pos=np.linalg.norm(delta[:,:2],axis=1).max();rot=np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2]))).max();loss=np.maximum(0,a['nominal_coverage']-a[row.variant+'_coverage']).max();E=max(pos/eps['position'],rot/eps['rotation'],loss/eps['coverage_loss']);check('E',E,row.object_error)
                j=np.sum((np.diff(actual[:,:2],n=3,axis=0)/.01**3)**2)*.01;gain=(j0-j)/j0 if j0>1e-8 else 0.;check('jerk_gain',gain,row.jerk_gain)
                reps=meta['repetitions'][row.variant];assert len(reps)==tcfg['repetitions'];check('mean_timing',np.mean([r['total_seconds'] for r in reps]),row.mean_seconds);check('median_timing',np.median([r['total_seconds'] for r in reps]),row.median_seconds)
                for r in reps:assert r['total_seconds']>=r['render_seconds']>=0;assert r['candidate_count']==row.candidate_count;timings.append({'state_id':row.state_id,'initial_pose_family':row.initial_pose_family,'split':row.split,'variant':row.variant,'method':row.method,'profile':row.profile,**r})
                p=meta['stage_profiles'][row.variant];assert p['unattributed_seconds']>-1e-6,p;stages.append({'state_id':row.state_id,'initial_pose_family':row.initial_pose_family,'split':row.split,'variant':row.variant,'method':row.method,'profile':row.profile,**{k:v for k,v in p.items() if k!='call_counts'},**p['call_counts']})
        frames.append(f)
    f=pd.concat(frames,ignore_index=True);f['joint']=f.active&(f.object_error<=1)&(f.jerk_gain>=.1);f['violation']=f.object_error>1
    return f,pd.DataFrame(timings),pd.DataFrame(stages),{'states':len(paths),'variant_rows':len(f),'timing_repetitions':len(timings),'max_recalculation_error':errors,'scope':'DenseE/jerk,nominalprefix/suffix,polynomialboundariesexceptGaussian,actionbounds,repetitioncount andaggregatewalltimes. Timed/profileoutputdeterminism checked live inrunner; instrumented stageprofiles separatefrommain uninstrumented latency.'}


def lock(f,t,cfg,tcfg,signature):
    curves=[];settings=[]
    for (method,profile),g in f.groupby(['method','profile']):
        active=g[g.active];contact=active[active.nominal_contact];lat=t[(t.method==method)&(t.profile==profile)].total_seconds;curves.append({'method':method,'profile':profile,'p95_seconds':float(lat.quantile(tcfg['latency_quantile'])),'joint':float(g.joint.mean()),'active':float(g.active.mean()),'risk':float(active.violation.mean()) if len(active) else None,'contact_risk':float(contact.violation.mean()) if len(contact) else None})
    for method in tcfg['methods']:
        for budget in tcfg['latency_budgets_seconds']:
            feasible=[r for r in curves if r['method']==method and r['p95_seconds']<=budget and r['risk'] is not None and r['contact_risk'] is not None and max(r['risk'],r['contact_risk'])<=cfg['target_violation_rate'] and r['active']>=cfg['minimum_active_coverage']]
            best=max(feasible,key=lambda r:(r['joint'],r['active'],-r['p95_seconds'])) if feasible else None;settings.append({'method':method,'budget_seconds':budget,'enabled':best is not None,'selection':best})
    return {'signature':signature,'rules':settings,'curves':curves,'scope':'Choose one candidate-count profile per method/common p95walltimecap using calibration only. Same locked verifier thresholds. No newrisk guarantee under profile/subset change; developmentrisk remeasured. If no profile meets time/risk/activity caps, refinement disabled at thatbudget. This is measuredcompute comparison, not latencyinjected robotexecution.'}


def summarize(f,t,stages,settings):
    summaries=[];budget_rows=[]
    for (split,method,profile),g in f.groupby(['split','method','profile']):
        active=g[g.active];lat=t[(t.split==split)&(t.method==method)&(t.profile==profile)].total_seconds
        summaries.append({'split':split,'method':method,'profile':profile,'states':len(g),**{k:mean_ci(g,k) for k in ['joint','active','jerk_gain','mean_seconds','candidate_count']},'risk':mean_ci(active,'violation'),'latency_p50_seconds':float(lat.median()),'latency_p95_seconds':float(lat.quantile(.95)),'latency_max_seconds':float(lat.max()),'additional_peak_gpu_bytes':int(t[(t.split==split)&(t.method==method)&(t.profile==profile)].additional_peak_gpu_bytes.max())})
    for split in f.split.unique():
        for rule in settings['rules']:
            row={'split':split,'method':rule['method'],'budget_seconds':rule['budget_seconds'],'enabled':rule['enabled']}
            if rule['enabled']:
                profile=rule['selection']['profile'];g=f[(f.split==split)&(f.method==rule['method'])&(f.profile==profile)];lat=t[(t.split==split)&(t.method==rule['method'])&(t.profile==profile)].copy();lat['deadline_exceeded']=lat.total_seconds>rule['budget_seconds'];row.update({'profile':profile,'joint':mean_ci(g,'joint'),'active':mean_ci(g,'active'),'risk':mean_ci(g[g.active],'violation'),'mean_seconds':mean_ci(g,'mean_seconds'),'deadline_exceedance':mean_ci(lat,'deadline_exceeded')})
            else:row.update({'profile':None,'reason':'No calibration-qualified profile; nominal fallback, no measured refinement latency.'})
            budget_rows.append(row)
    stage_summary=stages.fillna(0).groupby(['split','method','profile']).mean(numeric_only=True).reset_index().to_dict('records')
    return {'variants':summaries,'budget_comparison':budget_rows,'synchronized_stage_profile_means':stage_summary,'note':'Main walltime repetitions are uninstrumented. Stage means contain synchronization overhead and are diagnostic; do not substitute them for measured total. Sharedhost/device contention recorded inrunenvironment. No added actuationdelay simulated.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibrate','evaluate'],required=True);p.add_argument('--partial',action='store_true');args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());tcfg=json.loads(resolve('configs/followup_runtime.json').read_text());out=resolve(cfg['output'])/'runtime';manifest=json.loads((out/'manifest.json').read_text());sig=manifest['signature']
    if args.phase=='calibrate':
        f,t,s,a=load(out/'calibration',cfg,tcfg,sig,args.partial);write_json(out/'calibration/raw_audit.json',a);f.to_csv(out/'calibration/evaluation.csv',index=False);t.to_csv(out/'calibration/timings.csv',index=False);s.to_csv(out/'calibration/stages.csv',index=False)
        if not args.partial:settings=lock(f,t,cfg,tcfg,sig);write_json(out/'calibrated_settings.json',settings)
        print('CALIBRATION AUDIT',a,flush=True)
    else:
        settings=json.loads((out/'calibrated_settings.json').read_text());assert settings['signature']==sig;fs=[];ts=[];ss=[];audit={}
        for split in ['development','calibration']:
            f,t,s,a=load(out/split,cfg,tcfg,sig,args.partial);fs.append(f);ts.append(t);ss.append(s);audit[split]=a
        f=pd.concat(fs);t=pd.concat(ts);s=pd.concat(ss);analysis=summarize(f,t,s,settings);analysis['locked']=settings;analysis['scope']=tcfg['scope'];write_json(out/'analysis.json',analysis);write_json(out/'raw_audit.json',audit);f.to_csv(out/'evaluation.csv',index=False);t.to_csv(out/'timings.csv',index=False);s.to_csv(out/'stages.csv',index=False)
        for r in analysis['variants']:
            if r['split']=='development':print(r['method'],r['profile'],'U',r['joint']['mean'],'R',r['risk']['mean'],'p95ms',1000*r['latency_p95_seconds'],flush=True)
if __name__=='__main__':main()
