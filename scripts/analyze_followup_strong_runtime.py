"""Raw audit and shared p95 budget comparison for uncached strong refinement."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import controller_rollout
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args();base=resolve('artifacts/followup_v1');out=base/'strong_runtime';manifest=json.loads((out/'manifest.json').read_text());tcfg=manifest['config'];cfg=json.loads(resolve('configs/followup_v1.json').read_text());eps=cfg['tolerances'];data=ExpertDataset(cfg['dataset']);frames=[];times=[];errors={};counts={};oldselected=pd.read_csv(base/'strong_baselines/selected.csv');prior_matches=0
    for name,h in manifest['signature'].items():assert sha256(resolve(name))==h,name
    def check(name,x,y,tol=1e-7):
        e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0),e);assert e<=tol,(name,e)
    def jerk(trace):return float(np.sum((np.diff(trace[:,:2],n=3,axis=0)/.01**3)**2)*.01)
    for split in ['calibration','development']:
        paths=sorted((out/split).glob('*h05.json'));counts[split]=len(paths)
        if not args.partial:assert len(paths)==105
        for path in paths:
            meta=json.loads(path.read_text());assert meta['signature']==manifest['signature'];f=pd.read_csv(path.with_suffix('.csv'));assert len(f)==11;frames.append(f)
            with np.load(path.with_suffix('.npz')) as a,np.load((base/'strong_baselines'/path.name).with_suffix('.npz')) as old:
                ref=a['nominal_physics'];j0=jerk(ref)
                for row in f.itertuples():
                    key=row.variant;trace=a[key+'_physics'];action=a[key+'_actions'];nominal=a['nominal_actions'];check('fixed_prefix',trace[:30],ref[:30],1e-9);check('fixed_suffix',action[5:],nominal[5:]);assert np.max(np.abs(action))<=1+1e-6
                    norm=np.linalg.norm((action[:5]-nominal[:5])/data.scaler.scale_);check('action_norm',norm,row.actual_norm);assert bool(row.active)==(norm>1e-6 and row.predicted_gain>1e-8)
                    if row.method not in ['gaussian_action','lipo_full_command','groove_full_command']:check('polynomial_boundary',action[[0,4]],nominal[[0,4]])
                    d=trace[30:,2:5]-ref[30:,2:5];pos=np.linalg.norm(d[:,:2],axis=1).max();rot=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2]))).max();loss=np.maximum(0,a['nominal_coverage']-a[key+'_coverage']).max();E=max(pos/eps['position'],rot/eps['rotation'],loss/eps['coverage_loss']);check('object_E',E,row.object_error);check('jerk_gain',(j0-jerk(trace))/j0 if j0>1e-8 else 0.,row.jerk_gain)
                    q=controller_rollout(action,a['root'][:2],a['root'][5:7]);cost=jerk(np.concatenate([ref[:30,:2],q]));check('predicted_cost',cost,meta['decisions'][key]['predicted_selected_jerk'],2e-4)
                    reps=meta['timings'][key];assert len(reps)==tcfg['repetitions'];check('mean_time',np.mean([x['seconds'] for x in reps]),row.mean_seconds)
                    for rep in reps:assert rep['seconds']>0;times.append({'state_id':row.state_id,'initial_pose_family':row.initial_pose_family,'split':split,'variant':key,'method':row.method,'budget':row.budget,**rep})
                    # Same locked algorithm, now recomputing encoder/Jacobian instead of using cached artifacts.
                    if row.method in oldselected.method.unique() and (row.method!='nonlinear_wm' or row.budget==64):
                        previous=oldselected[(oldselected.state_id==row.state_id)&(oldselected.method==row.method)].iloc[0];check('prior_selected_action',action[:5],old['actions'][int(previous['index'])],2e-5);prior_matches+=1
    f=pd.concat(frames,ignore_index=True);f['joint']=f.active&(f.object_error<=1)&(f.jerk_gain>=.1);f['violation']=f.object_error>1;t=pd.DataFrame(times);summary=[];curves=[];budget_rows=[]
    for (split,variant),g in f.groupby(['split','variant']):
        lat=t[(t.split==split)&(t.variant==variant)];active=g[g.active];contact=active[active.nominal_contact];row={'split':split,'variant':variant,'method':g.method.iloc[0],'budget':int(g.budget.iloc[0]),'joint':mean_ci(g,'joint'),'active':mean_ci(g,'active'),'risk':mean_ci(active,'violation'),'contact_risk':mean_ci(contact,'violation'),'gain':mean_ci(g,'jerk_gain'),'mean_seconds':mean_ci(g,'mean_seconds'),'p50_seconds':float(lat.seconds.median()),'p95_seconds':float(lat.seconds.quantile(.95)),'additional_peak_gpu_bytes':int(lat.additional_peak_gpu_bytes.max())};summary.append(row)
        if split=='calibration':curves.append(row)
    if not args.partial:
        lock=[]
        for method in tcfg['methods']:
            for cap in tcfg['latency_budgets_seconds']:
                feasible=[r for r in curves if r['method']==method and r['p95_seconds']<=cap and r['active']['mean']>=cfg['minimum_active_coverage'] and r['risk']['mean'] is not None and r['contact_risk']['mean'] is not None and max(r['risk']['mean'],r['contact_risk']['mean'])<=cfg['target_violation_rate']]
                best=max(feasible,key=lambda r:(r['joint']['mean'],-r['p95_seconds'])) if feasible else None;rule={'method':method,'cap_seconds':cap,'enabled':best is not None,'variant':best['variant'] if best else None};lock.append(rule)
                if best:
                    g=f[(f.split=='development')&(f.variant==best['variant'])];lat=t[(t.split=='development')&(t.variant==best['variant'])].copy();lat['deadline_exceeded']=lat.seconds>cap;budget_rows.append({**rule,'joint':mean_ci(g,'joint'),'risk':mean_ci(g[g.active],'violation'),'deadline_exceedance':mean_ci(lat,'deadline_exceeded')})
                else:budget_rows.append(rule)
        target=out/'calibrated_budgets.json';locked={'signature':manifest['signature'],'rules':lock,'scope':'Calibration-selected finite profiles under common p95 caps and empirical risk/activity constraints; no hard runtime cap/actuation delay injection.'}
        if target.exists():assert json.loads(target.read_text())==locked
        else:write_json(target,locked)
    f.to_csv(out/'evaluation.csv',index=False);t.to_csv(out/'timings.csv',index=False);write_json(out/'raw_audit.json',{'partial':args.partial,'states':counts,'rows':len(f),'timing_repetitions':len(t),'prior_selected_action_checks':prior_matches,'max_recalculation_error':errors});write_json(out/'analysis.json',{'partial':args.partial,'summary':summary,'budget_comparison':budget_rows,'scope':tcfg['scope']})
    print('AUDIT',counts,len(f),len(t),'prior action checks',prior_matches,flush=True)
    for r in summary:print(r['split'],r['variant'],'U',r['joint']['mean'],'R',r['risk']['mean'],'p95ms',1000*r['p95_seconds'])
if __name__=='__main__':main()
