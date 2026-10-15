"""Audit Cube runtime and select computation profiles on calibration only."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci

BASE=resolve('artifacts/followup_v1/cube');read=lambda p:json.loads(p.read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();out=BASE/('runtime_smoke' if args.smoke else 'runtime');m=read(out/'manifest.json');cfg=m['cube_config'];rcfg=m['config'];assert m['smoke']==args.smoke
    for path,h in m['inputs'].items():assert sha256(resolve(path))==h,path
    complete=read(out/'execution_complete.json');assert complete['manifest_sha256']==sha256(out/'manifest.json')
    scale=np.asarray(m['normalization']['scale']);frames=[];errors={};states=0;reproduced=0
    def check(name,a,b,tol=1e-8):
        e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));errors[name]=max(errors.get(name,0.),e);assert e<=tol,(name,e)
    for split in ['calibration','development']:
        files=sorted((out/split).glob('e*.json'));assert len(files)==(1 if args.smoke else 96)
        for path in files:
            record=read(path);assert record['manifest_sha256']==sha256(out/'manifest.json')
            for parent,h in record['parent_sha256'].items():assert sha256(resolve(parent))==h,parent
            raw=pd.read_csv(BASE/split/path.with_suffix('.csv').name);timings=pd.read_csv(path.with_suffix('.csv'));assert len(timings)==len(rcfg['profiles'])*len(rcfg['methods'])*rcfg['repetitions'];states+=1
            assert timings.seconds.gt(0).all() and np.isfinite(timings.seconds).all()
            with np.load(path.with_suffix('.npz')) as a,np.load(BASE/split/path.with_suffix('.npz').name) as parent:
                for (profile,method),g in timings.groupby(['profile','method']):
                    key=profile+'__'+method;ids=a[key+'_candidate_ids'];actions=a[key+'_actions'];score=a[key+'_scores'];cost=a[key+'_costs'];norm=a[key+'_norm'];spec=rcfg['profiles'][profile]
                    assert set(g.repetition)==set(range(rcfg['repetitions'])) and g.candidate.nunique()==1
                    check('actions_exact_physical_parent',actions,parent['actions'][ids],1e-9);check('normalized_norm',np.linalg.norm(((actions-parent['nominal_actions'])/scale).reshape(len(ids),-1),axis=1),norm)
                    check('fixed_gripper',actions[:,:,4],np.broadcast_to(parent['nominal_actions'][:,4],actions[:,:,4].shape),0.)
                    assert (g.encoder_calls==(spec['bank']=='common' or method=='lewm_global')).all();assert (g.jacobian_calls==(spec['bank']=='common')).all();assert (g.finite_prediction_calls==(len(ids) if method=='lewm_global' else 0)).all()
                    rules=read(BASE/'calibrated_settings.json')['rules'] if spec['bank']=='common' else read(BASE/'proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules'];r=rules[method]
                    allowed=(norm>1e-6)&(norm<=r.get('radius',.4)+1e-6)&(score<=r.get('tau',-np.inf))&(cost<cost[0]-1e-10)
                    if r.get('disabled',False):allowed[:]=False
                    choices=np.flatnonzero(allowed);selected=int(ids[choices[np.lexsort((ids[choices],cost[choices]))[0]]]) if len(choices) else 0;assert (g.candidate==selected).all()
                    source=raw.iloc[selected]
                    check('selected_object_error',g.object_error,source.object_error);check('selected_jerk_gain',g.jerk_gain,source.jerk_gain);check('selected_norm',g.norm,source.norm)
                    assert (g.active==(source.norm>1e-6)).all();assert (g.joint==bool(source.norm>1e-6 and source.object_error<=1 and source.jerk_gain>=cfg['jerk_gain_threshold'])).all()
                    if profile in ['nonwm_full','wm_full']:
                        old=pd.read_csv(BASE/('proposal_ablation/evaluation.csv' if profile=='nonwm_full' else 'evaluation.csv'));old=old[(old.split==split)&(old.state_id==record['state_id'])&(old.method==method)]
                        if profile=='nonwm_full':old=old[old.bank=='non_wm_proposals']
                        assert len(old)==1 and int(old.iloc[0].candidate)==selected;reproduced+=1
            frames.append(timings)
    timing=pd.concat(frames,ignore_index=True);f=timing[timing.repetition==0].copy();f['violation']=f.object_error>1
    assert len(f)==states*len(rcfg['profiles'])*len(rcfg['methods']);summary=[]
    for (split,profile,method),g in f.groupby(['split','profile','method']):
        t=timing[(timing.split==split)&(timing.profile==profile)&(timing.method==method)];active=g[g.active]
        summary.append({'split':split,'profile':profile,'method':method,'states':len(g), **{key:mean_ci(g,key) for key in ['joint','jerk_gain','active']},'risk':mean_ci(active,'violation'),'contact_risk':mean_ci(active[active.phase!='approach'],'violation'),'p50_seconds':float(t.seconds.median()),'p95_seconds':float(t.seconds.quantile(.95))})
    settings=[]
    for method in rcfg['methods']:
        for cap in rcfg['budgets_seconds']:
            options=[x for x in summary if x['split']=='calibration' and x['method']==method and x['p95_seconds']<=cap and x['risk']['mean'] is not None and x['contact_risk']['mean'] is not None and max(x['risk']['mean'],x['contact_risk']['mean'])<=cfg['target_risk'] and x['active']['mean']>=cfg['minimum_active_fraction']]
            best=max(options,key=lambda x:(x['joint']['mean'],x['jerk_gain']['mean'],-x['p95_seconds'])) if options else None
            settings.append({'method':method,'cap_seconds':cap,'enabled':best is not None,'profile':best['profile'] if best else None,'calibration':best})
    write_json(out/'calibrated_budgets.json',{'smoke':args.smoke,'settings':settings,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Calibration-only p95, all/contact risk and minimum active fraction. Smoke approach-only has no contact evidence and cannot qualify deployment. No per-state outcome tuning. No guarantee of a hard deadline.'})
    budget_rows=[]
    for setting in settings:
        g=f[(f.split=='development')&(f.method==setting['method'])&(f.profile==(setting['profile'] or 'nonwm_full'))].copy();g['cap_seconds']=setting['cap_seconds'];g['enabled']=setting['enabled']
        if not setting['enabled']:
            g['active']=False;g['joint']=False;g['violation']=False;g['object_error']=0.;g['jerk_gain']=0.;g['norm']=0.;g['candidate']=0;g['seconds']=0.;g['profile']='disabled';g['deadline_exceedance']=0.
        else:
            t=timing[(timing.split=='development')&(timing.method==setting['method'])&(timing.profile==setting['profile'])].copy();t['exceeded']=t.seconds>setting['cap_seconds'];ex=t.groupby('state_id').exceeded.mean();g['deadline_exceedance']=g.state_id.map(ex)
        budget_rows.append(g)
    b=pd.concat(budget_rows,ignore_index=True);budget=[];paired=[]
    for (method,cap),g in b.groupby(['method','cap_seconds']):
        active=g[g.active];budget.append({'method':method,'cap_seconds':cap,'enabled':bool(g.enabled.iloc[0]),'profile':g.profile.iloc[0],**{k:mean_ci(g,k) for k in ['joint','jerk_gain','active','deadline_exceedance']},'risk':mean_ci(active,'violation')})
    for cap,g in b.groupby('cap_seconds'):
        baseline=g[g.method=='controller_displacement']
        for method in ['proprio_contact','lewm_global']:
            x=g[g.method==method].merge(baseline,on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one');x['delta_U']=x.joint.astype(float)-x.joint_base.astype(float);x['delta_gain']=x.jerk_gain-x.jerk_gain_base
            paired.append({'method':method,'baseline':'controller_displacement','cap_seconds':cap,'U_difference':mean_ci(x,'delta_U'),'jerk_gain_difference':mean_ci(x,'delta_gain')})
    timing.to_csv(out/'timings.csv',index=False);f.to_csv(out/'evaluation.csv',index=False);b.to_csv(out/'budget_evaluation.csv',index=False)
    write_json(out/'analysis.json',{'smoke':args.smoke,'summary':summary,'budget_comparison':budget,'paired':paired,'scope':rcfg['scope']+' '+rcfg['calibration']})
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'states':states,'outcomes':len(f),'timings':len(timing),'full_parent_choice_reproductions':reproduced,'max_error':errors,'manifest_sha256':sha256(out/'manifest.json'),'scope':'All fresh timing repetitions select same source actions; physical outcomes linked to unchanged original dense-bank audit. Actual future outcomes never used to choose action. No new physics or latency-injected execution claimed.'})
    print('AUDITED Cube runtime',states,'states',len(f),'outcomes',len(timing),'timings',flush=True)

if __name__=='__main__':main()
