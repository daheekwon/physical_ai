"""Recompute runtime choices, native scores, dense outcomes and budget eligibility."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from scripts.run_followup_cube import Model
from scripts.dinowm_runtime_context import load_context
from scripts.lewm_runtime_reference import native_scores
from scripts.output_replacement_core import physical
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('dinowm_runtime_warmed_smoke' if smoke else 'dinowm_runtime');protocol=read(out/'protocol.json');cfg=protocol['config'];assert protocol['smoke']==smoke;done=read(out/'execution_complete.json');assert done['protocol_sha256']==sha256(out/'protocol.json') and done['states']==(4 if smoke else 402)
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    torch.set_num_threads(cfg['torch_threads']);maincfg=read(resolve('configs/followup_v1.json'));ccfg=read(resolve('configs/followup_cube.json'));data=ExpertDataset(maincfg['dataset']);parent=pd.read_csv(BASE/'dinowm_selection/candidates.csv');rules={(x['environment'],x['method']):x['rule'] for x in read(BASE/'dinowm_selection/rules.json')['rules']};capture=BASE/('dinowm_interventions_smoke' if smoke else 'dinowm_interventions');rows=[];times=[];errors={};inputs={};candidate_checks=0
    def check(name,a,b,tol=1e-8,rtol=0):
        a=np.asarray(a);b=np.asarray(b);np.testing.assert_allclose(a,b,atol=tol,rtol=rtol,err_msg=name);finite=np.isfinite(a)&np.isfinite(b);e=float(abs(a-b)[finite].max()) if finite.any() else 0.;errors[name]=max(errors.get(name,0.),e)
    for env,splits in protocol['states'].items():
        norm=read(BASE/'cube/manifest.json')['normalization'];mean=data.scaler.mean_ if env=='pusht' else np.array(norm['mean']);scale=data.scaler.scale_ if env=='pusht' else np.array(norm['scale']);lewm=FrozenLeWM(maincfg['checkpoint'],data.scaler,'cpu') if env=='pusht' else Model({**ccfg,'device':'cpu'},mean,scale)
        for split,names in splits.items():
            for name in names:
                src=resolve(name);dest=out/env/split/src.name;record=read(dest);assert record['protocol_sha256']==sha256(out/'protocol.json')
                for p,h in record['inputs'].items():assert sha256(resolve(p))==h,p
                for extension in ['.json','.csv','.npz']:inputs[str(dest.with_suffix(extension).relative_to(resolve('.')))]=sha256(dest.with_suffix(extension))
                meta,c,lookup,raw=load_context(env,src,data,mean,scale);truth=physical(env,raw);timings=pd.read_csv(dest.with_suffix('.csv'));assert len(timings)==len(cfg['methods'])*cfg['repetitions'];times.append(timings);stateparent=parent[(parent.environment==env)&(parent.state_id==meta['state_id'])].set_index('candidate');native=np.load(capture/env/split/src.with_suffix('.npz').name)['predicted'].astype(float);delta=native-native[:1];dinos={'dino_patch_rms':np.sqrt(np.mean(np.sum(delta**2,axis=2),axis=1)),'dino_patch_mean':np.linalg.norm(delta.mean(1),axis=1)}
                with np.load(dest.with_suffix('.npz')) as a:
                    for method,g in timings.groupby('method'):
                        ids=a[method+'_candidate_ids'];actions=a[method+'_actions'];scores=a[method+'_scores'];cost=a[method+'_costs'];norms=a[method+'_norms'];assert ids[0]==0 and len(set(ids))==len(ids);assert all(i in lookup.values() for i in ids);check('action_parent',actions,raw['actions'][ids],cfg['action_reproduction_atol']);check('norm',norms,np.linalg.norm(((actions-actions[:1])/scale).reshape(len(ids),-1),axis=1));candidate_checks+=len(ids)
                        nonzero=np.flatnonzero(norms>1e-6);assert set(ids[nonzero])==set(stateparent.index[stateparent.norm>1e-6]);index={int(k):i for i,k in enumerate(ids)};pi=stateparent.index.to_numpy(int);ri=[index[k] for k in pi];check('cost_parent_relative',cost[ri],stateparent.predicted_cost,1e-4,1e-8);check('dense_outcome_parent',truth['dense_full'][pi],stateparent.dense_full,2e-5);check('actual_jerk_parent',truth['jerk_gain'][pi],stateparent.jerk_gain)
                        if method=='lewm_global':check('native_cpu_LeWM_score',scores,native_scores(env,lewm,c,actions),2e-5)
                        elif method.startswith('dino'):check('native_DINO_score',scores,dinos[method][ids],1e-8)
                        else:check('classical_score',scores[ri],stateparent[method],1e-8)
                        rule=rules[(env,method)];gain=(cost[0]-cost)/max(cost[0],1e-12);eligible=np.flatnonzero((norms>1e-6)&(gain>1e-8)&(scores<=rule['tau'])) if rule['enabled'] else []
                        chosen=min(eligible,key=lambda i:(cost[i],norms[i],int(ids[i]))) if len(eligible) else 0;cid=int(ids[chosen]);assert (g.candidate==cid).all();assert set(g.repetition)==set(range(cfg['repetitions']));assert (g.candidate_count==len(ids)).all();assert g.total_seconds.gt(0).all() and np.isfinite(g.total_seconds).all();check('timing_components',g.total_seconds,g.generation_seconds+g.score_seconds+g.selection_seconds,.02);assert (g.score_seconds+1e-8>=g.encode_seconds+g.prediction_seconds).all()
                        expected_encode=int(method not in ['controller','contact']);expected_predict=(int(np.ceil(len(ids)/64)) if env=='pusht' else len(ids)) if method=='lewm_global' else len(ids) if method.startswith('dino') else 0;assert (g.encode_calls==expected_encode).all() and (g.prediction_calls==expected_predict).all() and (g.jacobian_calls==0).all()
                        active=bool(norms[chosen]>1e-6);error=float(truth['dense_full'][cid]) if active else 0.;jerkgain=float(truth['jerk_gain'][cid]) if active else 0.;rows.append({'environment':env,'split':split,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'phase':str(g.phase.iloc[0]),'method':method,'candidate':cid,'active':active,'object_error':error,'violation':active and error>1,'joint':active and error<=1 and jerkgain>=.1,'jerk_gain':jerkgain,'predicted_gain':float(gain[chosen]),'parent_choice_match':bool(g.parent_choice_match.all())})
                print('AUDIT',env,split,meta['state_id'],flush=True)
        del lewm
    f=pd.DataFrame(rows);timing=pd.concat(times,ignore_index=True);assert len(f)==done['states']*len(cfg['methods']) and len(timing)==done['timings'];summary=[];paired=[]
    for (env,split,method),g in f.groupby(['environment','split','method']):
        t=timing[(timing.environment==env)&(timing.split==split)&(timing.method==method)];active=g[g.active];contact=active[active.phase=='contact'] if env=='pusht' else active[active.phase!='approach'];summary.append({'environment':env,'split':split,'method':method,'states':len(g),**{k:mean_ci(g,k) for k in ['active','joint','jerk_gain','predicted_gain','parent_choice_match']},'risk':mean_ci(active,'violation'),'contact_risk':mean_ci(contact,'violation'),'p50_seconds':float(t.total_seconds.median()),'p95_seconds':float(t.total_seconds.quantile(.95)),'component_medians':{k:float(t[k].median()) for k in ['generation_seconds','encode_seconds','prediction_seconds','selection_seconds']}})
    for (env,split),g in f.groupby(['environment','split']):
        for method in ['lewm_global','dino_patch_rms','dino_patch_mean','contact']:
            a=g[g.method==method].set_index('state_id');b=g[g.method=='controller'].set_index('state_id').loc[a.index];v=a.copy()
            for metric in ['joint','jerk_gain']:
                v['difference']=a[metric].astype(float)-b[metric].astype(float);paired.append({'environment':env,'split':split,'method':method,'baseline':'controller','metric':metric,**mean_ci(v,'difference')})
    budgets=[]
    for env in f.environment.unique():
        for method in cfg['methods']:
            cal=next(r for r in summary if r['environment']==env and r['split']=='calibration' and r['method']==method);dev=next(r for r in summary if r['environment']==env and r['split']=='development' and r['method']==method)
            for cap in cfg['budgets_seconds']:
                eligible=cal['risk']['mean'] is not None and cal['risk']['mean']<=cfg['target_risk'] and cal['active']['mean']>=cfg['minimum_active_fraction'] and cal['p95_seconds']<=cap
                if env=='pusht':eligible=eligible and cal['contact_risk']['mean'] is not None and cal['contact_risk']['mean']<=cfg['target_risk']
                t=timing[(timing.environment==env)&(timing.split=='development')&(timing.method==method)];budgets.append({'environment':env,'method':method,'cap_seconds':cap,'enabled':bool(eligible),'development_U':dev['joint'] if eligible else {'mean':0.,'ci95':[0.,0.]},'development_risk':dev['risk'] if eligible else {'mean':None,'ci95':[None,None]},'observed_timing_exceedance':float((t.total_seconds>cap).mean()) if eligible else 0.,'interpretation':'Calibration p95 eligibility; no hard deadline or delay injection. Disabled means nominal fallback.'})
    f.to_csv(out/'evaluation.csv',index=False);timing.to_csv(out/'timings.csv',index=False);write_json(out/'analysis.json',{'smoke':smoke,'states':done['states'],'outcomes':len(f),'timings':len(timing),'summary':summary,'paired':paired,'budget_eligibility':budgets,'scope':cfg['scope']});write_json(out/'raw_audit.json',{'smoke':smoke,'states':done['states'],'outcomes':len(f),'timings':len(timing),'candidate_checks':candidate_checks,'max_errors':errors,'inputs':inputs,'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'source_sha256':sha256(Path(__file__)),'scope':'Fresh actions and native CPU LeWM reference scores, native DINO distances from independently audited stored tokens, classical scores/costs from audited parent, all choices/call counts/timing components, dense physical outcomes recomputed and compared with independently audited parent; family statistics and calibration-only budget eligibility. No original GPU LeWM score identity assumed.'});print('AUDITED RUNTIME',done['states'],len(f),len(timing),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_dinowm_runtime.json'))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(args.smoke)
