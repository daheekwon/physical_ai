"""Compare causal robot-response predictors on the existing fixed Cube bank."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from zj_wam.io import resolve,sha256,write_json
from scripts.cube_causal_arx_core import fit,predict
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_causal_arx';SRC=BASE/'cube_policy_oracle';read=lambda p:json.loads(p.read_text())

def main():
    cfg=read(resolve('configs/followup_cube_causal_arx.json'));a=read(SRC/'raw_audit.json');done=read(SRC/'pipeline_complete.json');assert a['states']==48 and done['raw_audit_sha256']==sha256(SRC/'raw_audit.json');paths=sorted(SRC.glob('*_t*.json'));assert len(paths)==48;OUT.mkdir(exist_ok=True);rules=read(BASE/'cube/proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules'];files=[resolve(p) for p in ['configs/followup_cube_causal_arx.json','scripts/cube_causal_arx_core.py','scripts/run_followup_cube_causal_arx.py','zj_wam/cube_experiment.py','scripts/analyze_followup_response_selection.py']]+[SRC/'manifest.json',SRC/'raw_audit.json',SRC/'pipeline_complete.json',BASE/'cube/proposal_ablation/analysis.json'];files.extend(p.with_suffix(ext) for p in paths for ext in ['.json','.npz','.csv']);lock={'config':cfg,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'created_unix':time.time(),'scope':cfg['scope']}
    assert not (OUT/'protocol.json').exists();write_json(OUT/'protocol.json',lock);rows=[];fidelity=[]
    for p in paths:
        meta=read(p);raw=dict(np.load(p.with_suffix('.npz')));truth=pd.read_csv(p.with_suffix('.csv'));points=raw['input_past_points'];targets=raw['input_past_targets'];prefix=points[-75:,:3];arrays={};models={};actualcost=truth.eef_jerk.to_numpy();actualgain=truth.jerk_gain.to_numpy();E=truth.object_error.to_numpy();n=len(actualcost)
        for variant in cfg['variants']:
            model=fit(points,targets,variant,cfg);pred=np.array([predict(np.r_[x,raw['input_suffix_actions']],points[-1],points[-1]-points[-2],model,raw['input_workspace_bounds'],cfg) for x in raw['actions']]);assert np.isfinite(pred).all();q=np.concatenate([np.broadcast_to(prefix,(n,*prefix.shape)),pred[:,:,:3]],axis=1);cost=(np.diff(q,n=3,axis=1)**2).sum((1,2))/.002**5;gain=(cost[0]-cost)/max(cost[0],1e-12)
            if variant=='past_pd':np.testing.assert_allclose(pred,raw['predicted_robot'],atol=1e-10,rtol=0);np.testing.assert_allclose(cost,raw['costs'],atol=1e-2,rtol=1e-8)
            arrays[variant+'_predicted']=pred;arrays[variant+'_cost']=cost;arrays[variant+'_gain']=gain
            for key in ['A','B','c','gains','residual']:arrays[variant+'_'+key]=model[key]
            models[variant]={k:model[k] for k in ['records','supported_axes']};active=raw['norm']>1e-6;corr=spearmanr(gain[active],actualgain[active]).statistic if np.ptp(gain[active])>0 and np.ptp(actualgain[active])>0 else np.nan;actual_delta=raw['candidate_eef_position']-raw['candidate_eef_position'][:1];pred_delta=pred[:,:,:3]-pred[:1,:,:3];error=np.linalg.norm(pred_delta-actual_delta,axis=2)
            fidelity.append({'state_id':p.stem,'initial_pose_family':meta['seed'],'phase':meta['phase'],'variant':variant,'supported_axes':model['supported_axes'],'gain_rank_spearman':corr,'gain_mean_absolute_error':float(abs(gain[active]-actualgain[active]).mean()),'candidate_effect_position_RMSE_m':float(np.sqrt(np.mean(error[active]**2))),'nominal_position_RMSE_m':float(np.sqrt(np.mean(np.sum((pred[0,:,:3]-raw['candidate_eef_position'][0])**2,axis=1)))),'nominal_relative_jerk_error':float(abs(cost[0]-actualcost[0])/max(actualcost[0],1e-12))})
            for verifier in ['controller_displacement','proprio_contact','lewm_global','true_preservation']:
                if verifier=='true_preservation':allowed=active&(E<=1)
                else:
                    rule=rules[verifier];allowed=active&(raw['norm']<=rule.get('radius',.4)+1e-6)&(raw[verifier+'_scores']<=rule.get('tau',-np.inf))
                    if rule.get('disabled',False):allowed[:]=False
                eligible=np.flatnonzero(allowed&(cost<cost[0]-1e-10));chosen=min(eligible,key=lambda i:(cost[i],i)) if len(eligible) else 0
                if variant=='past_pd' and verifier!='true_preservation':assert chosen==meta['selected'][verifier]
                rows.append({'state_id':p.stem,'initial_pose_family':meta['seed'],'phase':meta['phase'],'variant':variant,'verifier':verifier,'candidate':chosen,'active':bool(chosen),'object_error':E[chosen],'violation':bool(E[chosen]>1),'jerk_gain':actualgain[chosen],'predicted_gain':gain[chosen],'joint':bool(chosen and E[chosen]<=1 and actualgain[chosen]>=.1)})
        dest=OUT/p.with_suffix('.npz').name;np.savez_compressed(dest,**arrays);write_json(dest.with_suffix('.json'),{'source':str(p.relative_to(resolve('.'))),'models':models,'data_sha256':sha256(dest),'protocol_sha256':sha256(OUT/'protocol.json')});print('ARX',p.stem,{v:m['supported_axes'] for v,m in models.items()},flush=True)
    f=pd.DataFrame(rows);g=pd.DataFrame(fidelity);f.to_csv(OUT/'selected.csv',index=False);g.to_csv(OUT/'fidelity.csv',index=False);summary=[];fs=[];paired=[]
    for (variant,verifier),x in f.groupby(['variant','verifier']):
        for phase in ['all']+sorted(x.phase.unique()):
            h=x if phase=='all' else x[x.phase==phase];summary.append({'variant':variant,'verifier':verifier,'phase':phase,'states':len(h),**{k:mean_ci(h,k) for k in ['active','joint','jerk_gain','predicted_gain']},'selected_risk':mean_ci(h[h.active],'violation')})
    for variant,x in g.groupby('variant'):
        fs.append({'variant':variant,**{k:mean_ci(x.dropna(subset=[k]),k) for k in ['supported_axes','gain_rank_spearman','gain_mean_absolute_error','candidate_effect_position_RMSE_m','nominal_position_RMSE_m','nominal_relative_jerk_error']}})
    for (variant,verifier),x in f[f.variant!='past_pd'].groupby(['variant','verifier']):
        h=x.merge(f[(f.variant=='past_pd')&(f.verifier==verifier)],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
        for metric in ['joint','jerk_gain','active']:h['difference']=h[metric].astype(float)-h[metric+'_base'].astype(float);paired.append({'variant':variant,'verifier':verifier,'metric':metric,**mean_ci(h,'difference')})
    write_json(OUT/'analysis.json',{'states':48,'families':12,'candidates':899,'selections':len(f),'summary':summary,'fidelity':fs,'paired':paired,'scope':cfg['scope']});write_json(OUT/'execution_complete.json',{'states':48,'selections':len(f),'analysis_sha256':sha256(OUT/'analysis.json'),'protocol_sha256':sha256(OUT/'protocol.json'),'scope':'Diagnostic analysis; independent fit/rollout/choice/statistical audit required.'})
if __name__=='__main__':main()
