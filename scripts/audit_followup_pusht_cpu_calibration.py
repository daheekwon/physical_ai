"""Fresh single-native CPU scores and independent calibration/selection checks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from scripts.run_followup_pusht_cpu_calibration import refreshed
from scripts.lewm_runtime_reference import native_scores
from scripts.audit_followup_output_replacement import selected
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('pusht_cpu_calibration_smoke' if smoke else 'pusht_cpu_calibration');protocol=read(out/'protocol.json');cfg=protocol['config'];done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['protocol_sha256']==sha256(out/'protocol.json');main=read(resolve('configs/followup_v1.json'));torch.set_num_threads(2);data=ExpertDataset(main['dataset']);model=FrozenLeWM(main['checkpoint'],data.scaler,'cpu');errors={};frames=[];forwards=0
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    for p,h in done['outputs'].items():assert sha256(out/p)==h,p
    def check(name,x,y,tol=1e-8):
        x=np.asarray(x,float);y=np.asarray(y,float);assert x.shape==y.shape;np.testing.assert_allclose(x,y,atol=tol,rtol=1e-10,equal_nan=True,err_msg=name);ok=np.isfinite(x)&np.isfinite(y);errors[name]=max(errors.get(name,0),float(abs(x[ok]-y[ok]).max()) if ok.any() else 0.)
    for name in protocol['sources']:
        src=resolve(name);p=out/'scores'/src.name;m=read(p);assert m['protocol_sha256']==sha256(out/'protocol.json') and m['data_sha256']==sha256(p.with_suffix('.npz'));raw=dict(np.load(src.with_suffix('.npz')));a=dict(np.load(p.with_suffix('.npz')));scores=native_scores('pusht',model,{'images':raw['history_images'],'past':raw['past_actions']},raw['actions'][a['candidate_ids']]);check('fresh_nativeCPU_LeWM',scores,a['scores'],2e-5);check('saved_latent_distance',np.linalg.norm(a['latents']-a['latents'][:1],axis=1),a['scores'],0);forwards+=len(scores);frames.append(refreshed(src,out,cfg));print('AUDIT CPU CAL',src.stem,len(scores),flush=True)
    f=pd.concat(frames,ignore_index=True);saved=pd.read_csv(out/'candidates.csv');assert len(f)==len(saved)
    for col in f:
        if col in ['environment','split','state_id','phase','generator']:np.testing.assert_array_equal(f[col],saved[col])
        else:check('export_'+col,f[col],saved[col],1e-4 if col=='predicted_cost' else 1e-8)
    cal=f[f.split=='calibration'];finite=cal.loc[(cal.norm>cfg['minimum_norm'])&np.isfinite(cal.lewm_global),'lewm_global'].to_numpy();grid=np.unique(np.r_[0.,1.,np.quantile(finite,np.linspace(0,1,cfg['threshold_quantiles']))]) if len(finite) else np.array([0.,1.]);curves=pd.read_csv(out/'calibration_curves.csv');check('grid',grid,curves.tau);best=None;bestv=None
    for i,tau in enumerate(grid):
        x=selected(cal,'lewm_global',{'enabled':True,'tau':float(tau)},cfg);active=x[x.active];contact=active[active.phase=='contact'];risk=float(active.violation.mean()) if len(active) else None;crisk=float(contact.violation.mean()) if len(contact) else None;af=float(x.active.mean());value=(float((x.active&(x.predicted_gain>=cfg['meaningful_gain'])).mean()),float(x.predicted_gain.mean()));valid=af>=cfg['minimum_active_fraction'] and risk is not None and risk<=cfg['target_risk'] and crisk is not None and crisk<=cfg['target_risk'];r=curves.iloc[i]
        for k,v in [('active_fraction',af),('risk',risk),('contact_risk',crisk),('predicted_useful',value[0]),('mean_predicted_gain',value[1])]:check('trial_'+k,np.nan if v is None else v,r[k])
        assert bool(r.valid)==valid
        if valid and (bestv is None or value>bestv):best={'enabled':True,'tau':float(tau)};bestv=value
    rule=read(out/'rule.json');assert rule['enabled']==bool(best)
    if best:check('tau',rule['tau'],best['tau'])
    rules=read(out/'rules.json')['rules'];old={r['method']:r['rule'] for r in read(BASE/'dinowm_selection/rules.json')['rules'] if r['environment']=='pusht'};rows=[]
    for method in cfg['methods']:
        if method!='lewm_global':assert rules[method]==old[method]
        else:assert rules[method]==rule
        for split,g in f.groupby('split'):
            x=selected(g,method,rules[method],cfg);x['method']=method;x['split']=split;rows.append(x)
    x=pd.concat(rows,ignore_index=True);keys=['split','method','state_id'];x=x.sort_values(keys).reset_index(drop=True);saved=pd.read_csv(out/'selected.csv').sort_values(keys).reset_index(drop=True);assert len(x)==len(saved)
    for col in x:
        if col in keys+['phase']:np.testing.assert_array_equal(x[col],saved[col])
        else:check('selection_'+col,x[col],saved[col])
    result=read(out/'analysis.json')
    for r in result['summary']:
        g=x[(x.split==r['split'])&(x.method==r['method'])];assert len(g)==r['states']
        for k in ['joint','active','jerk_gain','selected_risk']:
            v=g[g.active] if k=='selected_risk' else g;col='violation' if k=='selected_risk' else k
            if not len(v):assert r[k]['mean'] is None;continue
            avg,ci=cluster(v[col].to_numpy(),v.initial_pose_family.to_numpy());check('summary_mean',avg,r[k]['mean']);check('summary_ci',ci,r[k]['ci95'])
    assert result['states']==(2 if smoke else 210) and len(x)==(10 if smoke else 1050);write_json(out/'raw_audit.json',{'smoke':smoke,'states':result['states'],'candidate_rows':len(f),'nativeCPU_fresh_predictions':forwards,'threshold_trials':len(grid),'selections':len(x),'max_errors':errors,'source_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'helpers':{p:sha256(resolve(p)) for p in ['scripts/lewm_runtime_reference.py','scripts/audit_followup_output_replacement.py','scripts/audit_retrained_dinowm_readouts.py']},'scope':'Every updated CPUscore checked viafresh singlecandidate nativeforward; allthresholds and selections independentlyrecomputed. Physical/norm/controller/contact/DINO scores reusedfromhashverifiedpreviousfullbank audit; no freshphysicalexecution. Full210recalibratesLeWM only, otherparentrulesunchanged.'});print('AUDITED CPU CAL',result['states'],forwards,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
