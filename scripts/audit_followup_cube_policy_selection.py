"""Independent rule grid, sorted selections and family statistics for policy verifier."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_cube_policy_selection import frame,old_rules
from scripts.audit_followup_output_replacement import selected
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('cube_policy_selection_smoke' if smoke else 'cube_policy_selection');protocol=read(out/'protocol.json');cfg=protocol['config'];done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['protocol_sha256']==sha256(out/'protocol.json');errors={}
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    for p,h in done['outputs'].items():assert sha256(out/p)==h,p
    def check(name,x,y,tol=1e-8):
        x=np.asarray(x,float);y=np.asarray(y,float);assert x.shape==y.shape;np.testing.assert_allclose(x,y,atol=tol,rtol=1e-10,equal_nan=True,err_msg=name);finite=np.isfinite(x)&np.isfinite(y);e=float(abs(x[finite]-y[finite]).max()) if finite.any() else 0.;errors[name]=max(errors.get(name,0.),e)
    f=pd.concat([frame(resolve(p)) for p in protocol['sources']],ignore_index=True);stored=pd.read_csv(out/'candidates.csv');assert len(f)==len(stored)
    for col in f:
        if col in ['environment','split','state_id','phase']:np.testing.assert_array_equal(f[col],stored[col])
        else:check('candidate_export_'+col,f[col],stored[col],1e-4 if col=='predicted_cost' else 1e-8)
    rules={r['method']:r['rule'] for r in read(out/'rules.json')['rules']};old=old_rules();rows=[];trials=0;curves=pd.read_csv(out/'calibration_curves.csv') if not smoke else None
    for method in cfg['methods']:
        if smoke:assert rules[method]==old[method]
        else:
            cal=f[f.split=='calibration'];assert cal.initial_pose_family.nunique()==24 and 24<=cal.state_id.nunique()<=96 and set(cal.initial_pose_family)==set(range(340100,340124));values=cal.loc[(cal.norm>cfg['minimum_norm'])&np.isfinite(cal[method]),method];grid=np.unique(np.r_[0.,1.,np.quantile(values,np.linspace(0,1,cfg['threshold_quantiles']))]) if len(values) else np.array([0.,1.]);recorded=curves[curves.method==method].reset_index(drop=True);check('threshold_grid',grid,recorded.tau);best=None;valuebest=None
            for i,tau in enumerate(grid):
                x=selected(cal,method,{'enabled':True,'tau':float(tau)},cfg);active=x[x.active];contact=active[active.phase=='contact'];risk=float(active.violation.mean()) if len(active) else None;crisk=float(contact.violation.mean()) if len(contact) else None;af=float(x.active.mean());value=(float((x.active&(x.predicted_gain>=cfg['meaningful_gain'])).mean()),float(x.predicted_gain.mean()));valid=af>=cfg['minimum_active_fraction'] and risk is not None and risk<=cfg['target_risk']
                if cfg['enforce_pusht_contact_risk']:valid=valid and crisk is not None and crisk<=cfg['target_risk']
                r=recorded.iloc[i]
                for name,v in [('active_fraction',af),('risk',risk),('contact_risk',crisk),('predicted_useful',value[0]),('mean_predicted_gain',value[1])]:check('calibration_'+name,np.nan if v is None else v,r[name])
                assert bool(r.valid)==valid
                if valid and (valuebest is None or value>valuebest):best={'enabled':True,'tau':float(tau)};valuebest=value
                trials+=1
            assert rules[method]['enabled']==bool(best)
            if best:check('chosen_tau',best['tau'],rules[method]['tau'])
        for split,g in f.groupby('split',sort=True):
            x=selected(g,method,rules[method],cfg);x['method']=method;x['split']=split;rows.append(x)
    x=pd.concat(rows,ignore_index=True);keys=['split','method','state_id'];x=x.sort_values(keys).reset_index(drop=True);stored=pd.read_csv(out/'selected.csv').sort_values(keys).reset_index(drop=True);assert len(x)==len(stored)
    for col in x:
        if col in keys+['phase']:np.testing.assert_array_equal(x[col],stored[col])
        else:check('selection_'+col,x[col],stored[col])
    result=read(out/'analysis.json')
    for r in result['summary']:
        g=x[(x.split==r['split'])&(x.method==r['method'])];g=g if r['phase']=='all' else g[g.phase==r['phase']];assert len(g)==r['states'] and g.initial_pose_family.nunique()==r['families']
        for metric in ['active','joint','jerk_gain','predicted_gain','candidate_coverage','selected_risk']:
            h=g[g.active] if metric=='selected_risk' else g;col='violation' if metric=='selected_risk' else metric
            if not len(h):assert r[metric]['mean'] is None;continue
            avg,ci=cluster(h[col].to_numpy(),h.initial_pose_family.to_numpy());check('summary_mean',avg,r[metric]['mean']);check('summary_ci',ci,r[metric]['ci95'])
    for r in result['paired']:
        g=x[x.split==r['split']];a=g[g.method==r['method']].set_index('state_id');b=g[g.method==r['baseline']].set_index('state_id').loc[a.index];d=a[r['metric']].astype(float)-b[r['metric']].astype(float);avg,ci=cluster(d.to_numpy(),a.initial_pose_family.to_numpy());check('paired_mean',avg,r['mean']);check('paired_ci',ci,r['ci95'])
    assert result['states']==f.state_id.nunique() and len(x)==3*result['states'];assert set(f[f.split=='development'].initial_pose_family)==({340000} if smoke else set(range(340000,340012)));write_json(out/'raw_audit.json',{'smoke':smoke,'states':result['states'],'candidate_rows':len(f),'selections':len(x),'calibration_trials':trials,'max_errors':errors,'source_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'helpers':{p:sha256(resolve(p)) for p in ['scripts/analyze_followup_cube_policy_selection.py','scripts/audit_followup_output_replacement.py','scripts/audit_retrained_dinowm_readouts.py']},'scope':'Audited physical/native-model bank reused with hashgates. All raw-precision exports, threshold grids, calibrationconstraints, sorted selection, familybootstrap andpaired differences verified independently. Smoke uses old expert thresholds only; full uses24 new Cube calibration families and12 previously inspected development families. No independent final or repeated-correction claim.'});print('AUDITED POLICY SELECTION',result['states'],len(x),trials,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
