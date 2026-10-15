"""Five-verifier Cube policy comparison with independent calibration and selection checks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_cube_policy_selection import frame
from scripts.output_replacement_core import fit,selections
from scripts.audit_followup_output_replacement import selected
from scripts.analyze_followup_response_selection import mean_ci
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
HELPERS=['scripts/analyze_followup_cube_policy_selection.py','scripts/output_replacement_core.py','scripts/audit_followup_output_replacement.py','scripts/analyze_followup_response_selection.py','scripts/audit_retrained_dinowm_readouts.py','configs/followup_cube_policy_selection.json','configs/followup_cube_dino_policy.json']

def main(smoke):
    src=BASE/('cube_dino_policy_scores_smoke' if smoke else 'cube_dino_policy_scores');out=BASE/('cube_dino_policy_selection_smoke' if smoke else 'cube_dino_policy_selection');out.mkdir(exist_ok=True);m=read(src/'manifest.json');audit=read(src/'raw_audit.json');assert audit['smoke']==smoke and audit['manifest_sha256']==sha256(src/'manifest.json') and audit['source_sha256']==sha256(resolve('scripts/run_followup_cube_dino_policy_scores.py'))
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    cfg=read(resolve('configs/followup_cube_policy_selection.json'));extra=read(resolve('configs/followup_cube_dino_policy.json'));cfg['methods']=cfg['methods']+extra['metrics'];cfg['scope']=extra['scope'];frames=[];files=[src/'manifest.json',src/'raw_audit.json',BASE/'cube_policy_selection/rules.json',BASE/'cube_policy_selection/raw_audit.json',Path(__file__)]+[resolve(p) for p in HELPERS]
    for name in m['sources']:
        p=resolve(name);x=frame(p);tokens=src/p.with_suffix('.npz').name;meta=src/p.name;assert read(meta)['data_sha256']==sha256(tokens)
        with np.load(tokens,allow_pickle=False) as z:
            for key in extra['metrics']:x[key]=z[key].astype(float)
        frames.append(x);files.extend([tokens,meta])
    f=pd.concat(frames,ignore_index=True);assert f.state_id.nunique()==(8 if smoke else 138)
    protocol={'smoke':smoke,'config':cfg,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope'],'smoke_note':'Eight-state plumbing check. Original3 rules retained; DINO smoke calibration on4 technicalstates has no performance interpretation and is never reused by full analysis.'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    old={r['method']:r['rule'] for r in read(BASE/'cube_policy_selection/rules.json')['rules']};rules=[];curves=[];chosen=[];errors={};trials=0
    def check(key,x,y):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);np.testing.assert_allclose(x,y,atol=1e-8,rtol=1e-10,equal_nan=True,err_msg=key);valid=np.isfinite(x)&np.isfinite(y);errors[key]=max(errors.get(key,0.),float(abs(x[valid]-y[valid]).max()) if valid.any() else 0.)
    cal=f[f.split=='calibration']
    assert cal.initial_pose_family.nunique()==(1 if smoke else 24)
    for method in cfg['methods']:
        if method in old:rule=old[method]
        else:
            rule,curve=fit(cal,method,cfg,'cube');curves.extend({'method':method,**r} for r in curve)
            # Independent grid and sorted-selection implementation: fit is not its own oracle.
            vals=cal.loc[(cal.norm>cfg['minimum_norm'])&np.isfinite(cal[method]),method].to_numpy();grid=np.unique(np.r_[0.,1.,np.quantile(vals,np.linspace(0,1,cfg['threshold_quantiles']))]) if len(vals) else np.array([0.,1.]);check('grid',grid,[r['tau'] for r in curve]);best=None;bestvalue=None
            for tau,row in zip(grid,curve):
                v=selected(cal,method,{'enabled':True,'tau':float(tau)},cfg);act=v[v.active];risk=float(act.violation.mean()) if len(act) else None;af=float(v.active.mean());value=(float((v.active&(v.predicted_gain>=cfg['meaningful_gain'])).mean()),float(v.predicted_gain.mean()));valid=risk is not None and risk<=cfg['target_risk'] and af>=cfg['minimum_active_fraction'];assert row['valid']==valid
                for k,val in [('risk',risk),('active_fraction',af),('predicted_useful',value[0]),('mean_predicted_gain',value[1])]:check('trial_'+k,np.nan if val is None else val,np.nan if row[k] is None else row[k])
                if valid and (bestvalue is None or value>bestvalue):best={'enabled':True,'tau':float(tau)};bestvalue=value
                trials+=1
            independent=best or {'enabled':False,'tau':None};assert independent['enabled']==rule['enabled']
            if independent['enabled']:check('chosen_tau',independent['tau'],rule['tau'])
        rules.append({'method':method,'rule':rule})
        for split,g in f.groupby('split',sort=True):
            v=selections(g,method,rule,cfg);ind=selected(g,method,rule,cfg)
            for col in v:
                if col in ['state_id','phase']:np.testing.assert_array_equal(v[col],ind[col])
                else:check('choice_'+col,v[col],ind[col])
            v['split']=split;v['method']=method;chosen.append(v)
    chosen=pd.concat(chosen,ignore_index=True);oldchoices=pd.read_csv(BASE/'cube_policy_selection/selected.csv')
    for method in old:
        a=chosen[chosen.method==method].set_index('state_id');b=oldchoices[oldchoices.method==method].set_index('state_id').loc[a.index];np.testing.assert_array_equal(a.candidate,b.candidate)
    summary=[];paired=[]
    def interval(g,col):
        r=mean_ci(g,col)
        if len(g):avg,ci=cluster(g[col].to_numpy(),g.initial_pose_family.to_numpy());check('family_mean',avg,r['mean']);check('family_ci',ci,r['ci95'])
        return r
    for (split,method),g in chosen.groupby(['split','method']):
        for phase in ['all']+sorted(g.phase.unique()):
            v=g if phase=='all' else g[g.phase==phase];summary.append({'split':split,'method':method,'phase':phase,'states':len(v),'families':v.initial_pose_family.nunique(),**{k:interval(v,k) for k in ['active','joint','jerk_gain','predicted_gain','candidate_coverage']},'selected_risk':interval(v[v.active],'violation')})
    for split,g in chosen.groupby('split'):
        for method in extra['metrics']:
            for baseline in old:
                v=g[g.method==method].merge(g[g.method==baseline],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
                for metric in ['joint','active','jerk_gain']:v['difference']=v[metric].astype(float)-v[metric+'_base'].astype(float);paired.append({'split':split,'method':method,'baseline':baseline,'metric':metric,**interval(v,'difference')})
    f.to_csv(out/'candidates.csv',index=False);chosen.to_csv(out/'selected.csv',index=False);pd.DataFrame(curves).to_csv(out/'calibration_curves.csv',index=False);write_json(out/'rules.json',{'smoke':smoke,'rules':rules});write_json(out/'analysis.json',{'smoke':smoke,'states':f.state_id.nunique(),'candidates':len(f),'selections':len(chosen),'summary':summary,'paired':paired,'scope':cfg['scope'],'missing_nominal_phases':read(BASE/'cube_policy_calibration_bank/execution_complete.json')['missing_phases']})
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'raw_audit.json',{'smoke':smoke,'states':f.state_id.nunique(),'candidates':len(f),'selections':len(chosen),'new_DINO_calibration_trials':trials,'max_errors':errors,'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'selected_sha256':sha256(out/'selected.csv'),'source_sha256':sha256(Path(__file__)),'helpers':{p:sha256(resolve(p)) for p in HELPERS},'scope':'Parent physical/native score audit reused with hash gates; all new DINO grid trials and choices independently computed using sorted selection, all family statistics independently checked, original3 choices exactly unchanged. Calibration/development only; no independent final.'});print('CUBE DINO SELECTION AUDITED',len(f),len(chosen),trials,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
