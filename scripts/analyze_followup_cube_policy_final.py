"""Locked Cube final contrasts, with independent choices and family statistics."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_cube_policy_selection import frame
from scripts.output_replacement_core import selections
from scripts.audit_followup_output_replacement import selected
from scripts.analyze_followup_response_selection import mean_ci
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main(smoke):
    bank=BASE/('cube_policy_final_bank_smoke' if smoke else 'cube_policy_final_bank');out=BASE/('cube_policy_final_analysis_smoke' if smoke else 'cube_policy_final');out.mkdir(exist_ok=True);ba=read(bank/'raw_audit.json');manifest=read(bank/'manifest.json');protocol=read(BASE/'cube_policy_final/final_protocol.json');cfg=protocol['config'];selector=read(resolve('configs/followup_cube_policy_selection.json'));rules={x['method']:x['rule'] for x in protocol['rules']};assert ba['smoke']==smoke and ba['source_sha256']==sha256(resolve('scripts/audit_followup_cube_policy_final_bank.py'))
    for p,h in {**protocol['inputs'],**manifest['inputs'],**ba['inputs'],**ba['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    paths=[bank/f'{seed}_{phase}_t{step:03d}.json' for seed,phases in manifest['phases'].items() for phase,step in phases.items() if step is not None];assert len(paths)==ba['states'];f=pd.concat([frame(p) for p in paths],ignore_index=True);assert f.initial_pose_family.nunique()==(1 if smoke else 64);assert set(f.initial_pose_family)==({cfg['smoke_seed']} if smoke else set(protocol['test_seeds']));rows=[];errors={}
    def check(name,x,y):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);np.testing.assert_allclose(x,y,atol=1e-8,rtol=1e-10,equal_nan=True,err_msg=name);errors[name]=max(errors.get(name,0.),float(abs(x-y).max()) if x.size else 0.)
    for method in cfg['methods']:
        x=selections(f,method,rules[method],selector);ind=selected(f,method,rules[method],selector)
        for col in x:
            if col in ['state_id','phase']:np.testing.assert_array_equal(x[col],ind[col])
            else:check('selection_'+col,x[col],ind[col])
        for row in x.itertuples():meta=read(bank/(row.state_id+'.json'));assert row.candidate==meta['selected'][method]
        x['method']=method;rows.append(x)
    choices=pd.concat(rows,ignore_index=True);summary=[];paired=[]
    def interval(g,col):
        r=mean_ci(g,col)
        if len(g):avg,ci=cluster(g[col].to_numpy(),g.initial_pose_family.to_numpy());check('family_mean',avg,r['mean']);check('family_ci',ci,r['ci95'])
        return r
    for method,g in choices.groupby('method'):
        summary.append({'method':method,'states':len(g),'families':g.initial_pose_family.nunique(),**{k:interval(g,k) for k in ['active','joint','jerk_gain','predicted_gain','candidate_coverage']},'selected_risk':interval(g[g.active],'violation')})
    for method,baseline in [('lewm_global','controller_displacement'),('proprio_contact','controller_displacement'),('lewm_global','proprio_contact')]:
        g=choices[choices.method==method].merge(choices[choices.method==baseline],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
        for metric in ['joint','active','jerk_gain']:g['difference']=g[metric].astype(float)-g[metric+'_base'].astype(float);paired.append({'method':method,'baseline':baseline,'metric':metric,**interval(g,'difference')})
    primary=next(r for r in paired if (r['method'],r['baseline'],r['metric'])==(cfg['primary_method'],cfg['primary_baseline'],cfg['primary_metric']));p=next(r for r in summary if r['method']==cfg['primary_method']);risk=p['selected_risk']['mean'];supported=bool(primary['mean']>=cfg['practical_U_difference'] and primary['ci95'][0]>0 and risk is not None and risk<=cfg['target_risk'] and p['active']['mean']>=cfg['minimum_active_fraction'])
    # Separate direct reconstruction of the prespecified practical gate.
    direct=choices[choices.method==cfg['primary_method']];base=choices[choices.method==cfg['primary_baseline']].set_index('state_id').loc[direct.state_id];delta=direct.joint.to_numpy(float)-base.joint.to_numpy(float);avg,ci=cluster(delta,direct.initial_pose_family.to_numpy());direct_risk=direct[direct.active].violation.mean() if direct.active.any() else float('inf');assert supported==bool(avg>=.1 and ci[0]>0 and direct_risk<=.1 and direct.active.mean()>=.25)
    choices.to_csv(out/'selected.csv',index=False);f.to_csv(out/'candidates.csv',index=False);write_json(out/'analysis.json',{'smoke':smoke,'states':len(paths),'families':f.initial_pose_family.nunique(),'candidates':len(f),'selections':len(choices),'summary':summary,'paired':paired,'supported_practical_primary_advantage':supported,'missing_phases':{seed:[phase for phase,t in phases.items() if t is None] for seed,phases in manifest['phases'].items()},'scope':cfg['scope'],'denominator':'All eligible nominal phase roots, including failed-task families; absent phases reported, no correction-based filtering. State-weighted point estimates and family cluster bootstrap.'});write_json(out/'raw_audit.json',{'smoke':smoke,'states':len(paths),'families':f.initial_pose_family.nunique(),'candidates':len(f),'selections':len(choices),'max_errors':errors,'analysis_sha256':sha256(out/'analysis.json'),'selected_sha256':sha256(out/'selected.csv'),'bank_raw_audit_sha256':sha256(bank/'raw_audit.json'),'final_protocol_sha256':sha256(BASE/'cube_policy_final/final_protocol.json'),'source_sha256':sha256(Path(__file__)),'helpers':{p:sha256(resolve(p)) for p in ['scripts/analyze_followup_cube_policy_selection.py','scripts/output_replacement_core.py','scripts/audit_followup_output_replacement.py','scripts/analyze_followup_response_selection.py','scripts/audit_retrained_dinowm_readouts.py']},'scope':'Each choice crosschecked by independently sorted candidate table and prephysics stored selection; each family interval independently recomputed; direct primary gate independently checked. Rawmetrics from fully audited bank. No test-time fitting.'});print('FINAL CUBE ANALYZED AUDITED',len(paths),len(choices),supported,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
