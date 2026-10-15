"""Prespecified family inference for locked policy-final single corrections."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.output_replacement_core import physical
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('dinowm_policy_final_analysis_smoke' if smoke else 'dinowm_policy_final');out.mkdir(exist_ok=True);bank=BASE/('dinowm_policy_final_bank_smoke' if smoke else 'dinowm_policy_final_bank');audit=read(bank/'raw_audit.json');assert audit['states']==(3 if smoke else 192) and audit['source_sha256']==sha256(resolve('scripts/audit_followup_dinowm_policy_final_bank.py'));lock=read(BASE/'dinowm_policy_final/final_protocol.json');cfg=lock['config'];rows=[];files=[bank/'raw_audit.json',bank/'manifest.json',BASE/'dinowm_policy_final/final_protocol.json']
    for p,h in {**read(bank/'manifest.json')['inputs'],**audit['inputs'],**audit['audit_helpers'],**lock['inputs']}.items():assert sha256(resolve(p))==h,p
    for p in sorted(bank.glob('test/*_t*.json')):
        if p.name.endswith('.decision.json'):continue
        m=read(p);a=dict(np.load(p.with_suffix('.npz')));truth=physical('pusht',a);gain=(a['predicted_costs'][0]-a['predicted_costs'])/max(a['predicted_costs'][0],1e-12)
        for method in cfg['methods']:
            i=m['selected_before_physics'][method];active=bool(a['norms'][i]>1e-6);E=float(truth['dense_full'][i]);J=float(truth['jerk_gain'][i]);rows.append({'state_id':m['state_id'],'initial_pose_family':m['seed'],'phase':m['phase'],'method':method,'candidate':i,'active':active,'object_error':E,'violation':bool(E>1),'jerk_gain':J,'predicted_gain':gain[i],'joint':bool(active and E<=1 and J>=cfg['meaningful_gain'])})
        files.extend([p,p.with_suffix('.npz'),p.with_suffix('.decision.json')])
    f=pd.DataFrame(rows);assert len(f)==audit['states']*5;f.to_csv(out/'selected.csv',index=False);summary=[];paired=[]
    for method,g in f.groupby('method'):
        summary.append({'method':method,'states':len(g),'families':g.initial_pose_family.nunique(),**{k:mean_ci(g,k) for k in ['active','joint','jerk_gain','predicted_gain']},'selected_risk':mean_ci(g[g.active],'violation')})
    for method,baseline in [('dino_patch_rms','lewm_global'),('lewm_global','controller'),('contact','controller'),('dino_patch_mean','lewm_global'),('dino_patch_rms','contact')]:
        g=f[f.method==method].merge(f[f.method==baseline],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
        for metric in ['joint','active','jerk_gain']:
            g['difference']=g[metric].astype(float)-g[metric+'_base'].astype(float);paired.append({'method':method,'baseline':baseline,'metric':metric,**mean_ci(g,'difference')})
    primary=next(x for x in paired if x['method']==cfg['primary_method'] and x['baseline']==cfg['primary_baseline'] and x['metric']==cfg['primary_metric']);main=next(x for x in summary if x['method']==cfg['primary_method']);risk=main['selected_risk']['mean'];supported=primary['mean']>=cfg['practical_U_difference'] and primary['ci95'][0]>0 and risk is not None and risk<=cfg['target_risk'] and main['active']['mean']>=cfg['minimum_active_fraction']
    write_json(out/'analysis.json',{'smoke':smoke,'states':audit['states'],'families':int(f.initial_pose_family.nunique()),'selections':len(f),'summary':summary,'paired':paired,'supported_practical_primary_advantage':bool(supported),'scope':cfg['scope']});files +=[resolve(p) for p in ['scripts/analyze_followup_dinowm_policy_final.py','scripts/output_replacement_core.py','scripts/analyze_followup_response_selection.py']];write_json(out/'analysis_protocol.json',{'smoke':smoke,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'analysis_sha256':sha256(out/'analysis.json'),'selected_sha256':sha256(out/'selected.csv')});print('ANALYZED POLICY FINAL',audit['states'],len(f),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
