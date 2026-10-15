"""Calibrate fixed native verifiers on fresh-policy families after physical bank audit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.output_replacement_core import physical,fit,selections
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def frame(path):
    meta=read(path);a=dict(np.load(path.with_suffix('.npz')));n=len(a['actions']);truth=physical('cube',a);cost=a['costs'].astype(np.float64);sid=path.stem
    return pd.DataFrame({'environment':'cube','split':meta['split'],'state_id':sid,'initial_pose_family':meta['seed'],'phase':meta['phase'],'candidate':np.arange(n),'norm':a['norm'],'predicted_cost':cost,'predicted_gain':(cost[0]-cost)/max(cost[0],1e-12),'dense_full':truth['dense_full'],'jerk_gain':truth['jerk_gain'],**{m:a[m+'_scores'].astype(np.float64) for m in ['controller_displacement','proprio_contact','lewm_global']}})

def old_rules():
    old=read(BASE/'cube/proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules']
    return {m:{'enabled':not r.get('disabled',False),'tau':r['tau'] if not r.get('disabled',False) else None} for m,r in old.items()}

def main(smoke):
    bank=BASE/('cube_policy_calibration_bank_smoke' if smoke else 'cube_policy_calibration_bank');audit=read(bank/'raw_audit.json');assert audit['source_sha256']==sha256(resolve('scripts/audit_followup_cube_policy_calibration_bank.py'))
    cfg=read(resolve('configs/followup_cube_policy_selection.json'));out=BASE/('cube_policy_selection_smoke' if smoke else 'cube_policy_selection');out.mkdir(exist_ok=True);paths=sorted(bank.glob('*_t*.json'));assert len(paths)==audit['states'];files=[resolve(p) for p in ['scripts/analyze_followup_cube_policy_selection.py','scripts/output_replacement_core.py','scripts/analyze_followup_response_selection.py','configs/followup_cube_policy_selection.json','scripts/audit_followup_cube_policy_selection.py']]+[bank/'manifest.json',bank/'raw_audit.json',BASE/'cube/proposal_ablation/analysis.json'];files+=[p.with_suffix(ext) for p in paths for ext in ['.json','.npz','.csv']];protocol={'smoke':smoke,'config':cfg,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'sources':[str(p.relative_to(resolve('.'))) for p in paths],'scope':cfg['scope'],'smoke_scope':'Smoke transfers original expert thresholds without fitting; no calibration efficacy claim.'};write_json(out/'protocol.json',protocol);f=pd.concat([frame(p) for p in paths],ignore_index=True);f.to_csv(out/'candidates.csv',index=False);rules=[];curves=[];chosen=[];old=old_rules()
    for method in cfg['methods']:
        if smoke:rule=old[method];cc=[]
        else:rule,cc=fit(f[f.split=='calibration'],method,cfg,'cube')
        rules.append({'method':method,'rule':rule});curves.extend({'method':method,**r} for r in cc)
        for split,g in f.groupby('split',sort=True):
            s=selections(g,method,rule,cfg);s['method']=method;s['split']=split;chosen.append(s)
    chosen=pd.concat(chosen,ignore_index=True);chosen.to_csv(out/'selected.csv',index=False);pd.DataFrame(curves).to_csv(out/'calibration_curves.csv',index=False);write_json(out/'rules.json',{'smoke':smoke,'rules':rules});summary=[];paired=[]
    for (split,method),g in chosen.groupby(['split','method']):
        for phase in ['all']+sorted(g.phase.unique().tolist()):
            x=g if phase=='all' else g[g.phase==phase];summary.append({'split':split,'method':method,'phase':phase,'states':len(x),'families':x.initial_pose_family.nunique(),**{k:mean_ci(x,k) for k in ['active','joint','jerk_gain','predicted_gain','candidate_coverage']},'selected_risk':mean_ci(x[x.active],'violation')})
    for split,g in chosen.groupby('split'):
        for method,baseline in [('lewm_global','controller_displacement'),('proprio_contact','controller_displacement'),('lewm_global','proprio_contact')]:
            x=g[g.method==method].merge(g[g.method==baseline],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for metric in ['joint','active','jerk_gain']:
                x['difference']=x[metric].astype(float)-x[metric+'_base'].astype(float);paired.append({'split':split,'method':method,'baseline':baseline,'metric':metric,**mean_ci(x,'difference')})
    write_json(out/'analysis.json',{'smoke':smoke,'states':len(paths),'candidate_rows':len(f),'selections':len(chosen),'summary':summary,'paired':paired,'scope':protocol['scope'],'missing_phases':read(bank/'execution_complete.json')['missing_phases'],'denominator':'Eligible nominal phase roots; absent phases are reported, not imputed as candidate failures. Family bootstrap retains all sampled eligible roots. Calibration resubstitution is not independent test.'});write_json(out/'execution_complete.json',{'smoke':smoke,'states':len(paths),'protocol_sha256':sha256(out/'protocol.json'),'outputs':{p:sha256(out/p) for p in ['analysis.json','rules.json','selected.csv','candidates.csv','calibration_curves.csv']}});print('POLICY SELECTION',len(paths),len(chosen),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
