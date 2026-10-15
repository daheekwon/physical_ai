"""Independent selected-trace metrics, all locked choices, and family bootstrap checks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_policy_final import Coverage
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def interval(frame,column):
    families=sorted(frame.initial_pose_family.unique())
    if not families:return {'mean':None,'ci95':None,'families':0}
    groups=[frame[frame.initial_pose_family==k][column].astype(float).to_numpy() for k in families];sums=np.array([x.sum() for x in groups]);counts=np.array([len(x) for x in groups]);rng=np.random.default_rng(20261011);draw=rng.integers(0,len(groups),(2000,len(groups)));replicates=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
    return {'mean':float(np.concatenate(groups).mean()),'ci95':np.percentile(replicates,[2.5,97.5]).tolist(),'families':len(families)}

def main(smoke):
    out=BASE/('dinowm_policy_final_analysis_smoke' if smoke else 'dinowm_policy_final');bank=BASE/('dinowm_policy_final_bank_smoke' if smoke else 'dinowm_policy_final_bank');protocol=read(out/'analysis_protocol.json');r=read(out/'analysis.json');f=pd.read_csv(out/'selected.csv');lock=read(BASE/'dinowm_policy_final/final_protocol.json');assert protocol['smoke']==r['smoke']==smoke and protocol['analysis_sha256']==sha256(out/'analysis.json') and protocol['selected_sha256']==sha256(out/'selected.csv');errors={};rows=0
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    def check(name,x,y,tol=1e-9):
        np.testing.assert_allclose(x,y,atol=tol,rtol=0,err_msg=name);errors[name]=max(errors.get(name,0),float(np.max(abs(np.asarray(x)-np.asarray(y)))))
    for sid,g in f.groupby('state_id'):
        p=bank/'test'/(sid+'.json');meta=read(p);a=dict(np.load(p.with_suffix('.npz')));decision=read(p.with_suffix('.decision.json'));covfun=Coverage(meta['seed']);base=a['physics'][0];cov0=np.array([covfun(base[k]) for k in [79,109]]);q=base[:,:2];d3=q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3];J0=(d3*d3).sum()/.01**5;assert len(g)==5
        for row in g.itertuples():
            rule=lock['rules'][row.method];gain=(a['predicted_costs'][0]-a['predicted_costs'])/max(a['predicted_costs'][0],1e-12);ok=[i for i in range(len(gain)) if rule['enabled'] and a['norms'][i]>1e-6 and gain[i]>1e-8 and a['score_'+row.method][i]<=rule['tau']];chosen=min(ok,key=lambda i:(a['predicted_costs'][i],a['norms'][i],i)) if ok else 0;assert chosen==row.candidate==decision['selected_before_physics'][row.method]==meta['selected_before_physics'][row.method];x=a['physics'][chosen];delta=x[30:,2:5]-base[30:,2:5];pos=np.sqrt((delta[:,:2]**2).sum(1)).max()/2;rot=np.abs(np.angle(np.exp(1j*delta[:,2]))).max()/np.deg2rad(2);cv=np.array([covfun(x[k]) for k in [79,109]]);E=max(pos,rot,np.maximum(0,cov0-cv).max()/.02);q=x[:,:2];d3=q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3];J=(d3*d3).sum()/.01**5;gainreal=(J0-J)/J0 if J0>1e-8 else 0.;active=bool(a['norms'][chosen]>1e-6);check('object_error',E,row.object_error,1e-8);check('jerk_gain',gainreal,row.jerk_gain,1e-8);check('predicted_gain',gain[chosen],row.predicted_gain);assert active==row.active and (E>1)==row.violation and (active and E<=1 and gainreal>=.1)==row.joint;rows+=1
    def compare(expected,actual):
        assert expected['families']==actual['families']
        if expected['mean'] is None:assert actual['mean'] is None and actual['ci95'] is None
        else:check('summary_mean',expected['mean'],actual['mean']);check('summary_CI',expected['ci95'],actual['ci95'])
    for x in r['summary']:
        g=f[f.method==x['method']];assert len(g)==x['states'] and g.initial_pose_family.nunique()==x['families']
        for metric in ['active','joint','jerk_gain','predicted_gain']:compare(interval(g,metric),x[metric])
        compare(interval(g[g.active],'violation'),x['selected_risk'])
    for x in r['paired']:
        g=f[f.method==x['method']].merge(f[f.method==x['baseline']],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one');g['delta']=g[x['metric']].astype(float)-g[x['metric']+'_base'].astype(float);compare(interval(g,'delta'),x)
    cfg=lock['config'];p=next(x for x in r['paired'] if (x['method'],x['baseline'],x['metric'])==(cfg['primary_method'],cfg['primary_baseline'],cfg['primary_metric']));m=next(x for x in r['summary'] if x['method']==cfg['primary_method']);risk=m['selected_risk']['mean'];supported=p['mean']>=.1 and p['ci95'][0]>0 and risk is not None and risk<=.1 and m['active']['mean']>=.25;assert r['supported_practical_primary_advantage']==supported;assert rows==r['selections']==(15 if smoke else 960) and r['families']==(1 if smoke else 64)
    write_json(out/'raw_audit.json',{'smoke':smoke,'states':r['states'],'families':r['families'],'selections':rows,'max_errors':errors,'analysis_sha256':sha256(out/'analysis.json'),'selected_sha256':sha256(out/'selected.csv'),'source_sha256':sha256(Path(__file__)),'helpers':{'scripts/analyze_followup_policy_final.py':sha256(resolve('scripts/analyze_followup_policy_final.py'))},'scope':'All locked choices from raw scores, selected dense pose/independent coverage/jerk and all family intervals plus primary decision. Fullbank audit separately covers every candidate and fresh native/physical branches.'});print('AUDITED POLICY FINAL',rows,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
