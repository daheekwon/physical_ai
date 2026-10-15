"""Reconstruct matched support and selections without recalibrating any rule."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_cube_proposals import load_candidates


def main():
    base=resolve('artifacts/followup_v1/cube');dest=base/'matched_proposals';a=json.loads((dest/'analysis.json').read_text());cfg=json.loads(resolve('configs/followup_cube.json').read_text());manifest=json.loads((base/'manifest.json').read_text());assert a['parent_raw_audit_sha256']==sha256(base/'raw_audit.json');selected=pd.read_csv(dest/'evaluation.csv');directions=['random0','random1','lewm_low','lewm_high'];checks=0;maxE=0.;maxG=0.
    for split in ['calibration','development']:
        f=load_candidates(base,split,cfg,manifest)
        for state,g in f.groupby('state_id'):
            d=g[g.method.isin(directions)];valid=d.groupby(['radius','sign']).method.nunique();valid=set(valid[valid==4].index);shared=g[g.method.isin(['nominal','past_pd_optimizer'])];counts=[]
            for direction in directions:
                direction_rows=d[(d.method==direction)&np.array([(r,s) in valid for r,s in zip(d.radius,d.sign)])];bank=pd.concat([shared,direction_rows]);counts.append(len(bank));nominal=bank[bank.candidate==0].iloc[0]
                for method in ['controller_displacement','proprio_contact','lewm_global','oracle']:
                    rule=a['settings'][direction]['rules'].get(method,{});eligible=bank[bank.active].copy()
                    if method=='oracle':eligible=eligible[(eligible.object_error<=1)&(eligible.jerk_gain>0)];cost='eef_jerk'
                    else:
                        column={'controller_displacement':'controller_score','proprio_contact':'contact_score','lewm_global':'lewm_score'}[method]
                        if rule.get('disabled'):eligible=eligible.iloc[:0]
                        else:eligible=eligible[(eligible.norm<=rule['radius']+1e-6)&(eligible[column]<=rule['tau'])&(eligible.predicted_jerk_cost<nominal.predicted_jerk_cost-1e-10)]
                        cost='predicted_jerk_cost'
                    expected=eligible.iloc[np.lexsort((eligible.candidate.to_numpy(),eligible[cost].to_numpy()))[0]] if len(eligible) else nominal
                    observed=selected[(selected.split==split)&(selected.state_id==state)&(selected.bank==direction)&(selected.method==method)];assert len(observed)==1;row=observed.iloc[0];assert row.candidate==expected.candidate;maxE=max(maxE,abs(row.object_error-expected.object_error));maxG=max(maxG,abs(row.jerk_gain-expected.jerk_gain));checks+=1
            assert len(set(counts))==1
    assert checks==len(selected)==3072 and max(maxE,maxG)<1e-12
    write_json(dest/'raw_audit.json',{'selection_rows':checks,'max_parent_E_difference':maxE,'max_parent_jerk_gain_difference':maxG,'scope':'Independently reconstructed action-feasible common support, per-state equal bank counts, locked threshold/radius eligibility and cost/candidate tie-breaking; all3072selected rows link to unchanged fully raw-audited physical parent. No recalibration during audit.'});print('PASS',checks,maxE,maxG,flush=True)
if __name__=='__main__':main()
