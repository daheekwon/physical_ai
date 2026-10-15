"""Post-hoc candidate/approval/ranking decomposition, fixed dev/cal choices only."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve,write_json,sha256
from zj_wam.selection import accepted
from scripts.calibrate_followup_selection import load_bank,selected_rows
from scripts.analyze_followup_response_selection import mean_ci
from scripts.analyze_followup_cube_proposals import load_candidates
from scripts.analyze_followup_cube import choose

BASE=resolve('artifacts/followup_v1')

def load(p):return json.loads(p.read_text())


def main():
    out=BASE/'selection_gap';out.mkdir(exist_ok=True);records=[];checks=0
    parent=pd.read_csv(BASE/'selection/selected_evaluation.csv');parent=parent[parent.rule_source=='selector']
    rules=load(BASE/'selection/selector_rules.json')['rules']
    sources={str(p.relative_to(resolve('.'))):sha256(p) for p in [BASE/'selection/selector_rules.json',BASE/'selection/selected_evaluation.csv',BASE/'cube/calibrated_settings.json',BASE/'cube/evaluation.csv',BASE/'cube/raw_audit.json']}
    for split in ['development','calibration']:
        f=load_bank(BASE,'f1',split)
        for method,rule in rules.items():
            selected=selected_rows(f,method,rule);selected['split']=split
            x=selected.merge(parent,on=['split','state_id','method'],suffixes=('','_parent'),validate='one_to_one');assert len(x)==105
            np.testing.assert_array_equal(x['index'],x.index_parent);np.testing.assert_allclose(x.jerk_gain,x.jerk_gain_parent,atol=1e-12);np.testing.assert_allclose(x.object_error,x.object_error_parent,atol=1e-12);checks+=len(x)
            approved=accepted(f,method,rule)
            for state,positions in f.groupby('state_id').indices.items():
                positions=np.asarray(positions);g=f.iloc[positions];a=approved[positions];row=selected[selected.state_id==state].iloc[0]
                useful=(g.actual_norm.to_numpy()>1e-6)&(g.object_error.to_numpy()<=1)&(g.jerk_gain.to_numpy()>=.1)
                positive=(g.nominal_predicted_jerk.to_numpy()>1e-8)&((g.nominal_predicted_jerk-g.predicted_jerk).to_numpy()>1e-8*g.nominal_predicted_jerk.to_numpy())
                records.append({'task':'PushT','split':split,'method':method,'state_id':state,'initial_pose_family':row.initial_pose_family,
                    'candidate_exists':bool(useful.any()),'useful_approved':bool((useful&a).any()),'useful_rankable':bool((useful&a&positive).any()),
                    'selected_useful':bool(row.active and row.object_error<=1 and row.jerk_gain>=.1),'selected_unsafe':bool(row.active and row.object_error>1),'active':bool(row.active)})
    cube=BASE/'cube';cfg=load(resolve('configs/followup_cube.json'));manifest=load(cube/'manifest.json');rules=load(cube/'calibrated_settings.json')['rules'];parent=pd.read_csv(cube/'evaluation.csv')
    for split in ['development','calibration']:
        f=load_candidates(cube,split,cfg,manifest)
        for method,rule in rules.items():
            selected=choose(f,method,rule)
            x=selected.merge(parent,on=['split','state_id','method'],suffixes=('','_parent'),validate='one_to_one');assert len(x)==96
            np.testing.assert_array_equal(x.candidate,x.candidate_parent);np.testing.assert_allclose(x.object_error,x.object_error_parent,atol=1e-12);np.testing.assert_allclose(x.jerk_gain,x.jerk_gain_parent,atol=1e-12);checks+=len(x)
            column={'controller_displacement':'controller_score','proprio_contact':'contact_score','lewm_global':'lewm_score'}[method]
            for state,g in f.groupby('state_id'):
                row=selected[selected.state_id==state].iloc[0];nominal=g[g.candidate==0].iloc[0]
                useful=g.active&(g.object_error<=1)&(g.jerk_gain>=.1)
                a=g.active&(g.norm<=rule.get('radius',.4)+1e-6)&(g[column]<=rule.get('tau',-np.inf))
                if rule.get('disabled',False):a[:]=False
                positive=g.predicted_jerk_cost<nominal.predicted_jerk_cost-1e-10
                records.append({'task':'Cube','split':split,'method':method,'state_id':state,'initial_pose_family':row.initial_pose_family,
                    'candidate_exists':bool(useful.any()),'useful_approved':bool((useful&a).any()),'useful_rankable':bool((useful&a&positive).any()),
                    'selected_useful':bool(row.joint),'selected_unsafe':bool(row.active and row.object_error>1),'active':bool(row.active)})
    f=pd.DataFrame(records)
    assert (f.candidate_exists>=f.useful_approved).all() and (f.useful_approved>=f.useful_rankable).all() and (f.useful_rankable>=f.selected_useful).all()
    f['no_useful_candidate']=~f.candidate_exists
    f['rejected_all_useful']=f.candidate_exists&~f.useful_approved
    f['nonpositive_predicted_gain']=f.useful_approved&~f.useful_rankable
    f['ranking_miss']=f.useful_rankable&~f.selected_useful
    categories=['no_useful_candidate','rejected_all_useful','nonpositive_predicted_gain','ranking_miss','selected_useful']
    assert (f[categories].astype(int).sum(1)==1).all()
    summary=[]
    for (task,split,method),g in f.groupby(['task','split','method']):
        summary.append({'task':task,'split':split,'method':method,'states':len(g),**{k:mean_ci(g,k) for k in ['candidate_exists','useful_approved','useful_rankable','selected_useful','selected_unsafe','active',*categories[:-1]]}})
    f.to_csv(out/'state_decomposition.csv',index=False)
    write_json(out/'analysis.json',{'summary':summary,'scope':'Post-hoc developmental diagnosis only; original frozen rules and same fixed bank, no new candidate execution/fitting/calibration/finaltest use. Useful means active+dense original tolerance+actualjerkgain10percent. Candidate/approval/ranking losses partition allstates; unsafe selected also reported separately. Counterfactual physical labels only diagnostic. Candidate_exists is finite bank opportunity, not physical global freedom. Cube commonbank has WM directions even for controllerverifier.'})
    write_json(out/'raw_audit.json',{'source_sha256':sha256(Path(__file__)),'source_inputs':sources,'selected_rows_exactly_reproduced':checks,'decomposition_rows':len(f),'nested_stage_checks':True,'mutually_exclusive_exhaustive_loss_categories':True,'scope':'Independent decomposition algebra and exact parent choices/physicalmetrics; underlying parent physical audits retained.'})
    for r in summary:
        if r['split']=='development' and r['method'] in ['controller_pusher','vision_contact','vision_adaptive','controller_displacement','proprio_contact','lewm_global']:
            print(r['task'],r['method'],*[f'{k}={r[k]["mean"]:.4f}' for k in categories],flush=True)
    print('AUDITED',len(f),'state-method choices',flush=True)
if __name__=='__main__':main()
