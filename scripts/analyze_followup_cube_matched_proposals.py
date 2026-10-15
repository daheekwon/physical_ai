"""Same-count Cube direction comparison on common action-feasible radius/sign support."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scripts.analyze_followup_cube_proposals import load_candidates
from scripts.analyze_followup_cube import calibrate,choose
from scripts.analyze_followup_response_selection import mean_ci
from zj_wam.io import resolve,write_json,sha256


def main():
    cfg=json.loads(resolve('configs/followup_cube.json').read_text());base=resolve(cfg['output']);out=base/'matched_proposals';out.mkdir(exist_ok=True);manifest=json.loads((base/'manifest.json').read_text());directions=['random0','random1','lewm_low','lewm_high'];banks={};support=[];checks=[]
    for split in ['calibration','development']:
        f=load_candidates(base,split,cfg,manifest);d=f[f.method.isin(directions)];keys=['state_id','radius','sign'];counts=d.groupby(keys).method.nunique();cells=counts[counts==len(directions)].reset_index()[keys];matched=d.merge(cells,on=keys,validate='many_to_one');shared=f[f.method.isin(['nominal','past_pd_optimizer'])].copy()
        check=matched.groupby(keys).norm.agg(['min','max']);assert np.max(check['max']-check['min'])<1e-10
        support.append({'split':split,'all_states':f.state_id.nunique(),'states_with_matched_direction':matched.state_id.nunique(),'matched_radius_sign_cells':len(cells),'shared_nominal_optimizer_candidates':len(shared),'direction_candidates_per_bank':len(cells),'common_norm_max_error':float(np.max(check['max']-check['min']))})
        for direction in directions:
            banks[split,direction]=pd.concat([shared,matched[matched.method==direction]],ignore_index=True)
        reference=banks[split,'random0'].groupby('state_id').size()
        for direction in directions:
            pd.testing.assert_series_equal(banks[split,direction].groupby('state_id').size(),reference);checks.append({'split':split,'bank':direction,'states':len(reference),'candidates':int(reference.sum())})
    rows=[];settings={}
    for direction in directions:
        print('CALIBRATE',direction,flush=True);settings[direction]=calibrate(banks['calibration',direction],cfg)
        for split in ['calibration','development']:
            for method in ['controller_displacement','proprio_contact','lewm_global','oracle']:
                x=choose(banks[split,direction],method,settings[direction]['rules'].get(method,{}));x['bank']=direction;rows.append(x)
    f=pd.concat(rows,ignore_index=True);summary=[];paired=[]
    for (split,bank,method),g in f.groupby(['split','bank','method']):summary.append({'split':split,'bank':bank,'method':method,'states':len(g),'joint':mean_ci(g,'joint'),'active':mean_ci(g,'active'),'risk':mean_ci(g[g.active],'violation'),'gain':mean_ci(g,'jerk_gain')})
    for split in ['calibration','development']:
        for method in ['controller_displacement','proprio_contact','lewm_global','oracle']:
            for baseline in ['random0','random1']:
                b=f[(f.split==split)&(f.method==method)&(f.bank==baseline)]
                for direction in ['lewm_low','lewm_high']:
                    x=f[(f.split==split)&(f.method==method)&(f.bank==direction)].merge(b,on=['state_id','initial_pose_family'],validate='one_to_one',suffixes=('','_base'));x['delta_U']=x.joint.astype(float)-x.joint_base.astype(float);x['delta_gain']=x.jerk_gain-x.jerk_gain_base;paired.append({'split':split,'method':method,'bank':direction,'baseline':baseline,'U_difference':mean_ci(x,'delta_U'),'gain_difference':mean_ci(x,'delta_gain')})
    f.to_csv(out/'evaluation.csv',index=False);write_json(out/'analysis.json',{'summary':summary,'paired':paired,'support':support,'count_checks':checks,'settings':settings,'source_sha256':sha256(Path(__file__)),'parent_raw_audit_sha256':sha256(base/'raw_audit.json'),'scope':'One direction per bank + identical nominal/PD-optimizer candidates; same actual directionnorm/radius/sign and per-state feasible candidate count across four banks. Shared support chosen only from action feasibility, not physical outcomes. Each verifier separately calibrated on same calibrationfamilies/searchgrid. States without shared direction cells retain shared optimizer/nominal and remain in denominator. Offline common-support direction-quality diagnostic, not compute-matched or fully WM-free deployment (support construction uses availability of all directions). Realized EEFmotion/frequency not matched.'});print('COMPLETE',len(f),'selected rows',flush=True)
if __name__=='__main__':main()
