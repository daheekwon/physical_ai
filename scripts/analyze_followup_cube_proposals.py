"""Separate Cube verifier value from LeWM-generated direction availability."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_cube import calibrate,choose
from scripts.analyze_followup_response_selection import mean_ci


def load_candidates(out,split,cfg,manifest):
    verified=json.loads((out/'raw_audit.json').read_text())[split];assert verified['states']==96
    frames=[]
    for path in sorted((out/split).glob('e*.csv')):
        meta=json.loads(path.with_suffix('.json').read_text());assert meta['signature']==manifest['signature'];f=pd.read_csv(path)
        with np.load(path.with_suffix('.npz')) as a:
            pred=a['predicted_robot'];delta=pred-pred[:1];pos=np.linalg.norm(delta[:,:,:3],axis=-1);angle=np.abs(np.arctan2(np.sin(delta[:,:,3]),np.cos(delta[:,:,3])));f['controller_score']=np.maximum(pos.max(1)/cfg['position_tolerance_m'],angle.max(1)/cfg['rotation_tolerance_rad']);gate=bool(float(a['root_gripper_contact'])>.1 or np.any(np.r_[a['nominal_actions'][:,4],a['suffix_actions'][:,4]]>.1));f['contact_score']=np.maximum((pos+np.sqrt(3)*.02*angle).max(1)/cfg['position_tolerance_m'],angle.max(1)/cfg['rotation_tolerance_rad']) if gate else np.zeros(len(f));f['active']=f.norm>1e-6;f['joint']=f.active&(f.object_error<=1)&(f.jerk_gain>=cfg['jerk_gain_threshold']);f['violation']=f.object_error>1
        frames.append(f)
    allrows=pd.concat(frames,ignore_index=True);assert len(frames)==96 and len(allrows)==verified['candidate_rows'];return allrows


def main():
    cfg=json.loads(resolve('configs/followup_cube.json').read_text());out=resolve(cfg['output']);manifest=json.loads((out/'manifest.json').read_text());dest=out/'proposal_ablation';dest.mkdir(exist_ok=True);original=json.loads((out/'calibrated_settings.json').read_text());frames={s:load_candidates(out,s,cfg,manifest) for s in ['calibration','development']};settings={};rows=[];counts=[]
    for bank in ['common','non_wm_proposals']:
        bankframes={s:f if bank=='common' else f[~f.method.isin(['lewm_low','lewm_high'])].copy() for s,f in frames.items()};locked=original if bank=='common' else calibrate(bankframes['calibration'],cfg);settings[bank]=locked
        for split,f in bankframes.items():
            counts.append({'bank':bank,'split':split,'states':f.state_id.nunique(),'candidates':len(f),'generator_methods':sorted(f.method.unique())})
            for method in ['controller_displacement','proprio_contact','lewm_global','oracle']:
                x=choose(f,method,locked['rules'].get(method,{}));x['bank']=bank;rows.append(x)
    f=pd.concat(rows,ignore_index=True);summary=[];paired=[]
    for (split,bank,method),g in f.groupby(['split','bank','method']):summary.append({'split':split,'bank':bank,'method':method,'states':len(g),**{k:mean_ci(g,k) for k in ['joint','active','jerk_gain']},'risk':mean_ci(g[g.active],'violation')})
    for split in frames:
        for method in ['controller_displacement','proprio_contact','lewm_global','oracle']:
            a=f[(f.split==split)&(f.method==method)&(f.bank=='common')];b=f[(f.split==split)&(f.method==method)&(f.bank=='non_wm_proposals')];x=a.merge(b,on='state_id',suffixes=('','_nonwm'));x['joint_difference']=x.joint.astype(float)-x.joint_nonwm.astype(float);x['jerk_difference']=x.jerk_gain-x.jerk_gain_nonwm;paired.append({'split':split,'method':method,'common_minus_nonwm_joint':mean_ci(x,'joint_difference'),'common_minus_nonwm_jerk':mean_ci(x,'jerk_difference')})
    # Common-bank selections must reproduce the completed parent evaluation exactly.
    parent=pd.read_csv(out/'evaluation.csv');common=f[f.bank=='common'].merge(parent,on=['split','state_id','method'],suffixes=('','_parent'));assert len(common)==2*96*4
    np.testing.assert_array_equal(common.candidate,common.candidate_parent);np.testing.assert_allclose(common.object_error,common.object_error_parent,atol=1e-12);np.testing.assert_allclose(common.jerk_gain,common.jerk_gain_parent,atol=1e-12)
    f.to_csv(dest/'evaluation.csv',index=False);write_json(dest/'analysis.json',{'summary':summary,'paired':paired,'counts':counts,'settings':settings,'parent_raw_audit_sha256':sha256(out/'raw_audit.json'),'source_sha256':sha256(Path(__file__)),'common_parent_exact_reproduction_rows':len(common),'scope':'Same physically audited candidates; remove LeWM low/high directions for model-free proposal bank, retain polynomial random and past-PD optimizer generated without WM information. Calibrate each verifier separately on eachbank. LeWM verifier remains a model-based selector even on nonWMproposals; controller/contact selectors on that bank do not requireWM predictions. Common hasmorecandidates; betweenbank difference does not isolate direction quality at matched candidatecount or compute. No cachedbanktime claimedasfullruntime.'});print('COMPLETE',len(f),'selected rows',flush=True)
if __name__=='__main__':main()
