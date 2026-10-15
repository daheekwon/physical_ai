"""F7 response diagnostics and F8/F9 development/calibration evidence, no test tuning."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from zj_wam.io import resolve,write_json
from zj_wam.selection import accepted
from scripts.calibrate_followup_selection import load_bank,selected_rows,weighted_risk,selection_metrics

OUT=resolve('artifacts/followup_v1')

def wrap(x):return np.arctan2(np.sin(x),np.cos(x))

def mean_ci(frame,value,seed=20261011):
    g=frame.groupby('initial_pose_family')[value].agg(['sum','count']).to_numpy(float)
    if not len(g):return {'mean':None,'ci95':None,'families':0}
    rng=np.random.default_rng(seed);ids=rng.integers(len(g),size=(2000,len(g)))
    sampled=g[ids].sum(1);v=sampled[:,0]/sampled[:,1]
    return {'mean':float(g[:,0].sum()/g[:,1].sum()),'ci95':np.quantile(v,[.025,.975]).tolist(),'families':len(g)}


def response():
    rows=[];metadata=[]
    for split in ['development','calibration']:
      for p in sorted((OUT/'response_scores/interventions').glob(f'{split}*h05.json')):
        m=json.loads(p.read_text());metadata.append(m)
        with np.load(OUT/'interventions'/p.with_suffix('.npz').name) as a: truth=a['physics'][:,30:,2:5].copy()
        with np.load(p.with_suffix('.npz')) as a:predictions={k:a[k].copy() for k in a.files if k in ['state_fixed','state_local','state_adaptive','vision_fixed','vision_local','vision_adaptive','predicted_pose','predicted_history_pose']}
        source=pd.read_csv(p.with_suffix('.csv'));idx=source['index'].to_numpy(int);contact=bool(source.nominal_contact.iloc[0])
        real=truth-truth[0:1];real[:,:,2]=wrap(real[:,:,2])
        for name,pred in predictions.items():
          end=pred[:,49] if name.startswith(('state','vision')) else pred[:,0]
          effect=end-end[0:1];effect[:,2]=wrap(effect[:,2]);err=effect[idx]-real[idx,49];err[:,2]=wrap(err[:,2])
          row={'split':split,'state_id':m['state_id'],'initial_pose_family':m['initial_pose_family'],'nominal_contact':contact,'method':name,
               'effect_position_error_px':float(np.linalg.norm(err[:,:2],axis=1).mean()),'effect_rotation_error_deg':float(np.rad2deg(np.abs(err[:,2])).mean()),
               'nominal_position_error_px':float(np.linalg.norm(end[0,:2]-truth[0,49,:2])),
               'nominal_rotation_error_deg':float(np.rad2deg(abs(wrap(end[0,2]-truth[0,49,2]))))}
          if name.startswith(('state','vision')):
            pe=pred-pred[0:1];pe[:,:,2]=wrap(pe[:,:,2]);e=pe[idx]-real[idx];e[:,:,2]=wrap(e[:,:,2])
            row['dense_effect_position_max_error_px']=float(np.linalg.norm(e[:,:,:2],axis=-1).max(1).mean())
            row['dense_effect_rotation_max_error_deg']=float(np.rad2deg(np.abs(e[:,:,2])).max(1).mean())
          rows.append(row)
        rows.append({'split':split,'state_id':m['state_id'],'initial_pose_family':m['initial_pose_family'],'nominal_contact':contact,'method':'zero_effect',
                     'effect_position_error_px':float(np.linalg.norm(real[idx,49,:2],axis=-1).mean()),'effect_rotation_error_deg':float(np.rad2deg(np.abs(real[idx,49,2])).mean())})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'response_scores/diagnostic_state_metrics.csv',index=False)
    summary=[]
    for (split,method),f in frame.groupby(['split','method']):
      for stratum in ['all','contact','free']:
        g=f if stratum=='all' else f[f.nominal_contact==(stratum=='contact')]
        metrics={c:mean_ci(g.dropna(subset=[c]),c) for c in frame.columns if c.endswith(('_px','_deg'))}
        summary.append({'split':split,'method':method,'stratum':stratum,**metrics})
    vision={}
    for split in ['development','calibration']:
        ms=[m for m in metadata if m['split']==split]
        vision[split]={'states':len(ms),'valid':sum(m['vision_valid'] for m in ms),
                       **{key:{'mean':float(np.mean([m[key] for m in ms])),'p90':float(np.quantile([m[key] for m in ms],.9)),'max':float(np.max([m[key] for m in ms]))} for key in ['vision_position_error','vision_rotation_error_deg']}}
    write_json(OUT/'response_scores/diagnostics.json',{'scope':'Separate development/calibration; same endpoint for model accuracy comparison; dense physical models separately','summary':summary,'vision':vision})
    print('Response diagnostics:',len(frame),'state-method rows',flush=True)


def selection():
    cfg=json.loads(resolve('configs/followup_selection.json').read_text());directory=OUT/'selection'
    bank_rules=json.loads((directory/'bank_rules.json').read_text())['rules'];selector=json.loads((directory/'selector_rules.json').read_text())['rules']
    selections=[];bank_rows=[];summaries=[]
    for split in ['development','calibration']:
      bank=load_bank(OUT,'interventions',split)
      for method,rule in bank_rules.items():
        mask=accepted(bank,method,rule)
        for sid,g in bank.groupby('state_id',sort=True):
            take=mask[g.index];n=int(take.sum());v=int((g.object_error.to_numpy()[take]>1).sum())
            bank_rows.append({'split':split,'method':method,'state_id':sid,'initial_pose_family':int(g.initial_pose_family.iloc[0]),'nominal_contact':bool(g.nominal_contact.iloc[0]),'acceptance':float(take.mean()),'violation_mass':v/len(g),'accepted':n,'violations':v})
      f=load_bank(OUT,'f1',split)
      for origin,rules in [('bank',bank_rules),('selector',selector)]:
       for method,rule in rules.items():
        chosen=selected_rows(f,method,rule);chosen['split']=split;chosen['rule_source']=origin
        chosen['joint']=chosen.active & chosen.preserved & (chosen.jerk_gain>=cfg['meaningful_gain'])
        selections.append(chosen)
        for stratum in ['all','contact','free']:
            g=chosen if stratum=='all' else chosen[chosen.nominal_contact==(stratum=='contact')]
            active=g[g.active].copy();active['violation']=~active.preserved
            summaries.append({'split':split,'method':method,'rule_source':origin,'stratum':stratum,'states':len(g),'active_count':len(active),
                              'active_coverage':mean_ci(g,'active'),'joint':mean_ci(g,'joint'),'mean_gain':mean_ci(g,'jerk_gain'),'conditional_violation':mean_ci(active,'violation')})
      # Oracle selects from exactly the same bank using evaluation outcomes only.
      oracle=[]
      for sid,g in f.groupby('state_id'):
        valid=g[(g.object_error<=1)&(g.jerk_gain>0)&(g.actual_norm>1e-6)]
        gain=float(valid.jerk_gain.max()) if len(valid) else 0.
        oracle.append({'state_id':sid,'initial_pose_family':int(g.initial_pose_family.iloc[0]),'nominal_contact':bool(g.nominal_contact.iloc[0]),'gain':gain,'joint':gain>=.1})
      for stratum in ['all','contact','free']:
        g=pd.DataFrame(oracle);g=g if stratum=='all' else g[g.nominal_contact==(stratum=='contact')]
        summaries.append({'split':split,'method':'oracle','rule_source':'outcome_oracle','stratum':stratum,'states':len(g),'joint':mean_ci(g,'joint'),'mean_gain':mean_ci(g,'gain')})
    chosen=pd.concat(selections,ignore_index=True);chosen.to_csv(directory/'selected_evaluation.csv',index=False)
    b=pd.DataFrame(bank_rows);b.to_csv(directory/'bank_state_evaluation.csv',index=False)
    bs=[]
    for (split,method),f in b.groupby(['split','method']):
      for stratum in ['all','contact','free']:
        g=f if stratum=='all' else f[f.nominal_contact==(stratum=='contact')]
        agg=g.groupby('initial_pose_family')[['acceptance','violation_mass']].sum().to_numpy();rng=np.random.default_rng(20261011);ids=rng.integers(len(agg),size=(2000,len(agg)));s=agg[ids].sum(1);risk=np.divide(s[:,1],s[:,0],out=np.full(2000,np.nan),where=s[:,0]>0)
        bs.append({'split':split,'method':method,'stratum':stratum,'coverage':mean_ci(g,'acceptance'),
                   'risk':float(g.violation_mass.sum()/g.acceptance.sum()) if g.acceptance.sum()>0 else None,
                   'risk_ci95':np.nanquantile(risk,[.025,.975]).tolist() if np.isfinite(risk).any() else None})
    paired=[]
    methods=cfg['score_methods']
    for split in ['development','calibration']:
      for control in ['controller_pusher','vision_contact','vision_local']:
        base=chosen[(chosen.split==split)&(chosen.rule_source=='selector')&(chosen.method==control)]
        for method in methods:
          f=chosen[(chosen.split==split)&(chosen.rule_source=='selector')&(chosen.method==method)].merge(base,on='state_id',suffixes=('','_base'))
          f['joint_diff']=f.joint.astype(float)-f.joint_base.astype(float);f['gain_diff']=f.jerk_gain-f.jerk_gain_base
          paired.append({'split':split,'method':method,'baseline':control,'joint_difference':mean_ci(f,'joint_diff'),'mean_gain_difference':mean_ci(f,'gain_diff')})
    write_json(directory/'evaluation_summary.json',{'scope':'Development diagnostic and calibration resubstitution; not independent final test','selection':summaries,'bank':bs,'paired':paired})
    print('Saved selection evidence:',len(chosen),'state-method choices',flush=True)
    for r in summaries:
        if r['split']=='development' and r['stratum']=='all' and r['rule_source']=='selector':print(r['method'],'U',round(r['joint']['mean'],3),'C',round(r['active_coverage']['mean'],3),'R',r['conditional_violation']['mean'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['response','selection','all'],default='all');args=p.parse_args()
 if args.mode in ['response','all']:response()
 if args.mode in ['selection','all']:selection()
