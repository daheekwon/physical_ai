"""F8 and F9 distinct calibration distributions; no test access or unlock."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.selection import accepted,select_indices
from zj_wam.io import resolve,write_json,sha256


def load_bank(out,bank,split):
    paths=sorted((out/'response_scores'/bank).glob(f'{split}_*_h05.json'))
    expected=45 if split=='test' else 105
    if len(paths)!=expected:raise ValueError(f'{bank}/{split} incomplete: {len(paths)}/{expected}')
    f=pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in paths],ignore_index=True)
    if f.state_id.nunique()!=expected:raise ValueError('Incorrect unique state count')
    return f


def weighted_risk(frame,mask,contact=False):
    weights=1/frame.groupby('state_id').state_id.transform('size').to_numpy(float)
    include=mask & (frame.nominal_contact.to_numpy(bool) if contact else True)
    total=float(weights[include].sum())
    return float(weights[include & (frame.object_error.to_numpy()>1)].sum()/total) if total>0 else None


def selected_rows(frame,method,rule):
    choices=select_indices(frame,method,rule)
    rows=[]
    for sid,index in choices.items():
        state=frame[frame.state_id==sid].iloc[0]
        common={'state_id':sid,'initial_pose_family':int(state.initial_pose_family),'nominal_contact':bool(state.nominal_contact),'method':method}
        if index<0:rows.append({**common,'index':0,'active':False,'predicted_gain':0.,'jerk_gain':0.,'preserved':True,'object_error':0.,'norm':0.})
        else:
            row=frame.iloc[index];cost=float(row.nominal_predicted_jerk)
            rows.append({**common,'index':int(row['index']),'active':True,'predicted_gain':(cost-float(row.predicted_jerk))/cost,
                         'jerk_gain':float(row.jerk_gain),'preserved':bool(row.object_error<=1),'object_error':float(row.object_error),'norm':float(row.actual_norm)})
    return pd.DataFrame(rows)


def selection_metrics(rows,cfg):
    active=rows[rows.active];contact=active[active.nominal_contact]
    return {'active_coverage':float(rows.active.mean()),
            'risk':float((active.object_error>1).mean()) if len(active) else None,
            'contact_risk':float((contact.object_error>1).mean()) if len(contact) else None,
            'predicted_useful_coverage':float((rows.active & (rows.predicted_gain>=cfg['meaningful_gain'])).mean()),
            'actual_joint_rate':float((rows.active & rows.preserved & (rows.jerk_gain>=cfg['meaningful_gain'])).mean()),
            'mean_predicted_gain':float(rows.predicted_gain.mean()),'mean_actual_gain':float(rows.jerk_gain.mean())}


def fit(out,cfg,bank):
    directory=out/'selection';directory.mkdir(exist_ok=True)
    path=directory/('bank_rules.json' if bank=='interventions' else 'selector_rules.json')
    if path.exists():print('Already frozen:',path);return
    if (out/'protocol/final_selection_lock.json').exists():raise ValueError('Cannot calibrate after final test unlock')
    frame=load_bank(out,bank,'calibration');rules={};curves=[]
    for method in cfg['score_methods']:
        scores=frame[method+'_score'].to_numpy(float);finite=scores[np.isfinite(scores)]
        thresholds=np.unique(np.r_[0.,np.quantile(finite,np.linspace(0,1,cfg['threshold_quantiles']))]) if len(finite) else np.array([0.])
        best=None;best_value=None
        for radius in cfg['radii']:
          for tau in thresholds:
            rule={'enabled':True,'relative_radius':radius,'tau':float(tau)}
            if bank=='interventions':
                take=accepted(frame,method,rule);weights=1/frame.groupby('state_id').state_id.transform('size').to_numpy(float)
                coverage=float(weights[take].sum()/weights.sum())
                risk=weighted_risk(frame,take);contact_risk=weighted_risk(frame,take,True)
                contact_active_states=frame[take & frame.nominal_contact].state_id.nunique()
                metrics={'active_coverage':coverage,'risk':risk,'contact_risk':contact_risk,'contact_active_states':contact_active_states}
                value=(coverage,float(frame.loc[take,'actual_norm'].mean()) if take.any() else 0.)
            else:
                selected=selected_rows(frame,method,rule);metrics=selection_metrics(selected,cfg)
                contact_active_states=int((selected.active & selected.nominal_contact).sum())
                metrics['contact_active_states']=contact_active_states
                value=(metrics['predicted_useful_coverage'],metrics['mean_predicted_gain'])
            valid=metrics['active_coverage']>=cfg['minimum_active_coverage'] and metrics['risk'] is not None and metrics['risk']<=cfg['target_violation'] and metrics['contact_risk'] is not None and metrics['contact_risk']<=cfg['target_violation']
            curves.append({'method':method,**rule,**metrics,'valid':valid})
            if valid and (best_value is None or value>best_value):best={**rule,'calibration':metrics};best_value=value
        rules[method]=best or {'enabled':False,'relative_radius':0.,'tau':None,'reason':'No empirical-risk/coverage feasible rule in prespecified search'}
        print(bank,method,json.dumps(rules[method]),flush=True)
    pd.DataFrame(curves).to_csv(directory/f'{bank}_calibration_curves.csv',index=False)
    write_json(path,{'scope':'Calibration only; empirical constraints are not probabilistic safety guarantees','bank':bank,'config':cfg,
                    'rules':rules,'families':frame.initial_pose_family.nunique(),'states':frame.state_id.nunique(),
                    'source_sha256':sha256(Path(__file__)),'config_sha256':sha256(resolve('configs/followup_selection.json'))})
    if bank=='f1':
        pd.concat([selected_rows(frame,m,r) for m,r in rules.items()],ignore_index=True).to_csv(directory/'calibration_selected.csv',index=False)


def main():
    out=resolve('artifacts/followup_v1');cfg=json.loads(resolve('configs/followup_selection.json').read_text())
    fit(out,cfg,'interventions');fit(out,cfg,'f1')

if __name__=='__main__':main()
