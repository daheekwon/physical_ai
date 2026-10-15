"""Error complementarity and calibrated conjunction on the identical smoothing bank."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scripts.calibrate_followup_selection import load_bank,selected_rows,selection_metrics
from scripts.analyze_followup_response_selection import mean_ci
from zj_wam.selection import accepted
from zj_wam.io import resolve,write_json,sha256

def ratio_ci(f,numerator,denominator):
 if not len(f) or not f[denominator].sum():return {'mean':None,'ci95':None,'families':int(f.initial_pose_family.nunique())}
 a=f.groupby('initial_pose_family')[[numerator,denominator]].sum().to_numpy(float);rng=np.random.default_rng(20261011);ids=rng.integers(len(a),size=(2000,len(a)));s=a[ids].sum(1);valid=s[:,1]>0;v=s[valid,0]/s[valid,1]
 return {'mean':float(a[:,0].sum()/a[:,1].sum()),'ci95':np.quantile(v,[.025,.975]).tolist(),'families':len(a),'bootstrap_undefined':int((~valid).sum())}

def main():
 out=resolve('artifacts/followup_v1');dest=out/'hybrid_verifier';dest.mkdir(exist_ok=True)
 cfg=json.loads(resolve('configs/followup_selection.json').read_text());hcfg=json.loads(resolve('configs/followup_hybrid_verifier.json').read_text());individual=json.loads((out/'selection/selector_rules.json').read_text())['rules']
 frames={sp:load_bank(out,'f1',sp) for sp in ['development','calibration']};cross=[];curves=[];summary=[];choices=[];frozen={};path=dest/'calibrated_rules.json'
 if path.exists():frozen=json.loads(path.read_text())['rules']
 elif (out/'protocol/final_selection_lock.json').exists():raise ValueError('Cannot calibrate after test unlock')
 for left,right in hcfg['pairs']:
  method=left+'_and_'+right
  assert individual[left]['enabled'] and individual[right]['enabled'] and individual[left]['tau']>0 and individual[right]['tau']>0
  scales=[individual[m]['tau'] for m in [left,right]]
  for split,f in frames.items():
   a=accepted(f,left,individual[left]);b=accepted(f,right,individual[right]);weights=1/f.groupby('state_id').state_id.transform('size').to_numpy(float)
   f[method+'_score']=np.maximum(f[left+'_score']/scales[0],f[right+'_score']/scales[1])
   for label,mask in [('both',a&b),('lewm_only',a&~b),('contact_only',~a&b),('neither',~a&~b)]:
    x=pd.DataFrame({'initial_pose_family':f.initial_pose_family,'denominator':weights*mask,'violation':weights*mask*(f.object_error>1),'useful':weights*mask*(f.object_error<=1)*(f.jerk_gain>=.1),'all_weight':weights})
    cross.append({'split':split,'pair':method,'cell':label,'raw_candidates':int(mask.sum()),'state_weighted_coverage':ratio_ci(x,'denominator','all_weight'),'risk':ratio_ci(x,'violation','denominator'),'useful_fraction':ratio_ci(x,'useful','denominator')})
  if method not in frozen:
   f=frames['calibration'];values=f[method+'_score'].to_numpy();thresholds=np.unique(np.r_[0,np.quantile(values[np.isfinite(values)],np.linspace(0,1,cfg['threshold_quantiles']))]);best=None;best_value=None
   for radius in cfg['radii']:
    for tau in thresholds:
     rule={'enabled':True,'relative_radius':float(radius),'tau':float(tau)};selected=selected_rows(f,method,rule);m=selection_metrics(selected,cfg)
     valid=m['active_coverage']>=cfg['minimum_active_coverage'] and m['risk'] is not None and m['risk']<=cfg['target_violation'] and m['contact_risk'] is not None and m['contact_risk']<=cfg['target_violation']
     curves.append({'method':method,**rule,**m,'valid':valid});value=(m['predicted_useful_coverage'],m['mean_predicted_gain'])
     if valid and (best_value is None or value>best_value):best={**rule,'calibration':m,'component_scales':scales};best_value=value
   frozen[method]=best or {'enabled':False,'reason':'No calibrated feasible rule','component_scales':scales}
  for split,f in frames.items():
   x=selected_rows(f,method,frozen[method]);x['split']=split;x['joint']=x.active & x.preserved & (x.jerk_gain>=.1);choices.append(x)
   for stratum in ['all','contact']:
    g=x if stratum=='all' else x[x.nominal_contact];active=g[g.active].copy();active['violation']=~active.preserved
    summary.append({'split':split,'method':method,'stratum':stratum,**{k:mean_ci(g,k) for k in ['joint','active','jerk_gain']},'risk':mean_ci(active,'violation')})
 if not path.exists():
  write_json(path,{'rules':frozen,'config':hcfg,'source_sha256':sha256(Path(__file__)),'scope':'Calibration only. Fixed component ratio and common tuning budget.'});pd.DataFrame(curves).to_csv(dest/'calibration_curves.csv',index=False)
 selected=pd.concat(choices,ignore_index=True);selected.to_csv(dest/'selected.csv',index=False);base=pd.read_csv(out/'selection/selected_evaluation.csv');paired=[]
 for split in ['development','calibration']:
  for method in frozen:
   for baseline in ['vision_contact','vision_adaptive','lewm_global','lewm_object']:
    b=base[(base.split==split)&(base.method==baseline)&(base.rule_source=='selector')]
    x=selected[(selected.split==split)&(selected.method==method)].merge(b,on='state_id',suffixes=('','_base')).copy();x['joint_diff']=x.joint.astype(float)-x.joint_base.astype(float)
    paired.append({'split':split,'method':method,'baseline':baseline,'joint_difference':mean_ci(x,'joint_diff')})
 write_json(dest/'analysis.json',{'scope':hcfg['scope'],'cross_tab':cross,'summary':summary,'paired':paired,'selection_scope':'Hybrid max score consumes predictions only; physical outcomes used to fit calibration rules and evaluate afterward.'})
 for r in summary:
  if r['split']=='development' and r['stratum']=='all':print(r['method'],{k:r[k]['mean'] for k in ['joint','active','risk']},flush=True)
if __name__=='__main__':main()
