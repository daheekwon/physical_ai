"""Locked shift comparison, selected-action audit, and family uncertainty."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scripts.analyze_followup_response_selection import mean_ci
from zj_wam.adapters import ExpertDataset
from zj_wam.diagnostics import jerk_energy
from zj_wam.selection import select_indices
from zj_wam.io import resolve,write_json

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
 cfg=json.loads(resolve('configs/followup_v1.json').read_text());scfg=json.loads(resolve('configs/followup_shift_evaluation.json').read_text());out=resolve(cfg['output']);directory=out/'shift_evaluation';rules=json.loads((out/'selection/selector_rules.json').read_text())['rules'];gauss=json.loads((out/'strong_baselines/calibrated_settings.json').read_text())['rules']['gaussian_action'];data=ExpertDataset(cfg['dataset']);eps=cfg['tolerances'];frames=[];maxima={};physical_effects=[];counts={}
 def check(name,x,y,atol=1e-7):
  error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));maxima[name]=max(maxima.get(name,0),error)
  if error>atol:raise ValueError((name,error))
 for split in ['development','calibration']:
  paths=sorted(directory.glob(split+'*.json'));counts[split]=len(paths)
  if not args.partial and len(paths)!=105:raise ValueError(f'{split}: {len(paths)}/105')
  for p in paths:
   meta=json.loads(p.read_text());f=pd.read_csv(p.with_suffix('.csv'));assert len(f)==len(scfg['conditions'])*len(scfg['methods']);frames.append(f)
   with np.load(p.with_suffix('.npz')) as a:
    unshifted=a['nominal_nominal_physics']
    for condition,group in f.groupby('condition'):
     info=meta['decisions'][condition];count=info['candidate_count']+1;norm=a[condition+'_candidate_norms'];cost=a[condition+'_candidate_costs'];raw=a[condition+'_actions'];reference=a[condition+'_nominal_physics'];covref=a[condition+'_nominal_coverage'];j0=jerk_energy(reference[:,:2],.01)
     frame=pd.DataFrame({'state_id':meta['state_id'],'horizon':5,'actual_norm':norm[:count],'predicted_jerk':cost[:count],'nominal_predicted_jerk':cost[0]})
     for method in scfg['methods']:
      if method!='gaussian_action':frame[method+'_score']=a[condition+'_'+method+'_scores']
     for r in group.itertuples(index=False):
      if r.method=='gaussian_action':index=count if gauss['enabled'] and cost[-1]<cost[0]*(1-1e-8) and norm[-1]>1e-6 else 0
      else:index=max(0,select_indices(frame,r.method,rules[r.method])[meta['state_id']])
      assert index==r.index==info['selected_indices'][r.method]
      actual=a[condition+'_'+r.method+'_physics'];cov=a[condition+'_'+r.method+'_coverage'];check('prefix_same',actual[:30],reference[:30]);pos=np.linalg.norm(actual[30:,2:4]-reference[30:,2:4],axis=1).max();angle=actual[30:,4]-reference[30:,4];rot=np.abs(np.arctan2(np.sin(angle),np.cos(angle))).max();loss=np.maximum(0,covref-cov)
      error=max(pos/eps['position'],rot/eps['rotation'],loss.max()/eps['coverage_loss']);j=jerk_energy(actual[:,:2],.01);gain=(j0-j)/j0 if j0>1e-8 else 0
      check('object_error',error,r.object_error);check('jerk_gain',gain,r.jerk_gain);check('action_norm',np.linalg.norm((raw[index]-raw[0])/data.scaler.scale_),r.actual_norm,2e-6);check('predicted_gain',(cost[0]-cost[index])/max(cost[0],1e-8),r.predicted_gain)
     pos=np.linalg.norm(reference[30:,2:4]-unshifted[30:,2:4],axis=1).max();angle=reference[30:,4]-unshifted[30:,4];rot=np.abs(np.arctan2(np.sin(angle),np.cos(angle))).max()
     if not any(k in scfg['conditions'][condition] for k in ['pair_friction','kp_multiplier']):check('sensor_only_physics_identical',reference,unshifted,1e-9)
     physical_effects.append({'split':split,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'condition':condition,'position_change_px':float(pos),'rotation_change_deg':float(np.rad2deg(rot))})
 f=pd.concat(frames,ignore_index=True);f['joint']=f.active & f.preserved & (f.jerk_gain>=.1);f['violation']=f.object_error>1;f['jerk_gain_prediction_error']=np.abs(f.predicted_gain-f.jerk_gain);summary=[]
 for (split,condition,method),g in f.groupby(['split','condition','method']):
  for stratum in ['all','contact']:
   x=g if stratum=='all' else g[g.nominal_contact];active=x[x.active]
   summary.append({'split':split,'condition':condition,'method':method,'stratum':stratum,'states':len(x),**{m:mean_ci(x,m) for m in ['joint','active','jerk_gain','jerk_gain_prediction_error','vision_valid']},'risk':mean_ci(active,'violation')})
 paired=[]
 for (split,condition),g in f.groupby(['split','condition']):
  base=g[g.method=='vision_contact']
  for method in ['controller_pusher','vision_adaptive','lewm_global','gaussian_action']:
   x=g[g.method==method].merge(base,on='state_id',suffixes=('','_base')).copy();x['joint_diff']=x.joint.astype(float)-x.joint_base.astype(float);x['gain_diff']=x.jerk_gain-x.jerk_gain_base
   paired.append({'split':split,'condition':condition,'method':method,'baseline':'vision_contact','joint_difference':mean_ci(x,'joint_diff'),'gain_difference':mean_ci(x,'gain_diff')})
 f.to_csv(directory/'evaluation.csv',index=False);pd.DataFrame(physical_effects).to_csv(directory/'physical_effects.csv',index=False)
 write_json(directory/'analysis.json',{'scope':'Partial run, not final condition comparison' if args.partial else scfg['data_scope'],'counts':counts,'summary':summary,'paired':paired,'interpretation':'Reused nominal rules. Online gains only; allowance/compute factorial not performed here. Groundtruth changedparameters not supplied. Contact membership can vary byshift. Unknown controller gainalsochanges jerk prediction.'})
 write_json(directory/'raw_audit.json',{'states':counts,'rows':len(f),'max_absolute_recalculation_error':maxima,'verified':'Selections reconstructed from saved predictions/costs/norms only, then raw physical E/jerk and action norms. Sensor-only conditions retain exact nominal physics. Savedcoverage endpointsused; not an independent geometry coverage audit.'})
 for r in summary:
  if r['split']=='development' and r['stratum']=='all':print(r['condition'],r['method'],'U',round(r['joint']['mean'],3),'R',r['risk']['mean'])
if __name__=='__main__':main()
