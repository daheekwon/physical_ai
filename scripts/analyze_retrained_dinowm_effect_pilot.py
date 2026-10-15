"""Recompute intervention metrics from saved DINO tokens and physical traces."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_retrained_effect_pilot';read=lambda p:json.loads(p.read_text())


def corr(x,y):
    x=np.asarray(x);y=np.asarray(y);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<3 or np.ptp(x)<1e-12 or np.ptp(y)<1e-12:return None
    return float(spearmanr(x,y).statistic)


def main():
    m=read(OUT/'manifest.json');done=read(OUT/'execution_complete.json');assert done['states']==21 and done['manifest_sha256']==sha256(OUT/'manifest.json')
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    tol=read(resolve('configs/followup_v1.json'))['tolerances'];cube=read(resolve('configs/followup_cube.json'));rows=[];state_rows=[];errors={}
    for env,paths in m['states'].items():
        for name in paths:
            src=resolve(name);dest=OUT/env/src.name;meta=read(dest);assert meta['manifest_sha256']==sha256(OUT/'manifest.json');old=pd.read_csv(src.with_suffix('.csv'))
            with np.load(dest.with_suffix('.npz')) as a:pred=a['predicted'].astype(float).reshape(len(a['predicted']),-1);actual=a['actual'].astype(float).reshape(len(a['actual']),-1)
            assert pred.shape==actual.shape and pred.shape[1]==196*384 and np.isfinite(pred).all() and np.isfinite(actual).all()
            pdiff=pred-pred[:1];adiff=actual-actual[:1];ps=np.sqrt(np.mean(pdiff**2,axis=1));rs=np.sqrt(np.mean(adiff**2,axis=1));err=np.sqrt(np.mean((pdiff-adiff)**2,axis=1));den=np.linalg.norm(pdiff,axis=1)*np.linalg.norm(adiff,axis=1);cos=np.divide(np.sum(pdiff*adiff,axis=1),den,out=np.full(len(den),np.nan),where=den>1e-12)
            with np.load(src.with_suffix('.npz')) as a:
                if env=='pusht':
                    d=a['physics'][:,30:,2:5]-a['physics'][:1,30:,2:5];pos=np.linalg.norm(d[:,:,:2],axis=-1)/tol['position'];rot=np.abs(np.arctan2(np.sin(d[:,:,2]),np.cos(d[:,:,2])))/tol['rotation'];endpoint=np.maximum(pos[:,49],rot[:,49]);coverage=np.maximum(0,a['coverage'][:1]-a['coverage'])/tol['coverage_loss'];E=np.maximum(np.maximum(pos.max(1),rot.max(1)),coverage.max(1));ids=old['index'].to_numpy(int);norm=old.actual_norm.to_numpy();robot=old.pusher_rms.to_numpy()
                else:
                    pose=a['candidate_object_pose'];pos=np.linalg.norm(pose[:,:,:3]-pose[:1,:,:3],axis=-1)/cube['position_tolerance_m'];quat=pose[:,:,3:];quat=quat/np.linalg.norm(quat,axis=-1,keepdims=True);dots=np.clip(np.abs(np.sum(quat*quat[:1],axis=-1)),0,1);rot=2*np.arctan2(np.sqrt(np.maximum(0,1-dots*dots)),dots)/cube['rotation_tolerance_rad'];endpoint=np.maximum(pos[:,124],rot[:,124]);E=np.maximum(pos.max(1),rot.max(1));ids=old.candidate.to_numpy(int);norm=old.norm.to_numpy();eef=a['candidate_eef_position'];robot=np.sqrt(np.mean(np.sum((eef-eef[:1])**2,axis=-1),axis=1))[ids]
            maximum=float(np.max(np.abs(E[ids]-old.object_error.to_numpy())));errors[env+'_parent_physical_E']=max(errors.get(env+'_parent_physical_E',0.),maximum);assert maximum<2e-5,(env,name,maximum)
            local=[]
            for k,(i,normvalue) in enumerate(zip(ids,norm)):
                row={'environment':env,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'candidate':int(i),'method':old.iloc[k].method,'active':bool(normvalue>1e-6),'norm':float(normvalue),'predicted_rms':float(ps[i]),'actual_rms':float(rs[i]),'effect_rmse':float(err[i]),'effect_cosine':float(cos[i]) if np.isfinite(cos[i]) else None,'endpoint_pose_E_t5':float(endpoint[i]),'trajectory_E_t8':float(E[i]),'robot_rms':float(robot[k]),'jerk_gain':float(old.iloc[k].jerk_gain)};rows.append(row);local.append(row)
            f=pd.DataFrame(local);active=f[f.active];state_rows.append({'environment':env,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'active_candidates':len(active),'candidate_rows':len(f),'nominal_prediction_rmse':float(np.sqrt(np.mean((pred[0]-actual[0])**2))), 'predicted_vs_actual_change_rho':corr(active.predicted_rms,active.actual_rms),'predicted_vs_endpoint_object_rho':corr(active.predicted_rms,active.endpoint_pose_E_t5),'actual_vs_endpoint_object_rho':corr(active.actual_rms,active.endpoint_pose_E_t5),'predicted_vs_trajectory_object_rho':corr(active.predicted_rms,active.trajectory_E_t8),'actual_vs_trajectory_object_rho':corr(active.actual_rms,active.trajectory_E_t8),'actual_vs_robot_rho':corr(active.actual_rms,active.robot_rms),'mean_effect_cosine':float(active.effect_cosine.mean()) if active.effect_cosine.notna().any() else None})
    f=pd.DataFrame(rows);s=pd.DataFrame(state_rows);summary=[]
    keys=[k for k in s.columns if k.endswith('_rho')]+['mean_effect_cosine','nominal_prediction_rmse']
    for env,g in s.groupby('environment'):
        summary.append({'environment':env,'states':len(g),'families':int(g.initial_pose_family.nunique()),'candidate_rows':int(g.candidate_rows.sum()),**{k:mean_ci(g.dropna(subset=[k]),k) for k in keys}})
    f.to_csv(OUT/'evaluation.csv',index=False);s.to_csv(OUT/'state_metrics.csv',index=False)
    write_json(OUT/'analysis.json',{'summary':summary,'scope':m['config']['scope'],'metrics':m['config']['metrics'],'note':'Undefined correlations remain null, not zero. Different banks across tasks; no cross-task ranking or LeWM superiority contrast.'})
    write_json(OUT/'raw_audit.json',{'states':len(s),'candidate_rows':len(f),'max_error':errors,'manifest_sha256':sha256(OUT/'manifest.json'),'source_sha256':sha256(Path(__file__)),'scope':'Metrics recomputed from retained full tokens and physical arrays, original physicalE reproduced. No new physics and no model-forward rerun in this analyzer.'})
    print('AUDITED DINO physical effects',len(s),'states',len(f),'candidate rows',flush=True)
if __name__=='__main__':main()
