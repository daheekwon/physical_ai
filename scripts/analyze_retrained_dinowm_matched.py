"""Same-candidate score ranking diagnostic; no learned threshold or selection."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');PARENT=BASE/'dinowm_retrained_effect_pilot';OUT=BASE/'dinowm_retrained_matched_pilot';read=lambda p:json.loads(p.read_text())


def freeze():
    OUT.mkdir(exist_ok=True);parent=read(PARENT/'manifest.json');paths=[Path(__file__),PARENT/'manifest.json',resolve('scripts/analyze_followup_response_selection.py'),resolve('configs/followup_cube.json')]
    for name in parent['states']['pusht']:
        p=BASE/'response_scores/family_ablation'/Path(name).name;paths += [p.with_suffix(ext) for ext in ['.json','.csv','.npz']]
    m={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'methods':['dinowm_retrained','lewm_global','controller','action_norm'],'primary_baseline':'controller','scope':'Exploratory matched candidate ranking on3development families/task. Spearman versus5actionendpoint poseE and8actiontrajectoryE, ROC-AUC of8actionviolation for states containing both labels. No calibration/selection, no U/R or superiority verdict. Existing bank includes LeWM-generated candidates. Cross-model encoder/dataset/training/precision differ.'}
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==m
    else:write_json(OUT/'manifest.json',m)
    return m


def rho(x,y):
    if len(x)<3 or np.ptp(x)<1e-12 or np.ptp(y)<1e-12:return None
    return float(spearmanr(x,y).statistic)


def analyze(m):
    audit=read(PARENT/'raw_audit.json');assert audit['states']==21
    assert audit['manifest_sha256']==sha256(PARENT/'manifest.json')
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    parent=read(PARENT/'manifest.json');f=pd.read_csv(PARENT/'evaluation.csv');cube=read(resolve('configs/followup_cube.json'));records=[];joined=[];errors={}
    for env,paths in parent['states'].items():
        for name in paths:
            src=resolve(name);meta=read(src);g=f[(f.environment==env)&(f.state_id==meta['state_id'])].copy();ids=g.candidate.to_numpy(int)
            if env=='pusht':
                p=BASE/'response_scores/family_ablation'/src.name;old=pd.read_csv(p.with_suffix('.csv')).set_index('index').loc[ids];assert list(old.method)==list(g.method)
                with np.load(p.with_suffix('.npz')) as a:pred=a['predictions'].astype(float)
                recomputed=np.linalg.norm(pred-pred[:1],axis=-1).max(1)[ids];wm=old.lewm_global_score.to_numpy();ctrl=old.controller_pusher_score.to_numpy();err=float(np.max(np.abs(wm-recomputed)));assert err<2e-5
            else:
                old=pd.read_csv(src.with_suffix('.csv')).set_index('candidate').loc[ids]
                with np.load(src.with_suffix('.npz')) as a:pred=a['predicted_latent'].astype(float);robot=a['predicted_robot'].astype(float)
                recomputed=np.linalg.norm(pred-pred[:1],axis=1)[ids];wm=old.lewm_score.to_numpy();d=robot-robot[:1];pos=np.linalg.norm(d[:,:,:3],axis=-1);rot=np.abs(np.arctan2(np.sin(d[:,:,3]),np.cos(d[:,:,3])));ctrl=np.maximum(pos.max(1)/cube['position_tolerance_m'],rot.max(1)/cube['rotation_tolerance_rad'])[ids];err=float(np.max(np.abs(wm-recomputed)));assert err<2e-5
            errors[env+'_lewm_score_reproduction']=max(errors.get(env+'_lewm_score_reproduction',0.),err)
            assert np.max(np.abs(old.object_error.to_numpy()-g.trajectory_E_t8.to_numpy()))<2e-5
            score_map={'dinowm_retrained':g.predicted_rms.to_numpy(),'lewm_global':wm,'controller':ctrl,'action_norm':g.norm.to_numpy()}
            for method,score in score_map.items():
                active=g.active.to_numpy(bool);x=score[active];y=g.trajectory_E_t8.to_numpy()[active];end=g.endpoint_pose_E_t5.to_numpy()[active];labels=y>1
                if not np.isfinite(x).all():raise ValueError((env,meta['state_id'],method,'nonfinite score'))
                auc=float(roc_auc_score(labels,x)) if len(np.unique(labels))==2 else None
                records.append({'environment':env,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'method':method,'active_candidates':int(active.sum()),'endpoint_rho':rho(x,end),'trajectory_rho':rho(x,y),'violation_auc':auc})
                for i,v in zip(ids,score):joined.append({'environment':env,'state_id':meta['state_id'],'candidate':int(i),'method':method,'score':float(v)})
    s=pd.DataFrame(records);summary=[];paired=[]
    for (env,method),g in s.groupby(['environment','method']):summary.append({'environment':env,'method':method,**{k:mean_ci(g.dropna(subset=[k]),k) for k in ['endpoint_rho','trajectory_rho','violation_auc']}})
    for env,g in s.groupby('environment'):
        for baseline in ['controller','lewm_global','action_norm']:
            x=g[g.method=='dinowm_retrained'].merge(g[g.method==baseline],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for k in ['endpoint_rho','trajectory_rho','violation_auc']:x[k+'_delta']=x[k]-x[k+'_base']
            paired.append({'environment':env,'method':'dinowm_retrained','baseline':baseline,**{k:mean_ci(x.dropna(subset=[k+'_delta']),k+'_delta') for k in ['endpoint_rho','trajectory_rho','violation_auc']}})
    s.to_csv(OUT/'state_metrics.csv',index=False);pd.DataFrame(joined).to_csv(OUT/'scores.csv',index=False)
    write_json(OUT/'analysis.json',{'summary':summary,'paired':paired,'scope':m['scope']})
    write_json(OUT/'raw_audit.json',{'states':21,'state_method_rows':len(s),'candidate_method_rows':len(joined),'max_error':errors,'manifest_sha256':sha256(OUT/'manifest.json'),'parent_inputs':{name:sha256(PARENT/name) for name in ['manifest.json','raw_audit.json','evaluation.csv']}})
    print('AUDITED matched DINO rankings',len(s),'state-method rows',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int);args=p.parse_args();m=freeze();start=time.monotonic()
    if args.dependency_pid:
        while not (PARENT/'raw_audit.json').exists():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Effect pilot ended before full audit; inspect failure before continuing')
            if time.monotonic()-start>3600:raise RuntimeError('Observation window elapsed; revalidate process')
            time.sleep(5)
    analyze(m)
if __name__=='__main__':main()
