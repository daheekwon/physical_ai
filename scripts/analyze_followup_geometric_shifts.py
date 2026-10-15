"""Audit dense outcomes and reproduce prior baseline controls before shift contrasts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import controller_rollout
from zj_wam.diagnostics import polynomial_basis
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    base=resolve('artifacts/followup_v1');out=base/'geometric_shifts';manifest=json.loads((out/'manifest.json').read_text());cfg=json.loads(resolve('configs/followup_v1.json').read_text());eps=cfg['tolerances'];data=ExpertDataset(cfg['dataset']);rules=json.loads((base/'strong_baselines/calibrated_settings.json').read_text())['rules'];prior_selected=pd.read_csv(base/'strong_baselines/selected.csv');frames=[];maxima={};counts={};controls=0
    for p,h in manifest['signature'].items():assert sha256(resolve(p))==h,p
    def check(name,a,b,tol=1e-7):
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maxima[name]=max(maxima.get(name,0),error);assert error<=tol,(name,error)
    def jerk(trace):return np.sum((np.diff(trace[:,:2],n=3,axis=0)/.01**3)**2)*.01
    for split in ['development','calibration']:
        paths=sorted(out.glob(split+'*h05.json'));counts[split]=len(paths)
        if not args.partial:assert len(paths)==105,(split,len(paths))
        for path in paths:
            meta=json.loads(path.read_text());assert meta['signature']==manifest['signature'];f=pd.read_csv(path.with_suffix('.csv'));assert len(f)==40;frames.append(f)
            oldpath=base/'shift_evaluation'/(meta['state_id']+'.json');oldframe=pd.read_csv(oldpath.with_suffix('.csv'))
            with np.load(path.with_suffix('.npz')) as a,np.load(oldpath.with_suffix('.npz')) as old,np.load((base/'strong_baselines'/path.name).with_suffix('.npz')) as strong:
                for r in f.itertuples():
                    name=r.condition;key=name+'_'+r.method;actual=a[key+'_physics'];reference=a[name+'_nominal_physics'];action=a[key+'_actions'];nominal=a[name+'_nominal_actions'];suffix=a[name+'_suffix'];root=a[name+'_root'];past=a[name+'_past']
                    check('reference_reproduces_prior_shift',reference,old[name+'_nominal_physics'],1e-9);check('prefix_equal',actual[:30],reference[:30],1e-9)
                    delta=actual[30:,2:5]-reference[30:,2:5];pos=np.linalg.norm(delta[:,:2],axis=1).max();rot=np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2]))).max();loss=np.maximum(0,a[name+'_nominal_coverage']-a[key+'_coverage']).max();E=max(pos/eps['position'],rot/eps['rotation'],loss/eps['coverage_loss']);check('object_error',E,r.object_error);check('preserved',float(E<=1),float(r.preserved),0.)
                    j0=jerk(reference);j=jerk(actual);check('jerk_gain',(j0-j)/j0 if j0>1e-8 else 0.,r.jerk_gain)
                    norm=np.linalg.norm((action-nominal)/data.scaler.scale_);check('action_norm',norm,r.actual_norm);assert norm<=rules[r.method]['radius']*np.sqrt(10)+2e-6;assert np.max(np.abs(action))<=1+1e-6
                    pq=controller_rollout(np.concatenate([action,suffix]),root[:2],root[5:7]);pred=jerk(np.concatenate([reference[:30,:2],pq]));check('predicted_jerk',pred,r.predicted_selected_jerk,2e-4)
                    if r.method.endswith('polynomial_executed'):check('fixed_boundary_actions',action[[0,4]],nominal[[0,4]])
                    assert bool(r.active)==(norm>1e-6 and r.predicted_gain>1e-8)
                    if r.method=='gaussian_action':
                        oldrow=oldframe[(oldframe.condition==name)&(oldframe.method==r.method)].iloc[0];check('gaussian_previous_action',action,old[name+'_actions'][int(oldrow['index'])]);check('gaussian_previous_physics',actual,old[key+'_physics'],1e-9);controls+=1
                    if name=='nominal':
                        selected=prior_selected[(prior_selected.state_id==r.state_id)&(prior_selected.method==r.method)].iloc[0];check('nominal_previous_strong_action',action,strong['actions'][int(selected['index'])],2e-6)
                        if r.method!='gaussian_action':check('nominal_previous_strong_E',E,selected.object_error,2e-6)
                    if name in ['rgb_noise_5','rgb_noise_15']:check('nonvisual_noise_invariance',action,a['nominal_'+r.method+'_actions'],1e-10)
    f=pd.concat(frames,ignore_index=True);f['joint']=f.active&(f.object_error<=1)&(f.jerk_gain>=.1);f['violation']=f.object_error>1;f['gain_prediction_error']=np.abs(f.jerk_gain-f.predicted_gain);summary=[]
    # Reuse audited previous rows for simultaneous method comparison.
    previous=pd.read_csv(base/'shift_evaluation/evaluation.csv');previous=previous[previous.method!='gaussian_action'];previous=previous.merge(f[['state_id','condition']].drop_duplicates(),on=['state_id','condition'],validate='many_to_one');combined=pd.concat([f,previous],ignore_index=True)
    for (split,condition,method),g in combined.groupby(['split','condition','method']):
        for stratum in ['all','contact']:
            x=g if stratum=='all' else g[g.nominal_contact];summary.append({'split':split,'condition':condition,'method':method,'stratum':stratum,'states':len(x),'joint':mean_ci(x,'joint'),'active':mean_ci(x,'active'),'risk':mean_ci(x[x.active],'violation'),'gain':mean_ci(x,'jerk_gain')})
    paired=[]
    for (split,condition),g in combined.groupby(['split','condition']):
        for baseline in ['vision_adaptive','lewm_global','gaussian_action']:
            b=g[g.method==baseline]
            for method in manifest['methods']:
                if method==baseline:continue
                x=g[g.method==method].merge(b,on=['state_id','initial_pose_family'],validate='one_to_one',suffixes=('','_base'));x['joint_difference']=x.joint.astype(float)-x.joint_base.astype(float);paired.append({'split':split,'condition':condition,'method':method,'baseline':baseline,'joint_difference':mean_ci(x,'joint_difference')})
    write_json(out/'raw_audit.json',{'partial':args.partial,'states':counts,'new_rows':len(f),'gaussian_exact_replication_controls':controls,'max_errors':maxima,'scope':'Source hashes, paired root/reference, raw dense E/jerk, actionnorm/bounds/polynomial endpoints, prior Gaussian exact reproduction in every condition, nominal prior strong candidate reproduction, RGB-only invariance for proprioceptive methods. Coverage uses saved simulator endpoint computation.'})
    write_json(out/'analysis.json',{'partial':args.partial,'summary':summary,'paired':paired,'scope':manifest['scope']});f.to_csv(out/'evaluation.csv',index=False);combined.to_csv(out/'combined_evaluation.csv',index=False)
    print('AUDIT',counts,len(f),'new rows',controls,'exact Gaussian controls',flush=True)
    for r in summary:
        if r['split']=='development' and r['stratum']=='all':print(r['condition'],r['method'],'U',r['joint']['mean'],'R',r['risk']['mean'])
if __name__=='__main__':main()
