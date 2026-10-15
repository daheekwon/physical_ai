"""Cube finite-bank raw audit, frozen calibration and paired-family evaluation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json
from zj_wam.cube_experiment import fit_response,predict_response,predicted_cost,angular_jerk
from scripts.analyze_followup_response_selection import mean_ci


def load(folder,cfg,manifest,partial=False):
    files=sorted(folder.glob('e*.json'));expected=4*len(manifest['splits'].get(folder.name,[]))
    if not partial and folder.name!='pilot':assert len(files)==expected,(folder,len(files),expected)
    frames=[];errors={};states=[]
    def check(name,x,y,tol=1e-7):
        error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),error);assert error<=tol,(name,error)
    scale=np.asarray(manifest['normalization']['scale'])
    for path in files:
        meta=json.loads(path.read_text());assert meta['signature']==manifest['signature'];f=pd.read_csv(path.with_suffix('.csv'))
        with np.load(path.with_suffix('.npz')) as a:
            actions=a['actions'];norm=np.linalg.norm(((actions-a['nominal_actions'])/scale).reshape(len(f),-1),axis=1);check('normalized_action_norm',norm,f.norm);check('gripper_fixed',actions[:,:,4],a['nominal_actions'][:,4]);check('suffix_common_length',len(a['suffix_actions']),cfg['suffix']);check('polynomial_coordinates',((a['coefficients']@a['polynomial_basis'].T).reshape(actions.shape)*scale),actions-a['nominal_actions']);check('endpoint_correction',actions[:,[0,-1]],np.broadcast_to(a['nominal_actions'][[0,-1]],actions[:,[0,-1]].shape));check('raw_bounds',np.maximum(0,np.abs(actions)-1),0.,1e-7)
            p=a['candidate_object_pose'];pos=np.linalg.norm(p[:,:,:3]-p[0,:,:3],axis=-1);q=p[:,:,3:]/np.linalg.norm(p[:,:,3:],axis=-1,keepdims=True);theta=2*np.arccos(np.clip(np.abs(np.sum(q*q[0],axis=-1)),0,1));E=np.maximum(pos.max(1)/cfg['position_tolerance_m'],theta.max(1)/cfg['rotation_tolerance_rad']);check('object_E',E,f.object_error);check('position_max',pos.max(1),f.object_position_max_m);check('rotation_max',theta.max(1),f.object_rotation_max_rad)
            actual=a['candidate_eef_position'];prefix=a['prefix_eef_position'];trace=np.concatenate([np.broadcast_to(prefix,(len(f),*prefix.shape)),actual],axis=1);J=np.sum((np.diff(trace,n=3,axis=1)/cfg['physics_dt']**3)**2,axis=(1,2))*cfg['physics_dt'];check('relative_EEF_jerk',J/np.maximum(1,J),f.eef_jerk.to_numpy()/np.maximum(1,J));gain=(J[0]-J)/J[0] if J[0]>1e-12 else np.zeros(len(f));check('EEF_jerk_gain',gain,f.jerk_gain)
            joints=np.concatenate([np.broadcast_to(a['prefix_qpos'][:,:6],(len(f),75,6)),a['candidate_qpos'][:,:,:6]],axis=1);jj=np.sum((np.diff(joints,n=3,axis=1)/cfg['physics_dt']**3)**2,axis=(1,2))*cfg['physics_dt'];check('relative_arm_jerk',jj/np.maximum(1,jj),f.joint_jerk.to_numpy()/np.maximum(1,jj));angular=np.array([angular_jerk(np.r_[a['prefix_eef_quaternion'],x],cfg['physics_dt']) for x in a['candidate_eef_quaternion']]);check('relative_angular_jerk',angular/np.maximum(1,angular),f.angular_jerk.to_numpy()/np.maximum(1,angular))
            check('physical_timing',np.diff(a['candidate_time'],axis=1),cfg['physics_dt'],1e-12);check('root_aligned_timestamps',a['candidate_time'][:,0],meta['timestep']*cfg['control_dt']+cfg['physics_dt'],1e-10)
            low=a['past_points'];gains,residual=fit_response(low,a['past_targets'],cfg['physics_dt']);check('past_only_response_fit',gains,a['response_gains']);check('past_fit_residual',residual,a['response_fit_rmse']);v=(low[-1]-low[-2])/cfg['physics_dt'];pred=np.array([predict_response(np.r_[x,a['suffix_actions']],low[-1],v,gains,a['workspace_bounds'],cfg['physics_dt'],cfg['substeps']) for x in actions]);check('robot_response_predictions',pred,a['predicted_robot'])
            cost=np.array([predicted_cost(x,low[-75:],cfg['physics_dt'],cfg['yaw_length_scale_m']) for x in pred]);check('relative_predicted_cost',cost/np.maximum(1,cost),f.predicted_jerk_cost.to_numpy()/np.maximum(1,cost));scores=np.linalg.norm(a['predicted_latent']-a['predicted_latent'][0],axis=1);check('finite_lewm_score',scores,f.lewm_score,1e-6)
            delta=pred-pred[:1];position=np.linalg.norm(delta[:,:,:3],axis=-1);angle=np.abs(np.arctan2(np.sin(delta[:,:,3]),np.cos(delta[:,:,3])));controller=np.maximum(position.max(1)/cfg['position_tolerance_m'],angle.max(1)/cfg['rotation_tolerance_rad']);gate=bool(float(a['root_gripper_contact'])>.1 or np.any(np.r_[a['nominal_actions'][:,4],a['suffix_actions'][:,4]]>.1));contact=np.maximum((position+np.sqrt(3)*.02*angle).max(1)/cfg['position_tolerance_m'],angle.max(1)/cfg['rotation_tolerance_rad']) if gate else np.zeros(len(f))
            f['controller_score']=controller;f['contact_score']=contact;f['contact_gate']=gate;f['object_error']=E;f['jerk_gain']=gain;f['active']=norm>1e-6;f['joint']=f.active&(E<=1)&(gain>=cfg['jerk_gain_threshold']);f['violation']=E>1
            prediction_error=np.linalg.norm(pred[0,:,:3]-actual[0],axis=-1)
            states.append({'state_id':meta['state_id'],'episode':meta['episode'],'phase':meta['phase'],'candidates':len(f),'rejected_candidates':len(meta['rejected_candidates']),'nominal_robot_prediction_max_m':float(prediction_error.max()),'nominal_robot_prediction_rms_m':float(np.sqrt(np.mean(prediction_error**2))),'seconds':meta['seconds'],'generation_seconds':meta['generation_seconds'],'contact_gate':gate})
        frames.append(f)
    return pd.concat(frames,ignore_index=True),pd.DataFrame(states),{'states':len(files),'candidate_rows':sum(map(len,frames)),'max_recalculation_error':errors,'scope':'Allsaved500Hzpositions/quaternions andEEF/armjerk,gripper/boundaries/norms,past-onlygainfit,robotpredictions,finiteWMscore. No future physical labels supplied to candidate generator/selector except explicit oracle.'}


def choose(f,method,rule):
    selected=[]
    for state,g in f.groupby('state_id',sort=True):
        nominal=g[g.candidate==0].iloc[0]
        if method=='nominal' or rule.get('disabled',False):r=nominal
        elif method=='oracle':
            eligible=g[(g.object_error<=1)&g.active&(g.jerk_gain>0)];r=eligible.sort_values(['eef_jerk','candidate']).iloc[0] if len(eligible) else nominal
        else:
            score={'lewm_global':'lewm_score','controller_displacement':'controller_score','proprio_contact':'contact_score'}.get(method)
            eligible=g[g.active & (g.norm<=rule.get('radius',.4)+1e-6)]
            if score:eligible=eligible[eligible[score]<=rule['tau']]
            eligible=eligible[eligible.predicted_jerk_cost<nominal.predicted_jerk_cost-1e-10]
            r=eligible.sort_values(['predicted_jerk_cost','candidate']).iloc[0] if len(eligible) else nominal
        d=r.to_dict();d['generator_method']=d['method'];d['method']=method;selected.append(d)
    return pd.DataFrame(selected)


def calibrate(f,cfg):
    rules={};curves=[]
    for method,column in [('controller_displacement','controller_score'),('proprio_contact','contact_score'),('lewm_global','lewm_score')]:
        values=f.loc[f.active,column];thresholds=np.unique(np.r_[0.,np.quantile(values,np.linspace(0,1,41))]);candidates=[]
        for radius in cfg['radii']:
            for tau in thresholds:
                rule={'radius':radius,'tau':float(tau)};x=choose(f,method,rule);active=x[x.active];contact=active[active.phase.isin(['grasp','transport','release'])];risk=float(active.violation.mean()) if len(active) else None;cr=float(contact.violation.mean()) if len(contact) else None;valid=risk is not None and cr is not None and max(risk,cr)<=cfg['target_risk'] and x.active.mean()>=cfg['minimum_active_fraction'];accepted=f.active&(f.norm<=radius+1e-6)&(f[column]<=tau);row={'method':method,**rule,'feasible':bool(valid),'joint':float(x.joint.mean()),'active':float(x.active.mean()),'risk':risk,'contact_risk':cr,'candidate_coverage':float(accepted[f.active].mean()),'candidate_risk':float(f.loc[accepted,'violation'].mean()) if accepted.any() else None};candidates.append(row)
        valid=[x for x in candidates if x['feasible']]
        best=max(valid,key=lambda x:(x['joint'],x['active'],-x['tau'],-x['radius'])) if valid else None
        rules[method]=best if best else {'disabled':True,'reason':'No calibration rule meets selected-risk/contact-risk and active-fraction requirements.'};curves+=candidates
    return {'rules':rules,'curves':curves,'scope':'Calibration families only. Future truth used to fit threshold, never per-state execution selection. Empirical selected-risk criterion, no finite-sample safety guarantee. Samebank/predictedjerk ranking for three verifiers.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['pilot','calibration','evaluate'],required=True);p.add_argument('--partial',action='store_true');args=p.parse_args();cfg=json.loads(resolve('configs/followup_cube.json').read_text());out=resolve(cfg['output']);manifest=json.loads((out/'manifest.json').read_text())
    if args.split in ['pilot','calibration']:
        f,s,a=load(out/args.split,cfg,manifest,args.partial);write_json(out/args.split/'raw_audit.json',a);f.to_csv(out/args.split/'audited_candidates.csv',index=False);s.to_csv(out/args.split/'state_diagnostics.csv',index=False)
        if args.split=='calibration' and not args.partial:lock=calibrate(f,cfg);lock['signature']=manifest['signature'];write_json(out/'calibrated_settings.json',lock)
        oracle=choose(f,'oracle',{});write_json(out/args.split/'oracle_summary.json',{'states':len(oracle),'joint':float(oracle.joint.mean()),'by_phase':oracle.groupby('phase').joint.mean().to_dict(),'scope':'Finitebankoracle only; pilot exploratory if splitpilot.'});print(args.split,a,'oracleU',float(oracle.joint.mean()),flush=True)
    else:
        lock=json.loads((out/'calibrated_settings.json').read_text());assert lock['signature']==manifest['signature'];rows=[];audit={}
        for split in ['development','calibration']:
            f,s,a=load(out/split,cfg,manifest,args.partial);audit[split]=a;s.to_csv(out/split/'state_diagnostics.csv',index=False)
            for method in ['nominal','pd_jerk_only','controller_displacement','proprio_contact','lewm_global','oracle']:
                x=choose(f,method,lock['rules'].get(method,{}));rows.append(x)
        allrows=pd.concat(rows,ignore_index=True);summary=[]
        for (split,method),g in allrows.groupby(['split','method']):
            active=g[g.active];summary.append({'split':split,'method':method,'states':len(g),**{k:mean_ci(g,k) for k in ['joint','active','jerk_gain','drop_relative_to_nominal']},'risk':mean_ci(active,'violation')})
        allrows.to_csv(out/'evaluation.csv',index=False);write_json(out/'analysis.json',{'summary':summary,'locked':lock,'scope':cfg['scope']});write_json(out/'raw_audit.json',audit)
        for r in summary:
            if r['split']=='development':print(r['method'],'U',r['joint']['mean'],'R',r['risk']['mean'],flush=True)
if __name__=='__main__':main()
