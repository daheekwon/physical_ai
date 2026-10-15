"""Independent augmented-LS, recurrence, physical metric and family-CI audit."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_experiment import fit_response
from scripts.audit_followup_dinowm_policy_final import interval
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_causal_arx';read=lambda p:json.loads(p.read_text())
ERRORS={}
def check(name,x,y,atol=1e-8,rtol=0):
    a=np.asarray(x,float);b=np.asarray(y,float);np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,equal_nan=True,err_msg=name)
    d=abs(a-b);ERRORS[name]=max(ERRORS.get(name,0.),float(np.nanmax(d)) if d.size and np.isfinite(d).any() else 0.)
def statistic(actual,frame,column):
    expected=interval(frame,column);assert actual['families']==expected['families']
    if expected['mean'] is None:assert actual['mean'] is None and actual['ci95'] is None
    else:check('family_mean',actual['mean'],expected['mean']);check('family_ci95',actual['ci95'],expected['ci95'])
def independent_fit(p,t,variant,cfg):
    p=p[-251:];t=t[-250:];g,res=fit_response(p,t,.002)
    A=np.diag(1-.002*g[:,1]);B=np.diag(.002**2*g[:,0]);c=np.zeros(4)
    if variant=='past_pd':return A,B,c,g,res,[]
    increments=p[1:]-p[:-1];error=t[1:]-p[1:-1];error[:,3]=np.angle(np.exp(1j*error[:,3]));X=np.column_stack((increments[:-1],error));Y=increments[1:];records=[]
    groups=[([i,i+4],[i]) for i in range(4)] if variant=='diagonal_arx' else [(list(range(8)),list(range(4)))]
    for columns,outputs in groups:
        x=X[:,columns];y=Y[:,outputs];std=np.std(x,axis=0);xm=x.mean(0);ym=y.mean(0);ratio=0.;rho=None;accepted=False;reason='unexcited'
        if np.min(std)>=cfg['minimum_feature_std']:
            z=(x-xm)/std;s=np.linalg.svd(z,compute_uv=False);ratio=float(s[-1]/s[0]);reason='rank_deficient'
            if ratio>=cfg['minimum_singular_ratio']:
                aug=np.vstack((z,np.sqrt(cfg['ridge'])*np.eye(len(columns))));rhs=np.vstack((y-ym,np.zeros((len(columns),len(outputs)))))
                coef=(np.linalg.lstsq(aug,rhs,rcond=None)[0]/std[:,None]).T;intercept=ym-coef@xm
                aa=A.copy();bb=B.copy();cc=c.copy()
                for row,axis in enumerate(outputs):
                    aa[axis]=0;bb[axis]=0
                    for j,col in enumerate(columns):
                        if col<4:aa[axis,col]=coef[row,j]
                        else:bb[axis,col-4]=coef[row,j]
                    cc[axis]=intercept[row]
                qa=aa[np.ix_(outputs,outputs)];qb=bb[np.ix_(outputs,outputs)];n=len(outputs)
                transition=np.zeros((2*n,2*n));transition[:n,:n]=np.eye(n)-qb;transition[:n,n:]=qa;transition[n:,:n]=-qb;transition[n:,n:]=qa
                rho=float(max(abs(np.linalg.eigvals(transition))));reason='unstable'
                if np.isfinite(coef).all() and rho<=cfg['maximum_spectral_radius']:A,B,c=aa,bb,cc;accepted=True;reason='accepted'
        records.append({'accepted':accepted,'reason':reason,'minimum_std':float(std.min()),'singular_ratio':ratio,'spectral_radius':rho,'outputs':outputs,'columns':columns})
    return A,B,c,g,res,records

def recurrence(actions,past,A,B,c,bounds):
    position=past[-1].copy();increment=past[-1]-past[-2];out=[]
    for action in actions:
        target=position+action[:4]*np.array([.05,.05,.05,.3]);target[:3]=np.minimum(np.maximum(target[:3],bounds[0]),bounds[1])
        for j in range(25):
            e=target-position;e[3]=np.angle(np.exp(1j*e[3]));increment=np.einsum('ij,j->i',A,increment)+np.einsum('ij,j->i',B,e)+c;position=position+increment;out.append(position.copy())
    return np.asarray(out)
def energy(q):
    d=q[:,3:]-3*q[:,2:-1]+3*q[:,1:-2]-q[:,:-3];return np.einsum('nki,nki->n',d,d)/.002**5

def main():
    lock=read(OUT/'protocol.json');cfg=lock['config'];done=read(OUT/'execution_complete.json');analysis=read(OUT/'analysis.json');assert done['analysis_sha256']==sha256(OUT/'analysis.json') and done['protocol_sha256']==sha256(OUT/'protocol.json')
    for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
    assert cfg['history_physics_steps']==250 and cfg['physics_dt']==.002 and cfg['substeps']==25
    rules=read(BASE/'cube/proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules'];selected=pd.read_csv(OUT/'selected.csv');fidelity=pd.read_csv(OUT/'fidelity.csv');inputs={};candidate_count=0;prediction_count=0;choices=0
    for path in lock['sources']:
        p=resolve(path);meta=read(p);raw=dict(np.load(p.with_suffix('.npz')));savedpath=OUT/p.with_suffix('.npz').name;saved=dict(np.load(savedpath));m=read(savedpath.with_suffix('.json'));assert m['data_sha256']==sha256(savedpath) and m['protocol_sha256']==sha256(OUT/'protocol.json');past=raw['input_past_points'];n=len(raw['actions']);candidate_count+=n
        actualq=np.concatenate((np.broadcast_to(raw['prefix_eef_position'],(n,*raw['prefix_eef_position'].shape)),raw['candidate_eef_position']),axis=1);actualcost=energy(actualq);actualgain=(actualcost[0]-actualcost)/max(actualcost[0],1e-12)
        pose=raw['candidate_object_pose'];dp=np.linalg.norm(pose[:,:,:3]-pose[:1,:,:3],axis=2).max(1);quat=pose[:,:,3:7];quat=quat/np.linalg.norm(quat,axis=2,keepdims=True);dots=abs(np.sum(quat*quat[:1],axis=2));angles=2*np.arccos(np.clip(dots,0,1));E=np.maximum(dp/.002,angles.max(1)/np.deg2rad(2));truth=pd.read_csv(p.with_suffix('.csv'));check('physical_jerk',actualcost,truth.eef_jerk,1e-2,1e-8);check('physical_gain',actualgain,truth.jerk_gain);check('physical_object',E,truth.object_error,2e-5);assert np.array_equal(E<=1,truth.object_error.to_numpy()<=1)
        active=raw['norm']>1e-6
        for variant in cfg['variants']:
            A,B,c,g,res,records=independent_fit(past,raw['input_past_targets'],variant,cfg)
            for key,value in [('A',A),('B',B),('c',c),('gains',g),('residual',res)]:check('fit_'+key,value,saved[variant+'_'+key],1e-7)
            recorded=m['models'][variant];assert len(records)==len(recorded['records']);support=sum(len(x['outputs']) for x in records if x['accepted']);assert support==recorded['supported_axes']
            for x,y in zip(records,recorded['records']):
                for key in ['accepted','reason','outputs','columns']:assert x[key]==y[key],(p,key,x,y)
                for key in ['minimum_std','singular_ratio','spectral_radius']:
                    if x[key] is None:assert y[key] is None
                    else:check('fit_'+key,x[key],y[key],1e-7)
            pred=np.array([recurrence(np.concatenate((a,raw['input_suffix_actions'])),past,A,B,c,raw['input_workspace_bounds']) for a in raw['actions']]);prediction_count+=n;check('predicted_trace',pred,saved[variant+'_predicted'],2e-8)
            cost=energy(np.concatenate((np.broadcast_to(past[-75:,:3],(n,75,3)),pred[:,:,:3]),axis=1));check('expected_cost',cost,saved[variant+'_cost'],1e-2,1e-7);gain=(cost[0]-cost)/max(cost[0],1e-12);check('expected_gain',gain,saved[variant+'_gain'],1e-7)
            effect=pred[:,:,:3]-pred[:1,:,:3]-(raw['candidate_eef_position']-raw['candidate_eef_position'][:1]);corr=spearmanr(gain[active],actualgain[active]).statistic if np.ptp(gain[active])>0 and np.ptp(actualgain[active])>0 else np.nan
            values={'supported_axes':support,'gain_rank_spearman':corr,'gain_mean_absolute_error':np.mean(abs(gain[active]-actualgain[active])),'candidate_effect_position_RMSE_m':np.sqrt(np.mean(np.sum(effect[active]**2,axis=2))),'nominal_position_RMSE_m':np.sqrt(np.mean(np.sum((pred[0,:,:3]-raw['candidate_eef_position'][0])**2,axis=1))),'nominal_relative_jerk_error':abs(cost[0]-actualcost[0])/max(actualcost[0],1e-12)}
            record=fidelity[(fidelity.state_id==p.stem)&(fidelity.variant==variant)];assert len(record)==1
            for key,value in values.items():check('fidelity_'+key,value,record.iloc[0][key],1e-7)
            # Independently verified saved arithmetic preserves exact threshold/tie semantics.
            cost=saved[variant+'_cost'];gain=saved[variant+'_gain']
            for verifier in ['controller_displacement','proprio_contact','lewm_global','true_preservation']:
                eligible=[]
                for i in range(n):
                    if not active[i] or cost[i]>=cost[0]-1e-10:continue
                    if verifier=='true_preservation':allow=E[i]<=1
                    else:
                        rule=rules[verifier];allow=not rule.get('disabled',False) and raw['norm'][i]<=rule.get('radius',.4)+1e-6 and raw[verifier+'_scores'][i]<=rule.get('tau',-np.inf)
                    if allow:eligible.append(i)
                chosen=min(eligible,key=lambda i:(cost[i],i)) if eligible else 0
                x=selected[(selected.state_id==p.stem)&(selected.variant==variant)&(selected.verifier==verifier)];assert len(x)==1;x=x.iloc[0];assert chosen==x.candidate and x.initial_pose_family==meta['seed'] and x.phase==meta['phase'];assert bool(x.active)==bool(active[chosen]) and bool(x.violation)==bool(E[chosen]>1) and bool(x.joint)==bool(active[chosen] and E[chosen]<=1 and actualgain[chosen]>=.1);check('selected_E',x.object_error,E[chosen],2e-5);check('selected_gain',x.jerk_gain,actualgain[chosen]);check('selected_predicted_gain',x.predicted_gain,gain[chosen]);choices+=1
        for q in [savedpath,savedpath.with_suffix('.json')]:inputs[str(q.relative_to(resolve('.')))]=sha256(q)
        print('AUDIT ARX',p.stem,flush=True)
    for x in analysis['summary']:
        f=selected[(selected.variant==x['variant'])&(selected.verifier==x['verifier'])]
        if x['phase']!='all':f=f[f.phase==x['phase']]
        assert len(f)==x['states']
        for key in ['active','joint','jerk_gain','predicted_gain']:statistic(x[key],f,key)
        statistic(x['selected_risk'],f[f.active],'violation')
    for x in analysis['fidelity']:
        f=fidelity[fidelity.variant==x['variant']]
        for key in x:
            if key!='variant':statistic(x[key],f.dropna(subset=[key]),key)
    for x in analysis['paired']:
        f=selected[(selected.variant==x['variant'])&(selected.verifier==x['verifier'])].set_index('state_id');base=selected[(selected.variant=='past_pd')&(selected.verifier==x['verifier'])].set_index('state_id').loc[f.index];v=f.reset_index();v['delta']=f[x['metric']].to_numpy(float)-base[x['metric']].to_numpy(float);statistic(x,v,'delta')
    assert len(lock['sources'])==48 and candidate_count==899 and prediction_count==2697 and choices==len(selected)==576 and len(fidelity)==144
    for p in [OUT/'analysis.json',OUT/'selected.csv',OUT/'fidelity.csv',OUT/'protocol.json',OUT/'execution_complete.json']:inputs[str(p.relative_to(resolve('.')))]=sha256(p)
    write_json(OUT/'raw_audit.json',{'states':48,'families':12,'candidates':candidate_count,'independent_model_candidate_rollouts':prediction_count,'selections':choices,'fidelity_rows':len(fidelity),'max_errors':ERRORS,'inputs':inputs,'source_sha256':sha256(Path(__file__)),'audit_helpers':{p:sha256(resolve(p)) for p in ['zj_wam/cube_experiment.py','scripts/audit_followup_dinowm_policy_final.py']},'scope':'Independent augmented least squares vs normal equations for ARX; recurrence for all variants/candidates, dense SO3 geometry and actual XYZ jerk, every selected outcome and family CI. PastPD fitting reused existing audited implementation. No new simulation needed: parent physical bank already audited. Development diagnosis only.'});print('AUDITED CAUSAL ARX',candidate_count,choices,flush=True)
if __name__=='__main__':main()
