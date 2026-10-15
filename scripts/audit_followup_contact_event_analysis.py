"""Independent event effect geometry, readout inference, selected outcomes and intervals."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.audit_retrained_dinowm_readouts import infer
from scripts.audit_followup_dinowm_policy_final import interval
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text());wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi

def main(smoke):
    out=BASE/('contact_events_smoke' if smoke else 'contact_events');protocol=read(out/'protocol.json');cfg=protocol['config'];done=read(out/'analysis_complete.json');analysis=read(out/'analysis.json');assert done['analysis_sha256']==sha256(out/'analysis.json') and done['raw_audit_sha256']==sha256(out/'raw_audit.json');resolution=analysis['resolution'];errors={};checked=0
    for p,h in {**protocol['inputs'],**done['inputs'],**done['helpers']}.items():assert sha256(resolve(p))==h,p
    tables={name:pd.read_csv(out/(name+('.csv.gz' if name=='effects' else '.csv'))) for name in ['effects','state_effects','selected','physical_states','capture_states','paired_effects']};effect=tables['effects'];models={rep:dict(np.load(BASE/'dinowm_readouts/pusht'/f'{rep}__raw__ridge.npz')) for rep in cfg['representations']}
    def check(name,x,y,tol=1e-8):
        np.testing.assert_allclose(x,y,atol=tol,rtol=0,equal_nan=True,err_msg=name);x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);finite=np.isfinite(x)&np.isfinite(y);errors[name]=max(errors.get(name,0),float(np.max(abs(x[finite]-y[finite]))) if finite.any() else 0.)
    for spec in protocol['states']:
        sid=spec['state_id'];p=out/(sid+'.npz');a=dict(np.load(p));z=dict(np.load(p.with_suffix('.decoded.npz')));meta=read(p.with_suffix('.json'));n=len(a['actions']);active=np.flatnonzero(a['norms']>1e-6);true=a['physics'][:,79,2:5];tp=true[:,:2]-true[:1,:2];tr=wrap(true[:,2]-true[0,2]);root=a['physics'][0,29,2:5];dis=a['physics'][:,30:,2:5]-a['physics'][:1,30:,2:5];E=np.maximum.reduce([np.sqrt((dis[:,:,:2]**2).sum(2)).max(1)/2,abs(wrap(dis[:,:,2])).max(1)/np.deg2rad(2),np.maximum(0,a['coverage'][:1]-a['coverage']).max(1)/.02]);cost=(np.diff(a['physics'][:,:,:2],n=3,axis=1)**2).sum((1,2))/.01**5;gain=(cost[0]-cost)/cost[0] if cost[0]>1e-8 else np.zeros(n)
        delta=((a['actions']-a['actions'][:1])/a['input_scale']).reshape(n,-1);norm=np.linalg.norm(delta,axis=1);wrong=np.full(n,-1,dtype=int)
        for i in active:
            ratio=norm/norm[i];den=norm*norm[i];cos=np.divide(delta@delta[i],den,out=np.ones(n),where=den>1e-12);valid=(norm>1e-6)&(np.arange(n)!=i)&(ratio>=cfg['wrong_action_norm_ratio'][0])&(ratio<=cfg['wrong_action_norm_ratio'][1])&(cos<=cfg['wrong_action_max_cosine']);ids=np.flatnonzero(valid)
            if len(ids):wrong[i]=ids[np.argmin(cos[ids])]
        for rep,model in models.items():
            def feature(values,current=False):
                if rep=='lewm_frame':return a['history'][-1:] if current else values
                if current:return a['history'].reshape(1,-1)
                return np.concatenate([np.tile(a['history'][1:].reshape(1,-1),(len(values),1)),values],axis=1)
            real=infer(model,feature(a['actual']).astype(float));pred=infer(model,feature(a['predicted']).astype(float));current=infer(model,feature(None,True).astype(float));check('readout_real',real,z[rep+'__decoded_real']);check('readout_predicted',pred,z[rep+'__decoded_predicted']);check('readout_current',current,z[rep+'__decoded_current']);check('true_dp',tp,z[rep+'__true_dp']);check('true_dr',tr[:,None],z[rep+'__true_dr']);np.testing.assert_array_equal(wrong,z[rep+'__wrong_index'])
            def errors_for(y,indices=None):
                position=y[:,:2];angle=np.arctan2(y[:,2],y[:,3]);dp=position-position[:1];dr=wrap(angle-angle[0])
                if indices is not None:position=position[indices];angle=angle[indices];dp=dp[indices];dr=dr[indices]
                def cosine(x,y):
                    denom=np.sqrt((x*x).sum(1)*(y*y).sum(1));return np.divide((x*y).sum(1),denom,out=np.full(n,np.nan),where=denom>1e-12)
                return {'position_effect_error':np.sqrt(((dp-tp)**2).sum(1)),'rotation_effect_error':abs(wrap(dr-tr)),'position_absolute_error':np.sqrt(((position-true[:,:2])**2).sum(1)),'rotation_absolute_error':abs(wrap(angle-true[:,2])),'position_cosine':cosine(dp,tp),'rotation_cosine':cosine(dr[:,None],tr[:,None]),'predicted_position_magnitude':np.linalg.norm(dp,axis=1),'predicted_rotation_magnitude':abs(dr)}
            controls={'real_encoded':errors_for(real),'predicted':errors_for(pred),'zero_effect':errors_for(np.repeat(real[:1],n,axis=0)),'persistence':errors_for(np.repeat(current,n,axis=0)),'wrong_action':errors_for(pred,np.maximum(wrong,0))}
            for val in controls['wrong_action'].values():val[wrong<0]=np.nan
            for method,vals in controls.items():
                group=effect[(effect.state_id==sid)&(effect.representation==rep)&(effect.method==method)]
                for _,g in group.groupby(['event_id','role']):
                    ids=g.candidate.to_numpy(int);np.testing.assert_array_equal(ids,active);check('physical_E',g.physical_E,E[ids]);check('physical_gain',g.jerk_gain,gain[ids]);check('true_position_magnitude',g.true_position_magnitude,np.linalg.norm(tp,axis=1)[ids]);check('true_rotation_magnitude',g.true_rotation_magnitude,abs(tr)[ids])
                    for key,v in vals.items():check('effect_'+key,g[key],v[ids])
                    checked+=len(g)
            g=tables['capture_states'];g=g[(g.state_id==sid)&(g.representation==rep)];check('capture_position',g.current_position_error,np.full(len(g),np.linalg.norm(current[0,:2]-root[:2])));check('capture_rotation',g.current_rotation_error,np.full(len(g),abs(wrap(np.arctan2(current[0,2],current[0,3])-root[2]))))
        for row in tables['selected'][tables['selected'].state_id==sid].itertuples():
            i=meta['selected_before_physics'][row.method];assert row.candidate==i and row.active==bool(a['norms'][i]>1e-6);check('selected_E',row.object_error,E[i]);check('selected_gain',row.jerk_gain,gain[i]);assert row.violation==bool(E[i]>1) and row.joint==bool(row.active and E[i]<=1 and gain[i]>=.1)
        contact=a['contacts']>0;mask=contact[active]!=contact[0];has=(contact.sum(1)>0);first=np.where(has,contact.argmax(1)*.01+.01,np.nan);shift=abs(first-first[0]);q=a['physics'][active,30:,:2]-a['physics'][0,30:,:2];expected={'active_candidates':len(active),'preserved_fraction':float((E[active]<=1).mean()),'useful_fraction':float(((E[active]<=1)&(gain[active]>=.1)).mean()),'has_useful_candidate':bool(np.any((E[active]<=1)&(gain[active]>=.1))),'contact_mask_disagreement':float(mask.mean()),'contact_existence_flip_fraction':float((has[active]!=has[0]).mean()),'first_contact_shift_mean_seconds':float(np.nanmean(shift[active])) if np.isfinite(shift[active]).any() else np.nan,'first_contact_shift_support':float((has[active]&has[0]).mean()),'pusher_delta_rms':float(np.sqrt((q*q).sum(2).mean(1)).mean())}
        g=tables['physical_states'];g=g[g.state_id==sid]
        for key,value in expected.items():check('physical_state_'+key,g[key],np.full(len(g),value))
    keys=['event_id','event_type','role','state_id','initial_pose_family','representation','method'];state=tables['state_effects'].set_index(keys)
    for key,g in effect.groupby(keys,sort=False):
        row=state.loc[key];good=np.isfinite(g.position_effect_error)&np.isfinite(g.rotation_effect_error);v=g[good];assert row.active_candidates==len(g) and row.valid_candidates==len(v);check('control_support',row.control_support,len(v)/len(g));rep=key[-2]
        for axis in ['position','rotation']:
            resolved=v['true_'+axis+'_magnitude']>resolution[rep][axis];direction=resolved&np.isfinite(v[axis+'_cosine']);metrics={axis+'_effect_error':v[axis+'_effect_error'].mean(),axis+'_absolute_error':v[axis+'_absolute_error'].mean(),axis+'_resolvable_fraction':resolved.mean() if len(v) else np.nan,axis+'_direction_support':direction.mean() if len(v) else np.nan,axis+'_direction_agreement':(v.loc[direction,axis+'_cosine']>=0).mean() if direction.any() else np.nan}
            for metric,value in metrics.items():check('state_'+metric,row[metric],value)
    pairs=tables['paired_effects'].set_index(keys[:-1]+['control'])
    for key,g in effect.groupby(keys[:-1],sort=False):
        pred=g[g.method=='predicted'].set_index('candidate')
        for control in ['real_encoded','zero_effect','persistence','wrong_action']:
            c=g[g.method==control].set_index('candidate');valid=np.isfinite(c.position_effect_error)&np.isfinite(c.rotation_effect_error);row=pairs.loc[(*key,control)];assert row.matched_candidates==int(valid.sum())
            for axis in ['position','rotation']:check('paired_'+axis,row[axis+'_difference'],(pred.loc[valid,axis+'_effect_error']-c.loc[valid,axis+'_effect_error']).mean())
    definitions={'selection_summary':('selected',['event_type','role','method']),'physical_summary':('physical_states',['event_type','role']),'capture_summary':('capture_states',['event_type','role','representation']),'effect_summary':('state_effects',['event_type','role','representation','method']),'paired_effect_summary':('paired_effects',['event_type','role','representation','control'])}
    def compare(expected,actual):
        assert expected['families']==actual['families']
        if expected['mean'] is None:assert actual['mean'] is None and actual['ci95'] is None
        else:check('summary_mean',expected['mean'],actual['mean']);check('summary_CI',expected['ci95'],actual['ci95'])
    for name,(table,groupkeys) in definitions.items():
        for row in analysis[name]:
            g=tables[table]
            for k in groupkeys:g=g[g[k]==row[k]]
            assert len(g)==row['cases']
            for metric,value in row.items():
                if not isinstance(value,dict):continue
                v=g[g.active] if metric=='selected_risk' else g;column='violation' if metric=='selected_risk' else metric;compare(interval(v.dropna(subset=[column]),column),value)
    for row in analysis['paired_phase_errors']:
        g=tables['state_effects'];g=g[(g.representation==row['representation'])&(g.method==row['method'])&(g.event_type==row['event_type'])];v=g[g.role=='transition'].merge(g[g.role==row['reference']],on=['event_id','initial_pose_family'],suffixes=('','_reference'),validate='one_to_one');v['delta']=v[row['metric']]-v[row['metric']+'_reference'];compare(interval(v.dropna(subset=['delta']),'delta'),row)
    assert checked==len(effect)==analysis['effect_rows'];assert len(tables['selected'])==analysis['selection_rows'];write_json(out/'analysis_raw_audit.json',{'smoke':smoke,'states':analysis['states'],'cases':analysis['cases'],'families':analysis['families'],'effect_rows':checked,'selected_rows':analysis['selection_rows'],'max_errors':errors,'analysis_sha256':sha256(out/'analysis.json'),'analysis_complete_sha256':sha256(out/'analysis_complete.json'),'source_sha256':sha256(Path(__file__)),'helpers':{p:sha256(resolve(p)) for p in ['scripts/audit_retrained_dinowm_readouts.py','scripts/audit_followup_dinowm_policy_final.py']},'scope':'All decoded features, physical-unit effects and controls, support/resolution gates, selected/physical outcomes, state reductions and family intervals independently recalculated. Raw audit separately covers allnativeencodes/predictions and freshphysics/contacts.'});print('AUDITED CONTACT EFFECTS',analysis['states'],checked,analysis['selection_rows'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
