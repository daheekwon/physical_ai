"""Audit actual staged selector records against independent choice/physics equations."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_experiment import fit_response
from scripts.analyze_followup_response_selection import mean_ci
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main(split):
    out=BASE/'cube_adaptation_factorial'/split;protocol=read(out/'protocol.json');cfg=protocol['config'];settings=protocol['settings'];rule=settings['base_rule'];done=read(out/'execution_complete.json');assert done['protocol_sha256']==sha256(out/'protocol.json') and done['split']==split;errors={};rows=[];inputs={};count=0
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    def check(name,x,y,tol=0,rtol=0):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(name,x.shape,y.shape);np.testing.assert_allclose(x,y,atol=tol,rtol=rtol,equal_nan=True,err_msg=name);d=abs(x.astype(float)-y.astype(float));errors[name]=max(errors.get(name,0.),float(np.nanmax(d)) if d.size and np.isfinite(d).any() else 0.)
    for state_index,source in enumerate(protocol['sources']):
        src=resolve(source);sm=read(src);condition='nominal' if split=='nominal' else sm['condition'];path=out/condition/src.name;meta=read(path);assert meta['source_data_sha256']==sha256(src.with_suffix('.npz')) and meta['data_sha256']==sha256(path.with_suffix('.npz')) and meta['protocol_sha256']==sha256(out/'protocol.json');raw=dict(np.load(src.with_suffix('.npz')));saved=dict(np.load(path.with_suffix('.npz')));frame=pd.read_csv(path.with_suffix('.csv')).set_index('mode');assert len(frame)==6;offset=state_index%6;expected_order=cfg['methods'][offset:]+cfg['methods'][:offset];assert meta['execution_order']==expected_order
        p=raw['input_past_points'];targets=raw['input_past_targets'];gains,_=fit_response(p[:126],targets[:125],.002);state=p[125].copy();d=p[125]-p[124];pred=[]
        for target in targets[125:]:
            e=target-state;e[3]=np.angle(np.exp(1j*e[3]));d=(1-.002*gains[:,1])*d+.002**2*gains[:,0]*e;state=state+d;pred.append(state.copy())
        delta=np.array(pred)-p[126:];components=np.array([np.linalg.norm(delta[:,:3],axis=1).max()/.002,np.abs(np.angle(np.exp(1j*delta[:,3]))).max()/np.deg2rad(2)]);past_error=float(components.max());cost=raw['costs'];norm=raw['norm'];score=raw['proprio_contact_scores'];gain=(cost[0]-cost)/max(cost[0],1e-12);table=pd.DataFrame({'candidate':np.arange(len(cost)),'cost':cost,'norm':norm,'score':score,'gain':gain,'stage1':np.isin(raw['labels'][:,0],['nominal','past_pd_optimizer'])});pose=raw['candidate_object_pose'];r0=Rotation.from_quat(pose[0][:,[4,5,6,3]]);J=[]
        for trace in raw['candidate_eef_position']:
            q=np.r_[raw['prefix_eef_position'],trace];dd=q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3];J.append(float((dd*dd).sum()/.002**5))
        J=np.array(J)
        for mode in cfg['methods']:
            dec=meta['decisions'][mode];ids=np.array(dec['source_candidate_indices'],int);assert ids[0]==0 and np.all(np.diff(ids)>0);q=lambda k:saved[mode+'__'+k]
            for k in ['actions','coefficients','costs','norm','predicted_robot','solver_success','solver_iterations']:check('source_'+k,q(k),raw[k][ids])
            check('source_scores',q('scores'),raw['proprio_contact_scores'][ids]);np.testing.assert_array_equal(q('labels'),raw['labels'][ids]);check('source_optimizer_diagnostics',q('optimizer_diagnostics'),raw['optimizer_diagnostics']);check('basis',q('basis'),raw['basis'])
            radius=.4/(1+settings['alpha']*min(past_error,10)) if mode in ['allowance','both'] else (settings['constant_radius'] if mode=='constant_shrink' else .4);check('radius',radius,dec['radius'],1e-12)
            if mode in ['allowance','both']:check('past_error',past_error,dec['past_error'],1e-8);check('past_error_components',components,dec['past_error_components'],1e-8)
            else:assert dec['past_error'] is None and dec['past_error_components'] is None
            eligible=table[(table.norm>1e-6)&(table.norm<=radius+1e-6)&(table.score<=rule['tau'])&(table.gain>1e-8)] if rule['enabled'] else table.iloc[:0];first=eligible[eligible.stage1].sort_values(['cost','norm','candidate']);fi=int(first.iloc[0].candidate) if len(first) else 0;early=bool(mode in ['compute','both'] and fi>0 and gain[fi]>=.1 and gain[fi]>=.9*table[table.stage1].gain.max());available=table[table.stage1] if mode=='cheap_only' or early else table;np.testing.assert_array_equal(ids,available.candidate.to_numpy());allowed=eligible[eligible.stage1] if mode=='cheap_only' or early else eligible;ranked=allowed.sort_values(['cost','norm','candidate']);j=int(ranked.iloc[0].candidate) if len(ranked) else 0;assert j==dec['original_candidate']==int(ids[dec['selected']]);assert early==dec['early_stop'] and dec['scored_candidates']==len(ids) and dec['solver_calls']==5 and dec['stages']==(1 if mode=='cheap_only' or early else 2);assert dec['solver_iterations']==int(q('optimizer_diagnostics')[:,2].sum());assert all(x>=0 for x in dec['timing'].values()) and sum(dec['timing'].values())<=dec['seconds']+1e-8
            pos=float(np.linalg.norm(pose[j,:,:3]-pose[0,:,:3],axis=1).max());rot=float((Rotation.from_quat(pose[j][:,[4,5,6,3]])*r0.inv()).magnitude().max());E=max(pos/.002,rot/np.deg2rad(2));actual=(J[0]-J[j])/max(J[0],1e-12);active=bool(norm[j]>1e-6);stored=frame.loc[mode];check('physical_object_error',E,stored.object_error,2e-5);check('physical_jerk_gain',actual,stored.jerk_gain,1e-8);assert stored.candidate==j and bool(stored.active)==active and bool(stored.violation)==bool(active and E>1) and bool(stored.joint)==bool(active and E<=1 and actual>=.1)
            for key in ['seconds','radius','alpha','stages','early_stop','scored_candidates','solver_calls','solver_iterations','predicted_gain']:check('export_'+key,stored[key],dec[key],1e-8)
            for key,value in dec['timing'].items():check('component_time_'+key,stored['time_'+key],value,1e-8)
            rows.append(stored.to_dict()|{'mode':mode});count+=1
        for p0 in [path,path.with_suffix('.npz'),path.with_suffix('.csv')]:inputs[str(p0.relative_to(resolve('.')))]=sha256(p0)
        print('AUDIT CUBE FACTORIAL',split,condition,src.stem,count,flush=True)
    f=pd.DataFrame(rows);assert count==done['actual_selector_calls']==6*len(protocol['sources']);summary=[];paired=[]
    def ci(g,k):
        r=mean_ci(g,k)
        if len(g):m,c=cluster(g[k].to_numpy(),g.initial_pose_family.to_numpy());check('independent_mean',m,r['mean'],1e-8);check('independent_ci',c,r['ci95'],1e-8)
        return r
    for (population,condition,mode),g in f.groupby(['split','condition','mode']):
        summary.append({'split':population,'condition':condition,'mode':mode,'states':len(g),**{k:ci(g,k) for k in ['active','joint','jerk_gain','seconds','scored_candidates','solver_calls','early_stop']},'selected_risk':ci(g[g.active],'violation'),'seconds_p50':float(g.seconds.median()),'seconds_p95':float(g.seconds.quantile(.95)),'deadline_exceedance':{str(x):float((g.seconds>x).mean()) for x in cfg['deadline_seconds']}})
    for (population,condition),g in f.groupby(['split','condition']):
        base=g[g['mode']=='fixed']
        for mode in [x for x in cfg['methods'] if x!='fixed']:
            v=g[g['mode']==mode].merge(base,on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for k in ['joint','jerk_gain','seconds','scored_candidates']:v['difference']=v[k].astype(float)-v[k+'_base'].astype(float);paired.append({'split':population,'condition':condition,'mode':mode,'baseline':'fixed','metric':k,**ci(v,'difference')})
    f.to_csv(out/'evaluation.csv',index=False);write_json(out/'analysis.json',{'split':split,'states':len(protocol['sources']),'actual_selector_calls':count,'settings':settings,'summary':summary,'paired':paired,'additional_matched_budget_control_needed':bool(f[f['mode'].isin(['compute','both'])].early_stop.any()),'scope':protocol['scope']});write_json(out/'raw_audit.json',{'split':split,'states':len(protocol['sources']),'actual_selector_calls':count,'max_errors':errors,'inputs':inputs,'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'source_sha256':sha256(Path(__file__)),'helpers':{p:sha256(resolve(p)) for p in ['zj_wam/cube_experiment.py','scripts/analyze_followup_response_selection.py','scripts/audit_retrained_dinowm_readouts.py']},'scope':'Allgeneratedarrays matched auditedsource, independentpast recurrence/radius/stage/selection/solvercounts/timingledger and chosen densephysicalmetrics. Family statistics independentlycrosschecked. Actual walltime notrerun; no freshphysics inthisaudit, sourceallcandidatephysicalbank previouslyaudited.'});print('AUDITED CUBE FACTORIAL',split,count,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['nominal','shifted']);a=p.parse_args();main(a.split)
