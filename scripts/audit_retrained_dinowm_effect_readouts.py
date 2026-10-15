"""Independent raw-array inference, physical effect geometry and summary checks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from scripts.audit_retrained_dinowm_readouts import infer,proper_rotation,symmetries,cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def features(z,h,rep,q,current=False):
    if 'spatial' in rep:
        grid=z.reshape(-1,14,14,384);parts=[grid[:,r:r+7,c:c+7].mean(axis=(1,2)) for r,c in [(0,0),(0,7),(7,0),(7,7)]];x=np.concatenate(parts,axis=1)
    elif 'history' in rep:
        prior=h.mean(axis=1)[0:2 if current else 3] if current else h.mean(axis=1)[1:3]
        x=np.concatenate([np.tile(prior.reshape(1,-1),(len(z),1)),z.mean(axis=1)],axis=1)
    else:x=z.mean(axis=1)
    if 'projected' in rep:x=np.concatenate([x[:,i:i+384].astype(float)@q for i in range(0,x.shape[1],384)],axis=1)
    return x.astype(float)


def pose(env,y,kind):
    pos=y[:,:2] if env=='pusht' else y[:,:3]
    if env=='pusht':return pos,np.arctan2(y[:,2],y[:,3])
    r=proper_rotation(y)
    if kind=='symmetry':
        candidates=r[:,None]@symmetries()[None];scores=np.einsum('nsij,ij->ns',candidates,r[0]);chosen=[]
        for cs,sc in zip(candidates,scores):
            ids=np.flatnonzero(sc>=sc.max()-1e-10);j=min(ids,key=lambda j:tuple(np.round(cs[j].reshape(-1),10)));chosen.append(cs[j])
        r=np.asarray(chosen)
    return pos,r


def delta(env,p):
    pos,r=p;dp=pos-pos[:1]
    return dp,((r-r[0]+np.pi)%(2*np.pi)-np.pi)[:,None] if env=='pusht' else Rotation.from_matrix(r@r[0].T).as_rotvec()


def metric(env,absolute,effect,true_abs,true_effect):
    pos,r=absolute;dp,dr=effect;tp,tr=true_effect;tpos,trot=true_abs
    angle=lambda a,b:np.arccos(np.clip((np.einsum('nij,nij->n',a,b)-1)/2,-1,1))
    if env=='pusht':re=np.abs((dr[:,0]-tr[:,0]+np.pi)%(2*np.pi)-np.pi);ae=np.abs((r-trot+np.pi)%(2*np.pi)-np.pi)
    else:re=angle(Rotation.from_rotvec(dr).as_matrix(),Rotation.from_rotvec(tr).as_matrix());ae=angle(r,trot)
    def cos(x,y):
        n=np.sqrt(np.sum(x*x,axis=1))*np.sqrt(np.sum(y*y,axis=1));return np.divide(np.sum(x*y,axis=1),n,out=np.full(len(n),np.nan),where=n>1e-12)
    return {'position_effect_error':np.linalg.norm(dp-tp,axis=1),'rotation_effect_error':re,'position_absolute_error':np.linalg.norm(pos-tpos,axis=1),'rotation_absolute_error':ae,'position_cosine':cos(dp,tp),'rotation_cosine':cos(dr,tr),'predicted_position_magnitude':np.linalg.norm(dp,axis=1),'predicted_rotation_magnitude':np.linalg.norm(dr,axis=1)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();out=BASE/('dinowm_effect_readouts_smoke' if args.smoke else 'dinowm_effect_readouts');capture=BASE/('dinowm_interventions_smoke' if args.smoke else 'dinowm_interventions');protocol=read(out/'protocol.json');cfg=protocol['config'];done=read(out/'analysis_complete.json');assert done['smoke']==args.smoke and done['analysis_sha256']==sha256(out/'analysis.json')
    for name,h in protocol['inputs'].items():assert sha256(resolve(name))==h,name
    q=np.load(BASE/'dinowm_readouts/projection.npy');errors={};records=0;rows=0;decoded_checks=0;allframes=[]
    def check(name,x,y,tol=1e-7):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);assert x.shape==y.shape,(name,x.shape,y.shape);assert np.array_equal(np.isnan(x),np.isnan(y)),name;ok=np.isfinite(x)&np.isfinite(y);err=float(np.max(np.abs(x[ok]-y[ok]))) if ok.any() else 0.;errors[name]=max(errors.get(name,0.),err);assert err<=tol,(name,err)
    for meta_path in sorted(out.glob('*/*/*.json')):
        m=read(meta_path);records+=1;env=m['environment'];src=resolve(m['source']);old=pd.read_csv(src.with_suffix('.csv'));frame=pd.read_csv(meta_path.with_suffix('.csv.gz'));assert len(frame)==m['rows'];rows+=len(frame);allframes.append(frame[frame.active].copy());ids=old['index' if env=='pusht' else 'candidate'].to_numpy(int)
        for name,h in m['files'].items():assert sha256(meta_path.parent/name)==h,name
        tokens_path=capture/env/m['split']/src.with_suffix('.npz').name;assert m['tokens_sha256']==sha256(tokens_path)
        with np.load(tokens_path) as tokens,np.load(src.with_suffix('.npz')) as parent,np.load(meta_path.with_suffix('.npz')) as saved:
            if env=='pusht':p=parent['physics'][:,79,2:5];true_abs=(p[:,:2],p[:,2])
            else:p=parent['candidate_object_pose'][:,124];true_abs=(p[:,:3],Rotation.from_quat(p[:,[4,5,6,3]]).as_matrix())
            true_delta=delta(env,true_abs)
            for (rep,kind,learner),g in frame.groupby(['representation','target','readout']):
                key=rep+'__'+kind+'__'+learner
                with np.load(BASE/'dinowm_readouts'/env/(key+'.npz')) as raw:model=dict(raw)
                absolute={};effects={}
                for name,tokenkey in [('real','actual'),('predicted','predicted'),('current',None)]:
                    z=tokens[tokenkey] if tokenkey else tokens['history'][-1:];x=features(z,tokens['history'],rep,q,current=tokenkey is None);decoded=infer(model,x);check('raw_feature_inference',decoded,saved[key+'__decoded_'+name],1e-8);decoded_checks+=1
                    if name=='current':decoded=np.repeat(decoded,len(tokens['actual']),axis=0)
                    absolute[name]=pose(env,decoded,kind);effects[name]=delta(env,absolute[name])
                check('physical_dp',true_delta[0],saved[key+'__true_dp'],1e-12);check('physical_dr',true_delta[1],saved[key+'__true_dr'],1e-12)
                wrong=saved[key+'__wrong_index'];scale=np.array(read(resolve('data/checkpoints/dinowm-retrained-fafraob')/env/'swm_h5_norm_stats.json')['action_std']);d=((parent['actions']-parent['actions'][:1])/scale).reshape(len(parent['actions']),-1);norm=np.linalg.norm(d,axis=1)
                for i,j in enumerate(wrong):
                    if j<0:continue
                    assert i!=j and norm[i]>1e-6 and norm[j]>1e-6
                    ratio=norm[j]/norm[i];cos=np.dot(d[i],d[j])/(norm[i]*norm[j]);assert cfg['wrong_action_norm_ratio'][0]<=ratio<=cfg['wrong_action_norm_ratio'][1] and cos<=cfg['wrong_action_max_cosine']+1e-12;check('wrong_ratio',ratio,saved[key+'__wrong_norm_ratio'][i],1e-12);check('wrong_cosine',cos,saved[key+'__wrong_cosine'][i],1e-12)
                n=len(wrong);zero_abs=tuple(np.repeat(v[:1],n,axis=0) for v in absolute['real']);zero_delta=(np.zeros_like(true_delta[0]),np.zeros_like(true_delta[1]));safe=np.maximum(wrong,0)
                methods={'real_encoded':(absolute['real'],effects['real']),'predicted':(absolute['predicted'],effects['predicted']),'zero_effect':(zero_abs,zero_delta),'persistence':(absolute['current'],effects['current']),'wrong_action':(tuple(v[safe] for v in absolute['predicted']),tuple(v[safe] for v in effects['predicted']))}
                for method,(a,e) in methods.items():
                    measured=metric(env,a,e,true_abs,true_delta);record=g[g.method==method];np.testing.assert_array_equal(record.candidate,ids);check('physical_E_copy',record.physical_E,old.object_error,1e-12);check('jerk_copy',record.jerk_gain,old.jerk_gain,1e-12)
                    check('true_position_magnitude',np.linalg.norm(true_delta[0],axis=1)[ids],record.true_position_magnitude,1e-12)
                    check('true_rotation_magnitude',np.linalg.norm(true_delta[1],axis=1)[ids],record.true_rotation_magnitude,1e-12)
                    check('active_identity',old['actual_norm' if env=='pusht' else 'norm'].to_numpy()>1e-6,record.active,0.)
                    for name,v in measured.items():
                        if method=='wrong_action':v[wrong<0]=np.nan
                        # SO(3) trace/arccos rounds at identity; tolerate sub-microradian error.
                        check(name,v[ids],record[name],2e-7 if 'rotation' in name and 'cosine' not in name else 1e-7)
    assert records==(4 if args.smoke else 402)
    frame=pd.concat(allframes,ignore_index=True);resolution=read(out/'resolution.json');keys=['environment','representation','target','readout']
    for r in resolution['thresholds']:
        v=frame[(frame.split=='calibration')&(frame.method=='real_encoded')]
        for k in keys:v=v[v[k]==r[k]]
        count=v.state_id.value_counts().to_dict();w=np.array([1/count[s] for s in v.state_id]);
        for axis in ['position','rotation']:
            values=v[axis+'_effect_error'].to_numpy();order=np.argsort(values,kind='stable');cumulative=np.cumsum(w[order]);threshold=values[order[np.flatnonzero(cumulative>=cfg['resolution_quantile']*cumulative[-1])[0]]];check('resolution_'+axis,max(1e-9,threshold),r[axis],1e-12)
    states=pd.read_csv(out/'state_metrics.csv');analysis=read(out/'analysis.json');assert analysis['candidate_method_rows']==len(frame) and analysis['state_method_rows']==len(states)
    statekeys=['environment','split','state_id','representation','target','readout','method'];state_lookup=states.set_index(statekeys);thresholds={tuple(r[k] for k in keys):r for r in resolution['thresholds']}
    for key,g in frame.groupby(statekeys,sort=False):
        row=state_lookup.loc[key];valid=g[np.isfinite(g.position_effect_error)&np.isfinite(g.rotation_effect_error)];threshold=thresholds[(key[0],key[3],key[4],key[5])];rp=valid.true_position_magnitude>threshold['position'];rr=valid.true_rotation_magnitude>threshold['rotation']
        assert row.active_candidates==len(g) and row.valid_candidates==len(valid)
        values={'control_support':len(valid)/len(g),'physical_preservation':float((g.physical_E<=1).mean()),'position_resolvable_fraction':float(rp.mean()) if len(valid) else np.nan,'rotation_resolvable_fraction':float(rr.mean()) if len(valid) else np.nan,'position_direction_agreement':float((valid.loc[rp,'position_cosine']>=cfg['position_direction_cosine']).mean()) if rp.any() else np.nan,'rotation_direction_agreement':float((valid.loc[rr,'rotation_cosine']>=cfg['rotation_direction_cosine']).mean()) if rr.any() else np.nan}
        for col in ['position_effect_error','rotation_effect_error','position_absolute_error','rotation_absolute_error','predicted_position_magnitude','predicted_rotation_magnitude','true_position_magnitude','true_rotation_magnitude']:values[col]=float(valid[col].mean()) if len(valid) else np.nan
        for name,value in values.items():check('raw_to_state_'+name,value,row[name],1e-10)
    paired=pd.read_csv(out/'paired_state_metrics.csv');pairkeys=statekeys[:-1];pairlookup=paired.set_index(pairkeys+['control'])
    for key,g in frame.groupby(pairkeys,sort=False):
        predicted=g[g.method=='predicted'].set_index('candidate')
        for control in ['real_encoded','zero_effect','persistence','wrong_action']:
            c=g[g.method==control].set_index('candidate');valid=np.isfinite(c.position_effect_error)&np.isfinite(c.rotation_effect_error);row=pairlookup.loc[(*key,control)];assert row.matched_candidates==valid.sum()
            for axis in ['position','rotation']:
                difference=(predicted.loc[valid,axis+'_effect_error']-c.loc[valid,axis+'_effect_error']).mean() if valid.any() else np.nan;check('raw_paired_'+axis,difference,row[axis+'_difference'],1e-10)
    for r in analysis['paired_controls']:
        g=paired
        for k in ['environment','split','representation','target','readout','control']:g=g[g[k]==r[k]]
        assert int(g.matched_candidates.sum())==r['matched_candidates']
        for metric_name in ['position_difference','rotation_difference']:
            valid=g.dropna(subset=[metric_name]);value=r[metric_name]
            if not len(valid):assert value['mean'] is None;continue
            avg,ci=cluster(valid[metric_name].to_numpy(),valid.initial_pose_family.to_numpy());check('paired_summary_mean',avg,value['mean'],1e-10);check('paired_summary_ci',ci,value['ci95'],1e-10)
    for r in analysis['summary']:
        v=states
        for k in ['environment','split','representation','target','readout','method']:v=v[v[k]==r[k]]
        if r['phase']!='all':v=v[v.phase==r['phase']]
        assert len(v)==r['states']
        for k,value in r.items():
            if not isinstance(value,dict):continue
            g=v.dropna(subset=[k])
            if not len(g):assert value['mean'] is None and value['families']==0;continue
            avg,ci=cluster(g[k].to_numpy(),g.initial_pose_family.to_numpy());check('summary_mean',avg,value['mean'],1e-10);check('summary_ci',ci,value['ci95'],1e-10)
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'states':records,'candidate_method_rows_including_inactive':rows,'decoded_array_checks':decoded_checks,'max_error':errors,'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'resolution_sha256':sha256(out/'resolution.json'),'source_sha256':sha256(Path(__file__)),'scope':'Every saved decoded array from raw tokens/model weights; independent planar/SO3 effect geometry and supported wrong-action matches, original physical E/jerk identities, calibration resolution, raw-to-state means/directional gates, matched-control differences and reported family summary CIs. Collector raw audit covers physical traces and fresh forward samples. No acceptance U/R claim.'})
    print('AUDITED effect readouts',records,rows,decoded_checks,errors,flush=True)
if __name__=='__main__':main()
