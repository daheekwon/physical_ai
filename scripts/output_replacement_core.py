"""Frozen-readout replacement scores and deterministic calibration/selection."""
import numpy as np,pandas as pd
from scripts.retrained_dinowm_effect_readout_core import pose_effect
from scripts.analyze_followup_response_selection import mean_ci

def decoded_score(env,y,kind):
    dp,dr,_,_=pose_effect(env,y,kind=='symmetry')
    return np.maximum(np.linalg.norm(dp,axis=1)/(2 if env=='pusht' else .002),np.linalg.norm(dr,axis=1)/np.deg2rad(2))

def physical(env,a):
    if env=='pusht':
        p=a['physics'];d=p[:,30:,2:5]-p[:1,30:,2:5];pos=np.linalg.norm(d[:,:,:2],axis=-1)/2;rot=np.abs(np.arctan2(np.sin(d[:,:,2]),np.cos(d[:,:,2])))/np.deg2rad(2);dense=np.maximum(pos.max(1),rot.max(1));end=np.maximum(pos[:,49],rot[:,49]);cover=np.maximum(0,a['coverage'][:1]-a['coverage']).max(1)/.02;full=np.maximum(dense,cover);jerk=np.sum(np.diff(p[:,:,:2],n=3,axis=1)**2,axis=(1,2))/.01**5
    else:
        p=a['candidate_object_pose'];pos=np.linalg.norm(p[:,:,:3]-p[:1,:,:3],axis=-1)/.002;q=p[:,:,3:]/np.linalg.norm(p[:,:,3:],axis=-1,keepdims=True);dot=np.clip(abs(np.sum(q*q[:1],axis=-1)),0,1);rot=2*np.arccos(dot)/np.deg2rad(2);dense=np.maximum(pos.max(1),rot.max(1));end=np.maximum(pos[:,124],rot[:,124]);full=dense.copy();robot=np.concatenate([np.broadcast_to(a['prefix_eef_position'],(len(p),*a['prefix_eef_position'].shape)),a['candidate_eef_position']],axis=1);jerk=np.sum(np.diff(robot,n=3,axis=1)**2,axis=(1,2))/.002**5
    gain=(jerk[0]-jerk)/jerk[0] if jerk[0]>1e-8 else np.zeros(len(jerk))
    return {'true_endpoint':end,'dense_pose':dense,'dense_full':full,'jerk_gain':gain,'actual_jerk':jerk}

def selections(frame,level,rule,cfg):
    rows=[]
    for sid,g in frame.groupby('state_id',sort=True):
        active=g.norm.to_numpy()>cfg['minimum_norm'];approved=active&(g[level].to_numpy()<=rule['tau']) if rule['enabled'] else np.zeros(len(g),bool)
        eligible=approved&(g.predicted_gain.to_numpy()>cfg['minimum_predicted_gain']);ids=np.flatnonzero(eligible)
        i=int(ids[np.lexsort((g.candidate.to_numpy()[ids],g.norm.to_numpy()[ids],g.predicted_cost.to_numpy()[ids]))[0]]) if len(ids) else None
        r=g.iloc[i] if i is not None else None
        rows.append({'state_id':sid,'initial_pose_family':int(g.initial_pose_family.iloc[0]),'phase':str(g.phase.iloc[0]),'candidate':int(r.candidate) if r is not None else 0,'active':r is not None,'predicted_gain':float(r.predicted_gain) if r is not None else 0.,'jerk_gain':float(r.jerk_gain) if r is not None else 0.,'object_error':float(r.dense_full) if r is not None else 0.,'violation':bool(r is not None and r.dense_full>1),'joint':bool(r is not None and r.dense_full<=1 and r.jerk_gain>=cfg['meaningful_gain']),'candidate_coverage':float(approved.sum()/active.sum()) if active.any() else 0.,'candidate_violation_mass':float(np.sum(approved&(g.dense_full.to_numpy()>1))/active.sum()) if active.any() else 0.,'approved_count':int(approved.sum()),'active_candidates':int(active.sum())})
    return pd.DataFrame(rows)

def fit(frame,level,cfg,env):
    finite=frame.loc[(frame.norm>cfg['minimum_norm'])&np.isfinite(frame[level]),level].to_numpy();grid=np.unique(np.r_[0.,1.,np.quantile(finite,np.linspace(0,1,cfg['threshold_quantiles']))]) if len(finite) else np.array([0.,1.]);best=None;bestvalue=None;curves=[]
    for tau in grid:
        r=selections(frame,level,{'enabled':True,'tau':float(tau)},cfg);active=r[r.active];contact=active[active.phase=='contact'];risk=float(active.violation.mean()) if len(active) else None;crisk=float(contact.violation.mean()) if len(contact) else None
        valid=len(active)/len(r)>=cfg['minimum_active_fraction'] and risk is not None and risk<=cfg['target_risk']
        if env=='pusht' and cfg['enforce_pusht_contact_risk']:valid=valid and crisk is not None and crisk<=cfg['target_risk']
        value=(float((r.active&(r.predicted_gain>=cfg['meaningful_gain'])).mean()),float(r.predicted_gain.mean()))
        curves.append({'tau':float(tau),'active_fraction':float(r.active.mean()),'risk':risk,'contact_risk':crisk,'predicted_useful':value[0],'mean_predicted_gain':value[1],'valid':bool(valid)})
        if valid and (bestvalue is None or value>bestvalue):best={'enabled':True,'tau':float(tau),'calibration':curves[-1]};bestvalue=value
    return best or {'enabled':False,'tau':None,'reason':'No prespecified empiricalrisk/activecoverage feasible threshold'},curves

def summarize(selected):
    rows=[]
    keys=['environment','representation','target','mode','level','split']
    for key,g in selected.groupby(keys,sort=True):
        for phase in ['all']+sorted(g.phase.unique().tolist()):
            v=g if phase=='all' else g[g.phase==phase];a=v[v.active];den=v.candidate_coverage.sum()
            rows.append({**dict(zip(keys,key)),'phase':phase,'states':len(v),'active_states':len(a),**{m:mean_ci(v,m) for m in ['active','joint','jerk_gain','predicted_gain','candidate_coverage']},'selected_risk':mean_ci(a,'violation'),'candidate_risk':float(v.candidate_violation_mass.sum()/den) if den>0 else None})
    contrasts=[];group=['environment','representation','target','mode','split'];levels=['predicted','real_encoded','true_endpoint','dense_pose','dense_full']
    for key,g in selected.groupby(group,sort=True):
        for prev,nextlevel in zip(levels[:-1],levels[1:]):
            a=g[g.level==prev];b=g[g.level==nextlevel];v=b.merge(a,on=['state_id','initial_pose_family'],suffixes=('','_before'),validate='one_to_one')
            for metric in ['joint','jerk_gain','active']:
                v['difference']=v[metric].astype(float)-v[metric+'_before'].astype(float);contrasts.append({**dict(zip(group,key)),'from':prev,'to':nextlevel,'metric':metric,**mean_ci(v,'difference')})
    return {'summary':rows,'paired_adjacent_levels':contrasts,'interpretation':'Future levels are diagnostic oracles. Threshold transfer and separate calibration are distinct; calibration results are resubstitution. Same expected-jerk ranking, not actual-jerk oracle. Readout-selected development is not fresh test.'}
