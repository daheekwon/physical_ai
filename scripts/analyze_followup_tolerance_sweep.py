"""Physical-tolerance sweep with separate fixed, jointly calibrated and per-tolerance rules."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from scripts.calibrate_followup_selection import load_bank,selected_rows
from scripts.analyze_followup_response_selection import mean_ci


def packed(frame,methods):
    groups=list(frame.groupby('state_id',sort=True));n=len(groups);width=max(int(g['index'].max()) for _,g in groups)+1;arrays={k:np.full((n,width),np.inf) for k in ['cost','norm','gain','position','rotation','coverage']+[m+'_score' for m in methods]};meta=[]
    for i,(sid,g) in enumerate(groups):
        g=g.sort_values('index');idx=g['index'].to_numpy(int);assert len(np.unique(idx))==len(idx) and np.all(idx>0)
        # Keep position identical to each source bank index, including any sparse row ids.
        for key in arrays:arrays[key][i,0]=0.
        arrays['cost'][i,0]=float(g.nominal_predicted_jerk.iloc[0])
        for key,col in [('cost','predicted_jerk'),('norm','actual_norm'),('gain','jerk_gain'),('position','position_max'),('rotation','rotation_max')]+[(m+'_score',m+'_score') for m in methods]:arrays[key][i,idx]=g[col]
        arrays['coverage'][i,idx]=g[['coverage_loss','coverage_suffix_loss']].max(axis=1)
        meta.append({'state_id':sid,'initial_pose_family':int(g.initial_pose_family.iloc[0]),'nominal_contact':bool(g.nominal_contact.iloc[0])})
    arrays['state_ids']=np.array([x['state_id'] for x in meta])
    arrays['gain'][~np.isfinite(arrays['gain'])]=0
    return arrays,pd.DataFrame(meta)


def choose(a,method,rule,pm,rm,shared=False):
    n=len(a['cost']);idx=np.zeros(n,dtype=int)
    if not rule['enabled']:return idx
    f=min(pm,rm) if shared else 1.;radius=rule['relative_radius']*min(1,f);tau=rule['tau']*f
    valid=np.isfinite(a[method+'_score'])&(a[method+'_score']<=tau)&(a['norm']<=radius*np.sqrt(10)+1e-7)&(a['norm']>1e-6)
    cost=np.where(valid,a['cost'],np.inf)
    # Cost then norm then bank index, exactly matching original selector tie-breaking.
    order=np.lexsort((a['norm'],cost),axis=1);candidate=order[:,0];rows=np.arange(n);c=cost[rows,candidate];nom=a['cost'][:,0]
    improve=(nom>1e-8)&np.isfinite(c)&((nom-c)/np.maximum(nom,1e-8)>1e-8)
    idx[improve]=candidate[improve];return idx


def errors(a,eps,pm,rm):return np.maximum.reduce([a['position']/(eps['position']*pm),a['rotation']/(eps['rotation']*rm),a['coverage']/eps['coverage_loss']])


def metrics(a,meta,idx,e):
    np.testing.assert_array_equal(a['state_ids'],meta.state_id.to_numpy())
    rows=np.arange(len(idx));active=idx>0;contact=active&meta.nominal_contact.to_numpy();violation=e[rows,idx]>1;pred=(a['cost'][:,0]-a['cost'][rows,idx])/np.maximum(a['cost'][:,0],1e-8)
    return {'active':float(active.mean()),'risk':float(violation[active].mean()) if active.any() else None,'contact_risk':float(violation[contact].mean()) if contact.any() else None,'predicted_useful':float((active&(pred>=.1)).mean()),'predicted_gain':float(pred.mean())}


def fit(a,meta,method,grid,eps,cfg,shared):
    raw=a[method+'_score'][:,1:];finite=raw[np.isfinite(raw)];thresholds=np.unique(np.r_[0.,np.quantile(finite,np.linspace(0,1,cfg['threshold_quantiles']))]);best=None;value=None;curves=[]
    for radius in cfg['radii']:
        for tau in thresholds:
            rule={'enabled':True,'relative_radius':radius,'tau':float(tau)};stats=[]
            for pm,rm in grid:
                idx=choose(a,method,rule,pm,rm,shared);stats.append(metrics(a,meta,idx,errors(a,eps,pm,rm)))
            feasible=all(m['active']>=cfg['minimum_active_coverage'] and m['risk'] is not None and m['risk']<=cfg['target_violation'] and m['contact_risk'] is not None and m['contact_risk']<=cfg['target_violation'] for m in stats)
            score=(np.mean([m['predicted_useful'] for m in stats]),np.mean([m['predicted_gain'] for m in stats]));curves.append({'rule':rule,'feasible':feasible,'conditions':stats})
            if feasible and (value is None or score>value):best=rule;value=score
    return best or {'enabled':False,'relative_radius':0,'tau':None,'reason':'No shared feasible rule across all specified tolerance conditions' if shared else 'No feasible rule'},curves


def audit_raw(base,frames,eps):
    maximum={};count=0
    for split,frame in frames.items():
        for sid,g in frame.groupby('state_id'):
            with np.load(base/'f1'/f'{sid}_h05.npz') as a:
                trace=a['physics'][:,30:];p=np.linalg.norm(trace[:,:,2:4]-trace[:1,:,2:4],axis=-1).max(1);angle=trace[:,:,4]-trace[:1,:,4];r=np.abs(np.arctan2(np.sin(angle),np.cos(angle))).max(1);c=np.maximum(0,a['coverage'][:1]-a['coverage']).max(1);g=g.sort_values('index');ids=g['index'].to_numpy(int)
                for label,val,col in [('position',p,'position_max'),('rotation',r,'rotation_max'),('coverage',c,None)]:
                    expected=g[col].to_numpy() if col else g[['coverage_loss','coverage_suffix_loss']].max(axis=1).to_numpy();err=float(np.max(np.abs(val[ids]-expected)));maximum[label]=max(maximum.get(label,0.),err);assert err<1e-7
                e=np.maximum.reduce([p/eps['position'],r/eps['rotation'],c/eps['coverage_loss']]);assert np.max(np.abs(e[ids]-g.object_error))<1e-7;count+=len(g)
    return {'states':sum(f.state_id.nunique() for f in frames.values()),'candidate_rows':count,'max_errors':maximum,'scope':'100Hz position/rotation maxima recomputed from raw traces; saved endpoint coverages reused; base E verified. No new dynamics execution.'}


def main():
    start=time.perf_counter();base=resolve('artifacts/followup_v1');out=base/'tolerance_sweep';out.mkdir(exist_ok=True);cfg=json.loads(resolve('configs/followup_selection.json').read_text());fcfg=json.loads(resolve('configs/followup_v1.json').read_text());tcfg=json.loads(resolve('configs/followup_tolerance_sweep.json').read_text());methods=tcfg['methods'];grid=tcfg['multipliers'];eps=fcfg['tolerances'];frames={s:load_bank(base,'f1',s) for s in ['calibration','development']};audit=audit_raw(base,frames,eps);packs={s:packed(f,methods) for s,f in frames.items()};original=json.loads((base/'selection/selector_rules.json').read_text())['rules'];cal,calmeta=packs['calibration'];rules={};curves={}
    for split,(a,meta) in packs.items():
        for method in methods:
            expected=selected_rows(frames[split],method,original[method]).set_index('state_id').loc[meta.state_id,'index'].to_numpy()
            np.testing.assert_array_equal(choose(a,method,original[method],1,1),expected)
    for method in methods:
        shared,curve=fit(cal,calmeta,method,grid,eps,cfg,True);rules[method]={'shared':shared,'per_tolerance':{}};curves[method]={'shared':curve,'per_tolerance':{}}
        for pm,rm in grid:
            key=f'p{pm:g}_r{rm:g}';rule,curve=fit(cal,calmeta,method,[(pm,rm)],eps,cfg,False);rules[method]['per_tolerance'][key]=rule;curves[method]['per_tolerance'][key]=curve
        print(method,'shared enabled',shared['enabled'],flush=True)
    for method in methods:
        baseline=rules[method]['per_tolerance']['p1_r1']
        assert baseline['enabled']==original[method]['enabled']
        np.testing.assert_allclose([baseline['tau'],baseline['relative_radius']],[original[method]['tau'],original[method]['relative_radius']],rtol=1e-12,atol=1e-12)
    write_json(out/'calibrated_settings.json',{'config':tcfg,'rules':rules,'original_rules':{m:original[m] for m in methods},'source_sha256':sha256(Path(__file__)),'config_sha256':sha256(resolve('configs/followup_tolerance_sweep.json'))});write_json(out/'calibration_curves.json',curves)
    rows=[]
    for split,(a,meta) in packs.items():
        ids=np.arange(len(meta))
        for pm,rm in grid:
            key=f'p{pm:g}_r{rm:g}';e=errors(a,eps,pm,rm)
            for method in methods:
                for variant in tcfg['variants']:
                    rule=original[method] if variant=='fixed_original' else rules[method]['shared'] if variant=='shared_tolerance_input' else rules[method]['per_tolerance'][key];idx=choose(a,method,rule,pm,rm,variant=='shared_tolerance_input');f=meta.copy();f['split']=split;f['tolerance']=key;f['method']=method;f['variant']=variant;f['index']=idx;f['active']=idx>0;f['object_error']=e[ids,idx];f['jerk_gain']=a['gain'][ids,idx];rows.append(f)
            # Best actually observed safe candidate, only a diagnostic finite-bank oracle.
            quality=np.where((e<=1)&np.isfinite(a['norm']),a['gain'],-np.inf);idx=np.argmax(quality,axis=1);idx[quality[ids,idx]<=0]=0;f=meta.copy();f['split']=split;f['tolerance']=key;f['method']='finite_bank_oracle';f['variant']='oracle';f['index']=idx;f['active']=idx>0;f['object_error']=e[ids,idx];f['jerk_gain']=a['gain'][ids,idx];rows.append(f)
    f=pd.concat(rows,ignore_index=True);f['violation']=f.object_error>1;f['joint']=f.active&~f.violation&(f.jerk_gain>=.1);summary=[]
    for (split,tolerance,method,variant),g in f.groupby(['split','tolerance','method','variant']):summary.append({'split':split,'tolerance':tolerance,'method':method,'variant':variant,**{m:mean_ci(g,m) for m in ['joint','active','jerk_gain']},'risk':mean_ci(g[g.active],'violation')})
    resolution=json.loads((base/'lewm_audit/effects/resolution.json').read_text())['thresholds']
    f.to_csv(out/'evaluation.csv',index=False);write_json(out/'raw_audit.json',audit);write_json(out/'analysis.json',{'config':tcfg,'summary':summary,'probe_resolution':resolution,'seconds':time.perf_counter()-start,'interpretation':'Fixed original rule sensitivity, one jointly calibrated tolerance-input rule, and independently recalibrated reference are distinct. All development/calibration; no finaltest or general safety guarantee.'});print('Rows',len(f),'seconds',round(time.perf_counter()-start,1),flush=True)
if __name__=='__main__':main()
