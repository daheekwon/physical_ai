"""Independent final test analysis; no fitting or threshold selection is permitted."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib,pymunk
from zj_wam.io import resolve,write_json,sha256
from zj_wam.adapters import ExpertDataset,ReplaySimulator
from zj_wam.readouts import decode,effects,history_features
from zj_wam.selection import accepted
from scripts.audit_followup_streaming_and_strong import geometry
from scripts.calibrate_followup_selection import selected_rows
from scripts.calibrate_followup_strong_baselines import rows_for_setting
from scripts.analyze_followup_response_selection import mean_ci

BASE=resolve('artifacts/followup_v1')

def load(path):return json.loads(path.read_text())
def energy(q):return np.sum((q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3])**2)/.01**5

class Audit:
    def __init__(self):self.maxima={}
    def check(self,name,x,y,tol=1e-7):
        delta=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));self.maxima[name]=max(self.maxima.get(name,0.),delta)
        assert delta<=tol,(name,delta)

class Coverage:
    def __init__(self,data,meta,cfg):
        actions,states=data.episode(meta['episode']);sim=ReplaySimulator(actions,states,meta['timestep'],cfg['seed']+meta['episode']);sim.env.reset(seed=cfg['seed'])
        self.vertices=[list(x.get_vertices()) for x in sim.env.block.shapes];self.body=pymunk.Body(body_type=pymunk.Body.KINEMATIC);self.body.center_of_gravity=sim.env.block.center_of_gravity
        self.body.angle=float(states[-1,4]);self.body.position=states[-1,2:4].tolist();self.target=geometry(self.body,self.vertices);sim.close()
    def __call__(self,state):
        self.body.angle=float(state[4]);self.body.position=state[2:4].tolist();return geometry(self.body,self.vertices).intersection(self.target).area/self.target.area


def physical(trace,reference,h,coverage,cfg):
    d=trace[30:,2:5]-reference[30:,2:5];pos=np.linalg.norm(d[:,:2],axis=1).max();rot=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2]))).max()
    ct=max(0,coverage(reference[30+h*10-1])-coverage(trace[30+h*10-1]));cs=max(0,coverage(reference[-1])-coverage(trace[-1]));j=energy(trace[:,:2]);j0=energy(reference[:,:2]);eps=cfg['tolerances']
    return {'object_error':max(pos/eps['position'],rot/eps['rotation'],ct/eps['coverage_loss'],cs/eps['coverage_loss']),'position_max':pos,'rotation_max':rot,'coverage_loss':ct,'coverage_suffix_loss':cs,'jerk_gain':(j0-j)/j0 if j0>1e-8 else 0.,'physics_jerk':j}


def paths_for(folder,split,h=None,limit=None):
    pattern=split+('*h%02d.json'%h if h else '*.json');paths=sorted((BASE/folder).glob(pattern))
    if not limit:assert len(paths)==(45 if split=='test' else 105),(folder,h,len(paths))
    return paths[:limit]


def audit_banks(split,cfg,data,dest,limit):
    results={};oracles=[]
    for folder,h in [('f1',5),('interventions',5),('family_ablation',5),('strong_baselines',5),('f1',10),('f1',20)]:
        audit=Audit();nrows=0;paths=paths_for(folder,split,h,limit)
        for path in paths:
            meta=load(path);frame=pd.read_csv(path.with_suffix('.csv'));assert frame.state_id.eq(meta['state_id']).all() and frame.split.eq(split).all();cov=Coverage(data,meta,cfg)
            with np.load(path.with_suffix('.npz')) as a:
                reference=a['physics'][0];nominal=a['actions'][0];assert len(reference)==30+(h+3)*10
                for row in frame.itertuples():
                    trace=a['physics'][row.index];action=a['actions'][row.index];audit.check('fixed_prefix',trace[:30],reference[:30],1e-9);norm=np.linalg.norm((action-nominal)/data.scaler.scale_);audit.check('action_norm',norm,row.actual_norm,2e-6);assert np.max(np.abs(action))<=1+1e-6
                    for k,v in physical(trace,reference,h,cov,cfg).items():
                        if k=='physics_jerk':audit.check('relative_jerk',v/max(v,1),getattr(row,k)/max(v,1))
                        else:audit.check(k,v,getattr(row,k))
                    nrows+=1
                valid=frame[(frame.actual_norm>1e-6)&(frame.object_error<=1)&(frame.jerk_gain>0)];gain=float(valid.jerk_gain.max()) if len(valid) else 0.
                oracles.append({'bank':folder,'horizon':h,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'gain':gain,'joint':gain>=.1})
        results[f'{folder}_h{h}']={'states':len(paths),'rows':nrows,'maxima':audit.maxima}
    o=pd.DataFrame(oracles);o.to_csv(dest/'oracle_states.csv',index=False);write_json(dest/'bank_audit.json',{'partial':bool(limit),'banks':results,'scope':'Every raw candidate trace, full prefix+H+suffix jerk, same-time position/angle and independent polygon coverage from raw physical poses; candidate norms/bounds. Future labels used only for audit and diagnostic oracle.'})
    write_json(dest/'oracle_summary.json',[{'bank':b,'horizon':int(h),'joint':mean_ci(g,'joint'),'gain':mean_ci(g,'gain')} for (b,h),g in o.groupby(['bank','horizon'])]);print('BANK AUDIT',results,flush=True)


def source_frame(folder,split,h=5,limit=None):
    return pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in paths_for(folder,split,h,limit)],ignore_index=True)


def selection(split,cfg,dest,limit):
    rules=load(BASE/'selection/selector_rules.json')['rules'];bankrules=load(BASE/'selection/bank_rules.json')['rules'];fcfg=load(resolve('configs/followup_final_expert.json'));chosen=[];approval=[]
    link_audit=Audit()
    for bankname in ['f1','interventions','family_ablation']:
        for path in paths_for('response_scores/'+bankname,split,5,limit):
            scored=pd.read_csv(path.with_suffix('.csv'));raw=pd.read_csv((BASE/bankname/path.name).with_suffix('.csv'));joined=scored.merge(raw,on=['state_id','index','initial_pose_family'],validate='one_to_one',suffixes=('','_raw'));assert len(joined)==len(raw)==len(scored)
            for column in ['object_error','jerk_gain','actual_norm','position_max','rotation_max']:
                link_audit.check(bankname+'_'+column,joined[column],joined[column+'_raw'],1e-10)
    frame=source_frame('response_scores/f1',split,limit=limit)
    for method,rule in rules.items():
        f=selected_rows(frame,method,rule);f['pipeline']='common_bank';chosen.append(f)
    raw=source_frame('strong_baselines',split,limit=limit);strong=load(BASE/'strong_baselines/calibrated_settings.json')['rules']
    for method,rule in strong.items():
        f=rows_for_setting(raw,method,rule.get('setting'),rule.get('radius'),not rule['enabled']);f['pipeline']='strong_locked';chosen.append(f)
    family=source_frame('response_scores/family_ablation',split,limit=limit)
    for gen,g in family.groupby('method'):
        for verifier in fcfg['family_verifiers']:
            f=selected_rows(g.reset_index(drop=True),verifier,rules[verifier]);f['pipeline']='family_'+gen;chosen.append(f)
    bank=source_frame('response_scores/interventions',split,limit=limit)
    for method,rule in bankrules.items():
        mask=accepted(bank,method,rule)
        for sid,g in bank.groupby('state_id'):
            take=mask[g.index];approval.append({'state_id':sid,'initial_pose_family':int(g.initial_pose_family.iloc[0]),'method':method,'coverage':float(take.mean()),'violation_mass':float(np.sum(take&(g.object_error.to_numpy()>1))/len(g)),'accepted':int(take.sum())})
    f=pd.concat(chosen,ignore_index=True);f['joint']=f.active&(f.object_error<=1)&(f.jerk_gain>=.1);f['violation']=f.object_error>1;summary=[]
    for (pipeline,method),g in f.groupby(['pipeline','method']):
        for stratum in ['all','contact','free']:
            x=g if stratum=='all' else g[g.nominal_contact==(stratum=='contact')];summary.append({'pipeline':pipeline,'method':method,'stratum':stratum,'states':len(x),'joint':mean_ci(x,'joint'),'active':mean_ci(x,'active'),'risk':mean_ci(x[x.active],'violation'),'gain':mean_ci(x,'jerk_gain')})
    paired=[];common=f[f.pipeline=='common_bank']
    for method,baseline in [fcfg['primary_comparison'],*fcfg['secondary_comparisons']]:
        x=common[common.method==method].merge(common[common.method==baseline],on=['state_id','initial_pose_family'],validate='one_to_one',suffixes=('','_base'));x['delta_U']=x.joint.astype(float)-x.joint_base.astype(float);x['delta_gain']=x.jerk_gain-x.jerk_gain_base
        paired.append({'method':method,'baseline':baseline,'primary':[method,baseline]==fcfg['primary_comparison'],'U_difference':mean_ci(x,'delta_U'),'gain_difference':mean_ci(x,'delta_gain')})
    b=pd.DataFrame(approval);bs=[]
    for method,g in b.groupby('method'):
        fam=g.groupby('initial_pose_family')[['coverage','violation_mass']].sum().to_numpy();rng=np.random.default_rng(cfg['seed']);sample=fam[rng.integers(len(fam),size=(2000,len(fam)))].sum(1);risk=np.divide(sample[:,1],sample[:,0],out=np.full(2000,np.nan),where=sample[:,0]>0)
        bs.append({'method':method,'coverage':mean_ci(g,'coverage'),'risk':float(g.violation_mass.sum()/g.coverage.sum()) if g.coverage.sum()>0 else None,'risk_ci95':np.nanquantile(risk,[.025,.975]).tolist() if np.isfinite(risk).any() else None})
    f.to_csv(dest/'selected.csv',index=False);b.to_csv(dest/'approval.csv',index=False);write_json(dest/'selection.json',{'partial':bool(limit),'summary':summary,'paired':paired,'bank_approval':bs,'raw_candidate_link_audit':link_audit.maxima,'scope':'Rules loaded unchanged; common bank vs larger-family strong adapters vs fixed-verifier generator study separated. State-pose and supervised readout variants explicitly named. Conditional risk undefined when no active corrections.'});print('SELECTION',len(f),flush=True)


def diagnostics(split,cfg,dest,limit):
    records=[];capture=[];audit=Audit();paths=sorted((BASE/'lewm_audit/representations').glob(split+'_*.npz'));expected=15 if split=='test' else 35
    if not limit:assert len(paths)==expected
    for path in paths[:limit]:
        with np.load(path) as a:
            for name in ['projected_cls','encoder_cls','middle_cls','spatial_quadrants','history_projected']:
                for kind in ['ridge','mlp64']:
                    probe=joblib.load(BASE/f'lewm_audit/readouts/{name}_{kind}.joblib');pred=probe['output_scaler'].inverse_transform(probe['estimator'].predict(probe['input_scaler'].transform(a[name])));truth=a['targets'];pe=np.linalg.norm(pred[:,:2]-truth[:,:2],axis=1);angle=np.arctan2(pred[:,2],pred[:,3])-np.arctan2(truth[:,2],truth[:,3]);re=np.abs(np.arctan2(np.sin(angle),np.cos(angle)))
                    capture.extend({'representation':name,'readout':kind,'initial_pose_family':int(fam),'position_error':float(p),'rotation_error_deg':float(np.rad2deg(r))} for fam,p,r in zip(a['family'],pe,re))
    cap=pd.DataFrame(capture);write_json(dest/'capture.json',[{'representation':rep,'readout':kind,'position_error':mean_ci(g,'position_error'),'rotation_error_deg':mean_ci(g,'rotation_error_deg')} for (rep,kind),g in cap.groupby(['representation','readout'])])
    thresholds=load(BASE/'lewm_audit/effects/resolution.json')['thresholds'];frames=[]
    for path in paths_for('lewm_audit/effects',split,5,limit):
        frame=pd.read_csv(path.with_suffix('.csv'));frames.append(frame)
        with np.load(path.with_suffix('.npz')) as a,np.load((BASE/'interventions'/path.name).with_suffix('.npz')) as raw:
            true=raw['control'][:,7:8,2:5];audit.check('true_endpoint',a['true'],true);true_delta=effects(true);audit.check('true_effect',a['true_effect'],true_delta);wrong=a['wrong_indices'];assert np.all(wrong!=np.arange(1,len(true)))
            for (rep,kind,method),g in frame.groupby(['representation','readout','method']):
                probe=joblib.load(BASE/f'lewm_audit/readouts/{rep}_{kind}.joblib');real=decode(probe,a['real_'+rep])
                if method=='real_encoded':delta=effects(real)
                elif method in ['zero_effect','persistence']:delta=np.zeros_like(true_delta)
                else:
                    feature=a['predicted_projected'] if rep=='projected_cls' else history_features(raw['history_embedding'],a['predicted_projected']);pred=effects(decode(probe,feature));delta=pred if method=='predicted' else pred[np.r_[0,wrong]]
                ids=g['index'].to_numpy(int);err=delta[ids,0]-true_delta[ids,0];audit.check('effect_position_error',np.linalg.norm(err[:,:2],axis=1),g.position_effect_error);audit.check('effect_rotation_error',np.abs(np.arctan2(np.sin(err[:,2]),np.cos(err[:,2]))),g.rotation_effect_error)
    f=pd.concat(frames,ignore_index=True);summary=[]
    for (rep,kind,method),g in f.groupby(['representation','readout','method']):
        threshold=thresholds[rep+'_'+kind];g=g.copy();g['resolved']=g.position_effect>threshold['position'];g['direction_available']=g.predicted_position_magnitude>1e-8;valid=g[g.resolved&g.direction_available];summary.append({'representation':rep,'readout':kind,'method':method,'rows':len(g),'position_error':mean_ci(g,'position_effect_error'),'rotation_error_rad':mean_ci(g,'rotation_effect_error'),'resolved':mean_ci(g,'resolved'),'direction_available_when_resolved':mean_ci(g[g.resolved],'direction_available'),'resolved_direction_cosine':mean_ci(valid,'position_cosine')})
    write_json(dest/'effect_diagnostics.json',{'summary':summary,'max_audit_errors':audit.maxima,'scope':'Frozen calibration resolution and development-selected probes. Zero predicted vectors have undefined direction. Unresolved effects retained in physical selection statistics. Effect errors independently decoded from saved real/predicted features.'});print('DIAGNOSTICS',len(f),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['test','development','calibration'],default='test');p.add_argument('--phase',choices=['all','banks','selection','diagnostics','runtime','shifts','repeated'],default='all');p.add_argument('--limit',type=int);args=p.parse_args()
    if args.split=='test':
        from scripts.run_followup_final_expert import verify_lock
        verify_lock()
    cfg=load(resolve('configs/followup_v1.json'));dest=BASE/'final_expert'/('test' if args.split=='test' else 'validation_'+args.split);dest.mkdir(parents=True,exist_ok=True);data=ExpertDataset(cfg['dataset'])
    phases=['banks','selection','diagnostics','runtime','shifts','repeated'] if args.phase=='all' else [args.phase]
    for phase in phases:
        if phase=='banks':audit_banks(args.split,cfg,data,dest,args.limit)
        elif phase=='selection':selection(args.split,cfg,dest,args.limit)
        elif phase=='diagnostics':diagnostics(args.split,cfg,dest,args.limit)
        else:extended(phase,args.split,cfg,data,dest,args.limit)
    print('FINAL ANALYSIS PHASES',phases,flush=True)

# Extended raw audit routines are defined below before main is invoked.

def extended(phase,split,cfg,data,dest,limit):
    audit=Audit();frames=[];timings=[];counts={};budget_rows=[]
    folders={'runtime':['runtime/'+split,'strong_runtime/'+split],'shifts':['shift_evaluation','geometric_shifts'],'repeated':['repeated_refinement']}[phase]
    for folder in folders:
        paths=paths_for(folder,split,limit=limit);counts[folder]=len(paths)
        for path in paths:
            meta=load(path);frame=pd.read_csv(path.with_suffix('.csv'));spec=frame.iloc[0].to_dict();cov=Coverage(data,spec,cfg)
            with np.load(path.with_suffix('.npz')) as a:
                for row in frame.itertuples():
                    if phase=='runtime':
                        key=row.variant;reference=a['nominal_physics'];trace=a[key+'_physics'];nominal=a['nominal_actions'][:5];action=a[key+'_actions'][:5];h=5
                        audit.check('unchanged_suffix',a[key+'_actions'][5:],a['nominal_actions'][5:]);reps=meta['repetitions'][key] if folder.startswith('runtime/') else meta['timings'][key];assert len(reps)==3
                        values=[r.get('total_seconds',r.get('seconds')) for r in reps];audit.check('mean_seconds',np.mean(values),row.mean_seconds)
                        for r,v in zip(reps,values):assert v>0;timings.append({'pipeline':folder.split('/')[0],'variant':key,'method':row.method,'state_id':row.state_id,'initial_pose_family':row.initial_pose_family,'seconds':v,'additional_peak_gpu_bytes':r['additional_peak_gpu_bytes']})
                    elif phase=='shifts':
                        key=row.condition+'_'+row.method;reference=a[row.condition+'_nominal_physics'];trace=a[key+'_physics'];h=5
                        if folder=='geometric_shifts':nominal=a[row.condition+'_nominal_actions'];action=a[key+'_actions']
                        else:nominal=a[row.condition+'_actions'][0];action=a[row.condition+'_actions'][row.index]
                    else:
                        key=row.method;reference=a['nominal_physics'];trace=a[key+'_physics'];nominal=a['nominal_actions'];action=a[key+'_actions'];h=20
                    audit.check('fixed_prefix',trace[:30],reference[:30],1e-9);norm=np.linalg.norm((action-nominal)/data.scaler.scale_);stored=getattr(row,'actual_norm',getattr(row,'norm',None));audit.check('action_norm',norm,stored,2e-6);assert np.max(np.abs(action))<=1+1e-6
                    for long in ([False,True] if phase=='repeated' else [False]):
                        stop=len(trace) if long else 30+(h+3)*10;metrics=physical(trace[:stop],reference[:stop],h,cov,cfg);prefix='long_' if long else ''
                        for k,v in metrics.items():
                            if k=='physics_jerk':audit.check(prefix+'relative_jerk',v/max(v,1),getattr(row,prefix+k)/max(v,1))
                            else:audit.check(prefix+k,v,getattr(row,prefix+k))
            frame['pipeline']=folder.split('/')[0];frames.append(frame)
    f=pd.concat(frames,ignore_index=True)
    if phase=='shifts':f=f[~((f.pipeline=='shift_evaluation')&(f.method=='gaussian_action'))].copy()
    if phase=='repeated':f['active']=f.actual_norm>1e-6;f['nested_long_error']=np.maximum(f.object_error,f.long_object_error);f['long_joint']=f.active&(f.nested_long_error<=1)&(f.long_jerk_gain>=.1);f['long_violation']=f.nested_long_error>1
    f['joint']=f.active&(f.object_error<=1)&(f.jerk_gain>=.1);f['violation']=f.object_error>1;grouping=['pipeline','variant'] if phase=='runtime' else ['condition','method'] if phase=='shifts' else ['method'];summary=[]
    for keys,g in f.groupby(grouping):
        if not isinstance(keys,tuple):keys=(keys,)
        record=dict(zip(grouping,keys));record.update({'states':len(g),'joint':mean_ci(g,'joint'),'active':mean_ci(g,'active'),'risk':mean_ci(g[g.active],'violation'),'gain':mean_ci(g,'jerk_gain')})
        if phase=='runtime':
            t=pd.DataFrame(timings);lat=t[(t.pipeline==record['pipeline'])&(t.variant==record['variant'])];record.update({'p50_seconds':float(lat.seconds.median()),'p95_seconds':float(lat.seconds.quantile(.95))})
        if phase=='repeated':record.update({'long_joint':mean_ci(g,'long_joint'),'long_risk':mean_ci(g[g.active],'long_violation')})
        summary.append(record)
    if phase=='runtime':
        t=pd.DataFrame(timings);t.to_csv(dest/'runtime_timings.csv',index=False)
        for pipeline,asset in [('runtime','runtime/calibrated_settings.json'),('strong_runtime','strong_runtime/calibrated_budgets.json')]:
            if not (BASE/asset).exists():
                assert limit and split!='test';continue
            for rule in load(BASE/asset)['rules']:
                method=rule['method'];cap=rule.get('budget_seconds',rule.get('cap_seconds'));key=(method+'__'+rule['selection']['profile'] if pipeline=='runtime' and rule['enabled'] else rule.get('variant'));row={'pipeline':pipeline,'method':method,'cap_seconds':cap,'enabled':rule['enabled'],'variant':key}
                if rule['enabled']:
                    g=f[(f.pipeline==pipeline)&(f.variant==key)];lat=t[(t.pipeline==pipeline)&(t.variant==key)].copy();assert len(g);lat['miss']=lat.seconds>cap;row.update({'joint':mean_ci(g,'joint'),'risk':mean_ci(g[g.active],'violation'),'deadline_exceedance':mean_ci(lat,'miss')})
                budget_rows.append(row)
    f.to_csv(dest/(phase+'_evaluation.csv'),index=False);write_json(dest/(phase+'.json'),{'partial':bool(limit),'states':counts,'rows':len(f),'summary':summary,'budget_comparison':budget_rows,'max_raw_audit_errors':audit.maxima,'scope':'Frozen rules; dense paired physical pose and independent raw-pose polygon coverage audit. Runtime caps/profile choices loaded unchanged from calibration. Repeated long errors retain core coverage checks. Different source pipelines and correction families remain identified.'});print(phase.upper(),counts,len(f),flush=True)

if __name__=='__main__':main()
