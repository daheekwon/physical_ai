"""Independent native-distance/physical/calibration checks for DINO verifier study."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset
from scripts.analyze_followup_final_expert import Coverage
from scripts.audit_followup_output_replacement import selected,pd_positions
from scripts.audit_retrained_dinowm_readouts import cluster
from scripts.run_followup_dinowm_selection import make_frame
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('dinowm_selection_smoke' if args.smoke else 'dinowm_selection');capture=BASE/('dinowm_interventions_smoke' if args.smoke else 'dinowm_interventions');protocol=read(out/'protocol.json');cfg=protocol['config'];done=read(out/'execution_complete.json');assert done['smoke']==args.smoke and done['states']==(4 if args.smoke else 402) and done['protocol_sha256']==sha256(out/'protocol.json')
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    for p,h in done['files'].items():assert sha256(out/p)==h,p
    errors={}
    def check(name,x,y,tol=1e-8,rtol=0.):
        x=np.asarray(x,float);y=np.asarray(y,float);assert x.shape==y.shape,(name,x.shape,y.shape);np.testing.assert_allclose(x,y,atol=tol,rtol=rtol,equal_nan=True,err_msg=name);v=abs(x-y);errors[name]=max(errors.get(name,0.),float(np.nanmax(v)) if v.size and np.isfinite(v).any() else 0.)
    maincfg=read(resolve('configs/followup_v1.json'));data=ExpertDataset(maincfg['dataset']);raw=read(capture/'manifest.json');frames=[];records=0;native_arrays=0
    for env,splits in raw['states'].items():
        for split,names in splits.items():
            for name in names[:1] if args.smoke else names:
                src=resolve(name);meta=read(src);f,_=make_frame(env,split,src,capture,cfg);frames.append(f);records+=1
                with np.load(src.with_suffix('.npz'),allow_pickle=False) as z:a=dict(z)
                with np.load(capture/env/split/src.with_suffix('.npz').name,allow_pickle=False) as z:tokens=z['predicted'].astype(float)
                ids=f.candidate.to_numpy(int);d=tokens-tokens[0];rms=np.linalg.norm(d.reshape(len(d),-1),axis=1)/np.sqrt(d.shape[1]);mean=np.linalg.norm(tokens.mean(1)-tokens[0].mean(0),axis=1);native_arrays+=2
                fallback=bool((f.generator=='nominal_fallback').all())
                if fallback:assert len(f)==1 and f.norm.iloc[0]==0;continue
                check('native_patch_rms',rms[ids],f.dino_patch_rms,1e-10);check('native_patch_mean',mean[ids],f.dino_patch_mean,1e-10)
                assert not f.generator.isin(['lewm_low','lewm_high']).any() and (f.radius<=.2+1e-12).all() and (f.norm>1e-6).all()
                if env=='pusht':
                    physics=a['physics'];p=physics[:,30:,2:5];d=p-p[:1];pos=np.linalg.norm(d[:,:,:2],axis=2).max(1)/2;rot=np.abs((d[:,:,2]+np.pi)%(2*np.pi)-np.pi).max(1)/np.deg2rad(2);coverage=Coverage(data,meta,maincfg);cov=np.array([[coverage(trace[79]),coverage(trace[-1])] for trace in physics]);check('independent_polygon_coverage',cov,a['coverage'],1e-10);obj=np.maximum.reduce([pos,rot,np.maximum(0,cov[:1]-cov).max(1)/.02]);points=physics[:,:,:2];dt=.01
                    ep,_=data.episode(meta['episode']);suffix=ep[meta['timestep']+5:meta['timestep']+8];robot=np.array([pd_positions(np.r_[act,suffix],a['root']) for act in a['actions']]);ctrl=np.linalg.norm(robot-robot[:1],axis=2).max(1);check('pusht_controller_score',ctrl[ids],f.controller,1e-8);prefixed=np.concatenate([np.broadcast_to(physics[0,:30,:2],(len(robot),30,2)),robot],axis=1);cost=np.sum(np.diff(prefixed,n=3,axis=1)**2,axis=(1,2))/.01**5
                    original=pd.read_csv(BASE/'response_scores/f1'/src.with_suffix('.csv').name).set_index('index').loc[ids];check('audited_vision_contact_copy',original.vision_contact_score,f.contact,0.);check('native_LeWM_parent_copy',original.lewm_global_score,f.lewm_global,0.);scale=data.scaler.scale_
                else:
                    p=a['candidate_object_pose'];pos=np.linalg.norm(p[:,:,:3]-p[:1,:,:3],axis=2).max(1)/.002;q=p[:,:,3:];q=q/np.linalg.norm(q,axis=2,keepdims=True);angle=2*np.arccos(np.clip(np.abs(np.einsum('nki,ki->nk',q,q[0])),0,1));rot=angle.max(1)/np.deg2rad(2);obj=np.maximum(pos,rot);points=np.concatenate([np.broadcast_to(a['prefix_eef_position'],(len(p),*a['prefix_eef_position'].shape)),a['candidate_eef_position']],axis=1);dt=.002
                    robot=a['predicted_robot'];delta=robot-robot[:1];dp=np.linalg.norm(delta[:,:,:3],axis=2);dr=abs((delta[:,:,3]+np.pi)%(2*np.pi)-np.pi);ctrl=np.maximum(dp.max(1)/.002,dr.max(1)/np.deg2rad(2));contact=np.maximum((dp+np.sqrt(3)*.02*dr).max(1)/.002,dr.max(1)/np.deg2rad(2)) if float(a['root_gripper_contact'])>.1 or np.any(np.concatenate([a['nominal_actions'][:,4],a['suffix_actions'][:,4]])>.1) else np.zeros(len(p));check('cube_controller_score',ctrl[ids],f.controller,1e-10);check('cube_contact_score',contact[ids],f.contact,1e-10)
                    r=np.concatenate([np.broadcast_to(a['past_points'][-75:],(len(robot),75,4)),robot],axis=1).copy();r[:,:,3]*=.05;cost=np.sum(np.diff(r,n=3,axis=1)**2,axis=(1,2))/.002**5;latent=a['predicted_latent'];native=np.linalg.norm(latent-latent[:1],axis=-1);check('native_LeWM_cube_score',native[ids],f.lewm_global,1e-7);scale=np.array(read(BASE/'cube/manifest.json')['normalization']['scale'])
                jerk=np.sum(np.diff(points,n=3,axis=1)**2,axis=(1,2))/dt**5;gain=(jerk[0]-jerk)/jerk[0] if jerk[0]>1e-8 else np.zeros(len(jerk));check('dense_full_object_error',obj[ids],f.dense_full,2e-6);check('actual_jerk_gain',gain[ids],f.jerk_gain,1e-8);check('expected_cost',cost[ids],f.predicted_cost,1e-4,1e-9);check('expected_gain',(cost[0]-cost[ids])/max(cost[0],1e-12),f.predicted_gain,1e-8);check('applied_norm',np.linalg.norm(((a['actions'][ids]-a['actions'][:1])/scale).reshape(len(ids),-1),axis=1),f.norm,1e-8)
    # Original arithmetic is independently checked above, then retained for exact
    # <= membership at fitted quantiles instead of reusing rounded exported CSVs.
    frame=pd.concat(frames,ignore_index=True);saved=pd.read_csv(out/'candidates.csv');assert len(frame)==len(saved)
    for col in frame:
        if col in ['environment','split','state_id','phase','generator']:np.testing.assert_array_equal(frame[col],saved[col])
        else:check('export_'+col,frame[col],saved[col],1e-5 if col=='predicted_cost' else 1e-10)
    curves=pd.read_csv(out/'calibration_curves.csv');rules={(r['environment'],r['method']):r['rule'] for r in read(out/'rules.json')['rules']};rebuilt=[];trials=0
    for env,bank in frame.groupby('environment',sort=True):
        for method in cfg['methods']:
            cf=bank[bank.split=='calibration'];vals=cf.loc[(cf.norm>1e-6)&np.isfinite(cf[method]),method].to_numpy();grid=np.unique(np.r_[0.,1.,np.quantile(vals,np.linspace(0,1,41))]) if len(vals) else np.array([0.,1.]);recorded=curves[(curves.environment==env)&(curves.method==method)].reset_index(drop=True);check('threshold_grid',grid,recorded.tau,1e-10);best=None;bestvalue=None
            for i,tau in enumerate(grid):
                choices=selected(cf,method,{'enabled':True,'tau':float(tau)},cfg);active=choices[choices.active];contact=active[active.phase=='contact'];risk=float(active.violation.mean()) if len(active) else None;crisk=float(contact.violation.mean()) if len(contact) else None;af=float(choices.active.mean());value=(float((choices.active&(choices.predicted_gain>=.1)).mean()),float(choices.predicted_gain.mean()));valid=af>=.25 and risk is not None and risk<=.1
                if env=='pusht':valid=valid and crisk is not None and crisk<=.1
                row=recorded.iloc[i]
                for key,v in [('risk',risk),('contact_risk',crisk),('active_fraction',af),('predicted_useful',value[0]),('mean_predicted_gain',value[1])]:check('trial_'+key,np.nan if v is None else v,row[key],1e-10)
                assert bool(row.valid)==valid
                if valid and (bestvalue is None or value>bestvalue):best={'enabled':True,'tau':float(tau)};bestvalue=value
                trials+=1
            rule=best or {'enabled':False,'tau':None};assert rule['enabled']==rules[(env,method)]['enabled']
            if rule['enabled']:check('locked_threshold',rule['tau'],rules[(env,method)]['tau'],1e-10)
            for split in ['development','calibration']:
                v=selected(bank[bank.split==split],method,rule,cfg);v['environment']=env;v['method']=method;v['split']=split;rebuilt.append(v)
    frame=pd.concat(rebuilt,ignore_index=True);keys=['environment','split','method','state_id'];frame=frame.sort_values(keys).reset_index(drop=True);stored=pd.read_csv(out/'selected.csv').sort_values(keys).reset_index(drop=True);assert len(frame)==len(stored)
    for col in frame:
        if col in keys+['phase']:np.testing.assert_array_equal(frame[col],stored[col])
        else:check('selected_'+col,frame[col],stored[col],1e-10)
    analysis=read(out/'analysis.json');assert analysis['states']==records and analysis['selected_rows']==len(frame)
    for r in analysis['summary']:
        v=frame[(frame.environment==r['environment'])&(frame.split==r['split'])&(frame.method==r['method'])];v=v if r['phase']=='all' else v[v.phase==r['phase']];assert len(v)==r['states']
        for metric in ['active','joint','jerk_gain','predicted_gain','candidate_coverage','selected_risk']:
            g=v[v.active] if metric=='selected_risk' else v;col='violation' if metric=='selected_risk' else metric
            if not len(g):assert r[metric]['mean'] is None;continue
            avg,ci=cluster(g[col].to_numpy(),g.initial_pose_family.to_numpy());check('summary_mean',avg,r[metric]['mean'],1e-10);check('summary_ci',ci,r[metric]['ci95'],1e-10)
    for r in analysis['paired']:
        v=frame[(frame.environment==r['environment'])&(frame.split==r['split'])];a=v[v.method==r['method']].set_index('state_id');b=v[v.method==r['baseline']].set_index('state_id').loc[a.index];delta=a[r['metric']].astype(float)-b[r['metric']].astype(float);avg,ci=cluster(delta.to_numpy(),a.initial_pose_family.to_numpy());check('paired_mean',avg,r['mean'],1e-10);check('paired_ci',ci,r['ci95'],1e-10)
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'states':records,'candidate_rows':len(saved),'native_distance_arrays':native_arrays,'calibration_trials':trials,'selected_rows':len(frame),'max_errors':errors,'source_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'scope':'All native DINO distances, physical pose/coverage/jerk/norm, expected costs, non-WM membership and cap, independently recomputed calibration/selection/family intervals. PushT vision-contact scores reused from previously audited fixed source. Exact source arithmetic retained after verification to avoid CSV quantile boundary artifacts.'});print('AUDITED DINO SELECTION',records,len(saved),trials,len(frame),flush=True)

if __name__=='__main__':main()
