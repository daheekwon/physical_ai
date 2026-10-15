"""Audit new-expert nominal semantics, fixed windows, all outcomes and locked selections."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.policy_refinement import full_state
from zj_wam.diagnostics import polynomial_basis
from scripts.audit_followup_original_expert_adapter import scene
from scripts.original_expert_evaluation_core import original
from scripts.model_runtime_imports import load_dino
from scripts.lewm_runtime_reference import native_scores
from scripts.audit_followup_output_replacement import pd_positions
from scripts.analyze_followup_policy_final import Coverage
from scripts.analyze_followup_response_selection import mean_ci
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('original_expert_final_smoke' if smoke else 'original_expert_final');manifest=read(out/'manifest.json');cfg=manifest['config'];done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['manifest_sha256']==sha256(out/'manifest.json');rules=manifest['rules'];base=read(resolve('configs/followup_v1.json'));torch.set_num_threads(2);data=ExpertDataset(base['dataset']);lewm=FrozenLeWM(base['checkpoint'],data.scaler,'cpu');dino=load_dino('pusht');errors={};rows=[];nominal_rows=[];inputs={};branches=forwards=candidates=states=0;archive=resolve('data/datasets/diffusion-policy-official/pusht.zip');q=polynomial_basis(5)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    if not smoke:
        lock=read(out/'final_protocol.json');assert lock['rules']==rules and lock['test_original_episodes']==manifest['episodes']
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
    def check(name,x,y,tol=1e-8,rtol=0):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(name,x.shape,y.shape);np.testing.assert_allclose(x,y,atol=tol,rtol=rtol,err_msg=name);finite=np.isfinite(x)&np.isfinite(y);errors[name]=max(errors.get(name,0),float(abs(x[finite].astype(float)-y[finite].astype(float)).max()) if finite.any() else 0.)
    for ep in manifest['episodes']:
        nompath=out/f'e{ep:03d}_nominal.npz';n=dict(np.load(nompath));recorded,targets=original(archive,ep);check('original_states',recorded,n['recorded_states'],0);check('original_absolute_targets',targets,n['absolute_targets'],0);check('initial_pose',recorded[0],n['initial_pose'],0);steps=[10+int((len(targets)-18)*f) for f in cfg['window_fractions']];assert len(set(steps))==3;env=scene(recorded[0],False);trace=[];old=env.space.step
        def step(dt):old(dt);trace.append(full_state(env))
        env.space.step=step
        for k,target in enumerate(targets):
            check('absolute_nominal_control',full_state(env),n['states'][k],1e-8);check('absolute_nominal_RGB',env.render(),n['images'][k],0);check('independent_relative_conversion',(target-np.array(env.agent.position))/100.,n['actions'][k],1e-10);env.step(target)
        env.space.step=old;env.close();check('fresh_absolute_nominal_dense',trace,n['physics'],1e-8);check('last_control',n['states'][-1],n['physics'][-1],0);covfun=Coverage(ep);check('nominal_independent_coverage',[covfun(x) for x in n['states'][1:]],n['coverage'],1e-10);deviation=n['states'][:-1,:5]-recorded;angle=abs(np.arctan2(np.sin(deviation[:,4]),np.cos(deviation[:,4])));nominal_rows.append({'original_episode':ep,'actions':len(targets),'relative_prefix_outside_bound':int((abs(n['actions']).max(1)>1+cfg['nominal_bound_tolerance']).sum()),'recorded_position_max_px':float(np.linalg.norm(deviation[:,2:4],axis=1).max()),'recorded_rotation_max_deg':float(np.rad2deg(angle.max()))})
        for t in steps:
            p=out/f'e{ep:03d}_t{t:03d}.json';m=read(p);a=dict(np.load(p.with_suffix('.npz')));f=pd.read_csv(p.with_suffix('.csv'));decision=read(p.with_suffix('.decision.json'));assert m['data_sha256']==sha256(p.with_suffix('.npz')) and m['nominal_sha256']==sha256(nompath) and m['manifest_sha256']==sha256(out/'manifest.json');N=len(a['actions']);assert len(f)==N and m['step']==t and m['original_episode']==ep;eligible=bool(abs(n['actions'][t:t+8]).max()<=1+cfg['nominal_bound_tolerance']);assert eligible==m['eligible']==decision['eligible'];assert decision['selected_before_physics']==m['selected_before_physics'];check('history_images',a['input_images'],n['images'][[t-10,t-5,t]],0);check('past_actions',a['input_past'],n['actions'][t-10:t],0);check('nominal_chunk',a['actions'][0],n['actions'][t:t+5],0);check('suffix',a['input_suffix'],n['actions'][t+5:t+8],0);check('prefix',a['input_prefix'],n['physics'][t*10-30:t*10],0);assert not a['input_root'][[2,3,4,7,8,9]].any() and not a['input_proprio'][:,[2,3,4,7,8,9]].any();check('root_pusher',a['input_root'][[0,1,5,6]],n['states'][t,[0,1,5,6]],0);check('history_proprio',a['input_proprio'][:,[0,1,5,6]],n['states'][[t-10,t-5,t]][:,[0,1,5,6]],0)
            check('scale',a['input_scale'],data.scaler.scale_,0);check('mean',a['input_mean'],data.scaler.mean_,0);check('shared_prefix',a['physics'][:,:30],np.broadcast_to(a['input_prefix'],(N,30,10)),0);check('nominal_physics',a['physics'][0,30:],n['physics'][10*t:10*(t+8)],0);delta=((a['actions']-a['actions'][:1])/data.scaler.scale_).reshape(N,-1);check('polynomial_span',delta,delta@q@q.T,1e-9);norm=np.linalg.norm(delta,axis=1);check('norm',norm,a['norms']);check('csv_norm',norm,f.norm)
            if eligible:
                assert abs(a['actions']).max()<=1+1e-6;assert np.all(norm<=f.radius.to_numpy()*np.sqrt(10)+2e-6);robot=np.array([pd_positions(np.r_[action,a['input_suffix']],a['input_root']) for action in a['actions']]);check('independent_PD_response',robot,a['predicted_robot'],1e-9);points=np.concatenate([np.broadcast_to(a['input_prefix'][:,:2],(N,30,2)),robot],axis=1);cost=np.sum(np.diff(points,n=3,axis=1)**2,axis=(1,2))/.01**5;check('expected_cost',cost,a['predicted_costs'],1e-4,1e-10);check('controller_score',np.linalg.norm(robot-robot[:1],axis=2).max(1),a['score_controller']);check('native_CPU_LeWM',native_scores('pusht',lewm,{'images':a['input_images'],'past':a['input_past']},a['actions']),a['score_lewm_global'],2e-5);tokens=a['dino_predicted'].astype(float);d=tokens-tokens[:1];check('DINO_patch_RMS',np.linalg.norm(d.reshape(N,-1),axis=1)/14,a['score_dino_patch_rms'],1e-10);check('DINO_patch_mean',np.linalg.norm(d.mean(1),axis=1),a['score_dino_patch_mean'],1e-10);encoded=dino.encode(a['input_images'])
                for i in (range(N) if smoke else sorted({0,N-1})):
                    acts=torch.tensor(np.r_[a['input_past'],a['actions'][i]],dtype=torch.float32);normalized=(acts-torch.tensor(dino.stats['action_mean'],dtype=torch.float32))/torch.tensor(dino.stats['action_std'],dtype=torch.float32)
                    with torch.no_grad():pred=dino.model.rollout_last_visual(encoded,normalized.reshape(1,-1,10))[0].numpy()
                    check('fresh_native_DINO',pred,a['dino_predicted'][i],0);forwards+=1
            else:assert N==1 and not norm.any() and all(i==0 for i in m['selected_before_physics'].values())
            physics=a['physics'];de=physics[:,30:,2:5]-physics[:1,30:,2:5];pos=np.linalg.norm(de[:,:,:2],axis=2)/2;ang=abs((de[:,:,2]+np.pi)%(2*np.pi)-np.pi)/np.deg2rad(2);cv=np.array([[covfun(x[79]),covfun(x[109])] for x in physics]);check('independent_coverage',cv,a['coverage'],1e-10);E=np.maximum.reduce([pos.max(1),ang.max(1),np.maximum(0,cv[:1]-cv).max(1)/.02]);points=physics[:,:,:2];third=points[:,3:]-3*points[:,2:-1]+3*points[:,1:-2]-points[:,:-3];J=np.sum(third**2,axis=(1,2))/.01**5;gain=(J[0]-J)/J[0] if J[0]>1e-8 else np.zeros(N);check('dense_object_E',E,f.dense_full,1e-8);check('actual_gain',gain,f.jerk_gain,2e-8);check('actual_jerk',J,f.actual_jerk,1e-2,1e-8);phase='contact' if a['contacts'][0].any() else 'free';assert phase==m['phase']
            # Audit physical branches using the saved fixed relative prefix,
            # without the collection helper or object-contact callback.
            for i in sorted({0,N-1}):
                env=scene(recorded[0],True)
                for action in n['actions'][:t]:env.step(action)
                tr=[];old=env.space.step
                def step(dt):old(dt);tr.append(full_state(env))
                env.space.step=step
                for k,action in enumerate(np.r_[a['actions'][i],a['input_suffix']]):
                    env.step(action)
                    if k==4:check('fresh_endpoint_RGB',env.render(),a['endpoint_images'][i],0)
                env.space.step=old;env.close();check('fresh_branch',tr,physics[i,30:],0);branches+=1
            expected=(a['predicted_costs'][0]-a['predicted_costs'])/max(a['predicted_costs'][0],1e-12);check('csv_expected_gain',expected,f.predicted_gain);good=(norm>1e-6)&(E<=1)&(gain>=cfg['meaningful_gain'])
            for method in cfg['methods']:
                rule=rules[method];ids=[i for i in range(N) if eligible and rule['enabled'] and norm[i]>1e-6 and expected[i]>1e-8 and a['score_'+method][i]<=rule['tau']];i=min(ids,key=lambda i:(a['predicted_costs'][i],norm[i],i)) if ids else 0;assert i==m['selected_before_physics'][method];active=i!=0;rows.append({'original_episode':ep,'initial_pose_family':m['initial_pose_family'],'state_id':m['state_id'],'phase':phase,'eligible':eligible,'method':method,'candidate':i,'active':active,'object_error':float(E[i]),'violation':bool(active and E[i]>1),'jerk_gain':float(gain[i]),'joint':bool(active and E[i]<=1 and gain[i]>=cfg['meaningful_gain']),'finite_oracle_useful':bool(good.any())})
            for ext in ['.json','.npz','.csv','.decision.json']:inputs[str(p.with_suffix(ext).relative_to(resolve('.')))]=sha256(p.with_suffix(ext))
            states+=1;candidates+=N;print('AUDIT ORIGINAL EXPERT',ep,t,N,eligible,flush=True)
        inputs[str(nompath.relative_to(resolve('.')))]=sha256(nompath)
    assert states==done['states']==(3 if smoke else 63) and candidates==done['candidates'];f=pd.DataFrame(rows);f.to_csv(out/'selected.csv',index=False);pd.DataFrame(nominal_rows).to_csv(out/'nominal_quality.csv',index=False);summary=[];paired=[]
    for method,g in f.groupby('method'):
        summary.append({'method':method,'states':len(g),'families':g.initial_pose_family.nunique(),**{k:mean_ci(g,k) for k in ['joint','active','jerk_gain','eligible','finite_oracle_useful']},'selected_risk':mean_ci(g[g.active],'violation')})
    for method,baseline in [('dino_patch_rms','lewm_global'),('dino_patch_mean','lewm_global'),('lewm_global','controller'),('contact','controller'),('dino_patch_rms','contact')]:
        x=f[f.method==method].merge(f[f.method==baseline],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
        for metric in ['joint','active','jerk_gain']:
            x['difference']=x[metric].astype(float)-x[metric+'_base'].astype(float);stat=mean_ci(x,'difference');avg,ci=cluster(x.difference.to_numpy(),x.initial_pose_family.to_numpy());check('paired_mean',avg,stat['mean']);check('paired_interval',ci,stat['ci95']);paired.append({'method':method,'baseline':baseline,'metric':metric,**stat})
    for row in summary:
        g=f[f.method==row['method']]
        for metric in ['joint','active','jerk_gain','eligible','finite_oracle_useful','selected_risk']:
            x=g[g.active] if metric=='selected_risk' else g;col='violation' if metric=='selected_risk' else metric
            if not len(x):assert row[metric]['mean'] is None;continue
            avg,ci=cluster(x[col].to_numpy(),x.initial_pose_family.to_numpy());check('summary_mean',avg,row[metric]['mean']);check('summary_interval',ci,row[metric]['ci95'])
    write_json(out/'analysis.json',{'smoke':smoke,'states':states,'families':len(manifest['episodes']),'candidates':candidates,'selections':len(f),'ineligible_states':int((~f[f.method==cfg['methods'][0]].eligible).sum()),'summary':summary,'paired':paired,'scope':cfg['scope']});write_json(out/'raw_audit.json',{'smoke':smoke,'states':states,'candidates':candidates,'selections':len(f),'fresh_native_DINO_predictions':forwards,'fresh_physical_branches':branches,'max_errors':errors,'source_sha256':sha256(Path(__file__)),'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'inputs':inputs,'audit_helpers':{p:sha256(resolve(p)) for p in ['scripts/audit_followup_output_replacement.py','scripts/analyze_followup_policy_final.py','scripts/lewm_runtime_reference.py','scripts/audit_retrained_dinowm_readouts.py','scripts/analyze_followup_response_selection.py']},'scope':'All original nominal absolute-vs-relative replay, fixedfractions/causalinputs/bounds/polynomialspan, rawphysicalpreservation/jerk/coverage and prephysicsselection, familyintervals checked. EveryLeWMscore freshsingleforward; DINO allsmoke orfirstlast eachfullstate; freshnominal/lastcandidatephysicalbranch. Visioncontact uses previouslyauditedsharedmodel; no separate perceptionimplementation.'});print('AUDITED ORIGINAL EXPERT',states,candidates,len(f),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):main(args.smoke)
