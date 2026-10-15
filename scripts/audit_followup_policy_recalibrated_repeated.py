"""Fresh repeated-policy replay, causal bank verification and local/global diagnosis."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dill,hydra,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.policy_refinement import make_scene,full_state
from zj_wam.repeated_refinement import visual_history
from zj_wam.response_models import score_models
from zj_wam.diagnostics import polynomial_basis
from zj_wam.model_audit import predict_candidates
from scripts.analyze_followup_policy_final import Coverage,binomial_interval
from scripts.audit_followup_dinowm_policy_final import interval
from scripts.dinowm_runtime_core import generate
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def energy(q):
    d=q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3];return float((d*d).sum()/.01**5)
def deviation(actual,reference,cov,refcov):
    d=actual[:,2:5]-reference[:,2:5];pos=np.linalg.norm(d[:,:2],axis=1);rot=abs(np.angle(np.exp(1j*d[:,2])));loss=np.maximum(0,refcov-cov);return max(pos.max()/2,rot.max()/np.deg2rad(2),loss.max()/.02)
def robot_response(actions,root):
    q=root[:2].copy();v=root[5:7].copy();out=[]
    for a in actions:
        target=q+100*a
        for _ in range(10):v+=.01*(100*(target-q)-20*v);q+=.01*v;out.append(q.copy())
    return np.array(out)

def main(smoke):
    out=BASE/('policy_recalibrated_repeated_smoke' if smoke else 'policy_recalibrated_repeated');manifest=read(out/'manifest.json');cfg=manifest['config'];done=read(out/'execution_complete.json');assert done['manifest_sha256']==sha256(out/'manifest.json') and done['smoke']==smoke
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    maximum=manifest['max_actions'];base=read(resolve('configs/followup_v1.json'));rcfg=read(resolve('configs/followup_response_models.json'));pcfg=read(resolve('configs/followup_dinowm_policy_nominal.json'));torch.set_num_threads(cfg['torch_threads']);data=ExpertDataset(base['dataset']);model=FrozenLeWM(base['checkpoint'],data.scaler,'cpu');payload=torch.load(resolve(pcfg['policy_checkpoint']),map_location='cpu',pickle_module=dill,weights_only=False);policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload;policy.to(cfg['policy_device']).eval().requires_grad_(False);errors={};inputs={};episodes=[];localrows=[];fresh_policy=0;fresh_local=0;fresh_lewm=0;candidate_count=0;choice_count=0
    def check(name,a,b,tol=1e-8,rtol=0):
        a=np.asarray(a);b=np.asarray(b);np.testing.assert_allclose(a,b,atol=tol,rtol=rtol,equal_nan=True,err_msg=name);d=abs(a.astype(float)-b.astype(float));errors[name]=max(errors.get(name,0.),float(np.nanmax(d)) if d.size and np.isfinite(d).any() else 0.)
    for source in manifest['sources']:
        p=resolve(source);seed=read(p)['seed'];reference=dict(np.load(p.with_suffix('.npz')));split=p.parent.name;covfun=Coverage(seed);refcov=np.array([covfun(x) for x in reference['states'][1:maximum+1]]);refphysics=reference['physics'][:maximum*10];J0=energy(np.r_[reference['initial_state'][None,:2],refphysics[:,:2]])
        for method in cfg['methods']:
            dest=out/split/f'{seed}_{method}.json';meta=read(dest);a=dict(np.load(dest.with_suffix('.npz')));local=pd.read_csv(dest.with_suffix('.csv'));assert meta['recalibrated_manifest_sha256']==sha256(out/'manifest.json') and meta['nominal_source']==source and len(a['actions'])==maximum and a['physics'].shape==(maximum*10,10);check('initial',a['initial_state'],reference['initial_state'],0);check('control_dense',a['states'][1:],a['physics'][9::10],0);assert abs(a['actions']).max()<=1+1e-6;cov=np.array([covfun(x) for x in a['states'][1:]]);check('coverage',cov,a['coverage'],1e-10);J=energy(np.r_[a['initial_state'][None,:2],a['physics'][:,:2]]);check('episode_jerk',J,meta['physics_jerk'],1e-2,1e-9);hits=np.flatnonzero(cov>.95);assert meta['any_success']==bool(len(hits)) and meta['first_success_action']==(int(hits[0])+1 if len(hits) else None);check('final_coverage',cov[-1],meta['final_coverage']);eligible_count=0;active_count=0
            for k,d in enumerate(meta['decisions']):
                t=d['control_step'];n=min(8,maximum-t);assert t==8*k and d['policy_observation_times']==[max(0,t-1),t] and d['policy_noise_seed']==(base['seed']+1009*seed+t)%(2**31-1);q=a['states'][t,:2].copy();v=a['states'][t,5:7].copy();targets=a['policy_action_predictions'][k,1:1+n];nominal=[]
                for target in targets:
                    action=np.clip((target-q)/100,-1,1);nominal.append(action);target=q+100*action
                    for _ in range(10):v+=.01*(100*(target-q)-20*v);q+=.01*v
                nominal=np.array(nominal);check('nominal_compile',nominal,a['nominal_relative_chunks'][k,:n],1e-9);assert d['eligible']==bool(t>=10 and n==8)
                if not d['eligible']:check('unrefined',a['actions'][t:t+n],nominal,1e-9);assert not d['active'];continue
                eligible_count+=1;bank=resolve(d['bank']);raw=dict(np.load(bank));decision=read(bank.with_suffix('.json'));assert d['bank_sha256']==decision['data_sha256']==sha256(bank) and d['bank_decision_sha256']==sha256(bank.with_suffix('.json'));assert decision['selected_before_execution'] and decision['manifest_sha256']==sha256(out/'manifest.json');assert decision['step']==t and decision['seed']==seed and decision['method']==method;ids=[t-10,t-5,t];assert d['wm_observation_times']==ids;proprio=np.zeros((3,10));proprio[:,[0,1,5,6]]=a['states'][ids][:,[0,1,5,6]]
                c={'episode':seed,'timestep':t,'images':a['images'][ids],'past':a['actions'][t-10:t],'proprio':proprio,'prefix':a['physics'][t*10-30:t*10],'root':proprio[-1],'nominal':nominal[:5],'suffix':nominal[5:8],'scale':data.scaler.scale_,'mean':data.scaler.mean_}
                for key,value in c.items():check('causal_'+key,value,raw['input_'+key],0 if key!='nominal' and key!='suffix' else 1e-9)
                count=len(raw['actions']);candidate_count+=count;actions,labels,_,_,_,solver=generate('pusht',c,base);check('regenerated_candidates',actions,raw['actions'],1e-8);assert labels==[tuple(x) for x in decision['labels']];difference=((raw['actions']-raw['actions'][:1])/data.scaler.scale_).reshape(count,-1);norms=np.linalg.norm(difference,axis=1);check('norms',norms,raw['norms']);basis=polynomial_basis(5);check('polynomial',difference,difference@basis@basis.T,1e-8);assert np.max(abs(raw['actions']))<=1+1e-6
                robot=np.array([robot_response(np.r_[x,c['suffix']],c['root']) for x in raw['actions']]);costs=np.array([energy(np.r_[c['prefix'][:,:2],x]) for x in robot]);check('expected_cost',costs,raw['costs'],1e-2,1e-9)
                if method=='controller':score=np.linalg.norm(robot-robot[:1],axis=2).max(1)
                elif method=='contact':
                    observed,valid=visual_history(c['images'],c['proprio'],rcfg);values,_=score_models(raw['actions'],c['suffix'],observed[-1],observed,rcfg);score=values['contact'] if valid else np.full(count,np.inf)
                else:
                    _,context=model.predictor(c['images'],c['past'])
                    with torch.no_grad():latent=predict_candidates(model,context['embedding'],context['past'],raw['actions'],batch_size=64)[:,0]
                    score=np.linalg.norm(latent-latent[:1],axis=1);fresh_lewm+=count
                check('fresh_score_'+method,score,raw['scores'],1e-7 if method=='lewm_global' else 1e-8);rule=manifest['rules'][method];gain=(raw['costs'][0]-raw['costs'])/max(raw['costs'][0],1e-12);allowed=[i for i in range(count) if rule['enabled'] and norms[i]>1e-6 and gain[i]>1e-8 and raw['scores'][i]<=rule['tau']];chosen=min(allowed,key=lambda i:(raw['costs'][i],norms[i],i)) if allowed else 0;assert chosen==decision['selected']==d['selected_index'];check('delivered_correction',a['actions'][t:t+5],raw['actions'][chosen],0);check('unchanged_suffix',a['actions'][t+5:t+8],c['suffix'],1e-9);active=bool(norms[chosen]>1e-6);assert active==d['active'];active_count+=active;check('selected_norm',d['norm'],norms[chosen]);check('predicted_gain',d['predicted_gain'],gain[chosen]);choice_count+=1
                index=eligible_count-1;original=a['local_nominal_physics'][index];actual=a['physics'][t*10:(t+8)*10];nomcov=np.array([covfun(x) for x in original[9::10]]);check('local_nominal_coverage',nomcov,a['local_nominal_coverage'][index],1e-10);E=deviation(actual,original,cov[t:t+8][[4,7]],nomcov[[4,7]]);j0=energy(np.r_[c['prefix'][:,:2],original[:,:2]]);j1=energy(np.r_[c['prefix'][:,:2],actual[:,:2]]);g=(j0-j1)/j0 if j0>1e-8 else 0.;r=local.iloc[index];assert r.control_step==t and r.active==active;check('local_E',E,r.object_error);check('local_gain',g,r.jerk_gain);localrows.append({'split':split,'seed':seed,'initial_pose_family':seed,'method':method,'step':t,'active':active,'object_error':E,'violation':bool(E>1),'jerk_gain':g,'joint':bool(active and E<=1 and g>=.1)})
                # Counterfactual replay is evaluation-only and occurs after the episode.
                env=make_scene(seed)
                for x in a['actions'][:t]:env.step(x)
                trace=[];old=env.space.step
                def step(dt):old(dt);trace.append(full_state(env))
                env.space.step=step
                for x in nominal:env.step(x)
                env.space.step=old;env.close();check('fresh_local_nominal_physics',trace,original,0);fresh_local+=1
                for p1 in [bank,bank.with_suffix('.json')]:inputs[str(p1.relative_to(resolve('.')))]=sha256(p1)
            assert eligible_count==len(local)==meta['eligible_chunks'] and active_count==meta['active_chunks'];active=bool(active_count);first=next((x['control_step'] for x in meta['decisions'] if x['active']),maximum);check('precorrection_actions',a['actions'][:first],reference['actions'][:first],0);check('precorrection_physics',a['physics'][:first*10],refphysics[:first*10],0)
            env=make_scene(seed);trace=[];old=env.space.step;observations=[];policy.reset()
            def step(dt):old(dt);trace.append(full_state(env))
            env.space.step=step
            for t in range(maximum+1):
                check('fresh_control',full_state(env),a['states'][t],0);check('fresh_RGB',env.render(),a['images'][t],0);size=env.render_size;env.render_size=96;rgb=np.moveaxis(env.render().astype(np.float32)/255.,-1,0);env.render_size=size;observations.append({'image':rgb,'agent_pos':np.asarray(env.agent.position,dtype=np.float32)})
                if t in cfg['fresh_policy_check_steps'] and t<maximum:
                    ids=[t-1,t];batch={key:torch.from_numpy(np.stack([observations[i][key] for i in ids])[None]).to(cfg['policy_device'],dtype=torch.float32) for key in observations[-1]};torch.manual_seed((base['seed']+1009*seed+t)%(2**31-1))
                    with torch.no_grad():prediction=policy.predict_action(batch)
                    check('fresh_policy',prediction['action_pred'][0].cpu().numpy(),a['policy_action_predictions'][t//8],1e-5);fresh_policy+=1
                if t<maximum:env.step(a['actions'][t])
            env.space.step=old;env.close();check('fresh_dense',trace,a['physics'],0);E=deviation(a['physics'],refphysics,cov,refcov);gain=(J0-J)/J0 if J0>1e-8 else 0.;episodes.append({'split':split,'seed':seed,'initial_pose_family':seed,'method':method,'active':active,'active_chunks':active_count,'object_error':E,'violation':bool(E>1),'jerk_gain':gain,'joint':bool(active and E<=1 and gain>=.1),'any_success':bool(len(hits)),'final_coverage':cov[-1],'refinement_seconds':meta['refinement_seconds']})
            for ext in ['.json','.npz','.csv']:inputs[str(dest.with_suffix(ext).relative_to(resolve('.')))]=sha256(dest.with_suffix(ext))
            print('AUDIT RECALIBRATED',split,seed,method,'blocks',eligible_count,flush=True)
    f=pd.DataFrame(episodes);local=pd.DataFrame(localrows);summary=[];local_summary=[];paired=[]
    for (split,method),x in f.groupby(['split','method']):
        summary.append({'split':split,'method':method,'episodes':len(x),**{key:interval(x,key) for key in ['active','jerk_gain','joint','any_success','final_coverage','refinement_seconds']},'selected_risk':interval(x[x.active],'violation'),'risk_exact':binomial_interval(int(x[x.active].violation.sum()),int(x.active.sum()))})
        if method!='controller':
            baseframe=f[(f.split==split)&(f.method=='controller')].set_index('seed').loc[x.seed]
            for key in ['joint','jerk_gain','violation']:v=x.copy();v['delta']=x[key].to_numpy(float)-baseframe[key].to_numpy(float);paired.append({'split':split,'method':method,'baseline':'controller','metric':key,**interval(v,'delta')})
    for (split,method),x in local.groupby(['split','method']):local_summary.append({'split':split,'method':method,'blocks':len(x),**{key:interval(x,key) for key in ['active','jerk_gain','joint']},'selected_risk':interval(x[x.active],'violation')})
    assert len(f)==done['episodes'] and len(local)==done['eligible_blocks']==choice_count==fresh_local
    f.to_csv(out/'evaluation.csv',index=False);local.to_csv(out/'local_evaluation.csv',index=False);write_json(out/'analysis.json',{'smoke':smoke,'episodes':len(f),'families':int(f.seed.nunique()),'blocks':len(local),'summary':summary,'local_summary':local_summary,'paired':paired,'scope':cfg['scope']})
    write_json(out/'raw_audit.json',{'smoke':smoke,'episodes':len(f),'families':int(f.seed.nunique()),'blocks':len(local),'candidate_rows':candidate_count,'fresh_native_LeWM_predictions':fresh_lewm,'fresh_native_policy_predictions':fresh_policy,'fresh_local_nominal_branches':fresh_local,'all_episode_physics_and_RGB_replayed_exact':True,'max_errors':errors,'inputs':inputs,'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'source_sha256':sha256(Path(__file__)),'audit_helpers':{p:sha256(resolve(p)) for p in ['scripts/analyze_followup_policy_final.py','scripts/audit_followup_dinowm_policy_final.py']},'scope':'All causal inputs, candidate regeneration/geometry, independent controller dynamics and cost, fresh scores, locked selection, full physical/RGB replay, every local nominal branch and dense local/global geometry. Three fixed native policy calls per full episode; nominal parent audited separately. Development/calibration repeated transfer only.'});print('AUDITED RECALIBRATED REPEATED',len(f),len(local),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=read(resolve('configs/followup_policy_recalibrated_repeated.json'))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(a.smoke)
