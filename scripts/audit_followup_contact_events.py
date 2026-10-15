"""Audit event-window physics, native LeWM features, constraints and locked choices."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.policy_refinement import make_scene,full_state
from zj_wam.diagnostics import polynomial_basis
from zj_wam.repeated_refinement import visual_history
from zj_wam.response_models import score_models
from scripts.lewm_runtime_reference import native_scores
from scripts.audit_followup_output_replacement import pd_positions
from scripts.analyze_followup_policy_final import Coverage
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('contact_events_smoke' if smoke else 'contact_events');manifest=read(out/'protocol.json');done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['protocol_sha256']==sha256(out/'protocol.json');cfg=read(resolve('configs/followup_v1.json'));rcfg=read(resolve('configs/followup_response_models.json'));torch.set_num_threads(2);data=ExpertDataset(cfg['dataset']);lewm=FrozenLeWM(cfg['checkpoint'],data.scaler,'cpu');errors={};count=total=forwards=branches=0;inputs={};q=polynomial_basis(5)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    def check(name,x,y,tol=1e-8,rtol=0):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(name,x.shape,y.shape);np.testing.assert_allclose(x,y,atol=tol,rtol=rtol,err_msg=name);finite=np.isfinite(x)&np.isfinite(y);err=float(abs(x[finite].astype(float)-y[finite].astype(float)).max()) if finite.any() else 0.;errors[name]=max(errors.get(name,0),err)
    for spec in manifest['states']:
        seed=spec['seed'];t=spec['step'];p=out/f'{seed}_t{t:03d}.json';m=read(p);assert m['protocol_sha256']==sha256(out/'protocol.json') and m['data_sha256']==sha256(p.with_suffix('.npz'));a=dict(np.load(p.with_suffix('.npz')));parent=dict(np.load(resolve(m['source'])));f=pd.read_csv(p.with_suffix('.csv'));n=len(a['actions']);assert len(f)==n and len(a['physics'])==n and np.array_equal(f.candidate,np.arange(n));assert a['physics'].shape==(n,110,10) and a['contacts'].shape==(n,80)
        check('causal_images',a['input_images'],parent['images'][[t-10,t-5,t]],0);check('causal_past_actions',a['input_past'],parent['actions'][t-10:t],0);check('causal_prefix',a['input_prefix'],parent['physics'][10*t-30:10*t],0);check('nominal_actions',a['actions'][0],parent['actions'][t:t+5],0);check('suffix',a['input_suffix'],parent['actions'][t+5:t+8],0);check('input_scale',a['input_scale'],data.scaler.scale_,0);check('input_mean',a['input_mean'],data.scaler.mean_,0);assert not a['input_proprio'][:,[2,3,4,7,8,9]].any() and not a['input_root'][[2,3,4,7,8,9]].any();check('root_proprio',a['input_root'][[0,1,5,6]],parent['states'][t,[0,1,5,6]],0);check('historical_proprio',a['input_proprio'][:,[0,1,5,6]],parent['states'][[t-10,t-5,t]][:,[0,1,5,6]],0)
        check('shared_physics_prefix',a['physics'][:,:30],np.broadcast_to(a['input_prefix'],(n,30,10)),0);check('nominal_dense_replay',a['physics'][0,30:],parent['physics'][10*t:10*(t+8)],0);assert abs(a['actions']).max()<=1+1e-6
        delta=((a['actions']-a['actions'][:1])/data.scaler.scale_).reshape(n,-1);check('polynomial_span',delta,delta@q@q.T,1e-9);norm=np.linalg.norm(delta,axis=1);check('norm',norm,a['norms']);check('csv_norm',norm,f.norm);assert np.all(norm<=f.radius.to_numpy()*np.sqrt(10)+2e-6)
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],seed,t,5]));directions=rng.normal(size=(8,q.shape[1]));directions/=np.linalg.norm(directions,axis=1)[:,None]
        for i,row in enumerate(f.itertuples()):
            label=a['labels'][i];assert row.generator==label[0] and row.radius==float(label[1]) and row.sign==int(label[2]);assert row.generator=='nominal' or row.generator in ['optimizer_executed','optimizer_raw'] or row.generator.startswith('random_')
            if row.generator.startswith('random_'):check('random_candidate',delta[i],int(row.sign)*row.radius*np.sqrt(10)*(q@directions[int(row.generator.split('_')[1])]),1e-9)
        robot=np.array([pd_positions(np.r_[x,a['input_suffix']],a['input_root']) for x in a['actions']]);check('independent_controller_response',robot,a['predicted_robot'],1e-9);points=np.concatenate([np.broadcast_to(a['input_prefix'][:,:2],(n,30,2)),robot],axis=1);cost=np.sum(np.diff(points,n=3,axis=1)**2,axis=(1,2))/.01**5;check('predicted_cost',cost,a['predicted_costs'],1e-4,1e-10);check('csv_predicted_cost',cost,f.predicted_cost,1e-4,1e-10);check('expected_gain',(cost[0]-cost)/max(cost[0],1e-12),f.predicted_gain)
        check('controller_score',np.linalg.norm(robot-robot[:1],axis=2).max(1),a['score_controller']);observed,valid=visual_history(a['input_images'],a['input_proprio'],rcfg);scores,_=score_models(a['actions'],a['input_suffix'],observed[-1],observed,rcfg);contact=scores['contact'] if valid else np.full(n,np.inf);check('recomputed_vision_contact',contact,a['score_contact']);assert bool(valid)==m['visual_contact_valid']
        physics=a['physics'];diff=physics[:,30:,2:5]-physics[:1,30:,2:5];pos=np.linalg.norm(diff[:,:,:2],axis=2)/2;rot=abs((diff[:,:,2]+np.pi)%(2*np.pi)-np.pi)/np.deg2rad(2);covfun=Coverage(seed);cv=np.array([[covfun(x[79]),covfun(x[109])] for x in physics]);check('independent_coverage',cv,a['coverage'],1e-10);obj=np.maximum.reduce([pos.max(1),rot.max(1),np.maximum(0,cv[:1]-cv).max(1)/.02]);check('dense_object_error',obj,f.dense_full,1e-8);check('endpoint_object_error',np.maximum(pos[:,49],rot[:,49]),f.true_endpoint,1e-8);check('dense_pose_error',np.maximum(pos.max(1),rot.max(1)),f.dense_pose,1e-8)
        points=physics[:,:,:2];third=points[:,3:]-3*points[:,2:-1]+3*points[:,1:-2]-points[:,:-3];jerk=np.sum(third**2,axis=(1,2))/.01**5;check('independent_actual_jerk',jerk,f.actual_jerk,1e-2,1e-8);check('actual_gain',(jerk[0]-jerk)/jerk[0] if jerk[0]>1e-8 else np.zeros(n),f.jerk_gain,1e-8);assert m['phase']==('contact' if a['contacts'][0].any() else 'free') and np.all(a['contacts']>=0);assert (f.phase==m['phase']).all()
        hist=a['history'];actual=a['actual'];pred=a['predicted'];assert hist.shape==(3,192) and actual.shape==pred.shape==(n,192)
        pixels=np.concatenate([a['input_images'],a['endpoint_images']]);fresh=[]
        with torch.no_grad():
            for begin in range(0,len(pixels),7):
                batch=torch.stack([lewm.transform(im) for im in pixels[begin:begin+7]]);hidden=lewm.model.encoder(batch,interpolate_pos_encoding=True).last_hidden_state[:,0];fresh.append(lewm.model.projector(hidden).cpu().numpy())
        fresh=np.concatenate(fresh);check('direct_native_history',hist,fresh[:3],2e-5);check('direct_native_actual_all',actual,fresh[3:],2e-5)
        predictor,_=lewm.predictor(a['input_images'],a['input_past']);freshpred=[]
        with torch.no_grad():
            for action in a['actions']:freshpred.append(predictor(torch.tensor(lewm.normalize(action).reshape(-1))).cpu().numpy());forwards+=1
        freshpred=np.array(freshpred);check('fresh_native_all_predicted',pred,freshpred,2e-5);check('fresh_native_all_score',np.linalg.norm(freshpred-freshpred[:1],axis=1),a['score_lewm_global'],2e-5)
        for method in manifest['config']['methods']:check('csv_'+method,a['score_'+method],f[method],1e-7)
        for i in sorted({0,n-1}):
            env=make_scene(seed)
            for action in parent['actions'][:t]:env.step(action)
            check('fresh_root',full_state(env),parent['states'][t],0);trace=[];freshcontacts=[];counter=[0];old=env.space.step
            def collide(arbiter,space,data):
                env._handle_collision(arbiter,space,data)
                bodies={s.body for s in arbiter.shapes}
                if env.agent in bodies and env.block in bodies:counter[0]+=len(arbiter.contact_point_set.points)
            env.space.on_collision(0,0,post_solve=collide)
            def step(dt):before=counter[0];old(dt);trace.append(full_state(env));freshcontacts.append(counter[0]-before)
            env.space.step=step
            for k,action in enumerate(np.r_[a['actions'][i],a['input_suffix']]):
                env.step(action)
                if k==4:check('fresh_endpoint_RGB',env.render(),a['endpoint_images'][i],0)
            env.space.step=old;env.close();check('fresh_physical_branch',trace,physics[i,30:],0);check('fresh_contact_branch',freshcontacts,a['contacts'][i],0);branches+=1
        decision=read(p.with_suffix('.decision.json'));assert decision['protocol_sha256']==sha256(out/'protocol.json')
        for method in manifest['config']['methods']:
            rule=manifest['rules'][method];good=np.flatnonzero((norm>1e-6)&((cost[0]-cost)/max(cost[0],1e-12)>1e-8)&(a['score_'+method]<=rule['tau'])) if rule['enabled'] else []
            chosen=min(good,key=lambda i:(cost[i],norm[i],i)) if len(good) else 0
            assert chosen==decision['selected_before_physics'][method]==m['selected_before_physics'][method]
        inputs[str(p.with_suffix('.decision.json').relative_to(resolve('.')))]=sha256(p.with_suffix('.decision.json'))
        for ext in ['.json','.npz','.csv']:inputs[str(p.with_suffix(ext).relative_to(resolve('.')))]=sha256(p.with_suffix(ext))
        count+=1;total+=n;print('AUDIT POLICY BANK',m['state_id'],n,flush=True)
    assert count==done['states']==(10 if smoke else 120) and total==done['candidates'];write_json(out/'raw_audit.json',{'smoke':smoke,'states':count,'candidates':total,'fresh_native_LeWM_predictions':forwards,'fresh_actual_endpoint_encodes':total,'fresh_physical_branches':branches,'max_errors':errors,'inputs':inputs,'protocol_sha256':sha256(out/'protocol.json'),'source_sha256':sha256(Path(__file__)),'audit_helpers':{p:sha256(resolve(p)) for p in ['scripts/audit_followup_output_replacement.py','scripts/analyze_followup_policy_final.py','scripts/lewm_runtime_reference.py']},'scope':'All raw physical metrics, independent polygon coverage and PD response, causal inputs, polynomial/random/norm/action constraints; LeWM all fresh native latent predictions and all actual endpoint/history features via direct encoder/projector;2fresh physical branches/state; same previously audited vision-contact model recomputed, no new independent perception implementation. All three locked choices reconstructed; no finaltest or policydeployment claim.'});print('AUDITED POLICY BANK',count,total,forwards,branches,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):main(args.smoke)
