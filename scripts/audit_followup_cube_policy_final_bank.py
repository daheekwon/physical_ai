"""Independent dense-physics and selector audit of the finite Cube policy bank."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_experiment import fit_response
from zj_wam.cube_execution import DenseCube,snapshot,restore
from scripts.run_followup_cube import Model
from scripts.run_followup_cube_native_policy import scene
from scripts.audit_followup_cube_policy_refinement import response,energy
from scripts.analyze_followup_response_selection import mean_ci
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');SRC=BASE/'cube_policy_refinement';read=lambda p:json.loads(p.read_text())


from scripts.cube_policy_final_bank_adapter import source_dir,split,rules as final_rules
from scripts.analyze_followup_cube_plan_chunks import independent_compile

def main(smoke):
    out=BASE/('cube_policy_final_bank_smoke' if smoke else 'cube_policy_final_bank');manifest=read(out/'manifest.json');cfg=manifest['config'];done=read(out/'execution_complete.json');assert done['manifest_sha256']==sha256(out/'manifest.json') and done['smoke']==smoke
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    torch.set_num_threads(cfg['torch_threads']);parent=read(SRC/'manifest.json');ccfg={**parent['cube'],'device':'cpu','yaw_length_scale_m':0.};native=parent['native'];norm=read(BASE/'cube/manifest.json')['normalization'];mean=np.array(norm['mean']);scale=np.array(norm['scale']);model=Model(ccfg,mean,scale);rules=final_rules();errors={};rows=[];states=[];candidateframes=[];inputs={};replays=0;freshpred=0
    def check(name,x,y,atol=1e-8,rtol=0):
        x=np.asarray(x);y=np.asarray(y);np.testing.assert_allclose(x,y,atol=atol,rtol=rtol,err_msg=name);err=float(np.max(abs(x.astype(float)-y.astype(float)))) if x.size else 0.;errors[name]=max(errors.get(name,0.),err)
    for seedtext,phases in manifest['phases'].items():
        seed=int(seedtext)
        with np.load(source_dir(seed)/f'{seed}_nominal.npz') as z:source=dict(z)
        fresh=True;env=None;current=0
        if fresh:
            env,obs,info,_,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);images={0:env.render().copy()};check('fresh_plan',agent._plan,source['plan'],0)
        for phase,step in sorted(phases.items(),key=lambda x:300 if x[1] is None else x[1]):
            if step is None:continue
            path=out/f'{seed}_{phase}_t{step:03d}.json';meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(path.with_suffix('.npz'));bi=meta['source_block'];get=lambda k:source[f'block_{bi:03d}_{k}'];frame=pd.read_csv(path.with_suffix('.csv'));a=dict(np.load(path.with_suffix('.npz')));n=len(a['actions']);assert n==meta['candidates']==len(frame);candidateframes.append(frame)
            before=path.with_name(path.stem+'.before_physics.npz');choicepath=path.with_name(path.stem+'.choice.json');choice=read(choicepath);assert meta['choice_sha256']==sha256(choicepath) and choice['before_physics_sha256']==sha256(before) and choice['final_protocol_sha256']==sha256(BASE/'cube_policy_final/final_protocol.json')
            with np.load(before) as pre:
                for k in pre.files:check('before_physics_'+k,pre[k],a[k],0)
            for q0 in [before,choicepath]:inputs[str(q0.relative_to(resolve('.')))]=sha256(q0)
            for ext in ['.json','.csv','.npz']:inputs[str(path.with_suffix(ext).relative_to(resolve('.')))]=sha256(path.with_suffix(ext))
            actions=a['actions'];assert actions.dtype==np.float32 and np.max(abs(actions))<=1.;indices=np.minimum(np.floor((float(get('root_time'))+np.arange(8)*.05-float(source['plan_t_init'])+1e-7)/.05).astype(int),len(source['plan'])-1);compiled,_,_=independent_compile(source['plan'][indices],get('root'),get('velocity'),float(get('gripper')),get('gains'),float(get('gripper_gain')),source['bounds'],source['action_low'],source['action_high'],'past_pd_h5');check('same_nominal_actions',actions[0],compiled[:5],0);check('suffix',a['input_suffix_actions'],compiled[5:],0);check('past_actions',a['input_past_actions'],source['actions'][step-10:step],0);check('past_points',a['input_past_points'],get('past_points')[-251:],0);check('past_targets',a['input_past_targets'],get('past_targets')[-250:],0);check('fixed_gripper',actions[:,:,4],np.broadcast_to(actions[0,:,4],actions.shape[:2]),0);q=a['basis'];check('basis_orthonormal',q.T@q,np.eye(q.shape[1]),1e-12);difference=((actions-actions[:1])/scale).reshape(n,-1);norms=np.linalg.norm(difference,axis=1);check('normalized_norm',norms,a['norm']);check('polynomial_membership',difference,difference@q@q.T,1e-5);check('coefficients_to_applied',actions,(actions[0]+(a['coefficients']@q.T).reshape(-1,5,5)*scale).astype(np.float32),0);radius=a['labels'][:,1].astype(float);assert np.all(norms<=radius+1e-5)
            gains,res=fit_response(a['input_past_points'],a['input_past_targets'],.002);p=a['input_past_points'];v=(p[-1]-p[-2])/.002;robot=np.array([response(np.r_[x,a['input_suffix_actions']],p[-1],v,gains,a['input_workspace_bounds']) for x in actions]);check('independent_robot_prediction',robot,a['predicted_robot'],1e-10);cost=np.array([energy(np.r_[p[-75:,:3],x[:,:3]]) for x in robot]);check('expected_cost',cost,a['costs'],1e-2,1e-8);check('expected_csv_cost',cost,frame.predicted_jerk_cost,1e-2,1e-8)
            for k in ['qpos','qvel','object_pose','object_velocity','eef_position','eef_quaternion','robot_object_contacts','time']:
                check('nominal_first5_parent_'+k,a['candidate_'+k][0,:125],source['physics_'+k][step*25:(step+5)*25],0);check('physical_prefix_'+k,a['prefix_'+k],source['physics_'+k][step*25-75:step*25],0)
            pose=a['candidate_object_pose'];position=np.linalg.norm(pose[:,:,:3]-pose[:1,:,:3],axis=2).max(1);rotation=[]
            for trace in pose:
                r=Rotation.from_quat(trace[:,[4,5,6,3]]);r0=Rotation.from_quat(pose[0][:,[4,5,6,3]]);rotation.append(float((r*r0.inv()).magnitude().max()))
            e=np.maximum(position/.002,np.array(rotation)/np.deg2rad(2));jerk=np.array([energy(np.r_[a['prefix_eef_position'],x]) for x in a['candidate_eef_position']]);gain=(jerk[0]-jerk)/max(jerk[0],1e-12);check('object_error',e,frame.object_error,2e-5);check('actual_jerk',jerk,frame.eef_jerk,1e-2,1e-8);check('actual_gain',gain,frame.jerk_gain,1e-8);check('same_branch_clock',a['candidate_time'],np.broadcast_to(a['candidate_time'][0],a['candidate_time'].shape),0);check('physical_clock',a['candidate_time'][0],float(get('root_time'))+.002*np.arange(1,201),1e-9)
            delta=robot-robot[:1];dp=np.linalg.norm(delta[:,:,:3],axis=2);dr=abs(np.arctan2(np.sin(delta[:,:,3]),np.cos(delta[:,:,3])));score={'controller_displacement':np.maximum(dp.max(1)/.002,dr.max(1)/np.deg2rad(2))};contact=bool(float(a['input_root_gripper_contact'])>.1 or np.any(np.r_[actions[0,:,4],a['input_suffix_actions'][:,4]]>.1));score['proprio_contact']=np.maximum((dp+np.sqrt(3)*.02*dr).max(1)/.002,dr.max(1)/np.deg2rad(2)) if contact else np.zeros(n);embedding=model.encode(a['input_history_images']);past=torch.tensor(((a['input_past_actions']-mean)/scale).astype(np.float32).reshape(1,2,25));x0=((actions[0]-mean)/scale).reshape(-1);latent=[]
            with torch.no_grad():
                for act in actions:
                    future=torch.tensor(x0+((act-actions[0])/scale).reshape(-1),dtype=torch.float32);latent.append(model.model.predict(embedding,model.model.action_encoder(torch.cat([past,future.reshape(1,1,25)],1)))[:,-1].reshape(-1).numpy());freshpred+=1
            latent=np.array(latent);score['lewm_global']=np.linalg.norm(latent-latent[:1],axis=1)
            for method in cfg['methods']:check('native_score_'+method,score[method],a[method+'_scores'],2e-5 if method=='lewm_global' else 1e-8)
            # Exact saved arithmetic was checked independently above; use it for
            # original <= threshold membership, not a different rounded float.
            cost=a['costs'];norms=a['norm'];pg=(cost[0]-cost)/max(cost[0],1e-12);active=norms>1e-6;choices={}
            for method in cfg['methods']:
                rule=rules[method];ok=active&(pg>1e-8)&(a[method+'_scores']<=rule['tau']) if rule['enabled'] else np.zeros(n,bool)
                ids=np.flatnonzero(ok);chosen=min(ids,key=lambda i:(cost[i],norms[i],i)) if len(ids) else 0;assert chosen==meta['selected'][method]==choice['selected'][method];choices[method]=int(chosen)
            def best(mask,values):
                ids=np.flatnonzero(active&mask&(values<values[0]-1e-10));return int(min(ids,key=lambda i:(values[i],i))) if len(ids) else 0
            choices.update(unrestricted_expected=best(np.ones(n,bool),cost),physical_filter_expected=best(e<=1,cost),finite_physical_oracle=best(e<=1,jerk),unrestricted_physical_oracle=best(np.ones(n,bool),jerk))
            for method,i in choices.items():rows.append({'split':split(seed),'seed':seed,'initial_pose_family':seed,'phase':phase,'step':step,'method':method,'candidate':i,'active':bool(active[i]),'E':float(e[i]),'violation':bool(active[i] and e[i]>1),'joint':bool(active[i] and e[i]<=1 and gain[i]>=.1),'jerk_gain':float(gain[i]),'predicted_gain':float(pg[i])})
            states.append({'split':split(seed),'seed':seed,'initial_pose_family':seed,'phase':phase,'step':step,'candidates':n,'best_expected_gain':float(pg.max()),'any_expected_10percent':bool(np.any(active&(pg>=.1))),'best_unrestricted_actual_gain':float(gain.max()),'best_preserved_actual_gain':float(gain[e<=1].max()),'oracle_U':bool(np.any(active&(e<=1)&(gain>=.1)))})
            if fresh:
                if step>current:
                    for t in range(current,step):
                        prefix=meter.rollout(source['actions'][t:t+1])['physics']
                        for k,vv in prefix.items():check('fresh_replay_prefix_'+k,vv,source['physics_'+k][t*25:(t+1)*25],0)
                        if (t+1)%5==0:images[t+1]=env.render().copy()
                    current=step
                check('fresh_history_images',a['input_history_images'],np.array([images[step-10],images[step-5],images[step]]),0);check('fresh_root_contact',a['input_root_gripper_contact'],np.asarray(env.get_step_info()['proprio/gripper_contact']).reshape(-1)[0],0);snap=snapshot(env)
                for i in sorted(set([0,n-1])):
                    restore(env,snap);trace=meter.rollout(np.r_[actions[i],a['input_suffix_actions']])['physics']
                    for k,vv in trace.items():check('fresh_branch_'+k,vv,a['candidate_'+k][i],0)
                    replays+=1
                restore(env,snap)
            print('AUDIT ORACLE',seed,phase,n,flush=True)
        if env is not None:env.close()
    f=pd.DataFrame(rows);sf=pd.DataFrame(states);allc=pd.concat(candidateframes,ignore_index=True);assert len(sf)==done['states'] and len(allc)==done['candidates'];summary=[]
    for method,g in f.groupby('method'):
        for phase in ['all',*sorted(g.phase.unique())]:
            v=g if phase=='all' else g[g.phase==phase];summary.append({'method':method,'phase':phase,'states':len(v),**{k:mean_ci(v,k) for k in ['active','joint','jerk_gain','predicted_gain']},'selected_risk':mean_ci(v[v.active],'violation')})
    paired=[]
    for method in ['unrestricted_expected','physical_filter_expected','finite_physical_oracle','unrestricted_physical_oracle']:
        x=f[f.method==method].set_index(['seed','phase']);y=f[f.method=='controller_displacement'].set_index(['seed','phase']).loc[x.index];v=x.reset_index();v['delta_U']=x.joint.to_numpy(float)-y.joint.to_numpy(float);v['delta_gain']=x.jerk_gain.to_numpy()-y.jerk_gain.to_numpy();paired.append({'method':method,'baseline':'controller_displacement','U_difference':mean_ci(v,'delta_U'),'gain_difference':mean_ci(v,'delta_gain')})
    analysis={'smoke':smoke,'states':len(sf),'candidates':len(allc),'summary':summary,'paired':paired,'candidate_potential':{k:mean_ci(sf,k) for k in ['best_expected_gain','any_expected_10percent','best_unrestricted_actual_gain','best_preserved_actual_gain','oracle_U']},'scope':cfg['scope']};f.to_csv(out/'selected.csv',index=False);sf.to_csv(out/'state_potential.csv',index=False);allc.to_csv(out/'candidates.csv',index=False);write_json(out/'analysis.json',analysis);write_json(out/'raw_audit.json',{'smoke':smoke,'states':len(sf),'candidates':len(allc),'selected_rows':len(f),'fresh_native_LeWM_predictions':freshpred,'fresh_physical_branches_all_families':replays,'max_errors':errors,'inputs':inputs,'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'source_sha256':sha256(Path(__file__)),'audit_helpers':{p:sha256(resolve(p)) for p in ['scripts/analyze_followup_cube_plan_chunks.py','scripts/audit_followup_cube_policy_refinement.py','scripts/analyze_followup_response_selection.py']},'scope':'All candidate geometry, quantization, past response, expected/actual jerk, fullSO3 dense preservation, old nominal exactness and every selector; fresh native LeWM for allcandidates, fresh first/last physical branches at every family phase, every root RGB history/contact and independent8command compilation. Final nominal-calibrated native rules checked with full cost/norm/id tie break; chosen before physicalevaluation. Finite-oracle summaries diagnostic only, finalprimary decision separate.'});print('AUDITED POLICY ORACLE',len(sf),len(allc),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_cube_policy_calibration_bank.json'))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(args.smoke)
