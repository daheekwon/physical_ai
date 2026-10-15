"""Independent causal compiler, applied-candidate, branch and episode audit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,mujoco,torch
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_experiment import fit_response
from scripts.run_followup_cube_native_policy import scene
from scripts.run_followup_cube import Model
from scripts.analyze_followup_cube_plan_chunks import independent_compile,wrap,yaw
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def response(actions,root,velocity,gains,bounds):
    position=root.copy();v=velocity.copy();points=[]
    for a in actions:
        target=position+np.asarray(a[:4],float)*np.array([.05,.05,.05,.3]);target[:3]=np.clip(target[:3],bounds[0],bounds[1])
        for _ in range(25):
            error=target-position;error[3]=wrap(error[3]);v=v+.002*(gains[:,0]*error-gains[:,1]*v);position=position+.002*v;points.append(position.copy())
    return np.array(points)

def energy(p):return float(np.sum((p[3:]-3*p[2:-1]+3*p[1:-2]-p[:-3])**2)/.002**5)
def physical_error(p,q):
    pos=np.linalg.norm(p[:,:3]-q[:,:3],axis=1);a=Rotation.from_quat(p[:,[4,5,6,3]]);b=Rotation.from_quat(q[:,[4,5,6,3]]);ang=(a*b.inv()).magnitude();return max(float(pos.max()/.002),float(ang.max()/np.deg2rad(2))),float(pos.max()),float(ang.max())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('cube_policy_refinement_smoke' if args.smoke else 'cube_policy_refinement');m=read(out/'manifest.json');cfg=m['config'];compiler=m['compiler'];native=m['native'];done=read(out/'execution_complete.json');assert done['smoke']==args.smoke and done['manifest_sha256']==sha256(out/'manifest.json')
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    torch.set_num_threads(cfg['threads']);norm=read(BASE/'cube/manifest.json')['normalization'];mean=np.array(norm['mean']);scale=np.array(norm['scale']);model=Model(m['cube'],mean,scale);rules=read(BASE/'cube/proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules'];env,_,_,_,_=scene(cfg['development_seeds'][0],native);physical_model=env._model;grip=env._gripper_opening_joint_id;site=env._pinch_site_id;scratch=mujoco.MjData(physical_model);errors={};rows=[];localrows=[];solverrows=[];blocks=0;forwards=0;exact=0
    def check(name,x,y,tol=1e-10,rtol=0.):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(name,x.shape,y.shape);assert np.array_equal(np.isnan(x),np.isnan(y)),name;ok=np.isfinite(x)&np.isfinite(y);err=float(abs(x[ok].astype(float)-y[ok].astype(float)).max()) if ok.any() else 0.;errors[name]=max(errors.get(name,0.),err);assert np.all(abs(x[ok].astype(float)-y[ok].astype(float))<=tol+rtol*abs(y[ok])),(name,err,tol)
    seeds=cfg['development_seeds'][:1] if args.smoke else cfg['development_seeds']
    for seed in seeds:
      refpath=out/f'{seed}_nominal.npz'
      with np.load(refpath,allow_pickle=False) as z:reference={k:z[k] for k in ['physics_object_pose','physics_eef_position','goal_position','goal_quaternion','initial_qpos','initial_qvel']}
      for method in cfg['methods']:
        p=out/f'{seed}_{method}.json';meta=read(p);assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(p.with_suffix('.npz'))
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:a=dict(z)
        if args.smoke:
            assert meta['initial_smoke_physics_exact']
            with np.load(BASE/'cube_policy_refinement_initial_smoke_v1'/p.with_suffix('.npz').name) as old:
                for key in ['actions','success','physics_qpos','physics_qvel','physics_object_pose','physics_eef_position','physics_time']:check('initial_smoke_exact_'+key,a[key],old[key],0.)
        n=native['control_steps'];assert a['actions'].shape==(n,5) and a['actions'].dtype==np.float32 and abs(a['actions']).max()<=1
        for k in ['goal_position','goal_quaternion','initial_qpos','initial_qvel']:check('paired_'+k,a[k],reference[k],0.)
        for k in a:
            if k.startswith('control_'):check('dense_'+k,a[k],a['physics_'+k[8:]][24::25],0.)
        check('physical_clock',a['physics_time'],float(a['initial_time'])+.002*np.arange(1,n*25+1),1e-9)
        if method=='nominal':
            with np.load(BASE/'cube_plan_chunks'/f'{seed}_past_pd_h5.npz') as old:
                for k in ['actions','success','physics_qpos','physics_qvel','physics_object_pose','physics_eef_position','physics_time']:check('nominal_exact_'+k,a[k],old[k],0.)
            exact+=1
        grippers=np.clip(np.r_[a['initial_qpos'][grip],a['control_qpos'][:,grip]]/.8,0,1);roots=[]
        for step in range(n):
            if step==0:
                e,_,inf,_,_=scene(seed,native);root=np.r_[inf['proprio/effector_pos'],inf['proprio/effector_yaw'][0]];e.close()
            else:
                scratch.qpos[:]=a['physics_qpos'][step*25-2];mujoco.mj_kinematics(physical_model,scratch);mat=scratch.site_xmat[site].reshape(3,3);root=np.r_[scratch.site_xpos[site],np.arctan2(mat[1,0],mat[0,0])]
            roots.append(root)
        roots=np.array(roots);unwrapped=np.unwrap(np.r_[roots[0,3],yaw(a['physics_eef_quaternion'])]);points=np.c_[np.r_[a['initial_eef_position'][None],a['physics_eef_position']],unwrapped];delta=.5*(a['actions']+1)*(a['action_high']-a['action_low'])+a['action_low'];targets=roots+delta[:,:4];targets[:,:3]=np.clip(targets[:,:3],a['bounds'][0],a['bounds'][1]);targets[:,3]=roots[:,3]+wrap(targets[:,3]-roots[:,3]);targets[:,3]+=2*np.pi*np.round((unwrapped[1::25]-targets[:,3])/(2*np.pi));pasttargets=np.repeat(targets,25,axis=0);griptargets=np.clip(grippers[:-1]+delta[:,4],0,1);image_lookup={int(s):i for i,s in enumerate(a['history_image_control_indices'])};step=0;activeblocks=0
        for bi in range(meta['blocks']):
            get=lambda k:a[f'block_{bi:03d}_{k}'];h=1 if step<cfg['warmup_controls'] else 5;assert int(get('step'))==step and int(get('horizon'))==h and int(get('last_observation_control'))==step;count=min(step*25,compiler['fit_physics_history']);check('causal_root',get('root'),roots[step],1e-12);check('past_points',get('past_points'),points[step*25-count:step*25+1],1e-11);check('past_targets',get('past_targets'),pasttargets[step*25-count:step*25],1e-11);check('gripper_history',get('gripper_history'),grippers[:step+1],0.);check('gripper_targets',get('gripper_targets'),griptargets[:step],1e-12);vel=(points[step*25]-points[step*25-1])/.002 if step else np.zeros(4);check('past_velocity',vel,get('velocity'),1e-9)
            if count>=2:
                gains,res=fit_response(get('past_points'),get('past_targets'),.002);check('refit',gains,get('gains'),1e-8);check('refit_residual',res,get('residual'),1e-8)
            else:gains=np.tile(compiler['default_pd'],(4,1));check('default_gains',gains,get('gains'),0.)
            er=griptargets[:step]-grippers[:step];change=np.diff(grippers[:step+1]);den=er@er;gg=float(np.clip(er@change/den,0,1)) if den>1e-10 else compiler['default_gripper_gain'];check('gripper_fit',gg,get('gripper_gain'),1e-12);width=h+(3 if h==5 else 0);index=np.minimum(np.floor((float(get('root_time'))+.05*np.arange(width)-float(a['plan_t_init'])+1e-7)/.05).astype(int),len(a['plan'])-1);check('known_plan_indices',index,get('plan_indices'),0.);check('known_plan',a['plan'][index],get('targets'),0.)
            compiled,pr,pg=independent_compile(get('targets'),get('root'),get('velocity'),float(get('gripper')),get('gains'),float(get('gripper_gain')),a['bounds'],a['action_low'],a['action_high'],'past_pd_h5');check('nominal_compiler',compiled[:h],get('nominal_actions'),0.);check('suffix_compiler',compiled[h:],get('suffix_actions'),0.);check('compiler_prediction',pr,get('predicted'),1e-11);check('compiler_gripper',pg,get('predicted_gripper'),1e-12);check('executed_chunk',get('actions'),a['actions'][step:step+h],0.);check('gripper_coordinate_fixed',get('actions')[:,4],compiled[:h,4],0.)
            selected=int(get('selected'))
            if h==5:
                check('past_actions',get('past_actions'),a['actions'][step-10:step],0.);check('history_indices',get('history_control_indices'),[step-10,step-5,step],0.);check('causal_images',get('history_images'),a['history_images'][[image_lookup[k] for k in [step-10,step-5,step]]],0.)
                if method!='nominal':
                    bank=lambda k:a[f'bank_{bi:03d}_{k}'];actions=bank('actions');q=bank('basis');assert actions.dtype==np.float32 and abs(actions).max()<=1;check('bank_nominal',actions[0],compiled[:5],0.);check('bank_gripper_fixed',actions[:,:,4],np.broadcast_to(compiled[:5,4],actions.shape[:2]),0.);check('basis_orthonormal',q.T@q,np.eye(q.shape[1]),1e-12);diff=((actions-compiled[:5])/scale).reshape(len(actions),-1);norms=np.linalg.norm(diff,axis=1);check('applied_norm',norms,bank('norm'),1e-10);check('polynomial_quantization',diff,diff@q@q.T,cfg['polynomial_quantization_tolerance']);check('quantized_coefficients',actions,(compiled[:5]+(bank('coefficients')@q.T).reshape(-1,5,5)*scale).astype(np.float32),0.)
                    labels=bank('labels');assert len(labels)==len(actions);assert len(bank('solver_success'))==len(bank('solver_iterations'))==len(actions);diagnostics=bank('optimizer_diagnostics');check('optimizer_radius_grid',diagnostics[:,0],cfg['radii'],0.);assert np.all((diagnostics[:,2]>=0)&(diagnostics[:,2]<=80))
                    for dr in diagnostics:
                        solverrows.append({'seed':seed,'method':method,'step':step,'radius':float(dr[0]),'success':bool(dr[1]),'iterations':int(dr[2]),'normalized_objective':float(dr[3]),'accepted_candidate':bool(dr[4])})
                        optimizer_ids=np.flatnonzero((labels[:,0]=='past_pd_optimizer')&(labels[:,1].astype(float)==dr[0]));assert len(optimizer_ids)==int(dr[4])
                        if len(optimizer_ids):
                            oi=int(optimizer_ids[0]);assert bool(dr[1])==bool(bank('solver_success')[oi]) and int(dr[2])==int(bank('solver_iterations')[oi])
                    rng=np.random.default_rng(np.random.SeedSequence([m['cube']['seed'],seed,step]));random=[rng.normal(size=q.shape[1]) for _ in range(2)];random=[v/np.linalg.norm(v) for v in random]
                    for i,(kind,radius,sign) in enumerate(labels):
                        radius=float(radius);assert norms[i]<=radius+1e-5 if i else norms[i]==0
                        if kind.startswith('random'):check('random_direction',bank('coefficients')[i],int(sign)*radius*random[int(kind[-1])],1e-12)
                    gains2,res=fit_response(get('past_points'),get('past_targets'),.002);check('bank_gains',bank('gains'),gains2,1e-8);check('bank_residual',bank('residual'),res,1e-8);past=get('past_points');v=(past[-1]-past[-2])/.002;pred=np.array([response(np.r_[x,get('suffix_actions')],past[-1],v,gains2,a['bounds']) for x in actions]);check('independent_response',pred,bank('predicted_robot'),1e-10);cost=np.array([energy(np.r_[past[-75:,:3],v[:,:3]]) for v in pred]);check('expected_translation_jerk',cost,bank('costs'),1e-2,1e-8);d=pred-pred[:1];position=np.linalg.norm(d[:,:,:3],axis=-1);angle=abs(wrap(d[:,:,3]));contact=bool(float(get('root_gripper_contact'))>.1 or np.any(np.r_[compiled[:5,4],get('suffix_actions')[:,4]]>.1))
                    if method=='controller_displacement':score=np.maximum(position.max(1)/.002,angle.max(1)/np.deg2rad(2))
                    elif method=='proprio_contact':score=np.maximum((position+np.sqrt(3)*.02*angle).max(1)/.002,angle.max(1)/np.deg2rad(2)) if contact else np.zeros(len(actions))
                    else:
                        embedding=model.encode(get('history_images'));pastact=torch.tensor(((get('past_actions')-mean)/scale).astype(np.float32).reshape(1,2,25));latents=[]
                        with torch.no_grad():
                            for x in actions:
                                normact=torch.tensor(((x-mean)/scale).astype(np.float32).reshape(1,1,25));ac=torch.cat([pastact,normact],1);latents.append(model.model.predict(embedding,model.model.action_encoder(ac))[:,-1].reshape(-1).numpy());forwards+=1
                        latent=np.array(latents);score=np.linalg.norm(latent-latent[:1],axis=1)
                    check('independent_verifier',score,bank('scores'),2e-5);rule=rules[method];allow=(norms>1e-6)&(norms<=rule.get('radius',.4)+1e-6)&(bank('scores')<=rule.get('tau',-np.inf))&(bank('costs')<bank('costs')[0]-1e-10)
                    if rule.get('disabled',False):allow[:]=False
                    ids=np.flatnonzero(allow);choice=int(ids[np.lexsort((ids,bank('costs')[ids]))[0]]) if len(ids) else 0;assert selected==choice;check('selected_applied_actions',actions[choice],get('actions'),0.);assert int(get('encoder_calls'))==int(method=='lewm_global') and int(get('jacobian_calls'))==0 and int(get('finite_prediction_calls'))==(len(actions) if method=='lewm_global' else 0)
                else:assert selected==0;check('nominal_delivery',get('actions'),compiled[:5],0.)
                local=lambda tag,k:a[f'local_{bi:03d}_{tag}_{k}'];check('local_future_clock',local('nominal','time'),float(get('root_time'))+.002*np.arange(1,201),1e-9);check('local_same_clock',local('nominal','time'),local('modified','time'),0.);row=next(r for r in meta['local_rows'] if r['block']==bi)
                for key in ['qpos','qvel','object_pose','object_velocity','eef_position','eef_quaternion','robot_object_contacts','time']:
                    check('local_prefix',local('prefix',key),a['physics_'+key][step*25-75:step*25],0.);check('branch_equals_execution',local('modified',key)[:125],a['physics_'+key][step*25:(step+5)*25],0.)
                    if not selected:check('zero_branch_exact',local('modified',key),local('nominal',key),0.)
                e,pp,rr=physical_error(local('modified','object_pose'),local('nominal','object_pose'));j=energy(np.r_[local('prefix','eef_position'),local('modified','eef_position')]);j0=energy(np.r_[local('prefix','eef_position'),local('nominal','eef_position')]);gain=(j0-j)/j0 if j0>1e-12 else 0.;check('local_E',e,row['object_error'],2e-5);check('local_gain',gain,row['jerk_gain'],1e-8);activeblocks+=bool(selected);localrows.append({'seed':seed,'initial_pose_family':seed,'method':method,'step':step,'active':bool(selected),'E':e,'gain':gain,'joint':bool(selected and e<=1 and gain>=.1),'violation':bool(selected and e>1)})
            step+=h;blocks+=1
        assert step==n and meta['local_blocks']==57;E,pos,rot=physical_error(a['physics_object_pose'],reference['physics_object_pose']);j=energy(a['physics_eef_position']);j0=energy(reference['physics_eef_position']);gain=(j0-j)/j0;distance=np.linalg.norm(a['control_object_pose'][:,:3]-a['goal_position'],axis=1);success=distance<=.04;check('task_success',success,a['success'],0.);assert bool(success[-1])==meta['success_final'];rows.append({'seed':seed,'initial_pose_family':seed,'method':method,'active':activeblocks>0,'active_blocks':activeblocks,'global_E':E,'position_max_m':pos,'rotation_max_rad':rot,'global_violation':bool(activeblocks and E>1),'global_preserved':E<=1,'jerk_gain':gain,'joint':bool(activeblocks and E<=1 and gain>=.1),'task_success_any':bool(success.any()),'task_success_final':bool(success[-1]),'released_success_final':bool(success[-1] and a['control_robot_object_contacts'][-1]==0),'final_goal_distance_m':float(distance[-1]),'mean_generation_seconds':float(np.mean([r['generation_seconds'] for r in meta['local_rows']]))});print('AUDITED',seed,method,'activeblocks',activeblocks,flush=True)
    env.close();f=pd.DataFrame(rows);local=pd.DataFrame(localrows);assert len(f)==done['episodes'];f.to_csv(out/'evaluation.csv',index=False);local.to_csv(out/'local_evaluation.csv',index=False);solvers=pd.DataFrame(solverrows);solvers.to_csv(out/'solver_diagnostics.csv',index=False);assert len(solvers)==len(seeds)*3*57*5;summary=[]
    for method,g in f.groupby('method'):
        l=local[local.method==method];summary.append({'method':method,'episodes':len(g),**{key:mean_ci(g,key) for key in ['active','active_blocks','global_preserved','jerk_gain','joint','task_success_any','task_success_final','released_success_final','position_max_m','rotation_max_rad','final_goal_distance_m','mean_generation_seconds']},'global_selected_risk':mean_ci(g[g.active],'global_violation'),'local_joint':mean_ci(l,'joint'),'local_selected_risk':mean_ci(l[l.active],'violation')})
    contrasts=[]
    for method in ['proprio_contact','lewm_global']:
        v=f[f.method==method].merge(f[f.method=='controller_displacement'],on=['seed','initial_pose_family'],suffixes=('','_control'))
        for metric in ['joint','jerk_gain','global_preserved','task_success_final']:
            v['difference']=v[metric].astype(float)-v[metric+'_control'].astype(float);contrasts.append({'method':method,'reference':'controller_displacement','metric':metric,**mean_ci(v,'difference')})
    write_json(out/'analysis.json',{'smoke':args.smoke,'summary':summary,'paired_contrasts':contrasts,'scope':cfg['scope']});write_json(out/'raw_audit.json',{'smoke':args.smoke,'episodes':len(f),'families':len(seeds),'blocks':blocks,'local_blocks':len(local),'physics_samples':len(f)*7500,'exact_nominal_episodes':exact,'optimizer_attempts':len(solvers),'optimizer_failures':int((~solvers.success).sum()),'unconverged_candidates_retained':int((~solvers.success & solvers.accepted_candidate).sum()),'fresh_LeWM_candidate_predictions':forwards,'max_errors':errors,'source_sha256':sha256(Path(__file__)),'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'scope':'Causalroots/history/knownplan andindependentcompiler; float32gripperfixedcandidatebasis/radius/seed/response/translationcost; everyWMcandidatefreshforward; selectedindex beforephysics; localbranch equalsactualexecution; ownnominalwholetrajectory/jerk/goal andfamily summaries. No latency inserted.'});print('COMPLETE AUDIT',len(f),len(local),flush=True)
if __name__=='__main__':main()
