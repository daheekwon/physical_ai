"""Independent candidate geometry, scores, metrics and fresh shifted-root audit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import restore
from zj_wam.cube_experiment import fit_response
from scripts.run_followup_cube import Model
from scripts.cube_shift_candidate_bank_core import replay_root,selector_context
from scripts.audit_followup_cube_shift_history import independent_response,energy
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main(smoke):
    out=BASE/('cube_shift_candidate_bank_smoke' if smoke else 'cube_shift_candidate_bank');manifest=read(out/'manifest.json');cfg=manifest['config'];done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['manifest_sha256']==sha256(out/'manifest.json')
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    hm=read(BASE/'cube_shift_history/manifest.json');conditions={x['name']:x for x in hm['conditions']};parent=read(BASE/'cube_policy_refinement/manifest.json');native=parent['native'];ccfg={**parent['cube'],'device':'cpu','yaw_length_scale_m':0.};norm=read(BASE/'cube/manifest.json')['normalization'];mean=np.array(norm['mean']);scale=np.array(norm['scale']);torch.set_num_threads(cfg['torch_threads']);model=Model(ccfg,mean,scale);errors={};inputs={};rows=[];count=0;nc=0;nf=0;npred=0
    def check(name,x,y,tol=0,rtol=0):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(name,x.shape,y.shape);np.testing.assert_allclose(x,y,atol=tol,rtol=rtol,err_msg=name);errors[name]=max(errors.get(name,0.),float(abs(x.astype(float)-y.astype(float)).max()) if x.size else 0.)
    for source in manifest['sources']:
        cp=resolve(source);cm=read(cp);path=out/cm['condition']/cp.name;meta=read(path);a=dict(np.load(path.with_suffix('.npz')));assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(path.with_suffix('.npz'));bp=path.with_name(path.stem+'.before_physics.npz');assert sha256(bp)==meta['before_physics_sha256']
        with np.load(bp) as pre:
            assert not any(k.startswith('candidate_') or k.startswith('prefix_') for k in pre.files)
            for k in pre.files:
                if pre[k].dtype.kind in 'US':np.testing.assert_array_equal(pre[k],a[k])
                else:check('prephysics_'+k,pre[k],a[k])
        hp=resolve(cm['history_source']);hm0=read(hp);history=dict(np.load(hp.with_suffix('.npz')));compiled=dict(np.load(cp.with_suffix('.npz')));context=selector_context(history,compiled);frame=pd.read_csv(path.with_suffix('.csv'));actions=a['actions'];n=len(actions);assert n==meta['candidates']==len(frame) and actions.dtype==np.float32 and np.max(abs(actions))<=1
        for k,v in context.items():check('causal_input_'+k,a['input_'+k],np.asarray(v))
        check('nominal_actions',actions[0],context['nominal_actions']);check('fixed_gripper',actions[:,:,4],np.broadcast_to(actions[0,:,4],actions.shape[:2]));q=a['basis'];check('orthonormal_basis',q.T@q,np.eye(q.shape[1]),1e-12);diff=((actions-actions[:1])/scale).reshape(n,-1);norms=np.linalg.norm(diff,axis=1);check('norm',norms,a['norm'],1e-8);check('polynomial',diff,diff@q@q.T,1e-5);check('quantized_coefficients',actions,(actions[0]+(a['coefficients']@q.T).reshape(-1,5,5)*scale).astype(np.float32));assert np.all(norms<=a['labels'][:,1].astype(float)+1e-5)
        points=context['past_points'];gains,res=fit_response(points,context['past_targets'],.002);v=(points[-1]-points[-2])/.002;robot=np.array([independent_response(np.r_[act,context['suffix_actions']],points[-1],v,gains,context['workspace_bounds']) for act in actions]);check('independent_response',robot,a['predicted_robot'],1e-10);cost=np.array([energy(np.r_[points[-75:,:3],p[:,:3]]) for p in robot]);check('predicted_cost',cost,a['costs'],1e-2,1e-8);check('csv_predicted_cost',cost,frame.predicted_cost,1e-2,1e-8)
        delta=robot-robot[:1];dp=np.linalg.norm(delta[:,:,:3],axis=-1);dr=abs(np.angle(np.exp(1j*delta[:,:,3])));scores={'controller_displacement':np.maximum(dp.max(1)/.002,dr.max(1)/np.deg2rad(2))};contact=context['root_gripper_contact']>.1 or np.any(np.r_[actions[0,:,4],context['suffix_actions'][:,4]]>.1);scores['proprio_contact']=np.maximum((dp+np.sqrt(3)*.02*dr).max(1)/.002,dr.max(1)/np.deg2rad(2)) if contact else np.zeros(n)
        embedding=model.encode(context['history_images']);past=torch.tensor(((context['past_actions']-mean)/scale).astype(np.float32).reshape(1,2,25));x0=((actions[0]-mean)/scale).reshape(-1);latent=[]
        with torch.no_grad():
            for action in actions:
                x=torch.tensor(x0+((action-actions[0])/scale).reshape(-1),dtype=torch.float32);latent.append(model.model.predict(embedding,model.model.action_encoder(torch.cat([past,x.reshape(1,1,25)],1)))[:,-1].reshape(-1).numpy());npred+=1
        latent=np.array(latent);scores['lewm_global']=np.linalg.norm(latent-latent[:1],axis=1)
        for m in cfg['methods']:check('native_score_'+m,scores[m],a[m+'_scores'],2e-5 if m=='lewm_global' else 1e-8)
        pose=a['candidate_object_pose'];pos=np.linalg.norm(pose[:,:,:3]-pose[:1,:,:3],axis=2).max(1);r0=Rotation.from_quat(pose[0][:,[4,5,6,3]]);rot=np.array([(Rotation.from_quat(p[:,[4,5,6,3]])*r0.inv()).magnitude().max() for p in pose]);E=np.maximum(pos/.002,rot/np.deg2rad(2));jerk=np.array([energy(np.r_[a['prefix_eef_position'],p]) for p in a['candidate_eef_position']]);gain=(jerk[0]-jerk)/max(jerk[0],1e-12);check('object_error',E,frame.object_error,2e-5);check('actual_jerk',jerk,frame.eef_jerk,1e-2,1e-8);check('actual_gain',gain,frame.jerk_gain,1e-8);check('same_branch_clock',a['candidate_time'],np.broadcast_to(a['candidate_time'][0],a['candidate_time'].shape));check('physical_clock',a['candidate_time'][0],float(history['root_time'])+.002*np.arange(1,201),1e-9)
        for k,value in history.items():
            if k.startswith('warmup_'):check('prefix_'+k,a['prefix_'+k[7:]],value[-75:])
        env,meter,root=replay_root(hm0,history,conditions[cm['condition']],native,hm['config']);info=env.get_step_info();check('actual_root_proprio',np.r_[info['proprio/effector_pos'],info['proprio/effector_yaw'][0]],compiled['input_root_proprio'],1e-12);check('actual_root_gripper',info['proprio/gripper_opening'][0],compiled['input_gripper_history'][-1])
        for i in sorted({0,n-1}):
            restore(env,root);trace=meter.rollout(np.r_[actions[i],context['suffix_actions']])['physics']
            for k,value in trace.items():check('fresh_branch_'+k,value,a['candidate_'+k][i])
            nf+=1
        env.close();active=norms>1e-6;safe=E<=1;rows.append({'state_id':cp.stem,'initial_pose_family':cm['seed'],'phase':cm['phase'],'condition':cm['condition'],'candidates':n,'finite_oracle_useful':bool(np.any(active&safe&(gain>=.1))),'best_actual_safe_gain':float(gain[safe].max()),'best_predicted_gain':float(((cost[0]-cost)/max(cost[0],1e-12)).max())})
        for p in [path,path.with_suffix('.npz'),path.with_suffix('.csv'),bp]:inputs[str(p.relative_to(resolve('.')))]=sha256(p)
        count+=1;nc+=n;print('AUDIT SHIFT BANK',cm['seed'],cm['phase'],cm['condition'],count,flush=True)
    assert count==done['contexts'] and nc==done['candidates'] and npred==nc;pd.DataFrame(rows).to_csv(out/'candidate_potential.csv',index=False);write_json(out/'raw_audit.json',{'smoke':smoke,'contexts':count,'candidates':nc,'fresh_LeWM_predictions':npred,'fresh_physical_branches':nf,'fresh_history_RGB':3*count,'max_errors':errors,'inputs':inputs,'manifest_sha256':sha256(out/'manifest.json'),'source_sha256':sha256(Path(__file__)),'scope':'All candidate geometry, expected/actual XYZ jerk, dense fullSO3 preservation and native scores independently recomputed. Every shifted root/history freshly replayed with previously audited preparation helper; first/last candidate branches physically reproduced in every context. Scores/actions persisted before evaluation. No threshold fitting/selection efficacy or allowance/compute adaptation claim.'});print('AUDITED SHIFT BANK',count,nc,nf,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=read(resolve('configs/followup_cube_shift_candidate_bank.json'))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(a.smoke)
