"""Fresh shifted-history replay and causal before/after response-fit diagnosis."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube,snapshot,restore
from zj_wam.cube_experiment import fit_response,predict_response
from scripts.run_followup_cube_native_policy import scene
from scripts.cube_shift_preflight_core import baseline
from scripts.cube_shift_history_core import control_rollout,points,warm_history,prepare_history
from scripts.audit_followup_dinowm_policy_final import interval
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');PARENT=BASE/'cube_policy_refinement';read=lambda p:json.loads(p.read_text())

def independent_response(actions,p,v,g,bounds):
    p=p.copy();d=.002*v;out=[]
    for action in actions:
        target=p+action[:4]*np.array([.05,.05,.05,.3]);target[:3]=np.clip(target[:3],bounds[0],bounds[1])
        for _ in range(25):
            e=target-p;e[3]=np.angle(np.exp(1j*e[3]));d=(1-.002*g[:,1])*d+.002**2*g[:,0]*e;p=p+d;out.append(p.copy())
    return np.array(out)
def energy(q):
    d=q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3];return float((d*d).sum()/.002**5)
def main(smoke):
    out=BASE/('cube_shift_history_smoke' if smoke else 'cube_shift_history');manifest=read(out/'manifest.json');cfg=manifest['config'];done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['manifest_sha256']==sha256(out/'manifest.json');paths=[resolve(p) for p in manifest['sources']];native=read(PARENT/'manifest.json')['native'];seeds=sorted({read(p)['seed'] for p in paths});inputs={};errors={};rows=[];count=0;fresh_samples=0;fresh_rgb=0
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    def check(name,a,b,tol=0,rtol=0):
        a=np.asarray(a);b=np.asarray(b);np.testing.assert_allclose(a,b,atol=tol,rtol=rtol,err_msg=name);errors[name]=max(errors.get(name,0.),float(np.max(abs(a.astype(float)-b.astype(float)))) if a.size else 0.)
    for seed in seeds:
        source=dict(np.load(PARENT/f'{seed}_nominal.npz'));env,obs,info,_,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);initial=meter.measure();original=baseline(env);original['visual_statistics']={k:np.array(getattr(env._model.stat,k),copy=True) for k in ['center','extent','meansize']};current=0;chunks=[];commands=[]
        for p in sorted([p for p in paths if read(p)['seed']==seed],key=lambda p:read(p)['step']):
            meta=read(p);t=meta['step'];change=t-10;assert change>=1
            if change>current:
                trace,targets=control_rollout(env,meter,source['actions'][current:change]);chunks.append(trace);commands.extend(targets)
                for key,value in trace.items():check('prechange_parent_'+key,value,source['physics_'+key][current*25:change*25])
                current=change
            full={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]};pre=points(initial,full);root=snapshot(env)
            for condition in manifest['conditions']:
                dest=out/condition['name']/p.name;m=read(dest);raw=dict(np.load(dest.with_suffix('.npz')));assert m['data_sha256']==sha256(dest.with_suffix('.npz')) and m['manifest_sha256']==sha256(out/'manifest.json') and m['step']==t and m['change_step']==change
                fresh=warm_history(env,meter,root,original,condition,source['actions'][change:t],pre,np.array(commands),cfg)
                for key,value in fresh.items():check('fresh_history_'+key,value,raw[key])
                fresh_samples+=250;fresh_rgb+=3;check('actions',raw['past_actions'],source['actions'][change:t]);check('nominal_actions',raw['nominal_actions'],source['actions'][t:t+5]);check('suffix_actions',raw['suffix_actions'],source['actions'][t+5:t+8]);check('history_times',raw['history_control_indices'],[t-10,t-5,t]);length=min(250,change*25);assert int(raw['fit_physics_steps'])==length and raw['prechange_points'].shape==raw['postchange_fit_points'].shape==(length+1,4) and raw['prechange_targets'].shape==raw['postchange_fit_targets'].shape==(length,4);check('equal_budget_newpoints',raw['postchange_fit_points'],raw['past_points'][-length-1:]);check('equal_budget_newtargets',raw['postchange_fit_targets'],raw['past_targets'][-length:]);check('past_physical_root',raw['past_points'][-1,:3],raw['root_eef_position']);check('history_clock',raw['warmup_time'],raw['change_root_time']+.002*np.arange(1,251),1e-9)
                if condition['name']=='nominal':
                    parentraw=dict(np.load(p.with_suffix('.npz')));check('original_RGB_fixture',raw['history_images'],parentraw['input_history_images'])
                    for key in ['qpos','qvel','object_pose','object_velocity','eef_position','eef_quaternion','robot_object_contacts','time']:check('original_warmup_fixture_'+key,raw['warmup_'+key],source['physics_'+key][change*25:t*25])
                for key in ['center','extent','meansize']:check('preserved_visual_'+key,raw['parameter_visual_'+key],original['visual_statistics'][key])
                assert float(raw['root_time'])==float(raw['warmup_time'][-1]);actions=np.r_[raw['nominal_actions'],raw['suffix_actions']];future=meter.rollout(actions)['physics'];fresh_samples+=200
                for key,value in future.items():check('fresh_future_'+key,value,raw['nominal_future_'+key])
                fit_output={};prefix=raw['warmup_eef_position'][-75:];actualcost=energy(np.r_[prefix,raw['nominal_future_eef_position']]);rootp=raw['past_points'][-1];velocity=(raw['past_points'][-1]-raw['past_points'][-2])/.002
                for variant,pk,tk in [('frozen_prechange','prechange_points','prechange_targets'),('updated_postchange','postchange_fit_points','postchange_fit_targets')]:
                    g,res=fit_response(raw[pk],raw[tk],.002);pred=predict_response(actions,rootp,velocity,g,raw['workspace_bounds'],.002,25);ind=independent_response(actions,rootp,velocity,g,raw['workspace_bounds']);check('independent_response',pred,ind,1e-10);J=energy(np.r_[prefix,pred[:,:3]]);pos=np.sqrt(np.mean(np.sum((pred[:,:3]-raw['nominal_future_eef_position'])**2,axis=1)));rows.append({'state_id':p.stem,'initial_pose_family':seed,'phase':meta['phase'],'condition':condition['name'],'model':variant,'fit_steps':length,'eef_position_RMSE_m':float(pos),'eef_jerk_relative_error':float(abs(J-actualcost)/max(actualcost,1e-12)),'actual_jerk':actualcost,'predicted_jerk':J});fit_output.update({variant+'_gains':g,variant+'_residual':res,variant+'_predicted':pred})
                derived=dest.with_name(dest.stem+'.response.npz');np.savez_compressed(derived,**fit_output)
                for q in [dest,dest.with_suffix('.npz'),derived]:inputs[str(q.relative_to(resolve('.')))]=sha256(q)
                count+=1;print('AUDIT SHIFT HISTORY',seed,meta['phase'],condition['name'],count,flush=True)
            prepare_history(env,root,original,manifest['conditions'][0]);restore(env,root)
        env.close()
    f=pd.DataFrame(rows);summary=[];paired=[];interaction=[]
    for (condition,model),x in f.groupby(['condition','model']):summary.append({'condition':condition,'model':model,'states':len(x),**{key:interval(x,key) for key in ['eef_position_RMSE_m','eef_jerk_relative_error']}})
    for condition,x in f.groupby('condition'):
        old=x[x.model=='frozen_prechange'].set_index('state_id');new=x[x.model=='updated_postchange'].set_index('state_id').loc[old.index];v=old.reset_index()
        for key in ['eef_position_RMSE_m','eef_jerk_relative_error']:v[key+'_improvement']=old[key].to_numpy()-new[key].to_numpy();paired.append({'condition':condition,'metric':key,'direction':'positive means update reduces error',**interval(v,key+'_improvement')})
        if condition!='sham_forward':
            base=f[f.condition=='sham_forward'];bo=base[base.model=='frozen_prechange'].set_index('state_id').loc[old.index];bn=base[base.model=='updated_postchange'].set_index('state_id').loc[old.index]
            for key in ['eef_position_RMSE_m','eef_jerk_relative_error']:v['interaction']=old[key].to_numpy()-new[key].to_numpy()-(bo[key].to_numpy()-bn[key].to_numpy());interaction.append({'condition':condition,'metric':key,'direction':'update benefit relative to same-forward sham update benefit',**interval(v,'interaction')})
    assert count==done['condition_contexts'] and len(f)==2*count and fresh_samples==450*count and fresh_rgb==3*count;f.to_csv(out/'response_fidelity.csv',index=False);write_json(out/'analysis.json',{'smoke':smoke,'states':done['states'],'families':len(seeds),'condition_contexts':count,'forecast_rows':len(f),'summary':summary,'paired':paired,'condition_interaction':interaction,'scope':'Past-only matched-sample PD coefficient refresh, nominal future8 prediction fidelity. Fixedprechange versus updatedpostchange fitting; ordinary phase/time update benefit controlled with same-forward sham. Not correction selection, risk calibration or adaptive smoothing success.'});write_json(out/'raw_audit.json',{'smoke':smoke,'states':done['states'],'families':len(seeds),'condition_contexts':count,'fresh_history_RGB':fresh_rgb,'fresh_physics_samples':fresh_samples,'forecast_rows':len(f),'max_errors':errors,'inputs':inputs,'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'source_sha256':sha256(Path(__file__)),'audit_helpers':{p:sha256(resolve(p)) for p in ['scripts/audit_followup_dinowm_policy_final.py']},'scope':'All original nominal RGB/physics fixtures and causal shifted histories, visualstats, equalfitbudget and futurephysical branches freshly replayed. Existing audited model-preparation/history helpers reused; independently checked indexing, known past/future separation and model recurrence. No future robot/object fit.'});print('AUDITED SHIFT HISTORY',count,len(f),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
