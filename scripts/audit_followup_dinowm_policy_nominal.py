"""Independent nominal policy replay, action compilation and sparse policy forwards."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dill,hydra,numpy as np,pandas as pd,torch
from zj_wam.io import resolve,sha256,write_json
from zj_wam.policy_refinement import make_scene,full_state
from scripts.analyze_followup_policy_final import Coverage,energy
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def main(smoke):
    out=BASE/('dinowm_policy_nominal_smoke' if smoke else 'dinowm_policy_nominal');manifest=read(out/'manifest.json');cfg=manifest['config'];done=read(out/'execution_complete.json');assert done['manifest_sha256']==sha256(out/'manifest.json') and done['smoke']==smoke and not done['reserved_test_outcomes_evaluated'];basecfg=read(resolve('configs/followup_v1.json'));torch.set_num_threads(basecfg['threads']);ckpt=resolve(cfg['policy_checkpoint']);assert sha256(ckpt)==manifest['checkpoint_sha256'];payload=torch.load(ckpt,map_location='cpu',pickle_module=dill,weights_only=False);policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload;policy.to(cfg['policy_device']).eval().requires_grad_(False);errors={};rows=[];inputs={};forwards=0;blocks=0
    for p,h in manifest['input_hashes'].items():assert sha256(resolve(p))==h,p
    for p,h in manifest['prior_initial_inputs'].items():assert sha256(resolve(p))==h,p
    def check(name,x,y,tol=1e-8):
        x=np.asarray(x);y=np.asarray(y);np.testing.assert_allclose(x,y,atol=tol,rtol=0,err_msg=name);e=float(np.nanmax(abs(x.astype(float)-y.astype(float)))) if x.size else 0.;errors[name]=max(errors.get(name,0.),e)
    for split in ['development','calibration']:
        paths=sorted((out/split).glob('*_nominal.json'));assert len(paths)==(1 if split=='development' else 0) if smoke else len(paths)==len(cfg[split+'_seeds'])
        for path in paths:
            meta=read(path);seed=meta['seed'];a=dict(np.load(path.with_suffix('.npz')));assert manifest['initials'][str(seed)]['split']==split;check('initial_manifest',a['initial_state'],manifest['initials'][str(seed)]['initial_state'],0);check('control_dense',a['states'][1:],a['physics'][9::10],0);assert len(a['actions'])==300 and abs(a['actions']).max()<=1 and meta['active_chunks']==0;covfun=Coverage(seed);cov=np.array([covfun(x) for x in a['states'][1:]]);check('independent_polygon_coverage',cov,a['coverage'],1e-10);check('reported_jerk',energy(np.r_[a['initial_state'][None,:2],a['physics'][:,:2]]),meta['physics_jerk'],1e-3);check('reported_final_coverage',cov[-1],meta['final_coverage'],1e-10);hits=np.flatnonzero(cov>.95);assert meta['first_success_action']==(int(hits[0])+1 if len(hits) else None)
            for k,d in enumerate(meta['decisions']):
                t=d['control_step'];n=min(8,300-t);targets=a['policy_action_predictions'][k,1:1+n];q=a['states'][t,:2].copy();v=a['states'][t,5:7].copy();compiled=[]
                for target in targets:
                    act=np.clip((target-q)/100.,-1,1);compiled.append(act);position_target=q+100*act
                    for _ in range(10):v=v+.01*(100*(position_target-q)-20*v);q=q+.01*v
                check('independent_target_compilation',compiled,a['actions'][t:t+n],1e-9);check('nominal_chunk',compiled,a['nominal_relative_chunks'][k,:n],1e-9);assert d['policy_observation_times']==[max(0,t-1),t] and d['policy_noise_seed']==(basecfg['seed']+1009*seed+t)%(2**31-1);assert not d['active']
                if d['eligible']:assert d['wm_observation_times']==[t-10,t-5,t]
                blocks+=1
            env=make_scene(seed);trace=[];original=env.space.step
            def step(dt):original(dt);trace.append(full_state(env))
            env.space.step=step;obs=[];policy.reset()
            for t in range(301):
                check('fresh_control',full_state(env),a['states'][t],0);check('fresh_RGB',env.render(),a['images'][t],0);old=env.render_size;env.render_size=96;rgb=np.moveaxis(env.render().astype(np.float32)/255.,-1,0);env.render_size=old;obs.append({'image':rgb,'agent_pos':np.asarray(env.agent.position,dtype=np.float32)})
                if t in cfg['matched_chunk_control_steps']:
                    ids=[t-1,t];batch={key:torch.from_numpy(np.stack([obs[i][key] for i in ids])[None]).to(cfg['policy_device'],dtype=torch.float32) for key in obs[-1]};torch.manual_seed((basecfg['seed']+1009*seed+t)%(2**31-1))
                    with torch.no_grad():prediction=policy.predict_action(batch)
                    check('fresh_native_policy_prediction',prediction['action_pred'][0].cpu().numpy(),a['policy_action_predictions'][t//8],1e-5);forwards+=1
                if t<300:env.step(a['actions'][t])
            env.space.step=original;env.close();check('fresh_dense_replay',trace,a['physics'],0);local=pd.read_csv(path.with_suffix('.csv'));assert len(local)==35 and not local.active.any();check('nominal_local_E',local.object_error,np.zeros(len(local)),1e-8);check('nominal_local_gain',local.jerk_gain,np.zeros(len(local)),1e-8)
            for ext in ['.json','.npz','.csv']:inputs[str(path.with_suffix(ext).relative_to(resolve('.')))]=sha256(path.with_suffix(ext))
            rows.append({'split':split,'seed':seed,'success_any':bool(len(hits)),'final_coverage':float(cov[-1]),'max_coverage':float(cov.max()),'first_success_action':int(hits[0])+1 if len(hits) else None,'policy_calls':meta['policy_calls'],'jerk':meta['physics_jerk']});print('AUDIT FRESH NOMINAL',split,seed,flush=True)
    assert len(rows)==done['episodes'];pd.DataFrame(rows).to_csv(out/'evaluation.csv',index=False);write_json(out/'raw_audit.json',{'smoke':smoke,'episodes':len(rows),'physics_samples':len(rows)*3000,'compiled_policy_blocks':blocks,'fresh_native_policy_predictions':forwards,'all_physics_and_RGB_replayed_exact':True,'reserved_test_outcomes_evaluated':False,'max_errors':errors,'inputs':inputs,'manifest_sha256':sha256(out/'manifest.json'),'source_sha256':sha256(Path(__file__)),'scope':'All nominal300control/3000physics and301RGB per episode reproduced; independent target compilation and polygon coverage;3native policy calls at fixed windows rerun; original checkpointhash and family separation manifest checked. No refinement efficacy claim.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
