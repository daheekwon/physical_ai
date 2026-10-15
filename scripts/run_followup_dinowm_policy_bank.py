"""New-policy-family common polynomial bank; scores precede all counterfactual physics."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.policy_refinement import make_scene,full_state,coverage,context_at
from zj_wam.repeated_refinement import visual_history
from zj_wam.response_models import score_models
from zj_wam.model_audit import predict_candidates
from scripts.model_runtime_imports import load_dino
from scripts.dinowm_runtime_core import generate
from scripts.output_replacement_core import physical
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_dinowm_policy_bank.json','configs/followup_dinowm_policy_nominal.json','configs/followup_v1.json','configs/followup_response_models.json','scripts/run_followup_dinowm_policy_bank.py','scripts/dinowm_runtime_core.py','scripts/model_runtime_imports.py','scripts/retrained_dinowm_core.py','scripts/output_replacement_core.py','zj_wam/adapters.py','zj_wam/policy_refinement.py','zj_wam/followup.py','zj_wam/repeated_refinement.py','zj_wam/response_models.py','zj_wam/diagnostics.py','zj_wam/model_audit.py']

def context(a,seed,t,scale,mean):
    images,past,proprio,prefix,ids=context_at(t,a['images'][:t+1],a['states'][:t+1],a['actions'][:t],a['physics'][:10*t]);root=proprio[-1].copy()
    # Controller input intentionally has no GT object pose/velocity.
    assert np.all(root[[2,3,4,7,8,9]]==0)
    return {'episode':seed,'timestep':t,'images':images,'past':past,'proprio':proprio,'prefix':prefix,'root':root,'nominal':a['actions'][t:t+5],'suffix':a['actions'][t+5:t+8],'scale':scale,'mean':mean}

def branch(seed,t,original,modified):
    env=make_scene(seed)
    for action in original[:t]:env.step(action)
    root=full_state(env);history_image=env.render().copy();trace=[];contact=[];cv=[];endpoint=None;counter=[0];old=env.space.step
    def collide(arbiter,space,data):
        env._handle_collision(arbiter,space,data);bodies={s.body for s in arbiter.shapes}
        if env.agent in bodies and env.block in bodies:counter[0]+=len(arbiter.contact_point_set.points)
    env.space.on_collision(0,0,post_solve=collide)
    def step(dt):
        before=counter[0];old(dt);trace.append(full_state(env));contact.append(counter[0]-before)
    env.space.step=step
    for k,action in enumerate(np.r_[modified,original[t+5:t+8]]):
        env.step(action);cv.append(coverage(env))
        if k==4:endpoint=env.render().copy()
    env.space.step=old;env.close()
    return {'root':root,'image':history_image,'physics':np.array(trace),'contacts':np.array(contact),'coverage':np.array(cv),'endpoint':endpoint}

def main(smoke):
    cfg=read(resolve('configs/followup_dinowm_policy_bank.json'));base=read(resolve('configs/followup_v1.json'));rcfg=read(resolve('configs/followup_response_models.json'));torch.set_num_threads(cfg['torch_threads']);src=BASE/('dinowm_policy_nominal_smoke' if smoke else 'dinowm_policy_nominal');parent=read(src/'manifest.json');audit=read(src/'raw_audit.json');assert audit['episodes']==(1 if smoke else 36) and audit['all_physics_and_RGB_replayed_exact']
    if not smoke:assert read(src/'pipeline_complete.json')['raw_audit_sha256']==sha256(src/'raw_audit.json')
    out=BASE/('dinowm_policy_bank_smoke' if smoke else 'dinowm_policy_bank');out.mkdir(exist_ok=True);paths=sorted(src.glob('*/*_nominal.json'));assert len(paths)==(1 if smoke else 36);states=[{'seed':read(p)['seed'],'split':p.parent.name,'step':t,'source':str(p.relative_to(resolve('.')))} for p in paths for t in parent['config']['matched_chunk_control_steps']];files=[resolve(p) for p in SOURCES]+[src/'manifest.json',src/'raw_audit.json',resolve(base['checkpoint'])/'weights.pt'];files+=sorted((BASE/'protocol/dinowm_retrained_inventory').glob('*.json'))
    for p in paths:files.extend([p,p.with_suffix('.npz'),p.with_suffix('.csv')])
    manifest={'smoke':smoke,'config':cfg,'states':states,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    data=ExpertDataset(base['dataset']);lewm=FrozenLeWM(base['checkpoint'],data.scaler,'cpu');dino=load_dino('pusht');count=0;total=0
    for p in paths:
        seed=read(p)['seed'];split=p.parent.name;a=dict(np.load(p.with_suffix('.npz')));folder=out/split;folder.mkdir(exist_ok=True)
        for t in parent['config']['matched_chunk_control_steps']:
            dest=folder/f'{seed}_t{t:03d}.json'
            if dest.exists():m=read(dest);assert m['manifest_sha256']==sha256(out/'manifest.json') and m['data_sha256']==sha256(dest.with_suffix('.npz'));count+=1;total+=m['candidates'];continue
            start=time.perf_counter();c=context(a,seed,t,data.scaler.scale_,data.scaler.mean_);actions,labels,robot,costs,norms,solver=generate('pusht',c,base);scores={'controller':np.linalg.norm(robot-robot[:1],axis=2).max(1)};observed,valid=visual_history(c['images'],c['proprio'],rcfg);sc,_=score_models(actions,c['suffix'],observed[-1],observed,rcfg);scores['contact']=sc['contact'] if valid else np.full(len(actions),np.inf)
            _,lc=lewm.predictor(c['images'],c['past']);encoded=dino.encode(c['images'])
            with torch.no_grad():
                zl=predict_candidates(lewm,lc['embedding'],lc['past'],actions,batch_size=64)[:,0];zd=np.array([dino.predict(encoded,c['past'],x)[0].numpy() for x in actions])
            scores['lewm_global']=np.linalg.norm(zl-zl[:1],axis=1);delta=zd.astype(float)-zd[:1].astype(float);scores['dino_patch_rms']=np.sqrt(np.mean(np.sum(delta**2,axis=2),axis=1));scores['dino_patch_mean']=np.linalg.norm(delta.mean(1),axis=1);raw={'actions':actions,'labels':np.array(labels,dtype=str),'norms':norms,'predicted_robot':robot,'predicted_costs':costs,'lewm_predicted':zl,'dino_predicted':zd,**{'score_'+k:v for k,v in scores.items()},**{'input_'+k:np.asarray(v) for k,v in c.items()}};physics=[];contacts=[];covers=[];endpoints=[]
            for i,x in enumerate(actions):
                b=branch(seed,t,a['actions'],x);np.testing.assert_array_equal(b['root'],a['states'][t]);np.testing.assert_array_equal(b['image'],a['images'][t]);physics.append(np.r_[c['prefix'],b['physics']]);contacts.append(b['contacts']);covers.append(b['coverage'][[4,7]]);endpoints.append(b['endpoint'])
                if i==0:np.testing.assert_array_equal(b['physics'],a['physics'][t*10:(t+8)*10]);np.testing.assert_allclose(b['coverage'],a['coverage'][t:t+8],atol=0,rtol=0)
            raw.update(physics=np.array(physics),contacts=np.array(contacts),coverage=np.array(covers),endpoint_images=np.array(endpoints));truth=physical('pusht',raw);phase='contact' if np.any(raw['contacts'][0]) else 'free';gain=(costs[0]-costs)/max(costs[0],1e-12);sid=f'{seed}_t{t:03d}';rows=[]
            for i,label in enumerate(labels):rows.append({'environment':'pusht','split':split,'state_id':sid,'initial_pose_family':seed,'phase':phase,'candidate':i,'generator':label[0],'radius':label[1],'sign':label[2],'norm':norms[i],'predicted_cost':costs[i],'predicted_gain':gain[i],**{k:truth[k][i] for k in truth},**{k:scores[k][i] for k in scores}})
            assert all(np.isfinite(raw[k]).all() for k in ['actions','physics','coverage','lewm_predicted','dino_predicted','predicted_costs']);np.savez_compressed(dest.with_suffix('.npz'),**raw);pd.DataFrame(rows).to_csv(dest.with_suffix('.csv'),index=False);write_json(dest,{'seed':seed,'split':split,'step':t,'state_id':sid,'candidates':len(actions),'phase':phase,'visual_contact_valid':bool(valid),'solver':solver,'source':str(p.relative_to(resolve('.'))),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'seconds':time.perf_counter()-start});count+=1;total+=len(actions);print('FRESH POLICY BANK',split,sid,len(actions),round(time.perf_counter()-start,2),flush=True)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    assert count==len(states);write_json(out/'execution_complete.json',{'smoke':smoke,'states':count,'candidates':total,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Independent physics/model/action audit required before calibration; no heldout test.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_dinowm_policy_bank.json'))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(args.smoke)
