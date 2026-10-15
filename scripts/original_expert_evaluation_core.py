"""Read official trajectories and replay fixed nominal branches with explicit command semantics."""
import numpy as np,zarr
from zj_wam.policy_refinement import full_state,coverage,context_at
from scripts.audit_followup_original_expert_adapter import scene
from scripts.dinowm_runtime_core import generate
from zj_wam.repeated_refinement import visual_history
from zj_wam.response_models import score_models
from zj_wam.model_audit import predict_candidates
import torch

def original(path,episode):
    store=zarr.ZipStore(str(path),mode='r');r=zarr.open_group(store,path='pusht/pusht_cchi_v7_replay.zarr',mode='r');ends=r['meta/episode_ends'][:];start=0 if episode==0 else int(ends[episode-1]);end=int(ends[episode]);states=r['data/state'][start:end].astype(float);targets=r['data/action'][start:end].astype(float);store.close();return states,targets

def compile_targets(targets,initial):
    # Known pusher dynamics only: no object state or simulator future input.
    q=np.asarray(initial[:2],float).copy();v=np.zeros(2);actions=[]
    for target in targets:
        actions.append((target-q)/100.)
        for _ in range(10):v+=.01*(100*(target-q)-20*v);q+=.01*v
    return np.array(actions)

def nominal(initial,targets):
    actions=compile_targets(targets,initial);env=scene(initial,True);physics=[];states=[full_state(env)];images=[env.render().copy()];cov=[];old=env.space.step
    def step(dt):old(dt);physics.append(full_state(env))
    env.space.step=step
    for target,action in zip(targets,actions):
        np.testing.assert_allclose(np.array(env.agent.position)+100*action,target,atol=1e-9,rtol=0);env.step(action);states.append(full_state(env));images.append(env.render().copy());cov.append(coverage(env))
    env.space.step=old;env.close();return {'actions':actions,'physics':np.array(physics),'states':np.array(states),'images':np.array(images),'coverage':np.array(cov),'initial_pose':np.asarray(initial)}

def window_steps(n,cfg):
    lo=cfg['history_actions'];hi=n-cfg['horizon']-cfg['suffix'];assert hi>lo;steps=[lo+int((hi-lo)*x) for x in cfg['window_fractions']];assert len(set(steps))==3 and min(steps)>=10 and max(steps)+8<=n;return steps

def context(a,family,t,mean,scale):
    images,past,proprio,prefix,_=context_at(t,a['images'][:t+1],a['states'][:t+1],a['actions'][:t],a['physics'][:t*10]);return {'episode':family,'timestep':t,'images':images,'past':past,'proprio':proprio,'root':proprio[-1],'prefix':prefix,'nominal':a['actions'][t:t+5],'suffix':a['actions'][t+5:t+8],'mean':mean,'scale':scale}

def branch(initial,a,t,modified):
    env=scene(initial,True)
    for action in a['actions'][:t]:env.step(action)
    np.testing.assert_array_equal(full_state(env),a['states'][t]);np.testing.assert_array_equal(env.render(),a['images'][t]);counter=[0];contacts=[];physics=[];cov=[];endpoint=None;old=env.space.step
    def collide(arbiter,space,data):
        env._handle_collision(arbiter,space,data);bodies={s.body for s in arbiter.shapes}
        if env.agent in bodies and env.block in bodies:counter[0]+=len(arbiter.contact_point_set.points)
    env.space.on_collision(0,0,post_solve=collide)
    def step(dt):before=counter[0];old(dt);physics.append(full_state(env));contacts.append(counter[0]-before)
    env.space.step=step
    for k,action in enumerate(np.r_[modified,a['actions'][t+5:t+8]]):
        env.step(action);cov.append(coverage(env))
        if k==4:endpoint=env.render().copy()
    env.space.step=old;env.close();return {'physics':np.array(physics),'contacts':np.array(contacts),'coverage':np.array(cov)[[4,7]],'endpoint_images':endpoint}

def scores(c,cfg,rcfg,lewm,dino):
    actions,labels,robot,costs,norms,solver=generate('pusht',c,cfg);score={'controller':np.linalg.norm(robot-robot[:1],axis=2).max(1)};observed,valid=visual_history(c['images'],c['proprio'],rcfg);v,_=score_models(actions,c['suffix'],observed[-1],observed,rcfg);score['contact']=v['contact'] if valid else np.full(len(actions),np.inf);_,lc=lewm.predictor(c['images'],c['past']);encoded=dino.encode(c['images'])
    with torch.no_grad():zl=predict_candidates(lewm,lc['embedding'],lc['past'],actions,batch_size=64)[:,0];zd=np.array([dino.predict(encoded,c['past'],a)[0].numpy() for a in actions])
    score['lewm_global']=np.linalg.norm(zl-zl[:1],axis=1).astype(float);d=zd.astype(float)-zd[:1].astype(float);score['dino_patch_rms']=np.sqrt(np.mean(np.sum(d*d,axis=2),axis=1));score['dino_patch_mean']=np.linalg.norm(d.mean(1),axis=1)
    return {'actions':actions,'labels':np.array(labels,dtype=str),'predicted_robot':robot,'predicted_costs':costs,'norms':norms,'lewm_predicted':zl,'dino_predicted':zd,**{'score_'+m:v for m,v in score.items()}},solver,bool(valid)

def choose(raw,rules,cfg):
    gain=(raw['predicted_costs'][0]-raw['predicted_costs'])/max(raw['predicted_costs'][0],1e-12);choices={}
    for method in cfg['methods']:
        rule=rules[method];ok=(raw['norms']>1e-6)&(gain>1e-8)&(raw['score_'+method]<=rule['tau']) if rule['enabled'] else np.zeros(len(gain),bool);ids=np.flatnonzero(ok);choices[method]=int(ids[np.lexsort((ids,raw['norms'][ids],raw['predicted_costs'][ids]))[0]]) if len(ids) else 0
    return choices
