"""Validate original expert absolute targets on already-exposed families only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,zarr
from stable_worldmodel.envs.pusht.env import PushT
from zj_wam.io import resolve,sha256,write_json
from zj_wam.policy_refinement import full_state
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def scene(initial,relative):
    env=PushT(resolution=224,relative=relative,render_action=False);env.reset(seed=0);env._setup();env.agent.position=initial[:2].tolist();env.agent.velocity=(0.,0.);env.block.angle=float(initial[4]);env.block.position=initial[2:4].tolist();env.block.velocity=(0.,0.);env.block.angular_velocity=0.;env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block);env.goal_pose=np.array([256.,256.,np.pi/4]);env._set_goal_state(np.array([256.,400.,256.,256.,np.pi/4,0.,0.]));np.testing.assert_array_equal(full_state(env)[:5],initial);return env

def rollout(initial,targets,relative_actions=None):
    env=scene(initial,relative_actions is not None);physics=[];control=[full_state(env)];images=[env.render().copy()];converted=[];old=env.space.step
    def step(dt):old(dt);physics.append(full_state(env))
    env.space.step=step
    for i,target in enumerate(targets):
        converted.append((target-np.array(env.agent.position))/100.);env.step(target if relative_actions is None else relative_actions[i]);control.append(full_state(env));images.append(env.render().copy())
    env.space.step=old;env.close();return {'physics':np.array(physics),'states':np.array(control),'images':np.array(images),'relative_actions':np.array(converted)}

def main():
    audit=read(BASE/'protocol/original_expert_new_families.json');path=resolve('data/datasets/diffusion-policy-official/pusht.zip');assert sha256(path)==audit['archive_sha256'];store=zarr.ZipStore(str(path),mode='r');root=zarr.open_group(store,path='pusht/pusht_cchi_v7_replay.zarr',mode='r');ends=root['meta/episode_ends'][:];starts=np.r_[0,ends[:-1]];ids=[1,2,3];reserved={r['episode'] for r in audit['reserved_expert_episodes']};assert not set(ids)&reserved and all(audit['old_family_match_for_official_episode'][i]>=0 for i in ids);out=BASE/'original_expert_adapter';out.mkdir(exist_ok=True);rows=[]
    for i in ids:
        states=root['data/state'][starts[i]:ends[i]].astype(float);targets=root['data/action'][starts[i]:ends[i]].astype(float);a=rollout(states[0],targets);b=rollout(states[0],targets,a['relative_actions']);d=abs(a['physics']-b['physics']);assert d.max()<1e-8,(i,d.max());np.testing.assert_array_equal(a['images'],b['images']);np.testing.assert_allclose(a['relative_actions'],b['relative_actions'],atol=1e-10,rtol=0);err=a['states'][:-1,:5]-states;angle=abs(np.arctan2(np.sin(err[:,4]),np.cos(err[:,4])));row={'episode':i,'actions':len(targets),'absolute_relative_fullstate_max_error':float(d.max()),'absolute_relative_RGB_exact':True,'relative_action_abs_max':float(abs(a['relative_actions']).max()),'relative_actions_outside_bound':int((abs(a['relative_actions']).max(1)>1).sum()),'recorded_pusher_max_error_px':float(np.linalg.norm(err[:,:2],axis=1).max()),'recorded_object_max_error_px':float(np.linalg.norm(err[:,2:4],axis=1).max()),'recorded_object_max_error_deg':float(np.rad2deg(angle.max()))};rows.append(row);np.savez_compressed(out/f'episode_{i:03d}.npz',recorded_states=states,absolute_targets=targets,**{'absolute_'+k:v for k,v in a.items()},**{'relative_'+k:v for k,v in b.items()});print(row,flush=True)
    store.close();write_json(out/'audit.json',{'development_original_episodes':ids,'reserved_episodes_evaluated':False,'rows':rows,'source_sha256':sha256(Path(__file__)),'archive_sha256':sha256(path),'initial_family_audit_sha256':sha256(BASE/'protocol/original_expert_new_families.json'),'raw_inputs':{p.name:sha256(p) for p in sorted(out.glob('episode_*.npz'))},'scope':'Three prespecified already-exposed initialfamilies. SameSWMscene initialrecordedpose and zero missingvelocities; targetabsolute and compiledfrozenrelative densephysics/RGB equivalence. Recorded expert reproduction error reported separately. No clipped action in pairedconversion; any futureoutofbound policy must be fixed before21familytest. No modelscore/threshold/outcome on reserved21.'})
if __name__=='__main__':main()
