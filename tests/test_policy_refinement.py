import numpy as np
from zj_wam.policy_refinement import make_scene,full_state,absolute_to_relative,context_at,evaluate_local,coverage


def test_absolute_compilation_matches_current_controller_and_obeys_bounds():
 env=make_scene(210000);root=full_state(env);targets=np.array([[260.,100.],[300.,140.],[340.,190.],[430.,250.],[480.,320.],[400.,380.],[370.,360.],[350.,330.]])
 actions,meta=absolute_to_relative(targets,root[:2],root[5:7]);assert np.max(np.abs(actions))<=1
 q=root[:2].copy();v=root[5:7].copy()
 for a in actions:
  target=q+100*a
  for _ in range(10):v+=(100*(target-q)-20*v)*.01;q+=v*.01
  env.step(a);np.testing.assert_allclose(full_state(env)[:2],q,atol=1e-5,rtol=0)
 assert meta['clipped_actions']>0;env.close()


def test_context_strips_object_state_and_uses_only_past_timestamps():
 t=16;images=np.arange(17)[:,None,None,None];states=np.arange(170).reshape(17,10);actions=np.arange(32).reshape(16,2);physics=np.arange(1600).reshape(160,10)
 im,past,proprio,prefix,ids=context_at(t,images,states,actions,physics)
 assert ids==[6,11,16];np.testing.assert_array_equal(im[:,0,0,0],ids);np.testing.assert_array_equal(past,actions[6:16]);np.testing.assert_array_equal(proprio[:,[2,3,4,7,8,9]],0)


def test_local_counterfactual_replays_identical_prefix_without_snapshot_cache_loss():
 env=make_scene(210000);delivered=np.tile([.12,-.08],(16,1));nominal=np.tile([-.1,.05],(8,1));alltrace=[];original=env.space.step
 def step(dt):original(dt);alltrace.append(full_state(env))
 env.space.step=step
 for a in delivered:env.step(a)
 prefix=np.array(alltrace[-30:]);start=len(alltrace);cov=[]
 for a in nominal:env.step(a);cov.append(coverage(env))
 actual={'physics':np.array(alltrace[start:]),'coverage':cov};metric,trace,_=evaluate_local(210000,delivered,nominal,actual,prefix,{'position':2.,'rotation':np.deg2rad(2),'coverage_loss':.02})
 np.testing.assert_array_equal(trace,actual['physics']);assert metric['object_error']==0 and metric['jerk_gain']==0
 env.space.step=original;env.close()


def test_legacy_initialization_matches_official_native_fixture():
 env=make_scene(100000)
 expected=[235.,124.,299.61288530901334,145.20158073114663,-.9049274284249247,0.,0.,0.,0.,0.]
 np.testing.assert_allclose(full_state(env),expected,atol=1e-12,rtol=0);env.close()
