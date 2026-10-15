import numpy as np
from zj_wam.followup import DenseSimulator, controller_rollout, controller_design, optimize_controller, dense_metrics
from zj_wam.adapters import ReplaySimulator
from zj_wam.diagnostics import jerk_energy


def case():
    initial=np.array([256.,190.,256.,256.,0.,0.,0.])
    states=np.tile(initial,(50,1))
    actions=np.tile([0.,.3],(50,1));actions[12:16]=[.2,-.1]
    return actions,states


def test_dense_instrumentation_unchanged_and_controller_matches_contact_execution():
    a,s=case();sim=DenseSimulator(a,s,10,42);base=ReplaySimulator(a,s,10,42)
    ref=sim.execute_dense(a[10:15],a[15:18])
    other=base.execute(a[10:15])
    np.testing.assert_array_equal(ref['control'][:5,:7],other['states'])
    assert ref['physics'].shape==(80,10)
    assert ref['physics_trace'].shape==(110,10)
    assert (ref['physics_contacts']>0).any()
    root=ref['root_full_state']
    pred=controller_rollout(a[10:18],root[:2],root[5:7])
    np.testing.assert_allclose(pred,ref['physics'][:,:2],atol=1e-3,rtol=0)
    repeat=sim.execute_dense(a[10:15],a[15:18])
    np.testing.assert_array_equal(ref['physics_trace'],repeat['physics_trace'])
    sim.close();base.close()


def test_controller_optimizer_reduces_measured_physics_jerk_within_bounds():
    a,s=case();sim=DenseSimulator(a,s,10,42)
    ref=sim.execute_dense(a[10:15],a[15:18]);root=ref['root_full_state']
    raw,info=optimize_controller(a[10:15],a[15:18],root,sim.prefix_physics,np.array([.2,.3]),.7)
    assert info['feasible']
    result=sim.execute_dense(raw,a[15:18])
    assert result['physics_jerk']<ref['physics_jerk']*.99
    q,base,response,j0,design=controller_design(a[10:15],a[15:18],root,sim.prefix_physics,[.2,.3])
    expected=np.sum((j0+design@info['coefficient'])**2)*.01
    np.testing.assert_allclose(expected,result['physics_jerk'],rtol=1e-5)
    np.testing.assert_allclose(raw[[0,-1]],a[[10,14]],atol=1e-12)
    sim.close()


def test_dense_preservation_detects_between_control_rotation_and_suffix_coverage():
    a,s=case();sim=DenseSimulator(a,s,10,42)
    ref=sim.execute_dense(a[10:15],a[15:18]);tol={'position':2.,'rotation':np.deg2rad(2),'coverage_loss':.02}
    changed=dict(ref);changed['physics']=ref['physics'].copy();changed['physics'][1,4]+=.1
    metrics=dense_metrics(changed,ref,tol)
    assert not metrics['preserved'] and metrics['rotation_control_max']==0
    changed=dict(ref);changed['coverage_suffix']=ref['coverage_suffix']-.03
    assert not dense_metrics(changed,ref,tol)['preserved']
    sim.close()


def test_effect_orientation_wrap_and_native_history_alignment():
    from zj_wam.readouts import effects,errors,history_features
    pose=np.array([[[1.,2.,np.deg2rad(179)]],[[3.,2.,np.deg2rad(-179)]]])
    np.testing.assert_allclose(effects(pose)[1,0],[2.,0.,np.deg2rad(2)],atol=1e-12)
    context=np.array([[[0.],[1.],[2.]]]);future=np.array([[[3.],[4.]],[[30.],[40.]]])
    got=history_features(context,future)
    np.testing.assert_array_equal(got[0],[[1,2,3],[2,3,4]])
    np.testing.assert_array_equal(got[1],[[1,2,30],[2,30,40]])
