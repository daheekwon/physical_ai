import numpy as np
from zj_wam.cube_experiment import basis,fit_response,predict_response,angular_jerk
from zj_wam.cube_execution import quat_error


def test_cube_polynomial_preserves_gripper_and_boundaries():
    q=basis();assert q.shape==(25,8)
    np.testing.assert_allclose(q.T@q,np.eye(8),atol=1e-12)
    delta=(q@np.arange(8)).reshape(5,5)
    np.testing.assert_allclose(delta[:,4],0,atol=1e-14)
    np.testing.assert_allclose(delta[[0,-1]],0,atol=1e-14)


def test_past_pd_fit_recovers_excited_dynamics():
    rng=np.random.default_rng(91);actions=np.c_[rng.uniform(-.6,.6,(50,4)),np.zeros(50)]
    root=np.array([.4,0,.2,0.]);gains=np.array([[120.,23.],[85.,16.],[140.,22.],[100.,18.]])
    bounds=np.array([[-5,-5,-5],[5,5,5]]);points=predict_response(actions,root,np.zeros(4),gains,bounds)
    points=np.r_[root[None],points];targets=[]
    for i,a in enumerate(actions):targets.extend([points[i*25]+a[:4]*[.05,.05,.05,.3]]*25)
    fitted,residual=fit_response(points,np.array(targets),.002)
    np.testing.assert_allclose(fitted,gains,rtol=.01,atol=.1)
    assert np.max(residual)<.02


def test_quaternion_sign_does_not_change_rotation_or_angular_jerk():
    t=np.arange(60)*.002;angle=.4*t;q=np.c_[np.cos(angle/2),np.zeros((60,2)),np.sin(angle/2)]
    flipped=q.copy();flipped[::2]*=-1
    assert np.max(quat_error(q,flipped))<1e-7
    assert angular_jerk(q,.002)<1e-12
    np.testing.assert_allclose(angular_jerk(flipped,.002),angular_jerk(q,.002),atol=1e-12)


def test_cube_readout_rotation_projection_and_relative_effect():
    from scipy.spatial.transform import Rotation
    from zj_wam.cube_readouts import target,rotations,capture_errors,effects
    r=Rotation.from_euler('z',np.array([[0.],[.2]]));q=r.as_quat()[:,[3,0,1,2]];pose=np.c_[np.array([[.2,0,.1],[.201,0,.1]]),q]
    y=target(pose,np.zeros((2,6)),np.zeros((2,3)));np.testing.assert_allclose(rotations(y).as_matrix(),r.as_matrix(),atol=1e-12)
    e=capture_errors(y,y);assert all(np.max(x)<1e-12 for x in e.values())
    dp,tp,dr,tr=effects(y,y);np.testing.assert_allclose(dp[1],[.001,0,0],atol=1e-12);np.testing.assert_allclose(dr.magnitude(),[0,.2],atol=1e-12)
    # Projection must yield a proper rotation even for a reflected linear output.
    bad=y.copy();bad[:,3:12]=np.tile(np.diag([1,1,-1]).reshape(1,9),(2,1));np.testing.assert_allclose(np.linalg.det(rotations(bad).as_matrix()),1.,atol=1e-12)


def test_cube_visual_symmetry_metric_does_not_replace_full_rotation():
    from scipy.spatial.transform import Rotation
    from zj_wam.cube_readouts import target,capture_errors
    a=target(np.array([[0.,0.,.02,1.,0.,0.,0.]]),np.zeros((1,6)),np.zeros((1,3)))
    q=Rotation.from_rotvec([0,0,np.pi/2]).as_quat()[[3,0,1,2]]
    b=target(np.r_[0.,0.,.02,q][None],np.zeros((1,6)),np.zeros((1,3)))
    e=capture_errors(a,b);np.testing.assert_allclose(e['rotation_rad'],np.pi/2,atol=1e-12);assert e['rotation_mod_cube_symmetry_rad'][0]<1e-12


def test_cube_canonical_target_is_invariant_under_cube_group():
    from scipy.spatial.transform import Rotation
    from zj_wam.cube_readouts import target,canonical_target
    reference=Rotation.from_rotvec([.2,-.3,.4]);group=Rotation.create_group('O');rotations=reference*group;q=rotations.as_quat()[:,[3,0,1,2]];poses=np.c_[np.zeros((24,3)),q];y=target(poses,np.zeros((24,6)),np.zeros((24,3)));canonical=canonical_target(y)
    np.testing.assert_allclose(canonical[:,3:12],np.broadcast_to(canonical[0,3:12],(24,9)),atol=1e-9)
    # Fundamental-domain boundary must not depend on group enumeration.
    rotations=Rotation.from_rotvec([0,0,np.pi/4])*group;q=rotations.as_quat()[:,[3,0,1,2]];y=target(np.c_[np.zeros((24,3)),q],np.zeros((24,6)),np.zeros((24,3)));canonical=canonical_target(y)
    np.testing.assert_allclose(canonical[:,3:12],np.broadcast_to(canonical[0,3:12],(24,9)),atol=1e-9)


def test_cube_future_history_uses_last_two_past_frames_and_endpoint():
    from zj_wam.cube_readouts import current_features,future_features
    h=np.arange(576).reshape(3,192);future=np.full((2,192),1000.)
    np.testing.assert_array_equal(current_features(h,'frame'),h[-1])
    x=future_features(h,future,'history').reshape(2,3,192)
    np.testing.assert_array_equal(x[:,:2],np.broadcast_to(h[-2:],(2,2,192)))
    np.testing.assert_array_equal(x[:,2],future)


def test_cube_canonical_effect_alignment_uses_decoded_nominal_only():
    from scipy.spatial.transform import Rotation
    from zj_wam.cube_readouts import target,effects
    rotations=Rotation.from_rotvec(np.array([[0,0,.75],[0,0,.80]]));q=rotations.as_quat()[:,[3,0,1,2]];truth=target(np.c_[np.zeros((2,3)),q],np.zeros((2,6)),np.zeros((2,3)))
    from zj_wam.cube_readouts import canonical_target
    decoded=canonical_target(truth);_,_,effect,actual=effects(decoded,truth,True)
    np.testing.assert_allclose((effect*actual.inv()).magnitude(),0,atol=1e-10)
    assert effects(decoded,truth,False)[2].magnitude()[1]>1.
