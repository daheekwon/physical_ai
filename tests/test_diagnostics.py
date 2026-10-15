import numpy as np
from scipy.fft import dct
from zj_wam.diagnostics import (DiagnosticSimulator, jerk_energy, polynomial_basis,
                               frequency_matched_directions)
from zj_wam.adapters import ReplaySimulator


def test_physical_jerk_units_on_cubic_trajectory():
    dt=.1
    t=np.arange(12)*dt
    q=np.column_stack([t**3,2*t**3])
    np.testing.assert_allclose(jerk_energy(q,dt),9*(6**2+12**2)*dt)


def test_polynomial_rank_boundaries_and_family_membership():
    for horizon in [5,10,20]:
        q=polynomial_basis(horizon)
        assert q.shape==(2*horizon,4)
        np.testing.assert_allclose(q.T@q,np.eye(4),atol=1e-12)
        np.testing.assert_allclose(q[[0,1,-2,-1]],0,atol=1e-12)
        u=np.linspace(0,1,horizon)
        basis=np.kron(np.column_stack([u**2*(1-u)**2,u**2*(1-u)**2*(2*u-1)]),np.eye(2))
        np.testing.assert_allclose(q@q.T@basis,basis,atol=1e-12)


def test_frequency_control_matches_entire_temporal_power_spectrum():
    rng=np.random.default_rng(1)
    d=rng.normal(size=10);d/=np.linalg.norm(d)
    power=np.sum(dct(d.reshape(5,2),axis=0,norm='ortho')**2,axis=1)
    for matched in frequency_matched_directions(d,rng):
        np.testing.assert_allclose(np.linalg.norm(matched),1)
        np.testing.assert_allclose(np.sum(dct(matched.reshape(5,2),axis=0,norm='ortho')**2,axis=1),power,atol=1e-12)


def test_instrumentation_preserves_contact_dynamics_and_boundary_support():
    initial=np.array([256.,190.,256.,256.,0.,0.,0.])
    states=np.tile(initial,(50,1));actions=np.tile([0.,.3],(50,1))
    actions[12:16]=[.2,-.1]
    base=ReplaySimulator(actions,states,10,42)
    diagnostic=DiagnosticSimulator(actions,states,10,42)
    reference=base.execute(actions[10:15])
    measured=diagnostic.execute_diagnostic(actions[10:15],actions[15:18])
    np.testing.assert_array_equal(measured['states'],reference['states'])
    np.testing.assert_array_equal(measured['contacts'],reference['contacts'])
    assert measured['control_trace'].shape==(11,7)
    assert measured['physics_trace'].shape==(83,7)
    assert measured['executed_jerk']>0
    repeat=diagnostic.execute_diagnostic(actions[10:15],actions[15:18])
    np.testing.assert_array_equal(measured['physics_trace'],repeat['physics_trace'])
    base.close();diagnostic.close()


def test_model_selection_does_not_consult_executed_jerk_or_object_errors():
    import pandas as pd
    from scripts.analyze_additional_phase1 import select_model
    frame=pd.DataFrame([
        {'relative_radius':.1,'predicted_change':.02,'roughness_reduction':.2,
         'action_roughness':8.,'applied_radius':.3,'trajectory_index':1,
         'executed_jerk':100.,'object_error':10.},
        {'relative_radius':.1,'predicted_change':.03,'roughness_reduction':.1,
         'action_roughness':9.,'applied_radius':.3,'trajectory_index':2,
         'executed_jerk':1.,'object_error':0.},
    ])
    rule={'relative_radius':.1,'tau':.04}
    assert select_model(frame,rule).trajectory_index==1
    frame['executed_jerk']=[1.,100.];frame['object_error']=[0.,10.]
    assert select_model(frame,rule).trajectory_index==1
