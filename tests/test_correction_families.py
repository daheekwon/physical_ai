import numpy as np
from zj_wam.correction_families import spline_basis,interior_basis,terminal_constrained
from zj_wam.followup import controller_design,optimize_controller


def test_spline_orthonormal_and_endpoint_zero():
    for h in [5,10,20]:
        b=spline_basis(h)
        np.testing.assert_allclose(b.T@b,np.eye(b.shape[1]),atol=1e-12)
        np.testing.assert_allclose(b[[0,1,-2,-1]],0,atol=1e-12)


def test_actual_terminal_constraint_preserves_position_and_velocity():
    h=5;nominal=np.zeros((h,2));suffix=np.zeros((3,2));root=np.array([100,200,0,0,0,0,0,0,0,0.]);prefix=np.tile(root,(30,1));scale=np.ones(2)
    basis,info=terminal_constrained(interior_basis(h),nominal,suffix,root,prefix,scale)
    assert basis.shape[1]==2
    _,_,response,_,_=controller_design(nominal,suffix,root,prefix,scale,basis)
    np.testing.assert_allclose(response[h*10-1],0,atol=1e-9)
    np.testing.assert_allclose(response[h*10-1]-response[h*10-2],0,atol=1e-9)
    raw,meta=optimize_controller(nominal,suffix,root,prefix,scale,.2,basis=basis)
    assert meta['feasible']
