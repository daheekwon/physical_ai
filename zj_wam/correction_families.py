"""Predeclared correction families and physical terminal constraints."""
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import null_space
from zj_wam.followup import controller_design


def orthonormal_columns(matrix):
    u,s,_=np.linalg.svd(matrix,full_matrices=False)
    keep=s>max(1e-12,s.max(initial=0)*1e-10)
    return u[:,keep]


def interior_basis(h):
    # Values at h grid points uniquely define a degree <=h-1 interpolant per axis.
    # Only the first/last correction values are fixed; derivative constraints differ.
    return np.eye(2*h)[:,2:-2]


def spline_basis(h):
    degree=3;coefficients=min(12,max(8,h));inner=np.linspace(0,1,coefficients-degree+1)[1:-1]
    knots=np.r_[np.zeros(degree+1),inner,np.ones(degree+1)]
    spline=BSpline(knots,np.eye(coefficients),degree)
    constraints=np.stack([spline(0),spline(1),spline.derivative()(0),spline.derivative()(1)])
    coeff=null_space(constraints);values=spline(np.linspace(0,1,h))@coeff
    scalar=orthonormal_columns(values)
    return np.kron(scalar,np.eye(2))


def terminal_constrained(basis,nominal,suffix,root,prefix,scale):
    _,_,response,_,_=controller_design(nominal,suffix,root,prefix,scale,basis)
    i=len(nominal)*10-1
    boundary=np.concatenate([response[i],(response[i]-response[i-1])/.01],axis=0)
    coefficients=null_space(boundary,rcond=1e-9)
    return basis@coefficients,{'original_dimension':basis.shape[1],'dimension':coefficients.shape[1],
                              'constraint_rank':int(basis.shape[1]-coefficients.shape[1]),'boundary_map':boundary}
