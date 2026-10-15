import numpy as np
from zj_wam.cumulative_allowance import deviation_score,remaining_margin
EPS={'position':2.,'rotation':np.deg2rad(2.)}

def test_existing_error_is_not_erased_by_zero_correction():
    ref=np.zeros((1,80,3));pred=ref.copy();pred[:,:,0]=3
    assert deviation_score(pred,ref,EPS)[0]==1.5
    root=np.zeros(10);other=root.copy();root[2]=3
    margin,error=remaining_margin(root,other,EPS)
    assert margin==0 and error==1.5

def test_time_aligned_maximum_and_wrapped_orientation():
    ref=np.zeros((1,80,3));ref[:,:,2]=np.pi-.01;pred=ref.copy();pred[:,:,2]=-np.pi+.01
    np.testing.assert_allclose(deviation_score(pred,ref,EPS),[.02/EPS['rotation']])
    pred[0,40,0]=4
    assert deviation_score(pred,ref,EPS)[0]==2
    assert np.linalg.norm(pred[0,-1,:2]-ref[0,-1,:2])==0
