import json,numpy as np
from zj_wam.response_models import point_geometry,contact_rollout,score_models,controller_batch
from zj_wam.followup import controller_rollout
from zj_wam.io import resolve

cfg=json.loads(resolve('configs/followup_response_models.json').read_text())

def test_contact_geometry_and_pushing_symmetry():
    p=np.array([[0.,-20.],[76.,20.],[40.,-20.]])
    pose=np.zeros((3,3));d,n=point_geometry(p,pose)
    np.testing.assert_allclose(d,[20,16,20])
    np.testing.assert_allclose(n[[0,2]],[[0,-1],[0,-1]])
    root=np.array([0.,-30.,0.,0.,0.,0.,0.,0.,0.,0.])
    path=np.column_stack([np.zeros(50),np.linspace(-29,20,50)])[None]
    result,c=contact_rollout(path,root,cfg)
    assert c.any() and result[0,-1,1]>0
    np.testing.assert_allclose(result[:,:,2],0,atol=1e-12)
    root[0]=76;path[:,:,0]=76
    result,c=contact_rollout(path,root,cfg)
    assert not c.any();np.testing.assert_allclose(result,0,atol=1e-12)

def test_batch_controller_and_zero_correction_are_consistent():
    root=np.array([100.,100.,256.,256.,0.,10.,20.,0.,0.,0.])
    a=np.random.default_rng(3).uniform(-.3,.3,(3,5,2));b=controller_batch(a,root)
    for i in range(3):np.testing.assert_allclose(b[i],controller_rollout(a[i],root[:2],root[5:7]),atol=1e-10)
    history=np.tile(root,(30,1));identical=np.tile(a[0:1],(3,1,1))
    scores,_=score_models(identical,np.zeros((3,2)),root,history,cfg)
    for v in scores.values():np.testing.assert_allclose(v,0,atol=1e-12)
