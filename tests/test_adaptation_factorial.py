import json
from pathlib import Path
import numpy as np
from zj_wam.adaptation_factorial import make_stage,score,pick,past_holdout_error
from zj_wam.repeated_refinement import generate_bank
from zj_wam.response_models import score_models,controller_batch,contact_rollout
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/followup_response_models.json').read_text())

def inputs():
    nominal=np.tile([.03,.015],(5,1));suffix=np.tile([.02,.01],(3,1));past=np.tile([.03,.015],(10,1))
    root=np.array([100.,100.,260.,260.,0.,0.,0.,0.,0.,0.]);history=np.tile(root,(3,1));prefix=np.tile(root,(30,1));scale=np.array([.1,.1]);return nominal,suffix,past,root,history,prefix,scale

def test_staged_bank_is_same_candidate_multiset_and_scores():
    n,s,p,r,h,pre,scale=inputs();radii=[.05,.1,.2,.4]
    a,_,_=make_stage(n,s,r,pre,p,scale,123,radii,1);b,_,_=make_stage(n,s,r,pre,p,scale,123,radii,2);new=np.concatenate([a,b])
    old,_=generate_bank(n,s,r,pre,p,scale,123,radii)
    assert len(new)==len(old)
    remaining=list(range(len(old)))
    for x in new:
        j=min(remaining,key=lambda i:np.max(np.abs(old[i]-x)))
        np.testing.assert_allclose(x,old[j],atol=1e-12);remaining.remove(j)
    c,norm,v=score(new,n,s,h,pre,scale,CFG);expected,_=score_models(new,s,r,h,CFG)
    np.testing.assert_allclose(v,expected['adaptive'],atol=1e-12)
    cb,nb,vb=score(np.concatenate([n[None],b]),n,s,h,pre,scale,CFG)
    np.testing.assert_allclose(v[len(a):],vb[1:],atol=1e-12)

def test_radius_and_validity_gate_are_real_constraints():
    costs=np.array([100.,60.,30.]);norms=np.array([0.,.1,.8]);scores=np.array([0.,.2,.1])
    assert pick(costs,norms,scores,True,.3,.4)==2
    assert pick(costs,norms,scores,True,.3,.05)==1
    assert pick(costs,norms,scores,False,.3,.4)==0

def test_past_holdout_detects_endpoint_mismatch():
    n,s,p,r,h,pre,scale=inputs()
    # No object contact; exact past controller endpoint supplied as current observation.
    q=controller_batch(p[-5:][None],h[1]);h[2,:2]=q[0,-1]
    error,parts=past_holdout_error(h,p,CFG)
    assert error<1e-9
    h[2,2]+=4
    error,parts=past_holdout_error(h,p,CFG)
    assert abs(parts[0]-2)<1e-9 and error>=2

def test_compute_arm_really_skips_second_stage(monkeypatch):
    import zj_wam.adaptation_factorial as module
    n,s,p,r,h,pre,scale=inputs();acfg=json.loads((ROOT/'configs/followup_adaptation_factorial.json').read_text())
    from zj_wam.diagnostics import polynomial_basis
    n=(polynomial_basis(5)@np.full(4,.1)).reshape(5,2)
    s=np.zeros((3,2))
    scale=np.ones(2)
    monkeypatch.setattr(module,'visual_history',lambda images,proprio,cfg:(h,True))
    original=module.make_stage;calls=[]
    def tracked(*args,**kwargs):
        calls.append(args[-1]);return original(*args,**kwargs)
    monkeypatch.setattr(module,'make_stage',tracked)
    rule={'relative_radius':.4,'tau':.65,'enabled':True}
    _,d,_=module.decide(None,None,p,pre,n,s,scale,123,[.05,.1,.2,.4],CFG,acfg,rule,'compute')
    assert d['early_stop'] and calls==[1] and d['scored_candidates']==5
    calls.clear()
    _,d,_=module.decide(None,None,p,pre,n,s,scale,123,[.05,.1,.2,.4],CFG,acfg,rule,'fixed')
    assert calls==[1,2] and d['scored_candidates']>5
