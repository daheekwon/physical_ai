import numpy as np
from zj_wam.followup import DenseSimulator
from zj_wam.adapters import ReplaySimulator
from zj_wam.repeated_refinement import visual_history


def test_proprio_history_matches_render_timestamps_after_modified_prefix():
    states=np.tile([256.,190.,256.,256.,0.,0.,0.],(40,1));actions=np.tile([0.,.3],(40,1));actions[11:14]=[.2,-.1]
    for t in [10,20]:
        sim=DenseSimulator(actions,states,t,42);sim.restore(capture=True)
        for k,when in enumerate([t-10,t-5,t]):
            reference=ReplaySimulator(actions,states,when,42);actual=reference.restore()
            np.testing.assert_array_equal(sim.context_control[k,:7],actual);reference.close()
        sim.close()


def test_visual_history_strips_privileged_object_state(monkeypatch):
    import zj_wam.repeated_refinement as module
    monkeypatch.setattr(module,'estimate_pose',lambda image,cfg:(np.array([1.,2.,.3]),{'valid':True}))
    images=np.zeros((3,4,4,3),dtype=np.uint8);proprio=np.arange(30,dtype=float).reshape(3,10)
    first,valid=visual_history(images,proprio,{})
    proprio[:,[2,3,4,7,8,9]]+=10000
    second,_=visual_history(images,proprio,{})
    assert valid
    np.testing.assert_array_equal(first,second)
