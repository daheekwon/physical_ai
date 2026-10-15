import numpy as np
from zj_wam.adapters import ReplaySimulator


def test_restore_replays_hidden_velocity_and_contact_history():
    initial = np.array([256.,190.,256.,256.,0.,0.,0.])
    states = np.tile(initial,(50,1))
    actions = np.tile([0.,.3],(50,1))
    sim = ReplaySimulator(actions,states,8,42)
    sim.restore()
    # A position-only restoration would lose this nonzero block velocity.
    assert np.linalg.norm(sim.env.block.velocity) > 1
    continuous = sim.execute(actions[:5],restore=False)
    replayed = sim.execute(actions[:5])
    np.testing.assert_array_equal(continuous['states'],replayed['states'])
    assert np.all(continuous['contacts']>0)
    assert 0 <= continuous['task_score'] <= 1
    sim.close()


def test_goal_setup_does_not_advance_recorded_initial_pose():
    initial=np.array([256.,190.,256.,256.,0.,0.,0.])
    states=np.tile(initial,(20,1))
    states[-1]=[256.,256.,256.,256.,0.,0.,0.]  # deliberately overlapping goal
    sim=ReplaySimulator(np.zeros((20,2)),states,0,42)
    np.testing.assert_array_equal(sim.restore(),initial)
    sim.close()
