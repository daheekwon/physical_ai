import numpy as np
from zj_wam.followup import DenseSimulator
from zj_wam.shifted_execution import ShiftedSimulator,noisy_history

def case():
 states=np.tile([256.,190.,256.,256.,0.,0.,0.],(45,1));actions=np.tile([0.,.3],(45,1));return actions,states

def test_null_shift_is_identical_and_gain_shift_replays_actual_past():
 a,s=case();base=DenseSimulator(a,s,20,42);null=ShiftedSimulator(a,s,20,42);changed=ShiftedSimulator(a,s,20,42,condition={'kp_multiplier':.8})
 ref=base.execute_dense(a[20:25],a[25:28]);same=null.execute_dense(a[20:25],a[25:28]);np.testing.assert_array_equal(ref['physics_trace'],same['physics_trace'])
 im,past=changed.restore(capture=True);root=changed.root_full_state.copy();first=changed.execute_dense(a[20:25],a[25:28]);second=changed.execute_dense(a[20:25],a[25:28]);np.testing.assert_array_equal(first['physics_trace'],second['physics_trace']);assert changed.env.k_p==80 and np.linalg.norm(root[:2]-ref['root_full_state'][:2])>.01
 np.testing.assert_array_equal(past,a[10:20]);assert len(im)==3
 for x in [base,null,changed]:x.close()

def test_noise_is_reproducible_without_mutating_clean_images():
 images=np.full((3,16,16,3),127,dtype=np.uint8);noisy=noisy_history(images,{'rgb_noise_std':15},42);np.testing.assert_array_equal(noisy,noisy_history(images,{'rgb_noise_std':15},42));assert not np.array_equal(noisy,images);assert np.all(images==127)
