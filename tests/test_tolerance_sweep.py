"""Regression for state/metadata alignment in tolerance calibration."""
import numpy as np
import pandas as pd
import pytest
from scripts.analyze_followup_tolerance_sweep import packed,choose,metrics,errors
from scripts.calibrate_followup_selection import selected_rows


def fixture_frame():
    rows=[]
    for state,contact in [('a',True),('b',False)]:
        for index,norm,cost,score,error in [(1,.1,80.,.2,0.),(3,.5,50.,.4,3.)]:
            rows.append(dict(state_id=state,initial_pose_family=1 if state=='a' else 2,nominal_contact=contact,index=index,horizon=5,actual_norm=norm,predicted_jerk=cost,nominal_predicted_jerk=100.,jerk_gain=1-cost/100.,position_max=error,rotation_max=0.,coverage_loss=0.,coverage_suffix_loss=0.,object_error=error/2.,example_score=score))
    return pd.DataFrame(rows)


def test_sparse_source_bank_and_nominal_fallback_match_selector():
    frame=fixture_frame();a,meta=packed(frame,['example']);rule={'enabled':True,'relative_radius':.4,'tau':.3}
    assert a['cost'].shape==(2,4) and np.all(a['cost'][:,0]==100.)
    idx=choose(a,'example',rule,1,1)
    np.testing.assert_array_equal(idx,selected_rows(frame,'example',rule)['index'])
    np.testing.assert_array_equal(choose(a,'example',{**rule,'tau':.01},1,1),[0,0])


def test_metadata_from_another_split_cannot_silently_set_contact_risk():
    a,meta=packed(fixture_frame(),['example']);idx=np.ones(2,dtype=int);e=errors(a,{'position':2.,'rotation':.035,'coverage_loss':.02},1,1)
    assert metrics(a,meta,idx,e)['contact_risk']==0.
    with pytest.raises(AssertionError):metrics(a,meta.iloc[::-1],idx,e)
