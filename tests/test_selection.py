import pandas as pd,numpy as np
from zj_wam.selection import accepted,select_indices


def test_acceptance_and_selection_ignore_execution_labels_and_preserve_nominal_fallback():
    frame=pd.DataFrame({'state_id':['a','a','b'],'horizon':[5]*3,'actual_norm':[.1,.2,.1],
                        'x_score':[.1,.2,2.],'predicted_jerk':[8.,7.,9.],'nominal_predicted_jerk':[10.]*3,'nominal_physics_jerk':[10.]*3,
                        'object_error':[10.,0.,0.],'jerk_gain':[-.5,.3,.1],'nominal_contact':[True,False,True]})
    rule={'enabled':True,'tau':.3,'relative_radius':.1}
    assert select_indices(frame,'x',rule)=={'a':1,'b':-1}
    frame['nominal_physics_jerk']=[.1,1000.,1.];frame['object_error']=[0.,100.,100.];frame['jerk_gain']=[.9,-20.,-30.];frame['nominal_contact']=False
    assert select_indices(frame,'x',rule)=={'a':1,'b':-1}
    assert not accepted(frame,'x',{'enabled':False}).any()


def test_nonfinite_risk_proxy_is_never_accepted():
    frame=pd.DataFrame({'horizon':[5,5],'actual_norm':[.1,.1],'x_score':[np.inf,np.nan]})
    assert not accepted(frame,'x',{'enabled':True,'tau':100,'relative_radius':.4}).any()
