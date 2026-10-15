import numpy as np,torch
from zj_wam.strong_optimizers import nonlinear_solve,quadratic_solve,geometric_solve
from zj_wam.followup import controller_design

def test_nonlinear_solver_keeps_a_model_feasible_improvement_with_budget():
    predict=lambda a:torch.stack([a[0]+.2*a[0]**3,a[1]])
    nominal=np.zeros((1,2));j0=np.array([1.,1.]);design=-np.eye(2)
    raw,info=nonlinear_solve(predict,torch.zeros(2),np.eye(2),nominal,np.eye(2),j0,design,.8,.2,[np.zeros(2),np.ones(2)*.3],budget=32)
    assert info['forward_calls']<=32 and info['backward_calls']<32
    assert info['objective_ratio']<1
    assert np.linalg.norm(predict(torch.tensor(raw.reshape(-1))).numpy())<=.2*(1+2e-5)

def test_geometric_adapter_validates_all_prefixes_and_equal_area_candidates():
    h=5;nominal=np.array([[.1,.2],[-.1,.1],[.2,-.1],[.1,.2],[-.1,.1]])
    root=np.zeros(10);root[:2]=100;prefix=np.tile(root,(30,1));scale=np.ones(2);q=np.eye(2*h)
    _,base,response,_,_=controller_design(nominal,np.zeros((3,2)),root,prefix,scale,q)
    raw,info=geometric_solve(nominal,q,base,response,prefix[[9,19,29],:2],.5,1.,'groove')
    assert len(info['regions'])==5
    assert np.max(np.abs(raw))<=1+1e-6
    selected=info['regions'][info['selected_region']];ref=info['regions'][0]
    assert selected['jerk_score']<=ref['jerk_score']+1e-8
    assert selected['deviation']<=1.5*max(ref['deviation'],1e-12)+1e-8
    assert selected['feasible']

def test_strong_setting_execution_labels_do_not_change_activation():
    import pandas as pd
    from scripts.calibrate_followup_strong_baselines import rows_for_setting
    f=pd.DataFrame({'state_id':['a','b'],'initial_pose_family':[1,2],'nominal_contact':[True,False],
                    'method':['nonlinear_wm']*2,'setting':[1.,1.],'relative_radius':[.4,.4],'nominal_predicted_jerk':[10.,10.],
                    'predicted_jerk':[8.,11.],'actual_norm':[.1,.1],'index':[1,1],'jerk_gain':[-.5,.9],'object_error':[5.,0.],
                    'preserved':[False,True],'optimizer_seconds':[.2,.3],'forward_calls':[20,30],'backward_calls':[19,29]})
    before=rows_for_setting(f,'nonlinear_wm',1.,.4)
    assert before.active.tolist()==[True,False]
    f['jerk_gain']=[.99,-.99];f['object_error']=[0.,100.];f['preserved']=[True,False]
    after=rows_for_setting(f,'nonlinear_wm',1.,.4)
    assert after.active.tolist()==before.active.tolist()
    assert after['index'].tolist()==before['index'].tolist()
