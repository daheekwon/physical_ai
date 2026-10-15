import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from zj_wam.runtime_benchmark import causal_inputs,stages
from zj_wam import repeated_refinement as repeated
from zj_wam import policy_refinement as policy
from zj_wam import followup
from scripts.analyze_followup_runtime import lock


def test_runtime_causal_inputs_remove_object_truth():
    proprio=np.arange(30).reshape(3,10);prefix=np.arange(300).reshape(30,10);a,b=causal_inputs(proprio,prefix)
    np.testing.assert_array_equal(a[:,[0,1,5,6]],proprio[:,[0,1,5,6]])
    np.testing.assert_array_equal(b[:,[0,1,5,6]],prefix[:,[0,1,5,6]])
    assert np.count_nonzero(a[:,[2,3,4,7,8,9]])==0 and np.count_nonzero(b[:,[2,3,4,7,8,9]])==0


def test_runtime_profiles_restore_patched_functions_on_failure():
    import torch
    model=SimpleNamespace(device='cpu',predictor=lambda:None,model=SimpleNamespace(predict=lambda:None,encode=lambda:None))
    before=[repeated.generate_bank,policy.generate_bank,followup.minimize,model.predictor,model.model.predict]
    with pytest.raises(RuntimeError):
        with stages(model):raise RuntimeError('profile interrupted')
    after=[repeated.generate_bank,policy.generate_bank,followup.minimize,model.predictor,model.model.predict]
    assert before==after


def test_latency_cap_requires_calibration_risk_and_time():
    rows=[];times=[]
    for profile,cost,joint,risk in [('small',.01,False,False),('medium',.04,True,False),('full',.08,True,True)]:
        for state in range(4):
            rows.append({'method':'m','profile':profile,'active':True,'nominal_contact':True,'violation':risk,'joint':joint})
            times.append({'method':'m','profile':profile,'total_seconds':cost})
    cfg={'target_violation_rate':.1,'minimum_active_coverage':.25};tcfg={'latency_quantile':.95,'methods':['m'],'latency_budgets_seconds':[.005,.02,.05,.1]}
    d=lock(pd.DataFrame(rows),pd.DataFrame(times),cfg,tcfg,{})
    assert not d['rules'][0]['enabled']
    assert [r['selection']['profile'] for r in d['rules'][1:]]==['small','medium','medium']
