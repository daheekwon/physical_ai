"""Acceptance/selection consume predictions and candidate magnitudes only."""
import numpy as np


def accepted(frame,method,rule):
    if not rule.get('enabled',False):return np.zeros(len(frame),dtype=bool)
    score=frame[method+'_score'].to_numpy(float)
    return (np.isfinite(score)&(score<=rule['tau'])&
            (frame.actual_norm.to_numpy()<=rule['relative_radius']*np.sqrt(2*frame.horizon.to_numpy())+1e-7)&
            (frame.actual_norm.to_numpy()>1e-6))


def select_indices(frame,method,rule,minimum_gain=1e-8):
    """Index is a dataframe row index; -1 means nominal fallback. No execution labels."""
    allowed=accepted(frame,method,rule)
    result={}
    for sid,positions in frame.groupby('state_id',sort=True).indices.items():
        positions=np.asarray(positions)
        take=positions[allowed[positions]]
        if len(take):
            costs=frame.predicted_jerk.to_numpy()[take]
            norms=frame.actual_norm.to_numpy()[take]
            chosen=take[np.lexsort((norms,costs))[0]]
            nominal=float(frame.nominal_predicted_jerk.iloc[positions[0]])
            gain=(nominal-float(frame.predicted_jerk.iloc[chosen]))/nominal if nominal>1e-8 else 0.
            result[sid]=int(chosen) if gain>minimum_gain else -1
        else:result[sid]=-1
    return result
