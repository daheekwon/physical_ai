"""Frozen PushT filter/geometric settings; only robot history is consumed."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from zj_wam.followup import controller_design, controller_rollout
from zj_wam.diagnostics import jerk_energy
from zj_wam.strong_optimizers import geometric_solve


def refine(method, nominal, suffix, root, prefix, past, scale, rule):
    h = len(nominal)
    # Explicitly exclude object pose/contact and physical shift parameters.
    robot = np.zeros_like(root)
    robot[[0, 1, 5, 6]] = root[[0, 1, 5, 6]]
    history = np.zeros_like(prefix)
    history[:, :2] = prefix[:, :2]
    def cost(a):
        q = controller_rollout(np.concatenate([a, suffix]), robot[:2], robot[5:7])
        return jerk_energy(np.concatenate([history[:, :2], q]), .01)
    before = cost(nominal)
    if not rule['enabled']:
        return nominal.copy(), {'active': False, 'actual_norm': 0., 'predicted_gain': 0., 'predicted_nominal_jerk': before, 'predicted_selected_jerk': before, 'solver_success': True}
    radius = rule['radius'] * np.sqrt(2*h)
    if method == 'gaussian_action':
        candidate = gaussian_filter1d(np.concatenate([past[-3:], nominal, suffix]), rule['setting'], axis=0, mode='nearest')[3:3+h]
        delta = (candidate-nominal)/scale
        candidate = nominal + min(1., radius/max(np.linalg.norm(delta), 1e-12))*(candidate-nominal)
        info = {'solver_success': True}
    else:
        polynomial = method.endswith('_polynomial_executed')
        basis = None if polynomial else np.eye(2*h)
        q, base, response, j0, design = controller_design(nominal, suffix, robot, history, scale, basis)
        candidate, info = geometric_solve(nominal, q*np.tile(scale,h)[:,None], base, response,
            history[[9,19,29],:2], radius, rule['setting'], method.split('_')[0], (j0,design) if polynomial else None)
    norm = float(np.linalg.norm((candidate-nominal)/scale))
    after = cost(candidate)
    gain = (before-after)/before if before>1e-8 else 0.
    active = norm>1e-6 and gain>1e-8
    assert np.max(np.abs(candidate))<=1+1e-6
    assert norm<=radius+2e-6
    if not active:
        candidate = nominal.copy(); norm=0.; after=before; gain=0.
    return candidate, {'active': bool(active), 'actual_norm': norm, 'predicted_gain': gain,
        'predicted_nominal_jerk': before, 'predicted_selected_jerk': after, 'solver_success': info['solver_success'], 'optimizer': info}
