"""Calibration-only model allowance; this is an estimate, never a safety proof."""
from dataclasses import dataclass
import numpy as np
import torch
from zj_wam.sensitivity import action_jacobian, eigensystem


@dataclass
class ActionAllowance:
    candidate_basis: np.ndarray
    sensitivity_values: np.ndarray
    allowed_correction_scale: np.ndarray
    predicted_change_budget: float
    diagnostics: dict


def estimate_action_allowance(predict, a0, calibration, phase='pooled', low_rank=3,
                              valid_action=None, radius_grid=None):
    """Return basis and signed sampled scales using only frozen model outputs.

    `predict` closes over the observation/history; no simulator outcome is accepted.
    Columns of allowed_correction_scale are minus, plus. Combination candidates
    still require the SAME joint predicted-change check. Intervals between sampled
    scales are not certified. Calibration artifact must come from calibration split.
    """
    if calibration['source_split'] != 'calibration':
        raise ValueError('Allowance must be fitted on calibration trajectories only')
    rule = calibration['rules'].get(phase, calibration['rules']['pooled'])
    jac = action_jacobian(predict, a0)
    values, vectors = eigensystem(jac)
    basis = vectors[:, :low_rank]
    allowed = np.zeros((basis.shape[1], 2))
    with torch.no_grad():
        nominal = predict(a0)
        for col in range(basis.shape[1]):
            d = torch.tensor(basis[:, col], dtype=a0.dtype, device=a0.device)
            for sign_id, sign in enumerate([-1, 1]):
                for radius in (calibration['radius_grid'] if radius_grid is None else radius_grid):
                    if radius > rule['radius']:
                        break
                    candidate = a0 + sign * radius * d
                    if valid_action is not None and not valid_action(candidate):
                        break
                    error = float(torch.linalg.vector_norm(predict(candidate) - nominal))
                    if error > rule['predicted_change']:
                        break
                    allowed[col, sign_id] = radius
    return ActionAllowance(basis, values, allowed, rule['predicted_change'], {
        'calibration_version': calibration['version'],
        'source_split': calibration['source_split'], 'phase': phase,
        'joint_candidate_check_required': True,
        'continuous_region_certified': False,
    })
