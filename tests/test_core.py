import numpy as np
import torch
from zj_wam.sensitivity import action_jacobian, eigensystem, symmetric_bound_radius
from zj_wam.metrics import rotation_error, cluster_bootstrap


def test_known_rotated_sensitivity():
    q = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
    matrix = torch.diag(torch.tensor([1., 9.], dtype=torch.float64)) @ q.T
    jac = action_jacobian(lambda a: matrix @ a, torch.zeros(2, dtype=torch.float64))
    values, vectors = eigensystem(jac)
    np.testing.assert_allclose(values, [1, 81])
    assert abs(vectors[:, 0] @ q[:, 0].numpy()) > 1 - 1e-10


def test_common_radius_handles_both_signs_without_clipping():
    a = np.array([0.95, 0.0])
    directions = [np.array([1., 0.]), np.array([0.6, 0.8])]
    radius = symmetric_bound_radius(a, directions, np.array([-1., -1.]), np.array([1., 1.]))
    assert abs(radius - 0.05) < 1e-10
    for d in directions:
        for sign in [-1, 1]:
            assert np.max(np.abs(a + sign * radius * d)) <= 1 + 1e-10


def test_angles_across_wrap():
    assert abs(rotation_error(np.pi - 0.01, -np.pi + 0.01) - 0.02) < 1e-10


def test_bootstrap_keeps_correlated_rows_together():
    # Replicating every observation within each trajectory must not narrow its CI.
    ci = cluster_bootstrap([0, 1], [0, 1], np.random.default_rng(7), 1000)
    repeated = cluster_bootstrap([0] * 100 + [1] * 100, [0] * 100 + [1] * 100,
                                 np.random.default_rng(7), 1000)
    assert ci == repeated == [0., 1.]


def test_allowance_finds_signed_finite_limit_and_rejects_test_calibration():
    import pytest
    from zj_wam.allowance import estimate_action_allowance
    # A quadratic response is locally insensitive but finite changes are visible.
    def predict(a):
        return torch.stack([a[0] ** 2, 2 * a[1]])
    calibration = {'source_split':'calibration','version':'unit-test',
                   'rules':{'pooled':{'radius':1.,'predicted_change':.1}},
                   'radius_grid':[.1,.2,.4,.8]}
    result = estimate_action_allowance(predict, torch.zeros(2), calibration, low_rank=1)
    np.testing.assert_allclose(result.allowed_correction_scale, [[.2,.2]])
    assert not result.diagnostics['continuous_region_certified']
    with pytest.raises(ValueError):
        estimate_action_allowance(predict,torch.zeros(2),dict(calibration,source_split='test'))
