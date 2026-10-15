"""Action-input derivatives and equal-norm perturbations (no weight updates)."""
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def action_jacobian(predict, action):
    # The math kernel supports the forward-mode derivatives used by jacfwd.
    with sdpa_kernel(SDPBackend.MATH):
        jac = torch.func.jacfwd(predict)(action)
    return jac.reshape(-1, action.numel()).detach()


def eigensystem(jac):
    matrix = jac.double().T @ jac.double()
    values, vectors = torch.linalg.eigh(matrix)
    return values.clamp_min(0).cpu().numpy(), vectors.cpu().numpy()


def make_directions(vectors, rng, n_random=5, n_mix=5, low_rank=3):
    result = {"low": vectors[:, 0], "high": vectors[:, -1]}
    for k in range(n_random):
        v = rng.normal(size=vectors.shape[0])
        result[f"random_{k}"] = v / np.linalg.norm(v)
    for k in range(1, min(low_rank, vectors.shape[1])):
        result[f"low_basis_{k}"] = vectors[:, k]
    for k in range(n_mix):
        c = rng.normal(size=min(low_rank, vectors.shape[1]))
        result[f"low_mix_{k}"] = vectors[:, :len(c)] @ (c / np.linalg.norm(c))
    return result


def symmetric_bound_radius(action, directions, low, high):
    """Largest common radius permitting BOTH signs without clipping."""
    margin = np.minimum(action - low, high - action).reshape(-1)
    if np.min(margin) < -1e-6:
        return 0.0
    result = np.inf
    for direction in directions:
        d = np.abs(np.asarray(direction).reshape(-1))
        nz = d > 1e-12
        result = min(result, float(np.min(np.maximum(margin[nz], 0) / d[nz])))
    return result
