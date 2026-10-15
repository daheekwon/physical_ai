import numpy as np


def rotation_error(a, b):
    return np.abs(np.arctan2(np.sin(np.asarray(a) - b), np.cos(np.asarray(a) - b)))


def outcome_metrics(candidate, nominal, position_scale=120.0):
    pos = np.linalg.norm(candidate["states"][:, 2:4] - nominal["states"][:, 2:4], axis=1)
    rot = rotation_error(candidate["states"][:, 4], nominal["states"][:, 4])
    return {
        "position_error": float(pos[-1]), "rotation_error": float(rot[-1]),
        "position_path_mean": float(pos.mean()), "position_path_max": float(pos.max()),
        "rotation_path_mean": float(rot.mean()), "rotation_path_max": float(rot.max()),
        "object_error": float(pos[-1] / position_scale + rot[-1] / np.pi),
        "task_score_difference": float(candidate["task_score"] - nominal["task_score"]),
        "official_state_distance_difference": float(candidate["official_state_distance"] - nominal["official_state_distance"]),
        "candidate_task_success": bool(candidate["task_success"]),
        "candidate_contact": bool(np.any(candidate["contacts"])),
    }


def preservation_mask(frame, tolerances):
    return ((frame.position_error <= tolerances["position"])
            & (frame.rotation_error <= tolerances["rotation"])
            & (frame.task_score_difference >= -tolerances["task_score_loss"]))


def cluster_bootstrap(values, clusters, rng, n_boot=2000):
    """CI for mean, resampling trajectories with all their dependent rows."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    valid = np.isfinite(values)
    values, clusters = values[valid], clusters[valid]
    ids = np.unique(clusters)
    if not len(ids):
        return [None, None]
    groups = [values[clusters == k] for k in ids]
    sums = np.array([v.sum() for v in groups])
    counts = np.array([len(v) for v in groups])
    draws = rng.integers(len(ids), size=(n_boot, len(ids)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return np.quantile(means, [0.025, 0.975]).tolist()
