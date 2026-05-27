"""Path post-processing: collision-aware shortcutting and B-spline smoothing.

Raw sampling-based paths are jagged. Shortcutting greedily replaces detours
with straight segments whenever the shortcut is collision-free (the workhorse);
B-spline fitting then optionally rounds the corners for a visually smooth,
higher-resolution trajectory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manipdyn.planning.collision import CollisionChecker


def shortcut_path(
    path: np.ndarray,
    checker: CollisionChecker,
    iterations: int = 200,
    resolution: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Randomized collision-aware shortcutting.

    Repeatedly picks two points on the path and, if the straight segment
    between them is collision-free, splices out everything in between.
    """
    path = [np.asarray(p, dtype=float) for p in path]
    if len(path) <= 2:
        return np.array(path)
    rng = np.random.default_rng(seed)

    for _ in range(iterations):
        if len(path) <= 2:
            break
        i = rng.integers(0, len(path) - 1)
        j = rng.integers(i + 1, len(path))
        if j - i <= 1:
            continue
        if not checker.edge_in_collision(path[i], path[j], resolution):
            path = path[: i + 1] + path[j:]
    return np.array(path)


def smooth_bspline(path: np.ndarray, n_points: int = 0, smoothing: float = 0.0) -> np.ndarray:
    """Fit a B-spline through the path and resample it.

    Returns the input unchanged if there are too few points or SciPy fails.
    """
    import scipy.interpolate

    path = np.asarray(path, dtype=float)
    if len(path) < 4:
        return path

    # Drop near-duplicate consecutive waypoints (splprep requires distinct knots).
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-4])
    path = path[keep]
    if len(path) < 4:
        return path

    try:
        k = min(3, len(path) - 1)
        tck, _ = scipy.interpolate.splprep(path.T, s=smoothing, k=k)
        u_fine = np.linspace(0.0, 1.0, n_points or len(path) * 5)
        return np.array(scipy.interpolate.splev(u_fine, tck)).T
    except Exception:
        return path
