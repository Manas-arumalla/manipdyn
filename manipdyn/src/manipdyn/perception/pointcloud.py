"""Turn a depth image into a metric, world-frame point cloud and clean it up.

Everything here is pure NumPy + SciPy (already dependencies), so the perception
core needs no extra packages. The pipeline is:

    depth ──deproject──> camera-frame points ──R,t──> world points
          ──voxel_downsample──> ──remove_plane (RANSAC)──> ──largest_cluster──>

``deproject`` implements the pinhole model validated in the tests; the rest are
standard cloud-cleaning steps used to isolate an object resting on a table.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def deproject(
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    extrinsics: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray | None = None,
    stride: int = 1,
    max_depth: float | None = None,
) -> np.ndarray:
    """Deproject a depth image to an ``(N, 3)`` world-frame point cloud.

    Parameters
    ----------
    depth:
        ``(H, W)`` metric depth (metres), as returned by :meth:`Camera.depth`.
    intrinsics:
        ``(fx, fy, cx, cy)`` from :attr:`Camera.intrinsics`.
    extrinsics:
        ``(R, t)`` camera-to-world pose from :attr:`Camera.extrinsics`.
    mask:
        Optional ``(H, W)`` bool array; only ``True`` pixels are kept.
    stride:
        Subsample step over pixels (``1`` keeps all).
    max_depth:
        Optional far cutoff (metres) to drop background/sky pixels.
    """
    fx, fy, cx, cy = intrinsics
    R, t = extrinsics
    h, w = depth.shape
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth[vv, uu]

    keep = np.isfinite(z) & (z > 1e-4)
    if max_depth is not None:
        keep &= z < max_depth
    if mask is not None:
        keep &= mask[vv, uu]

    uu, vv, z = uu[keep], vv[keep], z[keep]
    x_cam = (uu - cx) / fx * z
    y_cam = -(vv - cy) / fy * z
    pts_cam = np.stack([x_cam, y_cam, -z], axis=-1)
    return pts_cam @ R.T + t


def voxel_downsample(points: np.ndarray, voxel: float = 0.004) -> np.ndarray:
    """Keep one point per ``voxel``-sized cube (order-stable). No-op if empty."""
    if len(points) == 0 or voxel <= 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def remove_plane(
    points: np.ndarray,
    thresh: float = 0.006,
    iters: int = 120,
    seed: int = 0,
) -> np.ndarray:
    """Drop the dominant plane (table/floor) via RANSAC; return the rest.

    Returns the input unchanged if fewer than 3 points. Deterministic for a
    fixed ``seed``.
    """
    n = len(points)
    if n < 3:
        return points
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = 0
    for _ in range(iters):
        i, j, k = rng.choice(n, 3, replace=False)
        normal = np.cross(points[j] - points[i], points[k] - points[i])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        dist = np.abs((points - points[i]) @ normal)
        inliers = dist < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None:
        return points
    return points[~best_inliers]


def largest_cluster(points: np.ndarray, eps: float = 0.012, min_size: int = 8) -> np.ndarray:
    """Return the biggest Euclidean cluster (single-link within ``eps``).

    Returns the input unchanged if it is smaller than ``min_size`` (nothing to
    separate) or empty.
    """
    n = len(points)
    if n < min_size:
        return points
    tree = cKDTree(points)
    label = np.full(n, -1, dtype=int)
    current = 0
    for start in range(n):
        if label[start] != -1:
            continue
        label[start] = current
        stack = [start]
        while stack:
            p = stack.pop()
            for q in tree.query_ball_point(points[p], eps):
                if label[q] == -1:
                    label[q] = current
                    stack.append(q)
        current += 1
    counts = np.bincount(label)
    return points[label == counts.argmax()]
