"""Estimate an object's pose from a point cloud, and a one-call sensing helper.

``estimate_object_pose`` turns a cleaned cloud into a graspable estimate: a
bounding-box centre, the height of the top surface, the footprint extents, and
PCA axes for orientation. For a partial top-down view the **axis-aligned
bounding-box centre** is a robust XY estimate — its extremes are set by the
object's edges, so it is not pulled toward the camera the way a raw centroid is
under oblique foreshortening.

``sense_object_pose`` wires a :class:`~manipdyn.perception.camera.Camera`
through the cloud pipeline. Two honest modes:

* ``segmentation=True`` (default) uses MuJoCo's ground-truth segmentation buffer
  to select the object's pixels — a stand-in for a perfect instance segmenter.
  It reads *which pixels are the object*, never the object's pose, so the grasp
  target still comes from geometry, not from ``data.xpos``.
* ``segmentation=False`` is fully sensor-only: deproject everything, drop the
  table/floor plane, and keep the largest cluster inside an optional workspace
  box (scene knowledge, not object pose).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from manipdyn.perception.pointcloud import (
    deproject,
    largest_cluster,
    remove_plane,
    voxel_downsample,
)

if TYPE_CHECKING:
    from manipdyn.perception.camera import Camera
    from manipdyn.sim.world import World


@dataclass
class ObjectEstimate:
    """A perceived object pose, everything needed for a top-down grasp."""

    center: np.ndarray  # (3,) axis-aligned bounding-box centre
    top_xy: np.ndarray  # (2,) top-face centre — the top-down grasp target
    top_z: float  # height of the top surface (m)
    dims: np.ndarray  # (3,) bounding-box extents (m)
    R: np.ndarray  # (3,3) PCA axes (columns), largest-variance first
    n_points: int  # number of points the estimate is built from


def estimate_object_pose(points: np.ndarray) -> ObjectEstimate:
    """Estimate an :class:`ObjectEstimate` from an ``(N, 3)`` object cloud."""
    p = np.asarray(points, dtype=float)
    if len(p) < 4:
        raise ValueError(f"need >= 4 points to estimate a pose, got {len(p)}")
    lo, hi = p.min(axis=0), p.max(axis=0)
    center = (lo + hi) / 2.0
    dims = hi - lo
    top_z = float(hi[2])
    top_xy = center[:2].copy()

    # PCA of the cloud for a coarse orientation frame.
    c = p.mean(axis=0)
    cov = np.cov((p - c).T)
    evals, evecs = np.linalg.eigh(cov)
    R = evecs[:, np.argsort(evals)[::-1]]
    return ObjectEstimate(
        center=center, top_xy=top_xy, top_z=top_z, dims=dims, R=R, n_points=len(p)
    )


def object_geom_ids(world: World, body: str = "object") -> list[int]:
    """Geom ids belonging to ``body`` (used to segment it from the scene)."""
    m = world.model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid == -1:
        raise ValueError(f"No body named {body!r} in scene.")
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid]


def segment_mask(seg: np.ndarray, geom_ids: list[int]) -> np.ndarray:
    """Boolean pixel mask selecting the given geom ids from a seg image."""
    ids = np.asarray(geom_ids)
    return np.isin(seg[..., 0], ids) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)


def sense_object_pose(
    camera: Camera,
    *,
    segmentation: bool = True,
    object_body: str = "object",
    workspace: tuple[float, float, float, float, float, float] | None = None,
    voxel: float = 0.003,
) -> ObjectEstimate:
    """Perceive an object with ``camera`` and return its pose estimate.

    ``workspace`` (used only when ``segmentation=False``) is an
    ``(xlo, xhi, ylo, yhi, zlo, zhi)`` world-frame crop box.
    """
    world = camera.world
    depth = camera.depth()
    intr, extr = camera.intrinsics, camera.extrinsics

    if segmentation:
        seg = camera.segmentation()
        mask = segment_mask(seg, object_geom_ids(world, object_body))
        pts = deproject(depth, intr, extr, mask=mask)
    else:
        pts = deproject(depth, intr, extr, max_depth=None)
        if workspace is not None:
            xlo, xhi, ylo, yhi, zlo, zhi = workspace
            inside = (
                (pts[:, 0] >= xlo)
                & (pts[:, 0] <= xhi)
                & (pts[:, 1] >= ylo)
                & (pts[:, 1] <= yhi)
                & (pts[:, 2] >= zlo)
                & (pts[:, 2] <= zhi)
            )
            pts = pts[inside]
        pts = remove_plane(pts)
        pts = largest_cluster(pts)

    pts = voxel_downsample(pts, voxel)
    if len(pts) < 5:
        raise RuntimeError(
            "perception found too few object points — is the object in view and unoccluded?"
        )
    return estimate_object_pose(pts)
