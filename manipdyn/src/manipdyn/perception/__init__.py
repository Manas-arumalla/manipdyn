"""Perception: simulated RGB-D vision for the UR5e lab.

Render colour/depth/segmentation from a scene camera, deproject depth into a
metric point cloud, and estimate a graspable object pose — so the arm can be
driven from what it *sees* rather than from privileged simulator state.

    from manipdyn.perception import Camera, sense_object_pose
    from manipdyn.sim import World

    world = World(scene="scene_pick")
    cam = Camera(world, "overhead")
    est = sense_object_pose(cam)      # ObjectEstimate: top_xy, top_z, dims, ...
"""

from __future__ import annotations

from manipdyn.perception.camera import Camera
from manipdyn.perception.estimate import (
    ObjectEstimate,
    estimate_object_pose,
    movable_bodies,
    object_geom_ids,
    segment_mask,
    select_object,
    sense_object_pose,
    sense_objects,
)
from manipdyn.perception.pointcloud import (
    cluster_all,
    deproject,
    largest_cluster,
    remove_plane,
    voxel_downsample,
)

__all__ = [
    "Camera",
    "ObjectEstimate",
    "estimate_object_pose",
    "sense_object_pose",
    "sense_objects",
    "select_object",
    "movable_bodies",
    "object_geom_ids",
    "segment_mask",
    "deproject",
    "voxel_downsample",
    "remove_plane",
    "largest_cluster",
    "cluster_all",
]
