"""Perception tests: RGB-D rendering, deprojection, and object-pose estimation.

These need an offscreen GL backend (as the render smoke test does); they skip
cleanly when one is unavailable.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from manipdyn.sim import World


def _world_with_cube(x=-0.47, y=-0.16):
    """scene_pick with the cube nudged to a spot the home gripper doesn't occlude."""
    world = World(scene="scene_pick", ee_site="pinch")
    jid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    adr = world.model.jnt_qposadr[jid]
    world.model.qpos0[adr : adr + 2] = [x, y]
    world.reset()
    world.forward()
    return world


def _camera(world, name="overhead", w=640, h=480):
    from manipdyn.perception import Camera

    try:
        return Camera(world, name, width=w, height=h)
    except Exception as exc:  # no GL backend on this runner
        pytest.skip(f"offscreen GL unavailable: {exc}")


def test_cameras_present_and_physics_unchanged():
    # Adding cameras must not add DOFs/actuators or disturb arm discovery.
    world = World(scene="scene_pick", ee_site="pinch")
    assert world.n_arm == 6
    for name in ("overhead", "wrist"):
        assert mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, name) != -1


def test_render_shapes_and_intrinsics():
    world = _world_with_cube()
    cam = _camera(world)
    with cam:
        rgb, depth, seg = cam.rgb(), cam.depth(), cam.segmentation()
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        assert depth.shape == (480, 640) and np.all(depth >= 0)
        assert seg.shape == (480, 640, 2)
        fx, fy, cx, cy = cam.intrinsics
        assert fx > 0 and fy > 0
        assert cx == pytest.approx(319.5) and cy == pytest.approx(239.5)


def test_segmentation_selects_object():
    from manipdyn.perception import object_geom_ids, segment_mask

    world = _world_with_cube()
    cam = _camera(world)
    with cam:
        mask = segment_mask(cam.segmentation(), object_geom_ids(world))
    assert mask.sum() > 20  # the cube occupies a healthy patch of pixels


def test_estimate_object_pose_matches_truth():
    from manipdyn.perception import sense_object_pose

    world = _world_with_cube()
    oid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "object")
    true = world.data.xpos[oid]
    cam = _camera(world)
    with cam:
        est = sense_object_pose(cam, segmentation=True)
    # XY within a centimetre, top surface within a few millimetres.
    assert np.linalg.norm(est.top_xy - true[:2]) < 0.010
    assert abs(est.top_z - (true[2] + 0.025)) < 0.006
    assert est.n_points >= 20


def _clutter_world():
    """scene_clutter with the arm parked clear of the table for a clean look."""
    world = World(scene="scene_clutter", ee_site="pinch")
    look = world.home_qpos_arm.copy()
    look[1] -= 0.5
    world.reset(look)
    world.forward()
    return world


def test_sense_objects_finds_all_cubes():
    from manipdyn.perception import sense_objects

    world = _clutter_world()
    cam = _camera(world)
    with cam:
        objs = sense_objects(cam, segmentation=True)

    labels = {o.label for o in objs}
    assert labels == {"cube_red", "cube_green", "cube_blue"}
    for o in objs:
        bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, o.label)
        true_xy = world.data.xpos[bid][:2]
        assert np.linalg.norm(o.top_xy - true_xy) < 0.010


def test_sensor_only_clustering_counts_objects():
    from manipdyn.perception import sense_objects

    world = _clutter_world()
    cam = _camera(world)
    with cam:
        objs = sense_objects(
            cam,
            segmentation=False,
            workspace=(-0.68, -0.30, -0.26, 0.02, 0.37, 0.60),
        )
    assert len(objs) == 3  # three cubes recovered without knowing the count


def test_select_object_by_label_and_proximity():
    from manipdyn.perception import select_object, sense_objects

    world = _clutter_world()
    cam = _camera(world)
    with cam:
        objs = sense_objects(cam, segmentation=True)

    assert select_object(objs, label="cube_green").label == "cube_green"
    blue = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "cube_blue")
    nearest = select_object(objs, near=world.data.xpos[blue][:2])
    assert nearest.label == "cube_blue"


def test_deproject_convention_recovers_a_known_point():
    """A single depth pixel deprojects back to a point on the cube's top face."""
    from manipdyn.perception import deproject, object_geom_ids, segment_mask

    world = _world_with_cube()
    oid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "object")
    true = world.data.xpos[oid]
    cam = _camera(world)
    with cam:
        mask = segment_mask(cam.segmentation(), object_geom_ids(world))
        pts = deproject(cam.depth(), cam.intrinsics, cam.extrinsics, mask=mask)
    assert len(pts) > 20
    # The highest points are the top face; their height matches the true top.
    top_z = pts[:, 2].max()
    assert abs(top_z - (true[2] + 0.025)) < 0.006
