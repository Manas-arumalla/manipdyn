"""Procedural (parametric) scene generation via MjSpec."""

from __future__ import annotations

import numpy as np
import pytest

from manipdyn.models.procedural import build_clutter_scene
from manipdyn.perception import movable_bodies
from manipdyn.sim import World


def test_build_has_arm_and_n_cubes():
    model = build_clutter_scene(n_cubes=4, seed=1)
    world = World(model=model, ee_site="pinch")
    assert world.n_arm == 6  # arm discovery works on a procedural model
    assert movable_bodies(world) == [f"cube_{i}" for i in range(4)]


def test_build_is_deterministic():
    a = World(model=build_clutter_scene(n_cubes=5, seed=7))
    b = World(model=build_clutter_scene(n_cubes=5, seed=7))
    assert np.allclose(a.data.xpos, b.data.xpos)


def test_different_seeds_differ():
    a = World(model=build_clutter_scene(n_cubes=4, seed=1))
    b = World(model=build_clutter_scene(n_cubes=4, seed=2))
    assert not np.allclose(a.data.xpos, b.data.xpos)


def test_too_many_cubes_raises():
    with pytest.raises(RuntimeError, match="could not place"):
        build_clutter_scene(n_cubes=40, seed=0)


def test_perception_finds_procedural_cubes():
    from manipdyn.perception import sense_objects

    model = build_clutter_scene(n_cubes=4, seed=3)
    world = World(model=model, ee_site="pinch")
    look = world.home_qpos_arm.copy()
    look[1] -= 0.5  # park the arm clear of the table
    world.reset(look)
    world.forward()
    try:
        from manipdyn.perception import Camera

        cam = Camera(world, "overhead")
    except Exception as exc:  # no GL backend on this runner
        pytest.skip(f"offscreen GL unavailable: {exc}")
    with cam:
        objs = sense_objects(cam, segmentation=True)
    assert len(objs) == 4
