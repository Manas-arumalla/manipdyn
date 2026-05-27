"""Planners must return collision-free paths from start to goal."""

from __future__ import annotations

import numpy as np
import pytest

from manipdyn.planning import (
    PRM,
    RRT,
    CollisionChecker,
    InformedRRTStar,
    RRTConnect,
    RRTStar,
    shortcut_path,
)
from manipdyn.sim import World

Q_START = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
Q_GOAL = np.array([-1.5, -1.5708, 1.0, -1.5708, -1.5708, 0.0])


def _make(planner_cls, world):
    if planner_cls is PRM:
        return planner_cls(world, seed=0, n_samples=150, k_neighbors=12)
    if planner_cls in (RRTStar, InformedRRTStar):
        return planner_cls(world, seed=0, max_iter=800)
    return planner_cls(world, seed=0, max_iter=5000)


@pytest.mark.parametrize("planner_cls", [RRT, RRTConnect, RRTStar, InformedRRTStar, PRM])
def test_planner_returns_collision_free_path(planner_cls):
    world = World(scene="scene")  # scene with an obstacle box
    planner = _make(planner_cls, world)

    path = planner.plan(Q_START, Q_GOAL)
    assert path is not None, f"{planner_cls.name} found no path"
    assert path.shape[1] == 6
    assert np.allclose(path[0], Q_START) and np.allclose(path[-1], Q_GOAL)
    assert all(not planner.checker.in_collision(q) for q in path), "path has a colliding node"


def test_endpoints_in_collision_returns_none():
    world = World(scene="scene")
    planner = RRTConnect(world, seed=0)
    # A configuration jammed into the floor.
    q_bad = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    if planner.checker.in_collision(q_bad):
        assert planner.plan(Q_START, q_bad) is None


def test_shortcut_preserves_safety_and_endpoints():
    world = World(scene="scene")
    planner = RRT(world, seed=0, max_iter=5000)
    checker = CollisionChecker(world)

    path = planner.plan(Q_START, Q_GOAL)
    assert path is not None
    short = shortcut_path(path, checker, iterations=100, seed=0)

    assert len(short) <= len(path)
    assert np.allclose(short[0], Q_START) and np.allclose(short[-1], Q_GOAL)
    assert all(not checker.in_collision(q) for q in short)
