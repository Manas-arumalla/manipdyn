"""Reproducible benchmark scenarios.

* ``reach_targets`` — a fixed set of reachable goals (joint config + its
  forward-kinematics Cartesian position), so every controller is asked to reach
  the *same* poses.
* ``planner_query`` — a start/goal pair in an obstacle scene for planners.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manipdyn.control.base import Target
from manipdyn.sim import World

# Fixed, reachable, collision-free joint goals (chosen once for reproducibility).
_REACH_JOINT_GOALS = (
    (1.0, -1.1, 1.2, -1.6, -1.4, 0.4),
    (-0.8, -1.0, 1.0, -1.5, -1.5, 0.0),
    (0.6, -1.4, 1.4, -1.2, -1.6, 0.5),
)


def reach_targets(scene: str = "scene_base") -> list[Target]:
    """Targets carrying both the joint goal ``q`` and its Cartesian goal ``x``."""
    world = World(scene=scene)
    targets = []
    for q in _REACH_JOINT_GOALS:
        q = np.array(q, dtype=float)
        world.set_arm_qpos(q)
        world.forward()
        targets.append(Target(q=q.copy(), x=world.ee_pos.copy()))
    return targets


@dataclass
class PlannerQuery:
    scene: str
    q_start: np.ndarray
    q_goal: np.ndarray


def planner_query() -> PlannerQuery:
    """A start/goal pair whose straight-line joint motion is *blocked* by the
    pillar in ``scene_obstacle`` (the direct interpolation collides through the
    middle of the motion), so planners must actually find a detour."""
    return PlannerQuery(
        scene="scene_obstacle",
        q_start=np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0]),
        q_goal=np.array([-1.4, -1.2, 1.4, -1.7, -1.57, 0.0]),
    )
