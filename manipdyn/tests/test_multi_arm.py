"""Multi-robot: two UR5e arms in one shared simulation."""

from __future__ import annotations

import mujoco
import numpy as np

from manipdyn.control import ComputedTorqueController, Target
from manipdyn.models.procedural import build_two_arm_scene, two_arm_worlds


def test_two_arm_scene_builds_and_discovers_both():
    model = build_two_arm_scene()
    left, right = two_arm_worlds(model)

    assert left.n_arm == 6 and right.n_arm == 6
    assert left.ee_site_name == "left_pinch"
    assert right.ee_site_name == "right_pinch"
    # Two arms (16 actuated joints incl. grippers) + one free object.
    assert model.nu == 16
    assert model.neq == 2  # left_grasp + right_grasp welds
    # The two arms are driven by disjoint actuators.
    assert not set(left.arm_actuator_ids) & set(right.arm_actuator_ids)


def test_arms_are_independently_controlled_in_one_sim():
    model = build_two_arm_scene()
    left, right = two_arm_worlds(model)

    goal_l = np.array([0.6, -1.1, 1.2, -1.6, -1.4, 0.4])
    goal_r = np.array([-0.6, -1.0, 1.0, -1.5, -1.5, 0.0])
    cl = ComputedTorqueController(left, kp=400, kd=40)
    cr = ComputedTorqueController(right, kp=400, kd=40)

    for _ in range(1500):
        left.apply_arm_torque(cl.compute(Target(q=goal_l)))
        right.apply_arm_torque(cr.compute(Target(q=goal_r)))
        mujoco.mj_step(model, left.data)  # one shared step

    assert np.linalg.norm(goal_l - left.qpos_arm) < 0.02
    assert np.linalg.norm(goal_r - right.qpos_arm) < 0.02
