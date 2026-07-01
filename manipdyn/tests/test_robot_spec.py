"""RobotSpec parity tests: the spec-based World must behave exactly as before.

These pin down that introducing ``RobotSpec`` did not change the default UR5e
path — same joints, torque limits, home, and end-effector — and that the spec
is actually what drives discovery (not a leftover hardcoding).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from manipdyn.sim import UR5E, RobotSpec, World
from manipdyn.sim.world import ARM_JOINT_NAMES


def test_default_is_ur5e():
    world = World(scene="scene_base")
    assert world.robot is UR5E
    assert world.robot.arm_joint_names == ARM_JOINT_NAMES
    # The home posture is exactly the previously hardcoded configuration.
    assert np.allclose(world.home_qpos_arm, [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def test_explicit_spec_matches_default():
    """Passing an equal spec discovers an identical arm (spec drives discovery)."""
    default = World(scene="scene_base")
    explicit = World(scene="scene_base", robot=UR5E)

    assert np.array_equal(explicit.arm_joint_ids, default.arm_joint_ids)
    assert np.array_equal(explicit.arm_dof_adr, default.arm_dof_adr)
    assert np.array_equal(explicit.arm_actuator_ids, default.arm_actuator_ids)
    assert np.allclose(explicit.torque_limits, default.torque_limits)
    assert explicit.ee_site_name == default.ee_site_name
    assert np.allclose(explicit.ee_pos, default.ee_pos)
    assert explicit.n_arm == default.n_arm == 6


def test_spec_home_flows_through():
    """A spec's home posture is what World resets to."""
    tweaked = dataclasses.replace(UR5E, home_qpos=(0.1, -1.0, 1.0, -1.5, -1.5, 0.2))
    world = World(scene="scene_base", robot=tweaked)
    assert np.allclose(world.home_qpos_arm, tweaked.home_qpos)
    assert np.allclose(world.qpos_arm, tweaked.home_qpos)  # reset_home used it


def test_wrong_joint_names_raise():
    bad = RobotSpec(
        name="bogus",
        arm_joint_names=("nope_1", "nope_2"),
        ee_site_candidates=("attachment_site",),
        home_qpos=(0.0, 0.0),
    )
    with pytest.raises(RuntimeError, match="bogus"):
        World(scene="scene_base", robot=bad)
