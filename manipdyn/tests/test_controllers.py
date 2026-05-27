"""Each controller in the zoo must actually drive the arm to its target."""

from __future__ import annotations

import numpy as np
import pytest

from manipdyn.control import (
    ComputedTorqueController,
    ILQRController,
    ImpedanceController,
    LQRController,
    MPPIController,
    OSCController,
    PIDController,
    Target,
    TSIDController,
)
from manipdyn.kinematics import IKSolver
from manipdyn.sim import World

Q_TARGET = np.array([1.0, -1.1, 1.2, -1.6, -1.4, 0.4])


def _run(world: World, controller, target: Target, seconds: float) -> float:
    controller.reset()
    for _ in range(int(seconds / world.timestep)):
        world.step(controller.compute(target))
    return float(np.linalg.norm(Q_TARGET - world.qpos_arm))


@pytest.mark.parametrize(
    "make, tol",
    [
        (lambda w: PIDController(w), 0.05),
        (lambda w: ComputedTorqueController(w, kp=400, kd=40), 0.02),
        (lambda w: LQRController(w), 0.05),
    ],
)
def test_joint_controllers_settle(make, tol):
    world = World(scene="scene_base")
    world.reset(world.home_qpos_arm)
    err = _run(world, make(world), Target(q=Q_TARGET), seconds=5.0)
    assert err < tol, f"final joint error {err:.4f} exceeded {tol}"


def test_ilqr_controller_settles():
    world = World(scene="scene_base")
    world.reset(world.home_qpos_arm)
    controller = ILQRController(world, horizon=80, control_dt=0.02)
    err = _run(world, controller, Target(q=Q_TARGET), seconds=3.0)
    assert err < 0.05, f"iLQR final joint error {err:.4f} too large"


@pytest.mark.parametrize("make", [ImpedanceController, OSCController, TSIDController])
def test_cartesian_controllers_reach(make):
    world = World(scene="scene_base")
    world.reset(world.home_qpos_arm)

    # Cartesian goal = FK of a known reachable config.
    world.set_arm_qpos(Q_TARGET)
    world.forward()
    x_goal = world.ee_pos.copy()
    world.reset(world.home_qpos_arm)

    controller = make(world)
    controller.reset()
    for _ in range(int(6.0 / world.timestep)):
        world.step(controller.compute(Target(x=x_goal)))

    pos_err = float(np.linalg.norm(x_goal - world.ee_pos))
    assert pos_err < 0.03, f"EE position error {pos_err * 1000:.1f} mm too large"


def test_mppi_reduces_error():
    world = World(scene="scene_base")
    world.reset(world.home_qpos_arm)
    # Small, fast settings to keep the test quick.
    mppi = MPPIController(world, horizon=15, n_samples=20, seed=0)
    start = float(np.linalg.norm(Q_TARGET - world.qpos_arm))
    err = _run(world, mppi, Target(q=Q_TARGET), seconds=1.5)
    assert err < start, f"MPPI did not reduce error ({start:.3f} -> {err:.3f})"


def test_ik_converges_to_reachable_pose():
    world = World(scene="scene_base")
    world.set_arm_qpos(Q_TARGET)
    world.forward()
    x_goal = world.ee_pos.copy()
    world.reset(world.home_qpos_arm)

    result = IKSolver(world).solve(x_goal, q_guess=world.home_qpos_arm)
    assert result.success, f"IK failed: error {result.error:.4f} after {result.iterations} iters"

    # Verify the returned configuration actually reaches the goal.
    world.set_arm_qpos(result.q)
    world.forward()
    assert np.linalg.norm(world.ee_pos - x_goal) < 1e-2
