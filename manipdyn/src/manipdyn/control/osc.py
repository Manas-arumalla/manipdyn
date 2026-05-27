"""Operational-Space Control (Khatib).

Decouples end-effector dynamics with the task-space inertia matrix and uses the
dynamically-consistent null space for a secondary posture task:

    Lambda = (J M^-1 J^T)^-1                       # task-space inertia
    F      = Lambda (Kp e + Kd e_dot)              # decoupled task force
    tau    = J^T F + N tau_posture  [+ gravity comp]
    N      = I - J^T (M^-1 J^T Lambda)^T           # null-space projector

Unlike Jacobian-transpose impedance, OSC accounts for the manipulator's
configuration-dependent inertia, giving consistent task-space stiffness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manipdyn.control.base import Controller, Target

if TYPE_CHECKING:
    from manipdyn.sim.world import World

Gain = float | np.ndarray


class OSCController(Controller):
    name = "osc"
    target_space = "cartesian"

    def __init__(
        self,
        world: World,
        kp: float = 150.0,
        kd: float | None = None,
        null_kp: float = 10.0,
        null_kd: float = 5.0,
        damping: float = 1e-4,
        gravity_comp: bool = True,
    ):
        super().__init__(world)
        self.kp = kp
        # Critically damped by default.
        self.kd = kd if kd is not None else 2.0 * np.sqrt(kp)
        self.null_kp = null_kp
        self.null_kd = null_kd
        self.damping = damping
        self.gravity_comp = gravity_comp
        self.q_home = world.home_qpos_arm.copy()

    def compute(self, target: Target) -> np.ndarray:
        world = self.world
        jp, _ = world.ee_jacobian()  # (3, n_arm)
        M = world.mass_matrix()  # (n_arm, n_arm)
        x = world.ee_pos
        v = world.qvel_arm
        xdot = jp @ v
        target_xdot = np.zeros(3) if target.xdot is None else target.xdot

        # Task-space inertia Lambda = (J M^-1 J^T)^-1, with light damping.
        minv_jt = np.linalg.solve(M, jp.T)  # M^-1 J^T
        task_inertia = jp @ minv_jt
        lam = np.linalg.inv(task_inertia + self.damping * np.eye(3))

        force = lam @ (self.kp * (target.x - x) + self.kd * (target_xdot - xdot))
        tau_task = jp.T @ force

        # Dynamically-consistent null-space posture task (return toward home).
        j_bar = minv_jt @ lam  # M^-1 J^T Lambda
        null_proj = np.eye(self.n_arm) - jp.T @ j_bar.T
        tau_posture = self.null_kp * (self.q_home - world.qpos_arm) - self.null_kd * v
        tau = tau_task + null_proj @ tau_posture

        if self.gravity_comp:
            tau = tau + world.bias_force()
        return tau
