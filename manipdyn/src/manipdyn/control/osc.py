"""Operational-Space Control (Khatib).

Decouples end-effector dynamics with the task-space inertia matrix and uses the
dynamically-consistent null space for a secondary posture task:

    Lambda = (J M^-1 J^T)^-1                       # task-space inertia
    F      = Lambda (Kp e + Kd e_dot)              # decoupled task force
    tau    = J^T F + N tau_posture  [+ gravity comp]
    N      = I - J^T (M^-1 J^T Lambda)^T           # null-space projector

Unlike Jacobian-transpose impedance, OSC accounts for the manipulator's
configuration-dependent inertia, giving consistent task-space stiffness.

The task is position-only (``J`` is the 3-row position Jacobian) unless the
:class:`Target` carries a desired orientation ``R``, in which case the task
becomes the full 6-DOF pose: ``J`` stacks the position and rotation Jacobians
and the orientation error is the standard column-cross-product residual
``e_o = 1/2 * sum_i (r_i x r_i^des)``.
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
        kp_rot: float | None = None,
        kd_rot: float | None = None,
    ):
        super().__init__(world)
        self.kp = kp
        # Critically damped by default.
        self.kd = kd if kd is not None else 2.0 * np.sqrt(kp)
        self.null_kp = null_kp
        self.null_kd = null_kd
        self.damping = damping
        self.gravity_comp = gravity_comp
        # Orientation gains (used only when a target orientation is supplied).
        self.kp_rot = kp if kp_rot is None else kp_rot
        self.kd_rot = 2.0 * np.sqrt(self.kp_rot) if kd_rot is None else kd_rot
        self.q_home = world.home_qpos_arm.copy()

    def compute(self, target: Target) -> np.ndarray:
        world = self.world
        jp, jr = world.ee_jacobian()  # each (3, n_arm)
        M = world.mass_matrix()  # (n_arm, n_arm)
        v = world.qvel_arm

        # Position task (always present).
        xdot = jp @ v
        target_xdot = np.zeros(3) if target.xdot is None else target.xdot
        acc = self.kp * (target.x - world.ee_pos) + self.kd * (target_xdot - xdot)
        jac = jp

        # Optional orientation task -> full 6-DOF operational-space control.
        if target.R is not None:
            r_cur, r_des = world.ee_rot(), np.asarray(target.R, dtype=float)
            e_o = 0.5 * sum(np.cross(r_cur[:, i], r_des[:, i]) for i in range(3))
            acc = np.concatenate([acc, self.kp_rot * e_o - self.kd_rot * (jr @ v)])
            jac = np.vstack([jp, jr])

        # Task-space inertia Lambda = (J M^-1 J^T)^-1, with light damping.
        minv_jt = np.linalg.solve(M, jac.T)  # M^-1 J^T
        lam = np.linalg.inv(jac @ minv_jt + self.damping * np.eye(jac.shape[0]))
        tau_task = jac.T @ (lam @ acc)

        # Dynamically-consistent null-space posture task (return toward home).
        j_bar = minv_jt @ lam  # M^-1 J^T Lambda
        null_proj = np.eye(self.n_arm) - jac.T @ j_bar.T
        tau_posture = self.null_kp * (self.q_home - world.qpos_arm) - self.null_kd * v
        tau = tau_task + null_proj @ tau_posture

        if self.gravity_comp:
            tau = tau + world.bias_force()
        return tau
