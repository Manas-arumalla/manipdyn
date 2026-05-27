"""Task-Space Inverse Dynamics as a constrained QP (TSID).

The principled, constraint-aware cousin of OSC and the template for modern
whole-body control. Each tick solves for joint accelerations ``a`` that best
realize a desired end-effector acceleration while *strictly* respecting the
arm's torque limits:

    min_a   w_task ||J a - a*_task||^2 + w_post ||a - a_posture||^2 + w_reg ||a||^2
    s.t.    -tau_max <= M(q) a + h(q,v) <= tau_max         (torque limits)

with ``a*_task = Kp (x* - x) + Kd (xdot* - xdot)``. The realized torque is
``tau = M a + h``. The QP is solved with OSQP; if it ever fails we fall back to
a damped-least-squares solution so the loop never stalls.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import osqp
from scipy import sparse

from manipdyn.control.base import Controller, Target

if TYPE_CHECKING:
    from manipdyn.sim.world import World


class TSIDController(Controller):
    name = "tsid"
    target_space = "cartesian"

    def __init__(
        self,
        world: World,
        kp: float = 150.0,
        kd: float | None = None,
        w_task: float = 1.0,
        w_posture: float = 1e-2,
        w_reg: float = 1e-4,
        posture_kp: float = 10.0,
        posture_kd: float = 5.0,
    ):
        super().__init__(world)
        self.kp = kp
        self.kd = kd if kd is not None else 2.0 * np.sqrt(kp)
        self.w_task = w_task
        self.w_posture = w_posture
        self.w_reg = w_reg
        self.posture_kp = posture_kp
        self.posture_kd = posture_kd
        self.tau_max = world.torque_limits
        self.q_home = world.home_qpos_arm.copy()

    def compute(self, target: Target) -> np.ndarray:
        world = self.world
        jp, _ = world.ee_jacobian()  # (3, n_arm)
        M = world.mass_matrix()  # (n_arm, n_arm)
        h = world.bias_force()
        x = world.ee_pos
        v = world.qvel_arm
        xdot = jp @ v
        target_xdot = np.zeros(3) if target.xdot is None else target.xdot

        a_task = self.kp * (target.x - x) + self.kd * (target_xdot - xdot)
        a_post = self.posture_kp * (self.q_home - world.qpos_arm) - self.posture_kd * v

        # QP in joint acceleration `a`.
        P = (
            self.w_task * (jp.T @ jp)
            + self.w_posture * np.eye(self.n_arm)
            + self.w_reg * np.eye(self.n_arm)
        )
        g = -self.w_task * (jp.T @ a_task) - self.w_posture * a_post

        # Torque-limit constraints: -tau_max <= M a + h <= tau_max.
        lo = -self.tau_max - h
        hi = self.tau_max - h

        a = self._solve_qp(P, g, M, lo, hi)
        if a is None:  # fallback: damped least squares + nullspace posture
            jp_pinv = jp.T @ np.linalg.inv(jp @ jp.T + 1e-4 * np.eye(3))
            a = jp_pinv @ a_task + (np.eye(self.n_arm) - jp_pinv @ jp) @ a_post

        tau = M @ a + h
        return np.clip(tau, -self.tau_max, self.tau_max)

    def _solve_qp(self, P, g, A, lo, hi) -> np.ndarray | None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # quiet OSQP's deprecation notices
                prob = osqp.OSQP()
                prob.setup(
                    P=sparse.csc_matrix(P),
                    q=g,
                    A=sparse.csc_matrix(A),
                    l=lo,
                    u=hi,
                    verbose=False,
                )
                res = prob.solve()
            if res.x is not None and res.x[0] is not None and "solved" in str(res.info.status):
                return np.asarray(res.x, dtype=float)
        except Exception:
            pass
        return None
