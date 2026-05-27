"""Infinite-horizon Linear-Quadratic Regulator.

Linearizes the MuJoCo dynamics about the goal configuration, solves the
continuous-time algebraic Riccati equation (CARE), and applies the optimal
state feedback ``u = -K (x - x*)`` plus gravity compensation.

LQR is fundamentally a *regulator*: the gain ``K`` is optimal for the
linearization point. We therefore (re)linearize lazily whenever the commanded
target moves significantly, so the same object works both for fixed-goal
regulation (one CARE solve) and for slowly-varying references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve_continuous_are

from manipdyn.control.base import Controller, Target
from manipdyn.dynamics import linearize

if TYPE_CHECKING:
    from manipdyn.sim.world import World


class LQRController(Controller):
    name = "lqr"
    target_space = "joint"

    def __init__(
        self,
        world: World,
        q_pos: float = 1000.0,
        q_vel: float = 10.0,
        r: float = 1.0,
        relinearize_tol: float = 1e-2,
    ):
        super().__init__(world)
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r = r
        self.relinearize_tol = relinearize_tol
        self.arm_act = world.arm_actuator_ids
        self._K: np.ndarray | None = None
        self._lin_q: np.ndarray | None = None  # arm config of current linearization

    def reset(self) -> None:
        self._K = None
        self._lin_q = None

    def _solve_gain(self, target_q_arm: np.ndarray) -> None:
        world = self.world
        m = world.model
        nv, nu = m.nv, m.nu

        q_full = world.data.qpos.copy()
        q_full[world.arm_qpos_adr] = target_q_arm
        A, B = linearize(world, q_full)

        Q = np.zeros((2 * nv, 2 * nv))
        Q[:nv, :nv] = np.eye(nv) * self.q_pos
        Q[nv:, nv:] = np.eye(nv) * self.q_vel
        R = np.eye(nu) * self.r

        P = solve_continuous_are(A, B, Q, R)
        self._K = np.linalg.solve(R, B.T @ P)  # R^-1 B^T P
        self._lin_q = np.asarray(target_q_arm, dtype=float).copy()

    def compute(self, target: Target) -> np.ndarray:
        if self._K is None or np.linalg.norm(target.q - self._lin_q) > self.relinearize_tol:
            self._solve_gain(target.q)

        m = self.world.model
        x = np.concatenate([self.world.data.qpos[: m.nq], self.world.data.qvel[: m.nv]])
        x_target = np.concatenate([self._target_qpos(target.q), np.zeros(m.nv)])

        u = -self._K @ (x - x_target)
        return u[self.arm_act] + self.world.bias_force()

    def _target_qpos(self, target_q_arm: np.ndarray) -> np.ndarray:
        q_full = self.world.data.qpos[: self.world.model.nq].copy()
        q_full[self.world.arm_qpos_adr] = target_q_arm
        return q_full
