"""iLQR controller: optimize a trajectory once, execute it with feedback.

Wraps :class:`~manipdyn.trajopt.ilqr.ILQR` behind the standard controller
interface. On the first call (or when the goal changes) it solves for an
optimal torque trajectory and time-varying gains, then tracks it:

    u_t = U*[k] + K[k] (x - X*[k])

where ``k`` indexes the coarse control trajectory and each coarse control is
held for ``control_dt / sim_dt`` fine steps. The optimized torques already
include gravity compensation, so none is added. Requires an arm-only scene
(``nq == nv``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manipdyn.control.base import Controller, Target
from manipdyn.trajopt.ilqr import ILQR, ILQRResult

if TYPE_CHECKING:
    from manipdyn.sim.world import World


class ILQRController(Controller):
    name = "ilqr"
    target_space = "joint"

    def __init__(
        self,
        world: World,
        horizon: int = 100,
        control_dt: float = 0.02,
        relinearize_tol: float = 1e-2,
        hold_kp: float = 200.0,
        hold_kd: float = 30.0,
        **ilqr_kwargs,
    ):
        super().__init__(world)
        self.ilqr = ILQR(world, horizon=horizon, control_dt=control_dt, **ilqr_kwargs)
        self.substeps = max(1, round(control_dt / world.timestep))
        self.relinearize_tol = relinearize_tol
        self.hold_kp = hold_kp
        self.hold_kd = hold_kd
        self._result: ILQRResult | None = None
        self._goal: np.ndarray | None = None
        self._k = 0

    def reset(self) -> None:
        self._result = None
        self._goal = None
        self._k = 0

    def _state(self) -> np.ndarray:
        nv = self.world.model.nv
        return np.concatenate([self.world.data.qpos[:nv], self.world.data.qvel[:nv]])

    def _ensure_plan(self, target_q: np.ndarray) -> None:
        if (
            self._result is None
            or self._goal is None
            or np.linalg.norm(target_q - self._goal) > self.relinearize_tol
        ):
            self._result = self.ilqr.optimize(target_q, self._state())
            self._goal = np.asarray(target_q, dtype=float).copy()
            self._k = 0

    def compute(self, target: Target) -> np.ndarray:
        self._ensure_plan(target.q)
        res = self._result
        coarse_idx = self._k // self.substeps
        self._k += 1

        if coarse_idx >= res.U.shape[0]:
            # Optimized horizon exhausted: hold the goal with a stabilizing
            # PD + gravity compensation (the one-step terminal gain alone does
            # not regulate against fine-vs-coarse integration drift).
            q, v = self.world.qpos_arm, self.world.qvel_arm
            return self.world.bias_force() + self.hold_kp * (target.q - q) - self.hold_kd * v

        return res.U[coarse_idx] + res.K[coarse_idx] @ (self._state() - res.X[coarse_idx])
