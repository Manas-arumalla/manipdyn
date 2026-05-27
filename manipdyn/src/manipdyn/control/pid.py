"""Joint-space PID controller with anti-windup — the baseline of the zoo.

tau = Kp (q* - q) + Ki integral(q* - q) dt + Kd (v* - v)  [+ gravity comp]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manipdyn.control.base import Controller, Target

if TYPE_CHECKING:
    from manipdyn.sim.world import World

Gain = float | np.ndarray


class PIDController(Controller):
    name = "pid"
    target_space = "joint"

    def __init__(
        self,
        world: World,
        kp: Gain = 300.0,
        ki: Gain = 8.0,
        kd: Gain = 60.0,
        integral_limit: float = 20.0,
        gravity_comp: bool = True,
    ):
        super().__init__(world)
        self.dt = world.timestep
        self.kp = self._broadcast(kp)
        self.ki = self._broadcast(ki)
        self.kd = self._broadcast(kd)
        self.integral_limit = float(integral_limit)
        self.gravity_comp = gravity_comp
        self._integral = np.zeros(self.n_arm)

    def _broadcast(self, g: Gain) -> np.ndarray:
        return np.full(self.n_arm, float(g)) if np.isscalar(g) else np.asarray(g, dtype=float)

    def reset(self) -> None:
        self._integral = np.zeros(self.n_arm)

    def compute(self, target: Target) -> np.ndarray:
        q, v = self.world.qpos_arm, self.world.qvel_arm
        target_v = np.zeros(self.n_arm) if target.v is None else target.v

        pos_err = target.q - q
        vel_err = target_v - v

        self._integral += pos_err * self.dt
        np.clip(self._integral, -self.integral_limit, self.integral_limit, out=self._integral)

        tau = self.kp * pos_err + self.ki * self._integral + self.kd * vel_err
        if self.gravity_comp:
            tau = tau + self.world.bias_force()
        return tau
