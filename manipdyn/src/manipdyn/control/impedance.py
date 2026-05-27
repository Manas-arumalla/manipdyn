"""Cartesian impedance control (Jacobian transpose).

Renders the end-effector as a spring-damper toward a Cartesian target without
inverting the Jacobian, so it degrades gracefully near singularities:

    F   = Kp (x* - x) + Kd (xdot* - xdot)
    tau = J_p^T F  [+ gravity comp]

Compliant and contact-safe, at the cost of steady-state error under load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manipdyn.control.base import Controller, Target

if TYPE_CHECKING:
    from manipdyn.sim.world import World

Gain = float | np.ndarray


class ImpedanceController(Controller):
    name = "impedance"
    target_space = "cartesian"

    def __init__(
        self,
        world: World,
        kp: Gain = 800.0,
        kd: Gain = 80.0,
        gravity_comp: bool = True,
    ):
        super().__init__(world)
        self.kp = self._broadcast(kp)
        self.kd = self._broadcast(kd)
        self.gravity_comp = gravity_comp

    @staticmethod
    def _broadcast(g: Gain) -> np.ndarray:
        return np.full(3, float(g)) if np.isscalar(g) else np.asarray(g, dtype=float)

    def compute(self, target: Target) -> np.ndarray:
        jp, _ = self.world.ee_jacobian()  # (3, n_arm)
        x = self.world.ee_pos
        xdot = jp @ self.world.qvel_arm
        target_xdot = np.zeros(3) if target.xdot is None else target.xdot

        force = self.kp * (target.x - x) + self.kd * (target_xdot - xdot)
        tau = jp.T @ force
        if self.gravity_comp:
            tau = tau + self.world.bias_force()
        return tau
