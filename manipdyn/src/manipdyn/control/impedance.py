"""Cartesian impedance control (Jacobian transpose).

Renders the end-effector as a spring-damper toward a Cartesian target without
inverting the Jacobian, so it degrades gracefully near singularities:

    F   = Kp (x* - x) + Kd (xdot* - xdot)
    tau = J_p^T F  [+ gravity comp]

Compliant and contact-safe, at the cost of steady-state error under load. When
the :class:`Target` carries a desired orientation ``R``, an orientation wrench
``M_o = Kp_rot e_o - Kd_rot omega`` is added through the rotation Jacobian
(``tau += J_r^T M_o``); with ``R`` omitted the controller is position-only and
unchanged.
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
        kp_rot: Gain = 30.0,
        kd_rot: Gain = 3.0,
    ):
        super().__init__(world)
        self.kp = self._broadcast(kp)
        self.kd = self._broadcast(kd)
        self.gravity_comp = gravity_comp
        # Orientation gains (used only when a target orientation is supplied).
        self.kp_rot = self._broadcast(kp_rot)
        self.kd_rot = self._broadcast(kd_rot)

    @staticmethod
    def _broadcast(g: Gain) -> np.ndarray:
        return np.full(3, float(g)) if np.isscalar(g) else np.asarray(g, dtype=float)

    def compute(self, target: Target) -> np.ndarray:
        jp, jr = self.world.ee_jacobian()  # each (3, n_arm)
        v = self.world.qvel_arm
        x = self.world.ee_pos
        xdot = jp @ v
        target_xdot = np.zeros(3) if target.xdot is None else target.xdot

        force = self.kp * (target.x - x) + self.kd * (target_xdot - xdot)
        tau = jp.T @ force

        # Optional orientation spring-damper through the rotation Jacobian.
        if target.R is not None:
            r_cur, r_des = self.world.ee_rot(), np.asarray(target.R, dtype=float)
            e_o = 0.5 * sum(np.cross(r_cur[:, i], r_des[:, i]) for i in range(3))
            moment = self.kp_rot * e_o - self.kd_rot * (jr @ v)
            tau = tau + jr.T @ moment

        if self.gravity_comp:
            tau = tau + self.world.bias_force()
        return tau
