"""Computed-Torque Control (feedback linearization).

Cancels the manipulator's nonlinear dynamics using MuJoCo inverse dynamics:

    tau = M(q) (a_des + Kp e + Kd e_dot) + C(q, v) + g(q)

evaluated via :func:`mujoco.mj_inverse` on a private ``MjData`` so the live
simulation state is untouched. With exact dynamics this linearizes the
closed loop to a decoupled second-order system per joint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from manipdyn.control.base import Controller, Target

if TYPE_CHECKING:
    from manipdyn.sim.world import World

Gain = float | np.ndarray


class ComputedTorqueController(Controller):
    name = "ctc"
    target_space = "joint"

    def __init__(self, world: World, kp: Gain = 100.0, kd: Gain = 20.0):
        super().__init__(world)
        self.kp = self._broadcast(kp)
        self.kd = self._broadcast(kd)
        self._data = mujoco.MjData(world.model)
        self._dof_adr = world.arm_dof_adr

    def _broadcast(self, g: Gain) -> np.ndarray:
        return np.full(self.n_arm, float(g)) if np.isscalar(g) else np.asarray(g, dtype=float)

    def compute(self, target: Target) -> np.ndarray:
        q, v = self.world.qpos_arm, self.world.qvel_arm
        target_v = np.zeros(self.n_arm) if target.v is None else target.v
        target_a = np.zeros(self.n_arm) if target.a is None else target.a

        a_des = target_a + self.kp * (target.q - q) + self.kd * (target_v - v)

        d = self._data
        d.qpos[:] = self.world.data.qpos
        d.qvel[:] = self.world.data.qvel
        d.qacc[:] = 0.0
        d.qacc[self._dof_adr] = a_des

        mujoco.mj_inverse(self.world.model, d)
        return d.qfrc_inverse[self._dof_adr].copy()
