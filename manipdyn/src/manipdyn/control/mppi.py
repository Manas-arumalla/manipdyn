"""Model-Predictive Path Integral control (sampling-based MPC).

Samples noisy control sequences, rolls each out through the *true* MuJoCo
dynamics, and forms an information-theoretic (softmax) weighted average of the
perturbations — no gradients, no linearization, handles the full nonlinear
model and contacts:

    cost_k    = sum_t [ w_pos ||q* - q||^2 + w_vel ||v||^2 + w_ctrl ||u||^2 ]
    weight_k  = softmax(-cost_k / lambda)
    U        += sum_k weight_k * noise_k

The rollouts are pure-Python ``mj_step`` loops, so this is the most expensive
controller in the zoo (≈ n_samples × horizon steps per tick). It is a prime
candidate for GPU acceleration via MuJoCo MJX — see docs/theory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from manipdyn.control.base import Controller, Target

if TYPE_CHECKING:
    from manipdyn.sim.world import World


class MPPIController(Controller):
    name = "mppi"
    target_space = "joint"

    def __init__(
        self,
        world: World,
        horizon: int = 30,
        n_samples: int = 50,
        noise_sigma: float = 2.0,
        lambda_: float = 0.05,
        w_pos: float = 5000.0,
        w_vel: float = 20.0,
        w_ctrl: float = 1e-3,
        seed: int | None = None,
    ):
        super().__init__(world)
        self.horizon = horizon
        self.n_samples = n_samples
        self.sigma = noise_sigma
        self.lam = lambda_
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_ctrl = w_ctrl

        self._d = mujoco.MjData(world.model)
        self.arm_act = world.arm_actuator_ids
        self.arm_dof = world.arm_dof_adr
        self.arm_qpos = world.arm_qpos_adr
        self.U = np.zeros((horizon, self.n_arm))
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.U = np.zeros((self.horizon, self.n_arm))

    def compute(self, target: Target) -> np.ndarray:
        m = self.world.model
        d = self._d
        q_target = target.q

        # Warm-start: shift the previous schedule one step forward.
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        noise = self.rng.normal(0.0, self.sigma, (self.n_samples, self.horizon, self.n_arm))
        costs = np.zeros(self.n_samples)

        qpos0 = self.world.data.qpos.copy()
        qvel0 = self.world.data.qvel.copy()

        for k in range(self.n_samples):
            d.qpos[:] = qpos0
            d.qvel[:] = qvel0
            d.time = 0.0
            mujoco.mj_forward(m, d)
            for t in range(self.horizon):
                u = self.U[t] + noise[k, t]
                d.ctrl[self.arm_act] = u + d.qfrc_bias[self.arm_dof]  # gravity-comped
                mujoco.mj_step(m, d)

                q_err = q_target - d.qpos[self.arm_qpos]
                v = d.qvel[self.arm_dof]
                costs[k] += self.w_pos * q_err @ q_err + self.w_vel * v @ v + self.w_ctrl * u @ u

        # Information-theoretic weighting (subtract min for stability).
        weights = np.exp(-(costs - costs.min()) / self.lam)
        weights /= weights.sum()

        self.U += np.einsum("k,kht->ht", weights, noise)
        return self.U[0] + self.world.bias_force()
