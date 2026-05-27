"""Damped least-squares (Levenberg-Marquardt) inverse kinematics.

Solves for the 6 arm joint angles that place the end-effector site at a target
position (and, optionally, orientation). Runs on a private ``MjData`` so it
never disturbs the live simulation state, and clamps to the model's joint
limits. Returns a structured :class:`IKResult` (instead of the original
``q``-or-``None``, which different call sites unpacked inconsistently).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from manipdyn.sim.world import World


@dataclass
class IKResult:
    q: np.ndarray  # arm configuration (n_arm,)
    success: bool  # converged within tolerance
    error: float  # final task-space error norm
    iterations: int  # iterations used


class IKSolver:
    def __init__(
        self,
        world: World,
        step_size: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-3,
        damping: float = 0.1,
    ):
        self.world = world
        self.model = world.model
        self._data = mujoco.MjData(world.model)
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.site_id = world.ee_site_id
        self._qpos_adr = world.arm_qpos_adr
        self._dof_adr = world.arm_dof_adr
        self._limits = world.joint_limits

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        q_guess: np.ndarray | None = None,
    ) -> IKResult:
        d = self._data
        # Seed from a guess or the world's current configuration.
        d.qpos[:] = self.world.data.qpos
        d.qvel[:] = 0.0
        if q_guess is not None:
            d.qpos[self._qpos_adr] = np.asarray(q_guess, dtype=float)

        target_pos = np.asarray(target_pos, dtype=float)
        use_ori = target_quat is not None
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        lam2 = self.damping**2

        err_norm = np.inf
        i = 0
        for i in range(self.max_iter):
            mujoco.mj_forward(self.model, d)
            curr_pos = d.site_xpos[self.site_id]
            err_pos = target_pos - curr_pos

            if use_ori:
                curr_quat = np.zeros(4)
                mujoco.mju_mat2Quat(curr_quat, d.site_xmat[self.site_id])
                err_rot = np.zeros(3)
                mujoco.mju_subQuat(err_rot, target_quat, curr_quat)
                err = np.concatenate([err_pos, err_rot])
            else:
                err = err_pos

            err_norm = float(np.linalg.norm(err))
            if err_norm < self.tol:
                return IKResult(d.qpos[self._qpos_adr].copy(), True, err_norm, i)

            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.site_id)
            J = (
                np.vstack([jacp[:, self._dof_adr], jacr[:, self._dof_adr]])
                if use_ori
                else jacp[:, self._dof_adr]
            )

            # dq = J^T (J J^T + lambda^2 I)^-1 err
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(JJt.shape[0]), err)

            q_new = d.qpos[self._qpos_adr] + self.step_size * dq
            d.qpos[self._qpos_adr] = np.clip(q_new, self._limits[:, 0], self._limits[:, 1])

        return IKResult(d.qpos[self._qpos_adr].copy(), False, err_norm, i + 1)
