"""Iterative LQR (iLQR) trajectory optimization on the MuJoCo dynamics.

Computes a locally optimal open-loop torque sequence (and time-varying feedback
gains) that drives the arm to a goal while minimizing a quadratic cost:

    J = lf(x_N) + sum_t  w_q||q-q*||^2 + w_v||v||^2 + w_u||u||^2

iLQR alternates a **backward pass** (Riccati-like recursion using the dynamics
Jacobians f_x, f_u from :func:`mujoco.mjd_transitionFD`) and a line-searched
**forward pass**, with Levenberg-Marquardt regularization on the control
Hessian for robustness. (Gauss-Newton iLQR; full DDP would add the dynamics'
second-order terms.)

To keep the horizon affordable it optimizes on a *private model copy at a
coarser control timestep*; the controller wrapper executes each control for the
corresponding number of fine simulation steps with feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from manipdyn.sim.world import World


@dataclass
class ILQRResult:
    X: np.ndarray  # (N+1, 2*nv) nominal states
    U: np.ndarray  # (N, m)      nominal controls (arm torque)
    K: np.ndarray  # (N, m, 2*nv) feedback gains
    cost: float
    converged: bool
    iterations: int
    control_dt: float


class ILQR:
    def __init__(
        self,
        world: World,
        horizon: int = 100,
        control_dt: float = 0.02,
        w_q: float = 10.0,
        w_v: float = 0.1,
        w_u: float = 1e-4,
        w_qf: float = 200.0,
        w_vf: float = 2.0,
        reg_init: float = 1e-6,
        max_iter: int = 50,
        tol: float = 1e-4,
    ):
        # Private model copy at the (coarser) control timestep.
        self.model = mujoco.MjModel.from_xml_path(world.scene_path)
        self.model.opt.timestep = control_dt
        self.data = mujoco.MjData(self.model)
        if self.model.nq != self.model.nv:
            raise ValueError("iLQR currently supports arm-only scenes (nq == nv).")

        self.nv = self.model.nv
        self.n = 2 * self.nv
        self.arm_act = np.asarray(world.arm_actuator_ids, dtype=int)
        self.arm_qpos = np.asarray(world.arm_qpos_adr, dtype=int)
        self.arm_dof = np.asarray(world.arm_dof_adr, dtype=int)
        self.m = len(self.arm_act)

        self.H = horizon
        self.control_dt = control_dt
        self.max_iter = max_iter
        self.tol = tol
        self.reg_init = reg_init

        # State cost weight vectors (per state dimension).
        self._w_x = np.concatenate([np.full(self.nv, w_q), np.full(self.nv, w_v)])
        self._w_xf = np.concatenate([np.full(self.nv, w_qf), np.full(self.nv, w_vf)])
        self._w_u = w_u

    # -- dynamics ---------------------------------------------------------
    def _set(self, x: np.ndarray, u: np.ndarray | None = None) -> None:
        self.data.qpos[:] = x[: self.nv]
        self.data.qvel[:] = x[self.nv :]
        self.data.ctrl[:] = 0.0
        if u is not None:
            self.data.ctrl[self.arm_act] = u

    def _state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        self._set(x, u)
        mujoco.mj_step(self.model, self.data)
        return self._state()

    def _jacobians(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._set(x, u)
        mujoco.mj_forward(self.model, self.data)
        A = np.zeros((self.n, self.n))
        B = np.zeros((self.n, self.model.nu))
        mujoco.mjd_transitionFD(self.model, self.data, 1e-6, True, A, B, None, None)
        return A, B[:, self.arm_act]

    def _rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        X = np.zeros((self.H + 1, self.n))
        X[0] = x0
        for t in range(self.H):
            X[t + 1] = self._step(X[t], U[t])
        return X

    # -- cost -------------------------------------------------------------
    def _traj_cost(self, X: np.ndarray, U: np.ndarray, x_star: np.ndarray) -> float:
        run = np.sum(self._w_x * (X[:-1] - x_star) ** 2) + self._w_u * np.sum(U**2)
        term = np.sum(self._w_xf * (X[-1] - x_star) ** 2)
        return float(run + term)

    # -- optimize ---------------------------------------------------------
    def optimize(
        self,
        q_goal: np.ndarray,
        x0: np.ndarray,
        U_init: np.ndarray | None = None,
    ) -> ILQRResult:
        n, m, H = self.n, self.m, self.H
        x_star = np.zeros(n)
        x_star[self.arm_qpos] = np.asarray(q_goal, dtype=float)

        # Warm start: gravity-compensation torque at the start state.
        if U_init is None:
            self._set(x0)
            mujoco.mj_forward(self.model, self.data)
            u0 = self.data.qfrc_bias[self.arm_dof].copy()
            U = np.tile(u0, (H, 1))
        else:
            U = U_init.copy()

        X = self._rollout(x0, U)
        cost = self._traj_cost(X, U, x_star)
        reg = self.reg_init
        K = np.zeros((H, m, n))
        converged = False

        Wxx = np.diag(2.0 * self._w_x)
        Wxxf = np.diag(2.0 * self._w_xf)
        Wuu = 2.0 * self._w_u * np.eye(m)

        last_it = 0
        for it in range(self.max_iter):
            last_it = it
            fx = np.zeros((H, n, n))
            fu = np.zeros((H, n, m))
            for t in range(H):
                fx[t], fu[t] = self._jacobians(X[t], U[t])

            # Backward pass.
            Vx = Wxxf @ (X[-1] - x_star)
            Vxx = Wxxf.copy()
            k = np.zeros((H, m))
            diverged = False
            for t in reversed(range(H)):
                lx = Wxx @ (X[t] - x_star)
                lu = Wuu @ U[t]
                Qx = lx + fx[t].T @ Vx
                Qu = lu + fu[t].T @ Vx
                Qxx = Wxx + fx[t].T @ Vxx @ fx[t]
                Quu = Wuu + fu[t].T @ Vxx @ fu[t] + reg * np.eye(m)
                Qux = fu[t].T @ Vxx @ fx[t]
                try:
                    Quu_inv = np.linalg.inv(Quu)
                except np.linalg.LinAlgError:
                    diverged = True
                    break
                k[t] = -Quu_inv @ Qu
                K[t] = -Quu_inv @ Qux
                Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
                Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
                Vxx = 0.5 * (Vxx + Vxx.T)

            if diverged:
                reg *= 10.0
                if reg > 1e10:
                    break
                continue

            # Forward pass with backtracking line search.
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
                Xn = np.zeros_like(X)
                Un = np.zeros_like(U)
                Xn[0] = x0
                for t in range(H):
                    Un[t] = U[t] + alpha * k[t] + K[t] @ (Xn[t] - X[t])
                    Xn[t + 1] = self._step(Xn[t], Un[t])
                cost_new = self._traj_cost(Xn, Un, x_star)
                if cost_new < cost:
                    rel = (cost - cost_new) / max(cost, 1e-9)
                    X, U, cost = Xn, Un, cost_new
                    reg = max(reg / 2.0, 1e-8)
                    improved = True
                    if rel < self.tol:
                        converged = True
                    break

            if not improved:
                reg *= 10.0
                if reg > 1e10:
                    break
            if converged:
                break

        return ILQRResult(X, U, K, cost, converged, last_it + 1, self.control_dt)
