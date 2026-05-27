"""Metrics for the benchmark: controller rollouts and planner queries.

All controllers — joint-space and Cartesian — are scored in a *common* space:
end-effector position error toward the goal. That makes the comparison fair
across methods that track different set-point types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from manipdyn.control.base import Controller, Target
from manipdyn.sim.world import World


@dataclass
class ControllerRun:
    name: str
    success: bool
    final_err_mm: float  # final EE position error (mm)
    settle_time_s: float  # time to stay within tolerance (s)
    ee_rmse_mm: float  # RMS EE error over the run (mm)
    effort: float  # mean ||tau||^2 (Nm^2)
    peak_torque_nm: float  # peak ||tau|| (Nm)
    compute_ms: float  # mean per-step compute time (ms)
    t: np.ndarray = field(default=None, repr=False)
    err_mm: np.ndarray = field(default=None, repr=False)


@dataclass
class PlannerRun:
    name: str
    success: bool
    plan_time_s: float
    path_len: float  # joint-space path length (rad), shortcutted
    raw_nodes: int
    collision_free: bool


def _settle_time(err: np.ndarray, dt: float, tol: float) -> float:
    below = err < tol
    for i in range(len(below)):
        if below[i:].all():
            return i * dt
    return len(err) * dt


def rollout_controller(
    world: World,
    controller: Controller,
    target: Target,
    *,
    duration: float = 3.0,
    settle_tol_mm: float = 20.0,
) -> ControllerRun:
    """Run one controller to a goal and measure performance.

    ``target`` must have ``x`` set (the Cartesian goal); joint-space
    controllers additionally use ``target.q``.
    """
    controller.reset()
    world.reset(world.home_qpos_arm)
    n = int(duration / world.timestep)
    x_goal = np.asarray(target.x, dtype=float)

    t = np.empty(n)
    err = np.empty(n)
    tau_sq = np.empty(n)
    ctime = np.empty(n)

    for i in range(n):
        t0 = time.perf_counter()
        tau = controller.compute(target)
        ctime[i] = (time.perf_counter() - t0) * 1e3
        world.step(tau)
        t[i] = world.time
        err[i] = np.linalg.norm(x_goal - world.ee_pos)
        tau_sq[i] = float(tau @ tau)

    finite = bool(np.all(np.isfinite(err)) and np.all(np.isfinite(tau_sq)))
    final_mm = float(err[-1] * 1e3) if finite else float("inf")
    tol_m = settle_tol_mm / 1e3
    return ControllerRun(
        name=controller.name,
        success=finite and final_mm < settle_tol_mm,
        final_err_mm=final_mm,
        settle_time_s=_settle_time(err, world.timestep, tol_m) if finite else duration,
        ee_rmse_mm=float(np.sqrt(np.mean(err**2)) * 1e3) if finite else float("inf"),
        effort=float(np.mean(tau_sq)) if finite else float("inf"),
        peak_torque_nm=float(np.sqrt(np.max(tau_sq))) if finite else float("inf"),
        compute_ms=float(np.mean(ctime)),
        t=t,
        err_mm=err * 1e3,
    )
