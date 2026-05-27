"""Run controllers and planners across scenarios and aggregate metrics."""

from __future__ import annotations

import time

import numpy as np

from manipdyn.benchmark.metrics import PlannerRun, rollout_controller
from manipdyn.benchmark.scenarios import planner_query, reach_targets
from manipdyn.control import CONTROLLERS
from manipdyn.planning import PLANNERS, shortcut_path
from manipdyn.sim import World
from manipdyn.tuning import tuned_controller

# Per-planner budgets for the benchmark (kept modest so a full run is quick).
_PLANNER_KW = {
    "rrt": {"max_iter": 5000},
    "rrt_connect": {"max_iter": 5000},
    "rrt_star": {"max_iter": 1500},
    "informed_rrt_star": {"max_iter": 1500},
    "prm": {"n_samples": 200, "k_neighbors": 12},
}


def _mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("inf")


def benchmark_controllers(
    scene: str = "scene_base",
    controllers: list[str] | None = None,
    duration: float = 3.0,
    settle_tol_mm: float = 20.0,
) -> list[dict]:
    """Run each controller (tuned gains) on every reach target; aggregate."""
    targets = reach_targets(scene)
    names = controllers or list(CONTROLLERS)
    rows = []
    for name in names:
        runs = []
        for target in targets:
            world = World(scene=scene)
            controller = tuned_controller(name, world)
            runs.append(
                rollout_controller(
                    world, controller, target, duration=duration, settle_tol_mm=settle_tol_mm
                )
            )
        rows.append(
            {
                "controller": name,
                "success_rate": float(np.mean([r.success for r in runs])),
                "final_err_mm": _mean([r.final_err_mm for r in runs]),
                "settle_s": _mean([r.settle_time_s for r in runs]),
                "rmse_mm": _mean([r.ee_rmse_mm for r in runs]),
                "effort": _mean([r.effort for r in runs]),
                "peak_torque_nm": _mean([r.peak_torque_nm for r in runs]),
                "compute_ms": _mean([r.compute_ms for r in runs]),
            }
        )
    return rows


def _path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def benchmark_planners(
    planners: list[str] | None = None,
    n_trials: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Run each planner over several seeds on the obstacle query; aggregate."""
    query = planner_query()
    names = planners or list(PLANNERS)
    rows = []
    for name in names:
        runs: list[PlannerRun] = []
        for trial in range(n_trials):
            world = World(scene=query.scene)
            planner = PLANNERS[name](world, seed=seed + trial, **_PLANNER_KW.get(name, {}))
            t0 = time.perf_counter()
            path = planner.plan(query.q_start, query.q_goal)
            dt = time.perf_counter() - t0
            if path is None:
                runs.append(PlannerRun(name, False, dt, float("inf"), 0, False))
                continue
            short = shortcut_path(path, planner.checker, iterations=100, seed=seed + trial)
            collision_free = all(not planner.checker.in_collision(q) for q in path)
            runs.append(PlannerRun(name, True, dt, _path_length(short), len(path), collision_free))
        ok = [r for r in runs if r.success]
        rows.append(
            {
                "planner": name,
                "success_rate": float(np.mean([r.success for r in runs])),
                "plan_time_s": _mean([r.plan_time_s for r in runs]),
                "path_len_rad": _mean([r.path_len for r in ok]) if ok else float("inf"),
                "raw_nodes": _mean([float(r.raw_nodes) for r in ok]) if ok else float("inf"),
                "collision_free": all(r.collision_free for r in ok) if ok else False,
            }
        )
    return rows
