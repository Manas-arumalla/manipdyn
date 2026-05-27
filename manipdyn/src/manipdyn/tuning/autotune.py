"""Black-box optimization of controller gains.

Treats a controller's gains as decision variables and minimizes a performance
cost measured from a closed-loop rollout:

    cost = w_err * final_error + w_settle * settle_time + w_effort * mean|tau|

Two roles in this project:
  1. **Make controllers better** — find good gains automatically instead of by
     hand.
  2. **Fair benchmarking** — give every controller the *same* tuning budget on
     the *same* scenario, so the comparison reflects the method, not how long
     someone fiddled with one set of gains.

Combines a global search (random search, SciPy differential evolution, or dual
annealing) with a bounded Nelder-Mead polish of the best candidate. The cost is
a plain callable, so any optimizer can be dropped in later (CMA-ES, Bayesian
optimization, Optuna).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from manipdyn.control.base import Controller, Target
from manipdyn.sim.world import World

WorldFactory = Callable[[], World]
ControllerFactory = Callable[..., Controller]


@dataclass
class TuneResult:
    best_params: dict[str, float]
    best_cost: float
    n_evals: int
    history: list[float] = field(default_factory=list)


def _settle_time(errors: np.ndarray, dt: float, tol: float) -> float:
    """First time the error stays below ``tol`` for the rest of the run."""
    below = errors < tol
    for i in range(len(below)):
        if below[i:].all():
            return i * dt
    return len(errors) * dt  # never settled -> full duration (penalty)


def evaluate_controller(
    world: World,
    controller: Controller,
    target: Target,
    duration: float = 4.0,
    settle_tol: float = 0.05,
    weights: tuple[float, float, float] = (10.0, 1.0, 0.001),
) -> dict[str, float]:
    """Roll out a controller and return performance metrics."""
    controller.reset()
    n = int(duration / world.timestep)
    errs = np.empty(n)
    taus = np.empty(n)
    use_joint = target.q is not None

    for i in range(n):
        tau = controller.compute(target)
        world.step(tau)
        taus[i] = np.linalg.norm(tau)
        errs[i] = (
            np.linalg.norm(target.q - world.qpos_arm)
            if use_joint
            else np.linalg.norm(target.x - world.ee_pos)
        )

    if not np.all(np.isfinite(errs)) or not np.all(np.isfinite(taus)):
        return {"cost": 1e6, "final_error": np.inf, "settle_time": duration, "effort": np.inf}

    w_err, w_settle, w_effort = weights
    final_error = float(errs[-1])
    settle = _settle_time(errs, world.timestep, settle_tol)
    effort = float(taus.mean())
    cost = w_err * final_error + w_settle * settle + w_effort * effort
    return {"cost": cost, "final_error": final_error, "settle_time": settle, "effort": effort}


def tune_controller(
    world_factory: WorldFactory,
    controller_factory: ControllerFactory,
    param_space: dict[str, tuple[float, float]],
    target: Target,
    *,
    duration: float = 4.0,
    method: str = "random",
    n_evals: int = 40,
    polish: bool = True,
    seed: int = 0,
    **eval_kwargs,
) -> TuneResult:
    """Optimize controller gains over ``param_space`` to minimize the cost.

    Parameters
    ----------
    world_factory:
        Returns a *fresh* :class:`World` for each evaluation (so rollouts are
        independent).
    controller_factory:
        ``controller_factory(world, **params) -> Controller``.
    param_space:
        Maps each gain name to ``(low, high)`` bounds.
    method:
        Global search strategy: ``"random"`` (default), ``"de"`` (differential
        evolution) or ``"anneal"`` (dual annealing).
    polish:
        If true, refine the best candidate with bounded Nelder-Mead.
    """
    names = list(param_space)
    bounds = [param_space[n] for n in names]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    history: list[float] = []

    def objective(vec: np.ndarray) -> float:
        # Clip into bounds so gradient-free local steps stay feasible.
        vec = np.clip(np.asarray(vec, dtype=float), lo, hi)
        params = {n: float(v) for n, v in zip(names, vec, strict=True)}
        world = world_factory()
        world.reset(world.home_qpos_arm)
        controller = controller_factory(world, **params)
        cost = evaluate_controller(world, controller, target, duration, **eval_kwargs)["cost"]
        history.append(cost)
        return cost

    # --- global search -------------------------------------------------
    if method == "de":
        from scipy.optimize import differential_evolution

        res = differential_evolution(
            objective,
            bounds,
            maxiter=max(1, n_evals // (10 * len(names))),
            popsize=10,
            seed=seed,
            polish=False,
            tol=1e-3,
        )
        best_vec, best_cost = np.clip(res.x, lo, hi), float(res.fun)
    elif method == "anneal":
        from scipy.optimize import dual_annealing

        res = dual_annealing(objective, list(zip(lo, hi, strict=True)), maxiter=n_evals, seed=seed)
        best_vec, best_cost = np.clip(res.x, lo, hi), float(res.fun)
    else:  # random search
        rng = np.random.default_rng(seed)
        best_vec, best_cost = lo.copy(), np.inf
        for _ in range(n_evals):
            vec = rng.uniform(lo, hi)
            c = objective(vec)
            if c < best_cost:
                best_cost, best_vec = c, vec

    # --- local polish (bounded Nelder-Mead) ----------------------------
    if polish:
        from scipy.optimize import minimize

        res = minimize(
            objective,
            best_vec,
            method="Nelder-Mead",
            options={"maxiter": max(15, n_evals // 2), "xatol": 1e-3, "fatol": 1e-4},
        )
        if float(res.fun) < best_cost:
            best_vec, best_cost = np.clip(res.x, lo, hi), float(res.fun)

    return TuneResult(
        best_params={n: float(v) for n, v in zip(names, best_vec, strict=True)},
        best_cost=best_cost,
        n_evals=len(history),
        history=history,
    )
