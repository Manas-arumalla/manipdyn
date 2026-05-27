"""Optimize the gains of every controller and save them as presets.

Runs the auto-tuner (global search + Nelder-Mead polish) on each controller in
``TUNE_SPECS`` against a common regulation scenario, then writes the best gains
to ``manipdyn/tuning/tuned_gains.json``. Prints a baseline-vs-tuned table.

Run (from the manipdyn/ directory):
    python scripts/optimize_controllers.py
"""

from __future__ import annotations

import datetime as _dt
import json

import numpy as np

from manipdyn.control import CONTROLLERS, Target
from manipdyn.sim import World
from manipdyn.tuning import TUNE_SPECS, tune_controller
from manipdyn.tuning.autotune import evaluate_controller
from manipdyn.tuning.presets import _PRESETS_FILE

SCENE = "scene_base"
Q_TARGET = np.array([1.0, -1.1, 1.2, -1.6, -1.4, 0.4])


def _make_world() -> World:
    return World(scene=SCENE)


def _target_for(target_space: str, x_goal: np.ndarray) -> Target:
    return Target(q=Q_TARGET) if target_space == "joint" else Target(x=x_goal)


def main() -> None:
    # Cartesian goal = forward kinematics of the joint target.
    w = _make_world()
    w.set_arm_qpos(Q_TARGET)
    w.forward()
    x_goal = w.ee_pos.copy()

    results: dict[str, dict] = {}
    print(f"{'controller':12s} {'baseline':>10s} {'tuned':>10s} {'improve':>9s}   params")
    print("-" * 78)

    for name, spec in TUNE_SPECS.items():
        target = _target_for(spec.target_space, x_goal)

        # Baseline cost with library defaults.
        world = _make_world()
        world.reset(world.home_qpos_arm)
        baseline = evaluate_controller(
            world, CONTROLLERS[name](world), target, duration=spec.duration
        )["cost"]

        # Optimized gains.
        res = tune_controller(
            _make_world,
            spec.factory,
            spec.space,
            target,
            method=spec.method,
            n_evals=spec.n_evals,
            polish=spec.polish,
            duration=spec.duration,
            seed=0,
        )

        improve = (baseline - res.best_cost) / max(baseline, 1e-9) * 100.0
        improved = res.best_cost < baseline
        entry = {
            "cost": round(min(res.best_cost, float(baseline)), 4),
            "baseline_cost": round(float(baseline), 4),
            "method": spec.method,
            "n_evals": res.n_evals,
            "improved": improved,
        }
        # Only adopt tuned gains if they actually beat the defaults; otherwise
        # fall back to library defaults (e.g. expensive samplers tuned on a
        # small budget can regress).
        if improved:
            entry["params"] = {k: round(v, 4) for k, v in res.best_params.items()}
        results[name] = entry

        tag = "" if improved else "  (kept defaults)"
        pretty = {k: round(v, 2) for k, v in res.best_params.items()}
        print(f"{name:12s} {baseline:10.3f} {res.best_cost:10.3f} {improve:8.1f}%   {pretty}{tag}")

    results["_meta"] = {
        "scene": SCENE,
        "target_joint": Q_TARGET.tolist(),
        "cost_weights": "w_err=10, w_settle=1, w_effort=1e-3",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    _PRESETS_FILE.write_text(json.dumps(results, indent=2))
    print("-" * 78)
    print(f"Saved tuned gains -> {_PRESETS_FILE}")


if __name__ == "__main__":
    main()
