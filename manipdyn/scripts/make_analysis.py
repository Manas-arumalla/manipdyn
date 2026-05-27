"""Deep-dive analysis figures (publication style), into ``benchmarks/results/``:

  * ``traj_profiles.png``  — a time-optimal trajectory's joint velocity and
    acceleration profiles, with the limits drawn: the classic bang-bang shape
    that rides the constraints (TOPP).
  * ``planner_paths.png``  — the end-effector paths each planner produces around
    the pillar, in top (x-y) and side (x-z) views with the obstacle drawn.
  * ``tuning_convergence.png`` — best-so-far cost vs evaluation for the
    black-box auto-tuner (global search then a Nelder-Mead polish).
  * ``controller_heatmap.png`` — a controller × metric matrix, each metric
    normalized so the colour shows who wins each criterion.

Run from the manipdyn/ directory:
    python scripts/make_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from manipdyn.benchmark.harness import benchmark_controllers  # noqa: E402
from manipdyn.control import Target  # noqa: E402
from manipdyn.planning import PLANNERS, shortcut_path  # noqa: E402
from manipdyn.sim import World  # noqa: E402
from manipdyn.trajectory import parameterize_time_optimal  # noqa: E402
from manipdyn.tuning import TUNE_SPECS, tune_controller  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
plt.rcParams.update(
    {
        "savefig.dpi": 160,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#dddddd",
        "font.size": 11,
    }
)

_OBS_Q_START = np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0])
_OBS_Q_GOAL = np.array([-1.4, -1.2, 1.4, -1.7, -1.57, 0.0])
_PLANNER_KW = {
    "rrt": {"max_iter": 6000, "goal_bias": 0.2},
    "rrt_connect": {"max_iter": 5000},
    "rrt_star": {"max_iter": 3000, "goal_bias": 0.2},
    "informed_rrt_star": {"max_iter": 3000, "goal_bias": 0.2},
    "prm": {"n_samples": 500, "k_neighbors": 15},
}


def traj_profiles() -> Path:
    vmax, amax = np.full(6, 1.5), np.full(6, 2.5)
    waypoints = np.array(
        [
            [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
            [1.0, -1.1, 1.2, -1.6, -1.4, 0.4],
            [-1.0, -0.8, 1.0, -1.5, -1.5, 0.0],
        ]
    )
    tr = parameterize_time_optimal(waypoints, vmax, amax, n_samples=240)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    for j in range(6):
        a1.plot(tr.t, tr.qd[:, j], color=cmap(j), lw=1.4, label=f"joint {j + 1}")
        a2.plot(tr.t, tr.qdd[:, j], color=cmap(j), lw=1.4)
    for v in (vmax[0], -vmax[0]):
        a1.axhline(v, ls="--", color="k", lw=0.9, alpha=0.6)
    for v in (amax[0], -amax[0]):
        a2.axhline(v, ls="--", color="k", lw=0.9, alpha=0.6)
    a1.set(title="Time-optimal trajectory — joint velocities (dashed = ±limit)", ylabel="rad/s")
    a2.set(title="Joint accelerations (dashed = ±limit)", ylabel="rad/s²", xlabel="time (s)")
    a1.legend(ncol=6, fontsize=8, loc="upper center")
    fig.suptitle(
        "TOPP — the speed profile rides the velocity/acceleration limits",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "traj_profiles.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def planner_paths() -> Path:
    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("tab10")
    for k, (name, kw) in enumerate(_PLANNER_KW.items()):
        w = World(scene="scene_obstacle")
        planner = PLANNERS[name](w, seed=0, **kw)
        path = planner.plan(_OBS_Q_START, _OBS_Q_GOAL)
        path = shortcut_path(path, planner.checker, iterations=150, seed=0)
        pts = []
        for i in range(len(path) - 1):
            for a in np.linspace(0, 1, 30):
                q = (1 - a) * path[i] + a * path[i + 1]
                w.set_arm_qpos(q)
                w.forward()
                pts.append(w.ee_pos.copy())
        pts = np.array(pts)
        ax_xy.plot(pts[:, 0], pts[:, 1], color=cmap(k), lw=1.8, label=name)
        ax_xz.plot(pts[:, 0], pts[:, 2], color=cmap(k), lw=1.8, label=name)

    # obstacle pillar: box at (-0.57, 0.30), half (0.05, 0.05, 0.225), z in [0, 0.45]
    import matplotlib.patches as mpatches

    ax_xy.add_patch(mpatches.Rectangle((-0.62, 0.25), 0.10, 0.10, color="#3a5fcd", alpha=0.5))
    ax_xz.add_patch(mpatches.Rectangle((-0.62, 0.0), 0.10, 0.45, color="#3a5fcd", alpha=0.5))
    ax_xy.set(title="End-effector path — top view (x-y)", xlabel="x (m)", ylabel="y (m)")
    ax_xz.set(
        title="End-effector path — side view (x-z), pillar in blue", xlabel="x (m)", ylabel="z (m)"
    )
    ax_xy.axis("equal")
    ax_xy.legend(fontsize=8)
    fig.suptitle(
        "Planner solutions in the workspace — every route avoids the pillar",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / "planner_paths.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def tuning_convergence() -> Path:
    w = World(scene="scene_base")
    w.set_arm_qpos(np.array([1.0, -1.1, 1.2, -1.6, -1.4, 0.4]))
    w.forward()
    x_goal = w.ee_pos.copy()
    q_target = np.array([1.0, -1.1, 1.2, -1.6, -1.4, 0.4])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    n_global = 40
    for k, name in enumerate(("pid", "ctc", "osc")):
        spec = TUNE_SPECS[name]
        target = Target(q=q_target) if spec.target_space == "joint" else Target(x=x_goal)
        res = tune_controller(
            lambda: World(scene="scene_base"),
            spec.factory,
            spec.space,
            target,
            method="random",
            n_evals=n_global,
            polish=True,
            duration=2.5,
            seed=0,
        )
        best = np.minimum.accumulate(res.history)
        ax.plot(range(1, len(best) + 1), best, color=cmap(k), lw=1.8, label=name)
    ax.axvline(n_global + 0.5, ls="--", color="k", lw=0.9, alpha=0.6)
    ax.text(n_global + 1, ax.get_ylim()[1], "  Nelder-Mead polish →", fontsize=9, va="top")
    ax.set_yscale("log")
    ax.set(
        title="Auto-tuning convergence — best-so-far cost",
        xlabel="objective evaluation",
        ylabel="cost (log)",
    )
    ax.legend()
    fig.tight_layout()
    p = OUT / "tuning_convergence.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def controller_heatmap() -> Path:
    rows = benchmark_controllers()
    names = [r["controller"] for r in rows]
    # metric -> (raw values, lower_is_better)
    metrics = {
        "accuracy\n(final err)": ([r["final_err_mm"] for r in rows], True),
        "settle time": ([r["settle_s"] for r in rows], True),
        "effort ‖τ‖²": ([r["effort"] for r in rows], True),
        "peak torque": ([r["peak_torque_nm"] for r in rows], True),
        "compute/step": ([r["compute_ms"] for r in rows], True),
    }
    score = np.zeros((len(names), len(metrics)))
    raw = np.zeros_like(score)
    for j, (vals, lower_better) in enumerate(metrics.values()):
        v = np.log10(np.maximum(np.array(vals, float), 1e-6))  # log scale: metrics span decades
        s = (v - v.min()) / (np.ptp(v) + 1e-12)
        score[:, j] = 1.0 - s if lower_better else s  # 1 = best
        raw[:, j] = vals

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(score, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), labels=list(metrics), fontsize=9)
    ax.set_yticks(range(len(names)), labels=names)
    for i in range(len(names)):
        for j in range(len(metrics)):
            val = raw[i, j]
            txt = f"{val:.2g}" if (val >= 0.01 or val == 0) else f"{val:.0e}"
            sc = score[i, j]
            color = "white" if (sc < 0.30 or sc > 0.70) else "#1a1a1a"  # contrast on dark cells
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, fontweight="bold", color=color)
    ax.set_title("Controller scorecard — green = best in column (normalized)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="normalized score (1 = best)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = OUT / "controller_heatmap.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    print("traj_profiles:      ", traj_profiles())
    print("planner_paths:      ", planner_paths())
    print("tuning_convergence: ", tuning_convergence())
    print("controller_heatmap: ", controller_heatmap())


if __name__ == "__main__":
    main()
