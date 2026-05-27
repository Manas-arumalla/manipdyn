"""Turn benchmark results into a Markdown report and publication-style plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# A clean, consistent look across every figure.
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.7,
        "legend.fontsize": 9,
        "legend.framealpha": 0.92,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

# array-valued keys (e.g. convergence curves) are for plots, not tables/JSON
_ARRAY_KEYS = ("curve_t", "curve_err_mm")


def _fmt(v: float) -> str:
    if isinstance(v, float):
        if v == float("inf"):
            return "—"
        if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
            return f"{v:.2e}"
        return f"{v:.3g}"
    return str(v)


def _scalar_cols(rows: list[dict]) -> list[str]:
    return [k for k in rows[0] if k not in _ARRAY_KEYS]


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no results)_"
    cols = _scalar_cols(rows)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def _palette(n: int) -> list:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _annotate(ax, xs, ys, labels) -> None:
    for x, y, lab in zip(xs, ys, labels, strict=True):
        ax.annotate(
            lab,
            (x, y),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            fontweight="bold",
        )


def plot_controllers(rows: list[dict], path: Path) -> Path:
    names = [r["controller"] for r in rows]
    colors = dict(zip(names, _palette(len(names)), strict=True))
    eps = 1e-4  # mm floor so log axes behave near machine-precision residuals

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (A) convergence curves — the headline research plot
    ax = axes[0, 0]
    for r in rows:
        t = np.asarray(r.get("curve_t", []))
        e = np.maximum(np.asarray(r.get("curve_err_mm", [])), eps)
        if t.size:
            ax.semilogy(t, e, label=r["controller"], color=colors[r["controller"]], lw=1.6)
    ax.set(title="Convergence — EE error vs time", xlabel="time (s)", ylabel="EE error (mm)")
    ax.legend(ncol=2, loc="upper right")

    # (B) accuracy vs control effort (lower-left is better)
    ax = axes[0, 1]
    xs = [max(r["final_err_mm"], eps) for r in rows]
    ys = [r["effort"] for r in rows]
    for n, x, y in zip(names, xs, ys, strict=True):
        ax.scatter(x, y, s=70, color=colors[n], edgecolor="k", linewidth=0.5, zorder=3)
    _annotate(ax, xs, ys, names)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        title="Accuracy vs effort",
        xlabel="final EE error (mm, log)",
        ylabel="control effort ‖τ‖² (log)",
    )

    # (C) settle time
    ax = axes[1, 0]
    order = sorted(range(len(rows)), key=lambda i: rows[i]["settle_s"])
    ax.bar(
        [names[i] for i in order],
        [rows[i]["settle_s"] for i in order],
        color=[colors[names[i]] for i in order],
    )
    ax.set(title="Settle time — lower better", ylabel="settle time (s)")
    ax.tick_params(axis="x", rotation=45)

    # (D) compute per step
    ax = axes[1, 1]
    order = sorted(range(len(rows)), key=lambda i: rows[i]["compute_ms"])
    ax.bar(
        [names[i] for i in order],
        [rows[i]["compute_ms"] for i in order],
        color=[colors[names[i]] for i in order],
    )
    ax.set_yscale("log")
    ax.set(title="Compute per control step — lower better", ylabel="time (ms, log)")
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Controller benchmark — UR5e reach, tuned gains", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_planners(rows: list[dict], path: Path) -> Path:
    names = [r["planner"] for r in rows]
    colors = dict(zip(names, _palette(len(names)), strict=True))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.bar(names, [r["success_rate"] for r in rows], color=[colors[n] for n in names])
    ax.set(title="Success rate — higher better", ylabel="fraction solved", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    ax.bar(names, [r["plan_time_s"] for r in rows], color=[colors[n] for n in names])
    ax.set_yscale("log")
    ax.set(title="Plan time — lower better", ylabel="time (s, log)")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 0]
    ax.bar(names, [r["path_len_rad"] for r in rows], color=[colors[n] for n in names])
    ax.set(title="Path length — lower better", ylabel="joint path length (rad)")
    ax.tick_params(axis="x", rotation=30)

    # (D) speed vs optimality trade-off
    ax = axes[1, 1]
    xs = [r["plan_time_s"] for r in rows]
    ys = [r["path_len_rad"] for r in rows]
    for n, x, y in zip(names, xs, ys, strict=True):
        ax.scatter(x, y, s=90, color=colors[n], edgecolor="k", linewidth=0.5, zorder=3)
    _annotate(ax, xs, ys, names)
    ax.set_xscale("log")
    ax.set(title="Speed vs optimality", xlabel="plan time (s, log)", ylabel="path length (rad)")

    fig.suptitle("Planner benchmark — blocked obstacle query", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(controller_rows: list[dict], planner_rows: list[dict], outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # JSON keeps the scalar metrics only (curves are large and plot-only).
    def _scalars(rows):
        return [{k: v for k, v in r.items() if k not in _ARRAY_KEYS} for r in rows]

    (outdir / "results.json").write_text(
        json.dumps(
            {"controllers": _scalars(controller_rows), "planners": _scalars(planner_rows)}, indent=2
        ),
        encoding="utf-8",
    )
    if controller_rows:
        plot_controllers(controller_rows, outdir / "controllers.png")
    if planner_rows:
        plot_planners(planner_rows, outdir / "planners.png")

    md = ["# manipdyn benchmark results", ""]
    if controller_rows:
        md += [
            "## Controllers",
            "",
            "Reach scenarios on `scene_base`, tuned gains, scored by end-effector"
            " error toward the goal.",
            "",
            markdown_table(controller_rows),
            "",
            "![controllers](controllers.png)",
            "",
        ]
    if planner_rows:
        md += [
            "## Planners",
            "",
            "Start→goal queries in an obstacle scene, averaged over seeds.",
            "",
            markdown_table(planner_rows),
            "",
            "![planners](planners.png)",
            "",
        ]
    report = outdir / "BENCHMARK.md"
    report.write_text("\n".join(md), encoding="utf-8")
    return report
