"""Turn benchmark results into a Markdown report and comparison plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402


def _fmt(v: float) -> str:
    if isinstance(v, float):
        if v == float("inf"):
            return "—"
        if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
            return f"{v:.2e}"
        return f"{v:.3g}"
    return str(v)


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no results)_"
    cols = list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def _bar(ax, rows, key, label, key_name="controller", log=False):
    names = [r[key_name] for r in rows]
    vals = [r[key] if r[key] != float("inf") else 0.0 for r in rows]
    ax.bar(names, vals, color="#3a7ca5")
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)


def plot_controllers(rows: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _bar(axes[0, 0], rows, "final_err_mm", "Final EE error (mm) — lower better")
    _bar(axes[0, 1], rows, "settle_s", "Settle time (s) — lower better")
    _bar(axes[1, 0], rows, "effort", "Control effort  mean‖τ‖² — lower better", log=True)
    _bar(axes[1, 1], rows, "compute_ms", "Compute / step (ms) — lower better", log=True)
    fig.suptitle("Controller benchmark (tuned gains, reach scenarios)", fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_planners(rows: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _bar(axes[0], rows, "success_rate", "Success rate — higher better", key_name="planner")
    _bar(axes[1], rows, "plan_time_s", "Plan time (s) — lower better", key_name="planner", log=True)
    _bar(axes[2], rows, "path_len_rad", "Path length (rad) — lower better", key_name="planner")
    fig.suptitle("Planner benchmark (obstacle scene)", fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(controller_rows: list[dict], planner_rows: list[dict], outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "results.json").write_text(
        json.dumps({"controllers": controller_rows, "planners": planner_rows}, indent=2),
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
