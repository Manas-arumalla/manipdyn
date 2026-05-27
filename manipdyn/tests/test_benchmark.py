"""The benchmark harness runs and produces a well-formed report."""

from __future__ import annotations

from manipdyn.benchmark import benchmark_controllers, benchmark_planners
from manipdyn.benchmark.report import markdown_table, write_report

CONTROLLER_KEYS = {
    "controller",
    "success_rate",
    "final_err_mm",
    "settle_s",
    "rmse_mm",
    "effort",
    "peak_torque_nm",
    "compute_ms",
}
PLANNER_KEYS = {
    "planner",
    "success_rate",
    "plan_time_s",
    "path_len_rad",
    "raw_nodes",
    "collision_free",
}


def test_benchmark_controllers_smoke():
    rows = benchmark_controllers(controllers=["pid", "lqr"], duration=1.0)
    assert {r["controller"] for r in rows} == {"pid", "lqr"}
    for r in rows:
        assert CONTROLLER_KEYS <= set(r)
        assert 0.0 <= r["success_rate"] <= 1.0


def test_benchmark_planners_smoke():
    rows = benchmark_planners(planners=["rrt_connect"], n_trials=1)
    assert rows[0]["planner"] == "rrt_connect"
    assert PLANNER_KEYS <= set(rows[0])


def test_write_report_creates_artifacts(tmp_path):
    rows_c = benchmark_controllers(controllers=["pid"], duration=1.0)
    rows_p = benchmark_planners(planners=["rrt_connect"], n_trials=1)
    report = write_report(rows_c, rows_p, tmp_path)
    assert report.exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "controllers.png").exists()
    assert (tmp_path / "planners.png").exists()
    assert "controller" in markdown_table(rows_c)
