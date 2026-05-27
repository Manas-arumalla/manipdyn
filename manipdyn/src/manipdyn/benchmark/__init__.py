"""The benchmark suite — the centerpiece of manipdyn.

Runs every controller and planner on common, reproducible scenarios and emits a
metrics table + comparison plots, so the project answers *which method wins,
and when* with data rather than adjectives. Controllers are instantiated with
their tuned gains (see :mod:`manipdyn.tuning`) for a fair comparison.
"""

from manipdyn.benchmark.harness import benchmark_controllers, benchmark_planners
from manipdyn.benchmark.metrics import ControllerRun, PlannerRun, rollout_controller
from manipdyn.benchmark.scenarios import reach_targets

__all__ = [
    "benchmark_controllers",
    "benchmark_planners",
    "rollout_controller",
    "ControllerRun",
    "PlannerRun",
    "reach_targets",
]
