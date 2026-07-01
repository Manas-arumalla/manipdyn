# manipdyn documentation

These pages document the theory, architecture, and usage of the package. They
render as a [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) site
(`mkdocs serve` from the package root, using `mkdocs.yml`) and are kept in
lock-step with each module.

## Theory notes

| Topic | Covers |
|-------|--------|
| [Control zoo](theory/control.md) | PID, Computed-Torque, LQR, Cartesian Impedance, OSC, MPPI — math, trade-offs, API |
| [Kinematics](theory/kinematics.md) | Damped least-squares inverse kinematics |
| [Planning](theory/planning.md) | RRT, RRT-Connect, RRT\*, Informed RRT\*, PRM; smoothing; collision |
| [Optimization & optimal control](theory/optimization.md) | iLQR, TSID (QP), time-optimal timing, controller auto-tuning |
| [Benchmark suite](theory/benchmark.md) | reproducible controller + planner comparison, metrics, results |
| [Deeper analysis](analysis.md) | trajectory profiles, planner workspace paths, tuning convergence, controller scorecard |
| [Control center (GUI)](gui.md) | mode-based PySide6 app — Watch Sim (interactive viewer + telemetry) or Run Sim (headless) |
| [Reinforcement learning](rl.md) | Gymnasium reach env + SAC baseline vs. classical controllers |
| [Tasks: pick-and-place](tasks.md) | the full grasp → carry → place pipeline |
| [Perception & vision](perception.md) | RGB-D camera → point cloud → object-pose estimate driving the grasp |

## Architecture (current)

```
World (MuJoCo wrapper) ── state, M(q), J, bias, render
   │
   ├── kinematics/  IKSolver           target_x  -> q
   ├── dynamics/    linearize()        (A, B) for LQR
   ├── trajopt/     ILQR               optimal torque trajectory + gains
   ├── trajectory/  parameterize_*     geometric path -> timed trajectory
   ├── control/     Controller(ABC)    Target    -> arm torque
   │                  pid · ctc · lqr · ilqr · impedance · osc · tsid · mppi
   ├── tuning/      tune_controller    optimize gains (also fair benchmarking)
   ├── planning/    Planner(ABC)       q_start, q_goal -> collision-free path
   │                  rrt · rrt_connect · rrt_star · informed_rrt_star · prm
   └── perception/  Camera             depth -> point cloud -> object pose
```

Everything is driven through small, typed interfaces (`Target`, `Controller`,
`Planner`) so the benchmark harness and GUI can swap methods freely.
