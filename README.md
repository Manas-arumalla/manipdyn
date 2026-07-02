<div align="center">

# 🦾 manipdyn — 6-DOF Manipulator Planning & Control Lab

**A MuJoCo-physics lab for the UR5e: a benchmarked zoo of 8 controllers and 5 motion planners, with trajectory optimization, automatic gain tuning, a reinforcement-learning baseline, an interactive control center, and a full pick-and-place — all behind one clean, tested, documented package.**

[![CI](https://github.com/Manas-arumalla/manipdyn/actions/workflows/ci.yml/badge.svg)](https://github.com/Manas-arumalla/manipdyn/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Lint](https://img.shields.io/badge/lint-ruff-purple.svg)

<img src="manipdyn/media/pick_place.gif" width="58%" alt="pick and place"/>
<br/>
<img src="manipdyn/media/reach_osc.gif" width="38%" alt="operational-space reach"/>
<img src="manipdyn/media/obstacle_avoidance.gif" width="38%" alt="RRT-Connect obstacle avoidance"/>

<sub>Pick-and-place (top); operational-space reach and RRT-Connect obstacle avoidance (bottom) — all rendered headlessly by the library.</sub>

</div>

---

> **Related:** [openarm-control](https://github.com/Manas-arumalla/openarm-control) — the bimanual follow-up: these manipulation methods scaled to a 2×7-DOF OpenArm v2 platform with learned policies.

## Why I built this

I wanted one place to implement the classical and modern manipulator methods
behind a single interface and compare them fairly — on identical, reproducible,
auto-tuned scenarios — instead of judging them one demo at a time. So I built a
benchmark that puts every controller and planner on the same bench and reports
the numbers.

## Highlights

| Area | What's included |
|------|-----------------|
| **Control (8)** | PID · Computed-Torque · LQR · **iLQR** · Cartesian Impedance · OSC · **TSID (constrained QP)** · **MPPI** |
| **Planning (5)** | RRT · **RRT-Connect** · RRT\* · **Informed RRT\*** · PRM (+ collision checking, shortcut & B-spline smoothing) |
| **Optimization** | iLQR trajectory optimization · time-optimal path parameterization (TOPP) · **black-box controller auto-tuning** |
| **Learning** | Gymnasium reach envs + an **SAC** baseline (state- and vision-conditioned), scored against the classical controllers |
| **Perception** | a simulated **RGB-D camera** → point cloud → object-pose estimate that drives the grasp from vision, not ground-truth state; multi-object/clutter |
| **Parametric** | procedural scene generation with **MjSpec** (random clutter, deterministic per seed); a **RobotSpec** de-hardcodes the arm |
| **Multi-robot** | two UR5e arms in **one shared sim**, independently controlled; a two-arm **cube handover** |
| **Benchmark** | one command → metrics table + comparison plots, with fair auto-tuned gains |
| **Tasks** | a complete **pick-and-place** (grasp → base-rotation carry → stable place) |
| **Task planning** | a symbolic **STRIPS** layer turning declarative goals into pick/place/stack sequences |
| **GUI** | a mode-based PySide6 control center — Watch Sim (interactive MuJoCo viewer + live telemetry) or Run Sim (headless results) across every mode |
| **Engineering** | installable package · typed interfaces · `pytest` suite · headless rendering · ruff · GitHub Actions CI |

## Results

Every method is scored on the **same** auto-tuned scenarios, so the comparison is
fair. Regenerate the whole table with **`manipdyn bench`**.

**Controllers** — three reach targets on `scene_base`, tuned gains, scored by
end-effector error:

| controller | success | final err | settle | RMSE¹ | effort ‖τ‖² | compute |
|------------|:-------:|----------:|-------:|------:|------------:|--------:|
| computed-torque | 3/3 | **8e-13 mm** | 0.23 s | 72 mm | 6.0e3 | 0.013 ms |
| lqr | 3/3 | 2e-8 mm | 0.34 s | 80 mm | 2.2e3 | 0.015 ms |
| osc | 3/3 | 0.008 mm | **0.18 s** | 67 mm | 6.9e3 | 0.054 ms |
| tsid | 3/3 | 0.025 mm | 0.20 s | **67 mm** | 2.3e3 | 1.41 ms |
| ilqr | 3/3 | 0.011 mm | 0.29 s | 75 mm | 2.2e3 | 0.13 ms |
| pid | 3/3 | 0.23 mm | 0.26 s | 74 mm | 6.9e3 | **0.008 ms** |
| impedance | 3/3 | 2.9 mm | 0.55 s | 71 mm | 4.2e3 | 0.013 ms |
| mppi | 2/3 | 13.2 mm | 2.12 s | 150 mm | **1.8e3** | 26.3 ms |

**Planners** — a blocked start/goal query on `scene_obstacle` (straight-line
motion collides); every planner finds a collision-free detour:

| planner | success | plan time | path length |
|---------|:-------:|----------:|------------:|
| RRT | 5/5 | **12 ms** | 1.61 rad |
| RRT-Connect | 5/5 | 17 ms | 1.69 rad |
| RRT\* | 5/5 | 12.9 s | 1.57 rad |
| Informed RRT\* | 5/5 | 24.5 s | **1.42 rad** |
| PRM | 5/5 | 6.7 s | 5.32 rad |

<sub>¹ RMSE is taken over the whole reach, so it reflects the approach transient,
not the steady-state error (the "final err" column).</sub>

## Quickstart

```bash
cd manipdyn
pip install -e ".[gui,rl]"          # core + GUI + reinforcement learning

manipdyn bench                      # run the benchmark -> benchmarks/results/
manipdyn gui                        # launch the control center
python scripts/make_pick_place.py   # render the pick-and-place demo
```

```python
import numpy as np
from manipdyn.sim import World
from manipdyn.control import Target
from manipdyn.tuning import tuned_controller

world = World(scene="scene_base")
ctrl = tuned_controller("ctc", world)          # computed-torque, tuned gains
goal = np.array([1.0, -1.1, 1.2, -1.6, -1.4, 0.4])
for _ in range(1500):
    world.step(ctrl.compute(Target(q=goal)))
print("final joint error:", np.linalg.norm(goal - world.qpos_arm))
```

## Benchmark plots

The same comparison, plotted (regenerate with `manipdyn bench`):

<img src="manipdyn/benchmarks/results/controllers.png" width="80%" alt="controller benchmark"/>

Auto-tuning (global search + Nelder-Mead polish) cut controller cost **41–65%**
vs. hand-picked defaults. The SAC policy reaches **80%** of random goals to
within 3 cm on the same physics. Full tables and the math behind every method
live in [`manipdyn/docs/`](manipdyn/docs/).

Every method is also shown *running*, not just charted — all 8 controllers
reaching and all 5 planners avoiding, side by side
([`manipdyn/docs/theory/benchmark.md`](manipdyn/docs/theory/benchmark.md)).

## Analysis

Beyond the headline tables, each method is studied in depth — full write-up in
[`manipdyn/docs/analysis.md`](manipdyn/docs/analysis.md).

**Controller scorecard** — eight controllers across five criteria, green = best in each column. No method wins everything:

<img src="manipdyn/benchmarks/results/controller_heatmap.png" width="78%" alt="controller scorecard heatmap"/>

**Planner routes in the workspace** — the same blocked query solved by all five planners. RRT / RRT-Connect / RRT\* / Informed take a tight detour past the pillar; PRM's uniform roadmap loops wide and high:

<img src="manipdyn/benchmarks/results/planner_paths.png" width="92%" alt="planner end-effector paths, top and side views"/>

**Time-optimal trajectories (TOPP)** — joint velocities and accelerations ride their limits (dashed), the bang-bang structure that makes the timing optimal:

<img src="manipdyn/benchmarks/results/traj_profiles.png" width="78%" alt="time-optimal velocity and acceleration profiles"/>

**Auto-tuning convergence** — best-so-far cost under a global search, then a bounded Nelder-Mead polish (dashed line):

<img src="manipdyn/benchmarks/results/tuning_convergence.png" width="68%" alt="auto-tuning convergence curves"/>

## Control center

A mode-based PySide6 desktop app drives the whole lab interactively — pick a
**mode** (Reach · Obstacle Avoidance · Pick & Place · RL Reach · Benchmark),
configure it, then **Watch Sim** (opens the interactive MuJoCo viewer + live
telemetry) or **Run Sim** (headless evaluation → results & plots).

<img src="manipdyn/media/gui.gif" width="80%" alt="interactive control center"/>

## Repository layout

```
manipdyn/            the package — planning, control, benchmark, GUI, docs, demos
code/                an earlier pure-NumPy trajectory simulator
Manipulator Test/    an earlier MuJoCo prototype
```

Full package documentation: [`manipdyn/README.md`](manipdyn/README.md).

## Attribution & license

`manipdyn` source is **MIT** (see [LICENSE](LICENSE)). The UR5e model is derived
from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
/ ROS-Industrial UR5e description and is **BSD-3-Clause** (see its bundled
`LICENSE` under `manipdyn/src/manipdyn/models/ur5e_model/`).
