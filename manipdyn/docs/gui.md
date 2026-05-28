# Control center (GUI)

A mode-based PySide6 desktop app for driving the whole lab interactively.

```bash
pip install -e "./manipdyn[gui]"
manipdyn gui
```

![control center](../media/gui.png)

## Modes

Pick a **mode**; the configuration panel changes to match it:

| Mode | What it does | Configure |
|------|--------------|-----------|
| **Reach** | a controller drives the EE to a Cartesian target | controller, tuned/manual gains, target (x, y, z) |
| **Obstacle Avoidance** | a planner finds a detour around the pillar, tracked by computed-torque control | planner |
| **Pick & Place** | the full cube-on-table task (top-down grasp → base-rotation carry → place) | — |
| **RL Reach** | the learned SAC policy reaches a sampled goal | goal seed |
| **Benchmark** | scores every controller and planner, writes tables + plots | which / duration / trials |

## Two ways to run — consistent across modes

* **Watch Sim** opens the **interactive MuJoCo viewer** (orbit / zoom the real
  3D scene) *and* an embedded live view, and streams live telemetry — the
  stat cards (time, metric, phase) and the real-time error plot — as it runs.
* **Run Sim** runs the same scenario **headless** in a background thread,
  faster than real time, and reports results/plots (no viewer). Benchmark mode
  is Run-only and additionally writes `benchmarks/results/`.

Physics is stepped on the main thread (so the MuJoCo GL context is never touched
cross-thread), while the heavy setup — IK, planning, grasp solving, policy
loading — runs in a worker so the UI never freezes.

## Architecture

The GUI imports the `manipdyn` library directly: each mode is a small *engine*
that builds its scenario in `prepare()` and advances one control step per
`step()`, returning telemetry. Both the live (`QTimer` + viewer) and headless
(worker thread) drivers consume that same interface, and the pick-and-place
engine reuses the exact pipeline in `manipdyn.tasks.pick_place` that the
headless demo renders — so the GUI never drifts from the rest of the package.
