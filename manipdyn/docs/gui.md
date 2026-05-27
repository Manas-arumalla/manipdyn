# Control center (GUI)

A PySide6 desktop app for driving the lab interactively.

```bash
pip install -e "./manipdyn[gui]"
manipdyn gui
```

![control center](../media/gui.png)

## What it does

* **Embedded live 3D view** — the actual MuJoCo scene rendered in-window as the
  arm moves (a `QTimer` steps physics on the main thread and streams offscreen
  frames, so the GL context is never touched cross-thread and the UI stays
  responsive).
* **Scene picker** — `scene_base`, `scene` (with obstacle), `scene_base_gripper`.
* **Every controller** — all 8 in a dropdown, with **per-controller gain
  fields** auto-populated from each controller's tuning spec (and pre-filled
  with the *tuned* gains when "Use tuned gains" is checked).
* **Planner integration** — pick RRT / RRT-Connect / RRT\* / Informed RRT\* /
  PRM for joint-space controllers; the chosen planner finds a path, which is
  shortcut and time-parameterized, then tracked. Falls back to a direct move if
  no path is found.
* **Cartesian target** — type an (x, y, z) goal; IK supplies the joint goal for
  joint-space controllers, and the red marker shows the target.
* **Live telemetry** — a real-time end-effector-error plot.
* **One-click benchmark** — runs the controller benchmark in a background
  thread and shows the results table.

## Architecture

Unlike the v1 prototype (which wrote a JSON config and `subprocess`-launched a
separate script from a fixed directory), this GUI **imports the `manipdyn`
library directly**. It builds a `World`, a `Controller`, and (optionally) a
planned trajectory in-process, so it is robust to the working directory, easy
to extend, and shares exactly the same code paths the benchmark uses.
