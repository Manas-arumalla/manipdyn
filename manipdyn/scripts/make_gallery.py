"""Render method-comparison galleries (headless, offscreen).

Every benchmarked method gets a visible simulation, not just a number:

  * ``controllers.gif`` — all 8 controllers reaching the *same* target on
    ``scene_base``, side by side in a labeled grid.
  * ``planners.gif``    — all 5 planners' paths executed around the obstacle
    pillar (each plan is time-parameterized and tracked by computed-torque
    control), side by side in a labeled grid.

Run from the manipdyn/ directory:
    python scripts/make_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manipdyn.control import CONTROLLERS, ComputedTorqueController, Target
from manipdyn.planning import PLANNERS, shortcut_path
from manipdyn.render.recorder import save_gif
from manipdyn.sim import World
from manipdyn.trajectory import parameterize_time_optimal
from manipdyn.tuning import tuned_controller

MEDIA = Path(__file__).resolve().parents[1] / "media"
PANEL_W, PANEL_H = 300, 230


def _font(size: int):
    for name in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


_FONT = _font(20)


def _camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.1, -0.05, 0.45)
    cam.distance = 1.95
    cam.azimuth = 140
    cam.elevation = -20
    return cam


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((9, 7), text, font=_FONT, fill=(0, 0, 0))  # shadow
    draw.text((8, 6), text, font=_FONT, fill=(255, 220, 90))
    return np.asarray(img)


def _grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    rows = (len(panels) + cols - 1) // cols
    h, w, _ = panels[0].shape
    canvas = np.full((rows * h, cols * w, 3), 24, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = p
    return canvas


def controllers_gallery() -> Path:
    names = list(CONTROLLERS)
    q_goal = np.array([1.0, -1.1, 1.2, -1.6, -1.4, 0.4])

    # the Cartesian goal is the forward kinematics of q_goal — one target shared
    # by every panel (joint-space and task-space controllers both use it).
    probe = World(scene="scene_base")
    probe.set_arm_qpos(q_goal)
    probe.forward()
    x_goal = probe.ee_pos.copy()
    target = Target(q=q_goal, x=x_goal)

    worlds, ctrls, renderers = [], [], []
    for name in names:
        w = World(scene="scene_base")
        w.reset(w.home_qpos_arm)
        w.set_target_marker(x_goal)
        worlds.append(w)
        ctrls.append(tuned_controller(name, w))
        renderers.append(mujoco.Renderer(w.model, height=PANEL_H, width=PANEL_W))

    cam = _camera()
    frames = []
    for step in range(1300):
        for w, c in zip(worlds, ctrls, strict=True):
            w.step(c.compute(target))
        if step % 20 == 0:
            panels = []
            for w, r, name in zip(worlds, renderers, names, strict=True):
                r.update_scene(w.data, camera=cam)
                panels.append(_label(r.render(), name.upper()))
            frames.append(_grid(panels, cols=4))
    for r in renderers:
        r.close()
    return save_gif(frames, MEDIA / "controllers.gif", fps=18, palettesize=48, max_width=880)


def planners_gallery() -> Path:
    names = list(PLANNERS)
    q_start = np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0])
    q_goal = np.array([-1.4, -1.2, 1.4, -1.7, -1.57, 0.0])
    cam = _camera()
    cam.lookat[:] = (-0.35, 0.25, 0.35)
    cam.distance = 2.1
    cam.azimuth = 55

    # budgets large enough that every planner solves the blocked query at seed 0
    budgets = {
        "rrt": {"max_iter": 6000, "goal_bias": 0.2},
        "rrt_connect": {"max_iter": 5000},
        "rrt_star": {"max_iter": 3000, "goal_bias": 0.2},
        "informed_rrt_star": {"max_iter": 3000, "goal_bias": 0.2},
        "prm": {"n_samples": 500, "k_neighbors": 15},
    }
    worlds, trajs, ctrls, renderers = [], [], [], []
    for name in names:
        w = World(scene="scene_obstacle")
        planner = PLANNERS[name](w, seed=0, **budgets.get(name, {}))
        path = planner.plan(q_start, q_goal)
        if path is None:
            raise RuntimeError(f"{name} found no path")
        path = shortcut_path(path, planner.checker, iterations=150, seed=0)
        trajs.append(
            parameterize_time_optimal(path, np.full(6, 1.2), np.full(6, 2.5), n_samples=200)
        )
        w.reset(q_start)
        worlds.append(w)
        ctrls.append(ComputedTorqueController(w, kp=600, kd=50))
        renderers.append(mujoco.Renderer(w.model, height=PANEL_H, width=PANEL_W))

    end = max(t.duration for t in trajs) + 0.6
    frames, step = [], 0
    while worlds[0].time <= end:
        for w, t, c in zip(worlds, trajs, ctrls, strict=True):
            tt = min(w.time, t.duration)
            q = np.array([np.interp(tt, t.t, t.q[:, j]) for j in range(6)])
            v = np.array([np.interp(tt, t.t, t.qd[:, j]) for j in range(6)])
            w.step(c.compute(Target(q=q, v=v)))
        if step % 22 == 0:
            panels = []
            for w, r, name in zip(worlds, renderers, names, strict=True):
                r.update_scene(w.data, camera=cam)
                panels.append(_label(r.render(), name.replace("_", "-").upper()))
            frames.append(_grid(panels, cols=5))
        step += 1
    for r in renderers:
        r.close()
    return save_gif(frames, MEDIA / "planners.gif", fps=18, palettesize=48, max_width=1000)


def main() -> None:
    print("controllers:", controllers_gallery())
    print("planners:   ", planners_gallery())


if __name__ == "__main__":
    main()
