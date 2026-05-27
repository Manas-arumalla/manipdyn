"""Generate the project's demo GIFs (headless, offscreen-rendered).

Produces, into ``manipdyn/media/``:
  * ``reach_osc.gif``           — operational-space control reaching a target
  * ``obstacle_avoidance.gif``  — RRT-Connect plan, time-parameterized, tracked
  * ``pick_place.gif``          — gripper pick-and-place (see make_pick_place.py)

Run (from manipdyn/):
    python scripts/make_demos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipdyn.control import Target
from manipdyn.planning import RRTConnect, shortcut_path
from manipdyn.render import Recorder
from manipdyn.sim import World
from manipdyn.trajectory import parameterize_time_optimal
from manipdyn.tuning import tuned_controller

MEDIA = Path(__file__).resolve().parents[1] / "media"


def _fk(world: World, q: np.ndarray) -> np.ndarray:
    world.set_arm_qpos(q)
    world.forward()
    return world.ee_pos.copy()


def _interp(timed, t: float) -> tuple[np.ndarray, np.ndarray]:
    n = timed.q.shape[1]
    q = np.array([np.interp(t, timed.t, timed.q[:, j]) for j in range(n)])
    qd = np.array([np.interp(t, timed.t, timed.qd[:, j]) for j in range(n)])
    return q, qd


def demo_reach() -> Path:
    world = World(scene="scene_base")
    world.reset(world.home_qpos_arm)
    q_goal = np.array([1.1, -1.0, 1.1, -1.6, -1.4, 0.5])
    x_goal = _fk(world, q_goal)
    world.reset(world.home_qpos_arm)
    world.set_target_marker(x_goal)

    controller = tuned_controller("osc", world)
    controller.reset()
    target = Target(x=x_goal)

    cam = dict(lookat=(0.15, -0.1, 0.4), distance=1.7, azimuth=140, elevation=-22)
    with Recorder(world, width=480, height=380, fps=25, **cam) as rec:
        for i in range(int(3.5 / world.timestep)):
            world.step(controller.compute(target))
            if i % 8 == 0:
                rec.capture()
        return rec.save_gif(MEDIA / "reach_osc.gif", palettesize=48, max_width=380)


def demo_obstacle() -> Path:
    # A pillar stands directly in the straight-line swing between start and goal,
    # so the planner has to lift the end-effector up and over it.
    world = World(scene="scene_obstacle")
    q_start = np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0])
    q_goal = np.array([-1.4, -1.2, 1.4, -1.7, -1.57, 0.0])

    planner = RRTConnect(world, seed=3, max_iter=8000)
    path = planner.plan(q_start, q_goal)
    if path is None:
        raise RuntimeError("planner failed for the obstacle demo")
    path = shortcut_path(path, planner.checker, iterations=150, seed=3)

    timed = parameterize_time_optimal(path, np.full(6, 1.2), np.full(6, 2.5), n_samples=200)

    world.reset(q_start)
    controller = tuned_controller("ctc", world)
    controller.reset()

    end = timed.duration + 1.0
    cam = dict(lookat=(-0.35, 0.25, 0.35), distance=2.0, azimuth=55, elevation=-25)
    with Recorder(world, width=480, height=380, fps=25, **cam) as rec:
        i = 0
        while world.time <= end:
            q_ref, qd_ref = _interp(timed, min(world.time, timed.duration))
            world.step(controller.compute(Target(q=q_ref, v=qd_ref)))
            if i % 8 == 0:
                rec.capture()
            i += 1
        return rec.save_gif(MEDIA / "obstacle_avoidance.gif", palettesize=48, max_width=380)


def main() -> None:
    print("reach:    ", demo_reach())
    print("obstacle: ", demo_obstacle())


if __name__ == "__main__":
    main()
