"""M1 end-to-end demo: drive the UR5e to a target with PID, headless.

Loads a scene, regulates the arm to a target joint configuration using the
PID baseline (+ model gravity compensation), records the motion to a GIF, and
prints tracking metrics. No interactive viewer required.

Run:
    python manipdyn/scripts/demo_headless.py
or, after `pip install -e ./manipdyn`:
    python -m manipdyn.scripts.demo_headless
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipdyn.control import PIDController, Target
from manipdyn.render import Recorder
from manipdyn.sim import World


def main() -> None:
    world = World(scene="scene_base")
    print(world)

    q_start = world.home_qpos_arm.copy()
    q_target = np.array([1.2, -1.0, 1.2, -1.6, -1.4, 0.5])

    world.reset(q_start)
    world.set_target_marker(_fk_site(world, q_target))

    controller = PIDController(world, kp=300.0, ki=8.0, kd=60.0)
    controller.reset()
    target = Target(q=q_target)

    duration = 4.0
    n_steps = int(duration / world.timestep)
    capture_every = max(1, int((1.0 / 30.0) / world.timestep))

    errors = []
    media = Path(__file__).resolve().parents[1] / "media"
    with Recorder(world, width=640, height=480, fps=30) as rec:
        for i in range(n_steps):
            world.step(controller.compute(target))
            errors.append(float(np.linalg.norm(q_target - world.qpos_arm)))
            if i % capture_every == 0:
                rec.capture()
        gif_path = rec.save_gif(media / "m1_pid_track.gif")

    errors = np.array(errors)
    print("\n--- PID tracking (joint-space regulation) ---")
    print(f"  steps:        {n_steps}  ({duration}s @ dt={world.timestep})")
    print(f"  final error:  {errors[-1]:.4f} rad")
    print(f"  RMSE:         {np.sqrt(np.mean(errors**2)):.4f} rad")
    print(f"  settled <0.05 rad at: {_settle_time(errors, world.timestep, 0.05):.2f} s")
    print(f"  GIF saved:    {gif_path}")


def _fk_site(world: World, q_arm: np.ndarray) -> np.ndarray:
    """Forward-kinematics the EE site for a given arm config (non-destructive)."""
    saved = world.qpos_arm
    world.set_arm_qpos(q_arm)
    world.forward()
    pos = world.ee_pos
    world.set_arm_qpos(saved)
    world.forward()
    return pos


def _settle_time(errors: np.ndarray, dt: float, tol: float) -> float:
    below = np.where(errors < tol)[0]
    return float(below[0] * dt) if below.size else float("nan")


if __name__ == "__main__":
    main()
