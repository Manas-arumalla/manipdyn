"""Perception-driven pick-and-place demo GIF: what the robot sees, and what it does.

Renders the pick-and-place side by side — the scene on the left, the overhead
RGB-D camera view on the right — with a red marker at the pose estimated from
vision. The grasp is driven by that estimate (not ground-truth state), so the
arm is seen closing exactly on the perceived point.

Run from the manipdyn/ directory:
    python scripts/make_perception_demo.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from manipdyn.perception import Camera, sense_object_pose
from manipdyn.render.recorder import save_gif
from manipdyn.sim import World
from manipdyn.tasks import pick_place
from manipdyn.tasks.pick_place import run, solve

MEDIA = Path(__file__).resolve().parents[1] / "media"
H = 340
CAPTURE_EVERY = 52


def main() -> None:
    world = World(scene=pick_place.SCENE, ee_site=pick_place.EE_SITE)

    # Park the arm clear of the cube for an unobstructed look, then sense.
    look = world.home_qpos_arm.copy()
    look[1] -= 0.5
    world.reset(look)
    world.forward()
    overhead = Camera(world, "overhead", width=480, height=H)
    est = sense_object_pose(overhead, segmentation=True)

    # Show the perceived grasp point as a red marker (visible in both views).
    sid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
    world.model.site_pos[sid] = [est.top_xy[0], est.top_xy[1], est.top_z]
    world.model.site_rgba[sid] = [1.0, 0.1, 0.1, 0.9]
    world.model.site_size[sid] = [0.012, 0.0, 0.0]

    # Left: framed scene view. Right: the overhead camera the estimate came from.
    scene = mujoco.Renderer(world.model, height=H, width=560)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (-0.18, -0.31, 0.42)
    cam.distance = 1.45
    cam.azimuth = 215
    cam.elevation = -18

    plan = solve(world, object_xy=est.top_xy)
    frames = []
    last = None
    for i, info in enumerate(run(world, plan)):
        last = info
        if i % CAPTURE_EVERY == 0:
            scene.update_scene(world.data, camera=cam)
            left = scene.render()
            right = overhead.rgb()
            frames.append(np.hstack([left, right]))
    scene.close()
    overhead.close()

    gif = save_gif(frames, MEDIA / "perception.gif", fps=16, palettesize=32, max_width=560)
    print(
        f"frames={len(frames)}  final: cube_z={last['cube_pos'][2]:.3f} "
        f"tilt={last['cube_tilt_deg']:.1f} place_err={last['place_err_mm']:.1f}mm"
    )
    print(f"GIF saved: {gif}")


if __name__ == "__main__":
    main()
