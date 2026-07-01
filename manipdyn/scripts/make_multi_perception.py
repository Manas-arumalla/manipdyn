"""Multi-object perception demo: detect several cubes and pick a target.

Renders the overhead RGB-D camera on the clutter scene, estimates the pose of
every cube from vision, and selects one target (here: the red cube) — the kind
of "find the right object among clutter" step that precedes a grasp. Saves a
two-panel figure (overhead RGB, and the labelled point cloud with the pick).

Run from the manipdyn/ directory:
    python scripts/make_multi_perception.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manipdyn.perception import Camera, deproject, sense_objects
from manipdyn.sim import World

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "perception_multi.png"


def main() -> None:
    world = World(scene="scene_clutter", ee_site="pinch")
    look = world.home_qpos_arm.copy()
    look[1] -= 0.5  # park the arm clear for an unobstructed look
    world.reset(look)
    world.forward()

    cam = Camera(world, "overhead", width=640, height=480)
    rgb = cam.rgb()
    cloud = deproject(cam.depth(), cam.intrinsics, cam.extrinsics, max_depth=1.1)
    objs = sense_objects(cam, segmentation=True)
    cam.close()

    target = next(o for o in objs if o.label == "cube_red")
    for o in objs:
        print(f"{o.label:11s} top_xy={np.round(o.top_xy, 3)} dims={np.round(o.dims, 3)}")
    print(f"selected target: {target.label} at {np.round(target.top_xy, 3)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].imshow(rgb)
    ax[0].set_title("overhead RGB (clutter)")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Table-top slice of the cloud for a clean top view.
    top = cloud[cloud[:, 2] > 0.30]
    ax[1].scatter(top[:, 0], top[:, 1], c="0.8", s=3)
    colors = {"cube_red": "#e0301e", "cube_green": "#2ca02c", "cube_blue": "#3457d5"}
    for o in objs:
        ax[1].scatter(
            *o.top_xy, s=90, c=colors.get(o.label, "k"), edgecolors="k", label=o.label, zorder=3
        )
    ax[1].scatter(
        *target.top_xy,
        s=260,
        facecolors="none",
        edgecolors="k",
        linewidths=2.0,
        label="target",
        zorder=4,
    )
    ax[1].set_title("estimated poses (target ringed)")
    ax[1].set_xlabel("x (m)")
    ax[1].set_ylabel("y (m)")
    ax[1].set_aspect("equal")
    ax[1].legend(loc="best", fontsize=8)

    fig.suptitle("Multi-object perception: estimate every cube, select a target", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"figure saved: {OUT}")


if __name__ == "__main__":
    main()
