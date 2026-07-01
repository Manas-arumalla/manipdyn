"""Perception demo: show what the arm *sees* and the pose it estimates.

Renders the fixed overhead RGB-D camera and the eye-in-hand wrist camera on the
pick scene, deprojects the depth into a world-frame point cloud, and estimates
the cube's top-down grasp pose from vision alone (no privileged simulator pose).
Saves a four-panel figure.

Run from the manipdyn/ directory:
    python scripts/make_perception.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from manipdyn.perception import (
    Camera,
    deproject,
    object_geom_ids,
    segment_mask,
    sense_object_pose,
)
from manipdyn.sim import World

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "perception.png"


def main() -> None:
    world = World(scene="scene_pick", ee_site="pinch")
    # Park the arm clear of the cube so the overhead view is unobstructed
    # (what you'd do in practice: look first, then move in).
    look = world.home_qpos_arm.copy()
    look[1] -= 0.5  # lift the shoulder
    world.reset(look)
    world.forward()

    oid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "object")
    true = world.data.xpos[oid]

    overhead = Camera(world, "overhead", width=640, height=480)
    wrist = Camera(world, "wrist", width=480, height=480)

    rgb = overhead.rgb()
    depth = overhead.depth()
    wrist_rgb = wrist.rgb()
    mask = segment_mask(overhead.segmentation(), object_geom_ids(world))
    cloud = deproject(depth, overhead.intrinsics, overhead.extrinsics, mask=mask)
    est = sense_object_pose(overhead, segmentation=True)
    overhead.close()
    wrist.close()

    err_mm = float(np.linalg.norm(est.top_xy - true[:2]) * 1e3)
    print(
        f"estimated top_xy={np.round(est.top_xy, 4)} true={np.round(true[:2], 4)} "
        f"err={err_mm:.1f} mm  top_z={est.top_z:.3f}  points={est.n_points}"
    )

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    ax[0].imshow(rgb)
    ax[0].set_title("overhead RGB")
    d = np.where(depth > 0, depth, np.nan)
    im = ax[1].imshow(d, cmap="viridis")
    ax[1].set_title("overhead depth (m)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[2].imshow(wrist_rgb)
    ax[2].set_title("eye-in-hand (wrist)")
    for a in ax[:3]:
        a.set_xticks([])
        a.set_yticks([])

    sc = ax[3].scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], s=6, cmap="plasma")
    ax[3].scatter(*true[:2], marker="o", s=140, facecolors="none", edgecolors="k", label="true")
    ax[3].scatter(*est.top_xy, marker="x", s=140, c="red", label=f"estimate ({err_mm:.1f} mm)")
    ax[3].set_title("segmented point cloud (top view)")
    ax[3].set_xlabel("x (m)")
    ax[3].set_ylabel("y (m)")
    ax[3].set_aspect("equal")
    ax[3].legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=ax[3], fraction=0.046, pad=0.04, label="height (m)")

    fig.suptitle(
        "Perception: object pose estimated from a simulated RGB-D camera "
        "(drives the grasp instead of ground-truth pose)",
        fontsize=12,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"figure saved: {OUT}")


if __name__ == "__main__":
    main()
