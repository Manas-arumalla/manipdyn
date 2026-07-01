"""Benchmark: perception-driven vs oracle pick-and-place over random placements.

For each randomized cube position on the pick table, runs the full pick-and-place
twice — once from the ground-truth pose (oracle) and once from the pose estimated
by the overhead RGB-D camera (perception) — and reports grasp success, place
error, and the perception error that drives the gap. This is the "does vision
cost us anything?" axis of the lab's fair-comparison thesis.

Run from the manipdyn/ directory:
    python scripts/benchmark_perception.py --trials 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from manipdyn.perception import Camera, sense_object_pose
from manipdyn.sim import World
from manipdyn.tasks import pick_place
from manipdyn.tasks.pick_place import run, solve

OUTDIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
BASE_XY = np.array([-0.49, -0.13])


def _place_cube(world: World, x: float, y: float) -> np.ndarray:
    jid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    adr = world.model.jnt_qposadr[jid]
    world.model.qpos0[adr : adr + 2] = [x, y]  # survives reset() inside run()
    world.reset()
    world.forward()
    oid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "object")
    return world.data.xpos[oid][:2].copy()


def _rollout(world: World, object_xy: np.ndarray | None) -> dict:
    plan = solve(world, object_xy=object_xy)
    last = None
    for tel in run(world, plan):
        last = tel
    return last


def _succeeded(tel: dict) -> bool:
    return tel["cube_pos"][2] > 0.30 and tel["cube_tilt_deg"] < 25 and tel["place_err_mm"] < 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--spread", type=float, default=0.03, help="+/- placement range (m)")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for i in range(args.trials):
        dx, dy = rng.uniform(-args.spread, args.spread, 2)
        x, y = BASE_XY + [dx, dy]

        w = World(scene=pick_place.SCENE, ee_site=pick_place.EE_SITE)
        true = _place_cube(w, x, y)
        cam = Camera(w, "overhead")
        est = sense_object_pose(cam, segmentation=True)
        cam.close()
        perc_err = float(np.linalg.norm(est.top_xy - true) * 1e3)
        oracle = _rollout(w, None)

        w = World(scene=pick_place.SCENE, ee_site=pick_place.EE_SITE)
        _place_cube(w, x, y)
        percept = _rollout(w, est.top_xy)

        rows.append(
            dict(
                trial=i,
                perc_err_mm=perc_err,
                oracle_ok=_succeeded(oracle),
                oracle_place_mm=oracle["place_err_mm"],
                percept_ok=_succeeded(percept),
                percept_place_mm=percept["place_err_mm"],
            )
        )
        print(
            f"trial {i:2d}  pos=({x:+.3f},{y:+.3f})  perc_err={perc_err:4.1f}mm  "
            f"oracle={'ok ' if rows[-1]['oracle_ok'] else 'MISS'} "
            f"({oracle['place_err_mm']:.1f}mm)  "
            f"percept={'ok ' if rows[-1]['percept_ok'] else 'MISS'} "
            f"({percept['place_err_mm']:.1f}mm)"
        )

    n = len(rows)
    o_succ = sum(r["oracle_ok"] for r in rows)
    p_succ = sum(r["percept_ok"] for r in rows)
    perc_errs = np.array([r["perc_err_mm"] for r in rows])

    md = OUTDIR / "perception_bench.md"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(md, "w") as f:
        f.write("# Perception vs oracle pick-and-place\n\n")
        f.write(
            f"{n} randomized cube placements on the pick table (+/-{args.spread * 1e3:.0f} mm).\n\n"
        )
        f.write("| driver | grasp success | mean place err |\n")
        f.write("|--------|--------------:|---------------:|\n")
        f.write(
            f"| oracle (ground-truth pose) | {o_succ}/{n} | "
            f"{np.mean([r['oracle_place_mm'] for r in rows]):.1f} mm |\n"
        )
        f.write(
            f"| **perception (RGB-D)** | {p_succ}/{n} | "
            f"{np.mean([r['percept_place_mm'] for r in rows]):.1f} mm |\n\n"
        )
        f.write(
            f"Perception pose error: mean {perc_errs.mean():.1f} mm, "
            f"max {perc_errs.max():.1f} mm.\n"
        )

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(perc_errs, bins=10, color="#4c78a8", edgecolor="k")
    ax[0].set_title("perception pose error")
    ax[0].set_xlabel("estimate error (mm)")
    ax[0].set_ylabel("trials")
    pe = [r["perc_err_mm"] for r in rows]
    pp = [r["percept_place_mm"] for r in rows]
    ok = [r["percept_ok"] for r in rows]
    ax[1].scatter(
        [e for e, k in zip(pe, ok) if k],
        [p for p, k in zip(pp, ok) if k],
        c="#2ca02c",
        label="grasped",
        s=40,
    )
    ax[1].scatter(
        [e for e, k in zip(pe, ok) if not k],
        [p for p, k in zip(pp, ok) if not k],
        c="#d62728",
        label="missed",
        s=40,
    )
    ax[1].set_title("place error vs perception error")
    ax[1].set_xlabel("perception error (mm)")
    ax[1].set_ylabel("place error (mm)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "perception_bench.png", dpi=110)

    print(
        f"\noracle {o_succ}/{n}   perception {p_succ}/{n}   "
        f"perc err mean {perc_errs.mean():.1f}mm max {perc_errs.max():.1f}mm"
    )
    print(f"report: {md}")


if __name__ == "__main__":
    main()
