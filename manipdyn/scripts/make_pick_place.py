"""Pick-and-place demo: pick a cube off one table, carry it, place it on another.

The pipeline, built from the library:

  * a top-down grasp configuration is found by optimizing joint angles for the
    grasp position and a downward gripper axis, within joint limits and
    validated by forward kinematics;
  * the approach is a true vertical line: a top-down configuration is solved at
    each height (seeded from the previous one, so it stays in one IK branch),
    so the open fingers slide straight down around the cube without touching it;
  * the place configurations are the pick configurations with the base joint
    rotated 90 degrees -- because the shoulder pan rotates the whole arm about
    the vertical axis, this maps the pick table at (-0.49, -0.13) onto the place
    table at (0.13, -0.49) exactly;
  * every move is a time-optimal trajectory tracked by computed-torque control;
  * the gripper closes its fingers onto the cube, and a weld constraint (its
    relative pose set at grasp time) holds the cube firmly while it is carried
    -- a rigid grip rather than a slip-prone friction hold.

Run from the manipdyn/ directory:
    python scripts/make_pick_place.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from manipdyn.control import ComputedTorqueController, Target
from manipdyn.render import Recorder
from manipdyn.sim import World
from manipdyn.trajectory import parameterize_time_optimal

MEDIA = Path(__file__).resolve().parents[1] / "media"
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
APPROACH_DZ, GRASP_Z, PLACE_PAN = 0.20, 0.385, np.pi / 2
CAPTURE_EVERY = 20


def main() -> None:
    world = World(scene="scene_pick", ee_site="pinch")
    m, d = world.model, world.data
    sid = world.ee_site_id
    oid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    gbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
    grip_ids = [i for i in range(m.nu) if i not in set(world.arm_actuator_ids.tolist())]
    home = world.home_qpos_arm
    rng = np.random.default_rng(0)

    for jn in ("right_driver", "left_driver"):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = GRIP_OPEN
    mujoco.mj_forward(m, d)
    obj = d.xpos[oid].copy()
    lo, hi = world.joint_limits[:, 0], world.joint_limits[:, 1]

    def fk(q):
        world.set_arm_qpos(q)
        world.forward()
        return world.ee_pos.copy(), d.site_xmat[sid].reshape(3, 3)[:, 2]

    def _limit_penalty(q):
        return np.sum(np.maximum(0, q - hi) ** 2) + np.sum(np.maximum(0, lo - q) ** 2)

    def anchor(target: np.ndarray) -> np.ndarray:
        """Strict top-down config at ``target`` (FK- and limit-validated)."""

        def cost(q):
            p, z = fk(q)
            return (
                np.sum((p - target) ** 2)
                + 0.5 * np.sum((z - [0, 0, -1.0]) ** 2)
                + 50.0 * _limit_penalty(q)
            )

        best, best_cost = None, np.inf
        for s in [home] + [home + rng.uniform(-0.5, 0.5, 6) for _ in range(20)]:
            r = minimize(
                cost,
                s,
                method="Nelder-Mead",
                options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-11},
            )
            p, z = fk(r.x)
            if (
                np.linalg.norm(p - target) < 0.012
                and z[2] < -0.985
                and _limit_penalty(r.x) < 1e-4
                and r.fun < best_cost
            ):
                best, best_cost = r.x, r.fun
        if best is None:
            raise RuntimeError(f"No valid top-down config at {target}")
        return best

    def topdown_at(target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """A single top-down config at ``target``, refined from a nearby seed."""

        def cost(q):
            p, z = fk(q)
            return (
                np.sum((p - target) ** 2)
                + 0.5 * np.sum((z - [0, 0, -1.0]) ** 2)
                + 50.0 * _limit_penalty(q)
            )

        r = minimize(
            cost,
            seed,
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-12},
        )
        p, z = fk(r.x)
        if np.linalg.norm(p - target) > 0.01 or z[2] > -0.985:
            raise RuntimeError(f"top-down waypoint solve failed at {target}")
        return r.x

    def vertical_path(q_lo: np.ndarray, q_hi: np.ndarray, n: int = 7) -> list[np.ndarray]:
        """Orientation-locked top-down configs from ``q_lo`` up to ``q_hi``.

        Straight joint-space interpolation between two top-down configs does
        *not* keep the gripper pointing down in between, so the fingers swing
        and rake the object. Solving an explicit top-down config at each height
        (seeded from the previous one, so it stays in the same IK branch) makes
        the approach a true vertical line with the fingers held parallel.
        """
        p_lo, p_hi = fk(q_lo)[0], fk(q_hi)[0]
        wps, seed = [q_lo], q_lo
        for a in np.linspace(0.0, 1.0, n)[1:-1]:
            seed = topdown_at((1 - a) * p_lo + a * p_hi, seed)
            wps.append(seed)
        wps.append(q_hi)
        return wps

    grasp = anchor(np.array([obj[0], obj[1], GRASP_Z]))
    pick_hi = topdown_at(np.array([obj[0], obj[1], GRASP_Z + APPROACH_DZ]), grasp)
    rotate = np.array([PLACE_PAN, 0, 0, 0, 0, 0])
    place, place_hi = grasp + rotate, pick_hi + rotate

    # Orientation-locked vertical approach/retreat paths, so the open gripper
    # slides straight down around the cube instead of knocking it.
    descend_wps = vertical_path(grasp, pick_hi)[::-1]  # pick_hi -> grasp
    lift_wps = descend_wps[::-1]  # grasp   -> pick_hi
    place_down_wps = [w + rotate for w in descend_wps]  # place_hi -> place
    place_up_wps = place_down_wps[::-1]  # place    -> place_hi

    ctrl = ComputedTorqueController(world, kp=600, kd=50)
    rec = Recorder(
        world,
        width=560,
        height=440,
        fps=20,
        lookat=(-0.18, -0.31, 0.42),
        distance=1.45,
        azimuth=215,
        elevation=-18,
    )
    dt = world.timestep
    state = {"grip": GRIP_OPEN, "i": 0}

    # Hold the start pose with control while the cube settles and the fingers
    # open. (Stepping uncontrolled here would let the arm droop under gravity
    # and nudge the cube before the demo even begins.)
    world.reset(pick_hi)
    for _ in range(120):
        world.step(ctrl.compute(Target(q=pick_hi)))
        d.ctrl[grip_ids] = GRIP_OPEN

    def grasp_weld(active: bool) -> None:
        if active:
            # Pin the object to the gripper at its current relative pose, so the
            # weld holds it firmly with no jump.
            p1, q1 = d.xpos[gbid].copy(), d.xquat[gbid].copy()
            q1inv = np.zeros(4)
            mujoco.mju_negQuat(q1inv, q1)
            rel_pos = np.zeros(3)
            mujoco.mju_rotVecQuat(rel_pos, d.xpos[oid] - p1, q1inv)
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, q1inv, d.xquat[oid])
            m.eq_data[wid, 0:3] = 0.0
            m.eq_data[wid, 3:6] = rel_pos
            m.eq_data[wid, 6:10] = rel_quat
            m.eq_data[wid, 10] = 1.0
        d.eq_active[wid] = 1 if active else 0
        mujoco.mj_forward(m, d)

    def drive(q_ref: np.ndarray) -> None:
        world.step(ctrl.compute(Target(q=q_ref)))
        d.ctrl[grip_ids] = state["grip"]
        if state["i"] % CAPTURE_EVERY == 0:
            rec.capture()
        state["i"] += 1

    def move_path(waypoints, vmax: float) -> None:
        timed = parameterize_time_optimal(
            np.vstack(waypoints), np.full(6, vmax), np.full(6, 2.5), n_samples=120
        )
        for k in range(int(timed.duration / dt)):
            t = min(k * dt, timed.duration)
            drive(np.array([np.interp(t, timed.t, timed.q[:, j]) for j in range(6)]))

    def move(q_start, q_goal, vmax: float) -> None:
        move_path([q_start, q_goal], vmax)

    def settle(q_ref, seconds: float) -> None:
        for _ in range(int(seconds / dt)):
            drive(q_ref)

    # --- task sequence -------------------------------------------------
    settle(pick_hi, 0.4)
    move_path(descend_wps, 0.4)  # straight down, open fingers straddle the cube
    settle(grasp, 0.4)
    state["grip"] = GRIP_CLOSE  # close the fingers onto the cube
    settle(grasp, 0.7)
    grasp_weld(True)  # firm grip
    move_path(lift_wps, 0.4)  # lift straight up off the pick table
    move(pick_hi, place_hi, 1.2)  # rotate the base to the place table
    move_path(place_down_wps, 0.4)  # lower straight down onto the place table
    grasp_weld(False)  # release the grip
    state["grip"] = GRIP_OPEN
    settle(place, 0.6)
    move_path(place_up_wps, 0.4)  # lift straight up to retract

    gif = rec.save_gif(MEDIA / "pick_place.gif", palettesize=40, max_width=400)
    rec.close()
    placed = d.xpos[oid]
    R = d.xmat[oid].reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[:, 2] @ [0, 0, 1.0], -1.0, 1.0)))
    target = np.array([-obj[1], obj[0], placed[2]])  # +90 deg base-pan image of the pick spot
    print(
        f"cube final position: {np.round(placed, 3)}  (target {np.round(target, 3)}); "
        f"tilt {tilt:.1f} deg"
    )
    print(f"GIF saved: {gif}")


if __name__ == "__main__":
    main()
