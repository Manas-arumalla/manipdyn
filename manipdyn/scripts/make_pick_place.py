"""Pick-and-place demo: top-down grasp, base-rotate transport, place.

A clean, robust pipeline built from the library:

  * **grasp config**: robust multi-seed optimization of joint angles for the
    grasp position + a downward gripper axis, **within joint limits** and
    FK-validated (unconstrained solves return out-of-limit poses the actuators
    can't hold; plain orientation IK is unreliable on this gripper),
  * **approach config**: lifted straight up from the grasp config with tiny,
    limit-clamped Jacobian steps (in-branch, vertical, orientation-preserving),
  * **place configs**: the pick configs with the base joint rotated 90° — since
    the shoulder pan rotates the whole arm about the vertical axis, this maps
    the pickup column (0.45, 0) to the place column (0, 0.45) exactly,
  * **time-optimal parameterization + computed-torque tracking** for every move,
  * a **kinematic grasp**: while held, the object's pose tracks the gripper
    rigidly (velocity zeroed, so release is gentle). MuJoCo's weld equality
    keeps its *compile-time* relative pose, which would fling the object on
    activation; tracking the captured grasp transform is robust and exact.

Run (from manipdyn/):
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
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.015
APPROACH_DZ, GRASP_Z, PLACE_PAN = 0.20, 0.35, np.pi / 2
CAPTURE_EVERY = 16


def main() -> None:
    world = World(scene="scene_pick", ee_site="pinch")
    m, d = world.model, world.data
    sid = world.ee_site_id
    oid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    gbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    obj_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
    obj_dadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
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
        def cost(q):
            p, z = fk(q)
            return (
                np.sum((p - target) ** 2)
                + 0.05 * np.sum((z - [0, 0, -1.0]) ** 2)
                + 50.0 * _limit_penalty(q)
            )

        best, best_cost = None, np.inf
        for s in [home] + [home + rng.uniform(-0.6, 0.6, 6) for _ in range(14)]:
            r = minimize(
                cost,
                s,
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-11},
            )
            p, z = fk(r.x)
            if (
                np.linalg.norm(p - target) < 0.01
                and z[2] < -0.9
                and _limit_penalty(r.x) < 1e-4
                and r.fun < best_cost
            ):
                best, best_cost = r.x, r.fun
        if best is None:
            raise RuntimeError(f"No valid top-down config at {target}")
        return best

    def lift(q: np.ndarray, xy: np.ndarray, dz: float, n: int = 600) -> np.ndarray:
        q = q.copy()
        target_z = fk(q)[0][2] + dz
        for _ in range(n):
            world.set_arm_qpos(q)
            world.forward()
            p = world.ee_pos
            err = np.array([xy[0] - p[0], xy[1] - p[1], target_z - p[2]])
            if np.linalg.norm(err) < 5e-4:
                break
            jp, _ = world.ee_jacobian()
            q = np.clip(
                q + 0.05 * (jp.T @ np.linalg.solve(jp @ jp.T + 1e-4 * np.eye(3), err)), lo, hi
            )
        return q

    grasp = anchor(np.array([obj[0], obj[1], GRASP_Z]))
    pick_hi = lift(grasp, np.array([obj[0], obj[1]]), APPROACH_DZ)
    rotate = np.array([PLACE_PAN, 0, 0, 0, 0, 0])
    place, place_hi = grasp + rotate, pick_hi + rotate

    ctrl = ComputedTorqueController(world, kp=600, kd=50)
    # Framed camera: pickup column (+x) on the left, place target (+y) on the
    # right, so the carry reads clearly as an A -> B transport.
    rec = Recorder(
        world,
        width=520,
        height=420,
        fps=20,
        lookat=(0.2, 0.2, 0.1),
        distance=1.6,
        azimuth=225,
        elevation=-40,
    )
    dt = world.timestep
    grasp_point = np.array([obj[0], obj[1], GRASP_Z])
    state = {
        "grip": GRIP_OPEN,
        "hold": False,
        "attached": False,
        "rel_pos": None,
        "rel_quat": None,
        "i": 0,
    }

    world.reset(pick_hi)
    d.ctrl[grip_ids] = GRIP_OPEN
    for _ in range(50):
        mujoco.mj_step(m, d)

    def _capture_rel():
        # object pose expressed in the gripper_base frame
        p1, q1 = d.xpos[gbid].copy(), d.xquat[gbid].copy()
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, q1)
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, d.xpos[oid] - p1, q1inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, q1inv, d.xquat[oid])
        state["rel_pos"], state["rel_quat"] = rel_pos, rel_quat

    def _attach():
        # place the object at the captured offset from the (now-moved) gripper
        p1, q1 = d.xpos[gbid], d.xquat[gbid]
        wp = np.zeros(3)
        mujoco.mju_rotVecQuat(wp, state["rel_pos"], q1)
        d.qpos[obj_qadr : obj_qadr + 3] = p1 + wp
        new_q = np.zeros(4)
        mujoco.mju_mulQuat(new_q, q1, state["rel_quat"])
        d.qpos[obj_qadr + 3 : obj_qadr + 7] = new_q
        d.qvel[obj_dadr : obj_dadr + 6] = 0.0  # no residual velocity -> gentle release
        mujoco.mj_forward(m, d)

    def drive(q_ref: np.ndarray) -> None:
        world.step(ctrl.compute(Target(q=q_ref)))
        if state["hold"]:
            if not state["attached"] and np.linalg.norm(world.ee_pos - grasp_point) < 0.06:
                _capture_rel()
                state["attached"] = True
            if state["attached"]:
                _attach()
        else:
            state["attached"] = False
        d.ctrl[grip_ids] = state["grip"]
        if state["i"] % CAPTURE_EVERY == 0:
            rec.capture()
        state["i"] += 1

    def move(q_start, q_goal, vmax: float) -> None:
        timed = parameterize_time_optimal(
            np.vstack([q_start, q_goal]), np.full(6, vmax), np.full(6, 2.5), n_samples=80
        )
        for k in range(int(timed.duration / dt)):
            t = min(k * dt, timed.duration)
            drive(np.array([np.interp(t, timed.t, timed.q[:, j]) for j in range(6)]))

    def settle(q_ref, seconds: float) -> None:
        for _ in range(int(seconds / dt)):
            drive(q_ref)

    # --- task sequence -------------------------------------------------
    settle(pick_hi, 0.4)
    move(pick_hi, grasp, 0.6)  # descend onto the object
    settle(grasp, 0.4)
    state["grip"], state["hold"] = GRIP_CLOSE, True
    settle(grasp, 0.6)  # close + grasp (kinematic attach latches)
    move(grasp, pick_hi, 0.6)  # lift
    move(pick_hi, place_hi, 1.2)  # rotate the base to the place column
    move(place_hi, place, 0.6)  # lower
    state["grip"], state["hold"] = GRIP_OPEN, False
    settle(place, 0.6)  # release (object settles onto floor)
    move(place, place_hi, 0.6)  # retract

    gif = rec.save_gif(MEDIA / "pick_place.gif", palettesize=48, max_width=360)
    rec.close()
    placed = d.xpos[oid]
    print(f"object final position: {np.round(placed, 3)}  (placed near [0, 0.45])")
    print(f"GIF saved: {gif}")


if __name__ == "__main__":
    main()
