"""Pick-and-place task: a reusable, driver-agnostic pipeline.

The motion is exposed as a **generator** that advances the simulation one
control step at a time and yields telemetry. That lets any caller drive it —
the headless GIF script renders an offscreen frame each step, the GUI steps it
from a timer and syncs the live viewer — without duplicating the pipeline.

Pipeline (see ``docs/tasks.md`` for the why):
  * a strict top-down grasp config is found by multi-seed optimization;
  * the approach is an orientation-locked vertical line so the open gripper
    slides straight down around the cube without knocking it;
  * the place config is the pick config with the base joint rotated 90°, which
    maps the pick table onto the place table exactly;
  * a weld (its relative pose set at grasp time) carries the cube rigidly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import minimize

from manipdyn.control import ComputedTorqueController, Target
from manipdyn.sim import World
from manipdyn.trajectory import parameterize_time_optimal

SCENE = "scene_pick"
EE_SITE = "pinch"
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
APPROACH_DZ, GRASP_Z, PLACE_PAN = 0.20, 0.385, np.pi / 2


@dataclass
class PickPlacePlan:
    grasp: np.ndarray
    pick_hi: np.ndarray
    place: np.ndarray
    place_hi: np.ndarray
    descend_wps: list[np.ndarray]
    lift_wps: list[np.ndarray]
    place_down_wps: list[np.ndarray]
    place_up_wps: list[np.ndarray]
    cube_xy: np.ndarray


def solve(world: World, object_xy: np.ndarray | None = None) -> PickPlacePlan:
    """Solve the grasp/approach/place configurations for the current scene.

    ``object_xy`` optionally overrides the cube's ``(x, y)`` — pass a perceived
    position (see :func:`manipdyn.perception.sense_object_pose`) to drive the
    grasp from vision instead of the simulator's ground-truth pose. Left
    ``None`` (the default), the true object position is used, so existing
    callers are unaffected.
    """
    m, d = world.model, world.data
    sid = world.ee_site_id
    oid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    home = world.home_qpos_arm
    rng = np.random.default_rng(0)
    lo, hi = world.joint_limits[:, 0], world.joint_limits[:, 1]

    for jn in ("right_driver", "left_driver"):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = GRIP_OPEN
    mujoco.mj_forward(m, d)
    obj = d.xpos[oid].copy()
    if object_xy is not None:
        obj[:2] = np.asarray(object_xy, dtype=float)

    def fk(q):
        world.set_arm_qpos(q)
        world.forward()
        return world.ee_pos.copy(), d.site_xmat[sid].reshape(3, 3)[:, 2]

    def limit_penalty(q):
        return np.sum(np.maximum(0, q - hi) ** 2) + np.sum(np.maximum(0, lo - q) ** 2)

    def cost_at(target):
        def cost(q):
            p, z = fk(q)
            return (
                np.sum((p - target) ** 2)
                + 0.5 * np.sum((z - [0, 0, -1.0]) ** 2)
                + 50.0 * limit_penalty(q)
            )

        return cost

    def anchor(target):
        best, best_cost = None, np.inf
        for s in [home, *[home + rng.uniform(-0.5, 0.5, 6) for _ in range(20)]]:
            r = minimize(
                cost_at(target),
                s,
                method="Nelder-Mead",
                options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-11},
            )
            p, z = fk(r.x)
            if (
                np.linalg.norm(p - target) < 0.012
                and z[2] < -0.985
                and limit_penalty(r.x) < 1e-4
                and r.fun < best_cost
            ):
                best, best_cost = r.x, r.fun
        if best is None:
            raise RuntimeError(f"no valid top-down config at {target}")
        return best

    def topdown_at(target, seed):
        r = minimize(
            cost_at(target),
            seed,
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-12},
        )
        p, z = fk(r.x)
        if np.linalg.norm(p - target) > 0.01 or z[2] > -0.985:
            raise RuntimeError(f"top-down waypoint solve failed at {target}")
        return r.x

    def vertical_path(q_lo, q_hi, n=7):
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
    descend_wps = vertical_path(grasp, pick_hi)[::-1]
    return PickPlacePlan(
        grasp=grasp,
        pick_hi=pick_hi,
        place=grasp + rotate,
        place_hi=pick_hi + rotate,
        descend_wps=descend_wps,
        lift_wps=descend_wps[::-1],
        place_down_wps=[w + rotate for w in descend_wps],
        place_up_wps=[w + rotate for w in descend_wps][::-1],
        cube_xy=obj[:2].copy(),
    )


def run(world: World, plan: PickPlacePlan | None = None) -> Iterator[dict]:
    """Drive the full pick-and-place, yielding telemetry once per control step.

    The caller advances the generator and may render/sync/capture between steps.
    Grip commands and the carry weld are applied internally.
    """
    plan = plan or solve(world)
    m, d = world.model, world.data
    oid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    gbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
    grip_ids = [i for i in range(m.nu) if i not in set(world.arm_actuator_ids.tolist())]
    ctrl = ComputedTorqueController(world, kp=600, kd=50)
    dt = world.timestep
    grip = GRIP_OPEN
    place_target = np.array([-plan.cube_xy[1], plan.cube_xy[0]])  # +90° base-pan image

    def telemetry(phase: str) -> dict:
        R = d.xmat[oid].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[:, 2] @ [0, 0, 1.0], -1.0, 1.0))))
        cube = d.xpos[oid].copy()
        return {
            "phase": phase,
            "t": float(world.time),
            "cube_pos": cube,
            "cube_tilt_deg": tilt,
            "place_err_mm": float(np.linalg.norm(cube[:2] - place_target) * 1e3),
            "ee_pos": world.ee_pos.copy(),
        }

    def drive(q_ref, phase):
        world.step(ctrl.compute(Target(q=q_ref)))
        d.ctrl[grip_ids] = grip
        return telemetry(phase)

    def settle(q_ref, seconds, phase):
        for _ in range(int(seconds / dt)):
            yield drive(q_ref, phase)

    def move_path(wps, vmax, phase):
        timed = parameterize_time_optimal(
            np.vstack(wps), np.full(6, vmax), np.full(6, 2.5), n_samples=120
        )
        for k in range(int(timed.duration / dt)):
            t = min(k * dt, timed.duration)
            yield drive(np.array([np.interp(t, timed.t, timed.q[:, j]) for j in range(6)]), phase)

    def grasp_weld(active: bool):
        if active:
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

    # Hold the start pose under control so the arm does not droop into the cube.
    world.reset(plan.pick_hi)
    for _ in range(120):
        world.step(ctrl.compute(Target(q=plan.pick_hi)))
        d.ctrl[grip_ids] = GRIP_OPEN
        yield telemetry("ready")

    yield from settle(plan.pick_hi, 0.4, "approach")
    yield from move_path(plan.descend_wps, 0.4, "descend")
    yield from settle(plan.grasp, 0.4, "align")
    grip = GRIP_CLOSE
    yield from settle(plan.grasp, 0.7, "grasp")
    grasp_weld(True)
    yield from move_path(plan.lift_wps, 0.4, "lift")
    yield from move_path([plan.pick_hi, plan.place_hi], 1.2, "carry")
    yield from move_path(plan.place_down_wps, 0.4, "lower")
    grasp_weld(False)
    grip = GRIP_OPEN
    yield from settle(plan.place, 0.6, "release")
    yield from move_path(plan.place_up_wps, 0.4, "retract")
