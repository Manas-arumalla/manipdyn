"""Two-arm handover demo: left arm picks a cube, hands it to the right arm.

Builds the two-arm scene, plans joint waypoints for each arm with IK, tracks them
with computed-torque control, and transfers the cube between grippers with a weld
(its relative pose set at hand-off time, as in the single-arm pick-and-place).
Renders a side view to a GIF.

Run from the manipdyn/ directory:
    python scripts/make_handover.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from manipdyn.control import ComputedTorqueController, Target
from manipdyn.kinematics import IKSolver
from manipdyn.models.procedural import build_two_arm_scene, two_arm_worlds
from manipdyn.render.recorder import save_gif

MEDIA = Path(__file__).resolve().parents[1] / "media"
H = 360
CAPTURE_EVERY = 40


def main() -> None:
    model = build_two_arm_scene()
    left, right = two_arm_worlds(model)
    d = left.data

    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    eq = {
        p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{p}grasp")
        for p in ("left_", "right_")
    }
    gb = {
        p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{p}gripper_base")
        for p in ("left_", "right_")
    }

    cube = d.xpos[oid].copy()
    hand = np.array([-0.30, 0.0, 0.52])  # hand-off point, reachable by both
    place = np.array([-0.30, -0.085, 0.40])  # a spot on the table, in the right arm's reach

    ik_l, ik_r = IKSolver(left), IKSolver(right)
    home_l, home_r = left.home_qpos_arm.copy(), right.home_qpos_arm.copy()
    l_hi = ik_l.solve(cube + [0, 0, 0.15], q_guess=home_l).q
    l_pick = ik_l.solve(cube + [0, 0, 0.03], q_guess=l_hi).q
    l_hand = ik_l.solve(hand, q_guess=l_hi).q
    r_hand = ik_r.solve(hand + [0, -0.02, 0.0], q_guess=home_r).q
    r_over = ik_r.solve(place + [0, 0, 0.16], q_guess=r_hand).q
    r_down = ik_r.solve(place + [0, 0, -0.05], q_guess=r_over).q  # aim low; we stop early

    cl = ComputedTorqueController(left, kp=500, kd=45)
    cr = ComputedTorqueController(right, kp=500, kd=45)
    dt = left.timestep

    renderer = mujoco.Renderer(model, height=H, width=620)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Look along -X so the two arms (separated along Y) appear side by side.
    cam.lookat[:] = (-0.32, 0.0, 0.42)
    cam.distance = 2.15
    cam.azimuth = 180
    cam.elevation = -20

    state = {"l": home_l.copy(), "r": home_r.copy(), "i": 0}
    frames: list[np.ndarray] = []

    def tick() -> None:
        left.apply_arm_torque(cl.compute(Target(q=state["l"])))
        right.apply_arm_torque(cr.compute(Target(q=state["r"])))
        mujoco.mj_step(model, d)
        if state["i"] % CAPTURE_EVERY == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
        state["i"] += 1

    def move(arm: str, q_to: np.ndarray, seconds: float) -> None:
        q_from = state[arm].copy()
        for k in range(int(seconds / dt)):
            state[arm] = q_from + (q_to - q_from) * (k + 1) / int(seconds / dt)
            tick()

    def hold(seconds: float) -> None:
        for _ in range(int(seconds / dt)):
            tick()

    def lower_until(arm: str, q_to: np.ndarray, seconds: float, stop_z: float) -> None:
        """Interpolate toward q_to but stop once the cube is near the table."""
        q_from = state[arm].copy()
        n = int(seconds / dt)
        for k in range(n):
            state[arm] = q_from + (q_to - q_from) * (k + 1) / n
            tick()
            if d.xpos[oid][2] <= stop_z:
                break

    def weld(prefix: str, active: bool) -> None:
        e, b = eq[prefix], gb[prefix]
        if active:
            q1 = d.xquat[b].copy()
            q1inv = np.zeros(4)
            mujoco.mju_negQuat(q1inv, q1)
            rel_pos = np.zeros(3)
            mujoco.mju_rotVecQuat(rel_pos, d.xpos[oid] - d.xpos[b], q1inv)
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, q1inv, d.xquat[oid])
            model.eq_data[e, 0:3] = 0.0
            model.eq_data[e, 3:6] = rel_pos
            model.eq_data[e, 6:10] = rel_quat
            model.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1 if active else 0
        mujoco.mj_forward(model, d)

    hold(0.4)
    move("l", l_hi, 1.0)
    move("l", l_pick, 0.8)
    hold(0.3)
    weld("left_", True)  # left grasps
    move("l", l_hi, 0.8)  # lift
    move("l", l_hand, 1.3)  # carry to hand-off
    hold(0.3)
    move("r", r_hand, 1.6)  # right comes to receive
    hold(0.3)
    weld("right_", True)  # right grasps
    weld("left_", False)  # left lets go
    hold(0.3)
    move("l", home_l, 1.2)  # left retracts
    move("r", r_over, 1.0)  # right carries over the place spot
    lower_until("r", r_down, 1.4, stop_z=0.405)  # lower until the cube meets the table
    hold(0.2)
    weld("right_", False)  # right releases
    hold(0.3)
    move("r", r_over, 0.8)  # retract

    tilt = float(
        np.degrees(np.arccos(np.clip(d.xmat[oid].reshape(3, 3)[:, 2] @ [0, 0, 1.0], -1.0, 1.0)))
    )
    final = d.xpos[oid].copy()
    print(
        f"cube final {np.round(final, 3)} tilt {tilt:.1f} deg  place err "
        f"{np.linalg.norm(final[:2] - place[:2]) * 1e3:.0f} mm  frames {len(frames)}"
    )

    renderer.close()
    gif = save_gif(frames, MEDIA / "handover.gif", fps=16, palettesize=32, max_width=560)
    print(f"GIF saved: {gif}")


if __name__ == "__main__":
    main()
