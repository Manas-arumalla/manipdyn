# Tasks: pick-and-place

The pick-and-place demo (`scripts/make_pick_place.py`) chains the library's
components into a complete manipulation task and renders it headlessly.

![pick and place](../media/pick_place.gif)

## Pipeline

1. **Grasp configuration** — a robust multi-seed optimization finds joint angles
   that place the gripper at the grasp point with its approach axis pointing
   **down**, *within joint limits* and validated by forward kinematics.
   (Unconstrained solves return out-of-limit elbow/wrist angles the actuators
   can't hold; plain orientation IK is unreliable on this gripper.)
2. **Approach configuration** — the grasp config lifted straight up with small,
   limit-clamped **Jacobian steps**, so it stays in the same IK branch (a clean
   vertical, orientation-preserving move rather than a wild reconfiguration).
3. **Place configurations** — the pick configs with the base joint rotated 90°.
   Because the shoulder pan rotates the whole arm about the vertical axis, this
   maps the pickup column `(0.45, 0)` to the place column `(0, 0.45)` *exactly*,
   so the carry is a clean base rotation needing no planning.
4. **Execution** — every move is a time-optimal trajectory tracked by
   computed-torque control.
5. **Grasp** — while held, the object's pose tracks the gripper rigidly from the
   captured grasp transform (velocity zeroed, so release is gentle). This is
   more robust than a MuJoCo weld equality, which preserves its *compile-time*
   relative pose and would fling the object on activation.

## The object

A wide, heavy base with a thin grippable post: the gripper grips the post high
(≈ 0.35 m — a comfortable height where the arm doesn't contort into the floor),
while the low centre of mass keeps the object upright when it's set down.

## Lessons (documented so they aren't re-learned)

* Top-down grasps below ≈ 0.35 m make IK contort an arm link **through the
  floor** — keep grasps high.
* A free object must rest **exactly** on the floor (center = half-height) or the
  initial settling drop topples a tall object.
* Support stands under the object get hit by the arm on approach; a free-
  standing object (excluded from the approach's collision set, or simply low
  enough not to obstruct) avoids this.
