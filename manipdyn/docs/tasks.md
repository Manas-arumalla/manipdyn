# Tasks: pick-and-place

The pick-and-place demo (`scripts/make_pick_place.py`) chains the library's
components into a complete manipulation task and renders it headlessly: a cube
is picked off one table, carried, and placed on a second table.

![pick and place](../media/pick_place.gif)

## Pipeline

1. **Grasp configuration** — a multi-seed optimization finds joint angles that
   place the gripper at the cube with its approach axis pointing **down**,
   *within joint limits* and validated by forward kinematics. The cube sits in
   the arm's natural top-down reach, so this is a clean upright posture rather
   than a contorted one. (Plain orientation IK is unreliable on this gripper.)
2. **Approach** — a true vertical line. A top-down configuration is solved at
   each height (seeded from the one below it, so it stays in the same IK
   branch), so the open fingers slide **straight down** around the cube without
   raking it. A plain joint-space interpolation between two top-down configs
   does *not* keep the gripper pointing down in between, so the fingers would
   swing sideways and knock the cube over.
3. **Place configurations** — the pick configurations with the base joint
   rotated 90°. Because the shoulder pan rotates the whole arm about the
   vertical axis, this maps the pick table at `(-0.49, -0.13)` onto the place
   table at `(0.13, -0.49)` *exactly*, so the carry is a clean base rotation
   needing no planning.
4. **Execution** — every move is a time-optimal trajectory tracked by
   computed-torque control.
5. **Grasp** — two options. By default (`use_weld=True`) the fingers close on the
   cube and a weld holds it rigidly while it is carried; the weld's relative pose
   is set *at grasp time* from the live gripper-to-cube transform (a default weld
   keeps its *compile-time* pose and would fling the cube on activation). This is
   deterministic and robust.

   With **`use_weld=False`** the weld is dropped entirely and the cube is held by
   a **real contact grasp** — the fingers close and *friction alone* carries it
   through the lift, the base rotation, and the place. With this gripper and cube
   it completes the whole task (placed within a couple of millimetres, upright),
   so the weld is a convenience, not a crutch. Pass it through
   `pick_place.run(world, plan, use_weld=False)`.

## The object

A 50 mm cube on a small table. Being compact, it is gripped about its centre
and cannot topple, and a second identical table at the base-rotated location
receives it — a clean, self-contained pick-and-place rather than a fragile
balancing act.

## Design notes

* This arm's natural **top-down workspace is on one side** of the base — at its
  home pose the gripper already points straight down at ≈ `(-0.49, -0.13, 0.39)`.
  Top-down targets outside that zone have no natural IK solution; the optimizer
  is forced into contorted, limit-straining postures that drive a link through
  the table. So the tables sit in the natural zone.
* The arm is held under **closed-loop control from the very first step**. An
  uncontrolled settle would let it droop under gravity and nudge the cube before
  the motion even begins.
* The descent and retreat are **orientation-locked** vertical lines (see step 2),
  which is what keeps the gripper from sweeping into the cube or the table.
