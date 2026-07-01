"""Procedural (parametric) scene generation with ``mujoco.MjSpec``.

Build scenes in Python instead of hand-writing MJCF: load a base arm scene into
an editable spec, add a table, a parametric number of randomly-placed cubes, and
an overhead camera, then compile to an ``MjModel`` that :class:`~manipdyn.sim.world.World`
can wrap directly::

    from manipdyn.sim import World
    from manipdyn.models.procedural import build_clutter_scene

    model = build_clutter_scene(n_cubes=4, seed=1)
    world = World(model=model, ee_site="pinch")

This is the basis for domain randomization (fresh layouts every reset/episode)
and for perception/RL over scenes that never had to be authored by hand. Builds
are deterministic in ``seed``.
"""

from __future__ import annotations

import mujoco
import numpy as np

from manipdyn.models import scene_path
from manipdyn.sim.robot import UR5E, prefixed
from manipdyn.sim.world import World

# A distinct, readable palette; cubes cycle through it.
_PALETTE = (
    (0.90, 0.22, 0.12, 1.0),  # red
    (0.20, 0.75, 0.28, 1.0),  # green
    (0.20, 0.34, 0.90, 1.0),  # blue
    (0.95, 0.75, 0.15, 1.0),  # yellow
    (0.60, 0.30, 0.75, 1.0),  # purple
    (0.20, 0.75, 0.80, 1.0),  # cyan
)


def _poisson_xy(
    rng: np.random.Generator,
    n: int,
    half: tuple[float, float],
    min_sep: float,
    tries: int = 200,
) -> list[tuple[float, float]]:
    """Sample ``n`` points in a centred rectangle with a minimum separation."""
    pts: list[tuple[float, float]] = []
    for _ in range(n):
        for _ in range(tries):
            p = (rng.uniform(-half[0], half[0]), rng.uniform(-half[1], half[1]))
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep**2 for q in pts):
                pts.append(p)
                break
        else:
            raise RuntimeError(
                f"could not place {n} cubes with separation {min_sep} m — "
                "reduce n_cubes or min_sep, or widen the table."
            )
    return pts


def build_clutter_scene(
    n_cubes: int = 3,
    seed: int = 0,
    base_scene: str = "scene_base_gripper",
    table_center: tuple[float, float] = (-0.49, -0.13),
    table_half: tuple[float, float] = (0.16, 0.12),
    cube_half: float = 0.025,
    min_sep: float = 0.075,
) -> mujoco.MjModel:
    """Build a UR5e + gripper scene with a table and ``n_cubes`` random cubes.

    Cubes are named ``cube_0 .. cube_{n-1}`` (each free-jointed, so
    :func:`manipdyn.perception.movable_bodies` finds them) and coloured from a
    fixed palette. An ``overhead`` camera matching the pick scene is added.
    Returns a compiled :class:`mujoco.MjModel`.
    """
    rng = np.random.default_rng(seed)
    spec = mujoco.MjSpec.from_file(scene_path(base_scene))
    wb = spec.worldbody

    cx, cy = table_center
    top_z = 0.355
    # Table top + four legs.
    wb.add_geom(
        name="pt_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[cx, cy, top_z],
        size=[table_half[0], table_half[1], 0.01],
        rgba=[0.62, 0.43, 0.26, 1.0],
    )
    for sx in (table_half[0] - 0.02, -(table_half[0] - 0.02)):
        for sy in (table_half[1] - 0.02, -(table_half[1] - 0.02)):
            wb.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[cx + sx, cy + sy, 0.1725],
                size=[0.013, 0.013, 0.1725],
                rgba=[0.40, 0.27, 0.16, 1.0],
            )

    # Random, non-overlapping cubes resting on the table top.
    inner = (table_half[0] - cube_half - 0.015, table_half[1] - cube_half - 0.015)
    rest_z = top_z + 0.01 + cube_half
    for i, (dx, dy) in enumerate(_poisson_xy(rng, n_cubes, inner, min_sep)):
        body = wb.add_body(name=f"cube_{i}", pos=[cx + dx, cy + dy, rest_z])
        body.add_freejoint(name=f"cube_{i}_free")
        body.add_geom(
            name=f"cube_{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[cube_half, cube_half, cube_half],
            rgba=list(_PALETTE[i % len(_PALETTE)]),
            mass=0.05,
            friction=[1.5, 0.05, 0.001],
        )

    wb.add_camera(
        name="overhead",
        pos=[-0.18, -0.31, 1.15],
        fovy=58,
        xyaxes=[1, 0, 0, 0, 1, 0],
    )
    return spec.compile()


LEFT_PREFIX, RIGHT_PREFIX = "left_", "right_"


def build_two_arm_scene(
    separation: float = 0.95,
    base_scene: str = "scene_base_gripper",
    cube_xy: tuple[float, float] = (-0.30, 0.10),
    table_center: tuple[float, float] = (-0.30, 0.0),
) -> mujoco.MjModel:
    """Two UR5e + grippers facing each other across a table, with a graspable cube.

    The arms are attached with ``left_`` / ``right_`` name prefixes (via
    ``MjSpec``) so they coexist in one model. Two weld equalities —
    ``left_grasp`` and ``right_grasp`` (gripper_base <-> object) — are provided
    inactive, so a task can grasp with one arm and hand the object to the other.
    Wrap the result with :func:`two_arm_worlds`.
    """
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.5, 0.53, 0.58, 1]
    )
    spec.worldbody.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1])
    spec.worldbody.add_light(pos=[-0.6, 0.6, 1.6], dir=[0.4, -0.4, -1])

    half = separation / 2.0
    for prefix, y, yaw in ((LEFT_PREFIX, half, 0.0), (RIGHT_PREFIX, -half, np.pi)):
        child = mujoco.MjSpec.from_file(scene_path(base_scene))
        frame = spec.worldbody.add_frame(
            pos=[0.0, y, 0.0], quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        )
        frame.attach_body(child.worldbody.first_body(), prefix, "")

    # A shared table and a graspable cube.
    tx, ty = table_center
    spec.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[tx, ty, 0.355],
        size=[0.12, 0.12, 0.01],
        rgba=[0.62, 0.43, 0.26, 1.0],
    )
    cube = spec.worldbody.add_body(name="object", pos=[cube_xy[0], cube_xy[1], 0.39])
    cube.add_freejoint(name="object_free")
    cube.add_geom(
        name="object_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.025],
        rgba=[0.90, 0.30, 0.12, 1.0],
        mass=0.05,
        friction=[1.5, 0.05, 0.001],
    )

    for prefix in (LEFT_PREFIX, RIGHT_PREFIX):
        spec.add_equality(
            name=f"{prefix}grasp",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=f"{prefix}gripper_base",
            name2="object",
            active=False,
        )
    return spec.compile()


def two_arm_worlds(model: mujoco.MjModel) -> tuple[World, World]:
    """Wrap a two-arm model as ``(left, right)`` Worlds sharing one simulation.

    Both wrap the same model/data; step only ``left`` (or ``mj_step`` once) after
    applying each arm's torque with ``apply_arm_torque``.
    """
    left = World(model=model, robot=prefixed(UR5E, LEFT_PREFIX), ee_site=f"{LEFT_PREFIX}pinch")
    right = World(
        model=model,
        robot=prefixed(UR5E, RIGHT_PREFIX),
        ee_site=f"{RIGHT_PREFIX}pinch",
        data=left.data,
        home=False,
    )
    right.set_arm_qpos(right.home_qpos_arm)
    left.forward()
    return left, right
