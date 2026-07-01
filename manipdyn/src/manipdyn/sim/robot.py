"""RobotSpec: the per-robot facts a :class:`~manipdyn.sim.world.World` needs.

``World`` discovers joints, DOFs, and actuators from the loaded model *by name*
and never hardcodes ``6`` — a ``RobotSpec`` just supplies those names (plus
end-effector site candidates and a home posture) for a given arm. The default
is the UR5e, so existing behaviour is byte-for-byte unchanged; supporting a
different arm is a matter of passing a different spec:

    world = World(scene="my_scene", robot=UR5E)   # default
    world = World(scene="panda_scene", robot=PANDA_SPEC)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    """Names and home posture that let ``World`` drive a particular arm.

    Parameters
    ----------
    name:
        Human-readable id (e.g. ``"ur5e"``).
    arm_joint_names:
        The arm's actuated joints, in kinematic order. ``World`` looks each up
        in the model and skips any that are absent, so a spec may list a
        superset.
    ee_site_candidates:
        End-effector site names to try, in priority order.
    home_qpos:
        A sane home configuration for the arm joints (one value per joint).
    """

    name: str
    arm_joint_names: tuple[str, ...]
    ee_site_candidates: tuple[str, ...]
    home_qpos: tuple[float, ...]


def prefixed(spec: RobotSpec, prefix: str) -> RobotSpec:
    """A copy of ``spec`` with ``prefix`` prepended to every joint/site name.

    ``mujoco.MjSpec`` attaches a sub-model under a name prefix, so a prefixed
    spec lets ``World`` discover one arm of a multi-robot scene (e.g.
    ``prefixed(UR5E, "left_")`` finds ``left_shoulder_pan_joint`` etc.).
    """
    return RobotSpec(
        name=f"{prefix}{spec.name}",
        arm_joint_names=tuple(prefix + n for n in spec.arm_joint_names),
        ee_site_candidates=tuple(prefix + c for c in spec.ee_site_candidates),
        home_qpos=spec.home_qpos,
    )


#: The UR5e — the lab's default arm. These are exactly the values ``World`` used
#: before ``RobotSpec`` existed, so the default path is unchanged.
UR5E = RobotSpec(
    name="ur5e",
    arm_joint_names=(
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ),
    ee_site_candidates=("attachment_site", "eef_site", "pinch"),
    home_qpos=(0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
)
