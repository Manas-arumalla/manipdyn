"""World: a thin wrapper around a MuJoCo UR5e model.

Design notes:
  * Load scenes by absolute path so it runs from any working directory.
  * Discover the arm's joints / DOFs / actuators *from the model* by name,
    rather than hardcoding ``6`` and slicing ``[:6]``.
  * Expose the handful of physics quantities controllers actually need
    (end-effector pose, Jacobian, mass matrix, bias/gravity force) behind a
    small, typed surface so controllers never touch ``mujoco`` directly.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from manipdyn.models import scene_path
from manipdyn.sim.robot import UR5E, RobotSpec

# Backwards-compatible alias: the UR5e arm joints, in kinematic order. New code
# should reach for a :class:`RobotSpec`; this is kept for existing imports.
ARM_JOINT_NAMES: tuple[str, ...] = UR5E.arm_joint_names


class World:
    """A loaded UR5e MuJoCo model plus convenience accessors.

    Parameters
    ----------
    scene:
        Scene name (e.g. ``"scene_base"``) or ``.xml`` filename bundled with
        the package, or an absolute path to any MJCF file.
    timestep:
        Optional override for the integrator timestep (seconds).
    robot:
        Which arm to drive. Defaults to the UR5e (:data:`UR5E`); pass another
        :class:`RobotSpec` to load a different manipulator's joints/home.
    """

    def __init__(
        self,
        scene: str = "scene_base",
        timestep: float | None = None,
        ee_site: str | None = None,
        robot: RobotSpec = UR5E,
    ):
        # Accept either a bundled scene name or a direct path to any MJCF file.
        if os.path.isfile(scene):
            path = os.path.abspath(scene)
        else:
            path = scene_path(scene)

        self.scene_path = path
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        if timestep is not None:
            self.model.opt.timestep = float(timestep)

        self.robot = robot
        self._ee_site_override = ee_site
        self._discover_arm()
        self._discover_ee_site()
        self.reset_home()

    # ------------------------------------------------------------------ setup
    def _discover_arm(self) -> None:
        """Locate arm joints/DOFs/actuators by name (robust to a gripper)."""
        m = self.model
        joint_ids, qpos_adr, dof_adr = [], [], []
        for name in self.robot.arm_joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                continue
            joint_ids.append(jid)
            qpos_adr.append(int(m.jnt_qposadr[jid]))
            dof_adr.append(int(m.jnt_dofadr[jid]))

        if not joint_ids:
            raise RuntimeError(
                f"No {self.robot.name} arm joints found in model; is this the right scene/robot?"
            )

        self.arm_joint_ids = np.array(joint_ids, dtype=int)
        self.arm_qpos_adr = np.array(qpos_adr, dtype=int)
        self.arm_dof_adr = np.array(dof_adr, dtype=int)
        self.n_arm = len(joint_ids)

        # Actuators whose transmission targets an arm joint, in joint order.
        arm_act = []
        for jid in joint_ids:
            for a in range(m.nu):
                if m.actuator_trnid[a, 0] == jid:
                    arm_act.append(a)
                    break
        self.arm_actuator_ids = np.array(arm_act, dtype=int)

        # Per-joint torque limits. These UR5e actuators are `motor`s
        # (gain=gear=1), so the torque bound comes from ctrlrange, not the
        # (unset) forcerange — using forcerange naively clamps torque to zero.
        limits = []
        for a in self.arm_actuator_ids:
            gear = abs(float(m.actuator_gear[a, 0])) or 1.0
            if m.actuator_forcelimited[a]:
                bound = float(np.abs(m.actuator_forcerange[a]).max())
            elif m.actuator_ctrllimited[a]:
                bound = gear * float(np.abs(m.actuator_ctrlrange[a]).max())
            else:
                bound = np.inf
            limits.append(bound)
        self.torque_limits = np.array(limits, dtype=float)

        # Joint position limits from the model.
        self.joint_limits = m.jnt_range[self.arm_joint_ids].copy()

    def _discover_ee_site(self) -> None:
        m = self.model
        self.ee_site_id = -1
        self.ee_site_name = None
        base = self.robot.ee_site_candidates
        candidates = (self._ee_site_override, *base) if self._ee_site_override else base
        for name in candidates:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid != -1:
                self.ee_site_id = sid
                self.ee_site_name = name
                break
        if self.ee_site_id == -1:
            raise RuntimeError(f"No end-effector site found (tried {base}).")

    # ----------------------------------------------------------------- basics
    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def home_qpos_arm(self) -> np.ndarray:
        """A sane home configuration for the arm joints (from the robot spec)."""
        return np.array(self.robot.home_qpos, dtype=float)

    def reset_home(self) -> None:
        self.reset(self.home_qpos_arm)

    def reset(self, qpos_arm: np.ndarray | None = None) -> None:
        """Reset state; optionally set the arm configuration."""
        mujoco.mj_resetData(self.model, self.data)
        if qpos_arm is not None:
            self.set_arm_qpos(qpos_arm)
        self.forward()

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------------- arm state
    @property
    def qpos_arm(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adr].copy()

    @property
    def qvel_arm(self) -> np.ndarray:
        return self.data.qvel[self.arm_dof_adr].copy()

    def set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[self.arm_qpos_adr] = np.asarray(q, dtype=float)

    def set_arm_qvel(self, v: np.ndarray) -> None:
        self.data.qvel[self.arm_dof_adr] = np.asarray(v, dtype=float)

    # --------------------------------------------------------------- dynamics
    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def ee_rot(self) -> np.ndarray:
        """Rotation matrix of the EE site (3x3); columns are its axes in world."""
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def ee_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """Position and rotation Jacobians of the EE site, arm columns only.

        Returns ``(Jp, Jr)`` each shaped ``(3, n_arm)``.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp[:, self.arm_dof_adr], jacr[:, self.arm_dof_adr]

    def mass_matrix(self) -> np.ndarray:
        """Joint-space inertia matrix M(q) restricted to the arm DOFs."""
        full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, full, self.data.qM)
        idx = self.arm_dof_adr
        return full[np.ix_(idx, idx)]

    def bias_force(self) -> np.ndarray:
        """Coriolis + centrifugal + gravity force on the arm (qfrc_bias)."""
        return self.data.qfrc_bias[self.arm_dof_adr].copy()

    # gravity-compensation torque is exactly the bias force at zero velocity,
    # but using the live bias is the usual feedforward in a torque loop.
    gravity_comp = bias_force

    # ------------------------------------------------------------------- step
    def apply_arm_torque(self, tau: np.ndarray) -> None:
        """Write a torque command to the arm actuators (clipped to limits)."""
        tau = np.clip(np.asarray(tau, dtype=float), -self.torque_limits, self.torque_limits)
        self.data.ctrl[self.arm_actuator_ids] = tau

    def step(self, tau_arm: np.ndarray | None = None) -> None:
        """Optionally apply an arm torque, then advance one timestep."""
        if tau_arm is not None:
            self.apply_arm_torque(tau_arm)
        mujoco.mj_step(self.model, self.data)

    # -------------------------------------------------------------- target viz
    def set_target_marker(self, pos: np.ndarray) -> None:
        """Move the red ``target_marker`` site if the scene has one."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
        if sid != -1:
            self.model.site_pos[sid] = np.asarray(pos, dtype=float)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"World(nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}, "
            f"n_arm={self.n_arm}, ee_site='{self.ee_site_name}', dt={self.timestep})"
        )
