"""Collision checking against the MuJoCo scene.

Runs on a private ``MjData`` (so the live sim is never disturbed), sets only
the arm DOFs of an otherwise-current configuration, and reports a collision
when any active contact penetrates beyond a margin. Edge checking interpolates
between configurations at a fixed joint-space resolution.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from manipdyn.sim.world import World


class CollisionChecker:
    def __init__(
        self,
        world: World,
        margin: float = 1e-3,
        ignore_geoms: Iterable[str] | None = None,
    ):
        """Collision checker for the arm against the scene.

        ``margin``: a contact counts as a collision when its penetration depth
        exceeds ``margin`` (a small positive clearance treats near-touching as
        unsafe). ``ignore_geoms``: geom names to exclude from collision (e.g. a
        graspable object the arm is allowed to approach/carry)."""
        self.model = world.model
        self._d = mujoco.MjData(world.model)
        self.arm_qpos_adr = world.arm_qpos_adr
        self.margin = margin
        self._base_qpos = world.data.qpos.copy()
        self._ignore_ids: set[int] = set()
        for name in ignore_geoms or ():
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self._ignore_ids.add(gid)

    def in_collision(self, q_arm: np.ndarray) -> bool:
        d = self._d
        d.qpos[:] = self._base_qpos
        d.qpos[self.arm_qpos_adr] = q_arm
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self._ignore_ids or c.geom2 in self._ignore_ids:
                continue
            if c.dist < self.margin:
                return True
        return False

    def edge_in_collision(self, q1: np.ndarray, q2: np.ndarray, resolution: float = 0.05) -> bool:
        """True if any interpolated configuration on the segment collides."""
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        n = max(2, int(np.linalg.norm(q2 - q1) / resolution) + 1)
        for alpha in np.linspace(0.0, 1.0, n):
            if self.in_collision(q1 * (1.0 - alpha) + q2 * alpha):
                return True
        return False
