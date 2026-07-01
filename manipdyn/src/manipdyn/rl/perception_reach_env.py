"""A vision-conditioned reaching env: reach a cube the agent *perceives*.

Like :class:`~manipdyn.rl.reach_env.ReachEnv`, but the goal is not privileged
simulator state — each episode the cube is placed at a random spot and its
position is estimated from the overhead RGB-D camera (see
:mod:`manipdyn.perception`). The policy reaches the *perceived* cube, closing the
perception -> control loop under domain randomization.

* **Observation** (18-d): arm position (6), velocity (6), end-effector position
  (3), and the perceived goal vector ``goal - ee`` (3) — same shape as
  ``ReachEnv``, so the same policy architecture transfers.
* **Action** (6-d): joint-position targets about home, tracked by an internal
  PD + gravity-compensation loop.
* **Reward**: potential-based progress toward the perceived goal, a small control
  penalty, and a success bonus; the episode ends on reaching it.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from manipdyn.perception import Camera, sense_object_pose
from manipdyn.sim import World


class PerceptionReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scene: str = "scene_pick",
        horizon: int = 150,
        decimation: int = 10,
        tol: float = 0.04,
        action_range: float = 0.6,
        kp: float = 150.0,
        kd: float = 25.0,
        cube_range: float = 0.05,
        segmentation: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.world = World(scene=scene, ee_site="pinch")
        self.camera = Camera(self.world, "overhead", width=320, height=240)
        self.horizon = horizon
        self.decimation = decimation
        self.tol = tol
        self.action_range = action_range
        self.kp = kp
        self.kd = kd
        self.cube_range = cube_range
        self.segmentation = segmentation
        self.home = self.world.home_qpos_arm.copy()
        # A raised "ready" pose that clears the overhead camera's view of the cube.
        self.ready = self.home.copy()
        self.ready[1] -= 0.5
        self.rng = np.random.default_rng(seed)

        m = self.world.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self._obj_qadr = int(m.jnt_qposadr[jid])
        self._base_xy = m.qpos0[self._obj_qadr : self._obj_qadr + 2].copy()

        self.action_space = spaces.Box(-1.0, 1.0, (self.world.n_arm,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32)
        self._steps = 0
        self._prev_dist = 0.0
        self.goal = np.zeros(3)

    def _obs(self) -> np.ndarray:
        q, v, ee = self.world.qpos_arm, self.world.qvel_arm, self.world.ee_pos
        return np.concatenate([q, v, ee, self.goal - ee]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            super().reset(seed=seed)

        # Randomize the cube on the table (domain randomization).
        xy = self._base_xy + self.rng.uniform(-self.cube_range, self.cube_range, 2)
        self.world.model.qpos0[self._obj_qadr : self._obj_qadr + 2] = xy
        self.world.reset(self.ready)  # look from the raised pose
        self.world.forward()

        # Perceive the cube -> goal just above its top face.
        est = sense_object_pose(self.camera, segmentation=self.segmentation)
        self.goal = np.array([est.top_xy[0], est.top_xy[1], est.top_z + 0.03])

        self.world.reset(self.ready)  # start the episode from the ready pose
        self._steps = 0
        self._prev_dist = float(np.linalg.norm(self.goal - self.world.ee_pos))
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        q_des = self.home + action * self.action_range
        for _ in range(self.decimation):
            q, v = self.world.qpos_arm, self.world.qvel_arm
            self.world.step(self.kp * (q_des - q) - self.kd * v + self.world.bias_force())

        self._steps += 1
        dist = float(np.linalg.norm(self.goal - self.world.ee_pos))
        reward = 10.0 * (self._prev_dist - dist) - 0.1 * dist - 0.01 * float(action @ action)
        self._prev_dist = dist
        terminated = dist < self.tol
        if terminated:
            reward += 10.0
        truncated = self._steps >= self.horizon
        return self._obs(), float(reward), terminated, truncated, {"distance": dist}
