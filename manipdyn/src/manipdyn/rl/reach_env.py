"""A Gymnasium reaching environment for the UR5e.

The agent commands joint-position targets (tracked by an internal PD +
gravity-compensation loop) to drive the end-effector to a randomly sampled,
reachable Cartesian goal.

* **Observation** (18-d): arm position (6), velocity (6), end-effector position
  (3), and the goal vector ``goal - ee`` (3).
* **Action** (6-d): joint-position targets in ``[-1, 1]`` (mapped around the
  home configuration) tracked by an internal PD loop — the position-style
  action space standard in manipulation RL, far more sample-efficient than
  commanding raw torques.
* **Reward**: potential-based progress with a small control penalty and a
  success bonus; the episode terminates on reaching the goal.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from manipdyn.sim import World


class ReachEnv(gym.Env):
    metadata = {"render_modes": []}

    #: Forward-reaching nominal config; goals are sampled around its FK so the
    #: task is consistent and reachable (not scattered behind/under the base).
    GOAL_NOMINAL = np.array([0.0, -1.0, 1.0, -1.5, -1.5, 0.0])

    def __init__(
        self,
        scene: str = "scene_base",
        horizon: int = 150,
        decimation: int = 10,
        tol: float = 0.03,
        action_range: float = 2.0,
        kp: float = 150.0,
        kd: float = 25.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.world = World(scene=scene)
        self.horizon = horizon
        self.decimation = decimation
        self.tol = tol
        # Position-style action space: actions are joint targets about home,
        # tracked by an internal PD + gravity compensation.
        self.home = self.world.home_qpos_arm.copy()
        self.action_range = action_range
        self.kp = kp
        self.kd = kd
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (self.world.n_arm,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32)
        self._steps = 0
        self._prev_dist = 0.0
        self.goal = np.zeros(3)

    # ------------------------------------------------------------------
    def _sample_goal(self) -> np.ndarray:
        # FK of a config near the forward nominal -> reachable, front-region goal.
        q = self.GOAL_NOMINAL + self.rng.uniform(-0.5, 0.5, size=self.world.n_arm)
        self.world.set_arm_qpos(q)
        self.world.forward()
        return self.world.ee_pos.copy()

    def _obs(self) -> np.ndarray:
        q = self.world.qpos_arm
        v = self.world.qvel_arm
        ee = self.world.ee_pos
        return np.concatenate([q, v, ee, self.goal - ee]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        self.world.reset(self.world.home_qpos_arm)
        self.goal = self._sample_goal()
        self.world.reset(self.world.home_qpos_arm)
        self.world.set_target_marker(self.goal)
        self._steps = 0
        self._prev_dist = float(np.linalg.norm(self.goal - self.world.ee_pos))
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        q_des = self.home + action * self.action_range
        for _ in range(self.decimation):
            q, v = self.world.qpos_arm, self.world.qvel_arm
            tau = self.kp * (q_des - q) - self.kd * v + self.world.bias_force()
            self.world.step(tau)

        self._steps += 1
        dist = float(np.linalg.norm(self.goal - self.world.ee_pos))
        # Potential-based progress reward (dense, fast to learn) + penalties.
        reward = 10.0 * (self._prev_dist - dist) - 0.1 * dist - 0.01 * float(action @ action)
        self._prev_dist = dist
        terminated = dist < self.tol
        if terminated:
            reward += 10.0
        truncated = self._steps >= self.horizon
        return self._obs(), float(reward), terminated, truncated, {"distance": dist}
