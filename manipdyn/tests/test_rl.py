"""The Gymnasium reaching environment conforms to the Gym API."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from manipdyn.rl import ReachEnv  # noqa: E402


def test_reach_env_api():
    env = ReachEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (18,) and obs.dtype == np.float32
    assert env.action_space.shape == (6,)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (18,)
    assert isinstance(reward, float)
    assert "distance" in info


def test_reach_env_episode_truncates():
    env = ReachEnv(seed=1, horizon=20)
    env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(6, dtype=np.float32))
        done = term or trunc
        steps += 1
    assert steps <= 20


def test_reach_env_passes_sb3_checker():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    check_env(ReachEnv(seed=0), warn=True)


def _perception_env(seed=0):
    from manipdyn.rl import PerceptionReachEnv

    try:
        return PerceptionReachEnv(seed=seed)
    except Exception as exc:  # no GL backend for the camera
        pytest.skip(f"offscreen GL unavailable: {exc}")


def test_perception_reach_env_goal_comes_from_vision():
    import mujoco

    env = _perception_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (18,) and obs.dtype == np.float32
    oid = mujoco.mj_name2id(env.world.model, mujoco.mjtObj.mjOBJ_BODY, "object")
    true_xy = env.world.data.xpos[oid][:2]
    # The goal is the perceived cube position, close to the true one.
    assert np.linalg.norm(env.goal[:2] - true_xy) < 0.02


def test_perception_reach_env_passes_sb3_checker():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    check_env(_perception_env(seed=0), warn=True)
