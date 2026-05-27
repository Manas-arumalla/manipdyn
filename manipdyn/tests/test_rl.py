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
