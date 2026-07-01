"""Reinforcement-learning interface: a Gymnasium env over the manipulator.

Wraps :class:`~manipdyn.sim.world.World` as a standard ``gymnasium.Env`` so the
same physics the classical controllers use can train an RL policy (SAC/PPO via
Stable-Baselines3), and be compared against them. Requires the ``rl`` extra:
``pip install -e '.[rl]'``.
"""

from manipdyn.rl.reach_env import ReachEnv

__all__ = ["ReachEnv", "PerceptionReachEnv"]


def __getattr__(name: str):
    # Lazy import: PerceptionReachEnv needs a GL backend (camera), so only load
    # it on demand, keeping ``import manipdyn.rl`` light for headless callers.
    if name == "PerceptionReachEnv":
        from manipdyn.rl.perception_reach_env import PerceptionReachEnv

        return PerceptionReachEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
