"""Reinforcement-learning interface: a Gymnasium env over the manipulator.

Wraps :class:`~manipdyn.sim.world.World` as a standard ``gymnasium.Env`` so the
same physics the classical controllers use can train an RL policy (SAC/PPO via
Stable-Baselines3), and be compared against them. Requires the ``rl`` extra:
``pip install -e '.[rl]'``.
"""

from manipdyn.rl.reach_env import ReachEnv

__all__ = ["ReachEnv"]
