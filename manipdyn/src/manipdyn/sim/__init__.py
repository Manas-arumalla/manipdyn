"""Simulation layer: a thin, well-behaved wrapper around a MuJoCo model."""

from manipdyn.sim.world import ARM_JOINT_NAMES, World

__all__ = ["World", "ARM_JOINT_NAMES"]
