"""The uniform interface shared by the whole control zoo.

Every controller maps *(live world state, desired set-point)* to a **complete
arm joint-torque command** — i.e. each controller is responsible for its own
gravity / bias compensation. This keeps the benchmark harness trivial::

    tau = controller.compute(target)
    world.step(tau)

and lets joint-space methods (PID, computed-torque, LQR, MPPI) and task-space
methods (Cartesian impedance, OSC) live behind one call. A controller declares
its :attr:`target_space` so the harness/GUI know whether to populate the
joint (``q``) or Cartesian (``x``) fields of the :class:`Target`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from manipdyn.sim.world import World


@dataclass
class Target:
    """A desired set-point. Fields are filled in as available.

    Joint-space controllers read ``q`` (and optionally ``v``, ``a``);
    task-space controllers read ``x`` (and optionally ``xdot``). The harness
    typically fills both so any controller can consume the same object.
    """

    q: np.ndarray | None = None
    v: np.ndarray | None = None
    a: np.ndarray | None = None
    x: np.ndarray | None = None
    xdot: np.ndarray | None = None
    #: Desired end-effector orientation as a 3x3 rotation matrix. Optional;
    #: task-space controllers that support it track orientation when it is set
    #: and stay position-only when it is ``None``.
    R: np.ndarray | None = None


class Controller(ABC):
    """Abstract base for all torque-producing controllers."""

    #: Human-readable id used in benchmark tables and plots.
    name: str = "controller"
    #: Which field of :class:`Target` this controller consumes.
    target_space: Literal["joint", "cartesian"] = "joint"

    def __init__(self, world: World):
        self.world = world
        self.n_arm = world.n_arm

    def reset(self) -> None:  # noqa: B027 - optional override hook, intentionally not abstract
        """Clear internal state (integrators, schedules). Default: no-op."""

    @abstractmethod
    def compute(self, target: Target) -> np.ndarray:
        """Return the full arm joint-torque command (shape ``(n_arm,)``)."""
        raise NotImplementedError
