"""Planner interface and shared sampling-based-planning utilities.

Concrete planners (RRT, RRT*, RRT-Connect, Informed RRT*, PRM) subclass
:class:`Planner` and reuse the sampling / steering / validity helpers here.
A plan is a collision-free joint-space path returned as an ``(N, n_arm)``
array, or ``None`` if no path was found.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from manipdyn.planning.collision import CollisionChecker

if TYPE_CHECKING:
    from manipdyn.sim.world import World


class Node:
    """A tree node for sampling-based planners."""

    __slots__ = ("q", "parent", "cost")

    def __init__(self, q: np.ndarray, parent: Node | None = None, cost: float = 0.0):
        self.q = np.asarray(q, dtype=float)
        self.parent = parent
        self.cost = cost


def reconstruct(node: Node) -> list[np.ndarray]:
    """Walk parent pointers from ``node`` back to the root (root-first)."""
    path = []
    while node is not None:
        path.append(node.q)
        node = node.parent
    return path[::-1]


class Planner(ABC):
    name: str = "planner"

    def __init__(
        self,
        world: World,
        checker: CollisionChecker | None = None,
        step_size: float = 0.2,
        max_iter: int = 3000,
        goal_bias: float = 0.1,
        edge_resolution: float = 0.05,
        seed: int | None = None,
    ):
        self.world = world
        self.checker = checker or CollisionChecker(world)
        self.limits = np.asarray(world.joint_limits, dtype=float)  # (n_arm, 2)
        self.dim = world.n_arm
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.edge_resolution = edge_resolution
        self.rng = np.random.default_rng(seed)

    # -- sampling-planning primitives ------------------------------------
    def sample(self) -> np.ndarray:
        return self.rng.uniform(self.limits[:, 0], self.limits[:, 1])

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        delta = q_to - q_from
        dist = np.linalg.norm(delta)
        if dist <= self.step_size:
            return q_to.copy()
        return q_from + (delta / dist) * self.step_size

    def edge_valid(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        return not self.checker.edge_in_collision(q1, q2, self.edge_resolution)

    def endpoints_valid(self, q_start: np.ndarray, q_goal: np.ndarray) -> bool:
        return not (self.checker.in_collision(q_start) or self.checker.in_collision(q_goal))

    @abstractmethod
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        """Return a collision-free path ``(N, n_arm)`` or ``None``."""
        raise NotImplementedError
