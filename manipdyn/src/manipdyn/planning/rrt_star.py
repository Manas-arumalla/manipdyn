"""RRT* and Informed RRT* (asymptotically optimal planners).

* **RRT\*** — like RRT but chooses the lowest-cost parent within a radius and
  *rewires* nearby nodes, so the path cost converges toward optimal as samples
  accumulate (anytime).
* **Informed RRT\*** — once a solution of cost ``c_best`` exists, restricts
  sampling to the prolate-hyperspheroid (ellipsoid) that could possibly improve
  it, dramatically accelerating convergence to the optimum.
"""

from __future__ import annotations

import numpy as np

from manipdyn.planning.base import Node, Planner, reconstruct


class RRTStar(Planner):
    name = "rrt_star"

    def __init__(self, *args, connect_radius: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.connect_radius = connect_radius
        self._c_best = np.inf

    def _goal_or_sample(self, q_goal: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.goal_bias:
            return q_goal
        return self._sample_free()

    def _sample_free(self) -> np.ndarray:
        """Sampling distribution hook (overridden by Informed RRT*)."""
        return self.sample()

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        if not self.endpoints_valid(q_start, q_goal):
            return None

        self._c_best = np.inf
        tree = [Node(q_start, cost=0.0)]
        best_goal: Node | None = None

        for _ in range(self.max_iter):
            q_rand = self._goal_or_sample(q_goal)
            nearest = min(tree, key=lambda n: self.distance(n.q, q_rand))
            q_new = self.steer(nearest.q, q_rand)
            if self.checker.in_collision(q_new) or not self.edge_valid(nearest.q, q_new):
                continue

            neighbors = [n for n in tree if self.distance(n.q, q_new) < self.connect_radius]

            # Choose the lowest-cost collision-free parent.
            parent, c_min = nearest, nearest.cost + self.distance(nearest.q, q_new)
            for nb in neighbors:
                c = nb.cost + self.distance(nb.q, q_new)
                if c < c_min and self.edge_valid(nb.q, q_new):
                    parent, c_min = nb, c
            node = Node(q_new, parent, c_min)
            tree.append(node)

            # Rewire neighbors through the new node if cheaper.
            for nb in neighbors:
                c = node.cost + self.distance(node.q, nb.q)
                if c < nb.cost and self.edge_valid(node.q, nb.q):
                    nb.parent, nb.cost = node, c

            if self.distance(q_new, q_goal) <= self.step_size and self.edge_valid(q_new, q_goal):
                goal_cost = node.cost + self.distance(q_new, q_goal)
                if goal_cost < self._c_best:
                    best_goal, self._c_best = node, goal_cost

        if best_goal is None:
            return None
        return np.array(reconstruct(best_goal) + [q_goal])


class InformedRRTStar(RRTStar):
    name = "informed_rrt_star"

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        self._c_min = self.distance(q_start, q_goal)
        self._center = 0.5 * (q_start + q_goal)
        self._C = self._world_rotation(q_start, q_goal)
        return super().plan(q_start, q_goal)

    def _sample_free(self) -> np.ndarray:
        if not np.isfinite(self._c_best):
            return self.sample()
        return self._sample_informed()

    def _world_rotation(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        """Rotation aligning the ellipsoid's major axis with start->goal."""
        diff = q_goal - q_start
        norm = np.linalg.norm(diff)
        if norm < 1e-9:
            return np.eye(self.dim)
        a1 = diff / norm
        e1 = np.zeros(self.dim)
        e1[0] = 1.0
        U, _, Vt = np.linalg.svd(np.outer(a1, e1))
        d = np.ones(self.dim)
        d[-1] = np.linalg.det(U) * np.linalg.det(Vt)
        return U @ np.diag(d) @ Vt

    def _sample_informed(self) -> np.ndarray:
        c_best = max(self._c_best, self._c_min)
        r1 = c_best / 2.0
        ri = np.sqrt(max(c_best**2 - self._c_min**2, 0.0)) / 2.0
        L = np.diag([r1] + [ri] * (self.dim - 1))
        q = self._C @ L @ self._unit_ball() + self._center
        return np.clip(q, self.limits[:, 0], self.limits[:, 1])

    def _unit_ball(self) -> np.ndarray:
        x = self.rng.normal(size=self.dim)
        x /= np.linalg.norm(x)
        return x * self.rng.random() ** (1.0 / self.dim)
