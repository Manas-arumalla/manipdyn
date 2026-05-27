"""RRT and RRT-Connect.

* **RRT** — grow a single tree from the start, biased toward the goal.
* **RRT-Connect** — grow two trees (from start and goal) and repeatedly try to
  connect them. Bidirectional + greedy ``connect`` makes it dramatically faster
  than plain RRT, and it is the de-facto default for manipulators.
"""

from __future__ import annotations

import numpy as np

from manipdyn.planning.base import Node, Planner, reconstruct


class RRT(Planner):
    name = "rrt"

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        if not self.endpoints_valid(q_start, q_goal):
            return None

        tree = [Node(q_start)]
        for _ in range(self.max_iter):
            q_rand = q_goal if self.rng.random() < self.goal_bias else self.sample()
            nearest = min(tree, key=lambda n: self.distance(n.q, q_rand))
            q_new = self.steer(nearest.q, q_rand)

            if self.checker.in_collision(q_new) or not self.edge_valid(nearest.q, q_new):
                continue
            node = Node(q_new, nearest)
            tree.append(node)

            if self.distance(q_new, q_goal) <= self.step_size and self.edge_valid(q_new, q_goal):
                return np.array(reconstruct(node) + [q_goal])
        return None


class RRTConnect(Planner):
    name = "rrt_connect"

    _TRAPPED, _ADVANCED, _REACHED = 0, 1, 2

    def _extend(self, tree: list[Node], q_target: np.ndarray) -> tuple[int, Node]:
        nearest = min(tree, key=lambda n: self.distance(n.q, q_target))
        q_new = self.steer(nearest.q, q_target)
        if self.checker.in_collision(q_new) or not self.edge_valid(nearest.q, q_new):
            return self._TRAPPED, nearest
        node = Node(q_new, nearest)
        tree.append(node)
        reached = self.distance(q_new, q_target) < 1e-9
        return (self._REACHED if reached else self._ADVANCED), node

    def _connect(self, tree: list[Node], q_target: np.ndarray) -> tuple[int, Node]:
        status, node = self._ADVANCED, None
        while status == self._ADVANCED:
            status, node = self._extend(tree, q_target)
        return status, node

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        if not self.endpoints_valid(q_start, q_goal):
            return None

        start_tree, goal_tree = [Node(q_start)], [Node(q_goal)]
        tree_a, tree_b, a_is_start = start_tree, goal_tree, True

        for _ in range(self.max_iter):
            status, node_a = self._extend(tree_a, self.sample())
            if status != self._TRAPPED:
                status_b, node_b = self._connect(tree_b, node_a.q)
                if status_b == self._REACHED:
                    s_node, g_node = (node_a, node_b) if a_is_start else (node_b, node_a)
                    start_side = reconstruct(s_node)  # start -> connection
                    goal_side = reconstruct(g_node)[::-1]  # connection -> goal
                    return np.array(start_side + goal_side[1:])
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start
        return None
