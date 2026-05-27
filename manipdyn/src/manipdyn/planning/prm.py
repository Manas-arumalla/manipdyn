"""Probabilistic Roadmap (PRM) — a multi-query planner.

Builds a reusable roadmap once (sample free configurations, connect k-nearest
neighbors with collision-free edges), then answers many start/goal queries
cheaply by connecting them to the roadmap and running Dijkstra. Ideal when the
environment is static and many plans are needed.
"""

from __future__ import annotations

import heapq

import numpy as np

from manipdyn.planning.base import Planner


class PRM(Planner):
    name = "prm"

    def __init__(self, *args, n_samples: int = 300, k_neighbors: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.samples: np.ndarray | None = None
        self.adj: dict[int, list[tuple[int, float]]] = {}

    def build_roadmap(self) -> None:
        samples = []
        while len(samples) < self.n_samples:
            q = self.sample()
            if not self.checker.in_collision(q):
                samples.append(q)
        samples = np.array(samples)

        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(samples))}
        for i in range(len(samples)):
            dists = np.linalg.norm(samples - samples[i], axis=1)
            order = np.argsort(dists)
            connected = 0
            for j in order[1:]:
                if connected >= self.k_neighbors:
                    break
                if self.edge_valid(samples[i], samples[j]):
                    adj[i].append((int(j), float(dists[j])))
                    connected += 1
        self.samples, self.adj = samples, adj

    def _connect(self, q: np.ndarray) -> int:
        """Index of the nearest roadmap node reachable by a free edge, or -1."""
        dists = np.linalg.norm(self.samples - q, axis=1)
        for j in np.argsort(dists):
            if self.edge_valid(q, self.samples[j]):
                return int(j)
        return -1

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        if not self.endpoints_valid(q_start, q_goal):
            return None
        if self.samples is None:
            self.build_roadmap()

        s, g = self._connect(q_start), self._connect(q_goal)
        if s < 0 or g < 0:
            return None

        # Dijkstra over the roadmap.
        pq: list[tuple[float, int]] = [(0.0, s)]
        prev: dict[int, int] = {s: -1}
        cost = {s: 0.0}
        while pq:
            c, u = heapq.heappop(pq)
            if u == g:
                break
            if c > cost.get(u, np.inf):
                continue
            for v, w in self.adj[u]:
                nc = c + w
                if nc < cost.get(v, np.inf):
                    cost[v], prev[v] = nc, u
                    heapq.heappush(pq, (nc, v))

        if g not in prev:
            return None
        chain, u = [], g
        while u != -1:
            chain.append(u)
            u = prev[u]
        chain.reverse()
        return np.array([q_start, *[self.samples[i] for i in chain], q_goal])
