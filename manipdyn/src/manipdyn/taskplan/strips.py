"""A minimal STRIPS task planner (A* over symbolic states).

Task-level planning that sits above the geometric motion planner: describe the
world with predicates and a goal, and get back an ordered list of symbolic
actions (pick, place, stack, ...) that achieves it. Pure Python, no dependencies.

A predicate is a tuple such as ``("on", "a", "b")`` or ``("handempty",)``; a
state is a frozen set of predicates. :func:`plan` searches with A* using the
number of unmet goal predicates as an admissible-in-practice heuristic.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

Predicate = tuple
State = frozenset


@dataclass
class Action:
    """A grounded STRIPS action: preconditions and add/delete effects."""

    name: str
    pre: frozenset
    add: frozenset
    delete: frozenset

    def __post_init__(self) -> None:
        self.pre = frozenset(self.pre)
        self.add = frozenset(self.add)
        self.delete = frozenset(self.delete)

    def applicable(self, state: State) -> bool:
        return self.pre <= state

    def apply(self, state: State) -> State:
        return (state - self.delete) | self.add

    def __str__(self) -> str:
        return self.name


def plan(
    init: Iterable[Predicate],
    goal: Iterable[Predicate],
    actions: list[Action],
    max_expansions: int = 200_000,
) -> list[Action] | None:
    """Return an ordered list of actions from ``init`` to ``goal``, or ``None``.

    ``[]`` means the goal already holds. ``None`` means no plan was found within
    ``max_expansions`` node expansions.
    """
    init, goal = frozenset(init), frozenset(goal)
    if goal <= init:
        return []

    def heuristic(state: State) -> int:
        return len(goal - state)

    counter = itertools.count()
    frontier = [(heuristic(init), next(counter), init, [])]
    best_cost = {init: 0}
    expansions = 0

    while frontier and expansions < max_expansions:
        _, _, state, path = heapq.heappop(frontier)
        if goal <= state:
            return path
        expansions += 1
        g = len(path) + 1
        for action in actions:
            if action.pre <= state:
                nxt = (state - action.delete) | action.add
                if nxt not in best_cost or g < best_cost[nxt]:
                    best_cost[nxt] = g
                    heapq.heappush(
                        frontier, (g + heuristic(nxt), next(counter), nxt, [*path, action])
                    )
    return None
