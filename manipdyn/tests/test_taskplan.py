"""Symbolic (STRIPS) task planning over the pick/place/stack domain."""

from __future__ import annotations

from manipdyn.taskplan import all_on_table, blocks_world_actions, plan, stacked_state

BLOCKS = ["a", "b", "c"]
ACTIONS = blocks_world_actions(BLOCKS)


def _valid(init, goal, steps) -> bool:
    """A plan is valid if every action is applicable in turn and the goal holds."""
    state = frozenset(init)
    for action in steps:
        if not action.applicable(state):
            return False
        state = action.apply(state)
    return frozenset(goal) <= state


def test_builds_a_tower():
    init = all_on_table(BLOCKS)
    goal = {("on", "a", "b"), ("on", "b", "c")}
    steps = plan(init, goal, ACTIONS)
    assert steps and _valid(init, goal, steps)


def test_reverses_a_stack():
    init = stacked_state(["a", "b", "c"])  # a on the bottom
    goal = {("on", "b", "c"), ("on", "a", "b")}  # reversed
    steps = plan(init, goal, ACTIONS)
    assert steps and _valid(init, goal, steps)
    # Reversing a three-block stack cannot be done in fewer than a few moves.
    assert len(steps) >= 4


def test_move_block_to_table():
    init = stacked_state(["a", "b"])  # b on a
    goal = {("ontable", "b")}
    steps = plan(init, goal, ACTIONS)
    assert steps and _valid(init, goal, steps)


def test_goal_already_satisfied_is_empty_plan():
    init = all_on_table(BLOCKS)
    assert plan(init, {("ontable", "a")}, ACTIONS) == []


def test_unsolvable_goal_returns_none():
    init = all_on_table(BLOCKS)
    assert plan(init, {("on", "a", "a")}, ACTIONS) is None
