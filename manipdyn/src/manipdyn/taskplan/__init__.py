"""Task-level (symbolic) planning above the geometric motion planner.

Describe the world with predicates and a goal; get back an ordered sequence of
symbolic actions (pick, place, stack) to achieve it. A lightweight STRIPS layer
that turns declarative goals into a manipulation sequence — without any external
planner or middleware.

    from manipdyn.taskplan import plan, blocks_world_actions, all_on_table

    blocks = ["a", "b", "c"]
    init = all_on_table(blocks)
    goal = {("on", "a", "b"), ("on", "b", "c")}       # a on b on c
    steps = plan(init, goal, blocks_world_actions(blocks))
"""

from manipdyn.taskplan.blocks import all_on_table, blocks_world_actions, stacked_state
from manipdyn.taskplan.strips import Action, Predicate, State, plan

__all__ = [
    "Action",
    "Predicate",
    "State",
    "plan",
    "blocks_world_actions",
    "stacked_state",
    "all_on_table",
]
