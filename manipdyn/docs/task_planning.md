# Task planning

A lightweight **symbolic (STRIPS) planning** layer sits above the geometric
motion planner. Describe the world with predicates and a goal, and it returns an
ordered sequence of manipulation actions — declarative goals instead of
hand-scripted step lists, with no external planner or middleware.

```python
from manipdyn.taskplan import plan, blocks_world_actions, all_on_table

blocks = ["a", "b", "c"]
init = all_on_table(blocks)                     # all three on the table
goal = {("on", "a", "b"), ("on", "b", "c")}     # a on b on c
steps = plan(init, goal, blocks_world_actions(blocks))
# -> pickup(b), stack(b,c), pickup(a), stack(a,b)
```

## How it works

* A **predicate** is a tuple such as `("on", "a", "b")` or `("handempty",)`; a
  **state** is a frozen set of predicates.
* An `Action` carries preconditions and add/delete effects. The pick/place
  domain (`blocks_world_actions`) provides the grounded operators — `pickup`,
  `putdown`, `stack`, `unstack` — which map directly onto the manipulator's
  motion primitives (a grasp, a table place, a stack).
* `plan` searches from the initial state to the goal with A*, using the number
  of unmet goal predicates as the heuristic. It returns the action list, `[]`
  when the goal already holds, or `None` when no plan exists.

## From symbols to motion

Each symbolic action corresponds to a geometric routine already in the library:
`pickup`/`putdown` are the grasp and place of the [pick-and-place](tasks.md)
pipeline, and object poses come from [perception](perception.md). The planner
decides *what* to do and in *what order*; the motion layer decides *how*. Because
the plan is recomputed from the current state, a failed action can be re-planned
around rather than aborting the task.

Run `python scripts/plan_task.py` to print plans for building and reversing a
stack.
