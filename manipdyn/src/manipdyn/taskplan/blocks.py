"""A pick-and-place / stacking domain for the STRIPS planner.

The classic blocks-world operators, which map directly onto the manipulator's
motion primitives: ``pickup``/``putdown`` are a grasp and a table place,
``stack``/``unstack`` add and remove an object on top of another. Feed the
grounded actions to :func:`manipdyn.taskplan.plan` with an initial state and a
goal to get an ordered manipulation sequence.
"""

from __future__ import annotations

from manipdyn.taskplan.strips import Action


def blocks_world_actions(blocks: list[str]) -> list[Action]:
    """All grounded pick/place/stack/unstack actions for the given blocks."""
    actions: list[Action] = []
    for x in blocks:
        actions.append(
            Action(
                f"pickup({x})",
                pre={("clear", x), ("ontable", x), ("handempty",)},
                add={("holding", x)},
                delete={("clear", x), ("ontable", x), ("handempty",)},
            )
        )
        actions.append(
            Action(
                f"putdown({x})",
                pre={("holding", x)},
                add={("clear", x), ("ontable", x), ("handempty",)},
                delete={("holding", x)},
            )
        )
        for y in blocks:
            if x == y:
                continue
            actions.append(
                Action(
                    f"stack({x},{y})",
                    pre={("holding", x), ("clear", y)},
                    add={("on", x, y), ("clear", x), ("handempty",)},
                    delete={("holding", x), ("clear", y)},
                )
            )
            actions.append(
                Action(
                    f"unstack({x},{y})",
                    pre={("on", x, y), ("clear", x), ("handempty",)},
                    add={("holding", x), ("clear", y)},
                    delete={("on", x, y), ("clear", x), ("handempty",)},
                )
            )
    return actions


def stacked_state(order: list[str]) -> set[tuple]:
    """State predicates for a single stack, ``order[0]`` on the table upward."""
    preds: set[tuple] = {("handempty",), ("ontable", order[0]), ("clear", order[-1])}
    for lower, upper in zip(order, order[1:], strict=False):
        preds.add(("on", upper, lower))
    return preds


def all_on_table(blocks: list[str]) -> set[tuple]:
    """State predicates with every block clear and on the table."""
    preds: set[tuple] = {("handempty",)}
    for b in blocks:
        preds.add(("ontable", b))
        preds.add(("clear", b))
    return preds
