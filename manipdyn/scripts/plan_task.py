"""Task-planning demo: turn a declarative goal into a manipulation sequence.

Describes a blocks-world state and goal, then prints the ordered pick/place/stack
plan the STRIPS layer produces. Each symbolic action maps to a motion primitive
(a grasp, a table place, a stack) executed by the geometric pipeline.

Run from the manipdyn/ directory:
    python scripts/plan_task.py
"""

from __future__ import annotations

from manipdyn.taskplan import all_on_table, blocks_world_actions, plan, stacked_state


def show(title: str, init, goal) -> None:
    steps = plan(init, goal, blocks_world_actions(["a", "b", "c"]))
    print(f"\n{title}")
    if not steps:
        print("  (goal already satisfied)" if steps == [] else "  (no plan found)")
        return
    for i, action in enumerate(steps, 1):
        print(f"  {i}. {action}")


def main() -> None:
    blocks = ["a", "b", "c"]
    show(
        "Build a tower a-on-b-on-c from three blocks on the table:",
        all_on_table(blocks),
        {("on", "a", "b"), ("on", "b", "c")},
    )
    show(
        "Reverse a stack (a at the bottom -> a on top):",
        stacked_state(["a", "b", "c"]),
        {("on", "b", "c"), ("on", "a", "b")},
    )


if __name__ == "__main__":
    main()
