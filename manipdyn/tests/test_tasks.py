"""Pick-and-place: both the weld hold and the real contact (friction) grasp."""

from __future__ import annotations

import mujoco
import pytest

from manipdyn.sim import World
from manipdyn.tasks import pick_place


def _run_pick_place(use_weld: bool) -> dict:
    world = World(scene=pick_place.SCENE, ee_site=pick_place.EE_SITE)
    plan = pick_place.solve(world)
    last = None
    lifted = False
    for tel in pick_place.run(world, plan, use_weld=use_weld):
        last = tel
        if tel["cube_pos"][2] > 0.45:  # the cube left the table at some point
            lifted = True
    last["lifted"] = lifted
    return last


@pytest.mark.parametrize("use_weld", [True, False])
def test_pick_place_delivers_cube(use_weld):
    tel = _run_pick_place(use_weld)
    assert tel["lifted"], "cube was never lifted off the table"
    assert tel["cube_pos"][2] > 0.30, "cube ended on the floor"
    assert tel["cube_tilt_deg"] < 25, "cube toppled"
    assert tel["place_err_mm"] < 60, f"placed {tel['place_err_mm']:.0f} mm off target"


def test_contact_grasp_holds_by_friction():
    """With use_weld=False the cube is held by the fingers, not a weld constraint."""
    world = World(scene=pick_place.SCENE, ee_site=pick_place.EE_SITE)
    wid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
    plan = pick_place.solve(world)
    max_z = 0.0
    for tel in pick_place.run(world, plan, use_weld=False):
        max_z = max(max_z, tel["cube_pos"][2])
        assert world.data.eq_active[wid] == 0  # the weld is never activated
    assert max_z > 0.45  # yet the cube was carried aloft — friction did the work
