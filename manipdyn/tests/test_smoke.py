"""Foundation smoke tests: the package loads, simulates, controls, renders."""

from __future__ import annotations

import numpy as np
import pytest

from manipdyn.sim import World


def test_world_loads_and_discovers_arm():
    world = World(scene="scene_base")
    assert world.n_arm == 6
    assert world.arm_actuator_ids.shape[0] == 6
    assert world.ee_site_id != -1
    assert np.all(np.isfinite(world.ee_pos))
    # Torque limits come from ctrlrange, not the unset forcerange.
    assert np.allclose(world.torque_limits, [150, 150, 150, 28, 28, 28])


def test_mass_matrix_and_jacobian_shapes():
    world = World(scene="scene_base")
    M = world.mass_matrix()
    assert M.shape == (6, 6)
    assert np.allclose(M, M.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(M) > 0)  # SPD inertia

    jp, jr = world.ee_jacobian()
    assert jp.shape == (3, 6) and jr.shape == (3, 6)


def test_offscreen_render_produces_frame():
    pytest.importorskip("mujoco")
    from manipdyn.render import Recorder

    world = World(scene="scene_base")
    try:
        rec = Recorder(world, width=160, height=120)
    except Exception as exc:  # no GL backend on this runner
        pytest.skip(f"offscreen GL unavailable: {exc}")
    with rec:
        rec.capture()
        assert rec.frames[0].shape == (120, 160, 3)
        assert rec.frames[0].dtype == np.uint8
