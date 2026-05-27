"""Time-optimal parameterization must respect limits and boundary conditions."""

from __future__ import annotations

import numpy as np

from manipdyn.trajectory import parameterize_time_optimal

PATH = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.3, 0.4, 0.0, 0.0, 0.0],
        [1.0, -0.6, 0.8, -0.2, 0.0, 0.0],
        [1.2, -1.0, 1.2, -0.5, 0.3, 0.1],
    ]
)
VMAX = np.full(6, 2.0)
AMAX = np.full(6, 5.0)


def test_timing_respects_velocity_and_accel_limits():
    traj = parameterize_time_optimal(PATH, VMAX, AMAX, n_samples=300)
    assert traj.duration > 0
    assert np.all(np.abs(traj.qd) <= VMAX + 1e-2), "velocity limit violated"
    # Small tolerance for finite-difference curvature at polyline corners.
    assert np.all(np.abs(traj.qdd) <= AMAX * 1.05 + 1e-2), "accel limit violated"


def test_timing_starts_and_ends_at_rest_on_path():
    traj = parameterize_time_optimal(PATH, VMAX, AMAX, n_samples=300)
    assert np.allclose(traj.qd[0], 0.0, atol=1e-6)
    assert np.allclose(traj.qd[-1], 0.0, atol=1e-3)
    assert np.allclose(traj.q[0], PATH[0]) and np.allclose(traj.q[-1], PATH[-1])
    assert np.all(np.diff(traj.t) > 0), "time must be strictly increasing"


def test_tighter_limits_take_longer():
    fast = parameterize_time_optimal(PATH, VMAX, AMAX, n_samples=200)
    slow = parameterize_time_optimal(PATH, VMAX * 0.5, AMAX, n_samples=200)
    assert slow.duration > fast.duration
