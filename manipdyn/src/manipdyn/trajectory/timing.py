r"""Time-optimal path parameterization (TOPP, forward-backward integration).

A geometric planner returns *where* to go, not *when*. This module assigns
times to a joint-space path so the motion is as fast as possible while
respecting per-joint velocity and acceleration limits.

Parameterize the path by arc length s with tangent q'(s) and curvature q''(s).
With path speed f = ds/dt and tangential acceleration a = d(f)/dt,
$$ \dot q = q'(s)\,f, \qquad \ddot q = q'(s)\,a + q''(s)\,f^2. $$
Using x = f^2 (so dx/ds = 2a), velocity limits cap x directly and acceleration
limits bound a *given* x. A forward pass (accelerate as hard as possible) then a
backward pass (ensure we can still brake) yields the maximum feasible speed
profile — the classic numerical-integration solution to TOPP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TimedTrajectory:
    t: np.ndarray  # (M,) time stamps
    q: np.ndarray  # (M, n) positions
    qd: np.ndarray  # (M, n) velocities
    qdd: np.ndarray  # (M, n) accelerations

    @property
    def duration(self) -> float:
        return float(self.t[-1])


def _accel_bounds(
    dq: np.ndarray, ddq: np.ndarray, x: float, a_lim: np.ndarray
) -> tuple[float, float]:
    """Feasible tangential-acceleration interval at one path point.

    Per joint: -a_lim <= dq*a + ddq*x <= a_lim. Each joint gives an interval for
    ``a``; the intersection is returned as (a_lo, a_hi).
    """
    a_lo, a_hi = -np.inf, np.inf
    for dqi, ddqi, ai in zip(dq, ddq, a_lim, strict=True):
        if abs(dqi) < 1e-9:
            continue  # this joint does not constrain `a` here
        b1 = (ai - ddqi * x) / dqi
        b2 = (-ai - ddqi * x) / dqi
        lo, hi = min(b1, b2), max(b1, b2)
        a_lo, a_hi = max(a_lo, lo), min(a_hi, hi)
    return a_lo, a_hi


def parameterize_time_optimal(
    path: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    n_samples: int = 200,
) -> TimedTrajectory:
    """Assign a time-optimal, limit-respecting timing to a joint-space path."""
    path = np.asarray(path, dtype=float)
    vel_limits = np.abs(np.asarray(vel_limits, dtype=float))
    acc_limits = np.abs(np.asarray(acc_limits, dtype=float))
    n = path.shape[1]

    # 1. Cumulative arc length, then resample to uniform spacing in s.
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    length = s_cum[-1]
    if length < 1e-9:
        z = np.zeros((1, n))
        return TimedTrajectory(np.zeros(1), path[:1].copy(), z, z)

    s = np.linspace(0.0, length, n_samples)
    q = np.column_stack([np.interp(s, s_cum, path[:, j]) for j in range(n)])
    ds = length / (n_samples - 1)

    # 2. Path tangent q'(s) and curvature q''(s).
    dq = np.gradient(q, ds, axis=0)
    ddq = np.gradient(dq, ds, axis=0)

    # 3. Cap x = f^2 from two sources:
    #    - velocity limits:  |q'_j| f <= v_j           ->  x <= (v_j/|q'_j|)^2
    #    - curvature/accel:  |q''_j| f^2 <= a_j (a=0)   ->  x <= a_j/|q''_j|
    #    The second prevents centripetal acceleration alone from violating
    #    limits at high-curvature points (e.g. corners of a polyline path).
    with np.errstate(divide="ignore", invalid="ignore"):
        x_vel = np.min(vel_limits / (np.abs(dq) + 1e-12), axis=1) ** 2
        x_acc = np.min(acc_limits / (np.abs(ddq) + 1e-12), axis=1)
    x = np.minimum(x_vel, x_acc)
    x[0] = x[-1] = 0.0  # start and end at rest

    # 4a. Forward pass: accelerate as hard as the limits allow.
    for i in range(n_samples - 1):
        _, a_hi = _accel_bounds(dq[i], ddq[i], x[i], acc_limits)
        if np.isfinite(a_hi):
            x[i + 1] = min(x[i + 1], max(x[i] + 2.0 * a_hi * ds, 0.0))

    # 4b. Backward pass: guarantee we can brake in time.
    for i in range(n_samples - 1, 0, -1):
        a_lo, _ = _accel_bounds(dq[i], ddq[i], x[i], acc_limits)
        if np.isfinite(a_lo):
            x[i - 1] = min(x[i - 1], max(x[i] - 2.0 * a_lo * ds, 0.0))

    x = np.clip(x, 0.0, None)
    f = np.sqrt(x)

    # 5. Integrate time: dt_i = 2 ds / (f_i + f_{i+1}).
    t = np.zeros(n_samples)
    for i in range(n_samples - 1):
        avg = max(0.5 * (f[i] + f[i + 1]), 1e-6)
        t[i + 1] = t[i] + ds / avg

    # 6. Joint velocities/accelerations from the profile.
    a_path = np.gradient(x, ds, axis=0) / 2.0  # tangential accel a = dx/ds / 2
    qd = dq * f[:, None]
    qdd = dq * a_path[:, None] + ddq * x[:, None]
    return TimedTrajectory(t, q, qd, qdd)
