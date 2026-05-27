"""Finite-difference linearization of the MuJoCo dynamics for LQR.

Builds the continuous-time state-space matrices ``(A, B)`` for ``x = [q, v]``,
``u = ctrl`` about a configuration ``q*`` at rest. MuJoCo's
:func:`mujoco.mjd_transitionFD` returns the *discrete* transition Jacobians, so
we convert with ``A_c = (A_d - I) / dt`` and ``B_c = B_d / dt``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from manipdyn.sim.world import World


def linearize(
    world: World,
    q_full_target: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous-time ``(A, B)`` about a gravity-compensated equilibrium.

    Parameters
    ----------
    world:
        The world whose model is linearized. Its live state is preserved.
    q_full_target:
        Full ``qpos`` (length ``nq``) to linearize about; defaults to the
        current configuration.

    Returns
    -------
    (A, B):
        ``A`` is ``(2*nv, 2*nv)`` and ``B`` is ``(2*nv, nu)``.
    """
    m, d = world.model, world.data
    nv, nu = m.nv, m.nu

    qpos_save = d.qpos.copy()
    qvel_save = d.qvel.copy()
    ctrl_save = d.ctrl.copy()

    if q_full_target is not None:
        d.qpos[:] = q_full_target
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    # Linearize around the equilibrium where actuators hold against gravity.
    d.ctrl[:] = d.qfrc_bias[:nu]

    n = 2 * nv
    A_d = np.zeros((n, n))
    B_d = np.zeros((n, nu))
    mujoco.mjd_transitionFD(m, d, eps, True, A_d, B_d, None, None)

    dt = m.opt.timestep
    A_c = (A_d - np.eye(n)) / dt
    B_c = B_d / dt

    d.qpos[:] = qpos_save
    d.qvel[:] = qvel_save
    d.ctrl[:] = ctrl_save
    mujoco.mj_forward(m, d)

    return A_c, B_c
