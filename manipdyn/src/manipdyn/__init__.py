"""manipdyn — a MuJoCo-physics lab for 6-DOF manipulator planning and control.

A clean, importable, benchmark-driven package built around the UR5e in MuJoCo.

Example:
    from manipdyn.sim import World
    from manipdyn.control import PIDController
"""

from __future__ import annotations

__version__ = "0.1.0"

from manipdyn.sim import World

__all__ = ["World", "__version__"]
