"""PySide6 control center for manipdyn.

Library-backed (imports manipdyn directly — no subprocess/JSON hop), with an
embedded live MuJoCo view, per-controller gains, planner integration, live
telemetry, and one-click benchmarking.

Launch with ``manipdyn gui`` (requires the ``gui`` extra: ``pip install -e
'.[gui]'``).
"""

from manipdyn.gui.app import launch

__all__ = ["launch"]
