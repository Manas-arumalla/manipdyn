"""Load optimized controller gains and build controllers from them.

``optimize_controllers.py`` writes the tuned gains to ``tuned_gains.json``
(shipped inside the package). :func:`tuned_controller` builds a controller with
those gains, falling back to library defaults if none have been saved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from manipdyn.control import CONTROLLERS
from manipdyn.tuning.specs import TUNE_SPECS

if TYPE_CHECKING:
    from manipdyn.control.base import Controller
    from manipdyn.sim.world import World

_PRESETS_FILE = Path(__file__).resolve().parent / "tuned_gains.json"


def load_tuned_gains() -> dict:
    """Return the saved tuned-gains table (empty dict if not yet generated)."""
    if _PRESETS_FILE.exists():
        return json.loads(_PRESETS_FILE.read_text())
    return {}


def tuned_params(name: str) -> dict | None:
    """Optimized gains for a controller, or ``None`` if not tuned."""
    entry = load_tuned_gains().get(name)
    return entry.get("params") if entry else None


def tuned_controller(name: str, world: World) -> Controller:
    """Instantiate a controller with its optimized gains (or defaults)."""
    params = tuned_params(name)
    if params and name in TUNE_SPECS:
        return TUNE_SPECS[name].factory(world, **params)
    return CONTROLLERS[name](world)
