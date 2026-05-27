"""Bundled MuJoCo models and a resolver that works regardless of CWD.

Scenes are loaded by *absolute* path so MuJoCo resolves ``<include>`` and
``meshdir`` relative to the scene file — fixing the CWD-brittleness of the
original prototype, where everything had to run from one directory.
"""

from __future__ import annotations

from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent

# Scenes shipped with the package (without the .xml suffix).
AVAILABLE_SCENES = ("scene", "scene_base", "scene_base_gripper")


def models_dir() -> Path:
    """Absolute path to the bundled models directory."""
    return _MODELS_DIR


def scene_path(name: str) -> str:
    """Resolve a scene name (or filename) to an absolute path.

    >>> scene_path("scene_base")        # -> ".../models/scene_base.xml"
    >>> scene_path("scene_base.xml")    # also works
    """
    filename = name if name.endswith(".xml") else f"{name}.xml"
    path = _MODELS_DIR / filename
    if not path.exists():
        available = ", ".join(AVAILABLE_SCENES)
        raise FileNotFoundError(
            f"Scene '{name}' not found at {path}. Available scenes: {available}"
        )
    return str(path)


__all__ = ["models_dir", "scene_path", "AVAILABLE_SCENES"]
