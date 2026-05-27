"""Capture MuJoCo frames offscreen and write them to GIF/MP4.

This is what lets the project produce README demo assets and lets CI verify
rendering without a display, instead of relying on the interactive viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from manipdyn.sim.world import World


class Recorder:
    """Collects rendered frames from a :class:`World` for later export."""

    def __init__(
        self,
        world: World,
        width: int = 640,
        height: int = 480,
        camera: str | None = None,
        fps: int = 30,
        lookat: tuple[float, float, float] | None = None,
        distance: float | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
    ):
        self.world = world
        self.fps = fps
        self._renderer = mujoco.Renderer(world.model, height=height, width=width)
        self._camera = camera
        self.frames: list[np.ndarray] = []

        # Optional explicit free-camera framing (for clear, repeatable shots).
        self._free_cam = None
        if any(v is not None for v in (lookat, distance, azimuth, elevation)):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = lookat if lookat is not None else (0.0, 0.0, 0.3)
            cam.distance = distance if distance is not None else 2.0
            cam.azimuth = azimuth if azimuth is not None else 90.0
            cam.elevation = elevation if elevation is not None else -25.0
            self._free_cam = cam

    def capture(self) -> None:
        """Render the current world state and append it to the frame buffer."""
        if self._free_cam is not None:
            self._renderer.update_scene(self.world.data, camera=self._free_cam)
        elif self._camera is not None:
            self._renderer.update_scene(self.world.data, camera=self._camera)
        else:
            self._renderer.update_scene(self.world.data)
        self.frames.append(self._renderer.render())

    def save_gif(
        self,
        path: str | Path,
        fps: int | None = None,
        palettesize: int = 64,
        max_width: int | None = None,
    ) -> Path:
        return save_gif(self.frames, path, fps or self.fps, palettesize, max_width)

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def save_gif(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int = 30,
    palettesize: int = 64,
    max_width: int | None = None,
) -> Path:
    """Write HxWx3 uint8 frames to an animated GIF.

    ``palettesize`` reduces the color palette and ``max_width`` optionally
    downscales the frames — both shrink the file for README/web use.
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to save — did you call Recorder.capture()?")

    images = [Image.fromarray(f) for f in frames]
    if max_width and images[0].width > max_width:
        scale = max_width / images[0].width
        size = (max_width, int(images[0].height * scale))
        images = [im.resize(size) for im in images]
    # Quantize to a small palette + optimize: reliably small GIFs for the web.
    images = [im.quantize(colors=palettesize, method=Image.Quantize.MEDIANCUT) for im in images]

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path
