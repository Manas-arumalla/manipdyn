"""A simulated RGB-D camera on top of a named MuJoCo camera.

Renders colour, **metric depth**, and **segmentation** from a camera defined in
the scene, and exposes the pinhole intrinsics/extrinsics needed to turn a depth
image into a metric point cloud (see :mod:`manipdyn.perception.pointcloud`).

Depth convention (validated in ``tests/test_perception.py``): MuJoCo's depth
buffer is the linear perpendicular distance in **metres** along the camera's
optical axis, so the standard pinhole deprojection

    x = (u - cx) / fx * z,   y = -(v - cy) / fy * z,   z_cam = -z

followed by ``world = R_cam @ p_cam + t_cam`` recovers world points. The camera
looks along its local ``-z`` with ``+x`` right and ``+y`` up (OpenGL/MuJoCo
convention); the vertical field of view comes from ``model.cam_fovy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from manipdyn.sim.world import World


class Camera:
    """An RGB-D + segmentation view from a scene camera.

    Parameters
    ----------
    world:
        The :class:`~manipdyn.sim.world.World` whose model/data to render.
    name:
        Name of a ``<camera>`` in the scene (e.g. ``"overhead"`` or ``"wrist"``).
    width, height:
        Render resolution in pixels. Defaults stay within MuJoCo's default
        offscreen framebuffer (640x480); larger sizes need a ``<global
        offwidth=.../>`` in the scene's ``<visual>``.
    """

    def __init__(self, world: World, name: str = "overhead", width: int = 640, height: int = 480):
        self.world = world
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.cam_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.cam_id == -1:
            have = [
                mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(world.model.ncam)
            ]
            raise ValueError(f"No camera named {name!r} in scene (have: {have}).")
        self._renderer = mujoco.Renderer(world.model, height=self.height, width=self.width)

    # ---------------------------------------------------------------- intrinsics
    @property
    def fovy(self) -> float:
        """Vertical field of view in degrees (from the model)."""
        return float(self.world.model.cam_fovy[self.cam_id])

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        """Pinhole ``(fx, fy, cx, cy)`` in pixels (square pixels from ``fovy``)."""
        f = 0.5 * self.height / np.tan(0.5 * np.deg2rad(self.fovy))
        return f, f, (self.width - 1) / 2.0, (self.height - 1) / 2.0

    @property
    def extrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """Camera pose in world frame as ``(R, t)`` (3x3 rotation, 3-vector).

        Read from live ``data``; valid after any ``world.step()``/``forward()``.
        """
        R = self.world.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        t = self.world.data.cam_xpos[self.cam_id].copy()
        return R, t

    # ------------------------------------------------------------------- renders
    def rgb(self) -> np.ndarray:
        """Return an ``(H, W, 3)`` uint8 colour image."""
        self._renderer.disable_depth_rendering()
        self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(self.world.data, camera=self.cam_id)
        return self._renderer.render()

    def depth(self) -> np.ndarray:
        """Return an ``(H, W)`` float32 metric depth image (metres)."""
        self._renderer.disable_segmentation_rendering()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.world.data, camera=self.cam_id)
        out = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return out

    def segmentation(self) -> np.ndarray:
        """Return an ``(H, W, 2)`` int32 segmentation image.

        Channel 0 is the object id (e.g. geom id), channel 1 the object type
        (``mjtObj``); ``-1`` marks pixels with no geometry.
        """
        self._renderer.disable_depth_rendering()
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(self.world.data, camera=self.cam_id)
        out = self._renderer.render().copy()
        self._renderer.disable_segmentation_rendering()
        return out

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
