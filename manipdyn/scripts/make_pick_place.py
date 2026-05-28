"""Pick-and-place demo: pick a cube off one table, carry it, place it on another.

A thin driver over the reusable pipeline in :mod:`manipdyn.tasks.pick_place` —
it steps the task generator, renders an offscreen frame every few steps, and
writes the GIF. The same generator drives the live GUI.

Run from the manipdyn/ directory:
    python scripts/make_pick_place.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipdyn.render import Recorder
from manipdyn.sim import World
from manipdyn.tasks import pick_place

MEDIA = Path(__file__).resolve().parents[1] / "media"
CAPTURE_EVERY = 20


def main() -> None:
    world = World(scene=pick_place.SCENE, ee_site=pick_place.EE_SITE)
    rec = Recorder(
        world,
        width=560,
        height=440,
        fps=20,
        lookat=(-0.18, -0.31, 0.42),
        distance=1.45,
        azimuth=215,
        elevation=-18,
    )

    last = None
    for i, info in enumerate(pick_place.run(world)):
        last = info
        if i % CAPTURE_EVERY == 0:
            rec.capture()

    gif = rec.save_gif(MEDIA / "pick_place.gif", palettesize=40, max_width=400)
    rec.close()
    print(
        f"cube final position: {np.round(last['cube_pos'], 3)}  "
        f"tilt {last['cube_tilt_deg']:.1f} deg  place error {last['place_err_mm']:.1f} mm"
    )
    print(f"GIF saved: {gif}")


if __name__ == "__main__":
    main()
