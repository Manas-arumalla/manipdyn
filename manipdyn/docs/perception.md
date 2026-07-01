# Perception & vision

The lab can drive the arm from what a **simulated RGB-D camera sees** instead of
from privileged simulator state. A fixed overhead camera and an eye-in-hand
wrist camera render colour, metric depth, and segmentation; the depth is
deprojected into a world-frame point cloud, and an object's top-down grasp pose
is estimated from that cloud — no reading of `data.xpos`.

![perception pipeline](../benchmarks/results/perception.png)

```python
from manipdyn.sim import World
from manipdyn.perception import Camera, sense_object_pose
from manipdyn.tasks import pick_place

world = World(scene="scene_pick", ee_site="pinch")
cam = Camera(world, "overhead")
est = sense_object_pose(cam)                    # ObjectEstimate from vision
plan = pick_place.solve(world, object_xy=est.top_xy)   # grasp what we see
```

## Pipeline

1. **Render** — `Camera` wraps a named `<camera>` and returns an `(H,W,3)` RGB
   image, an `(H,W)` metric **depth** image, and an `(H,W,2)` **segmentation**
   image, plus pinhole intrinsics `(fx, fy, cx, cy)` from `model.cam_fovy` and
   the camera-to-world extrinsics `(R, t)` from `data.cam_xpos/xmat`.
2. **Deproject** — each depth pixel becomes a metric camera-frame point and is
   transformed to the world frame:

   $$x=\frac{u-c_x}{f_x}\,z,\quad y=-\frac{v-c_y}{f_y}\,z,\quad p_\text{world}=R\,[x,\,y,\,-z]^\top+t.$$

   MuJoCo's depth is the linear perpendicular distance in metres along the
   optical axis (the camera looks down its local $-z$), so this standard pinhole
   model recovers world points. The convention is pinned by
   `tests/test_perception.py`.
3. **Clean** — voxel-downsample, drop the dominant plane (table/floor) with
   RANSAC, and keep the largest Euclidean cluster.
4. **Estimate** — `estimate_object_pose` returns the axis-aligned
   bounding-box centre, the top-surface height, footprint extents, and PCA
   axes. The **bounding-box centre** is a robust XY grasp target under a partial
   oblique view — its extremes are set by the object's edges, so it is not
   pulled toward the camera the way a raw centroid is by foreshortening.

## Two honest segmentation modes

`sense_object_pose(cam, segmentation=...)`:

* **`segmentation=True`** (default) uses MuJoCo's ground-truth segmentation
  buffer to pick the object's pixels — a stand-in for a perfect instance
  segmenter. It reads *which pixels are the object*, **never the object's
  pose**, so the grasp target still comes from geometry.
* **`segmentation=False`** is fully sensor-only: deproject everything, drop the
  table plane, and keep the largest cluster inside an optional workspace box
  (scene knowledge, not object pose). Cleaner when the arm is parked clear of
  the object; the ground-truth-segmentation mode is more robust to self-occlusion.

## Does vision cost anything?

`scripts/benchmark_perception.py` runs the full pick-and-place from the
ground-truth pose (oracle) and from the estimated pose (perception) over
randomized cube placements, on the same physics and the same success test —
the lab's fair-comparison thesis applied to perception:

| driver | grasp success | mean place err |
|--------|--------------:|---------------:|
| oracle (ground-truth pose) | 12/12 | 0.2 mm |
| **perception (RGB-D)** | 12/12 | 0.6 mm |

Perception pose error: mean **0.2 mm**, max **0.7 mm** with the arm parked clear
for the look. Numbers regenerate with `python scripts/benchmark_perception.py`.

## What this does and doesn't do

* It replaces the **oracle object pose** with a *sensed* one — the grasp is
  driven by vision.
* It does **not** change the grasp physics: the cube is still held by a weld set
  at grasp time (see [tasks](tasks.md)). Contact-based grasping is a separate
  follow-on.
* The two scene cameras are **non-physical** — no mass, DOF, or geometry — so
  adding them cannot alter any existing dynamics, controller, or benchmark.
