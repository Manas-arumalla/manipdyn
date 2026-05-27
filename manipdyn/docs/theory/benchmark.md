# The benchmark suite

Runs every controller and planner on common, reproducible scenarios and reports
measured metrics, so the methods can be compared quantitatively rather than by
eye.

```bash
manipdyn bench --which all          # controllers + planners -> benchmarks/results/
manipdyn bench --which controllers  # just the control zoo
```

Outputs land in `benchmarks/results/`: `BENCHMARK.md` (tables), `controllers.png`
/ `planners.png` (comparison plots), and `results.json` (raw numbers).

## Fairness

Every controller is instantiated with its **tuned gains**
([`tuned_controller`](optimization.md#tuned-presets)) and asked to reach the
**same** set of goals; all are scored in a common space — end-effector position
error toward the goal — regardless of whether they internally track a joint or
Cartesian set-point.

## Controller results (reach scenarios, `scene_base`)

| controller | final err (mm) | settle (s) | effort ‖τ‖² | peak τ (Nm) | compute (ms) |
|------------|---------------:|-----------:|------------:|------------:|-------------:|
| pid        | 0.23   | 0.26 | 6.9e3 | 875  | 0.008 |
| ctc        | 8e-13  | 0.23 | 6.0e3 | 523  | 0.014 |
| lqr        | 2e-8   | 0.34 | 2.2e3 | 352  | 0.010 |
| ilqr       | 0.01   | 0.29 | 2.2e3 | 220  | 0.12  |
| impedance  | 2.93   | 0.55 | 4.2e3 | 486  | 0.013 |
| osc        | 0.008  | 0.18 | 6.9e3 | 797  | 0.048 |
| tsid       | 0.025  | 0.20 | 2.3e3 | 233  | 1.02  |
| mppi       | 13.2   | 2.12 | 1.8e3 | 76   | 18.2  |

What the numbers say:

* **Computed-torque and LQR** are the accuracy champions (machine-precision /
  nm-scale residual) — exact feedback linearization and optimal linear feedback
  on a well-modeled arm.
* **OSC and TSID** settle fastest in task space; **TSID** does so with low
  torque and *guaranteed* limit satisfaction, but pays ~1.3 ms/step for the QP.
* **PID** is the cheapest to compute (~8 µs) and perfectly serviceable.
* **Impedance** leaves a few-mm steady-state error — expected for a
  Jacobian-transpose spring without inertia shaping.
* **MPPI** is the most expensive (~18 ms/step) and least precise here — it even
  misses the 2 cm tolerance on the hardest goal — but uses the *least peak
  torque*: the gradient-free sampler trades accuracy and compute for generality
  and needs no model derivatives.

All eight reaching the **same** target, side by side (`python scripts/make_gallery.py`):

![every controller reaching](../../media/controllers.gif)

## Planner results (obstacle scene)

The query is **genuinely blocked**: a straight-line joint move from start to
goal collides with the pillar through the middle of the motion, so every planner
has to find a detour up and over it.

| planner | success | plan time (s) | path len (rad) | nodes |
|---------|--------:|--------------:|---------------:|------:|
| rrt | 1.0 | 0.016 | 1.61 | 12 |
| rrt_connect | 1.0 | 0.020 | 1.69 | 15 |
| rrt_star | 1.0 | 16.4 | 1.57 | 6 |
| informed_rrt_star | 1.0 | 21.4 | **1.42** | 5 |
| prm | 1.0 | 6.35 | 5.32 | 4 |

All five solve it. **RRT and RRT-Connect** return a feasible path in ~20 ms.
**RRT\*** and **Informed RRT\*** spend seconds rewiring to shrink cost, and
Informed returns the **shortest** path (1.42 rad). **PRM** answers from a
precomputed roadmap, but its uniform samples route a longer, higher-arcing
detour that shortcutting can't straighten across the pillar — and denser
sampling barely helps, since the corridor over the pillar is a narrow passage.
PRM's payoff is reusing one roadmap across *many* queries, not single-query path
quality.

Each planner's path executed around the pillar, side by side:

![every planner avoiding the obstacle](../../media/planners.gif)

> Planner numbers are seed-controlled, so `manipdyn bench` reproduces them; plan
> times are wall-clock and depend on the machine.
