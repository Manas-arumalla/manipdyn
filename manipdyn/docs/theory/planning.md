# Motion planning

Planners search the 6-D joint space for a **collision-free** path from a start
configuration to a goal. All planners share the :class:`Planner` interface and
return an ``(N, 6)`` array (or ``None``), validated against the MuJoCo scene by
a :class:`CollisionChecker`.

## Collision checking

A configuration is *in collision* if, after setting the arm DOFs and running
`mj_kinematics` + `mj_collision`, any active contact penetrates beyond a small
margin. Edge checking interpolates between two configurations at a fixed
joint-space resolution and tests each sample — so an edge is only accepted if
the whole swept segment is clear. The checker runs on a private `MjData`, and
joint sampling bounds are read from the model (`jnt_range`), not hardcoded.

## Sampling-based planners

### RRT
Grow one tree from the start, each iteration sampling a random configuration
(with probability `goal_bias`, the goal itself), steering the nearest node a
step toward it, and keeping the new node if the edge is collision-free.
*Probabilistically complete*, fast to a first solution, but the path is jagged
and non-optimal.

### RRT-Connect
Grow **two** trees — from start and goal — and alternately try to *connect*
them with a greedy multi-step extension. Bidirectional search plus greedy
connection makes it dramatically faster than plain RRT; it is the de-facto
default for manipulators. (No optimality guarantee.)

### RRT\*
Like RRT, but for each new node it (a) chooses the lowest-cost collision-free
parent within a radius and (b) *rewires* nearby nodes through the new node when
that lowers their cost. The path cost converges to the optimum as samples grow
(*asymptotically optimal*), at higher per-iteration cost.

### Informed RRT\*
Once a solution of cost $c_\text{best}$ exists, only samples that *could*
improve it can help — those inside the prolate hyperspheroid with foci at start
and goal and transverse diameter $c_\text{best}$. Sampling is restricted to
this ellipsoid (a rotation + scaling of a unit ball), which sharply accelerates
convergence to the optimal cost.

### PRM
A **multi-query** roadmap: sample many free configurations once, connect each
to its $k$ nearest neighbors with collision-free edges, then answer each
start/goal query by connecting both to the roadmap and running Dijkstra. Pays a
one-time build cost; subsequent queries in a static scene are cheap.

## Path smoothing

* **Shortcutting** (the workhorse): repeatedly pick two points on the path and,
  if the straight segment between them is collision-free, splice out everything
  in between. Collision-aware and very effective at removing detours.
* **B-spline**: fit a spline through the (shortcut) waypoints and resample for a
  visually smooth, higher-resolution path.

A smoothed *geometric* path still has no timing — see
[time-optimal parameterization](optimization.md#time-optimal-path-parameterization).

### Choosing a planner

| Situation | Reach for |
|-----------|-----------|
| Need a path fast, single query | `rrt_connect` |
| Want a near-optimal path | `informed_rrt_star` |
| Many queries, static scene | `prm` |
| Teaching / baseline | `rrt`, `rrt_star` |
