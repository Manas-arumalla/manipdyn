# Inverse kinematics

The UR5e is a 6-DOF arm; we solve numerically with **damped least squares**
(Levenberg–Marquardt), which stays well-behaved through singularities where the
plain Jacobian pseudo-inverse blows up.

## Problem

Find joint angles $q$ placing the end-effector site at a target position
$x^\*$ (optionally also orientation $R^\*$). Define the task error
$$ e = \begin{bmatrix} x^\* - x(q) \\ \log(R^\* R(q)^\top) \end{bmatrix} $$
(the orientation part is the axis-angle of the rotation error; position-only
IK drops the bottom block).

## Damped least-squares update

The site Jacobian $J$ (`mj_jacSite`) relates joint to task velocity,
$\dot e = J \dot q$. Each iteration takes the step
$$ \Delta q = J^\top (J J^\top + \lambda^2 I)^{-1} e, \qquad q \leftarrow q + \alpha\,\Delta q, $$
clamped to the model's joint limits. The damping $\lambda^2$ trades convergence
speed for stability near singular configurations (it bounds $\lVert\Delta q\rVert$).

## API

```python
from manipdyn.kinematics import IKSolver
solver = IKSolver(world, damping=0.1, tol=1e-3, max_iter=100)
result = solver.solve(target_pos, q_guess=world.home_qpos_arm)
# result: IKResult(q, success, error, iterations)
```

The solver runs on a private `MjData`, so it never disturbs the live
simulation, and returns a structured `IKResult` (success flag + final error +
iteration count) instead of a bare array-or-`None`.
