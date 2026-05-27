# Optimization & optimal control

Beyond reactive feedback, `manipdyn` includes optimization-based methods that
*plan* control, *respect constraints*, and *tune themselves*.

## iLQR — trajectory optimization

Iterative LQR computes a locally optimal **open-loop torque trajectory** plus
time-varying feedback gains for

$$ \min_{u_{0:N-1}} \; \ell_f(x_N) + \sum_{t} w_q\lVert q_t - q^\*\rVert^2 + w_v\lVert v_t\rVert^2 + w_u\lVert u_t\rVert^2. $$

Each iteration:

1. **Rollout** the current controls through the MuJoCo dynamics to get the
   nominal $x_{0:N}$.
2. **Backward pass** — propagate a quadratic value-function approximation from
   the terminal cost, using the dynamics Jacobians $f_x, f_u$ (from
   `mjd_transitionFD`), to get control updates $k_t$ and gains $K_t$:
   $$ Q_{uu} = \ell_{uu} + f_u^\top V_{xx} f_u + \mu I, \quad k = -Q_{uu}^{-1}Q_u, \quad K = -Q_{uu}^{-1}Q_{ux}. $$
3. **Forward pass** with a backtracking line search on the step $\alpha$,
   accepting only if the total cost decreases.

Levenberg–Marquardt regularization $\mu$ on $Q_{uu}$ keeps the backward pass
well-posed. This is Gauss–Newton iLQR; full DDP adds the dynamics' second-order
tensors. For affordability, optimization runs on a private model copy at a
coarse control timestep; the `ilqr` controller then plays the trajectory back
with feedback and holds the goal with a stabilizing PD once the horizon ends.

## TSID — task-space inverse dynamics as a QP

OSC decouples the task but cannot enforce hard limits. **TSID** solves, every
tick, a small quadratic program for joint accelerations $a$:

$$ \min_{a}\; w_\text{task}\lVert J a - a^\*_\text{task}\rVert^2 + w_\text{post}\lVert a - a_\text{posture}\rVert^2 + w_\text{reg}\lVert a\rVert^2 $$
$$ \text{s.t.}\quad -\tau_\max \le M(q)\,a + h(q,v) \le \tau_\max, $$

with $a^\*_\text{task} = K_p(x^\*-x) + K_d(\dot x^\*-\dot x)$, and applies
$\tau = M a + h$. Because the torque map is linear in $a$, actuator limits are
linear inequality constraints — solved with OSQP. This is the template behind
modern whole-body controllers, and it degrades gracefully (a damped
least-squares fallback) if the QP ever fails.

## Time-optimal path parameterization

A geometric path says *where*, not *when*. Parameterizing by arc length $s$
with tangent $q'(s)$ and curvature $q''(s)$, and path speed $f = \dot s$,

$$ \dot q = q' f, \qquad \ddot q = q' a + q'' f^2, \quad a = \ddot s. $$

With $x = f^2$ (so $dx/ds = 2a$): velocity limits cap $x \le (v_j/|q'_j|)^2$,
and at high-curvature points the centripetal term alone bounds
$x \le a_j/|q''_j|$. A **forward pass** accelerates as hard as the acceleration
limits allow, a **backward pass** guarantees braking feasibility, and the
resulting speed profile is integrated to time stamps — the classic
numerical-integration solution to TOPP. The output is a `TimedTrajectory`
$(t, q, \dot q, \ddot q)$ that honors per-joint velocity and acceleration
limits.

## Controller auto-tuning

Gains are themselves optimization variables. `tuning.tune_controller` minimizes

$$ \text{cost} = w_e\,e_\text{final} + w_s\,t_\text{settle} + w_u\,\overline{\lVert\tau\rVert} $$

measured from closed-loop rollouts. A **global search** — random search,
differential evolution (`method="de"`), or dual annealing (`method="anneal"`) —
is followed by a bounded **Nelder–Mead polish** (the cost is a plain callable,
so CMA-ES / Bayesian optimization slot in later). This serves two ends: it
finds good gains automatically, and — crucially for the
[benchmark](../README.md) — it gives every controller the **same tuning budget
on the same scenario**, so results reflect the *method*, not hand-tuning effort.

### Tuned presets

`scripts/optimize_controllers.py` tunes *every* controller and saves the winning
gains to `manipdyn/tuning/tuned_gains.json`; `tuning.tuned_controller(name,
world)` builds a controller with those gains (falling back to defaults if a
controller wasn't improved — a regression guard prevents adopting worse gains).

Representative results (lower cost is better):

| controller | baseline | tuned | improvement |
|------------|---------:|------:|------------:|
| pid        | 0.678 | 0.237 | **−65%** |
| ctc        | 0.539 | 0.193 | **−64%** |
| lqr        | 0.699 | 0.251 | **−64%** |
| tsid       | 0.410 | 0.193 | **−53%** |
| osc        | 0.382 | 0.203 | **−47%** |
| ilqr       | 0.365 | 0.214 | **−41%** |
| impedance  | 0.316 | 0.284 | **−10%** |
| mppi       | 0.746 |   —   | kept defaults\* |

\*Sampling MPC is expensive to tune on a small budget, so its (reasonable)
defaults are retained. The benchmark uses these tuned gains for fair comparison.
