# The control zoo

Every controller maps the live arm state $(q, \dot q)$ and a set-point to a
**full joint-torque command** $\tau \in \mathbb{R}^6$ that is written to the
UR5e's torque actuators. Gravity/bias compensation is each controller's own
responsibility, so all methods are compared on equal footing.

The arm's rigid-body dynamics are
$$ M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) = \tau, $$
where $M$ is the inertia matrix, $C\dot q$ the Coriolis/centrifugal force, and
$g$ gravity. MuJoCo gives us $M(q)$ (`mj_fullM`) and the *bias* force
$h(q,\dot q) = C\dot q + g$ (`qfrc_bias`) directly.

---

## PID (`pid`) — joint space

$$ \tau = K_p\,(q^\* - q) + K_i \!\int (q^\* - q)\,dt + K_d\,(\dot q^\* - \dot q) + h(q,\dot q) $$

Integral term is clamped (anti-windup). The baseline: model-free feedback plus
gravity compensation. Robust and simple, but a single fixed gain set cannot
account for the configuration-dependent inertia, so tracking degrades during
fast motion.

## Computed-Torque Control (`ctc`) — joint space

Feedback linearization. Choose a desired acceleration
$a = \ddot q^\* + K_p e + K_d \dot e$ (with $e = q^\* - q$) and invert the
dynamics for the torque that realizes it:
$$ \tau = M(q)\,a + C(q,\dot q)\dot q + g(q). $$
We compute the right-hand side exactly with MuJoCo inverse dynamics
(`mj_inverse`). With an accurate model the closed loop becomes six decoupled
linear systems $\ddot e + K_d \dot e + K_p e = 0$ — excellent tracking. The
cost is a full dynamics evaluation each tick and sensitivity to model error.

## Linear-Quadratic Regulator (`lqr`) — joint space

Linearize the dynamics about the goal $q^\*$ (at rest, gravity-compensated) to
get $\dot x = A x + B u$ with $x = [q, \dot q]$, then minimize
$$ J = \int_0^\infty x^\top Q x + u^\top R u \; dt. $$
The optimal feedback is $u = -K x$, $K = R^{-1} B^\top P$, where $P$ solves the
continuous-time **algebraic Riccati equation**
$$ A^\top P + P A - P B R^{-1} B^\top P + Q = 0. $$
$A, B$ come from MuJoCo finite differences (`mjd_transitionFD`, discrete →
continuous). Optimal *near* the linearization point; we re-linearize lazily if
the target moves. `q_pos`, `q_vel`, `r` are the $Q$/$R$ weights.

## Cartesian impedance (`impedance`) — task space

Make the end-effector behave like a spring-damper toward a Cartesian target,
mapping the task wrench to torque with the **Jacobian transpose**:
$$ F = K_p (x^\* - x) + K_d (\dot x^\* - \dot x), \qquad \tau = J_p^\top F + h. $$
No matrix inverse ⇒ graceful near singularities and naturally compliant
(contact-safe). Trade-off: configuration-dependent apparent inertia and
steady-state error under sustained load.

## Operational-Space Control (`osc`) — task space

Khatib's OSC accounts for the arm's inertia in task space. With the task-space
inertia $\Lambda = (J M^{-1} J^\top)^{-1}$,
$$ F = \Lambda\,(K_p e + K_d \dot e), \qquad \tau = J^\top F + N\,\tau_\text{posture} + h, $$
and the dynamically-consistent null-space projector
$N = I - J^\top \bar J^\top,\; \bar J = M^{-1} J^\top \Lambda$ lets a secondary
posture task (return toward home) run *without* disturbing the end-effector.
Gives consistent task-space stiffness; needs $M$, $J$, and an inverse of the
$3\times3$ task inertia (lightly damped for robustness).

## Model-Predictive Path Integral (`mppi`) — joint space

Sampling-based MPC. Sample $K$ noisy torque sequences over a horizon $H$, roll
each through the **true nonlinear MuJoCo dynamics**, score by
$$ S_k = \sum_t w_p\lVert q^\*-q\rVert^2 + w_v\lVert\dot q\rVert^2 + w_u\lVert u\rVert^2, $$
and update the nominal sequence with the softmax-weighted perturbations
$U \mathrel{+}= \sum_k \frac{e^{-S_k/\lambda}}{\sum_j e^{-S_j/\lambda}}\,\varepsilon_k$.
Gradient-free, handles contacts and nonlinearity, but expensive
($\approx K\times H$ simulation steps per control tick) — the prime candidate
for GPU acceleration (MuJoCo MJX).

---

### Choosing a controller

| Need | Reach for |
|------|-----------|
| Simple, robust regulation | `pid` |
| Accurate trajectory tracking, good model | `ctc` |
| Optimal regulation to a pose | `lqr` |
| Compliant / contact tasks | `impedance` |
| Task-space control with redundancy | `osc` |
| Constraints / nonlinearity / no gradients | `mppi` |

The [benchmark suite](benchmark.md) replaces this table with **measured** numbers.
