"""Per-controller tuning specifications.

For each controller we declare:
  * ``space``    — the gains to optimize and their ``(low, high)`` bounds,
  * ``factory``  — how to build the controller from those gains,
  * ``target_space`` — whether it tracks a joint or Cartesian set-point,
  * ``eval``     — optional per-controller evaluation overrides (e.g. cheaper
                   settings / shorter rollouts for the expensive samplers).

This is the single source of truth shared by the auto-tuner, the
``optimize_controllers`` script, and (later) the benchmark harness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from manipdyn.control import (
    ComputedTorqueController,
    ILQRController,
    ImpedanceController,
    LQRController,
    MPPIController,
    OSCController,
    PIDController,
    TSIDController,
)


@dataclass
class TuneSpec:
    factory: Callable[..., object]
    space: dict[str, tuple[float, float]]
    target_space: str  # "joint" | "cartesian"
    method: str = "de"
    n_evals: int = 40
    duration: float = 4.0
    polish: bool = True
    eval: dict = field(default_factory=dict)


#: Tuning recipe for every controller in the zoo.
TUNE_SPECS: dict[str, TuneSpec] = {
    "pid": TuneSpec(
        factory=lambda w, kp, ki, kd: PIDController(w, kp=kp, ki=ki, kd=kd),
        space={"kp": (50.0, 800.0), "ki": (0.0, 20.0), "kd": (5.0, 150.0)},
        target_space="joint",
    ),
    "ctc": TuneSpec(
        factory=lambda w, kp, kd: ComputedTorqueController(w, kp=kp, kd=kd),
        space={"kp": (20.0, 600.0), "kd": (5.0, 120.0)},
        target_space="joint",
    ),
    "lqr": TuneSpec(
        factory=lambda w, q_pos, q_vel, r: LQRController(w, q_pos=q_pos, q_vel=q_vel, r=r),
        space={"q_pos": (100.0, 5000.0), "q_vel": (1.0, 100.0), "r": (0.05, 5.0)},
        target_space="joint",
        n_evals=30,
    ),
    "ilqr": TuneSpec(
        factory=lambda w, w_q, w_v, w_qf: ILQRController(
            w, horizon=80, control_dt=0.02, w_q=w_q, w_v=w_v, w_qf=w_qf
        ),
        space={"w_q": (1.0, 100.0), "w_v": (0.01, 5.0), "w_qf": (50.0, 1000.0)},
        target_space="joint",
        method="random",
        n_evals=15,
        duration=3.0,
    ),
    "impedance": TuneSpec(
        factory=lambda w, kp, kd: ImpedanceController(w, kp=kp, kd=kd),
        space={"kp": (100.0, 2500.0), "kd": (10.0, 400.0)},
        target_space="cartesian",
        duration=6.0,
    ),
    "osc": TuneSpec(
        factory=lambda w, kp, null_kp: OSCController(w, kp=kp, null_kp=null_kp),
        space={"kp": (50.0, 1000.0), "null_kp": (1.0, 50.0)},
        target_space="cartesian",
        duration=6.0,
    ),
    "tsid": TuneSpec(
        factory=lambda w, kp, w_posture: TSIDController(w, kp=kp, w_posture=w_posture),
        space={"kp": (50.0, 1000.0), "w_posture": (1e-3, 0.2)},
        target_space="cartesian",
        duration=6.0,
    ),
    "mppi": TuneSpec(
        # Sampling MPC is expensive: keep rollouts small and the budget low.
        factory=lambda w, noise_sigma, lambda_, w_pos: MPPIController(
            w,
            horizon=15,
            n_samples=20,
            noise_sigma=noise_sigma,
            lambda_=lambda_,
            w_pos=w_pos,
            seed=0,
        ),
        space={"noise_sigma": (0.5, 5.0), "lambda_": (0.01, 0.5), "w_pos": (1000.0, 10000.0)},
        target_space="joint",
        method="random",
        n_evals=10,
        duration=1.5,
        polish=False,  # rollouts are expensive; skip the extra local search
    ),
}
