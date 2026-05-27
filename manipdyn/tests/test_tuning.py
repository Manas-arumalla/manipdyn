"""Auto-tuning must not do worse than a poorly-chosen baseline."""

from __future__ import annotations

import numpy as np
import pytest

from manipdyn.control import CONTROLLERS, PIDController, Target
from manipdyn.control.base import Controller
from manipdyn.sim import World
from manipdyn.tuning import TUNE_SPECS, tune_controller, tuned_controller
from manipdyn.tuning.autotune import evaluate_controller

Q_TARGET = np.array([1.0, -1.1, 1.2, -1.6, -1.4, 0.4])


def test_tuning_improves_on_weak_gains():
    target = Target(q=Q_TARGET)

    world = World(scene="scene_base")
    world.reset(world.home_qpos_arm)
    weak = evaluate_controller(world, PIDController(world, kp=60, ki=1, kd=8), target)["cost"]

    result = tune_controller(
        lambda: World(scene="scene_base"),
        lambda world, kp, kd: PIDController(world, kp=kp, ki=5.0, kd=kd),
        {"kp": (50.0, 600.0), "kd": (5.0, 120.0)},
        target,
        n_evals=20,
        seed=0,
    )

    assert result.n_evals >= 20  # 20 global evals + Nelder-Mead polish
    assert result.best_cost <= weak, "tuning should not be worse than the weak baseline"
    assert 50.0 <= result.best_params["kp"] <= 600.0


def test_every_controller_has_a_tuning_spec():
    # The benchmark relies on a spec for each registered controller.
    assert set(TUNE_SPECS) == set(CONTROLLERS)


@pytest.mark.parametrize("name", list(CONTROLLERS))
def test_tuned_controller_builds_every_controller(name):
    world = World(scene="scene_base")
    controller = tuned_controller(name, world)
    assert isinstance(controller, Controller)
    assert controller.name == name
