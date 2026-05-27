"""Controllers: a uniform interface over the manipulator control zoo.

All controllers share the :class:`~manipdyn.control.base.Controller` interface
and return a *full* arm torque command, so the benchmark harness can swap them
freely. Use :data:`CONTROLLERS` to look one up by name.

================  ============  ===========================================
name              target_space  method
================  ============  ===========================================
``pid``           joint         PID + gravity compensation (baseline)
``ctc``           joint         computed torque / feedback linearization
``lqr``           joint         LQR via Riccati on linearized dynamics
``ilqr``          joint         iLQR optimal trajectory + feedback playback
``impedance``     cartesian     Jacobian-transpose spring-damper
``osc``           cartesian     operational-space control + null space
``tsid``          cartesian     task-space inverse dynamics (constrained QP)
``mppi``          joint         sampling-based model-predictive control
================  ============  ===========================================
"""

from manipdyn.control.base import Controller, Target
from manipdyn.control.ctc import ComputedTorqueController
from manipdyn.control.ilqr import ILQRController
from manipdyn.control.impedance import ImpedanceController
from manipdyn.control.lqr import LQRController
from manipdyn.control.mppi import MPPIController
from manipdyn.control.osc import OSCController
from manipdyn.control.pid import PIDController
from manipdyn.control.tsid import TSIDController

#: Registry mapping controller name -> class, for the benchmark/GUI.
CONTROLLERS: dict[str, type[Controller]] = {
    PIDController.name: PIDController,
    ComputedTorqueController.name: ComputedTorqueController,
    LQRController.name: LQRController,
    ILQRController.name: ILQRController,
    ImpedanceController.name: ImpedanceController,
    OSCController.name: OSCController,
    TSIDController.name: TSIDController,
    MPPIController.name: MPPIController,
}

__all__ = [
    "Controller",
    "Target",
    "PIDController",
    "ComputedTorqueController",
    "LQRController",
    "ILQRController",
    "ImpedanceController",
    "OSCController",
    "TSIDController",
    "MPPIController",
    "CONTROLLERS",
]
