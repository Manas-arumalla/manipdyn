"""
6-DOF Robotic Arm Trajectory Planning Simulator
"""

from robot import Robot, create_robot
from kinematics import Kinematics
from trajectory import TrajectoryGenerator, Trajectory, TrajectoryType
from control import PIDController, PIDGains
from simulation import Simulation, SimulationResult

__version__ = "1.0.0"

__all__ = [
    "Robot", "create_robot",
    "Kinematics",
    "TrajectoryGenerator", "Trajectory", "TrajectoryType",
    "PIDController", "PIDGains",
    "Simulation", "SimulationResult",
]
