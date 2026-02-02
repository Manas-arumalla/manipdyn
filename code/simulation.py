"""
Robot Arm Simulation

Closed-loop simulation with simplified dynamics and PID control.
Runs trajectory tracking and compares different planners.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

from robot import Robot
from kinematics import Kinematics
from trajectory import Trajectory, TrajectoryGenerator, TrajectoryType
from control import PIDController


# =============================================================================
# Types
# =============================================================================

@dataclass
class SimulationResult:
    """Container for simulation results."""
    time: np.ndarray
    q_desired: np.ndarray
    q_actual: np.ndarray
    dq_desired: np.ndarray
    dq_actual: np.ndarray
    tau: np.ndarray
    ee_desired: np.ndarray   # End-effector desired (x,y,z)
    ee_actual: np.ndarray    # End-effector actual
    tracking_error: np.ndarray
    trajectory_type: TrajectoryType


# =============================================================================
# Simulation
# =============================================================================

class Simulation:
    """
    Closed-loop trajectory tracking simulation.
    
    Uses simplified second-order dynamics:
    M*ddq + B*dq = tau
    """
    
    def __init__(self, robot: Robot, dt: float = 0.01):
        """
        Initialize simulation.
        
        Args:
            robot: Robot model
            dt: Time step (s)
        """
        self.robot = robot
        self.kinematics = Kinematics(robot)
        self.dt = dt
        
        # Simplified dynamics (diagonal inertia and damping)
        self.M = np.diag([5.0, 5.0, 3.0, 1.0, 1.0, 0.5])
        self.B = np.diag([2.0, 2.0, 1.5, 0.5, 0.5, 0.3])
        self.M_inv = np.linalg.inv(self.M)
        
        self.noise_scale = 0.001
    
    def simulate(self, trajectory: Trajectory) -> SimulationResult:
        """
        Run closed-loop simulation.
        
        Args:
            trajectory: Desired trajectory
            
        Returns:
            SimulationResult with tracking data
        """
        N = len(trajectory.time)
        n = self.robot.n_joints
        
        # Storage
        q_actual = np.zeros((N, n))
        dq_actual = np.zeros((N, n))
        tau = np.zeros((N, n))
        ee_desired = np.zeros((N, 3))
        ee_actual = np.zeros((N, 3))
        
        # Initial state
        q = trajectory.positions[0].copy()
        dq = np.zeros(n)
        
        # Controller
        controller = PIDController(n_joints=n)
        controller.reset()
        
        for i in range(N):
            q_actual[i] = q
            dq_actual[i] = dq
            
            # End-effector positions
            ee_desired[i] = self.kinematics.get_position(trajectory.positions[i])
            ee_actual[i] = self.kinematics.get_position(q)
            
            # Control
            tau[i] = controller.compute(
                trajectory.positions[i],
                trajectory.velocities[i],
                q, dq, self.dt
            )
            
            # Feedforward
            tau[i] += trajectory.accelerations[i] * np.diag(self.M) * 0.5
            
            # Dynamics: ddq = M^{-1}(tau - B*dq)
            ddq = self.M_inv @ (tau[i] - self.B @ dq)
            ddq += np.random.randn(n) * self.noise_scale
            
            # Integrate
            dq = dq + ddq * self.dt
            q = q + dq * self.dt
            q = self.robot.clamp_to_limits(q)
        
        tracking_error = np.linalg.norm(q_actual - trajectory.positions, axis=1)
        
        return SimulationResult(
            time=trajectory.time,
            q_desired=trajectory.positions,
            q_actual=q_actual,
            dq_desired=trajectory.velocities,
            dq_actual=dq_actual,
            tau=tau,
            ee_desired=ee_desired,
            ee_actual=ee_actual,
            tracking_error=tracking_error,
            trajectory_type=trajectory.trajectory_type
        )
    
    def run_comparison(
        self, 
        q_start: np.ndarray, 
        q_end: np.ndarray,
        T: float = 3.0,
        N: int = 300
    ) -> Dict[TrajectoryType, SimulationResult]:
        """
        Run simulation with all 3 trajectory planners.
        
        Returns:
            Dictionary of results per trajectory type
        """
        generator = TrajectoryGenerator(n_joints=self.robot.n_joints)
        results = {}
        
        for traj_type in TrajectoryType:
            print(f"Simulating {traj_type.value}...")
            trajectory = generator.generate(traj_type, q_start, q_end, T=T, N=N)
            results[traj_type] = self.simulate(trajectory)
            
            mean_err = np.mean(results[traj_type].tracking_error)
            print(f"  Mean error: {mean_err:.4f} rad")
        
        return results
    
    def run_waypoint_comparison(
        self, 
        waypoints: list,
        T_per_segment: float = 1.5,
        N_per_segment: int = 100
    ) -> Dict[TrajectoryType, SimulationResult]:
        """
        Run simulation with all 3 trajectory planners through multiple waypoints.
        
        Args:
            waypoints: List of joint configurations [q0, q1, q2, q3]
            T_per_segment: Time per segment
            N_per_segment: Samples per segment
            
        Returns:
            Dictionary of results per trajectory type
        """
        generator = TrajectoryGenerator(n_joints=self.robot.n_joints)
        results = {}
        
        for traj_type in TrajectoryType:
            print(f"Simulating {traj_type.value} ({len(waypoints)} waypoints)...")
            trajectory = generator.generate_multi_waypoint(
                traj_type, waypoints, 
                T_per_segment=T_per_segment, 
                N_per_segment=N_per_segment
            )
            results[traj_type] = self.simulate(trajectory)
            
            mean_err = np.mean(results[traj_type].tracking_error)
            print(f"  Mean error: {mean_err:.4f} rad")
        
        return results


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from .robot import create_robot
    
    robot = create_robot()
    sim = Simulation(robot)
    
    q_start = np.zeros(6)
    q_end = np.array([0.5, -0.5, 0.5, -0.3, 0.3, -0.2])
    
    results = sim.run_comparison(q_start, q_end)
    print("Done!")
