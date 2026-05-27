"""
Trajectory Generation

Implements 3 trajectory planners:
1. Cubic Polynomial - smooth position and velocity
2. Quintic Polynomial - smooth position, velocity, acceleration
3. Trapezoidal Velocity - constant acceleration phases
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Types
# =============================================================================

class TrajectoryType(Enum):
    """Available trajectory planner types."""
    CUBIC = "cubic"
    QUINTIC = "quintic"
    TRAPEZOIDAL = "trapezoidal"


@dataclass
class Trajectory:
    """Container for trajectory data."""
    time: np.ndarray           # Time points (N,)
    positions: np.ndarray       # Joint positions (N, n_joints)
    velocities: np.ndarray      # Joint velocities (N, n_joints)
    accelerations: np.ndarray   # Joint accelerations (N, n_joints)
    trajectory_type: TrajectoryType


# =============================================================================
# Trajectory Generator
# =============================================================================

class TrajectoryGenerator:
    """
    Joint-space trajectory generator.
    
    Generates smooth trajectories between configurations using
    polynomial or trapezoidal velocity profiles.
    """
    
    def __init__(self, n_joints: int):
        """Initialize with number of joints."""
        self.n_joints = n_joints
    
    # =========================================================================
    # Cubic Polynomial
    # =========================================================================
    
    def cubic_polynomial(
        self, 
        q_start: np.ndarray, 
        q_end: np.ndarray,
        v_start: np.ndarray = None,
        v_end: np.ndarray = None,
        T: float = 2.0, 
        N: int = 100
    ) -> Trajectory:
        """
        Generate cubic polynomial trajectory.
        
        q(t) = a0 + a1*t + a2*t² + a3*t³
        
        Boundary conditions: position and velocity at start/end.
        """
        v_start = v_start if v_start is not None else np.zeros(self.n_joints)
        v_end = v_end if v_end is not None else np.zeros(self.n_joints)
        
        t = np.linspace(0, T, N)
        positions = np.zeros((N, self.n_joints))
        velocities = np.zeros((N, self.n_joints))
        accelerations = np.zeros((N, self.n_joints))
        
        for j in range(self.n_joints):
            # Coefficients
            a0 = q_start[j]
            a1 = v_start[j]
            
            # Solve for a2, a3
            A = np.array([[T**2, T**3], [2*T, 3*T**2]])
            b = np.array([q_end[j] - a0 - a1*T, v_end[j] - a1])
            a2, a3 = np.linalg.solve(A, b)
            
            # Evaluate
            for i, ti in enumerate(t):
                positions[i, j] = a0 + a1*ti + a2*ti**2 + a3*ti**3
                velocities[i, j] = a1 + 2*a2*ti + 3*a3*ti**2
                accelerations[i, j] = 2*a2 + 6*a3*ti
        
        return Trajectory(t, positions, velocities, accelerations, TrajectoryType.CUBIC)
    
    # =========================================================================
    # Quintic Polynomial
    # =========================================================================
    
    def quintic_polynomial(
        self, 
        q_start: np.ndarray, 
        q_end: np.ndarray,
        v_start: np.ndarray = None,
        v_end: np.ndarray = None,
        a_start: np.ndarray = None,
        a_end: np.ndarray = None,
        T: float = 2.0, 
        N: int = 100
    ) -> Trajectory:
        """
        Generate quintic polynomial trajectory.
        
        q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        
        Boundary conditions: position, velocity, acceleration at start/end.
        """
        v_start = v_start if v_start is not None else np.zeros(self.n_joints)
        v_end = v_end if v_end is not None else np.zeros(self.n_joints)
        a_start = a_start if a_start is not None else np.zeros(self.n_joints)
        a_end = a_end if a_end is not None else np.zeros(self.n_joints)
        
        t = np.linspace(0, T, N)
        positions = np.zeros((N, self.n_joints))
        velocities = np.zeros((N, self.n_joints))
        accelerations = np.zeros((N, self.n_joints))
        
        for j in range(self.n_joints):
            a0, a1, a2 = q_start[j], v_start[j], a_start[j] / 2
            
            T2, T3, T4, T5 = T**2, T**3, T**4, T**5
            A = np.array([
                [T3, T4, T5],
                [3*T2, 4*T3, 5*T4],
                [6*T, 12*T2, 20*T3]
            ])
            b = np.array([
                q_end[j] - a0 - a1*T - a2*T2,
                v_end[j] - a1 - 2*a2*T,
                a_end[j] - 2*a2
            ])
            a3, a4, a5 = np.linalg.solve(A, b)
            
            for i, ti in enumerate(t):
                ti2, ti3, ti4, ti5 = ti**2, ti**3, ti**4, ti**5
                positions[i, j] = a0 + a1*ti + a2*ti2 + a3*ti3 + a4*ti4 + a5*ti5
                velocities[i, j] = a1 + 2*a2*ti + 3*a3*ti2 + 4*a4*ti3 + 5*a5*ti4
                accelerations[i, j] = 2*a2 + 6*a3*ti + 12*a4*ti2 + 20*a5*ti3
        
        return Trajectory(t, positions, velocities, accelerations, TrajectoryType.QUINTIC)
    
    # =========================================================================
    # Trapezoidal Velocity
    # =========================================================================
    
    def trapezoidal_velocity(
        self, 
        q_start: np.ndarray, 
        q_end: np.ndarray,
        v_max: float = 1.0,
        a_max: float = 2.0,
        T: float = None,
        N: int = 100
    ) -> Trajectory:
        """
        Generate trapezoidal velocity profile trajectory.
        
        Three phases:
        1. Acceleration (constant a)
        2. Cruise (constant v)
        3. Deceleration (constant -a)
        """
        dq = q_end - q_start
        
        # Compute time if not specified
        if T is None:
            T_needed = []
            for j in range(self.n_joints):
                d = abs(dq[j])
                if d < 1e-10:
                    continue
                t_accel = v_max / a_max
                d_accel = a_max * t_accel**2
                if d_accel >= d:
                    T_needed.append(2 * np.sqrt(d / a_max))
                else:
                    T_needed.append(2 * t_accel + (d - d_accel) / v_max)
            T = max(T_needed) if T_needed else 2.0
        
        T = max(T, 0.1)
        t = np.linspace(0, T, N)
        positions = np.zeros((N, self.n_joints))
        velocities = np.zeros((N, self.n_joints))
        accelerations = np.zeros((N, self.n_joints))
        
        # Symmetric profile: accel 25%, cruise 50%, decel 25%
        t_a = T / 4
        t_c = T / 2
        
        for j in range(self.n_joints):
            d = abs(dq[j])
            direction = np.sign(dq[j]) if d > 1e-10 else 0
            
            if d < 1e-10:
                positions[:, j] = q_start[j]
                continue
            
            # Scale velocity/acceleration to fit time
            v = d / (t_a + t_c)
            a = v / t_a
            
            for i, ti in enumerate(t):
                if ti <= t_a:
                    # Acceleration phase
                    positions[i, j] = q_start[j] + direction * 0.5 * a * ti**2
                    velocities[i, j] = direction * a * ti
                    accelerations[i, j] = direction * a
                elif ti <= t_a + t_c:
                    # Cruise phase
                    dt = ti - t_a
                    positions[i, j] = q_start[j] + direction * (0.5 * a * t_a**2 + v * dt)
                    velocities[i, j] = direction * v
                    accelerations[i, j] = 0
                else:
                    # Deceleration phase
                    dt = ti - t_a - t_c
                    d_cruise = 0.5 * a * t_a**2 + v * t_c
                    positions[i, j] = q_start[j] + direction * (d_cruise + v * dt - 0.5 * a * dt**2)
                    velocities[i, j] = direction * (v - a * dt)
                    accelerations[i, j] = -direction * a
            
            # Ensure exact final position
            positions[-1, j] = q_end[j]
            velocities[-1, j] = 0
            accelerations[-1, j] = 0
        
        return Trajectory(t, positions, velocities, accelerations, TrajectoryType.TRAPEZOIDAL)
    
    # =========================================================================
    # Generic Generator
    # =========================================================================
    
    def generate(
        self, 
        trajectory_type: TrajectoryType,
        q_start: np.ndarray, 
        q_end: np.ndarray,
        T: float = 2.0,
        N: int = 100,
        **kwargs
    ) -> Trajectory:
        """Generate trajectory of specified type."""
        if trajectory_type == TrajectoryType.CUBIC:
            return self.cubic_polynomial(q_start, q_end, T=T, N=N, **kwargs)
        elif trajectory_type == TrajectoryType.QUINTIC:
            return self.quintic_polynomial(q_start, q_end, T=T, N=N, **kwargs)
        elif trajectory_type == TrajectoryType.TRAPEZOIDAL:
            return self.trapezoidal_velocity(q_start, q_end, T=T, N=N, **kwargs)
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")


    def generate_multi_waypoint(
        self, 
        trajectory_type: TrajectoryType,
        waypoints: list,
        T_per_segment: float = 1.5,
        N_per_segment: int = 100,
        **kwargs
    ) -> Trajectory:
        """
        Generate trajectory through multiple waypoints.
        
        Args:
            trajectory_type: Type of trajectory planner
            waypoints: List of joint configurations [q0, q1, q2, ...]
            T_per_segment: Time for each segment
            N_per_segment: Samples per segment
            
        Returns:
            Combined trajectory through all waypoints
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        all_positions = []
        all_velocities = []
        all_accelerations = []
        all_times = []
        
        t_offset = 0.0
        
        for i in range(len(waypoints) - 1):
            q_start = waypoints[i]
            q_end = waypoints[i + 1]
            
            # Generate segment
            segment = self.generate(
                trajectory_type, q_start, q_end, 
                T=T_per_segment, N=N_per_segment, **kwargs
            )
            
            # Adjust time offset (skip first point except for first segment)
            if i == 0:
                all_times.append(segment.time + t_offset)
                all_positions.append(segment.positions)
                all_velocities.append(segment.velocities)
                all_accelerations.append(segment.accelerations)
            else:
                all_times.append(segment.time[1:] + t_offset)
                all_positions.append(segment.positions[1:])
                all_velocities.append(segment.velocities[1:])
                all_accelerations.append(segment.accelerations[1:])
            
            t_offset += T_per_segment
        
        return Trajectory(
            time=np.concatenate(all_times),
            positions=np.vstack(all_positions),
            velocities=np.vstack(all_velocities),
            accelerations=np.vstack(all_accelerations),
            trajectory_type=trajectory_type
        )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    gen = TrajectoryGenerator(n_joints=6)
    
    q_start = np.zeros(6)
    q_end = np.array([0.5, -0.3, 0.4, -0.2, 0.3, -0.1])
    
    for ttype in TrajectoryType:
        traj = gen.generate(ttype, q_start, q_end)
        print(f"{ttype.value}: {traj.positions.shape}")
