"""
Robot Arm Definition

6-DOF articulated robot arm using Denavit-Hartenberg (DH) convention.
Based on UR5-like dimensions for realistic simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DHParameters:
    """Denavit-Hartenberg parameters for a single joint."""
    a: float      # Link length (along x-axis)
    alpha: float  # Link twist (rotation about x-axis)
    d: float      # Link offset (along z-axis)
    theta: float  # Joint angle offset (rotation about z-axis)


@dataclass
class JointLimits:
    """Joint motion limits."""
    q_min: float   # Minimum position (rad)
    q_max: float   # Maximum position (rad)
    dq_max: float  # Maximum velocity (rad/s)
    tau_max: float # Maximum torque (Nm)


# =============================================================================
# Robot Class
# =============================================================================

class Robot:
    """
    6-DOF Articulated Robot Arm
    
    Configuration based on UR5 industrial robot:
    - 6 revolute joints
    - Anthropomorphic design (shoulder-elbow-wrist)
    - Standard DH convention
    """
    
    def __init__(self):
        """Initialize robot with UR5-like parameters."""
        
        # Link dimensions (meters)
        self.d1 = 0.089159   # Base height
        self.a2 = 0.425      # Upper arm length
        self.a3 = 0.39225    # Forearm length
        self.d4 = 0.10915    # Wrist 1 offset
        self.d5 = 0.09465    # Wrist 2 offset
        self.d6 = 0.0823     # Tool flange
        
        # DH Parameters: [a, alpha, d, theta_offset]
        self.dh_params = [
            DHParameters(a=0,        alpha=np.pi/2,  d=self.d1, theta=0),
            DHParameters(a=self.a2,  alpha=0,        d=0,       theta=0),
            DHParameters(a=self.a3,  alpha=0,        d=0,       theta=0),
            DHParameters(a=0,        alpha=np.pi/2,  d=self.d4, theta=0),
            DHParameters(a=0,        alpha=-np.pi/2, d=self.d5, theta=0),
            DHParameters(a=0,        alpha=0,        d=self.d6, theta=0),
        ]
        
        # Joint limits
        self.joint_limits = [
            JointLimits(q_min=-2*np.pi, q_max=2*np.pi, dq_max=3.0, tau_max=150),
            JointLimits(q_min=-2*np.pi, q_max=2*np.pi, dq_max=3.0, tau_max=150),
            JointLimits(q_min=-np.pi,   q_max=np.pi,   dq_max=3.0, tau_max=150),
            JointLimits(q_min=-2*np.pi, q_max=2*np.pi, dq_max=3.0, tau_max=28),
            JointLimits(q_min=-2*np.pi, q_max=2*np.pi, dq_max=3.0, tau_max=28),
            JointLimits(q_min=-2*np.pi, q_max=2*np.pi, dq_max=3.0, tau_max=28),
        ]
        
        self.n_joints = 6
        self.link_masses = [3.7, 8.4, 2.2, 1.2, 1.2, 0.2]  # kg
        self.gravity = np.array([0, 0, -9.81])
    
    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        return self.n_joints
    
    def get_home_position(self) -> np.ndarray:
        """Return home (zero) position."""
        return np.zeros(self.n_joints)
    
    def get_random_configuration(self) -> np.ndarray:
        """Generate random valid joint configuration."""
        q = np.zeros(self.n_joints)
        for i, limits in enumerate(self.joint_limits):
            q[i] = np.random.uniform(limits.q_min, limits.q_max)
        return q
    
    def check_joint_limits(self, q: np.ndarray) -> bool:
        """Check if configuration is within limits."""
        for i, (qi, limits) in enumerate(zip(q, self.joint_limits)):
            if qi < limits.q_min or qi > limits.q_max:
                return False
        return True
    
    def clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint values to limits."""
        q_clamped = q.copy()
        for i, limits in enumerate(self.joint_limits):
            q_clamped[i] = np.clip(q[i], limits.q_min, limits.q_max)
        return q_clamped


# =============================================================================
# Factory Function
# =============================================================================

def create_robot() -> Robot:
    """Create and return a robot instance."""
    return Robot()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    robot = create_robot()
    print(f"Created {robot.n_dof}-DOF Robot Arm")
    print(f"Home position: {robot.get_home_position()}")
