"""
PID Controller

Joint-space PID control for trajectory tracking.
Includes anti-windup and torque limiting.
"""

import numpy as np
from dataclasses import dataclass


# =============================================================================
# Types
# =============================================================================

@dataclass
class PIDGains:
    """PID controller gains."""
    Kp: np.ndarray  # Proportional
    Ki: np.ndarray  # Integral
    Kd: np.ndarray  # Derivative


# =============================================================================
# PID Controller
# =============================================================================

class PIDController:
    """
    Joint-space PID Controller.
    
    tau = Kp*(q_d - q) + Ki*∫(q_d - q)dt + Kd*(dq_d - dq)
    """
    
    def __init__(self, n_joints: int, gains: PIDGains = None):
        """
        Initialize controller.
        
        Args:
            n_joints: Number of joints
            gains: PID gains (default: tuned values)
        """
        self.n_joints = n_joints
        
        if gains is None:
            self.gains = PIDGains(
                Kp=np.ones(n_joints) * 100.0,
                Ki=np.ones(n_joints) * 5.0,
                Kd=np.ones(n_joints) * 20.0
            )
        else:
            self.gains = gains
        
        self.integral_error = np.zeros(n_joints)
        self.integral_limit = 10.0
        self.tau_max = np.array([150, 150, 150, 28, 28, 28])
    
    def reset(self):
        """Reset integral term."""
        self.integral_error = np.zeros(self.n_joints)
    
    def compute(
        self, 
        q_desired: np.ndarray, 
        dq_desired: np.ndarray,
        q_actual: np.ndarray, 
        dq_actual: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute control torques.
        
        Args:
            q_desired: Desired positions
            dq_desired: Desired velocities
            q_actual: Current positions
            dq_actual: Current velocities
            dt: Time step
            
        Returns:
            tau: Control torques
        """
        # Errors
        pos_error = q_desired - q_actual
        vel_error = dq_desired - dq_actual
        
        # Integral with anti-windup
        self.integral_error += pos_error * dt
        self.integral_error = np.clip(
            self.integral_error, 
            -self.integral_limit, 
            self.integral_limit
        )
        
        # PID law
        tau = (
            self.gains.Kp * pos_error + 
            self.gains.Ki * self.integral_error + 
            self.gains.Kd * vel_error
        )
        
        # Torque limits
        return np.clip(tau, -self.tau_max, self.tau_max)
    
    def set_gains(self, Kp: np.ndarray, Ki: np.ndarray, Kd: np.ndarray):
        """Update gains."""
        self.gains = PIDGains(Kp=Kp, Ki=Ki, Kd=Kd)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    controller = PIDController(n_joints=6)
    
    q_desired = np.array([0.5, -0.3, 0.4, -0.2, 0.3, -0.1])
    q, dq = np.zeros(6), np.zeros(6)
    
    for _ in range(100):
        tau = controller.compute(q_desired, np.zeros(6), q, dq, 0.01)
        dq += tau / 10.0 * 0.01
        q += dq * 0.01
    
    print(f"Final error: {np.linalg.norm(q - q_desired):.4f}")
