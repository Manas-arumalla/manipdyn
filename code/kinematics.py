"""
Forward and Inverse Kinematics

Implements:
- Forward Kinematics using DH transformation matrices
- Inverse Kinematics using Damped Least Squares (DLS)
- Jacobian computation for velocity mapping
"""

import numpy as np
from typing import Tuple, Optional, List

from robot import Robot


class Kinematics:
    """
    Kinematics solver for robot arm.
    
    Computes FK/IK transformations using standard DH convention.
    """
    
    def __init__(self, robot: Robot):
        """Initialize with robot model."""
        self.robot = robot
    
    # =========================================================================
    # DH Transformation
    # =========================================================================
    
    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Compute 4x4 DH transformation matrix.
        
        Args:
            a: Link length
            alpha: Link twist
            d: Link offset
            theta: Joint angle
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
    
    # =========================================================================
    # Forward Kinematics
    # =========================================================================
    
    def forward_kinematics(self, q: np.ndarray, return_all_frames: bool = False):
        """
        Compute end-effector pose from joint angles.
        
        Args:
            q: Joint angles (n_joints,)
            return_all_frames: If True, return intermediate frames
            
        Returns:
            4x4 end-effector transformation (or list of all frames)
        """
        T = np.eye(4)
        frames = [T.copy()]
        
        for dh, qi in zip(self.robot.dh_params, q):
            theta = dh.theta + qi
            Ti = self.dh_transform(dh.a, dh.alpha, dh.d, theta)
            T = T @ Ti
            frames.append(T.copy())
        
        return frames if return_all_frames else T
    
    def get_position(self, q: np.ndarray) -> np.ndarray:
        """Get end-effector position (x, y, z)."""
        T = self.forward_kinematics(q)
        return T[:3, 3]
    
    def get_orientation(self, q: np.ndarray) -> np.ndarray:
        """Get end-effector rotation matrix."""
        T = self.forward_kinematics(q)
        return T[:3, :3]
    
    # =========================================================================
    # Jacobian
    # =========================================================================
    
    def jacobian(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """
        Compute 6xN geometric Jacobian numerically.
        
        Args:
            q: Joint angles
            delta: Perturbation for differentiation
            
        Returns:
            6 x n_joints Jacobian (linear + angular velocities)
        """
        n = self.robot.n_joints
        J = np.zeros((6, n))
        
        T0 = self.forward_kinematics(q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]
        
        for i in range(n):
            q_plus = q.copy()
            q_plus[i] += delta
            
            T_plus = self.forward_kinematics(q_plus)
            p_plus = T_plus[:3, 3]
            R_plus = T_plus[:3, :3]
            
            # Linear velocity component
            J[:3, i] = (p_plus - p0) / delta
            
            # Angular velocity from rotation difference
            dR = (R_plus - R0) / delta
            J[3, i] = dR[2, 1]  # omega_x
            J[4, i] = dR[0, 2]  # omega_y
            J[5, i] = dR[1, 0]  # omega_z
        
        return J
    
    # =========================================================================
    # Inverse Kinematics
    # =========================================================================
    
    def inverse_kinematics(
        self, 
        target_pose: np.ndarray, 
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        damping: float = 0.1
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Compute IK using Damped Least Squares (Levenberg-Marquardt).
        
        Args:
            target_pose: 4x4 target transformation matrix
            q_init: Initial guess (default: home position)
            max_iterations: Maximum iterations
            tolerance: Position error tolerance (m)
            damping: Damping factor for stability
            
        Returns:
            q: Joint angles solution
            success: Whether solution found
            error: Final position error
        """
        q = q_init.copy() if q_init is not None else self.robot.get_home_position()
        
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]
        
        for _ in range(max_iterations):
            T_current = self.forward_kinematics(q)
            current_pos = T_current[:3, 3]
            current_rot = T_current[:3, :3]
            
            # Position error
            pos_error = target_pos - current_pos
            
            # Orientation error
            rot_error_matrix = target_rot @ current_rot.T
            rot_error = self._rotation_to_axis_angle(rot_error_matrix)
            
            # Combined error
            error = np.concatenate([pos_error, rot_error])
            error_norm = np.linalg.norm(pos_error)
            
            if error_norm < tolerance:
                return self.robot.clamp_to_limits(q), True, error_norm
            
            # DLS update: q += J^T (J J^T + λ²I)^{-1} e
            J = self.jacobian(q)
            JJT = J @ J.T
            damped = JJT + (damping ** 2) * np.eye(6)
            q_delta = J.T @ np.linalg.solve(damped, error)
            
            q = self.robot.clamp_to_limits(q + q_delta)
        
        final_error = np.linalg.norm(self.get_position(q) - target_pos)
        return q, final_error < tolerance * 10, final_error
    
    def _rotation_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle vector."""
        trace = np.trace(R)
        theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if theta < 1e-6:
            return np.zeros(3)
        
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))
        
        return axis * theta
    
    # =========================================================================
    # Manipulability
    # =========================================================================
    
    def manipulability(self, q: np.ndarray) -> float:
        """
        Compute Yoshikawa manipulability measure.
        
        Higher values = further from singularity.
        """
        J = self.jacobian(q)
        return np.sqrt(np.linalg.det(J @ J.T))


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from .robot import create_robot
    
    robot = create_robot()
    kin = Kinematics(robot)
    
    q = np.array([0, -np.pi/4, np.pi/4, 0, np.pi/4, 0])
    print(f"Position: {kin.get_position(q)}")
    print(f"Manipulability: {kin.manipulability(q):.4f}")
