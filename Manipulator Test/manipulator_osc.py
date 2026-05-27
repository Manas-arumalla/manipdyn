
import numpy as np
import mujoco

class OSCController:
    def __init__(self, model, data, kp=150, kd=20, null_kp=10):
        """
        Operational Space Controller (OSC) implementation.
        Controls end-effector pos/ori while handling redundancy in null-space.
        """
        self.model = model
        self.data = data
        self.kp = kp
        self.kd = kd
        self.null_kp = null_kp
        
        # End-effector
        self.site_name = "attachment_site" 
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        if self.site_id == -1:
            self.site_name = "eef_site"
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)

        # Integration: Sparse mass matrix
        self.nv = model.nv
        self.M_dense = np.zeros((self.nv, self.nv))
        
    def update(self, target_pos, target_vel=None):
        """
        Compute torque to reach target_pos (Cartesian).
        target_vel is optional feedforward (assume 0 if None).
        """
        if target_vel is None:
            target_vel = np.zeros(3)
            
        # 1. Get Jacobian (Translation only for now, can extend to 6D)
        # J: 3 x nv
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        
        # Select active joints (UR5e: 6DOF).
        # We'll use full Jacobian of size nv usually.
        # Ideally we control 3D Pos + 3D Ori (6D task). 
        # For simple OSC, let's do 3D Position first.
        # If nv > 6, we validly assume first 6 are Arm.
        J = jacp[:, :6] # 3 x 6
        
        # 2. Get Mass Matrix M(q)
        mujoco.mj_fullM(self.model, self.M_dense, self.data.qM)
        M = self.M_dense[:6, :6]
        
        # 3. Compute Lambda (Task Space Inertia)
        # Lambda = (J * M^-1 * J^T)^-1
        # Efficiently: Solve M * X = J^T  -> X = M^-1 J^T
        # Then Y = J * X = J M^-1 J^T
        # Then Lambda = inv(Y)
        
        # Or faster using MuJoCo solver if M is factorized? 
        # mj_solveM solves M*x = b.
        # But we need Explicit Lambda usually for F = Lambda * a.
        
        Minv_JT = np.linalg.solve(M, J.T) # Solve A*x=B (A=M, B=J^T) -> x = M^-1 J^T
        J_Minv_JT = J @ Minv_JT
        
        # Add damping for singularity stability?
        # J_Minv_JT += 1e-6 * np.eye(3)
        
        try:
            Lambda = np.linalg.inv(J_Minv_JT)
        except np.linalg.LinAlgError:
            Lambda = np.eye(3) # Fallback
            
        # 4. Compute Control Error (Task Space)
        x_curr = self.data.site_xpos[self.site_id]
        dx_curr = J @ self.data.qvel[:self.nv]
        
        x_err = target_pos - x_curr
        dx_err = target_vel - dx_curr
        
        # 5. Desired Task Force (Decoupled)
        # F = Lambda * (kp * e + kd * de)
        # Note: We ignore mu(q) and p(q) (Coriolis/Gravity in task space) 
        # because we will add Joint-Space Gravity Comp at the end.
        # Rigorous OSC subtracts J_dot * q_dot, but generally negligible for regulation.
        
        F_task = Lambda @ (self.kp * x_err + self.kd * dx_err)
        
        # 6. Map to Joint Torques
        tau_task = J.T @ F_task
        
        # 7. Nullspace Control (Secondary Task: Maintain Home Pose)
        # P = I - J^T * Lambda * J * M^-1  (Dynamically consistent pseudoinverse)
        # More simply: P = I - J_pinv * J
        # tau_null = P * (Kp_null * (q_home - q) - Kd_null * dq)
        
        # Calculate Dynamically Consistent J_bar = M^-1 J^T Lambda
        J_bar = Minv_JT @ Lambda
        
        # Null projection matrix N = I - J^T * J_bar^T
        # Check shapes: J: 3x6, J_bar: 6x3. J^T: 6x3.
        
        N = np.eye(6) - J.T @ J_bar.T
        
        # Secondary torque (Joint PD to home)
        q_home = np.zeros(6)
        q_home[1] = -1.57; q_home[2] = 1.57; q_home[3] = -1.57; q_home[4] = -1.57 # Nicer home
        
        # Current Arm State
        q_arm = self.data.qpos[:6]
        dq_arm = self.data.qvel[:6]
        
        tau_null_raw = self.null_kp * (q_home - q_arm) - 5.0 * dq_arm
        tau_null = N @ tau_null_raw
        
        # 8. Total Torque
        # Return only task + nullspace torque. 
        # Gravity comp will be added by the unified simulation loop.
        
        tau_total = tau_task + tau_null
        
        # If model has more joints (e.g. gripper), pad with zeros?
        # Actually caller expects 6D torque if we treat it as 6 control.
        # Sim loop handles the padding/application to d.ctrl[:6].
        
        return tau_total

    def get_control(self, target_pos, target_vel=None):
        return self.update(target_pos, target_vel)
