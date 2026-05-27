
import mujoco
import numpy as np

class IKSolver:
    def __init__(self, model, data, step_size=0.5, max_iter=100, tol=1e-3, damping=0.1):
        """
        Inverse Kinematics Solver using Damped Least Squares (DLS).
        """
        self.model = model
        self.data = data
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping # Damping factor (lambda)
        
        # End-Effector site name (Must exist in XML)
        self.site_name = "attachment_site" 
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        if self.site_id == -1:
            # Fallback to UR5e specific eef name if generic not found
            self.site_name = "eef_site"
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
            
    def solve(self, target_pos, target_quat=None, q_guess=None):
        """
        Solve IK for target_pos (x,y,z).
        Optional: target_quat (w,x,y,z) for orientation.
        Returns: q_solution (np.array) or None if failed.
        """
        if q_guess is not None:
            self.data.qpos[:6] = q_guess
            mujoco.mj_forward(self.model, self.data)
            
        # Jacobian Arrays (Allocated once)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        
        success = False
        err_norm = 0.0
        
        for i in range(self.max_iter):
            # 1. Forward Kinematics
            curr_pos = self.data.site_xpos[self.site_id]
            curr_quat = np.zeros(4)
            mujoco.mju_mat2Quat(curr_quat, self.data.site_xmat[self.site_id])
            
            # 2. Error Vector (dx)
            err_pos = target_pos - curr_pos
            
            err_rot = np.zeros(3)
            if target_quat is not None:
                mujoco.mju_subQuat(err_rot, target_quat, curr_quat)
            
            # Combine errors
            if target_quat is not None:
                err = np.concatenate([err_pos, err_rot])
                J_target = np.zeros((6, self.model.nv))
            else:
                err = err_pos
                J_target = np.zeros((3, self.model.nv))
                
            err_norm = np.linalg.norm(err)
            if err_norm < self.tol:
                success = True
                break
                
            # 3. Compute Jacobian
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            
            Jp = jacp[:, :6]
            Jr = jacr[:, :6]
            
            if target_quat is not None:
                J = np.vstack([Jp, Jr])
            else:
                J = Jp
                
            # 4. Damped Least Squares Update
            n = J.shape[0] 
            lambda_sq = self.damping ** 2
            
            # Matrix to invert: (J J^T + lambda^2 I)
            A = J @ J.T + lambda_sq * np.eye(n)
            
            try:
                # dq = J.T * inv(A) * err
                temp_x = np.linalg.solve(A, err)
                dq = J.T @ temp_x
            except np.linalg.LinAlgError:
                dq = np.zeros(6)
            
            # 5. Integrate State
            self.data.qpos[:6] += self.step_size * dq
            mujoco.mj_forward(self.model, self.data)
            
        if success:
            return self.data.qpos[:6].copy()
        else:
            return None
