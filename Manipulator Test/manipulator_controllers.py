
import numpy as np
from scipy.linalg import solve_continuous_are
import manipulator_dynamics
import mujoco

class LQRController:
    def __init__(self, model, data, Q_diag_pos, Q_diag_vel, R_diag, q_target):
        """
        LQR Controller linearized around q_target.
        """
        self.model = model
        self.data = data
        self.q_target = np.array(q_target)
        
        # 1. Linearize Dynamics around q_target
        print(f"Linearizing dynamics around {q_target}...")
        self.A, self.B = manipulator_dynamics.get_linearized_dynamics(model, data, self.q_target)
        
        # 2. Construct Cost Matrices
        # Q size: 2*nv x 2*nv (Pos, Vel)
        # R size: nu x nu
        nv = model.nv
        nu = model.nu
        
        self.Q = np.zeros((2*nv, 2*nv))
        self.Q[:nv, :nv] = np.eye(nv) * Q_diag_pos
        self.Q[nv:, nv:] = np.eye(nv) * Q_diag_vel
        
        self.R = np.eye(nu) * R_diag
        
        # 3. Solve LQR (CARE)
        # A.T * P + P * A - P * B * R^-1 * B.T * P + Q = 0
        print("Solving LQR Riccati Equation...")
        try:
            self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            
            # K = R^-1 * B.T * P
            self.K = np.linalg.inv(self.R) @ self.B.T @ self.P
            print("LQR Gain Computed Successfully.")
            print(f"K shape: {self.K.shape}")
            
        except Exception as e:
            print(f"LQR Computation Failed: {e}")
            self.K = np.zeros((nu, 2*nv))

    def update(self, q, v, target_q=None):
        """
        Compute control input u = -K * (x - x_desired)
        target_q: Optional override for trajectory tracking
        """
        if target_q is None:
            target_q = self.q_target
            
        # Error State
        q_err = q - target_q
        v_err = v - 0.0 # Target vel is 0
        
        x_err = np.concatenate([q_err, v_err])
        
        # Feedback Law
        u_lqr = -self.K @ x_err
        
        return u_lqr

class PIDController:
    def __init__(self, model, data, kp=100, ki=1, kd=20, output_limit=100):
        """
        Joint-Space PID Controller.
        u = Kp*e + Ki*int_e + Kd*dot_e
        """
        self.model = model
        self.data = data
        self.nq = 6 # Force 6 DoF for Arm Control
        self.kp = np.array([kp] * self.nq)
        self.ki = np.array([ki] * self.nq)
        self.kd = np.array([kd] * self.nq)
        self.output_limit = output_limit
        
        self.integral_error = np.zeros(self.nq)
        self.prev_time = 0.0
        
    def update(self, q, v, target_q):
        """
        Compute PID control signal.
        """
        # Time Step
        curr_time = self.data.time
        dt = curr_time - self.prev_time
        if dt <= 0: dt = 1e-4 # Protect against zero div
        self.prev_time = curr_time
        
        # Errors
        q_err = target_q - q
        v_err = 0 - v # Assuming target vel is 0
        
        # Integral
        self.integral_error += q_err * dt
        
        # Anti-windup (Simple Clamping)
        int_limit = 20.0
        self.integral_error = np.clip(self.integral_error, -int_limit, int_limit)
        
        # Control Law
        u_p = self.kp * q_err
        u_i = self.ki * self.integral_error
        u_d = self.kd * v_err
        
        u = u_p + u_i + u_d
        
        # Output Saturation
        u = np.clip(u, -self.output_limit, self.output_limit)
        
        return u

class CTCController:
    def __init__(self, model, data, kp=100, kd=20):
        """
        Computed Torque Control (Feedback Linearization).
        Uses inverse dynamics to cancel nonlinearities.
        """
        self.model = model
        # Use a separate data structure for inverse dynamics computations
        # to avoid modifying the main simulation state during step
        self.data_ctc = mujoco.MjData(model)
        self.kp = kp
        self.kd = kd
        
    def update(self, q, v, target_q, target_v=None, target_acc=None):
        """
        Calculates torque: tau = M(q)*(ddq_des + Kp*e + Kd*de) + C + g
        """
        if target_v is None: target_v = np.zeros_like(v)
        if target_acc is None: target_acc = np.zeros_like(v)
        
        # Sync internal data with current robot state
        self.data_ctc.qpos[:self.model.nq] = q
        self.data_ctc.qvel[:self.model.nv] = v
        self.data_ctc.qacc[:self.model.nv] = 0 # Clear accel just in case
        
        # Desired Acceleration (PD Output)
        q_err = target_q - q
        v_err = target_v - v
        
        aq = target_acc + self.kp * q_err + self.kd * v_err
        
        # Set desired accel in internal data
        self.data_ctc.qacc[:self.model.nv] = aq
        
        # Compute Inverse Dynamics
        # This computes `qfrc_inverse` which is the torque needed 
        # to produce `qacc` given `qpos` and `qvel`.
        # Includes M*aq + C + g
        mujoco.mj_inverse(self.model, self.data_ctc)
        
        # Return the computed torque
        return self.data_ctc.qfrc_inverse[:self.model.nu].copy()

class ImpedanceController:
    def __init__(self, model, data, site_name="attachment_site", kp=500, kd=50, limit=100):
        """
        Cartesian Impedance Controller (Jacobian Transpose).
        F = Kp * (x_des - x) + Kd * (v_des - v)
        tau = J.T * F
        """
        self.model = model
        self.data_imp = mujoco.MjData(model) # Separate data needed for Jac calculation
        
        self.kp = kp
        self.kd = kd
        self.limit = limit
        
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
             # Try fallback "eef_site"
             fallback = "eef_site"
             self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fallback)
             if self.site_id == -1:
                 print(f"CRITICAL ERROR: Site '{site_name}' and '{fallback}' NOT FOUND. Impedance will fail (Torque=0).")
                 self.site_id = 0
             else:
                 print(f"ImpedanceController: '{site_name}' not found. Using '{fallback}' (ID {self.site_id}).")
        else:
             print(f"ImpedanceController: Found '{site_name}' (ID {self.site_id}).")
             
        self.print_timer = 0
            
    def update(self, q, v, target_x, target_v_x=None):
        """
        Compute torque.
        target_x: (3,) Cartesian position
        """
        if target_v_x is None: target_v_x = np.zeros(3)
        
        # Sync internal state
        self.data_imp.qpos[:self.model.nq] = q
        self.data_imp.qvel[:self.model.nv] = v
        
        # Use mj_forward to ensure ALL kinematic/dynamic properties (including site matrices) are updated.
        # mj_kinematics might be insufficient for mj_jacSite depending on flags.
        mujoco.mj_forward(self.model, self.data_imp)
        
        # Current State
        curr_x = self.data_imp.site_xpos[self.site_id]
        
        # Jacobian (3xnv) -> Only use first 6 cols for Arm
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data_imp, jacp, jacr, self.site_id)
        
        # Slice to 6 DoF
        J_arm = jacp[:, :6] 
        
        # Current Cartesian Velocity: J * v_arm
        v_arm = v[:6] 
        curr_v_x = J_arm @ v_arm
        
        # Spring-Damper Force
        err_x = target_x - curr_x
        err_v = (target_v_x - curr_v_x) if target_v_x is not None else -curr_v_x
        
        F = self.kp * err_x + self.kd * err_v
        
        # Jacobian Transpose Logic: tau = J^T * F
        tau = J_arm.T @ F
        
        # Clamp
        tau = np.clip(tau, -self.limit, self.limit)
        
        return tau

class MPPIController:
    def __init__(self, model, data, horizon=30, n_samples=50, noise_sigma=2.0, lambda_=0.05):
        """
        Model Predictive Path Integral (MPPI) Controller.
        Sampling-based MPC.
        """
        self.model = model
        self.data = data
        self.horizon = horizon
        self.n_samples = n_samples
        self.nu = model.nu
        self.noise_sigma = noise_sigma
        self.lambda_ = lambda_
        
        # Rollout Data (Separate instance)
        self.d_rollout = mujoco.MjData(model)
        
        # Control Schedule (H x nu)
        self.U = np.zeros((self.horizon, self.nu))
        print(f"MPPI Initialized: H={horizon}, N={n_samples}, Sigma={noise_sigma}, Lambda={lambda_}")
        
    def update(self, q, v, target_q):
        """
        Compute control input using MPPI.
        """
        # 1. Shift Schedule
        self.U[:-1] = self.U[1:]
        self.U[-1] = np.zeros(self.nu) # Initialize last step with zeros
        
        # 2. Sample Noise (K x H x nu)
        noise = np.random.normal(0, self.noise_sigma, (self.n_samples, self.horizon, self.nu))
        
        costs = np.zeros(self.n_samples)
        
        # 3. Rollouts (Sequential loop - Python is slow, but model is simple)
        for k in range(self.n_samples):
            # Reset Rollout State
            self.d_rollout.qpos[:self.model.nq] = q
            self.d_rollout.qvel[:self.model.nv] = v
            # Need to zero out accel/forces? mj_step handles it usually.
            
            # Simulate Horizon
            for t in range(self.horizon):
                # Apply Control: u = U_mean + variance
                u_sample = self.U[t] + noise[k, t]
                
                # Clip?
                # u_sample = np.clip(u_sample, -50, 50)
                
                self.d_rollout.ctrl[:self.nu] = u_sample + self.d_rollout.qfrc_bias[:self.nu] # Add gravity comp to help
                
                mujoco.mj_step(self.model, self.d_rollout)
                
                # Compute Cost (State Cost + Control Cost)
                q_curr = self.d_rollout.qpos[:6]
                q_err = target_q - q_curr
                
                # Tuning: Increase state cost weight to prioritize reaching target
                state_cost = 5000 * np.dot(q_err, q_err) 
                
                # Velocity Cost (Damping): Penalize high speeds to reduce oscillation near target
                q_vel = self.d_rollout.qvel[:6]
                vel_cost = 20.0 * np.dot(q_vel, q_vel)
                
                ctrl_cost = 0.001 * np.dot(u_sample, u_sample)
                
                costs[k] += state_cost + vel_cost + ctrl_cost
                
        # 4. Compute Weights (Softmax)
        # Numerical stability trick: subtract min cost
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / self.lambda_)
        weights = exp_costs / np.sum(exp_costs)
        
        # 5. Update Mean Control Schedule
        # U_new = sum(w * (U + noise))
        weighted_noise = np.zeros((self.horizon, self.nu))
        for k in range(self.n_samples):
            weighted_noise += weights[k] * noise[k]
            
        self.U += weighted_noise
        
        # Return first control step
        return self.U[0].copy()
