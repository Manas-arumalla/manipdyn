
import mujoco
import numpy as np

def get_linearized_dynamics(model, data, q_target):
    """
    Compute Linearized Dynamics (A, B) around a target configuration q_target (with v=0).
    State x = [q, v] (Size 2*nq)
    Input u = [ctrl] (Size nu)
    
    Returns A, B matrices for the continuous system:
    dx/dt = A*x + B*u
    """
    nv = model.nv
    nu = model.nu
    
    # Save current state
    q_save = data.qpos.copy()
    v_save = data.qvel.copy()
    c_save = data.ctrl.copy()
    
    # Set linearization point
    data.qpos[:len(q_target)] = q_target
    data.qvel[:] = 0.0
    
    # Compute gravity compensation torque at this point
    mujoco.mj_forward(model, data)
    u_gravity = data.qfrc_bias[:nu].copy()
    data.ctrl[:] = u_gravity # Linearize around equilibrium (gravity compensated)
    
    # Finite Difference
    # A mapping from (q, v) -> (q_dot, v_dot)
    # B mapping from u -> (q_dot, v_dot)
    
    eps = 1e-6
    flg_centered = True
    
    # We need full A (2*nv x 2*nv) and B (2*nv x nu)
    # MuJoCo mjd_transitionFD computes discrete transition matrix if dt > 0 ?
    # Actually mjd_transitionFD computes discrete approximation.
    # To get continuous A, B we can rely on mj_forward derivatives or use small dt.
    # But usually for LQR, continuous A, B is preferred:
    # A = [ 0   I ]
    #     [ M^-1 K  M^-1 D ]
    # This is hard to extract directly efficiently without derivatives.
    
    # Let's use mjd_transitionFD which gives the discrete Jacobian for the next step.
    # A_d = I + A_c * dt
    # B_d = B_c * dt
    # So A_c approx (A_d - I)/dt, B_c approx B_d / dt
    
    n = 2 * nv
    m = nu
    
    A_d = np.zeros((n, n))
    B_d = np.zeros((n, m))
    
    # Compute discrete Jacobians
    mujoco.mjd_transitionFD(model, data, eps, flg_centered, A_d, B_d, None, None)
    
    dt = model.opt.timestep
    
    # Convert to Continuous
    # x_{k+1} = A_d x_k + B_d u_k
    # (x + dx*dt) = A_d x + B_d u
    # dx/dt = (A_d - I)/dt * x + (B_d)/dt * u
    
    A_c = (A_d - np.eye(n)) / dt
    B_c = B_d / dt
    
    # Restore State
    data.qpos[:] = q_save
    data.qvel[:] = v_save
    data.ctrl[:] = c_save
    mujoco.mj_forward(model, data) # Sync
    
    return A_c, B_c
