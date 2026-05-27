
import time
import numpy as np
import mujoco
import mujoco.viewer
import manipulator_ik
import manipulator_controllers

import json
import os

# Configuration
XML_PATH = "scene.xml"

# Default Params
Q_POS = 1000.0
Q_VEL = 10.0
R_VAL = 1.0
TARGET_POS = np.array([0.4, -0.4, 0.4])

# Load Config if exists
if os.path.exists("sim_config.json"):
    try:
        with open("sim_config.json", "r") as f:
            cfg = json.load(f)
            lqr_cfg = cfg.get("lqr", {})
            Q_POS = lqr_cfg.get("q_pos", Q_POS)
            R_VAL = lqr_cfg.get("r", R_VAL)
            
            tgt = cfg.get("target", None)
            if tgt: TARGET_POS = np.array(tgt)
            
            # Update XML Path
            if "xml_path" in cfg:
                XML_PATH = cfg["xml_path"]
            
            SHOW_TARGET = cfg.get("show_target", True)
                
            print(f"Loaded Config: Target={TARGET_POS}, XML={XML_PATH}")
    except:
        print("Failed to load config, using defaults.")
        SHOW_TARGET = True

# ...
    # Update marker
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
    if site_id != -1:
        m.site_pos[site_id] = TARGET_POS
        if SHOW_TARGET:
            m.site_rgba[site_id] = [1, 0, 0, 1]
        else:
            m.site_rgba[site_id] = [0, 0, 0, 0]

def main():
    # Load model
    print(f"Loading model from {XML_PATH}...")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # 1. Define Target Pose (Cartesian)
    target_pos = TARGET_POS # Forward-Right-Up
    print(f"Target Position: {target_pos}")
    
    # Update marker pos in viz
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
    if site_id != -1:
        m.site_pos[site_id] = target_pos
    
    # 2. Solve IK
    print("Solving Inverse Kinematics...")
    ik = manipulator_ik.IKSolver(m, d)
    
    # Reset to home to solve from a good seed
    d.qpos[:6] = np.zeros(6) 
    mujoco.mj_forward(m, d)
    
    q_sol, success, err = ik.solve(target_pos)
    
    if success:
        print(f"IK Solved! Error: {err:.4f}")
        print(f"Solution: {q_sol}")
    else:
        print(f"IK Failed to Converge. Error: {err:.4f}")
        # Proceed anyway to see best effort
        
    # 3. Move to Solution using LQR
    # We linearize LQR around the SOLUTION we just found.
    # This ensures perfect stability at the target.
    print("Computing LQR Gains for Target...")
    lqr = manipulator_controllers.LQRController(m, d, Q_POS, Q_VEL, R_VAL, q_sol)
    
    # Simulation loop
    print("Starting simulation to reach IK Target...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        
        # Reset state (Start from home)
        d.qpos[:6] = np.zeros(6) 
        d.qvel[:6] = np.zeros(6)
        
        while viewer.is_running():
            step_start = time.time()
            
            # Current
            q = d.qpos[:6]
            dq = d.qvel[:6]
            
            # Control Law (Regulator to q_sol)
            # Standard Regulator: u = -K(x - x_target)
            # LQR Controller class encapsulates this logic
            
            # Update target in LQR just in case (though initialized with it)
            lqr.q_target = q_sol
            
            u_lqr = lqr.update(q, dq)
            d.ctrl = d.qfrc_bias[:6] + u_lqr
            
            mujoco.mj_step(m, d)
            viewer.sync()
            
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
