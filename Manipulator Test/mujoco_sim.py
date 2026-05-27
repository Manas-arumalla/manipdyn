import time
import numpy as np
import mujoco
import mujoco.viewer
import manipulator_dynamics
import manipulator_controllers
import manipulator_ik
import manipulator_planner
import manipulator_osc
import json
import os
import matplotlib.pyplot as plt

# Configuration (Defaults)
XML_PATH = "scene.xml"
Q_POS = 1000.0
Q_VEL = 10.0
R_VAL = 1.0
START_POS = np.array([0.4, -0.4, 0.4]) # Default start
WAYPOINTS = [] # List of [x,y,z]
SHOW_TARGET = True
PLANNER_MODE = "direct"
CTRL_MODE = "lqr"

# Load Config
if os.path.exists("sim_config.json"):
    try:
        with open("sim_config.json", "r") as f:
            cfg = json.load(f)
            lqr_cfg = cfg.get("lqr", {})
            Q_POS = lqr_cfg.get("q_pos", Q_POS)
            Q_VEL = lqr_cfg.get("q_vel", Q_VEL)
            R_VAL = lqr_cfg.get("r", R_VAL)
            
            if "xml_path" in cfg: XML_PATH = cfg["xml_path"]
            
            SHOW_TARGET = cfg.get("show_target", True)
            PLANNER_MODE = cfg.get("planner", "direct")
            CTRL_MODE = cfg.get("controller", "lqr")
            
            if "start_pos" in cfg and cfg["start_pos"]:
                START_POS = np.array(cfg["start_pos"])
                
            if "waypoints" in cfg:
                WAYPOINTS = [np.array(wp) for wp in cfg["waypoints"]]
                
            # If no waypoints, use 'target' as single waypoint
            if not WAYPOINTS and "target" in cfg:
                WAYPOINTS = [np.array(cfg["target"])]

            print(f"Loaded: Plan={PLANNER_MODE}, Ctrl={CTRL_MODE}, WPs={len(WAYPOINTS)}")
    except Exception as e:
        print(f"Config Load Error: {e}")

def main():
    print(f"Loading {XML_PATH}...")
    try:
        m = mujoco.MjModel.from_xml_path(XML_PATH)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Load Error: {e}")
        return

    # Helper: Update Markers
    def update_markers(curr_target_idx):
        # Target Marker
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
        if sid != -1 and SHOW_TARGET and curr_target_idx < len(WAYPOINTS):
            m.site_pos[sid] = WAYPOINTS[curr_target_idx]
            m.site_rgba[sid] = [1, 0, 0, 1]

    # 1. SETUP START STATE
    # Solve IK for Start Position
    # Use dedicated MjData for IK to avoid messing up simulation state
    d_ik = mujoco.MjData(m)
    ik_solver = manipulator_ik.IKSolver(m, d_ik)
    q_home = np.array([0, -1.57, 0, -1.57, 0, 0])
    
    # Teleport to Start
    print(f"Setting Start Pose: {START_POS}")
    d.qpos[:6] = q_home 
    mujoco.mj_forward(m, d)
    q_start = ik_solver.solve(START_POS, q_guess=q_home)
    
    if q_start is None:
        print("Warning: Could not solve IK for Start Position. Using Home.")
        q_start = q_home
        
    d.qpos[:6] = q_start
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    
    # 2. TASK PREPARATION
    # We support two modes: 
    # A. "Legacy/Simple": Just a list of WAYPOINTS.
    # B. "Pro": A TASK_QUEUE with mixed actions (Move, Gripper, Sleep).
    
    TASKS = []
    
    # Check if we have a robust task queue from config
    if os.path.exists("sim_config.json"):
        with open("sim_config.json", "r") as f:
            c = json.load(f)
            if "tasks" in c and c["tasks"]:
                TASKS = c["tasks"]
                print(f"Loaded {len(TASKS)} complex tasks.")
            elif WAYPOINTS:
                # Convert WPs to Move Tasks
                print("Converting Waypoints to Move Tasks.")
                for wp in WAYPOINTS:
                    TASKS.append({"action": "move", "target": wp.tolist()})

    if not TASKS and not WAYPOINTS:
         print("No Tasks or Waypoints. Using Target Marker as single Task.")
         TASKS.append({"action": "move", "target": START_POS.tolist()})

    # 3. INITIALIZE CONTROLLER (ARM ONLY)
    controller = None
    if CTRL_MODE == "lqr":
        # Check dimensionality
        if m.nu > 6:
             q_start_full = d.qpos[:m.nu].copy() # 8-dof
             controller = manipulator_controllers.LQRController(m, d, Q_POS, Q_VEL, R_VAL, q_start_full)
        else:
             controller = manipulator_controllers.LQRController(m, d, Q_POS, Q_VEL, R_VAL, q_start)
    elif CTRL_MODE == "pid":
        controller = manipulator_controllers.PIDController(m, d, kp=200, ki=5, kd=50)
    elif CTRL_MODE == "ctc":
        controller = manipulator_controllers.CTCController(m, d, kp=100, kd=20)
    elif CTRL_MODE == "mpc":
        controller = manipulator_controllers.MPPIController(m, d)
    elif CTRL_MODE == "impedance":
        controller = manipulator_controllers.ImpedanceController(m, d, site_name="attachment_site", kp=800, kd=150)
    elif CTRL_MODE == "osc":
        controller = manipulator_osc.OSCController(m, d, kp=150, null_kp=10)

    # 4. SIMULATION LOOP
    print("Launching Viewer (Task Mode)...")
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
        start_time = time.time()
        
        # Reset to Start
        d.qpos[:6] = q_start
        d.qpos[6:] = 0 # Gripper open/neutral
        # d.ctrl is usually 6 for URe, but if gripper is added it might be 8.
        # Check Actuator count
        nu = m.nu
        print(f"Actuators: {nu}")
        
        # Init Gripper State
        gripper_target = 0.0 # 0=Open, 1=Close (mapped later)
        if nu > 6:
             d.ctrl[6:] = 0
        
        mujoco.mj_forward(m, d)
        
        # Task State Machine
        task_idx = 0
        task_state = "INIT" # INIT, PLANNING, MOVING, ACTION, DONE
        
        current_path = []
        path_node_idx = 0
        
        # Logging
        log_t, log_q, log_u, log_x, log_err_cart = [], [], [], [], []
        
        full_path_q = [] # For plotting history
        
        while viewer.is_running():
            step_start = time.time()
            t = d.time
            curr_q = d.qpos[:6]
            curr_v = d.qvel[:6]
            
            # --- TASK MACHINE ---
            if task_idx < len(TASKS):
                task = TASKS[task_idx]
                action = task.get("action", "move")
                
                if task_state == "INIT":
                    print(f"DTO [{task_idx}]: {action} -> {task.get('target', '')}")
                    
                    if action == "move":
                        # Plan Path
                        target_pos = np.array(task["target"])
                        update_markers(0) # Logic to show marker?
                        
                        # IK
                        d_ik.qpos[:6] = curr_q
                        q_goal = ik_solver.solve(target_pos, q_guess=curr_q)
                        
                        if q_goal is None:
                            print("  IK Failed. Skipping Task.")
                            task_state = "DONE"
                        else:
                            # Plan
                            if PLANNER_MODE in ["rrt", "rrt_star", "prm"]:
                                if PLANNER_MODE == "rrt_star": pl = manipulator_planner.RRTStarPlanner(m, d)
                                elif PLANNER_MODE == "prm": pl = manipulator_planner.PRMPlanner(m, d)
                                else: pl = manipulator_planner.RRTPlanner(m, d)
                                
                                raw_path = pl.plan(curr_q, q_goal)
                                if raw_path:
                                     # Smoothing?
                                     # ... (omitted for brevity, assume raw for now or standard smooth)
                                     current_path = raw_path
                                else:
                                     print("  Plan Failed. Direct.")
                                     current_path = [curr_q, q_goal]
                            else:
                                # Direct
                                current_path = [curr_q, q_goal]
                            
                            path_node_idx = 0
                            full_path_q.extend(current_path)
                            task_state = "MOVING"
                            
                    elif action == "gripper":
                        target_val = float(task.get("value", 0)) # 0 or 1
                        gripper_target = target_val * 255 # If mapped? Or just 0.0-1.0
                        # Robotiq: 0=Open, 255=Closed usually. 
                        # Our simple gripper: 0=Open, 0.04=Close.
                        # Let's map 0->0, 1->0.04
                        if target_val > 0.5:
                             # CLOSE -> 0.0 (Inner Limit)
                             gripper_target = 0.0
                        else:
                             # OPEN -> 0.04 (Outer Limit)
                             gripper_target = 0.04
                             
                        task_state = "ACTION"
                        action_start_time = t
                        
                    elif action == "sleep":
                        task_state = "ACTION"
                        action_start_time = t
                        
                if task_state == "MOVING":
                    # Track Path
                    if path_node_idx < len(current_path):
                        target_q_6 = current_path[path_node_idx]
                        dist = np.linalg.norm(target_q_6 - curr_q)
                        if dist < 0.1: # Waypoint reached
                            path_node_idx += 1
                        
                        # Control
                        # Handle Dimension Mismatch (6 DoF Plan vs 8 DoF Model)
                        if nu > 6 and CTRL_MODE == "lqr":
                             # Pad Target and Current
                             full_target = np.zeros(nu)
                             full_target[:6] = target_q_6
                             # Keep gripper at current pos or 0? 0 is fine for LQR reference.
                             
                             full_curr_q = d.qpos[:nu]
                             full_curr_v = d.qvel[:nu]
                             
                             u_full = controller.update(full_curr_q, full_curr_v, target_q=full_target)
                             u = u_full[:6]
                        else:
                             u = controller.update(curr_q, curr_v, target_q=target_q_6)
                             
                        d.ctrl[:6] = u + d.qfrc_bias[:6]
                    else:
                        print("  Move Complete.")
                        task_state = "DONE"
                        
                elif task_state == "ACTION":
                    # Hold Position
                    if nu > 6 and CTRL_MODE == "lqr":
                          full_curr_q = d.qpos[:nu]
                          full_curr_v = d.qvel[:nu]
                          # Target is current pos (hold)
                          u_full = controller.update(full_curr_q, full_curr_v, target_q=full_curr_q)
                          u = u_full[:6]
                    else:
                          u = controller.update(curr_q, curr_v, target_q=curr_q)
                          
                    d.ctrl[:6] = u + d.qfrc_bias[:6]
                    
                    # Apply Gripper
                    if nu > 6:
                        # Simple P-control for gripper or direct force
                        # Our XML actuator is 'position', so just set ctrl
                        # Actuators 6 and 7 (right drive, left driver)
                        d.ctrl[6] = gripper_target
                        if nu > 7: d.ctrl[7] = gripper_target
                        
                    if action == "gripper":
                         if t - action_start_time > 1.0: # Wait 1s for grip
                             task_state = "DONE"
                    elif action == "sleep":
                         dur = float(task.get("value", 1.0))
                         if t - action_start_time > dur:
                             task_state = "DONE"

                elif task_state == "DONE":
                    task_idx += 1
                    task_state = "INIT"
                    
            else:
                 # ALL DONE - HOLD
                 u = controller.update(curr_q, curr_v, target_q=curr_q)
                 d.ctrl[:6] = u + d.qfrc_bias[:6]
                 if nu > 6: 
                     d.ctrl[6] = gripper_target
                     if nu > 7: d.ctrl[7] = gripper_target

            timestep = m.opt.timestep
            mujoco.mj_step(m, d)
            viewer.sync()
            
            # --- LOGGING ---
            # --- LOGGING ---
            if len(log_t) == 0 or (t - log_t[-1] >= 0.05): # Log every ~0.05s (20Hz)
                 log_t.append(t)
                 log_q.append(curr_q.copy())
                 log_u.append(d.ctrl[:6].copy())
                 
                 # Cartesian Pos
                 x_curr = d.site_xpos[ik_solver.site_id].copy()
                 log_x.append(x_curr)
                 
                 # Error
                 if task_state == "MOVING" and 'target_q_6' in locals():
                      # Joint error norm
                      err = np.linalg.norm(target_q_6 - curr_q)
                      log_err_cart.append(err)
                 else:
                      log_err_cart.append(0.0)
            
            # Rate limit
            time_until_next = timestep - (time.time()-step_start)
            if time_until_next > 0: time.sleep(time_until_next)
                
    # --- PLOTTING ---
    print("Simulation finished. Generating plots...")
    log_t = np.array(log_t)
    log_q = np.array(log_q)
    log_u = np.array(log_u)
    log_x = np.array(log_x)
    
    # FIG 1: Performance Stats
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Joint Positions
    for j in range(6):
        axs[0].plot(log_t, log_q[:, j], label=f'q{j+1}')
    axs[0].set_title('Joint Positions')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].legend(loc='upper right', fontsize='small', ncol=2)
    axs[0].grid(True)
    
    # 2. Control Inputs
    for j in range(6):
        axs[1].plot(log_t, log_u[:, j], label=f'u{j+1}')
    axs[1].set_title('Control Inputs (Torque)')
    axs[1].set_ylabel('Torque (Nm)')
    axs[1].grid(True)
    
    # 3. Tracking Error
    axs[2].plot(log_t, log_err_cart, color='r', label='Tracking Error')
    if CTRL_MODE in ["lqr", "pid", "ctc", "mpc"]:
        axs[2].set_title('Joint Tracking Error (Norm)')
        axs[2].set_ylabel('Error (rad)')
    else:
        axs[2].set_title('Cartesian Tracking Error')
        axs[2].set_ylabel('Error (m)')
        
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    
    # FIG 2: 3D Path Comparison
    fig2 = plt.figure(figsize=(10, 8))
    ax3d = fig2.add_subplot(111, projection='3d')
    
    # Plot Actual
    ax3d.plot(log_x[:,0], log_x[:,1], log_x[:,2], label='Followed Path', color='b', linewidth=2)
    
    # Calculate Planned Path for visualization
    planned_x = []
    
    # If we have a planned joint path (RRT/RRT* or even Direct joint interpolation)
    if len(full_path_q) > 0:
        # Compute FK for joint path
        d_fk = mujoco.MjData(m)
        for q_node in full_path_q:
            d_fk.qpos[:6] = q_node
            mujoco.mj_kinematics(m, d_fk)
            # Use 'attachment_site' or 'eef_site' or whatever ik_solver uses
            # We assume ik_solver.site_id is correct
            planned_x.append(d_fk.site_xpos[ik_solver.site_id].copy())
    else:
        # Cartesian WPs
        planned_x.append(START_POS)
        planned_x.extend(WAYPOINTS)
        
    planned_x = np.array(planned_x)
    if len(planned_x) > 0:
        ax3d.plot(planned_x[:,0], planned_x[:,1], planned_x[:,2], label='Planned Path', color='g', linestyle='--', marker='o')
        # Mark Start/End
        ax3d.scatter(planned_x[0,0], planned_x[0,1], planned_x[0,2], color='k', s=50, label='Start')
        ax3d.scatter(planned_x[-1,0], planned_x[-1,1], planned_x[-1,2], color='r', s=50, label='Goal')

    ax3d.set_title(f'3D Path: Planned ({PLANNER_MODE}) vs Actual')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
