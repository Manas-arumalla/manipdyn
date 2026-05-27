
import time
import numpy as np
import mujoco
import mujoco.viewer
import manipulator_osc
import json
import os

# Configuration
XML_PATH = "scene.xml"
TARGET_POS = np.array([0.4, -0.4, 0.4])

# Load Config
if os.path.exists("sim_config.json"):
    try:
        with open("sim_config.json", "r") as f:
            cfg = json.load(f)
            tgt = cfg.get("target", None)
            if tgt: TARGET_POS = np.array(tgt)
            if "xml_path" in cfg: XML_PATH = cfg["xml_path"]
            SHOW_TARGET = cfg.get("show_target", True)
            print(f"Loaded Config: Target={TARGET_POS}, XML={XML_PATH}")
    except:
        pass
        SHOW_TARGET = True

def main():
    # ...
    # Update marker
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
    if site_id != -1:
        m.site_pos[site_id] = TARGET_POS
        if not SHOW_TARGET:
             m.site_rgba[site_id] = [0, 0, 0, 0]

def main():
    # Load model
    print(f"Loading model from {XML_PATH}...")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # Update marker
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
    if site_id != -1:
        m.site_pos[site_id] = TARGET_POS

    # Initialize OSC
    osc = manipulator_osc.OSCController(m, d)
    
    # Simulation loop
    print("Starting simulation with OSC...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        
        # Reset
        d.qpos[:6] = np.array([0, -1.57, 1.57, -1.57, -1.57, 0]) # Good start pose
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        
        while viewer.is_running():
            step_start = time.time()
            
            # Compute OSC Torque
            # Target is static Cartesian Position
            tau = osc.update(TARGET_POS)
            
            # Apply
            d.ctrl[:6] = tau
            
            mujoco.mj_step(m, d)
            viewer.sync()
            
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
