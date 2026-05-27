
import mujoco
import numpy as np
import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manipulator_controllers

def test_impedance():
    # Load Model
    xml_path = "scene_dynamic.xml"
    if not os.path.exists(xml_path):
        xml_path = "scene_base.xml"
        
    print(f"Loading {xml_path}...")
    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Init Controller
    print("Initializing ImpedanceController...")
    try:
        # Use same params as in mujoco_sim.py
        ctrl = manipulator_controllers.ImpedanceController(m, d, site_name="attachment_site", kp=800, kd=80)
    except Exception as e:
        print(f"Controller Init Failed: {e}")
        return

    # Set Robot State (Non-zero)
    d.qpos[:6] = [-1.57, -1.57, 1.57, -1.57, -1.57, 0] # Home-ish
    d.qvel[:6] = 0.0
    mujoco.mj_forward(m, d)
    
    curr_x = d.site_xpos[ctrl.site_id]
    print(f"Current EE Pos: {curr_x}")
    
    # Set Target Far Away
    target_x = curr_x + np.array([0.1, 0.1, 0.1])
    print(f"Target EE Pos: {target_x}")
    
    # Compute Control
    tau = ctrl.update(d.qpos[:6], d.qvel[:6], target_x)
    
    print("-" * 30)
    print("DIAGNOSTICS:")
    print(f"Calculate Tau: {tau}")
    print(f"Tau Norm: {np.linalg.norm(tau)}")
    
    if np.linalg.norm(tau) < 0.01:
        print("FAIL: Torque is effectively zero.")
        print("Debugging internals...")
        # debug jacobian
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, ctrl.site_id)
        print(f"J_pos (Sample):\n{jacp[:, :6]}")
        print(f"J Norm: {np.linalg.norm(jacp)}")
        
        err = target_x - curr_x
        F = 800 * err
        print(f"Force F: {F}")
        print(f"Projected Tau (J.T * F): {jacp.T @ F}")
    else:
        print("PASS: Controller produces torque.")

if __name__ == "__main__":
    test_impedance()
