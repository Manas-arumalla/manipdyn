
import mujoco
import numpy as np
import os
import sys
import time

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manipulator_controllers

def test_mpc_perf():
    # Load Model
    xml_path = "scene_dynamic.xml"
    if not os.path.exists(xml_path):
        xml_path = "scene_base.xml"
        
    print(f"Loading {xml_path}...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    print(f"Benchmarking MPPI with NEW DEFAULTS...")
    ctrl = manipulator_controllers.MPPIController(m, d)

    # Set Start
    d.qpos[:6] = [-1.57, -1.57, 1.57, -1.57, -1.57, 0] 
    mujoco.mj_forward(m, d)
    
    target_q = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 0])
    
    times = []
    errors = []
    
    # Run 200 steps
    print("Running 200 control steps...")
    for i in range(200):
        t0 = time.perf_counter()
        
        u = ctrl.update(d.qpos[:6], d.qvel[:6], target_q)
        
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000 # ms
        times.append(dt)
        
        # Apply
        d.ctrl[:6] = u + d.qfrc_bias[:6]
        mujoco.mj_step(m, d)
        
        err = np.linalg.norm(target_q - d.qpos[:6])
        errors.append(err)
        
        if i % 20 == 0:
            print(f"Step {i}: Time={dt:.2f}ms | Err={err:.3f} | U_norm={np.linalg.norm(u):.1f}")

    # Oscillation Detection
    # Count how many times error slope flips sign (local extrema)
    flips = 0
    slopes = np.diff(errors)
    for i in range(1, len(slopes)):
        if slopes[i] * slopes[i-1] < 0:
            flips += 1
            
    avg_time = np.mean(times)
    print("-" * 30)
    print(f"RESULTS:")
    print(f"Avg Compute Time: {avg_time:.2f} ms")
    print(f"Final Error: {errors[-1]:.3f}")
    print(f"Direction Flips (Oscillations): {flips}")
    
    if flips > 5:
        print("FAIL: Significant Oscillation detected.")
    else:
        print("PASS: Motion is stable.")
    
    if errors[-1] > 0.1: # 10cm/rad
         print("WARNING: Did not settle to goal.")

if __name__ == "__main__":
    test_mpc_perf()
