
import numpy as np
import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manipulator_planner
import mujoco

def test_planner():
    print("Testing Planner Module...")
    
    # 1. Load Model (Mocking data for collision checker)
    xml_path = "scene_base.xml"
    if os.path.exists("scene_dynamic.xml"):
        xml_path = "scene_dynamic.xml"
        
    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        print(f"Loaded {xml_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Test RRT* Init
    try:
        planner = manipulator_planner.RRTStarPlanner(m, d, max_iter=500)
        print("RRTStarPlanner Initialized.")
    except Exception as e:
        print(f"FAIL: RRTStarPlanner Init Error: {e}")
        return

    # 3. Test Plan (Short distance to ensure success)
    q_start = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0])
    q_goal = q_start + 0.1 # Small move
    
    print("Running RRT* Plan...")
    path = planner.plan(q_start, q_goal)
    
    if path is None:
        print("WARNING: RRT* failed to find path (might be valid if collision).")
    else:
        print(f"PASS: RRT* found path with {len(path)} nodes.")
        
    # NEW: Test PRM
    print("Testing PRM Planner...")
    try:
        prm = manipulator_planner.PRMPlanner(m, d, n_samples=50, k_neighbors=3) # Small for speed
        prm.build_roadmap()
        path_prm = prm.plan(q_start, q_goal)
        if path_prm:
             print(f"PASS: PRM found path with {len(path_prm)} nodes.")
        else:
             print("WARNING: PRM failed (maybe low samples).")
    except Exception as e:
        print(f"FAIL: PRM Error: {e}")
        
    # 4. Test Smoothing
    print("Testing B-Spline Smoothing...")
    try:
        # Create a jagged path
        jagged = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 0.5, 0, 0, 0, 0],
            [2, 1.5, 0, 0, 0, 0],
            [3, 1.0, 0, 0, 0, 0]
        ])
        smoothed = manipulator_planner.smooth_path_bspline(jagged)
        
        print(f"Jagged Points: {len(jagged)}")
        print(f"Smoothed Points: {len(smoothed)}")
        
        if len(smoothed) > len(jagged):
            print("PASS: Smoothing increased resolution.")
        else:
            print("WARNING: Smoothing did not increase resolution (Check scipy).")
            
    except Exception as e:
        print(f"FAIL: Smoothing Error: {e}")
        print("Did you install scipy? (pip install scipy)")

if __name__ == "__main__":
    test_planner()
