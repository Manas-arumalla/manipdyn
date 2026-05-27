
import numpy as np
import mujoco
import manipulator_planner
import time

XML_PATH = "scene.xml"

def main():
    print("--- RRT PLANNER VERIFICATION ---")
    
    # Load Model
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    
    print(f"DEBUG: Model loaded. Actuators: {m.nu}")
    
    # Initialize Planner
    planner = manipulator_planner.RRTPlanner(m, d, step_size=0.15, max_iter=3000)
    
    # Define Start and Goal
    # Start: Home
    q_start = np.array([0, -1.57, 0, -1.57, 0, 0])
    
    # Goal: Use a known safe pose (e.g., "Up" pose)
    # [0, -1.57, -1.57, -1.57, 1.57, 0] ?
    # Let's try to just rotate the base to 90 degrees (1.57) with same arm config.
    q_goal = np.array([1.57, -1.57, 0, -1.57, 0, 0]) 
    
    # Verify Goal Safety Explicitly
    if planner.checker.is_collision(q_goal):
        print("WARNING: Hardcoded Goal is in collision! Searching for random safe goal...")
        for _ in range(100):
            q_cand = np.random.uniform(-1, 1, 6)
            if not planner.checker.is_collision(q_cand):
                q_goal = q_cand
                print(f"Found safe random goal: {q_goal}")
                break
    
    print(f"DEBUG: Start Conf: {q_start}")
    print(f"DEBUG: Goal Conf:  {q_goal}")
    
    start_time = time.time()
    path = planner.plan(q_start, q_goal)
    elap = time.time() - start_time
    
    if path:
        print(f"SUCCESS: Path found in {elap:.3f}s")
        print(f"Path Length: {len(path)} nodes")
        
        # Verify Safety
        checker = planner.checker
        safe = True
        for i, q in enumerate(path):
            if checker.is_collision(q):
                print(f"ERROR: Waypoint {i} is in COLLISION!")
                safe = False
        
        if safe:
            print("VERIFICATION PASS: All waypoints valid.")
        else:
            print("VERIFICATION FAIL: Path contains collisions.")
            
        # Optional: Print Path
        # for p in path: print(np.round(p, 3))
    else:
        print("FAILURE: No path found.")

if __name__ == "__main__":
    main()
