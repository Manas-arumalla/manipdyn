# 6-DOF Robotic Arm Trajectory Planning Simulator

A Python-based robot arm simulator demonstrating forward/inverse kinematics, 3 trajectory planners, PID control, and 3D visualization with multi-waypoint support.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)

## Features

- **Forward/Inverse Kinematics** using Denavit-Hartenberg (DH) convention
- **3 Trajectory Planners**:
  - Cubic Polynomial (smooth position & velocity)
  - Quintic Polynomial (smooth position, velocity & acceleration)
  - Trapezoidal Velocity Profile (constant acceleration phases)
- **Multi-Waypoint Trajectories** - Plan through 4+ waypoints
- **PID Control** with anti-windup for trajectory tracking
- **Closed-Loop Simulation** with simplified dynamics
- **3D Visualization** comparing desired vs actual end-effector paths

## Project Structure

```
Manipulator/
├── code/
│   ├── robot.py          # 6-DOF robot DH parameters & joint limits
│   ├── kinematics.py     # FK/IK solver with Jacobian
│   ├── trajectory.py     # 3 trajectory generators + multi-waypoint
│   ├── control.py        # PID controller
│   ├── simulation.py     # Closed-loop simulation
│   └── main.py           # Demo with visualization
├── docs/
│   └── plots/            # Generated comparison plots
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Manas-arumalla/6DOF-Manipulator-trajectory-planning.git
cd 6DOF-Manipulator-trajectory-planning
pip install -r requirements.txt
```

## Usage

```bash
cd code
python main.py
```

This will:
1. Generate trajectories through 4 waypoints using all 3 planners
2. Simulate closed-loop PID tracking
3. Display 3D trajectory comparison (3 separate windows)
4. Save plots to `docs/plots/`

## Trajectory Planners

| Planner | Description | Smoothness |
|---------|-------------|------------|
| **Cubic** | 3rd order polynomial | Position, Velocity |
| **Quintic** | 5th order polynomial | Position, Velocity, Acceleration |
| **Trapezoidal** | Constant accel/cruise/decel | Velocity (discontinuous accel) |

## Output Plots

| Plot | Description |
|------|-------------|
| `3d_cubic.png` | Cubic trajectory 3D comparison |
| `3d_quintic.png` | Quintic trajectory 3D comparison |
| `3d_trapezoidal.png` | Trapezoidal trajectory 3D comparison |
| `joint_tracking.png` | Joint position tracking |
| `velocity_profiles.png` | Velocity profile comparison |
| `tracking_error.png` | Error over time |
| `all_joints.png` | All 6 joints tracking |

## Robot Model

Based on UR5 industrial robot:
- 6 revolute joints (anthropomorphic configuration)
- Standard DH convention
- Realistic joint limits and torque constraints

## Dependencies

- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4

## License

MIT License
