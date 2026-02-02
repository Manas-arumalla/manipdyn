"""
6-DOF Robotic Arm Trajectory Planning Simulator

Main script demonstrating:
1. 3 trajectory planners (Cubic, Quintic, Trapezoidal)
2. PID control for tracking
3. Multi-waypoint trajectories (4 points)
4. 3D visualization with desired vs actual comparison

Run: python main.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from robot import create_robot
from kinematics import Kinematics
from trajectory import TrajectoryType
from simulation import Simulation


# =============================================================================
# Configuration
# =============================================================================

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")

COLORS = {
    TrajectoryType.CUBIC: '#2ecc71',
    TrajectoryType.QUINTIC: '#3498db',
    TrajectoryType.TRAPEZOIDAL: '#e74c3c'
}

NAMES = {
    TrajectoryType.CUBIC: 'Cubic Polynomial',
    TrajectoryType.QUINTIC: 'Quintic Polynomial',
    TrajectoryType.TRAPEZOIDAL: 'Trapezoidal Velocity'
}


# =============================================================================
# Plotting Functions
# =============================================================================

def save_plot(fig, filename):
    """Save figure to docs/plots directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")


def plot_3d_trajectories(results, waypoint_positions):
    """Plot 3D trajectory for each planner in separate windows."""
    for traj_type, result in results.items():
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Desired path
        ax.plot(
            result.ee_desired[:, 0],
            result.ee_desired[:, 1],
            result.ee_desired[:, 2],
            'k--', linewidth=2.5, label='Desired', alpha=0.8
        )
        
        # Actual path
        ax.plot(
            result.ee_actual[:, 0],
            result.ee_actual[:, 1],
            result.ee_actual[:, 2],
            color=COLORS[traj_type], linewidth=2.5, label='Actual'
        )
        
        # Waypoint markers
        waypoint_colors = ['green', 'orange', 'purple', 'red']
        waypoint_labels = ['Start (P1)', 'P2', 'P3', 'End (P4)']
        markers = ['o', 's', 'D', '*']
        sizes = [150, 120, 120, 180]
        
        for i, (pos, color, label, marker, size) in enumerate(
            zip(waypoint_positions, waypoint_colors, waypoint_labels, markers, sizes)
        ):
            ax.scatter(*pos, c=color, s=size, marker=marker, label=label, zorder=5, edgecolors='black')
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title(f'{NAMES[traj_type]}\nMean Error: {np.mean(result.tracking_error):.4f} rad', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        
        fig.canvas.manager.set_window_title(f'{NAMES[traj_type]} - 3D Trajectory')
        save_plot(fig, f'3d_{traj_type.value}.png')


def plot_joint_tracking(results, joint_idx=0):
    """Plot joint tracking comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    for idx, (traj_type, result) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(result.time, result.q_desired[:, joint_idx], 'k--', lw=2, label='Desired')
        ax.plot(result.time, result.q_actual[:, joint_idx], color=COLORS[traj_type], lw=2, label='Actual')
        ax.set_ylabel(f'Joint {joint_idx+1} (rad)')
        ax.set_title(f'{NAMES[traj_type]}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Joint {joint_idx+1} Tracking Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'joint_tracking.png')


def plot_velocity_profiles(results, joint_idx=1):
    """Plot velocity profiles overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for traj_type, result in results.items():
        ax.plot(result.time, result.dq_desired[:, joint_idx],
                color=COLORS[traj_type], lw=2, label=NAMES[traj_type])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Joint {joint_idx+1} Velocity (rad/s)')
    ax.set_title('Velocity Profile Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'velocity_profiles.png')


def plot_tracking_error(results):
    """Plot tracking error over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for traj_type, result in results.items():
        mean = np.mean(result.tracking_error)
        ax.plot(result.time, result.tracking_error,
                color=COLORS[traj_type], lw=2, 
                label=f'{NAMES[traj_type]} (mean={mean:.4f})')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (rad)')
    ax.set_title('Tracking Error Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'tracking_error.png')


def plot_all_joints(results):
    """Plot all 6 joints for best trajectory."""
    best = min(results.keys(), key=lambda t: np.mean(results[t].tracking_error))
    result = results[best]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for j in range(6):
        ax = axes[j]
        ax.plot(result.time, result.q_desired[:, j], 'k--', lw=2, label='Desired')
        ax.plot(result.time, result.q_actual[:, j], 'b-', lw=2, label='Actual')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        ax.set_title(f'Joint {j+1}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'All Joints - {NAMES[best]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'all_joints.png')


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("6-DOF Robotic Arm Trajectory Planning Simulator")
    print("=" * 60)
    
    # Setup
    robot = create_robot()
    sim = Simulation(robot, dt=0.01)
    kin = Kinematics(robot)
    
    print(f"\nRobot: {robot.n_dof}-DOF Articulated Arm")
    
    # Define 4 waypoints for multi-point trajectory
    waypoints = [
        np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0]),      # Start (P1)
        np.array([0.4, -0.8, 0.8, -0.2, 0.7, 0.1]),     # Waypoint 2 (P2)
        np.array([0.6, -0.6, 1.0, -0.4, 0.9, 0.2]),     # Waypoint 3 (P3)
        np.array([0.8, -1.0, 1.2, -0.5, 1.0, 0.3]),     # End (P4)
    ]
    
    print(f"\nTrajectory with {len(waypoints)} waypoints:")
    waypoint_positions = []
    for i, q in enumerate(waypoints):
        ee_pos = kin.get_position(q)
        waypoint_positions.append(ee_pos)
        print(f"  P{i+1}: joints={np.round(q, 2)}  ->  EE={np.round(ee_pos, 3)} m")
    
    # Run simulations with 4 waypoints
    print("\n" + "-" * 60)
    print("Running trajectory simulations (4 waypoints)...")
    print("-" * 60)
    
    results = sim.run_waypoint_comparison(waypoints, T_per_segment=1.5, N_per_segment=150)
    
    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for traj_type, result in results.items():
        mean_err = np.mean(result.tracking_error)
        max_err = np.max(result.tracking_error)
        ee_err = np.linalg.norm(result.ee_desired[-1] - result.ee_actual[-1])
        
        print(f"\n{NAMES[traj_type].upper()}:")
        print(f"  Mean error: {mean_err:.4f} rad ({np.degrees(mean_err):.2f}°)")
        print(f"  Max error:  {max_err:.4f} rad ({np.degrees(max_err):.2f}°)")
        print(f"  EE error:   {ee_err*1000:.2f} mm")
    
    # Generate plots
    print("\n" + "-" * 60)
    print("Generating plots...")
    print("-" * 60)
    
    plot_3d_trajectories(results, waypoint_positions)
    plot_joint_tracking(results)
    plot_velocity_profiles(results)
    plot_tracking_error(results)
    plot_all_joints(results)
    
    print("\n" + "=" * 60)
    print(f"Done! Plots saved to: docs/plots/")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    main()
