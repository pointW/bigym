"""Test script to analyze joint angle changes when RBY1 base moves via mocap.

This script:
1. Loads RBY1 model with mocap body and weld constraint (pre-configured in XML)
2. Commands all joints to stay at 0
3. Moves the base by different distances (0.5, 1, 2, 5 cm) via mocap
4. Plots joint angles vs time to see coupling effects
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Set up paths
BASE_DIR = Path(__file__).parent.parent
RBY1_WITH_MOCAP_XML = BASE_DIR / "bigym/envs/xmls/rby1/model_act_consolidated_with_mocap.xml"

def load_model_with_mocap():
    """Load RBY1 model with mocap control (pre-configured in XML)."""
    
    model = mujoco.MjModel.from_xml_path(str(RBY1_WITH_MOCAP_XML))
    data = mujoco.MjData(model)
    
    # Verify mocap body exists
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_target")
    if mocap_id >= 0:
        mocap_idx = model.body_mocapid[mocap_id]
        if mocap_idx >= 0:
            print("✓ Mocap body 'base_target' found in model")
            print("✓ Weld constraint between base_target and rby1/base active")
        else:
            print("ERROR: base_target exists but is not a mocap body!")
            raise ValueError("base_target is not configured as mocap")
    else:
        print("ERROR: base_target mocap body not found in model!")
        raise ValueError("base_target not found")
    
    return model, data

def get_joint_info(model):
    """Get information about all controllable joints (excluding grippers and head)."""
    joint_info = []
    
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        
        # Skip gripper and head actuators
        if actuator_name and ('gripper' in actuator_name.lower() or 
                             'finger' in actuator_name.lower() or
                             'head' in actuator_name.lower()):
            continue
            
        # Get the joint this actuator controls
        joint_id = model.actuator_trnid[i, 0]
        if joint_id >= 0:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            
            # Skip base, wheel, and head joints
            if joint_name and not any(skip in joint_name for skip in ['world_j', 'wheel', 'head']):
                joint_info.append({
                    'actuator_id': i,
                    'joint_id': joint_id,
                    'actuator_name': actuator_name,
                    'joint_name': joint_name,
                    'qpos_idx': model.jnt_qposadr[joint_id],
                    'qvel_idx': model.jnt_dofadr[joint_id],
                })
    
    # Sort joints by type and number
    torso_joints = sorted([j for j in joint_info if 'torso' in j['joint_name']], 
                         key=lambda x: int(x['joint_name'].split('_')[-1]))
    left_arm_joints = sorted([j for j in joint_info if 'left_arm' in j['joint_name']],
                           key=lambda x: int(x['joint_name'].split('_')[-1]))
    right_arm_joints = sorted([j for j in joint_info if 'right_arm' in j['joint_name']],
                            key=lambda x: int(x['joint_name'].split('_')[-1]))
    
    return torso_joints + left_arm_joints + right_arm_joints

def run_base_movement_test(model, data, joint_info, base_displacement_cm, 
                          movement_time=0.5, total_time=2.0, axis='x'):
    """Run test with base movement via mocap while commanding joints to zero.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_info: List of joint information
        base_displacement_cm: Base displacement in cm
        movement_time: Time to complete base movement (seconds)
        total_time: Total simulation time (seconds)
        axis: Which axis to move along ('x', 'y', or 'xy')
    
    Returns:
        Dictionary with recorded data
    """
    base_displacement_m = base_displacement_cm / 100.0  # Convert to meters
    timestep = model.opt.timestep
    movement_steps = int(movement_time / timestep)
    total_steps = int(total_time / timestep)
    
    # Initialize data storage
    results = {
        'time': [],
        'base_pos': [],
        'joint_positions': {info['joint_name']: [] for info in joint_info},
        'joint_velocities': {info['joint_name']: [] for info in joint_info},
        'joint_forces': {info['joint_name']: [] for info in joint_info},
        'displacement': base_displacement_cm,
        'axis': axis
    }
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Get mocap body ID and index
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_target")
    if mocap_id < 0:
        raise ValueError("Mocap body 'base_target' not found")
    
    mocap_idx = model.body_mocapid[mocap_id]
    if mocap_idx < 0:
        raise ValueError("Body 'base_target' is not a mocap body")
    
    print(f"\nTesting base displacement: {base_displacement_cm}cm along {axis} axis")
    print(f"  Movement duration: {movement_time}s")
    print(f"  Total simulation: {total_time}s")
    print(f"  Using mocap control via base_target")
    
    # Run simulation
    for step in range(total_steps):
        time = step * timestep
        
        # Calculate base target position
        if step < movement_steps:
            # Smooth movement using cosine profile
            progress = (1 - np.cos(np.pi * step / movement_steps)) / 2
            if axis == 'x':
                target_x = base_displacement_m * progress
                target_y = 0
            elif axis == 'y':
                target_x = 0
                target_y = base_displacement_m * progress
            elif axis == 'xy':
                target_x = base_displacement_m * progress * np.cos(np.pi/4)
                target_y = base_displacement_m * progress * np.sin(np.pi/4)
        else:
            # Hold final position
            if axis == 'x':
                target_x = base_displacement_m
                target_y = 0
            elif axis == 'y':
                target_x = 0
                target_y = base_displacement_m
            elif axis == 'xy':
                target_x = base_displacement_m * np.cos(np.pi/4)
                target_y = base_displacement_m * np.sin(np.pi/4)
        
        # Set mocap position - this is the key part!
        data.mocap_pos[mocap_idx][0] = target_x
        data.mocap_pos[mocap_idx][1] = target_y
        data.mocap_pos[mocap_idx][2] = 0.0  # Keep at ground level
        
        # Command all joints to zero position
        for info in joint_info:
            data.ctrl[info['actuator_id']] = 0.0
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record data
        results['time'].append(time)
        results['base_pos'].append([data.qpos[0], data.qpos[1]])  # Actual base x,y position
        
        for info in joint_info:
            results['joint_positions'][info['joint_name']].append(
                np.rad2deg(data.qpos[info['qpos_idx']])
            )
            results['joint_velocities'][info['joint_name']].append(
                np.rad2deg(data.qvel[info['qvel_idx']])
            )
            results['joint_forces'][info['joint_name']].append(
                data.qfrc_actuator[info['qvel_idx']]
            )
    
    # Convert to numpy arrays
    results['time'] = np.array(results['time'])
    results['base_pos'] = np.array(results['base_pos'])
    for joint_name in results['joint_positions']:
        results['joint_positions'][joint_name] = np.array(results['joint_positions'][joint_name])
        results['joint_velocities'][joint_name] = np.array(results['joint_velocities'][joint_name])
        results['joint_forces'][joint_name] = np.array(results['joint_forces'][joint_name])
    
    return results

def plot_comparison(all_results, joint_info):
    """Plot joint angles for different base displacements.
    
    Args:
        all_results: List of result dictionaries for different displacements
        joint_info: List of joint information
    """
    # Organize joints by type
    torso_joints = [j['joint_name'] for j in joint_info if 'torso' in j['joint_name']]
    left_arm_joints = [j['joint_name'] for j in joint_info if 'left_arm' in j['joint_name']]
    right_arm_joints = [j['joint_name'] for j in joint_info if 'right_arm' in j['joint_name']]
    
    # Create figure with subplots for each joint group
    fig, axes = plt.subplots(3, 7, figsize=(21, 9))
    fig.suptitle('Joint Angles vs Time for Different Base Displacements (Mocap + Weld)', fontsize=14)
    
    # Color map for different displacements
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_results)))
    
    def plot_joint_group(joint_names, row, axes_row):
        """Plot a group of joints in a row."""
        for col, joint_name in enumerate(joint_names):
            if col >= 7:
                break
            
            ax = axes_row[col]
            
            # Plot for each displacement
            for result, color in zip(all_results, colors):
                displacement = result['displacement']
                axis = result['axis']
                label = f"{displacement}cm ({axis})"
                
                ax.plot(result['time'], result['joint_positions'][joint_name],
                       color=color, linewidth=1.5, label=label)
            
            ax.set_title(joint_name, fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Angle (deg)', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=7)
            
            # Add legend only to first plot
            if col == 0 and row == 0:
                ax.legend(loc='best', fontsize=6)
            
            # Add zero reference line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Plot each joint group
    plot_joint_group(torso_joints, 0, axes[0])
    plot_joint_group(left_arm_joints, 1, axes[1])
    plot_joint_group(right_arm_joints, 2, axes[2])
    
    # Hide unused subplots
    for row, joint_list in enumerate([torso_joints, left_arm_joints, right_arm_joints]):
        for col in range(len(joint_list), 7):
            axes[row, col].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.3, 0.5, 'Torso', transform=axes[0, 0].transAxes, 
                    fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.3, 0.5, 'Left Arm', transform=axes[1, 0].transAxes, 
                    fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[2, 0].text(-0.3, 0.5, 'Right Arm', transform=axes[2, 0].transAxes, 
                    fontsize=12, fontweight='bold', va='center', rotation=90)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'rby1_base_movement_joint_effects.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {filename}")
    
    # Create a second plot showing maximum deviations
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    fig2.suptitle('Maximum Joint Angle Deviations from Zero (Mocap + Weld)', fontsize=14)
    
    # Calculate max deviations for each joint and displacement
    joint_names = torso_joints + left_arm_joints + right_arm_joints
    x_positions = np.arange(len(joint_names))
    bar_width = 0.8 / len(all_results)
    
    for i, (result, color) in enumerate(zip(all_results, colors)):
        max_deviations = []
        for joint_name in joint_names:
            max_dev = np.max(np.abs(result['joint_positions'][joint_name]))
            max_deviations.append(max_dev)
        
        displacement = result['displacement']
        axis = result['axis']
        label = f"{displacement}cm ({axis})"
        
        x_offset = (i - len(all_results)/2 + 0.5) * bar_width
        ax2.bar(x_positions + x_offset, max_deviations, bar_width,
               color=color, label=label, alpha=0.8)
    
    ax2.set_xlabel('Joint', fontsize=10)
    ax2.set_ylabel('Max Deviation (deg)', fontsize=10)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save second figure
    filename2 = 'rby1_base_movement_max_deviations.png'
    fig2.savefig(filename2, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filename2}")
    
    # Create base position plot to verify mocap control
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
    fig3.suptitle('Base Position Tracking via Mocap Control', fontsize=14)
    
    for result, color in zip(all_results, colors):
        displacement = result['displacement']
        axis_dir = result['axis']
        label = f"{displacement}cm ({axis_dir})"
        
        # Plot X position
        ax3.plot(result['time'], result['base_pos'][:, 0] * 100, 
                color=color, linewidth=1.5, label=label)
        ax3.set_ylabel('X Position (cm)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=8)
        
        # Plot Y position
        ax4.plot(result['time'], result['base_pos'][:, 1] * 100, 
                color=color, linewidth=1.5, label=label)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Y Position (cm)', fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save base position plot
    filename3 = 'rby1_base_position_tracking.png'
    fig3.savefig(filename3, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filename3}")
    
    plt.show()

def main():
    """Main function to run base movement tests."""
    parser = argparse.ArgumentParser(description='Test RBY1 joint coupling with base movement via mocap')
    parser.add_argument('--displacements', type=float, nargs='+', 
                       default=[0.5, 1.0, 2.0, 5.0],
                       help='Base displacements to test in cm (default: 0.5 1.0 2.0 5.0)')
    parser.add_argument('--axis', type=str, default='x',
                       choices=['x', 'y', 'xy'],
                       help='Axis to move along (default: x)')
    parser.add_argument('--movement-time', type=float, default=0.5,
                       help='Time to complete base movement in seconds (default: 0.5)')
    parser.add_argument('--total-time', type=float, default=2.0,
                       help='Total simulation time in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    print("Loading RBY1 model with mocap control and weld constraint...")
    print("Using pre-configured XML: model_act_consolidated_with_mocap.xml")
    model, data = load_model_with_mocap()
    print(f"Model loaded successfully!")
    print(f"Timestep: {model.opt.timestep}s")
    
    # Get joint information
    joint_info = get_joint_info(model)
    print(f"\nFound {len(joint_info)} controllable joints")
    
    # Run tests for different displacements
    all_results = []
    for displacement in args.displacements:
        print(f"\n{'='*50}")
        result = run_base_movement_test(
            model, data, joint_info,
            base_displacement_cm=displacement,
            movement_time=args.movement_time,
            total_time=args.total_time,
            axis=args.axis
        )
        all_results.append(result)
    
    # Plot comparison
    print(f"\n{'='*50}")
    print("Plotting results...")
    plot_comparison(all_results, joint_info)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("Summary of Maximum Joint Deviations:")
    print("-" * 50)
    
    for result in all_results:
        displacement = result['displacement']
        axis = result['axis']
        print(f"\nBase displacement: {displacement}cm ({axis} axis)")
        
        max_overall = 0
        max_joint = ""
        
        for joint_name in result['joint_positions']:
            max_dev = np.max(np.abs(result['joint_positions'][joint_name]))
            if max_dev > max_overall:
                max_overall = max_dev
                max_joint = joint_name
        
        print(f"  Most affected joint: {max_joint} ({max_overall:.3f}°)")
        
        # Show top 5 affected joints
        deviations = []
        for joint_name in result['joint_positions']:
            max_dev = np.max(np.abs(result['joint_positions'][joint_name]))
            deviations.append((joint_name, max_dev))
        
        deviations.sort(key=lambda x: x[1], reverse=True)
        print("  Top 5 affected joints:")
        for joint_name, dev in deviations[:5]:
            print(f"    {joint_name:<15}: {dev:6.3f}°")
    
    print("\nTest complete!")
    print("\nIMPORTANT: This test shows joint deviations when the base moves via mocap.")
    print("If joints deviate significantly from zero, consider:")
    print("  1. Increasing joint stiffness (kp) in the XML")
    print("  2. Increasing joint damping (kv) in the XML")
    print("  3. Adjusting the weld constraint parameters")

if __name__ == "__main__":
    main()