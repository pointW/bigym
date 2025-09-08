"""Test script for tuning individual RBY1 joint gains.

This script sends a step command to a single joint after 0.1 seconds of baseline recording,
allowing for better analysis of the joint response from steady state.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Set up paths
BASE_DIR = Path(__file__).parent.parent
RBY1_XML = BASE_DIR / "bigym/envs/xmls/rby1/model_act_consolidated.xml"

def load_model():
    """Load the RBY1 model."""
    model = mujoco.MjModel.from_xml_path(str(RBY1_XML))
    data = mujoco.MjData(model)
    return model, data

def get_joint_info(model, target_joint_name=None):
    """Get information about controllable joints.
    
    Args:
        model: MuJoCo model
        target_joint_name: If specified, only return info for this joint
        
    Returns:
        List of joint information dictionaries
    """
    joint_info = []
    
    # Get all actuators (excluding grippers and head)
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
                # If target specified, only include that joint
                if target_joint_name is None or joint_name == target_joint_name:
                    joint_info.append({
                        'actuator_id': i,
                        'joint_id': joint_id,
                        'actuator_name': actuator_name,
                        'joint_name': joint_name,
                        'qpos_idx': model.jnt_qposadr[joint_id],
                        'qvel_idx': model.jnt_dofadr[joint_id],   # <-- add this
                    })
    
    return joint_info

def run_single_joint_test(model, data, joint_info, target_angle_deg=20, 
                         delay_seconds=0.1, total_seconds=2.0):
    """Run step response test for a single joint with delayed command.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_info: Joint information dictionary (single joint)
        target_angle_deg: Target angle in degrees
        delay_seconds: Time to wait before sending command
        total_seconds: Total simulation time
    
    Returns:
        Dictionary with recorded data
    """
    target_angle_rad = np.deg2rad(target_angle_deg)
    timestep = model.opt.timestep
    delay_steps = int(delay_seconds / timestep)
    total_steps = int(total_seconds / timestep)
    
    # Initialize data storage
    result = {
        'time': [],
        'position': [],
        'velocity': [],
        'force': [],
        'command': [],
        'target': target_angle_rad,
        'actuator_name': joint_info['actuator_name'],
        'joint_name': joint_info['joint_name'],
        'delay_time': delay_seconds
    }
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Get initial position
    initial_position = data.qpos[joint_info['qpos_idx']]
    
    print(f"Testing {joint_info['joint_name']}:")
    print(f"  Initial position: {np.rad2deg(initial_position):.2f}°")
    print(f"  Target: {target_angle_deg}° relative ({np.rad2deg(initial_position + target_angle_rad):.2f}° absolute)")
    print(f"  Command delay: {delay_seconds}s ({delay_steps} steps)")
    print(f"  Total duration: {total_seconds}s ({total_steps} steps)")
    
    # Run simulation
    for step in range(total_steps):
        # Set control command after delay
        if step < delay_steps:
            # Before delay: maintain initial position
            command = initial_position
        else:
            # After delay: apply step command
            command = initial_position + target_angle_rad
        
        # Apply control
        data.ctrl[joint_info['actuator_id']] = command
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record data
        time = step * timestep
        result['time'].append(time)
        result['position'].append(
            data.qpos[joint_info['qpos_idx']] - initial_position
        )
        # result['velocity'].append(data.qvel[joint_info['qpos_idx']])
        result['velocity'].append(data.qvel[joint_info['qvel_idx']])  # <-- use qvel_idx
        # result['force'].append(data.actuator_force[joint_info['actuator_id']])
        tau = data.qfrc_actuator[joint_info['qvel_idx']]
        result['force'].append(tau)
        result['command'].append(command - initial_position)
    
    # Convert to numpy arrays
    for key in ['time', 'position', 'velocity', 'force', 'command']:
        result[key] = np.array(result[key])
    
    return result

def plot_single_joint_response(result, target_angle_deg=20):
    """Plot detailed response for a single joint.
    
    Args:
        result: Dictionary with recorded data
        target_angle_deg: Target angle in degrees
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{result["joint_name"]} Step Response (Target: {target_angle_deg}°)', fontsize=14)
    
    delay_time = result['delay_time']
    
    # Plot 1: Position and Command
    ax1 = axes[0]
    ax1.plot(result['time'], np.rad2deg(result['position']), 'b-', linewidth=2, label='Position')
    ax1.plot(result['time'], np.rad2deg(result['command']), 'r--', linewidth=1, alpha=0.7, label='Command')
    ax1.axvline(x=delay_time, color='gray', linestyle=':', alpha=0.5, label='Command Start')
    ax1.axhline(y=target_angle_deg, color='g', linestyle='--', alpha=0.3, label='Target')
    ax1.set_ylabel('Position (deg)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Calculate metrics after command is sent
    post_delay_idx = np.where(result['time'] >= delay_time)[0]
    if len(post_delay_idx) > 0:
        post_delay_positions = np.rad2deg(result['position'][post_delay_idx])
        
        # Final value
        final_pos = post_delay_positions[-1]
        error = abs(final_pos - target_angle_deg)
        error_percent = (error / target_angle_deg) * 100 if target_angle_deg != 0 else 0
        
        # Rise time (10% to 90%)
        ten_percent = 0.1 * target_angle_deg
        ninety_percent = 0.9 * target_angle_deg
        rise_start_idx = np.where(post_delay_positions >= ten_percent)[0]
        rise_end_idx = np.where(post_delay_positions >= ninety_percent)[0]
        
        if len(rise_start_idx) > 0 and len(rise_end_idx) > 0:
            rise_time = result['time'][post_delay_idx[rise_end_idx[0]]] - result['time'][post_delay_idx[rise_start_idx[0]]]
        else:
            rise_time = None
        
        # Settling time (within 2% of target)
        threshold = 0.98 * target_angle_deg
        settling_idx = np.where(post_delay_positions >= threshold)[0]
        if len(settling_idx) > 0:
            settling_time = result['time'][post_delay_idx[settling_idx[0]]] - delay_time
        else:
            settling_time = None
        
        # Overshoot
        max_pos = np.max(post_delay_positions)
        overshoot = max(0, (max_pos - target_angle_deg) / target_angle_deg * 100)
        
        # Display metrics
        metrics_text = f"Final: {final_pos:.2f}° (Error: {error:.2f}° / {error_percent:.1f}%)"
        if rise_time:
            metrics_text += f"\nRise Time (10-90%): {rise_time:.3f}s"
        if settling_time:
            metrics_text += f"\nSettling Time (2%): {settling_time:.3f}s"
        if overshoot > 0:
            metrics_text += f"\nOvershoot: {overshoot:.1f}%"
        
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Velocity
    ax2 = axes[1]
    ax2.plot(result['time'], np.rad2deg(result['velocity']), 'g-', linewidth=2)
    ax2.axvline(x=delay_time, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Velocity (deg/s)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Display max velocity
    max_vel = np.max(np.abs(result['velocity'][post_delay_idx])) if len(post_delay_idx) > 0 else 0
    ax2.text(0.02, 0.98, f"Max Velocity: {np.rad2deg(max_vel):.1f} deg/s",
            transform=ax2.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 3: Force/Torque
    ax3 = axes[2]
    ax3.plot(result['time'], result['force'], 'r-', linewidth=2)
    ax3.axvline(x=delay_time, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Force/Torque (N·m)', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ax1.set_xlim(0.095, 0.105)
    # ax2.set_xlim(0.095, 0.105)
    # ax3.set_xlim(0.095, 0.105)
    
    # Display max force
    max_force = np.max(np.abs(result['force'][post_delay_idx])) if len(post_delay_idx) > 0 else 0
    steady_force = np.mean(result['force'][-100:]) if len(result['force']) > 100 else result['force'][-1]
    ax3.text(0.02, 0.98, f"Max Force: {max_force:.2f} N·m\nSteady State: {steady_force:.2f} N·m",
            transform=ax3.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    filename = f'rby1_{result["joint_name"]}_response.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {filename}")
    
def list_available_joints(model):
    """List all available joints for testing."""
    joint_info = get_joint_info(model)
    
    print("\nAvailable joints for testing:")
    print("-" * 40)
    
    # Group by type
    torso_joints = [j for j in joint_info if 'torso' in j['joint_name']]
    left_arm_joints = [j for j in joint_info if 'left_arm' in j['joint_name']]
    right_arm_joints = [j for j in joint_info if 'right_arm' in j['joint_name']]
    
    if torso_joints:
        print("\nTorso joints:")
        for j in sorted(torso_joints, key=lambda x: int(x['joint_name'].split('_')[-1])):
            print(f"  {j['joint_name']:<15} (actuator: {j['actuator_name']})")
    
    if left_arm_joints:
        print("\nLeft arm joints:")
        for j in sorted(left_arm_joints, key=lambda x: int(x['joint_name'].split('_')[-1])):
            print(f"  {j['joint_name']:<15} (actuator: {j['actuator_name']})")
    
    if right_arm_joints:
        print("\nRight arm joints:")
        for j in sorted(right_arm_joints, key=lambda x: int(x['joint_name'].split('_')[-1])):
            print(f"  {j['joint_name']:<15} (actuator: {j['actuator_name']})")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Test RBY1 single joint gain tuning')
    parser.add_argument('--joint', type=str, default='left_arm_3',
                       help='Joint name to test (e.g., left_arm_3). If not specified, list available joints.')
    parser.add_argument('--angle', type=float, default=20,
                       help='Target angle in degrees (default: 20)')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay before sending command in seconds (default: 0.1)')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Total simulation duration in seconds (default: 2.0)')
    parser.add_argument('--list', action='store_true',
                       help='List available joints and exit')
    
    args = parser.parse_args()
    
    print("Loading RBY1 model...")
    model, data = load_model()
    print(f"Model loaded successfully!")
    print(f"Timestep: {model.opt.timestep}s")
    
    # List joints if requested
    if args.list or args.joint is None:
        list_available_joints(model)
        if args.joint is None:
            print("\nUsage: python test_rby1_single_joint_gain.py --joint JOINT_NAME")
            print("Example: python test_rby1_single_joint_gain.py --joint left_arm_3")
        return
    
    # Get joint info for specified joint
    joint_info = get_joint_info(model, args.joint)
    
    if not joint_info:
        print(f"\nError: Joint '{args.joint}' not found!")
        list_available_joints(model)
        return
    
    # Run test
    print(f"\n" + "="*50)
    print(f"Testing joint: {args.joint}")
    print(f"Target angle: {args.angle}°")
    print(f"Command delay: {args.delay}s")
    print(f"Total duration: {args.duration}s")
    print("="*50 + "\n")
    
    result = run_single_joint_test(
        model, data, joint_info[0],
        target_angle_deg=args.angle,
        delay_seconds=args.delay,
        total_seconds=args.duration
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_single_joint_response(result, target_angle_deg=args.angle)
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()