"""Test script for tuning RBY1 joint gains.

This script sends a 20-degree step command to each joint and measures
the response over time, plotting position and force for each joint.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set up paths
BASE_DIR = Path(__file__).parent.parent
WORLD_XML = BASE_DIR / "bigym/envs/xmls/world.xml"
RBY1_XML = BASE_DIR / "bigym/envs/xmls/rby1/model_act_consolidated.xml"

def load_model():
    """Load the RBY1 model with world."""
    # Read the world XML
    with open(WORLD_XML, 'r') as f:
        world_xml = f.read()
    
    # Read the RBY1 model XML
    with open(RBY1_XML, 'r') as f:
        rby1_xml = f.read()
    
    # Create a combined XML that includes both
    combined_xml = f"""
    <mujoco model="rby1_test">
        <!-- Include world elements -->
        <compiler angle="radian" meshdir="../../../assets_mesh" texturedir="../../../assets_texture"/>
        
        <option timestep="0.002" impratio="20"/>
        
        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
        </visual>
        
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        </asset>
        
        <worldbody>
            <!-- Ground plane -->
            <geom name="floor" pos="0 0 0" size="0 0 0.05" type="plane" material="groundplane" condim="3"/>
            
            <!-- Include RBY1 robot -->
            <include file="{RBY1_XML}"/>
        </worldbody>
    </mujoco>
    """
    
    # Save combined XML temporarily
    temp_xml_path = "/tmp/rby1_test_combined.xml"
    with open(temp_xml_path, 'w') as f:
        f.write(combined_xml)
    
    # Load the model
    model = mujoco.MjModel.from_xml_path(str(RBY1_XML))
    data = mujoco.MjData(model)
    
    return model, data

def get_joint_info(model):
    """Get information about all controllable joints."""
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
            
            # Skip base, wheel, and head joints for this test
            if joint_name and not any(skip in joint_name for skip in ['world_j', 'wheel', 'head']):
                joint_info.append({
                    'actuator_id': i,
                    'joint_id': joint_id,
                    'actuator_name': actuator_name,
                    'joint_name': joint_name,
                    'qpos_idx': model.jnt_qposadr[joint_id]
                })
    
    # Sort joints into groups: torso, left_arm, right_arm
    torso_joints = [j for j in joint_info if 'torso' in j['joint_name']]
    left_arm_joints = [j for j in joint_info if 'left_arm' in j['joint_name']]
    right_arm_joints = [j for j in joint_info if 'right_arm' in j['joint_name']]
    
    # Sort each group by joint number
    torso_joints.sort(key=lambda x: int(x['joint_name'].split('_')[-1]))
    left_arm_joints.sort(key=lambda x: int(x['joint_name'].split('_')[-1]))
    right_arm_joints.sort(key=lambda x: int(x['joint_name'].split('_')[-1]))
    
    # Return in organized order
    return torso_joints + left_arm_joints + right_arm_joints

def run_step_response_test(model, data, joint_info, target_angle_deg=20, num_steps=1000):
    """Run step response test for all joints.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_info: List of joint information dictionaries
        target_angle_deg: Target angle in degrees
        num_steps: Number of simulation steps to run
    
    Returns:
        Dictionary with recorded data for each joint
    """
    target_angle_rad = np.deg2rad(target_angle_deg)
    
    # Initialize data storage
    results = {}
    for info in joint_info:
        results[info['joint_name']] = {
            'time': [],
            'position': [],
            'force': [],
            'target': target_angle_rad,
            'actuator_name': info['actuator_name']
        }
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Get initial positions
    initial_positions = {}
    for info in joint_info:
        initial_positions[info['joint_name']] = data.qpos[info['qpos_idx']]
    
    # Run simulation
    for step in range(num_steps):
        # Set control targets (relative to initial position)
        for info in joint_info:
            initial_pos = initial_positions[info['joint_name']]
            data.ctrl[info['actuator_id']] = initial_pos + target_angle_rad
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record data
        time = step * model.opt.timestep
        for info in joint_info:
            joint_name = info['joint_name']
            results[joint_name]['time'].append(time)
            results[joint_name]['position'].append(
                data.qpos[info['qpos_idx']] - initial_positions[joint_name]
            )
            # Record actuator force
            results[joint_name]['force'].append(data.actuator_force[info['actuator_id']])
    
    # Convert to numpy arrays
    for joint_name in results:
        for key in ['time', 'position', 'force']:
            results[joint_name][key] = np.array(results[joint_name][key])
    
    return results

def plot_results(results, target_angle_deg=20):
    """Plot step response results for all joints.
    
    Args:
        results: Dictionary with recorded data for each joint
        target_angle_deg: Target angle in degrees for reference line
    """
    # Organize joints by type
    joint_names = list(results.keys())
    torso_joints = sorted([j for j in joint_names if 'torso' in j], 
                          key=lambda x: int(x.split('_')[-1]))
    left_arm_joints = sorted([j for j in joint_names if 'left_arm' in j],
                            key=lambda x: int(x.split('_')[-1]))
    right_arm_joints = sorted([j for j in joint_names if 'right_arm' in j],
                             key=lambda x: int(x.split('_')[-1]))
    
    # Fixed layout: 3 rows (torso, left arm, right arm), 7 columns
    rows = 3
    cols = 7
    
    # Create figure with subplots for position
    fig1, axes1 = plt.subplots(rows, cols, figsize=(20, 9))
    fig1.suptitle(f'Joint Position Step Response (Target: {target_angle_deg}°)', fontsize=14)
    
    # Create figure with subplots for force
    fig2, axes2 = plt.subplots(rows, cols, figsize=(20, 9))
    fig2.suptitle(f'Joint Force Step Response (Target: {target_angle_deg}°)', fontsize=14)
    
    target_angle_rad = np.deg2rad(target_angle_deg)
    
    # Helper function to plot a joint
    def plot_joint(joint_name, data, ax1, ax2):
        # Plot position
        ax1.plot(data['time'], np.rad2deg(data['position']), 'b-', label='Position')
        ax1.axhline(y=target_angle_deg, color='r', linestyle='--', alpha=0.5, label='Target')
        # ax1.set_ylim(19.5, 20.5)
        ax1.set_xlabel('Time (s)', fontsize=8)
        ax1.set_ylabel('Position (deg)', fontsize=8)
        ax1.set_title(f'{joint_name}', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=6)
        ax1.tick_params(axis='both', labelsize=7)
        
        # Calculate and display settling metrics
        final_pos = np.rad2deg(data['position'][-1])
        error = abs(final_pos - target_angle_deg)
        error_percent = (error / target_angle_deg) * 100 if target_angle_deg != 0 else 0
        
        threshold = 0.999 * target_angle_deg
        settling_idx = np.where(np.rad2deg(data['position']) >= threshold)[0]
        settling_time = data['time'][settling_idx[0]] if len(settling_idx) > 0 else None
        
        info_text = f"Final: {final_pos:.1f}°\nError: {error:.1f}° ({error_percent:.1f}%)"
        if settling_time:
            info_text += f"\nSettling: {settling_time:.3f}s"
        ax1.text(0.95, 0.05, info_text, transform=ax1.transAxes, 
                fontsize=6, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot force
        ax2.plot(data['time'], data['force'], 'g-', label='Force')
        ax2.set_xlabel('Time (s)', fontsize=8)
        ax2.set_ylabel('Force (N·m)', fontsize=8)
        ax2.set_title(f'{joint_name}', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=6)
        ax2.tick_params(axis='both', labelsize=7)
        
        # Display max force
        max_force = np.max(np.abs(data['force']))
        ax2.text(0.95, 0.95, f"Max: {max_force:.1f} N·m", transform=ax2.transAxes,
                fontsize=6, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot torso joints (row 0)
    for i, joint_name in enumerate(torso_joints):
        if i < cols and joint_name in results:
            plot_joint(joint_name, results[joint_name], axes1[0, i], axes2[0, i])
    
    # Plot left arm joints (row 1)
    for i, joint_name in enumerate(left_arm_joints):
        if i < cols and joint_name in results:
            plot_joint(joint_name, results[joint_name], axes1[1, i], axes2[1, i])
    
    # Plot right arm joints (row 2)
    for i, joint_name in enumerate(right_arm_joints):
        if i < cols and joint_name in results:
            plot_joint(joint_name, results[joint_name], axes1[2, i], axes2[2, i])
    
    # Hide unused subplots
    for row in range(rows):
        if row == 0:  # Torso row
            for col in range(len(torso_joints), cols):
                axes1[row, col].axis('off')
                axes2[row, col].axis('off')
        elif row == 1:  # Left arm row
            for col in range(len(left_arm_joints), cols):
                axes1[row, col].axis('off')
                axes2[row, col].axis('off')
        elif row == 2:  # Right arm row
            for col in range(len(right_arm_joints), cols):
                axes1[row, col].axis('off')
                axes2[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figures
    fig1.savefig('rby1_joint_position_response.png', dpi=500, bbox_inches='tight')
    fig2.savefig('rby1_joint_force_response.png', dpi=500, bbox_inches='tight')
    print(f"Saved plots to rby1_joint_position_response.png and rby1_joint_force_response.png")
    
    # plt.show()

def main():
    """Main function to run the joint gain tuning test."""
    print("Loading RBY1 model...")
    model, data = load_model()
    
    print(f"Model loaded successfully!")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of joints: {model.njnt}")
    print(f"Timestep: {model.opt.timestep}s")
    
    # Get joint information
    joint_info = get_joint_info(model)
    print(f"\nFound {len(joint_info)} controllable joints (excluding base, wheels, and grippers)")
    
    for info in joint_info:
        print(f"  - {info['joint_name']} (actuator: {info['actuator_name']})")
    
    # Run step response test
    print(f"\nRunning step response test (20° target)...")
    results = run_step_response_test(model, data, joint_info, 
                                    target_angle_deg=-20, 
                                    num_steps=800)  # 4 seconds at 0.002s timestep
    
    # Plot results
    print("Plotting results...")
    plot_results(results, target_angle_deg=-20)
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()