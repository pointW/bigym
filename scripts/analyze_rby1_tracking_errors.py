#!/usr/bin/env python3
"""Analyze IK and tracking errors for RBY1 robot executing MovePlate demos."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mujoco

from bigym.envs.move_plates import MovePlate
from bigym.envs.manipulation import FlipCup
from bigym.envs.test_env import TestEnv
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from demonstrations.demo import Demo
from bigym.robots.configs.rby1 import RBY1
from bigym.const import HandSide


def compute_orientation_error(quat1, quat2):
    """Compute angular error between two quaternions in radians."""
    # Normalize quaternions
    quat1 = quat1 / np.linalg.norm(quat1)
    quat2 = quat2 / np.linalg.norm(quat2)
    
    # Compute dot product
    dot = np.clip(np.abs(np.dot(quat1, quat2)), -1.0, 1.0)
    
    # Convert to angle
    angle = 2 * np.arccos(dot)
    return angle


def analyze_rby1_tracking(demo_idx=0, max_steps=None, save_plots=True, show_plots=True):
    """Analyze RBY1 tracking errors step by step.
    
    Args:
        demo_idx: Which demo to analyze (default: 0)
        max_steps: Maximum number of steps to analyze (default: all)
        save_plots: If True, save plots to PDF
        show_plots: If True, display plots
    
    Returns:
        Dictionary containing error data
    """
    
    # Load RBY1 demo
    demo_dir = Path("rby1_cartesian_demos_moveplate")
    # demo_dir = Path("rby1_cartesian_demos_flipcup")
    demo_files = sorted(demo_dir.glob("rby1_cartesian_demo_*.safetensors"))
    
    if not demo_files or demo_idx >= len(demo_files):
        print(f"Demo {demo_idx} not found!")
        return None
    
    demo = Demo.from_safetensors(demo_files[demo_idx])
    print(f"Loaded demo {demo_idx} with seed {demo.seed}, {len(demo.timesteps)} timesteps")
    
    # Create RBY1 environment
    action_mode = RBY1CartesianActionModeWholeBody(
        direct_mode=False,  # Use standard PD control to see tracking errors
        block_until_reached=False,
        control_frequency=50
        # control_frequency=20
    )
    
    env = TestEnv(
        action_mode=action_mode,
        control_frequency=50,
        # control_frequency=20,
        render_mode=None,  # Headless for analysis
        robot_cls=RBY1
    )

    # physics = env.unwrapped._mojo.physics
    # model = physics.model._model
    # data = physics.data._data
    
    # # METHOD 1: Disable ALL collisions globally (including self-collisions)
    # # This is the most aggressive approach
    # print("\nMethod 1: Disabling ALL collisions globally...")
    # model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    # print("✅ All collisions disabled globally")
    
    
    print(f"Action space shape: {env.action_space.shape}")
    
    # Reset with demo seed
    env.reset(seed=demo.seed)
    
    # Storage for error tracking
    steps = []
    
    # IK errors (difference between IK solution and actual achieved poses)
    left_pos_ik_errors = []
    right_pos_ik_errors = []
    left_ori_ik_errors = []
    right_ori_ik_errors = []
    
    # Controller tracking errors (difference between target and achieved)
    left_pos_tracking_errors = []
    right_pos_tracking_errors = []
    left_ori_tracking_errors = []
    right_ori_tracking_errors = []
    
    # Base tracking errors
    base_pos_tracking_errors = []
    base_ori_tracking_errors = []
    
    # Joint tracking errors (difference between IK solution and achieved joint values)
    torso_joint_errors = []  # 6 DOF torso
    right_arm_joint_errors = []  # 7 DOF right arm
    left_arm_joint_errors = []  # 7 DOF left arm
    
    # Self-collision tracking
    self_collision_counts = []
    
    # IK solver info
    ik_costs = []
    ik_iterations = []
    ik_success = []
    
    print("\nStarting analysis...")
    
    num_steps = len(demo.timesteps) if max_steps is None else min(max_steps, len(demo.timesteps))
    
    for step_idx in range(num_steps):
        timestep = demo.timesteps[step_idx]
        
        # Get action
        action = timestep.info.get('demo_action')
        if action is None:
            action = timestep.executed_action
        if action is None:
            action = timestep.action
        
        if action is None:
            print(f"Step {step_idx}: No action found!")
            continue
        
        # Parse action to get target poses
        left_target_pos = action[0:3]
        left_target_ori_6d = action[3:9]
        right_target_pos = action[9:12]
        right_target_ori_6d = action[12:18]
        
        # Convert 6D rotation to quaternion for targets
        from bigym.rby1_cartesian_action_mode_whole_body import rotation_6d_to_matrix
        from pyquaternion import Quaternion
        
        left_rot_matrix = rotation_6d_to_matrix(left_target_ori_6d)
        right_rot_matrix = rotation_6d_to_matrix(right_target_ori_6d)
        
        left_target_quat = Quaternion(matrix=left_rot_matrix)
        right_target_quat = Quaternion(matrix=right_rot_matrix)
        
        # Convert to numpy arrays in wxyz format
        left_target_quat_np = np.array([left_target_quat.w, left_target_quat.x, 
                                        left_target_quat.y, left_target_quat.z])
        right_target_quat_np = np.array([right_target_quat.w, right_target_quat.x,
                                         right_target_quat.y, right_target_quat.z])
        
        # Step environment
        try:
            _, _, terminated, truncated, info = env.step(action)
            
            # Get IK solution info from action mode
            ik_solution, ik_info = action_mode.get_last_ik_solution()
            
            # Variables to store FK results from IK solution
            fk_left_pos = None
            fk_right_pos = None
            fk_left_quat = None
            fk_right_quat = None
            
            if ik_solution is not None and ik_info is not None:
                # Extract IK solver information
                if 'cost' in ik_info:
                    ik_costs.append(ik_info['cost'])
                else:
                    ik_costs.append(np.nan)
                
                if 'iterations' in ik_info:
                    ik_iterations.append(ik_info['iterations'])
                else:
                    ik_iterations.append(0)
                
                if 'success' in ik_info:
                    ik_success.append(1 if ik_info['success'] else 0)
                else:
                    ik_success.append(1)  # Assume success if not specified
                
                # Run FK on IK solution to get the end-effector poses it would produce
                # We need to temporarily set qpos to IK solution and compute FK
                physics = env.robot._mojo.physics
                model = physics.model._model
                data = physics.data._data
                
                # Save current qpos
                original_qpos = data.qpos.copy()
                
                # Set qpos to IK solution (only the relevant joints)
                # RBY1 qpos structure: [base(3), quat(4), wheels(4), torso(6), right_arm(7), gripper, left_arm(7)]
                data.qpos[:len(ik_solution)] = ik_solution
                
                # Forward kinematics to compute site positions
                mujoco.mj_forward(model, data)
                
                # Check for self-collisions
                num_contacts = data.ncon
                self_collision_count = 0
                collision_pairs = {}  # Track unique collision pairs with geom info
                for i in range(num_contacts):
                    contact = data.contact[i]
                    # Get geom and body names for the contact
                    geom1_id = contact.geom1
                    geom2_id = contact.geom2
                    if geom1_id >= 0 and geom2_id >= 0:
                        # Get geom names
                        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                        
                        body1_id = model.geom_bodyid[geom1_id]
                        body2_id = model.geom_bodyid[geom2_id]
                        # Check if both bodies belong to the robot (self-collision)
                        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                        if body1_name and body2_name:
                            if 'rby1' in body1_name.lower() and 'rby1' in body2_name.lower():
                                self_collision_count += 1
                                # Track collision pair with geom names
                                pair_key = tuple(sorted([body1_name, body2_name]))
                                geom_pair = f"{geom1_name} <-> {geom2_name}"
                                if pair_key not in collision_pairs:
                                    collision_pairs[pair_key] = {}
                                if geom_pair not in collision_pairs[pair_key]:
                                    collision_pairs[pair_key][geom_pair] = 0
                                collision_pairs[pair_key][geom_pair] += 1
                
                # Print collision details for first few steps
                if step_idx < 3 and collision_pairs:
                    print(f"\n  Self-collisions at step {step_idx}:")
                    for (body1, body2), geom_contacts in sorted(collision_pairs.items()):
                        print(f"    Bodies: {body1} <-> {body2}")
                        for geom_pair, count in sorted(geom_contacts.items()):
                            print(f"      Geoms: {geom_pair} ({count} contacts)")
                
                # Get FK end-effector positions from wrist sites
                left_site = env.robot._wrist_sites[HandSide.LEFT]
                right_site = env.robot._wrist_sites[HandSide.RIGHT]
                
                left_site_bind = physics.bind(left_site.mjcf)
                right_site_bind = physics.bind(right_site.mjcf)
                
                fk_left_pos = left_site_bind.xpos.copy()
                fk_right_pos = right_site_bind.xpos.copy()
                
                # Get FK orientations
                from scipy.spatial.transform import Rotation
                left_rot = Rotation.from_matrix(left_site_bind.xmat.reshape(3, 3))
                right_rot = Rotation.from_matrix(right_site_bind.xmat.reshape(3, 3))
                
                left_quat_xyzw = left_rot.as_quat()  # [x, y, z, w]
                right_quat_xyzw = right_rot.as_quat()
                fk_left_quat = np.array([left_quat_xyzw[3], left_quat_xyzw[0], 
                                        left_quat_xyzw[1], left_quat_xyzw[2]])
                fk_right_quat = np.array([right_quat_xyzw[3], right_quat_xyzw[0],
                                         right_quat_xyzw[1], right_quat_xyzw[2]])
                
                # Restore original qpos
                data.qpos[:] = original_qpos
                mujoco.mj_forward(model, data)
                
            else:
                # No IK info available
                ik_costs.append(np.nan)
                ik_iterations.append(0)
                ik_success.append(0)
                # Use targets as fallback
                fk_left_pos = left_target_pos
                fk_right_pos = right_target_pos
                fk_left_quat = left_target_quat_np
                fk_right_quat = right_target_quat_np
            
            # Get actual achieved end-effector positions
            left_site = env.robot._wrist_sites[HandSide.LEFT]
            right_site = env.robot._wrist_sites[HandSide.RIGHT]
            
            achieved_left_pos = left_site.get_position()
            achieved_right_pos = right_site.get_position()
            
            # Get achieved orientations
            from scipy.spatial.transform import Rotation
            physics = env.robot._mojo.physics
            left_site_bind = physics.bind(left_site.mjcf)
            right_site_bind = physics.bind(right_site.mjcf)
            
            left_rot = Rotation.from_matrix(left_site_bind.xmat.reshape(3, 3))
            right_rot = Rotation.from_matrix(right_site_bind.xmat.reshape(3, 3))
            
            # Convert to quaternions (wxyz format)
            left_quat_xyzw = left_rot.as_quat()  # [x, y, z, w]
            right_quat_xyzw = right_rot.as_quat()  # [x, y, z, w]
            achieved_left_quat = np.array([left_quat_xyzw[3], left_quat_xyzw[0], 
                                          left_quat_xyzw[1], left_quat_xyzw[2]])
            achieved_right_quat = np.array([right_quat_xyzw[3], right_quat_xyzw[0],
                                           right_quat_xyzw[1], right_quat_xyzw[2]])
            
            # Compute IK errors (FK of IK solution vs original target)
            # This tells us how well the IK solver solved for the target
            left_pos_ik_error = np.linalg.norm(fk_left_pos - left_target_pos)
            right_pos_ik_error = np.linalg.norm(fk_right_pos - right_target_pos)
            left_ori_ik_error = compute_orientation_error(fk_left_quat, left_target_quat_np)
            right_ori_ik_error = compute_orientation_error(fk_right_quat, right_target_quat_np)
            
            # Compute tracking errors (FK of IK solution vs what was actually achieved)
            # This tells us how well the controller tracked the IK solution
            left_pos_tracking_error = np.linalg.norm(fk_left_pos - achieved_left_pos)
            right_pos_tracking_error = np.linalg.norm(fk_right_pos - achieved_right_pos)
            left_ori_tracking_error = compute_orientation_error(fk_left_quat, achieved_left_quat)
            right_ori_tracking_error = compute_orientation_error(fk_right_quat, achieved_right_quat)
            
            # Compute base tracking errors
            # Get actual base position after step
            actual_base_pos = data.qpos[0:3].copy()  # [x, y, z]
            actual_base_quat = data.qpos[3:7].copy()  # [w, x, y, z]
            
            # Target base from IK solution
            target_base_pos = ik_solution[0:3] if ik_solution is not None else np.zeros(3)
            target_base_quat = ik_solution[3:7] if ik_solution is not None else np.array([1, 0, 0, 0])
            
            base_pos_tracking_error = np.linalg.norm(actual_base_pos[0:2] - target_base_pos[0:2])  # Only X,Y
            base_ori_tracking_error = compute_orientation_error(target_base_quat, actual_base_quat)
            
            # Compute joint tracking errors (IK solution vs achieved joint values)
            # RBY1 qpos indices: base(0-6), wheels(7-10), torso(11-16), right_arm(17-23), 
            # right_gripper(24-31), left_arm(32-38), left_gripper(39-46+)
            if ik_solution is not None:
                # Get actual joint values
                actual_torso_joints = data.qpos[11:17].copy()
                actual_right_arm_joints = data.qpos[17:24].copy()
                actual_left_arm_joints = data.qpos[32:39].copy()  # Left arm at 32-38!
                
                # Get target joint values from IK solution
                # IK solution contains full qpos, so use same indices
                target_torso_joints = ik_solution[11:17] if len(ik_solution) > 17 else actual_torso_joints
                target_right_arm_joints = ik_solution[17:24] if len(ik_solution) > 24 else actual_right_arm_joints
                target_left_arm_joints = ik_solution[32:39] if len(ik_solution) > 39 else actual_left_arm_joints
                
                # Compute RMS errors for each joint group
                torso_error = np.sqrt(np.mean((target_torso_joints - actual_torso_joints)**2))
                right_arm_error = np.sqrt(np.mean((target_right_arm_joints - actual_right_arm_joints)**2))
                left_arm_error = np.sqrt(np.mean((target_left_arm_joints - actual_left_arm_joints)**2))
            else:
                torso_error = 0.0
                right_arm_error = 0.0
                left_arm_error = 0.0
            
            # Store errors
            steps.append(step_idx)
            left_pos_ik_errors.append(left_pos_ik_error * 1000)  # Convert to mm
            right_pos_ik_errors.append(right_pos_ik_error * 1000)
            left_ori_ik_errors.append(np.degrees(left_ori_ik_error))  # Convert to degrees
            right_ori_ik_errors.append(np.degrees(right_ori_ik_error))
            
            left_pos_tracking_errors.append(left_pos_tracking_error * 1000)  # mm
            right_pos_tracking_errors.append(right_pos_tracking_error * 1000)
            left_ori_tracking_errors.append(np.degrees(left_ori_tracking_error))  # degrees
            right_ori_tracking_errors.append(np.degrees(right_ori_tracking_error))
            
            base_pos_tracking_errors.append(base_pos_tracking_error * 1000)  # mm
            base_ori_tracking_errors.append(np.degrees(base_ori_tracking_error))  # degrees
            
            # Store joint tracking errors (in radians)
            torso_joint_errors.append(torso_error)
            right_arm_joint_errors.append(right_arm_error)
            left_arm_joint_errors.append(left_arm_error)
            
            self_collision_counts.append(self_collision_count)
            
            # Print progress every 10 steps
            if step_idx % 10 == 0:
                print(f"Step {step_idx}: "
                      f"EE tracking L={left_pos_tracking_error*1000:.2f}mm, "
                      f"R={right_pos_tracking_error*1000:.2f}mm, "
                      f"Base={base_pos_tracking_error*1000:.2f}mm, "
                      f"Collisions={self_collision_count}")
            
            if info.get('task_success', False):
                print(f"✅ SUCCESS at step {step_idx}!")
                break
            
            if terminated or truncated:
                print(f"Episode ended at step {step_idx}")
                break
                
        except Exception as e:
            print(f"  ERROR at step {step_idx}: {e}")
            break
    
    env.close()
    
    # Create plots
    if len(steps) > 0:
        fig = plt.figure(figsize=(16, 16), dpi=300)
        
        # Plot 1: Controller Position tracking errors
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(steps, left_pos_tracking_errors, 'b-', label='Left EE', linewidth=2)
        ax1.plot(steps, right_pos_tracking_errors, 'r-', label='Right EE', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Position Error (mm)')
        ax1.set_title('Controller Tracking Error (FK(IK) vs Achieved Position)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Controller Orientation tracking errors
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(steps, left_ori_tracking_errors, 'b-', label='Left EE', linewidth=2)
        ax2.plot(steps, right_ori_tracking_errors, 'r-', label='Right EE', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Orientation Error (degrees)')
        ax2.set_title('Controller Tracking Error (FK(IK) vs Achieved Orientation)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: IK solver Position errors
        ax3 = plt.subplot(4, 2, 3)
        ax3.plot(steps, left_pos_ik_errors, 'b--', label='Left EE', linewidth=2)
        ax3.plot(steps, right_pos_ik_errors, 'r--', label='Right EE', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Position Error (mm)')
        ax3.set_title('IK Solver Error (FK(IK) vs Target Position)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: IK solver Orientation errors
        ax4 = plt.subplot(4, 2, 4)
        ax4.plot(steps, left_ori_ik_errors, 'b--', label='Left EE', linewidth=2)
        ax4.plot(steps, right_ori_ik_errors, 'r--', label='Right EE', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Orientation Error (degrees)')
        ax4.set_title('IK Solver Error (FK(IK) vs Target Orientation)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: Base tracking errors
        ax5 = plt.subplot(4, 2, 5)
        ax5.plot(steps, base_pos_tracking_errors, 'g-', label='Position (XY)', linewidth=2)
        ax5_twin = ax5.twinx()
        ax5_twin.plot(steps, base_ori_tracking_errors, 'm--', label='Orientation', linewidth=1.5)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Position Error (mm)', color='g')
        ax5_twin.set_ylabel('Orientation Error (degrees)', color='m')
        ax5.set_title('Base Tracking Error')
        ax5.tick_params(axis='y', labelcolor='g')
        ax5_twin.tick_params(axis='y', labelcolor='m')
        ax5.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 6: Joint tracking errors
        ax6 = plt.subplot(4, 2, 6)
        ax6.plot(steps, np.array(torso_joint_errors) * 1000, 'c-', label='Torso (6 DOF)', linewidth=2)
        ax6.plot(steps, np.array(right_arm_joint_errors) * 1000, 'r-', label='Right Arm (7 DOF)', linewidth=2)
        ax6.plot(steps, np.array(left_arm_joint_errors) * 1000, 'b-', label='Left Arm (7 DOF)', linewidth=2)
        ax6.set_xlabel('Step')
        ax6.set_ylabel('RMS Joint Error (mrad)')
        ax6.set_title('Joint Tracking Error (IK vs Achieved)')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Plot 7: IK solver metrics
        ax7 = plt.subplot(4, 2, 7)
        ax7_twin = ax7.twinx()
        
        # Plot iterations on left y-axis
        if any(not np.isnan(x) for x in ik_iterations):
            ax7.plot(steps, ik_iterations, 'g-', label='Iterations', linewidth=2)
            ax7.set_ylabel('IK Iterations', color='g')
            ax7.tick_params(axis='y', labelcolor='g')
        
        # Plot success rate on right y-axis  
        if len(ik_success) > 0:
            ax7_twin.plot(steps, ik_success, 'm--', label='Success', linewidth=1, alpha=0.5)
            ax7_twin.set_ylabel('IK Success', color='m')
            ax7_twin.tick_params(axis='y', labelcolor='m')
            ax7_twin.set_ylim([-0.1, 1.1])
        
        ax7.set_xlabel('Step')
        ax7.set_title('IK Solver Performance')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Combined comparison
        ax8 = plt.subplot(4, 2, 8)
        # Average position errors
        avg_pos_tracking = [(l + r) / 2 for l, r in zip(left_pos_tracking_errors, right_pos_tracking_errors)]
        avg_pos_ik = [(l + r) / 2 for l, r in zip(left_pos_ik_errors, right_pos_ik_errors)]
        
        ax8.plot(steps, avg_pos_tracking, 'b-', label='Tracking Error', linewidth=2)
        ax8.plot(steps, avg_pos_ik, 'r--', label='IK Error', linewidth=2)
        ax8.set_xlabel('Step')
        ax8.set_ylabel('Average Position Error (mm)')
        ax8.set_title('Tracking vs IK Error Comparison (Average of Both Arms)')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        plt.suptitle(f'RBY1 Tracking Analysis - Demo {demo_idx} (Seed: {demo.seed})', fontsize=14)
        plt.tight_layout()
        
        # Save plots if requested
        if save_plots:
            jpg_path = f'rby1_tracking_analysis_demo_{demo_idx}.jpg'
            plt.savefig(jpg_path)
        
        # Show plots if requested
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print("\nController Tracking Errors (FK(IK) vs Achieved):")
        print(f"  Left Position:  Mean={np.mean(left_pos_tracking_errors):.2f}mm, "
              f"Max={np.max(left_pos_tracking_errors):.2f}mm")
        print(f"  Right Position: Mean={np.mean(right_pos_tracking_errors):.2f}mm, "
              f"Max={np.max(right_pos_tracking_errors):.2f}mm")
        print(f"  Left Orientation:  Mean={np.mean(left_ori_tracking_errors):.2f}°, "
              f"Max={np.max(left_ori_tracking_errors):.2f}°")
        print(f"  Right Orientation: Mean={np.mean(right_ori_tracking_errors):.2f}°, "
              f"Max={np.max(right_ori_tracking_errors):.2f}°")
        print(f"  Base Position:  Mean={np.mean(base_pos_tracking_errors):.2f}mm, "
              f"Max={np.max(base_pos_tracking_errors):.2f}mm")
        print(f"  Base Orientation: Mean={np.mean(base_ori_tracking_errors):.2f}°, "
              f"Max={np.max(base_ori_tracking_errors):.2f}°")
        
        print("\nIK Solver Errors (FK(IK) vs Target):")
        print(f"  Left Position:  Mean={np.mean(left_pos_ik_errors):.2f}mm, "
              f"Max={np.max(left_pos_ik_errors):.2f}mm")
        print(f"  Right Position: Mean={np.mean(right_pos_ik_errors):.2f}mm, "
              f"Max={np.max(right_pos_ik_errors):.2f}mm")
        print(f"  Left Orientation:  Mean={np.mean(left_ori_ik_errors):.2f}°, "
              f"Max={np.max(left_ori_ik_errors):.2f}°")
        print(f"  Right Orientation: Mean={np.mean(right_ori_ik_errors):.2f}°, "
              f"Max={np.max(right_ori_ik_errors):.2f}°")
        
        if any(not np.isnan(x) for x in ik_iterations):
            valid_iterations = [x for x in ik_iterations if not np.isnan(x)]
            print(f"\nIK Solver:")
            print(f"  Iterations: Mean={np.mean(valid_iterations):.1f}, "
                  f"Max={np.max(valid_iterations):.0f}")
            print(f"  Success Rate: {np.mean(ik_success)*100:.1f}%")
        
        print(f"\nJoint Tracking Errors (IK solution vs Achieved):")
        print(f"  Torso RMS:     Mean={np.mean(torso_joint_errors)*1000:.2f}mrad, "
              f"Max={np.max(torso_joint_errors)*1000:.2f}mrad")
        print(f"  Right Arm RMS: Mean={np.mean(right_arm_joint_errors)*1000:.2f}mrad, "
              f"Max={np.max(right_arm_joint_errors)*1000:.2f}mrad")
        print(f"  Left Arm RMS:  Mean={np.mean(left_arm_joint_errors)*1000:.2f}mrad, "
              f"Max={np.max(left_arm_joint_errors)*1000:.2f}mrad")
        
        print(f"\nSelf-Collisions:")
        print(f"  Total steps with collisions: {sum(1 for x in self_collision_counts if x > 0)}")
        print(f"  Max collisions in a step: {max(self_collision_counts) if self_collision_counts else 0}")
        print(f"  Mean collisions per step: {np.mean(self_collision_counts):.2f}")
    
    # Return data for further analysis
    return {
        'steps': steps,
        'left_pos_tracking_errors': left_pos_tracking_errors,
        'right_pos_tracking_errors': right_pos_tracking_errors,
        'left_ori_tracking_errors': left_ori_tracking_errors,
        'right_ori_tracking_errors': right_ori_tracking_errors,
        'left_pos_ik_errors': left_pos_ik_errors,
        'right_pos_ik_errors': right_pos_ik_errors,
        'left_ori_ik_errors': left_ori_ik_errors,
        'right_ori_ik_errors': right_ori_ik_errors,
        'base_pos_tracking_errors': base_pos_tracking_errors,
        'base_ori_tracking_errors': base_ori_tracking_errors,
        'torso_joint_errors': torso_joint_errors,
        'right_arm_joint_errors': right_arm_joint_errors,
        'left_arm_joint_errors': left_arm_joint_errors,
        'ik_costs': ik_costs,
        'ik_iterations': ik_iterations,
        'ik_success': ik_success,
        'self_collision_counts': self_collision_counts
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze RBY1 tracking errors")
    parser.add_argument("--demo", type=int, default=0, help="Demo index to analyze")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps to analyze")
    parser.add_argument("--no-save", action="store_true", help="Don't save plots to PDF")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    args = parser.parse_args()
    
    analyze_rby1_tracking(
        demo_idx=args.demo,
        max_steps=args.max_steps,
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )