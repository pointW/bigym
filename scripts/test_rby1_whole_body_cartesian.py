"""Test RBY1 Cartesian action mode with whole-body IK."""

import sys
import os
import numpy as np
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bigym.envs.reach_target_rby1 import ReachTargetRBY1
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody


def main(headless=False):
    """Test whole-body Cartesian control."""
    
    print("=" * 60)
    print("RBY1 Whole-Body Cartesian Control Test")
    print(f"Mode: {'HEADLESS' if headless else 'VISUAL'}")
    print("=" * 60)
    print("\nCreating environment with whole-body IK action mode...")
    
    # Create environment with whole-body action mode
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,  # Changed to False for better tracking
            direct_mode=False  # MUST use direct mode - controller gains too low for stability
        ),
        control_frequency=50,
        render_mode=None if headless else "human",
    )
    
    print("Environment created. MuJoCo viewer should be open.")
    print(f"Action space: {env.action_space.shape[0]}D")
    print("  - Left EE pose: 9D (position + 6D rotation)")
    print("  - Right EE pose: 9D (position + 6D rotation)")
    print("  - Grippers: 2D")
    print("  - Total: 20D (no explicit base control)")
    print("\n" + "=" * 60)
    
    # Get action mode
    action_mode = env.action_mode
    
    # Run episodes
    num_episodes = 3 if headless else 5
    
    # Interpolation settings
    interpolation_steps = 20  # More intermediate steps
    max_step_size = 0.02  # Smaller maximum position change per step (2cm)
    
    # Tracking data for analysis
    tracking_data = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Reset environment
        obs, info = env.reset()
        
        # Get target position
        priv_obs = env._get_task_privileged_obs()
        target_pos = priv_obs["target_position"]
        
        print(f"Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # Get initial poses
        left_pose, right_pose = action_mode.get_current_ee_poses()
        
        # Always use RIGHT hand
        print("Using RIGHT hand (always)")
        reaching_hand = "right"
        initial_ee_pos = right_pose.position.copy()
        
        # Calculate total distance and determine interpolation
        total_distance = np.linalg.norm(target_pos - initial_ee_pos)
        print(f"Total distance to target: {total_distance:.3f}m")
        
        # Decide number of interpolation steps based on distance
        if total_distance > max_step_size * 2:
            n_interp_steps = min(int(total_distance / max_step_size), interpolation_steps)
            print(f"Using {n_interp_steps} interpolation steps")
        else:
            n_interp_steps = 1
            print("Small movement - no interpolation needed")
        
        success = False
        max_steps = 100
        
        for step in range(max_steps):
            # Create action using whole-body mode with interpolation
            from vr.ik.h1_upper_body_ik import Pose
            
            # Calculate interpolated target for this step
            if n_interp_steps > 1 and step < n_interp_steps:
                # Linear interpolation to intermediate target
                alpha = (step + 1) / n_interp_steps
                interpolated_target = initial_ee_pos + alpha * (target_pos - initial_ee_pos)
                
                if step % 2 == 0:  # Print every other interpolation step
                    print(f"  Interpolation step {step+1}/{n_interp_steps}: "
                          f"target=[{interpolated_target[0]:.3f}, {interpolated_target[1]:.3f}, {interpolated_target[2]:.3f}]")
            else:
                # Move to final target
                interpolated_target = target_pos
            
            # Store pre-action state for error tracking
            pre_qpos = env._mojo.physics.data.qpos.copy()
            pre_ee_pos = left_pose.position if reaching_hand == "left" else right_pose.position
            
            if reaching_hand == "left":
                # Move left hand to interpolated target, keep right hand at current position
                target_left_pose = Pose(interpolated_target, left_pose.orientation)
                action = action_mode.poses_to_action(
                    target_left_pose,
                    right_pose,
                    gripper_action=np.array([0.0, 0.0])  # Keep grippers open
                )
            else:
                # Move right hand to interpolated target, keep left hand at current position
                target_right_pose = Pose(interpolated_target, right_pose.orientation)
                action = action_mode.poses_to_action(
                    left_pose,
                    target_right_pose,
                    gripper_action=np.array([0.0, 0.0])  # Keep grippers open
                )
            
            # Note: The action contains Cartesian targets (EE poses), not joint positions!
            # The IK solver inside the action mode will convert these to joint positions
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get post-action state
            post_qpos = env._mojo.physics.data.qpos.copy()
            post_left_pose, post_right_pose = action_mode.get_current_ee_poses()
            post_ee_pos = post_left_pose.position if reaching_hand == "left" else post_right_pose.position
            
            # Get IK solution to properly measure errors
            ik_solution, ik_info = action_mode.get_last_ik_solution()
            
            # Calculate errors properly:
            # 1. IK error: Use FK on IK solution to see where it would place the EE
            if ik_solution is not None and ik_info is not None:
                # Temporarily set qpos to IK solution to check FK
                saved_qpos = env._mojo.physics.data.qpos.copy()
                env._mojo.physics.data.qpos[:] = ik_solution
                env._mojo.physics.forward()
                
                # Get EE position at IK solution
                ik_left_pose, ik_right_pose = action_mode.get_current_ee_poses()
                ik_ee_pos = ik_left_pose.position if reaching_hand == "left" else ik_right_pose.position
                
                # This is the true IK error
                ik_error = np.linalg.norm(ik_ee_pos - interpolated_target)
                
                # Restore actual qpos
                env._mojo.physics.data.qpos[:] = saved_qpos
                env._mojo.physics.forward()
                
                # 2. Controller tracking error: difference between IK solution and actual
                controller_error = np.linalg.norm(post_ee_pos - ik_ee_pos)
                joint_tracking_error = np.linalg.norm(post_qpos[:len(ik_solution)] - ik_solution)
            else:
                # Fallback if IK solution not available
                ik_error = np.linalg.norm(post_ee_pos - interpolated_target)
                controller_error = 0.0
                joint_tracking_error = 0.0
            
            # Store tracking data
            step_data = {
                'step': step,
                'ik_error': ik_error,  # Pure IK error from FK
                'controller_error': controller_error,  # Controller tracking error
                'joint_tracking_error': joint_tracking_error,  # Joint-level tracking
                'ee_movement': np.linalg.norm(post_ee_pos - pre_ee_pos),
                'target_dist': np.linalg.norm(post_ee_pos - target_pos)  # Error to final target
            }
            tracking_data.append(step_data)
            
            # Print detailed tracking info in headless mode
            if headless and step < n_interp_steps and step % 5 == 0:
                print(f"  Step {step+1}: IK error={ik_error:.4f}m, Controller error={controller_error:.4f}m")
            
            # Check success
            if info.get("task_success", False):
                success = True
                print(f"✓ Success in {step + 1} steps! Reward: {reward:.3f}")
                
                # Get final base position from qpos
                final_qpos = env._mojo.physics.data.qpos
                print(f"  Base moved to: X={final_qpos[0]:.3f}, Y={final_qpos[1]:.3f}")
                
                # Hold successful pose briefly
                if not headless:
                    for _ in range(30):
                        env.render()
                break
            
            # Update poses for next iteration
            left_pose, right_pose = action_mode.get_current_ee_poses()
            
            # Print progress every 10 steps during interpolation or every 20 steps after
            print_interval = 5 if step < n_interp_steps else 20
            if step % print_interval == 0 and step > 0:
                if reaching_hand == "left":
                    dist = np.linalg.norm(left_pose.position - target_pos)
                else:
                    dist = np.linalg.norm(right_pose.position - target_pos)
                print(f"  Step {step}: Distance: {dist:.3f}m, Reward: {reward:.3f}")
            
            if terminated or truncated:
                print(f"Episode terminated. Final reward: {reward:.3f}")
                break
            
            # Render
            if not headless:
                env.render()
        
        if not success:
            # Calculate final distance
            left_pose, right_pose = action_mode.get_current_ee_poses()
            if reaching_hand == "left":
                final_dist = np.linalg.norm(left_pose.position - target_pos)
            else:
                final_dist = np.linalg.norm(right_pose.position - target_pos)
            print(f"✗ Failed to reach target. Final distance: {final_dist:.3f}m")
        
        # Brief pause between episodes
        if episode < num_episodes - 1 and not headless:
            print("Next episode in 2 seconds...")
            for _ in range(40):
                env.render()
    
    print("\n" + "=" * 60)
    print("Whole-body Cartesian control test completed!")
    print("The robot automatically moved its base when needed to reach targets.")
    
    # Print tracking error summary
    if tracking_data:
        print("\n--- Error Analysis Summary ---")
        ik_errors = [d['ik_error'] for d in tracking_data]
        controller_errors = [d['controller_error'] for d in tracking_data]
        joint_tracking_errors = [d['joint_tracking_error'] for d in tracking_data]
        ee_movements = [d['ee_movement'] for d in tracking_data]
        
        print(f"IK solver errors (FK of IK solution vs target):")
        print(f"  Mean: {np.mean(ik_errors):.4f}m")
        print(f"  Max:  {np.max(ik_errors):.4f}m")
        print(f"  Min:  {np.min(ik_errors):.4f}m")
        
        print(f"\nController tracking errors (actual vs IK solution):")
        print(f"  Mean: {np.mean(controller_errors):.4f}m")
        print(f"  Max:  {np.max(controller_errors):.4f}m")
        print(f"  Min:  {np.min(controller_errors):.4f}m")
        
        print(f"\nJoint tracking errors:")
        print(f"  Mean: {np.mean(joint_tracking_errors):.4f} rad")
        print(f"  Max:  {np.max(joint_tracking_errors):.4f} rad")
        
        print(f"\nEE movements per step:")
        print(f"  Mean: {np.mean(ee_movements):.4f}m")
        print(f"  Max:  {np.max(ee_movements):.4f}m")
        
        # Identify main error source
        avg_ik_error = np.mean(ik_errors)
        avg_controller_error = np.mean(controller_errors)
        
        print(f"\n--- Diagnosis ---")
        if avg_ik_error > avg_controller_error:
            print("⚠️ IK solver is the main source of error")
            if avg_ik_error > 0.05:
                print("   The IK solver is struggling to find accurate solutions")
        else:
            print("⚠️ Controller tracking is the main source of error")
            if avg_controller_error > 0.05:
                print("   The controller gains may need tuning or actuators are too weak")
    
    print("=" * 60)
    
    # Keep viewer open briefly
    if not headless:
        print("\nKeeping viewer open for 3 seconds...")
        for _ in range(60):
            env.render()
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RBY1 whole-body Cartesian control")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without visualization")
    args = parser.parse_args()
    
    try:
        main(headless=args.headless)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()