"""Smooth visual test for RBY1 ReachTarget with proper arm coordination.

Similar to replay_demo.py but for RBY1 with Cartesian control.
"""

import sys
import os
import numpy as np
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bigym.envs.reach_target_rby1 import ReachTargetRBY1
from bigym.rby1_cartesian_action_mode import RBY1CartesianActionMode, rotation_matrix_to_6d
from bigym.const import HandSide


def determine_reaching_hand(target_pos):
    """Determine which hand should reach for the target based on position.
    
    Args:
        target_pos: Target position in world coordinates
        
    Returns:
        HandSide.LEFT or HandSide.RIGHT
    """
    # If target is on the left side (positive Y), use left hand
    # If target is on the right side (negative Y), use right hand
    # If centered, use closest hand based on X position
    if abs(target_pos[1]) < 0.1:  # Target is centered
        return HandSide.LEFT if target_pos[0] > 0.4 else HandSide.RIGHT
    else:
        return HandSide.LEFT if target_pos[1] > 0 else HandSide.RIGHT


def main():
    """Run smooth RBY1 reaching with proper arm coordination."""
    
    print("=" * 60)
    print("RBY1 Smooth Reaching Test")
    print("=" * 60)
    print("\nCreating environment...")
    
    # Create environment with higher control frequency for smoother motion
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionMode(),
        control_frequency=100,  # Higher frequency for smoother motion
        render_mode="human",  # Enable viewer
    )
    
    print("Environment created. MuJoCo viewer should be open.")
    print("\nViewer Controls:")
    print("- Left mouse: Rotate camera")
    print("- Right mouse: Move camera") 
    print("- Scroll: Zoom")
    print("- SPACE: Pause/unpause")
    print("- ESC: Exit")
    print("\n" + "=" * 60)
    
    # Get action mode
    action_mode = env.action_mode
    
    # Run episodes
    num_episodes = 20
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Reset environment
        obs, info = env.reset()
        
        # Get target position
        priv_obs = env._get_task_privileged_obs()
        target_pos = priv_obs["target_position"]
        
        # Determine which hand should reach
        reaching_hand = determine_reaching_hand(target_pos)
        reaching_hand_str = "LEFT" if reaching_hand == HandSide.LEFT else "RIGHT"
        
        print(f"Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"Reaching with: {reaching_hand_str} hand")
        
        # Get initial poses
        left_pose, right_pose = action_mode.get_current_ee_poses()
        
        # Store initial poses for the non-reaching hand
        if reaching_hand == HandSide.LEFT:
            reaching_pose_init = left_pose
            stable_pose_init = right_pose  # Store initial pose of stable hand
            reaching_idx = 0  # Left hand action indices
            stable_idx = 9  # Right hand action indices
        else:
            reaching_pose_init = right_pose
            stable_pose_init = left_pose  # Store initial pose of stable hand
            reaching_idx = 9  # Right hand action indices
            stable_idx = 0  # Left hand action indices
        
        # Track current poses for smooth interpolation
        stable_pose_current = stable_pose_init
        
        success = False
        max_steps = 200  # More steps to account for smoother motion
        
        # Parameters for smooth motion
        smoothing_factor = 0.4  # More aggressive to ensure reaching
        base_gain = 0.1  # Base movement gain
        stable_hand_smoothing = 0.05  # Stable hand correction
        
        for step in range(max_steps):
            # Create action
            action = np.zeros(env.action_space.shape[0])
            
            # Get current poses
            left_pose, right_pose = action_mode.get_current_ee_poses()
            
            if reaching_hand == HandSide.LEFT:
                reaching_pose_current = left_pose
            else:
                reaching_pose_current = right_pose
            
            # Smooth interpolation for reaching hand with adaptive gain
            reaching_error = target_pos - reaching_pose_current.position
            distance = np.linalg.norm(reaching_error)
            
            # Use adaptive smoothing: faster when far, even faster when very close to overcome IK errors
            if distance < 0.02:  # Very close, use direct target
                reaching_pos_target = target_pos  # Direct target when very close
            elif distance < 0.05:  # Getting close, speed up
                adaptive_smoothing = 0.8
                reaching_pos_target = reaching_pose_current.position + adaptive_smoothing * reaching_error
            elif distance < 0.1:  # Medium distance
                adaptive_smoothing = 0.6
                reaching_pos_target = reaching_pose_current.position + adaptive_smoothing * reaching_error
            else:  # Far away, use default
                reaching_pos_target = reaching_pose_current.position + smoothing_factor * reaching_error
            
            # Set reaching hand action
            action[reaching_idx:reaching_idx+3] = reaching_pos_target
            reaching_rot_6d = rotation_matrix_to_6d(reaching_pose_current.orientation.rotation_matrix)
            action[reaching_idx+3:reaching_idx+9] = np.clip(reaching_rot_6d, -1.0, 1.0)
            
            # Keep non-reaching hand stable with smooth interpolation to initial position
            # This maintains the hand's initial pose relative to the world frame
            if reaching_hand == HandSide.LEFT:
                # Keep right hand at its initial position with smooth correction
                current_right_pose = right_pose
                stable_error = stable_pose_init.position - current_right_pose.position
                stable_pose_target = current_right_pose.position + stable_hand_smoothing * stable_error
                action[stable_idx:stable_idx+3] = stable_pose_target
                stable_rot_6d = rotation_matrix_to_6d(stable_pose_init.orientation.rotation_matrix)
                action[stable_idx+3:stable_idx+9] = np.clip(stable_rot_6d, -1.0, 1.0)
            else:
                # Keep left hand at its initial position with smooth correction
                current_left_pose = left_pose
                stable_error = stable_pose_init.position - current_left_pose.position
                stable_pose_target = current_left_pose.position + stable_hand_smoothing * stable_error
                action[stable_idx:stable_idx+3] = stable_pose_target
                stable_rot_6d = rotation_matrix_to_6d(stable_pose_init.orientation.rotation_matrix)
                action[stable_idx+3:stable_idx+9] = np.clip(stable_rot_6d, -1.0, 1.0)
            
            # Base control - smooth movement toward target if needed
            target_dist_xy = np.linalg.norm(target_pos[:2])
            
            # Get current base estimate (from previous action or initial)
            if step == 0:
                base_x_current = 0.0
                base_y_current = 0.0
                base_rz_current = 0.0
            
            if target_dist_xy > 0.6:  # If target is far
                # Smoothly move base
                base_x_target = min(0.3, target_pos[0] - 0.5)
                base_y_target = target_pos[1] * 0.2
                base_rz_target = np.arctan2(target_pos[1], target_pos[0]) * 0.3
                
                base_x_current += base_gain * (base_x_target - base_x_current)
                base_y_current += base_gain * (base_y_target - base_y_current)
                base_rz_current += base_gain * (base_rz_target - base_rz_current)
            
            action[18] = np.clip(base_x_current, -2.0, 2.0)
            action[19] = np.clip(base_y_current, -2.0, 2.0)
            action[20] = np.clip(base_rz_current, -np.pi, np.pi)
            
            # Grippers (keep open)
            action[21] = 0.0
            action[22] = 0.0
            
            # Ensure action is within bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check success
            if info.get("task_success", False):
                success = True
                print(f"✓ Success in {step + 1} steps! Reward: {reward:.3f}")
                
                # Hold successful pose briefly
                for _ in range(30):
                    env.render()
                    time.sleep(0.02)
                break
            
            # Print progress every 50 steps
            if step % 50 == 0 and step > 0:
                if reaching_hand == HandSide.LEFT:
                    reaching_pos = left_pose.position
                else:
                    reaching_pos = right_pose.position
                dist_to_target = np.linalg.norm(reaching_pos - target_pos)
                print(f"  Step {step}: Distance to target: {dist_to_target:.3f}m, Reward: {reward:.3f}")
            
            if terminated or truncated:
                print(f"Episode terminated. Final reward: {reward:.3f}")
                break
            
            # Render for smooth visualization
            env.render()
            time.sleep(0.005)  # Smaller delay with higher control frequency
        
        if not success:
            # Calculate final distance
            reaching_hand_pos = left_pose.position if reaching_hand == HandSide.LEFT else right_pose.position
            final_dist = np.linalg.norm(reaching_hand_pos - target_pos)
            print(f"✗ Failed to reach target. Final distance: {final_dist:.3f}m")
        
        # Brief pause between episodes
        if episode < num_episodes - 1:
            print("Next episode in 1 second...")
            for _ in range(20):
                env.render()
                time.sleep(0.05)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("Keeping viewer open for 3 seconds...")
    print("=" * 60)
    
    # Keep viewer open briefly
    for _ in range(60):
        env.render()
        time.sleep(0.05)
    
    print("Closing environment...")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()