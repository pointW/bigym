"""Simple visual test for RBY1 ReachTarget with viewer.

Minimal test without camera observations.
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


def main():
    """Run RBY1 reaching with viewer."""
    
    print("=" * 60)
    print("RBY1 ReachTarget Visual Test")
    print("=" * 60)
    print("\nCreating environment with viewer...")
    
    # Create environment with human render mode (no cameras)
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionMode(floating_base=True),
        control_frequency=50,
        render_mode="human",  # This should open the viewer
    )
    
    print("Environment created. MuJoCo viewer should be open.")
    print("\nViewer Controls:")
    print("- Left mouse: Rotate camera")
    print("- Right mouse: Move camera") 
    print("- Scroll: Zoom")
    print("- Double-click: Center on object")
    print("- SPACE: Pause/unpause")
    print("- S: Show/hide contact forces")
    print("- F: Show/hide contact frames")
    print("- H: Show/hide help")
    print("- ESC: Exit")
    print("\n" + "=" * 60)
    
    # Get action mode
    action_mode = env.action_mode
    
    # Run episodes
    num_episodes = 10
    
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
        print(f"Initial L: [{left_pose.position[0]:.3f}, {left_pose.position[1]:.3f}, {left_pose.position[2]:.3f}]")
        print(f"Initial R: [{right_pose.position[0]:.3f}, {right_pose.position[1]:.3f}, {right_pose.position[2]:.3f}]")
        
        success = False
        max_steps = 50
        
        for step in range(max_steps):
            # Simple controller: move end-effectors toward target
            action = np.zeros(env.action_space.shape[0])
            
            # Proportional controller gains
            position_gain = 0.5
            
            # Left end-effector
            left_error = target_pos - left_pose.position
            action[0:3] = left_pose.position + position_gain * left_error
            left_rot_6d = rotation_matrix_to_6d(left_pose.orientation.rotation_matrix)
            action[3:9] = np.clip(left_rot_6d, -1.0, 1.0)
            
            # Right end-effector
            right_error = target_pos - right_pose.position
            action[9:12] = right_pose.position + position_gain * right_error
            right_rot_6d = rotation_matrix_to_6d(right_pose.orientation.rotation_matrix)
            action[12:18] = np.clip(right_rot_6d, -1.0, 1.0)
            
            # Base control
            # Move base if target is far
            target_dist_xy = np.linalg.norm(target_pos[:2])
            if target_dist_xy > 0.6:
                # Move base closer
                base_x = min(0.5, target_pos[0] - 0.4)
                base_y = target_pos[1] * 0.3
                base_rz = np.arctan2(target_pos[1], target_pos[0]) * 0.5
            else:
                base_x = 0.0
                base_y = 0.0
                base_rz = 0.0
            
            action[18] = np.clip(base_x, -2.0, 2.0)
            action[19] = np.clip(base_y, -2.0, 2.0)
            action[20] = np.clip(base_rz, -np.pi, np.pi)
            
            # Grippers (keep open)
            action[21] = 0.0
            action[22] = 0.0
            
            # Ensure action is within bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update poses for next iteration
            left_pose, right_pose = action_mode.get_current_ee_poses()
            
            # Check success
            if info.get("task_success", False):
                success = True
                print(f"✓ Success in {step + 1} steps! Reward: {reward:.3f}")
                
                # Hold successful pose briefly
                for _ in range(50):
                    env.render()
                    time.sleep(0.02)
                break
            
            if terminated or truncated:
                print(f"Episode terminated. Final reward: {reward:.3f}")
                break
            
            # Render periodically for smooth visualization
            if step % 2 == 0:
                env.render()
                time.sleep(0.01)
        
        if not success:
            # Calculate final distance
            left_dist = np.linalg.norm(left_pose.position - target_pos)
            right_dist = np.linalg.norm(right_pose.position - target_pos)
            min_dist = min(left_dist, right_dist)
            print(f"✗ Failed to reach target. Min distance: {min_dist:.3f}m")
        
        # Brief pause between episodes
        if episode < num_episodes - 1:
            print("Next episode starting...")
            time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("Viewer will remain open for 5 seconds...")
    print("=" * 60)
    
    # Keep viewer open briefly
    for _ in range(100):
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