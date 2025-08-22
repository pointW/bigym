"""Test script for RBY1 whole-body IK solver."""

import sys
import os
import numpy as np
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bigym.envs.reach_target_rby1 import ReachTargetRBY1
from bigym.rby1_cartesian_action_mode import RBY1CartesianActionMode
from bigym.ik.rby1_whole_body_ik import RBY1WholeBodyIK
import mujoco


def main():
    """Test whole-body IK solver."""
    
    print("=" * 60)
    print("RBY1 Whole-Body IK Test")
    print("=" * 60)
    
    # Create environment
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionMode(),
        control_frequency=50,
        render_mode="human",
    )
    
    print("\nEnvironment created. Testing whole-body IK solver...")
    
    # Get MuJoCo model and data
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    
    # Create whole-body IK solver
    ik_solver = RBY1WholeBodyIK(model, data)
    
    # Run test episodes
    num_episodes = 5
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Reset environment
        obs, info = env.reset()
        
        # Get target position
        priv_obs = env._get_task_privileged_obs()
        target_pos = priv_obs["target_position"]
        
        print(f"Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # Get current qpos
        current_qpos = data.qpos.copy()
        print(f"Initial base position: [{current_qpos[0]:.3f}, {current_qpos[1]:.3f}, {current_qpos[2]:.3f}]")
        
        # Determine which hand to use based on target position
        if target_pos[1] > 0:
            # Target on left side, use left hand
            print("Using LEFT hand for reaching")
            left_target = target_pos
            right_target = None
        else:
            # Target on right side, use right hand
            print("Using RIGHT hand for reaching")
            left_target = None
            right_target = target_pos
        
        # Solve whole-body IK
        print("\nSolving whole-body IK...")
        solution, success, info = ik_solver.solve(
            left_target_pos=left_target,
            right_target_pos=right_target,
            current_qpos=current_qpos,
            max_iterations=100,
            tolerance=0.001,
        )
        
        if success:
            print(f"✓ IK solved successfully in {info['iterations']} iterations")
            print(f"  Final base position: [{solution[0]:.3f}, {solution[1]:.3f}, {solution[2]:.3f}]")
            print(f"  Base movement: ΔX={solution[0]-current_qpos[0]:.3f}, ΔY={solution[1]-current_qpos[1]:.3f}")
            print(f"  Stability margin: {info['stability_margin']:.3f}m")
            
            if "left_position_error" in info["errors"]:
                print(f"  Left EE error: {info['errors']['left_position_error']:.4f}m")
            if "right_position_error" in info["errors"]:
                print(f"  Right EE error: {info['errors']['right_position_error']:.4f}m")
            
            # Apply solution to environment
            print("\nApplying IK solution...")
            
            # Set the full qpos
            data.qpos[:] = solution
            mujoco.mj_forward(model, data)
            
            # Visualize for a moment
            for _ in range(50):
                env.render()
                time.sleep(0.02)
            
            # Check if target is reached
            if left_target is not None:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rby1/end_effector_l")
                actual_pos = data.site_xpos[site_id]
            else:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rby1/end_effector_r")
                actual_pos = data.site_xpos[site_id]
            
            final_error = np.linalg.norm(actual_pos - target_pos)
            print(f"  Actual final error: {final_error:.4f}m")
            
            if final_error < 0.05:
                print("  ✓ Target reached successfully!")
            else:
                print("  ⚠ Target not reached within threshold")
                
        else:
            print(f"✗ IK failed after {info['iterations']} iterations")
            if "left_position_error" in info["errors"]:
                print(f"  Left EE error: {info['errors']['left_position_error']:.4f}m")
            if "right_position_error" in info["errors"]:
                print(f"  Right EE error: {info['errors']['right_position_error']:.4f}m")
        
        # Brief pause between episodes
        if episode < num_episodes - 1:
            print("\nNext episode in 2 seconds...")
            for _ in range(40):
                env.render()
                time.sleep(0.05)
    
    print("\n" + "=" * 60)
    print("Whole-body IK test completed!")
    print("=" * 60)
    
    # Keep viewer open briefly
    print("\nKeeping viewer open for 3 seconds...")
    for _ in range(60):
        env.render()
        time.sleep(0.05)
    
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