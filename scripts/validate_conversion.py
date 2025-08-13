"""Validate demo conversion: check if cartesian actions contain correct FK poses."""
import numpy as np
from pathlib import Path

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.const import HandSide
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from safetensors.numpy import load_file


def validate_conversion():
    """Validate that cartesian demo actions contain correct FK poses.
    
    This compares:
    1. Joint action → FK poses (ground truth)
    2. Cartesian action content (what we stored)
    
    Should have ZERO error if conversion is correct.
    """
    print("=" * 80)
    print("VALIDATING DEMO CONVERSION: FK POSES vs CARTESIAN ACTIONS")
    print("=" * 80)
    
    # Create ONLY joint environment
    joint_env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    # Load demos
    print("Loading demos...")
    
    # Load original joint demo
    demo_store = DemoStore()
    joint_metadata = Metadata.from_env(joint_env)
    joint_demos = demo_store.get_demos(joint_metadata, amount=1, frequency=50)
    joint_demo = joint_demos[0]
    
    # Load corresponding cartesian demo
    cartesian_demos_dir = Path("cartesian_demos_test")
    cartesian_demo_files = list(cartesian_demos_dir.glob("*.safetensors"))
    if not cartesian_demo_files:
        print("❌ No cartesian demos found!")
        return
    
    cartesian_demo_file = cartesian_demo_files[0]
    cartesian_demo_data = load_file(cartesian_demo_file)
    
    # Find action keys
    joint_actions = np.array([ts.executed_action for ts in joint_demo.timesteps])
    
    cartesian_action_key = None
    for key in cartesian_demo_data.keys():
        if 'action' in key.lower():
            cartesian_action_key = key
            break
    
    if not cartesian_action_key:
        print("❌ No action key found in cartesian demo!")
        return
    
    cartesian_actions = cartesian_demo_data[cartesian_action_key]
    
    print(f"✅ Loaded demos:")
    print(f"  Joint demo: {joint_actions.shape[0]} timesteps, action dim {joint_actions.shape[1]}")
    print(f"  Cartesian demo: {cartesian_actions.shape[0]} timesteps, action dim {cartesian_actions.shape[1]}")
    
    # Validate first several timesteps
    max_timesteps = min(10, joint_actions.shape[0], cartesian_actions.shape[0])
    print(f"\\nValidating first {max_timesteps} timesteps...")
    
    # Reset environment
    joint_env.reset()
    
    pose_errors = []
    base_errors = []
    prev_pelvis_pos = joint_env.robot.pelvis.get_position().copy()
    
    for step in range(max_timesteps):
        print(f"\\n--- TIMESTEP {step} ---")
        
        # Get actions for this timestep
        joint_action = joint_actions[step]
        cartesian_action = cartesian_actions[step]
        
        # Apply joint action to get FK poses (ground truth)
        joint_env.step(joint_action)
        
        # Get actual poses from FK
        left_site = joint_env.robot._wrist_sites[HandSide.LEFT]
        right_site = joint_env.robot._wrist_sites[HandSide.RIGHT]
        
        actual_left_pos = left_site.get_position()
        actual_right_pos = right_site.get_position()
        
        # Get actual pelvis movement
        current_pelvis_pos = joint_env.robot.pelvis.get_position()
        actual_base_movement = current_pelvis_pos - prev_pelvis_pos
        
        print(f"FK poses from joint action:")
        print(f"  Left:  {actual_left_pos}")
        print(f"  Right: {actual_right_pos}")
        print(f"  Base movement: {actual_base_movement}")
        
        # Parse cartesian action content
        idx = 0
        stored_left_pos = cartesian_action[idx:idx+3]
        idx += 3
        stored_left_ori = cartesian_action[idx:idx+6] 
        idx += 6
        stored_right_pos = cartesian_action[idx:idx+3]
        idx += 3
        stored_right_ori = cartesian_action[idx:idx+6]
        idx += 6
        stored_base_action = cartesian_action[idx:idx+3]
        idx += 3
        stored_grippers = cartesian_action[idx:]
        
        print(f"Stored in cartesian action:")
        print(f"  Left:  {stored_left_pos}")
        print(f"  Right: {stored_right_pos}")
        print(f"  Base action: {stored_base_action}")
        
        # Compare poses
        left_error = np.linalg.norm(actual_left_pos - stored_left_pos)
        right_error = np.linalg.norm(actual_right_pos - stored_right_pos)
        base_error = np.linalg.norm(actual_base_movement - stored_base_action)
        
        print(f"Validation errors:")
        print(f"  Left pose error:  {left_error:.6f}m ({left_error*1000:.1f}mm)")
        print(f"  Right pose error: {right_error:.6f}m ({right_error*1000:.1f}mm)")
        print(f"  Base action error: {base_error:.6f}m ({base_error*1000:.1f}mm)")
        
        pose_errors.append((left_error, right_error))
        base_errors.append(base_error)
        
        # Update previous pelvis position
        prev_pelvis_pos = current_pelvis_pos.copy()
    
    # Summary analysis
    print(f"\\n" + "=" * 80)
    print("CONVERSION VALIDATION SUMMARY")
    print("=" * 80)
    
    avg_left_error = np.mean([e[0] for e in pose_errors])
    avg_right_error = np.mean([e[1] for e in pose_errors])
    max_left_error = np.max([e[0] for e in pose_errors])
    max_right_error = np.max([e[1] for e in pose_errors])
    
    avg_base_error = np.mean(base_errors)
    max_base_error = np.max(base_errors)
    
    print(f"End-effector pose validation:")
    print(f"  Left EE  - Average: {avg_left_error*1000:.3f}mm, Max: {max_left_error*1000:.3f}mm")
    print(f"  Right EE - Average: {avg_right_error*1000:.3f}mm, Max: {max_right_error*1000:.3f}mm")
    
    print(f"\\nBase action validation:")
    print(f"  Average: {avg_base_error*1000:.3f}mm, Max: {max_base_error*1000:.3f}mm")
    
    # Check validation quality
    pose_threshold = 0.001  # 1mm tolerance
    base_threshold = 0.001  # 1mm tolerance
    
    pose_perfect = avg_left_error < pose_threshold and avg_right_error < pose_threshold
    base_perfect = avg_base_error < base_threshold
    
    print(f"\\nValidation results:")
    if pose_perfect and base_perfect:
        print("✅ PERFECT: Conversion is working correctly")
        print("   Cartesian actions exactly match FK poses and base movements")
    elif pose_perfect:
        print("👍 POSES GOOD: End-effector poses are correctly stored")
        print(f"⚠️  BASE ISSUES: Base actions have {avg_base_error*1000:.1f}mm average error")
    elif base_perfect:
        print(f"⚠️  POSE ISSUES: End-effector poses have {avg_left_error*1000:.1f}mm/{avg_right_error*1000:.1f}mm errors")
        print("👍 BASE GOOD: Base actions are correctly stored")
    else:
        print("❌ CONVERSION BROKEN: Both poses and base actions have errors")
        print("   Need to fix conversion algorithm")
    
    # Base movement analysis
    total_base_movement = np.sum([np.linalg.norm(stored_base_action) for stored_base_action in 
                                 [cartesian_actions[i][18:21] for i in range(max_timesteps)]])
    avg_base_magnitude = total_base_movement / max_timesteps
    
    print(f"\\nBase movement analysis:")
    print(f"  Average base action magnitude: {avg_base_magnitude:.6f}m")
    print(f"  Total base movement over {max_timesteps} steps: {total_base_movement:.6f}m")
    
    if avg_base_magnitude < 0.001:
        print("❌ MINIMAL BASE MOVEMENT: This will limit IK workspace")
        print("   Consider using tasks with more dynamic base movement")
    else:
        print("✅ SIGNIFICANT BASE MOVEMENT: Good for IK workspace")
    
    joint_env.close()


if __name__ == "__main__":
    validate_conversion()