#!/usr/bin/env python3
"""Comprehensive debug of RBY1 demo replay to identify root cause of failure.

This script checks:
1. Seed initialization correctness
2. Action parsing by the controller
3. IK error using FK
4. Joint tracking error
5. Controller behavior
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from bigym.envs.reach_target import ReachTarget
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from demonstrations.demo import Demo
from bigym.robots.configs.rby1 import RBY1
from bigym.const import HandSide
from vr.ik.h1_upper_body_ik import Pose
from pyquaternion import Quaternion


def parse_cartesian_action(action: np.ndarray) -> Dict:
    """Parse RBY1 Cartesian action into components."""
    if len(action) != 20:
        raise ValueError(f"Expected 20D action, got {len(action)}D")
    
    return {
        'left_pos': action[0:3],
        'left_ori_6d': action[3:9],
        'right_pos': action[9:12],
        'right_ori_6d': action[12:18],
        'grippers': action[18:20]
    }


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to rotation matrix."""
    # Reshape to get the two rows
    row1 = rotation_6d[:3]
    row2 = rotation_6d[3:6]
    
    # Normalize first row
    x = row1 / np.linalg.norm(row1)
    
    # Make second row orthogonal to first and normalize
    y = row2 - np.dot(row2, x) * x
    y = y / np.linalg.norm(y)
    
    # Third row is cross product
    z = np.cross(x, y)
    
    # Stack rows to form matrix
    matrix = np.vstack([x, y, z])
    
    return matrix


def debug_single_step(
    env,
    action_mode,
    action: np.ndarray,
    step_idx: int,
    verbose: bool = True
) -> Dict:
    """Debug a single step of RBY1 demo replay.
    
    Returns detailed diagnostics for the step.
    """
    # Parse action
    parsed = parse_cartesian_action(action)
    
    # Get current state BEFORE action
    pre_qpos = env._mojo.physics.data.qpos.copy()
    pre_left_pose, pre_right_pose = action_mode.get_current_ee_poses()
    
    # Target poses from action
    target_left_pos = parsed['left_pos']
    target_right_pos = parsed['right_pos']
    
    # Convert 6D rotation back to quaternion for target poses
    left_rot_matrix = rotation_6d_to_matrix(parsed['left_ori_6d'])
    right_rot_matrix = rotation_6d_to_matrix(parsed['right_ori_6d'])
    
    target_left_quat = Quaternion(matrix=left_rot_matrix)
    target_right_quat = Quaternion(matrix=right_rot_matrix)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP {step_idx} DEBUG")
        print(f"{'='*60}")
        print("\n1. ACTION PARSING:")
        print(f"   Target left pos:  {target_left_pos}")
        print(f"   Target right pos: {target_right_pos}")
        print(f"   Grippers: {parsed['grippers']}")
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get state AFTER action
    post_qpos = env._mojo.physics.data.qpos.copy()
    post_left_pose, post_right_pose = action_mode.get_current_ee_poses()
    
    # Get achieved positions
    left_site = env.robot._wrist_sites[HandSide.LEFT]
    right_site = env.robot._wrist_sites[HandSide.RIGHT]
    achieved_left_pos = left_site.get_position()
    achieved_right_pos = right_site.get_position()
    
    # Calculate errors
    left_pos_error = np.linalg.norm(achieved_left_pos - target_left_pos)
    right_pos_error = np.linalg.norm(achieved_right_pos - target_right_pos)
    
    # Check if IK was called and get solution
    ik_solution = None
    ik_error = None
    
    # Try to get IK solution from action mode
    if hasattr(action_mode, 'get_last_ik_solution'):
        ik_solution, ik_info = action_mode.get_last_ik_solution()
        if ik_solution is not None:
            # Calculate IK error by doing FK on IK solution
            saved_qpos = env._mojo.physics.data.qpos.copy()
            env._mojo.physics.data.qpos[:] = ik_solution
            env._mojo.physics.forward()
            
            ik_left_pos = left_site.get_position()
            ik_right_pos = right_site.get_position()
            
            ik_left_error = np.linalg.norm(ik_left_pos - target_left_pos)
            ik_right_error = np.linalg.norm(ik_right_pos - target_right_pos)
            ik_error = (ik_left_error + ik_right_error) / 2
            
            # Restore actual qpos
            env._mojo.physics.data.qpos[:] = saved_qpos
            env._mojo.physics.forward()
    
    # Joint movement
    joint_movement = np.linalg.norm(post_qpos - pre_qpos)
    
    # Base movement (first 3 DOF)
    base_movement = np.linalg.norm(post_qpos[:3] - pre_qpos[:3])
    base_position = post_qpos[:3]
    
    if verbose:
        print("\n2. ACHIEVED POSITIONS:")
        print(f"   Achieved left pos:  {achieved_left_pos}")
        print(f"   Achieved right pos: {achieved_right_pos}")
        
        print("\n3. TRACKING ERRORS:")
        print(f"   Left pos error:  {left_pos_error*1000:.1f}mm")
        print(f"   Right pos error: {right_pos_error*1000:.1f}mm")
        
        if ik_error is not None:
            print(f"\n4. IK SOLVER:")
            print(f"   IK error (FK of solution): {ik_error*1000:.1f}mm")
            print(f"   IK left error:  {ik_left_error*1000:.1f}mm")
            print(f"   IK right error: {ik_right_error*1000:.1f}mm")
        else:
            print("\n4. IK SOLVER: No IK solution available")
        
        print(f"\n5. MOVEMENT:")
        print(f"   Total joint movement: {joint_movement:.4f} rad")
        print(f"   Base movement: {base_movement*1000:.1f}mm")
        print(f"   Base position: [{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}]")
        
        print(f"\n6. REWARD & SUCCESS:")
        print(f"   Reward: {reward:.3f}")
        print(f"   Task success: {info.get('task_success', False)}")
    
    return {
        'step': step_idx,
        'target_left_pos': target_left_pos,
        'target_right_pos': target_right_pos,
        'achieved_left_pos': achieved_left_pos,
        'achieved_right_pos': achieved_right_pos,
        'left_pos_error': left_pos_error,
        'right_pos_error': right_pos_error,
        'ik_error': ik_error,
        'joint_movement': joint_movement,
        'base_movement': base_movement,
        'base_position': base_position,
        'reward': reward,
        'success': info.get('task_success', False)
    }


def verify_seed_initialization(env, demo_seed: int) -> bool:
    """Verify that environment seed initialization is correct."""
    print("\nVERIFYING SEED INITIALIZATION:")
    print(f"Demo seed: {demo_seed}")
    
    # Reset with seed
    obs, info = env.reset(seed=demo_seed)
    
    # Get target position (should be deterministic with seed)
    if hasattr(env, '_get_task_privileged_obs'):
        priv_obs = env._get_task_privileged_obs()
        if 'target_position' in priv_obs:
            target_pos = priv_obs['target_position']
            print(f"Target position: {target_pos}")
    
    # Reset again with same seed - should get same target
    obs2, info2 = env.reset(seed=demo_seed)
    
    if hasattr(env, '_get_task_privileged_obs'):
        priv_obs2 = env._get_task_privileged_obs()
        if 'target_position' in priv_obs2:
            target_pos2 = priv_obs2['target_position']
            if np.allclose(target_pos, target_pos2):
                print("✅ Seed initialization is deterministic")
                return True
            else:
                print("❌ Seed initialization NOT deterministic!")
                return False
    
    print("⚠️ Could not verify seed initialization")
    return True


def test_action_mode_directly(env, action_mode):
    """Test the action mode directly with a simple target."""
    print("\nTESTING ACTION MODE DIRECTLY:")
    
    # Get current poses
    left_pose, right_pose = action_mode.get_current_ee_poses()
    print(f"Initial right pos: {right_pose.position}")
    
    # Create a simple target (move right hand forward 5cm)
    target_pos = right_pose.position + np.array([0.05, 0.0, 0.0])
    target_right_pose = Pose(target_pos, right_pose.orientation)
    
    print(f"Target right pos: {target_pos}")
    
    # Create action
    action = action_mode.poses_to_action(
        left_pose,
        target_right_pose,
        gripper_action=np.array([0.0, 0.0])
    )
    
    print(f"Generated action shape: {action.shape}")
    
    # Parse and verify action
    parsed = parse_cartesian_action(action)
    print(f"Parsed target right pos: {parsed['right_pos']}")
    
    if np.allclose(parsed['right_pos'], target_pos):
        print("✅ Action mode correctly encodes target position")
    else:
        print("❌ Action mode INCORRECTLY encodes target position!")
        print(f"   Expected: {target_pos}")
        print(f"   Got: {parsed['right_pos']}")
        print(f"   Difference: {np.linalg.norm(parsed['right_pos'] - target_pos)*1000:.1f}mm")


def main(headless: bool = False):
    """Main debug function."""
    print("RBY1 DEMO REPLAY COMPREHENSIVE DEBUG")
    print("=" * 80)
    
    # Load first RBY1 demo
    demo_dir = Path("rby1_cartesian_demos_reachtarget")
    demo_files = sorted(demo_dir.glob("rby1_cartesian_demo_*.safetensors"))
    
    if not demo_files:
        print("No RBY1 demos found!")
        return
    
    demo = Demo.from_safetensors(demo_files[0])
    print(f"Loaded demo with seed {demo.seed}, {len(demo.timesteps)} timesteps")
    
    # Create RBY1 environment
    env = ReachTarget(
        action_mode=RBY1CartesianActionModeWholeBody(),
        control_frequency=50,
        render_mode=None if headless else "human",
        robot_cls=RBY1
    )
    
    action_mode = env.action_mode
    
    # Test 1: Verify seed initialization
    seed_ok = verify_seed_initialization(env, demo.seed)
    
    # Test 2: Test action mode directly
    test_action_mode_directly(env, action_mode)
    
    # Reset for demo replay
    print("\n" + "=" * 80)
    print("STARTING DEMO REPLAY DEBUG")
    print("=" * 80)
    
    env.reset(seed=demo.seed)
    
    # Track cumulative errors
    all_results = []
    
    # Debug first 5 steps in detail
    for step_idx in range(min(5, len(demo.timesteps))):
        timestep = demo.timesteps[step_idx]
        
        # Get action
        action = timestep.info.get('demo_action')
        if action is None:
            action = timestep.executed_action
        if action is None:
            print(f"Step {step_idx}: No action found!")
            continue
        
        # Debug this step
        result = debug_single_step(env, action_mode, action, step_idx, verbose=True)
        all_results.append(result)
        
        if not headless:
            env.render()
        
        if result['success']:
            print(f"\n✅ SUCCESS at step {step_idx}!")
            break
    
    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    if all_results:
        avg_left_error = np.mean([r['left_pos_error'] for r in all_results])
        avg_right_error = np.mean([r['right_pos_error'] for r in all_results])
        avg_ik_error = np.mean([r['ik_error'] for r in all_results if r['ik_error'] is not None])
        
        print(f"\nAverage tracking errors:")
        print(f"  Left hand:  {avg_left_error*1000:.1f}mm")
        print(f"  Right hand: {avg_right_error*1000:.1f}mm")
        if not np.isnan(avg_ik_error):
            print(f"  IK solver:  {avg_ik_error*1000:.1f}mm")
        
        # Check if errors are consistently high
        if avg_left_error > 0.3 or avg_right_error > 0.3:
            print("\n⚠️ DIAGNOSIS: Large tracking errors detected!")
            
            # Check Z heights
            first_result = all_results[0]
            target_z = first_result['target_right_pos'][2]
            achieved_z = first_result['achieved_right_pos'][2]
            
            print(f"\nHeight analysis:")
            print(f"  Target Z:   {target_z:.3f}m")
            print(f"  Achieved Z: {achieved_z:.3f}m")
            print(f"  Z error:    {abs(target_z - achieved_z)*1000:.1f}mm")
            
            if abs(target_z - achieved_z) > 0.2:
                print("\n❌ ROOT CAUSE: Target height is unreachable for RBY1!")
                print("   RBY1 cannot reach the H1 demo target heights.")
        else:
            print("\n✅ Tracking errors are reasonable")
    
    if not headless:
        input("\nPress Enter to close...")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive RBY1 demo debug")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    
    main(headless=args.headless)