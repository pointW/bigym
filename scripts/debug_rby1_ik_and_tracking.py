#!/usr/bin/env python3
"""Debug RBY1 demo replay focusing on IK, tracking, and control modes.

This script tests:
1. IK correctness at each step
2. Direct mode vs PD controller mode
3. Whether the issue is IK, tracking, or something else
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from bigym.envs.reach_target import ReachTarget
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from demonstrations.demo import Demo
from bigym.robots.configs.rby1 import RBY1
from bigym.const import HandSide
import mujoco

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


def test_ik_solution(env, target_left_pos, target_right_pos, target_left_quat=None, target_right_quat=None):
    """Test if IK can find a solution for the given targets."""
    
    # Import the IK solver
    from bigym.ik.rby1_whole_body_ik import RBY1WholeBodyIK
    from pyquaternion import Quaternion
    
    # Create IK solver
    ik_solver = RBY1WholeBodyIK(
        env.robot._mojo.physics.model.ptr,
        env.robot._mojo.physics.data.ptr
    )
    
    # Get current qpos
    current_qpos = env.robot._mojo.physics.data.qpos.copy()
    
    # Solve IK
    solution_qpos, success, info = ik_solver.solve(
        left_target_pos=target_left_pos,
        left_target_quat=target_left_quat,
        right_target_pos=target_right_pos,
        right_target_quat=target_right_quat,
        current_qpos=current_qpos,
        max_iterations=100,
        tolerance=1e-3,
    )
    
    # Test the solution by applying it
    if success:
        # Save current state
        saved_qpos = env.robot._mojo.physics.data.qpos.copy()
        
        # Apply IK solution
        env.robot._mojo.physics.data.qpos[:] = solution_qpos
        env.robot._mojo.physics.forward()
        
        # Check achieved positions and orientations
        left_site = env.robot._wrist_sites[HandSide.LEFT]
        right_site = env.robot._wrist_sites[HandSide.RIGHT]
        
        achieved_left = left_site.get_position()
        achieved_right = right_site.get_position()
        achieved_left_quat = left_site.get_quaternion()  # [w, x, y, z]
        achieved_right_quat = right_site.get_quaternion()
        
        left_pos_error = np.linalg.norm(achieved_left - target_left_pos)
        right_pos_error = np.linalg.norm(achieved_right - target_right_pos)
        
        # Calculate rotation errors if target quaternions provided
        left_rot_error = None
        right_rot_error = None
        if target_left_quat is not None:
            q_target = Quaternion(target_left_quat)
            q_achieved = Quaternion(achieved_left_quat)
            q_diff = q_target * q_achieved.inverse
            left_rot_error = 2 * np.arccos(np.clip(abs(q_diff.w), -1, 1)) * 180 / np.pi  # Convert to degrees
        
        if target_right_quat is not None:
            q_target = Quaternion(target_right_quat)
            q_achieved = Quaternion(achieved_right_quat)
            q_diff = q_target * q_achieved.inverse
            right_rot_error = 2 * np.arccos(np.clip(abs(q_diff.w), -1, 1)) * 180 / np.pi
        
        # Restore original state
        env.robot._mojo.physics.data.qpos[:] = saved_qpos
        env.robot._mojo.physics.forward()
        
        return {
            'success': True,
            'left_pos_error': left_pos_error,
            'right_pos_error': right_pos_error,
            'left_rot_error': left_rot_error,
            'right_rot_error': right_rot_error,
            'iterations': info['iterations'],
            'solution_qpos': solution_qpos
        }
    else:
        return {
            'success': False,
            'errors': info.get('errors', {}),
            'iterations': info['iterations']
        }


def test_direct_mode_tracking(env, action, verbose=True):
    """Test if using direct mode (no PD controller) helps."""
    
    from bigym.rby1_cartesian_action_mode_whole_body import rotation_6d_to_matrix
    from pyquaternion import Quaternion
    
    # Parse action to get targets
    parsed = parse_cartesian_action(action)
    target_left_pos = parsed['left_pos']
    target_right_pos = parsed['right_pos']
    
    # Convert 6D rotations to quaternions
    left_rot_matrix = rotation_6d_to_matrix(parsed['left_ori_6d'])
    right_rot_matrix = rotation_6d_to_matrix(parsed['right_ori_6d'])
    
    left_quat = Quaternion(matrix=left_rot_matrix, atol=1e-6, rtol=1e-6)
    right_quat = Quaternion(matrix=right_rot_matrix, atol=1e-6, rtol=1e-6)
    
    target_left_quat = np.array([left_quat.w, left_quat.x, left_quat.y, left_quat.z])
    target_right_quat = np.array([right_quat.w, right_quat.x, right_quat.y, right_quat.z])
    
    if verbose:
        print("\nTESTING DIRECT MODE (bypassing PD controller):")
        print(f"  Target left pos:  {target_left_pos}")
        print(f"  Target right pos: {target_right_pos}")
        # Convert to euler for readability
        from scipy.spatial.transform import Rotation
        left_euler = Rotation.from_quat([left_quat.x, left_quat.y, left_quat.z, left_quat.w]).as_euler('xyz', degrees=True)
        right_euler = Rotation.from_quat([right_quat.x, right_quat.y, right_quat.z, right_quat.w]).as_euler('xyz', degrees=True)
        print(f"  Target left euler:  {left_euler}")
        print(f"  Target right euler: {right_euler}")
    
    # Test IK solution with orientations
    ik_result = test_ik_solution(env, target_left_pos, target_right_pos, target_left_quat, target_right_quat)
    
    if ik_result['success']:
        if verbose:
            print(f"  ✅ IK found solution in {ik_result['iterations']} iterations")
            print(f"     Left pos error:  {ik_result['left_pos_error']*1000:.2f}mm")
            print(f"     Right pos error: {ik_result['right_pos_error']*1000:.2f}mm")
            if ik_result.get('left_rot_error') is not None:
                print(f"     Left rot error:  {ik_result['left_rot_error']:.1f}°")
            if ik_result.get('right_rot_error') is not None:
                print(f"     Right rot error: {ik_result['right_rot_error']:.1f}°")
        
        # Apply the IK solution directly
        env.robot._mojo.physics.data.qpos[:] = ik_result['solution_qpos']
        env.robot._mojo.physics.forward()
        
        # Now properly step the environment to get actual task success
        # Since we're in direct mode, we need to simulate what the env would do
        # The action has already been applied via qpos, so step with zero action
        # This will trigger proper reward and task success computation
        _, reward, _, _, info = env.step(np.zeros_like(action))
        
        # Get current positions for debug output
        left_site = env.robot._wrist_sites[HandSide.LEFT]
        right_site = env.robot._wrist_sites[HandSide.RIGHT]
        
        achieved_left = left_site.get_position()
        achieved_right = right_site.get_position()
        
        # Get target for distance calculation
        target_pos = None
        if hasattr(env, '_get_task_privileged_obs'):
            priv_obs = env._get_task_privileged_obs()
            if 'target_position' in priv_obs:
                target_pos = priv_obs['target_position']
        
        if target_pos is not None and verbose:
            left_to_target = np.linalg.norm(achieved_left - target_pos)
            right_to_target = np.linalg.norm(achieved_right - target_pos)
            min_dist = min(left_to_target, right_to_target)
            
            print(f"  Task target: {target_pos}")
            print(f"  Achieved positions:")
            print(f"    Left:  {achieved_left} (dist: {left_to_target:.3f}m)")
            print(f"    Right: {achieved_right} (dist: {right_to_target:.3f}m)")
            print(f"  Min distance to target: {min_dist:.3f}m")
        
        if verbose:
            print(f"  After stepping with IK solution:")
            print(f"    Reward: {reward:.3f}")
            print(f"    Task success: {info.get('task_success', False)}")
        
        # NOTE: We do NOT restore the original state here - we want to keep moving forward
        # The IK solution has been applied and we continue from this new state
        
        return {
            'ik_success': True,
            'ik_pos_error': (ik_result['left_pos_error'] + ik_result['right_pos_error']) / 2,
            'ik_rot_error': (
                (ik_result.get('left_rot_error', 0) + ik_result.get('right_rot_error', 0)) / 2 
                if ik_result.get('left_rot_error') is not None else None
            ),
            'reward': reward,
            'task_success': info.get('task_success', False)
        }
    else:
        if verbose:
            print(f"  ❌ IK failed after {ik_result['iterations']} iterations")
            print(f"     Errors: {ik_result.get('errors', {})}")
        return {
            'ik_success': False,
            'ik_pos_error': None,
            'ik_rot_error': None,
            'reward': None,
            'task_success': False
        }


def analyze_step_failure(env, action, step_idx, verbose=True):
    """Comprehensive analysis of why a step fails."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP {step_idx} ANALYSIS")
        print(f"{'='*60}")
    
    # Parse action
    parsed = parse_cartesian_action(action)
    target_left_pos = parsed['left_pos']
    target_right_pos = parsed['right_pos']
    
    # Get current state
    left_site = env.robot._wrist_sites[HandSide.LEFT]
    right_site = env.robot._wrist_sites[HandSide.RIGHT]
    current_left = left_site.get_position()
    current_right = right_site.get_position()
    
    # 1. Check if targets are reasonable
    left_dist = np.linalg.norm(target_left_pos - current_left)
    right_dist = np.linalg.norm(target_right_pos - current_right)
    
    if verbose:
        print("\n1. TARGET ANALYSIS:")
        print(f"   Current left:  {current_left}")
        print(f"   Target left:   {target_left_pos}")
        print(f"   Distance:      {left_dist*1000:.1f}mm")
        print(f"   Current right: {current_right}")
        print(f"   Target right:  {target_right_pos}")
        print(f"   Distance:      {right_dist*1000:.1f}mm")
    
    # 2. Test IK solution
    ik_result = test_ik_solution(env, target_left_pos, target_right_pos)
    
    if verbose:
        print("\n2. IK SOLVER TEST:")
        if ik_result['success']:
            print(f"   ✅ IK successful ({ik_result['iterations']} iterations)")
            print(f"   Left pos error:  {ik_result.get('left_pos_error', 0)*1000:.2f}mm")
            print(f"   Right pos error: {ik_result.get('right_pos_error', 0)*1000:.2f}mm")
            if ik_result.get('left_rot_error') is not None:
                print(f"   Left rot error:  {ik_result['left_rot_error']:.1f}°")
            if ik_result.get('right_rot_error') is not None:
                print(f"   Right rot error: {ik_result['right_rot_error']:.1f}°")
        else:
            print(f"   ❌ IK failed ({ik_result['iterations']} iterations)")
    
    # 3. Test with actual step
    _ = env.robot._mojo.physics.data.qpos.copy()  # Save state if needed
    
    _, reward, _, _, info = env.step(action)
    
    # Get achieved positions
    achieved_left = left_site.get_position()
    achieved_right = right_site.get_position()
    
    left_tracking_error = np.linalg.norm(achieved_left - target_left_pos)
    right_tracking_error = np.linalg.norm(achieved_right - target_right_pos)
    
    if verbose:
        print("\n3. PD CONTROLLER TRACKING:")
        print(f"   Achieved left:  {achieved_left}")
        print(f"   Left error:     {left_tracking_error*1000:.1f}mm")
        print(f"   Achieved right: {achieved_right}")
        print(f"   Right error:    {right_tracking_error*1000:.1f}mm")
        print(f"   Reward:         {reward:.3f}")
        print(f"   Task success:   {info.get('task_success', False)}")
    
    # 4. Check workspace limits
    base_pos = env.robot._mojo.physics.data.qpos[:3]
    
    if verbose:
        print("\n4. WORKSPACE ANALYSIS:")
        print(f"   Base position: {base_pos}")
        print(f"   Base movement: {np.linalg.norm(base_pos)*1000:.1f}mm from origin")
        
        # Check if targets are within RBY1 workspace
        # RBY1 has scale=1.3, so targets designed for H1 might be out of reach
        target_height_left = target_left_pos[2]
        target_height_right = target_right_pos[2]
        
        print(f"   Target heights: L={target_height_left:.3f}m, R={target_height_right:.3f}m")
        
        # Rough workspace check
        if target_height_left > 1.8 or target_height_right > 1.8:
            print("   ⚠️ Targets might be above RBY1 workspace!")
        if target_height_left < 0.3 or target_height_right < 0.3:
            print("   ⚠️ Targets might be below RBY1 workspace!")
    
    return {
        'step': step_idx,
        'ik_success': ik_result['success'],
        'ik_pos_error': ik_result.get('left_pos_error', 0) + ik_result.get('right_pos_error', 0) if ik_result['success'] else None,
        'ik_rot_error': None,  # Not calculated in this function yet
        'tracking_error_left': left_tracking_error,
        'tracking_error_right': right_tracking_error,
        'reward': reward,
        'task_success': info.get('task_success', False),
        'base_movement': np.linalg.norm(base_pos)
    }


def main(headless: bool = False, use_direct_mode: bool = False):
    """Main debug function.
    
    Args:
        headless: Run without visualization
        use_direct_mode: Test with direct IK application (bypass PD controller)
    """
    
    print("RBY1 IK AND TRACKING DEBUG")
    print("=" * 80)
    
    # Load first RBY1 demo
    demo_dir = Path("rby1_cartesian_demos_reachtarget")
    demo_files = sorted(demo_dir.glob("rby1_cartesian_demo_*.safetensors"))
    
    if not demo_files:
        # Try H1 demos as fallback
        print("No RBY1 demos found, trying H1 demos...")
        demo_dir = Path("demonstrations/h1_demos/ReachTarget")
        demo_files = sorted(demo_dir.glob("*.safetensors"))
        
        if not demo_files:
            print("No demos found!")
            return
    
    demo = Demo.from_safetensors(demo_files[0])
    print(f"Loaded demo: {demo_files[0].name}")
    print(f"  Seed: {demo.seed}")
    print(f"  Timesteps: {len(demo.timesteps)}")
    
    # Create RBY1 environment
    env = ReachTarget(
        action_mode=RBY1CartesianActionModeWholeBody(direct_mode=use_direct_mode),
        control_frequency=50,
        render_mode=None if headless else "human",
        robot_cls=RBY1
    )
    
    # Reset with demo seed
    env.reset(seed=demo.seed)
    if not headless:
        env.render()
    
    print("\n" + "="*80)
    print("STARTING ANALYSIS")
    print("="*80)
    
    results = []
    success_step = -1
    
    # Analyze first 10 steps
    for step_idx in range(min(50, len(demo.timesteps))):
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
        
        if use_direct_mode:
            # Parse action to get target positions and orientations
            parsed = parse_cartesian_action(action)
            target_left_pos = parsed['left_pos']
            target_right_pos = parsed['right_pos']
            
            # Convert 6D rotations to quaternions for comparison
            from bigym.rby1_cartesian_action_mode_whole_body import rotation_6d_to_matrix
            from pyquaternion import Quaternion
            
            left_rot_matrix = rotation_6d_to_matrix(parsed['left_ori_6d'])
            right_rot_matrix = rotation_6d_to_matrix(parsed['right_ori_6d'])
            
            left_quat = Quaternion(matrix=left_rot_matrix, atol=1e-6, rtol=1e-6)
            right_quat = Quaternion(matrix=right_rot_matrix, atol=1e-6, rtol=1e-6)
            
            # In direct mode, just step the environment normally
            # The action mode will handle IK internally
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get current positions for debug
            left_site = env.robot._wrist_sites[HandSide.LEFT]
            right_site = env.robot._wrist_sites[HandSide.RIGHT]
            left_pos = left_site.get_position()
            right_pos = right_site.get_position()
            left_quat_achieved = left_site.get_quaternion()  # [w, x, y, z]
            right_quat_achieved = right_site.get_quaternion()
            
            # Calculate IK errors
            left_pos_error = np.linalg.norm(left_pos - target_left_pos)
            right_pos_error = np.linalg.norm(right_pos - target_right_pos)
            
            # Calculate rotation errors
            q_left_achieved = Quaternion(left_quat_achieved)
            q_diff_left = left_quat * q_left_achieved.inverse
            left_rot_error = 2 * np.arccos(np.clip(abs(q_diff_left.w), -1, 1)) * 180 / np.pi
            
            q_right_achieved = Quaternion(right_quat_achieved)
            q_diff_right = right_quat * q_right_achieved.inverse
            right_rot_error = 2 * np.arccos(np.clip(abs(q_diff_right.w), -1, 1)) * 180 / np.pi
            
            # Get target
            target_pos = None
            if hasattr(env, '_get_task_privileged_obs'):
                priv_obs = env._get_task_privileged_obs()
                if 'target_position' in priv_obs:
                    target_pos = priv_obs['target_position']
            
            # Only print every 10 steps or on success
            if step_idx % 10 == 0 or info.get('task_success', False):
                print(f"\nStep {step_idx} (Direct Mode):")
                print(f"  Target left:  {target_left_pos}")
                print(f"  Achieved left:  {left_pos}")
                print(f"  Left pos error: {left_pos_error*1000:.1f}mm, rot error: {left_rot_error:.1f}°")
                print(f"  Target right: {target_right_pos}")
                print(f"  Achieved right: {right_pos}")
                print(f"  Right pos error: {right_pos_error*1000:.1f}mm, rot error: {right_rot_error:.1f}°")
                if target_pos is not None:
                    left_dist = np.linalg.norm(left_pos - target_pos)
                    right_dist = np.linalg.norm(right_pos - target_pos)
                    print(f"  Distance to task target: L={left_dist:.3f}m, R={right_dist:.3f}m")
                print(f"  Reward: {reward:.3f}")
                print(f"  Task success: {info.get('task_success', False)}")
            
            if info.get('task_success', False):
                print(f"\n✅ SUCCESS with direct mode at step {step_idx}!")
                success_step = step_idx
                break
            
            if terminated or truncated:
                print(f"\nEpisode ended at step {step_idx}")
                break
        else:
            # Comprehensive analysis
            result = analyze_step_failure(env, action, step_idx, verbose=True)
            results.append(result)
            
            if result['task_success']:
                print(f"\n✅ SUCCESS at step {step_idx}!")
                success_step = step_idx
                break
        
        if not headless:
            env.render()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        # IK success rate
        ik_successes = sum(1 for r in results if r['ik_success'])
        print(f"\nIK Success Rate: {ik_successes}/{len(results)} ({ik_successes/len(results)*100:.1f}%)")
        
        # Average tracking errors
        avg_left_error = np.mean([r['tracking_error_left'] for r in results])
        avg_right_error = np.mean([r['tracking_error_right'] for r in results])
        print(f"\nAverage Tracking Errors:")
        print(f"  Left hand:  {avg_left_error*1000:.1f}mm")
        print(f"  Right hand: {avg_right_error*1000:.1f}mm")
        
        # Base movement
        max_base_movement = max(r['base_movement'] for r in results)
        print(f"\nMax base movement: {max_base_movement*1000:.1f}mm")
        
        # Diagnosis
        print("\n" + "="*80)
        print("DIAGNOSIS:")
        print("="*80)
        
        if ik_successes == 0:
            print("❌ IK is failing - targets might be unreachable for RBY1")
            print("   Possible causes:")
            print("   - Targets designed for H1 are out of RBY1's workspace")
            print("   - Scale mismatch (RBY1 uses 1.3x scale)")
        elif ik_successes == len(results) and avg_left_error < 0.01 and avg_right_error < 0.01:
            print("✅ IK works perfectly, tracking is good")
            print("   The issue might be with task success criteria")
        elif ik_successes == len(results) and (avg_left_error > 0.1 or avg_right_error > 0.1):
            print("⚠️ IK works but PD tracking has large errors")
            print("   Possible solutions:")
            print("   - Use direct mode (bypass PD controller)")
            print("   - Tune PD gains")
            print("   - Reduce control frequency")
        else:
            print("⚠️ Mixed results - some IK failures, some tracking issues")
    
    if success_step >= 0:
        print(f"\n🎉 Task completed successfully at step {success_step}!")
    else:
        print(f"\n❌ Task not completed in {len(results)} steps")
    
    if not headless:
        input("\nPress Enter to close...")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug RBY1 IK and tracking")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--direct", action="store_true", help="Test with direct IK mode")
    args = parser.parse_args()
    
    main(headless=args.headless, use_direct_mode=args.direct)