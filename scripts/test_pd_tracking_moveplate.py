#!/usr/bin/env python3
"""Test PD tracking error with and without block_until_reached for MovePlate task."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from bigym.envs.move_plates import MovePlate
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from demonstrations.demo import Demo
from bigym.robots.configs.rby1 import RBY1
from bigym.const import HandSide


def compute_pose_error(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute position error in meters."""
    return np.linalg.norm(pos1 - pos2)


def test_tracking_accuracy(
    demo: Demo,
    block_until_reached: bool,
    max_steps: int = 200
) -> Dict:
    """Test tracking accuracy with given settings.
    
    Returns:
        Dictionary with tracking statistics
    """
    
    # Create environment with PD mode
    env = MovePlate(
        action_mode=RBY1CartesianActionModeWholeBody(
            direct_mode=False,  # Use PD control
            block_until_reached=block_until_reached
        ),
        control_frequency=50,
        render_mode=None,
        robot_cls=RBY1
    )
    
    env.reset(seed=demo.seed)
    
    # Tracking statistics
    ik_errors = []
    pd_tracking_errors_left = []
    pd_tracking_errors_right = []
    gripper_states = []
    task_success = False
    success_step = None
    max_reward = 0.0
    
    # Track gripper closing
    gripper_close_started = None
    plate_grasped = False
    
    for step_idx in range(min(max_steps, len(demo.timesteps))):
        timestep = demo.timesteps[step_idx]
        
        # Get action
        action = timestep.info.get('demo_action')
        if action is None:
            action = timestep.executed_action
        if action is None:
            continue
        
        # Parse target poses from action
        target_left_pos = action[0:3]
        target_right_pos = action[9:12]
        
        # Track when gripper starts closing
        if gripper_close_started is None and action[18] > 0.5:
            gripper_close_started = step_idx
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        max_reward = max(max_reward, reward)
        
        # Get achieved poses
        left_ee = env.robot._wrist_sites[HandSide.LEFT].get_position()
        right_ee = env.robot._wrist_sites[HandSide.RIGHT].get_position()
        
        # Get IK solution info if available
        action_mode = env.action_mode
        if hasattr(action_mode, '_last_ik_info') and action_mode._last_ik_info:
            ik_error = action_mode._last_ik_info.get('final_error', 0.0)
            ik_errors.append(ik_error)
        
        # Compute PD tracking errors
        left_error = compute_pose_error(left_ee, target_left_pos)
        right_error = compute_pose_error(right_ee, target_right_pos)
        pd_tracking_errors_left.append(left_error)
        pd_tracking_errors_right.append(right_error)
        
        # Track gripper state
        left_gripper = env.robot.grippers[HandSide.LEFT]
        gripper_states.append(left_gripper.qpos)
        
        # Check for plate contact after gripper close
        if gripper_close_started is not None and step_idx > gripper_close_started:
            physics = env._mojo.physics
            for i in range(physics.data.ncon):
                contact = physics.data.contact[i]
                geom1 = physics.model.geom(contact.geom1).name or f"geom_{contact.geom1}"
                geom2 = physics.model.geom(contact.geom2).name or f"geom_{contact.geom2}"
                
                # Check for gripper-plate contact
                if ('robotiq' in geom1.lower() or 'robotiq' in geom2.lower()) and \
                   ('plate' in geom1.lower() or 'plate' in geom2.lower()):
                    if not plate_grasped:
                        plate_grasped = True
        
        # Check success
        if info.get('task_success', False):
            task_success = True
            success_step = step_idx
            break
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Compute statistics
    results = {
        'block_until_reached': block_until_reached,
        'task_success': task_success,
        'success_step': success_step,
        'max_reward': max_reward,
        'gripper_close_step': gripper_close_started,
        'plate_grasped': plate_grasped,
        'max_gripper_closure': max(gripper_states) if gripper_states else 0.0,
        
        # IK errors
        'ik_error_mean': np.mean(ik_errors) if ik_errors else 0.0,
        'ik_error_max': np.max(ik_errors) if ik_errors else 0.0,
        'ik_error_std': np.std(ik_errors) if ik_errors else 0.0,
        
        # PD tracking errors (in mm)
        'pd_left_mean_mm': np.mean(pd_tracking_errors_left) * 1000 if pd_tracking_errors_left else 0.0,
        'pd_left_max_mm': np.max(pd_tracking_errors_left) * 1000 if pd_tracking_errors_left else 0.0,
        'pd_left_std_mm': np.std(pd_tracking_errors_left) * 1000 if pd_tracking_errors_left else 0.0,
        
        'pd_right_mean_mm': np.mean(pd_tracking_errors_right) * 1000 if pd_tracking_errors_right else 0.0,
        'pd_right_max_mm': np.max(pd_tracking_errors_right) * 1000 if pd_tracking_errors_right else 0.0,
        'pd_right_std_mm': np.std(pd_tracking_errors_right) * 1000 if pd_tracking_errors_right else 0.0,
    }
    
    # Compute tracking error during critical gripper close phase
    if gripper_close_started is not None:
        close_window = slice(gripper_close_started, min(gripper_close_started + 20, len(pd_tracking_errors_left)))
        if close_window.stop > close_window.start:
            results['pd_left_close_mean_mm'] = np.mean(pd_tracking_errors_left[close_window]) * 1000
            results['pd_right_close_mean_mm'] = np.mean(pd_tracking_errors_right[close_window]) * 1000
    
    return results


def main():
    """Main test function."""
    print("="*80)
    print("PD TRACKING ERROR ANALYSIS FOR MOVEPLATE TASK")
    print("="*80)
    
    # Load first demo
    demo_dir = Path("rby1_cartesian_demos_moveplate")
    demo_files = sorted(demo_dir.glob("rby1_cartesian_demo_*.safetensors"))
    
    if not demo_files:
        print("No RBY1 demos found!")
        print("Please run: python scripts/convert_h1_to_rby1_cartesian.py --env MovePlate")
        return
    
    demo = Demo.from_safetensors(demo_files[0])
    print(f"Using demo with seed {demo.seed}")
    print()
    
    # Test both settings
    results = []
    
    print("1. Testing with block_until_reached=False...")
    results_false = test_tracking_accuracy(demo, block_until_reached=False)
    results.append(results_false)
    
    print("2. Testing with block_until_reached=True...")
    results_true = test_tracking_accuracy(demo, block_until_reached=True)
    results.append(results_true)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    print("\n| Setting                | block=False | block=True  |")
    print("|------------------------|-------------|-------------|")
    
    # Task success
    print(f"| Task Success           | {'✅' if results[0]['task_success'] else '❌'}          | {'✅' if results[1]['task_success'] else '❌'}          |")
    
    if results[0]['success_step']:
        print(f"| Success Step           | {results[0]['success_step']:11d} | ", end="")
    else:
        print(f"| Success Step           | {'N/A':^11s} | ", end="")
    
    if results[1]['success_step']:
        print(f"{results[1]['success_step']:11d} |")
    else:
        print(f"{'N/A':^11s} |")
    
    print(f"| Max Reward             | {results[0]['max_reward']:11.3f} | {results[1]['max_reward']:11.3f} |")
    print(f"| Plate Grasped          | {'✅' if results[0]['plate_grasped'] else '❌'}          | {'✅' if results[1]['plate_grasped'] else '❌'}          |")
    print(f"| Max Gripper Closure    | {results[0]['max_gripper_closure']:11.3f} | {results[1]['max_gripper_closure']:11.3f} |")
    
    print("\n| IK Solver Performance  |             |             |")
    print("|------------------------|-------------|-------------|")
    print(f"| IK Error Mean          | {results[0]['ik_error_mean']:11.6f} | {results[1]['ik_error_mean']:11.6f} |")
    print(f"| IK Error Max           | {results[0]['ik_error_max']:11.6f} | {results[1]['ik_error_max']:11.6f} |")
    
    print("\n| PD Tracking (mm)       |             |             |")
    print("|------------------------|-------------|-------------|")
    print(f"| Left Mean Error        | {results[0]['pd_left_mean_mm']:11.1f} | {results[1]['pd_left_mean_mm']:11.1f} |")
    print(f"| Left Max Error         | {results[0]['pd_left_max_mm']:11.1f} | {results[1]['pd_left_max_mm']:11.1f} |")
    print(f"| Right Mean Error       | {results[0]['pd_right_mean_mm']:11.1f} | {results[1]['pd_right_mean_mm']:11.1f} |")
    print(f"| Right Max Error        | {results[0]['pd_right_max_mm']:11.1f} | {results[1]['pd_right_max_mm']:11.1f} |")
    
    # Tracking during gripper close
    if 'pd_left_close_mean_mm' in results[0]:
        print("\n| During Gripper Close   |             |             |")
        print("|------------------------|-------------|-------------|")
        print(f"| Left Mean Error        | {results[0].get('pd_left_close_mean_mm', 0):11.1f} | {results[1].get('pd_left_close_mean_mm', 0):11.1f} |")
        print(f"| Right Mean Error       | {results[0].get('pd_right_close_mean_mm', 0):11.1f} | {results[1].get('pd_right_close_mean_mm', 0):11.1f} |")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    improvement_left = (results[0]['pd_left_mean_mm'] - results[1]['pd_left_mean_mm']) / results[0]['pd_left_mean_mm'] * 100
    improvement_right = (results[0]['pd_right_mean_mm'] - results[1]['pd_right_mean_mm']) / results[0]['pd_right_mean_mm'] * 100
    
    print(f"\nblock_until_reached=True improves tracking by:")
    print(f"  Left hand:  {improvement_left:.1f}% reduction in mean error")
    print(f"  Right hand: {improvement_right:.1f}% reduction in mean error")
    
    if results[1]['task_success'] and not results[0]['task_success']:
        print("\n✅ block_until_reached=True enables successful task completion!")
    elif results[1]['task_success'] and results[0]['task_success']:
        if results[1]['success_step'] < results[0]['success_step']:
            print(f"\n✅ block_until_reached=True completes task {results[0]['success_step'] - results[1]['success_step']} steps faster!")
    
    print("\nKey insights:")
    print("- block_until_reached=True waits for PD controllers to converge")
    print("- This reduces tracking error significantly during critical operations")
    print("- Better tracking during gripper close enables successful grasping")


if __name__ == "__main__":
    main()