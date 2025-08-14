#!/usr/bin/env python3
"""Compare action poses, Direct mode results, and Joint mode results.

Since Cartesian achieved demos were created by:
1. Running joint demos
2. Recording achieved joint positions
3. Computing FK to get Cartesian poses

The Cartesian action should match the Joint mode's achieved pose.
"""
import sys
sys.path.insert(0, '/Users/dian/Documents/projects/bigym')

import numpy as np
from pathlib import Path

from bigym.envs.reach_target import ReachTarget
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.cartesian_action_mode_direct import CartesianActionModeDirect
from bigym.const import HandSide
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo


def compare_action_and_poses(seed, max_steps=20):
    """Compare action, Direct result, and Joint result for a given seed."""
    
    # Find joint demo
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    demo_store = DemoStore()
    metadata = Metadata.from_env(env)
    joint_demos = demo_store.get_demos(metadata, amount=60, frequency=50)
    
    joint_demo = None
    for demo in joint_demos:
        if demo.seed == seed:
            joint_demo = demo
            break
    
    env.close()
    
    if not joint_demo:
        print(f"Joint demo with seed {seed} not found")
        return None
    
    # Find achieved demo
    achieved_demo = None
    for i in range(60):
        demo_path = Path(f"cartesian_demos/cartesian_demo_{i:03d}.safetensors")
        if demo_path.exists():
            demo = Demo.from_safetensors(demo_path)
            if demo and demo.seed == seed:
                achieved_demo = demo
                break
    
    if not achieved_demo:
        print(f"Achieved demo with seed {seed} not found")
        return None
    
    print(f"\n{'='*80}")
    print(f"COMPARING SEED {seed}")
    print(f"{'='*80}")
    print(f"Joint demo steps: {len(joint_demo.timesteps)}")
    print(f"Achieved demo steps: {len(achieved_demo.timesteps)}")
    
    # Setup environments
    env_joint = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    env_direct = ReachTarget(
        action_mode=CartesianActionModeDirect(
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        ),
        control_frequency=50,
        render_mode=None,
    )
    
    # Reset both environments with same seed
    env_joint.reset(seed=seed)
    env_direct.reset(seed=seed)
    
    errors = []
    
    for step_idx in range(min(max_steps, len(joint_demo.timesteps), len(achieved_demo.timesteps))):
        # Get actions from both demos
        joint_action = joint_demo.timesteps[step_idx].info.get('demo_action')
        cartesian_action = achieved_demo.timesteps[step_idx].info.get('demo_action')
        
        if joint_action is None or cartesian_action is None:
            continue
        
        # Fix Cartesian action dimension if needed
        if len(cartesian_action) == 23:
            fixed_action = np.zeros(24)
            fixed_action[:18] = cartesian_action[:18]
            fixed_action[18:21] = cartesian_action[18:21]
            fixed_action[21] = 0
            fixed_action[22:] = cartesian_action[21:]
            cartesian_action = fixed_action
        
        # Extract target poses from Cartesian action
        target_left_pos = cartesian_action[:3]
        target_right_pos = cartesian_action[9:12]
        
        # Execute joint action
        joint_action = np.clip(joint_action, env_joint.action_space.low, env_joint.action_space.high)
        obs_joint, reward_joint, term_joint, trunc_joint, info_joint = env_joint.step(joint_action)
        
        # Get resulting poses from Joint mode
        left_site = env_joint.robot._wrist_sites[HandSide.LEFT]
        right_site = env_joint.robot._wrist_sites[HandSide.RIGHT]
        joint_left_pos = left_site.get_position().copy()
        joint_right_pos = right_site.get_position().copy()
        
        # Execute Cartesian action in Direct mode
        cartesian_action = np.clip(cartesian_action, env_direct.action_space.low, env_direct.action_space.high)
        obs_direct, reward_direct, term_direct, trunc_direct, info_direct = env_direct.step(cartesian_action)
        
        # Get resulting poses from Direct mode
        direct_left, direct_right = env_direct.action_mode.get_current_ee_poses()
        direct_left_pos = direct_left.position.copy()
        direct_right_pos = direct_right.position.copy()
        
        # Calculate errors
        error_target_vs_joint = np.linalg.norm(target_left_pos - joint_left_pos)
        error_target_vs_direct = np.linalg.norm(target_left_pos - direct_left_pos)
        error_joint_vs_direct = np.linalg.norm(joint_left_pos - direct_left_pos)
        
        # Print detailed info for first few steps or large errors
        if step_idx < 3 or error_target_vs_joint > 10e-3:
            print(f"\nStep {step_idx + 1}:")
            print(f"  Target (from Cartesian action):")
            print(f"    Left:  {target_left_pos}")
            print(f"    Right: {target_right_pos}")
            print(f"  Joint mode result:")
            print(f"    Left:  {joint_left_pos}")
            print(f"    Right: {joint_right_pos}")
            print(f"  Direct mode result:")
            print(f"    Left:  {direct_left_pos}")
            print(f"    Right: {direct_right_pos}")
            print(f"  Errors (left hand):")
            print(f"    Target vs Joint:  {error_target_vs_joint*1000:.2f}mm")
            print(f"    Target vs Direct: {error_target_vs_direct*1000:.2f}mm")
            print(f"    Joint vs Direct:  {error_joint_vs_direct*1000:.2f}mm")
            
            if error_target_vs_joint > 10e-3:
                print(f"  ⚠️ Large discrepancy! Target should match Joint result!")
        
        errors.append({
            'target_vs_joint': error_target_vs_joint,
            'target_vs_direct': error_target_vs_direct,
            'joint_vs_direct': error_joint_vs_direct,
            'joint_success': info_joint.get('task_success', False),
            'direct_success': info_direct.get('task_success', False)
        })
        
        # Check for success
        if info_joint.get('task_success', False) and info_direct.get('task_success', False):
            print(f"\n✅ Both modes succeed at step {step_idx + 1}")
            break
        elif info_joint.get('task_success', False):
            print(f"\n✅ Joint succeeds, ❌ Direct fails at step {step_idx + 1}")
            break
        elif info_direct.get('task_success', False):
            print(f"\n❌ Joint fails, ✅ Direct succeeds at step {step_idx + 1}")
            break
        
        if term_joint or trunc_joint or term_direct or trunc_direct:
            print(f"\n❌ Episode terminated at step {step_idx + 1}")
            break
    
    env_joint.close()
    env_direct.close()
    
    # Summary statistics
    if errors:
        avg_target_vs_joint = np.mean([e['target_vs_joint'] for e in errors])
        avg_target_vs_direct = np.mean([e['target_vs_direct'] for e in errors])
        avg_joint_vs_direct = np.mean([e['joint_vs_direct'] for e in errors])
        
        print(f"\nSummary for seed {seed}:")
        print(f"  Average errors:")
        print(f"    Target vs Joint:  {avg_target_vs_joint*1000:.2f}mm")
        print(f"    Target vs Direct: {avg_target_vs_direct*1000:.2f}mm")
        print(f"    Joint vs Direct:  {avg_joint_vs_direct*1000:.2f}mm")
        
        if avg_target_vs_joint > 5e-3:
            print(f"\n  ⚠️ UNEXPECTED: Target should match Joint achieved pose!")
            print(f"     This suggests the Cartesian demo conversion has issues.")
    
    return errors


def main():
    """Run comparison for multiple seeds."""
    print("="*80)
    print("ACTION-POSE CONSISTENCY TEST")
    print("="*80)
    print("\nTesting whether:")
    print("1. Cartesian action (from achieved demo)")
    print("2. Direct mode result")
    print("3. Joint mode result (from original demo)")
    print("are consistent.")
    print("\nSince Cartesian actions were derived from Joint achieved poses,")
    print("the Cartesian action SHOULD match the Joint result exactly.")
    
    # Test a few seeds
    test_seeds = [
        3873497653,  # Known to succeed in all modes
        478760999,   # Known to fail in Cartesian modes
        1983862525,  # Another successful seed
    ]
    
    all_results = {}
    
    for seed in test_seeds:
        results = compare_action_and_poses(seed, max_steps=20)
        if results:
            all_results[seed] = results
    
    # Overall analysis
    print("\n" + "="*80)
    print("OVERALL ANALYSIS")
    print("="*80)
    
    for seed, errors in all_results.items():
        avg_target_vs_joint = np.mean([e['target_vs_joint'] for e in errors])
        avg_target_vs_direct = np.mean([e['target_vs_direct'] for e in errors])
        
        print(f"\nSeed {seed}:")
        print(f"  Target vs Joint:  {avg_target_vs_joint*1000:.2f}mm (should be ~0)")
        print(f"  Target vs Direct: {avg_target_vs_direct*1000:.2f}mm")
        
        if avg_target_vs_joint > 5e-3:
            print(f"  ⚠️ Issue with demo conversion!")
    
    print("\nConclusions:")
    print("- If Target matches Joint: Demo conversion is correct")
    print("- If Direct matches Joint: Direct mode is perfectly accurate")
    print("- If neither match: There are systematic errors in the pipeline")


if __name__ == "__main__":
    main()