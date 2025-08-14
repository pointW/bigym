#!/usr/bin/env python3
"""Test that converted demos with preserved seeds can be replayed correctly."""
import sys
import numpy as np
from pathlib import Path

from bigym.envs.reach_target import ReachTarget
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.cartesian_action_mode import CartesianActionMode
from bigym.cartesian_action_mode_direct import CartesianActionModeDirect
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo


def test_preserved_seeds(n_demos=60):
    """Test all conversion methods with preserved seeds.
    
    Args:
        n_demos: Number of demos to test (default 60). Will test first n_demos from joint,
                 then find matching seeds in other demo sets.
    """
    print("="*80)
    print(f"TESTING DEMOS WITH PRESERVED SEEDS (n={n_demos})")
    print("="*80)
    print("\nConverted demos should now contain the original seed values")
    print("This allows proper replay with correct target positions")
    print("-"*80)
    
    results = {}
    seed_results = {}  # Track results per seed
    
    # Test 1: Original Joint Actions (baseline)
    print("\n1. ORIGINAL JOINT ACTIONS (baseline):")
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    metadata = Metadata.from_env(env)
    demo_store = DemoStore()
    joint_demos = demo_store.get_demos(metadata, amount=n_demos, frequency=50)
    
    # Collect seeds we'll test
    test_seeds = [demo.seed for demo in joint_demos]
    
    successes = []
    for i, demo in enumerate(joint_demos):
        # Reset with demo's seed for correct target
        env.reset(seed=demo.seed)
        print(f"  Demo {i+1} (seed={demo.seed}):", end=" ")
        
        success = False
        max_reward = 0
        success_step = None
        
        for step_idx, timestep in enumerate(demo.timesteps):
            action = timestep.info.get('demo_action')
            if action is None:
                continue
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            max_reward = max(max_reward, reward)
            
            if info.get('task_success', False):
                success = True
                success_step = step_idx + 1
                print(f"✅ Success at step {step_idx+1}")
                break
            
            if terminated or truncated:
                break
        
        if not success:
            print(f"❌ Failed (max_reward={max_reward:.3f})")
        
        successes.append(success)
        # Store per-seed results
        if demo.seed not in seed_results:
            seed_results[demo.seed] = {}
        seed_results[demo.seed]['joint'] = (success, success_step if success else None)
    
    joint_sr = sum(successes) / len(successes) * 100 if successes else 0
    results["Joint Actions"] = joint_sr
    env.close()
    
    # Test 2: Cartesian Achieved with preserved seed
    print("\n2. CARTESIAN ACHIEVED (with preserved seed):")
    env = ReachTarget(
        action_mode=CartesianActionMode(
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        ),
        control_frequency=50,
        render_mode=None,
    )
    
    successes = []
    tested_count = 0
    for i in range(60):
        demo_path = Path(f"cartesian_demos/cartesian_demo_{i:03d}.safetensors")
        if not demo_path.exists():
            continue
        
        demo = Demo.from_safetensors(demo_path)
        if demo is None:
            continue
        
        # Only test if this seed is in our test set
        if demo.seed not in test_seeds:
            continue
            
        tested_count += 1
        print(f"  Demo {tested_count} (seed={demo.seed}):", end=" ")
        env.reset(seed=demo.seed)
        
        success = False
        max_reward = 0
        success_step = None
        
        for step_idx, timestep in enumerate(demo.timesteps):
            action = timestep.info.get('demo_action')
            if action is None:
                action = timestep.action  # Try using the action field directly
            if action is None:
                continue
            
            # Fix dimension if needed
            if len(action) == 23:
                fixed_action = np.zeros(24)
                fixed_action[:18] = action[:18]
                fixed_action[18:21] = action[18:21]
                fixed_action[21] = 0
                fixed_action[22:] = action[21:]
                action = fixed_action
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            max_reward = max(max_reward, reward)
            
            if info.get('task_success', False):
                success = True
                success_step = step_idx + 1
                print(f"✅ Success at step {step_idx+1}")
                break
            
            if terminated or truncated:
                break
        
        if not success:
            print(f"❌ Failed (max_reward={max_reward:.3f})")
        
        successes.append(success)
        # Store per-seed results
        if demo.seed not in seed_results:
            seed_results[demo.seed] = {}
        seed_results[demo.seed]['achieved'] = (success, success_step if success else None)
    
    achieved_sr = sum(successes) / len(successes) * 100 if successes else 0
    results["Cartesian Achieved"] = achieved_sr
    env.close()
    
    # Test 3: Cartesian Target with preserved seed
    print("\n3. CARTESIAN TARGET (with preserved seed):")
    env = ReachTarget(
        action_mode=CartesianActionMode(
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        ),
        control_frequency=50,
        render_mode=None,
    )
    
    successes = []
    tested_count = 0
    for i in range(60):
        demo_path = Path(f"cartesian_demos_target/cartesian_demo_{i:03d}.safetensors")
        if not demo_path.exists():
            continue
        
        demo = Demo.from_safetensors(demo_path)
        if demo is None:
            continue
        
        # Only test if this seed is in our test set
        if demo.seed not in test_seeds:
            continue
            
        tested_count += 1
        print(f"  Demo {tested_count} (seed={demo.seed}):", end=" ")
        env.reset(seed=demo.seed)
        
        success = False
        max_reward = 0
        success_step = None
        
        for step_idx, timestep in enumerate(demo.timesteps):
            action = timestep.info.get('demo_action')
            if action is None:
                continue
            
            # Fix dimension if needed
            if len(action) == 23:
                fixed_action = np.zeros(24)
                fixed_action[:18] = action[:18]
                fixed_action[18:21] = action[18:21]
                fixed_action[21] = 0
                fixed_action[22:] = action[21:]
                action = fixed_action
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            max_reward = max(max_reward, reward)
            
            if info.get('task_success', False):
                success = True
                success_step = step_idx + 1
                print(f"✅ Success at step {step_idx+1}")
                break
            
            if terminated or truncated:
                break
        
        if not success:
            print(f"❌ Failed (max_reward={max_reward:.3f})")
        
        successes.append(success)
        # Store per-seed results
        if demo.seed not in seed_results:
            seed_results[demo.seed] = {}
        seed_results[demo.seed]['target'] = (success, success_step if success else None)
    
    target_sr = sum(successes) / len(successes) * 100 if successes else 0
    results["Cartesian Target"] = target_sr
    env.close()
    
    # Test 4: Cartesian Direct (with achieved demos)
    print("\n4. CARTESIAN DIRECT (achieved demos with direct qpos control):")
    env = ReachTarget(
        action_mode=CartesianActionModeDirect(
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        ),
        control_frequency=50,
        render_mode=None,
    )
    
    successes = []
    tested_count = 0
    for i in range(60):
        demo_path = Path(f"cartesian_demos/cartesian_demo_{i:03d}.safetensors")
        if not demo_path.exists():
            continue
        
        demo = Demo.from_safetensors(demo_path)
        if demo is None:
            continue
        
        # Only test if this seed is in our test set
        if demo.seed not in test_seeds:
            continue
            
        tested_count += 1
        print(f"  Demo {tested_count} (seed={demo.seed}):", end=" ")
        env.reset(seed=demo.seed)
        
        success = False
        max_reward = 0
        success_step = None
        
        for step_idx, timestep in enumerate(demo.timesteps):
            action = timestep.info.get('demo_action')
            if action is None:
                action = timestep.action  # Try using the action field directly
            if action is None:
                continue
            
            # Fix dimension if needed
            if len(action) == 23:
                fixed_action = np.zeros(24)
                fixed_action[:18] = action[:18]
                fixed_action[18:21] = action[18:21]
                fixed_action[21] = 0
                fixed_action[22:] = action[21:]
                action = fixed_action
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            max_reward = max(max_reward, reward)
            
            if info.get('task_success', False):
                success = True
                success_step = step_idx + 1
                print(f"✅ Success at step {step_idx+1}")
                break
            
            if terminated or truncated:
                break
        
        if not success:
            print(f"❌ Failed (max_reward={max_reward:.3f})")
        
        successes.append(success)
        # Store per-seed results
        if demo.seed not in seed_results:
            seed_results[demo.seed] = {}
        seed_results[demo.seed]['direct'] = (success, success_step if success else None)
    
    direct_sr = sum(successes) / len(successes) * 100 if successes else 0
    results["Cartesian Direct"] = direct_sr
    env.close()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY (With Preserved Seeds)")
    print("="*80)
    print("\n| Method               | Success Rate |")
    print("|----------------------|--------------|")
    
    for method, sr in results.items():
        status = "✅" if sr >= 90 else ("⚠️" if sr >= 50 else "❌")
        print(f"| {method:20s} | {status} {sr:5.1f}% |")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    joint_sr = results.get("Joint Actions", 0)
    achieved_sr = results.get("Cartesian Achieved", 0)
    target_sr = results.get("Cartesian Target", 0)
    direct_sr = results.get("Cartesian Direct", 0)
    
    print(f"""
Expected Results (with preserved seeds):
- Joint Actions: Should be ~100% (ground truth)
- Cartesian Target: Should closely match Joint Actions
- Cartesian Achieved: May be slightly lower due to PD controller errors
- Cartesian Direct: Should be ~100% (bypasses PD controller)

Actual Results:
- Joint Actions: {joint_sr:.1f}%
- Cartesian Target: {target_sr:.1f}%
- Cartesian Achieved: {achieved_sr:.1f}%
- Cartesian Direct: {direct_sr:.1f}%
""")
    
    if joint_sr >= 90:
        print("✅ Joint actions work correctly with proper seeds!")
    else:
        print("⚠️ Some joint demos may have issues even with correct seeds")
    
    if abs(target_sr - joint_sr) <= 10:
        print("✅ Target-based conversion successfully mimics joint control!")
    else:
        print("⚠️ Target-based conversion has some deviation from joint control")
    
    if achieved_sr >= joint_sr - 20:
        print("✅ Achieved-based conversion maintains reasonable performance!")
    else:
        print("⚠️ Achieved-based conversion shows significant performance drop")
    
    if direct_sr >= 95:
        print("✅ Direct mode achieves near-perfect accuracy as expected!")
    else:
        print("⚠️ Direct mode has lower accuracy than expected")
    
    # return seed_results
    
    print("\n" + "="*80)
    print("SEED PRESERVATION CHECK")
    print("="*80)
    
    # Collect all seeds from all three demo sets
    joint_seeds = set([demo.seed for demo in joint_demos])
    
    achieved_seeds = set()
    for i in range(60):
        demo_path = Path(f"cartesian_demos/cartesian_demo_{i:03d}.safetensors")
        if demo_path.exists():
            demo = Demo.from_safetensors(demo_path)
            if demo:
                achieved_seeds.add(demo.seed)
    
    target_seeds = set()
    for i in range(60):
        demo_path = Path(f"cartesian_demos_target/cartesian_demo_{i:03d}.safetensors")
        if demo_path.exists():
            demo = Demo.from_safetensors(demo_path)
            if demo:
                target_seeds.add(demo.seed)
    
    # Check set equality
    print("\n1. SEED SET COMPARISON:")
    print(f"  Joint demo seeds: {len(joint_seeds)} unique seeds")
    print(f"  Achieved demo seeds: {len(achieved_seeds)} unique seeds")
    print(f"  Target demo seeds: {len(target_seeds)} unique seeds")
    
    common_seeds = joint_seeds & achieved_seeds & target_seeds
    print(f"\n  Common seeds across all three sets: {len(common_seeds)}")
    
    if joint_seeds == achieved_seeds == target_seeds:
        print("  ✅ All three demo sets contain the exact same seeds!")
    else:
        print("  ⚠️ Seeds don't match perfectly across all sets")
        print(f"    Seeds only in joint: {len(joint_seeds - achieved_seeds - target_seeds)}")
        print(f"    Seeds only in achieved: {len(achieved_seeds - joint_seeds - target_seeds)}")
        print(f"    Seeds only in target: {len(target_seeds - joint_seeds - achieved_seeds)}")
    
    # Create table of results per seed
    print("\n2. PER-SEED SUCCESS TABLE:")
    print("  (✓ = success, ✗ = failure, - = not tested)")
    print("\n  | Seed         | Joint | Achieved | Target | Direct |")
    print("  |--------------|-------|----------|--------|--------|")
    
    # Sort seeds for consistent display
    all_seeds = sorted(seed_results.keys())
    
    for seed in all_seeds[:20]:  # Show first 20 seeds
        joint_result = seed_results[seed].get('joint', (None, None))
        achieved_result = seed_results[seed].get('achieved', (None, None))
        target_result = seed_results[seed].get('target', (None, None))
        direct_result = seed_results[seed].get('direct', (None, None))
        
        joint_mark = "✓" if joint_result[0] else ("✗" if joint_result[0] is False else "-")
        achieved_mark = "✓" if achieved_result[0] else ("✗" if achieved_result[0] is False else "-")
        target_mark = "✓" if target_result[0] else ("✗" if target_result[0] is False else "-")
        direct_mark = "✓" if direct_result[0] else ("✗" if direct_result[0] is False else "-")
        
        print(f"  | {seed:12d} | {joint_mark:^5} | {achieved_mark:^8} | {target_mark:^6} | {direct_mark:^6} |")
    
    if len(all_seeds) > 20:
        print(f"  | ... ({len(all_seeds) - 20} more seeds) |")
    
    # Count success patterns
    print("\n3. SUCCESS PATTERN ANALYSIS:")
    all_four_success = 0
    all_but_direct = 0
    all_but_achieved = 0
    joint_direct_only = 0
    none_success = 0
    other_patterns = 0
    
    for seed in seed_results:
        j = seed_results[seed].get('joint', (False,))[0]
        a = seed_results[seed].get('achieved', (False,))[0]
        t = seed_results[seed].get('target', (False,))[0]
        d = seed_results[seed].get('direct', (False,))[0]
        
        if j and a and t and d:
            all_four_success += 1
        elif j and a and t and not d:
            all_but_direct += 1
        elif j and not a and t and d:
            all_but_achieved += 1
        elif j and d and not a and not t:
            joint_direct_only += 1
        elif not j and not a and not t and not d:
            none_success += 1
        else:
            other_patterns += 1
    
    print(f"  All four succeed: {all_four_success}")
    print(f"  All except Direct: {all_but_direct}")
    print(f"  All except Achieved: {all_but_achieved}")
    print(f"  Joint + Direct only: {joint_direct_only}")
    print(f"  None succeed: {none_success}")
    print(f"  Other patterns: {other_patterns}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test demo conversion with preserved seeds")
    parser.add_argument("--n-demos", type=int, default=60, 
                        help="Number of demos to test (default: 60)")
    args = parser.parse_args()
    
    test_preserved_seeds(n_demos=args.n_demos)