#!/usr/bin/env python3
"""Test RBY1 robot performance against H1 baseline on the same demonstrations.

This script compares:
1. H1 with joint actions (baseline)
2. RBY1 with converted Cartesian actions (PD-controlled)
3. RBY1 with converted Cartesian actions (direct mode)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Type, List, Optional, Dict, Any, Tuple
import importlib

from bigym.action_modes import JointPositionActionMode
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo
from bigym.robots.configs.h1 import H1
from bigym.robots.configs.rby1 import RBY1


def get_environment_class(env_name: str) -> Type:
    """Dynamically import and return the environment class.
    
    Args:
        env_name: Name of the environment (e.g., 'ReachTarget', 'MovePlate')
        
    Returns:
        Environment class
    """
    # Map of environment names to their module paths
    env_modules = {
        'ReachTarget': 'bigym.envs.reach_target',
        'MovePlate': 'bigym.envs.move_plates',
        'MovePlates': 'bigym.envs.move_plates',  # Alias
        'PickCube': 'bigym.envs.pick_cube',
        'StackBlocks': 'bigym.envs.stack_blocks',
        'FlipCup': 'bigym.envs.manipulation',
        # Add more environments here as needed
    }
    
    # Handle MovePlates -> MovePlate mapping
    class_name = 'MovePlate' if env_name == 'MovePlates' else env_name
    
    if env_name not in env_modules:
        # Try to import from bigym.envs directly
        module_name = f"bigym.envs.{env_name.lower()}"
        try:
            module = importlib.import_module(module_name)
            return getattr(module, env_name)
        except (ImportError, AttributeError):
            raise ValueError(f"Unknown environment: {env_name}")
    
    module = importlib.import_module(env_modules[env_name])
    return getattr(module, class_name)


def get_default_camera_config(env_name: str) -> List[CameraConfig]:
    """Get default camera configuration for an environment.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        List of camera configurations
    """
    # Default configurations for known environments
    configs = {
        'ReachTarget': [
            CameraConfig(name="head", rgb=True, depth=False, resolution=(128, 128))
        ],
        'MovePlate': [
            CameraConfig("head", resolution=(84, 84)),
            CameraConfig("left_wrist", resolution=(84, 84)),
            CameraConfig("right_wrist", resolution=(84, 84)),
        ],
        'MovePlates': [
            CameraConfig("head", resolution=(84, 84)),
            CameraConfig("left_wrist", resolution=(84, 84)),
            CameraConfig("right_wrist", resolution=(84, 84)),
        ],
        'PickCube': [
            CameraConfig(name="head", rgb=True, depth=False, resolution=(128, 128))
        ],
        'StackBlocks': [
            CameraConfig(name="head", rgb=True, depth=False, resolution=(128, 128))
        ],
    }
    
    # Return the config if known, otherwise a minimal default
    return configs.get(env_name, [
        CameraConfig(name="head", rgb=True, depth=False, resolution=(128, 128))
    ])


def test_h1_baseline(
    env_class: Type,
    camera_configs: List[CameraConfig],
    n_demos: int,
    control_frequency: int = 50
) -> Tuple[Dict[int, Tuple[bool, Optional[int]]], List[Demo], float]:
    """Test H1 with original joint actions (baseline).
    
    Returns:
        Tuple of (seed_results, joint_demos, success_rate)
    """
    print("\n1. H1 JOINT ACTIONS (baseline):")
    
    # Create H1 environment with cameras to load demos
    env = env_class(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=None,
        robot_cls=H1
    )
    
    # Load demos
    metadata = Metadata.from_env(env)
    demo_store = DemoStore()
    joint_demos = demo_store.get_demos(metadata, amount=n_demos, frequency=control_frequency)
    
    # Close and recreate without cameras for faster rollout
    env.close()
    env = env_class(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=H1
    )
    
    seed_results = {}
    successes = []
    
    for i, demo in enumerate(joint_demos):
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
            _, reward, terminated, truncated, info = env.step(action)
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
        seed_results[demo.seed] = (success, success_step if success else None)
    
    env.close()
    
    success_rate = sum(successes) / len(successes) * 100 if successes else 0
    return seed_results, joint_demos, success_rate


def test_rby1_cartesian(
    env_class: Type,
    mode_name: str,
    demo_dir: str,
    test_seeds: List[int],
    direct_mode: bool,
    control_frequency: int = 50
) -> Tuple[Dict[int, Tuple[bool, Optional[int]]], float]:
    """Test RBY1 with cartesian actions.
    
    Returns:
        Tuple of (seed_results, success_rate)
    """
    print(f"\n{mode_name}:")
    
    env = env_class(
        action_mode=RBY1CartesianActionModeWholeBody(direct_mode=direct_mode, block_until_reached=False, control_frequency=control_frequency),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1
    )
    
    seed_results = {}
    successes = []
    tested_count = 0
    
    # Try to load demos from the specified directory
    demo_dir_path = Path(demo_dir)
    if not demo_dir_path.exists():
        print(f"  ⚠️ Demo directory not found: {demo_dir}")
        env.close()
        return seed_results, 0.0
    
    # Load all demos from directory
    demo_files = sorted(demo_dir_path.glob("rby1_cartesian_demo_*.safetensors"))
    
    for demo_file in demo_files:
        demo = Demo.from_safetensors(demo_file)
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
                action = timestep.executed_action
            if action is None:
                action = timestep.action
            if action is None:
                continue
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, reward, terminated, truncated, info = env.step(action)
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
        seed_results[demo.seed] = (success, success_step if success else None)
    
    env.close()
    
    success_rate = sum(successes) / len(successes) * 100 if successes else 0
    return seed_results, success_rate


def test_rby1_vs_h1(
    env_name: str,
    n_demos: int = 10,
    control_frequency: int = 50,
    rby1_demo_dir: Optional[str] = None,
    test_h1: bool = True,
    test_rby1_pd: bool = True,
    test_rby1_direct: bool = True
):
    """Test RBY1 performance against H1 baseline.
    
    Args:
        env_name: Name of the environment
        n_demos: Number of demos to test
        control_frequency: Control frequency
        rby1_demo_dir: Directory for RBY1 demos (auto-generated if None)
        test_h1: Whether to test H1 baseline
        test_rby1_pd: Whether to test RBY1 with PD control
        test_rby1_direct: Whether to test RBY1 with direct control
    """
    print("="*80)
    print(f"TESTING RBY1 vs H1 on {env_name} (n={n_demos})")
    print("="*80)
    
    # Show which tests will be run
    tests_to_run = []
    if test_h1:
        tests_to_run.append("H1 baseline")
    if test_rby1_pd:
        tests_to_run.append("RBY1 PD-controlled")
    if test_rby1_direct:
        tests_to_run.append("RBY1 direct")
    
    if not tests_to_run:
        print("No tests selected! Use --h1, --rby1-pd, or --rby1-direct flags.")
        return
    
    print(f"\nRunning tests: {', '.join(tests_to_run)}")
    print("-"*80)
    
    # Get environment class and camera config
    env_class = get_environment_class(env_name)
    camera_configs = get_default_camera_config(env_name)
    
    # Auto-generate directory name if not provided
    if rby1_demo_dir is None:
        rby1_demo_dir = f"rby1_cartesian_demos_{env_name.lower()}"
    
    results = {}
    all_seed_results = {}
    success_counts = {}
    test_seeds = set()
    h1_demos = []
    
    # Test 1: H1 Joint Actions (baseline)
    if test_h1:
        h1_seed_results, h1_demos, h1_sr = test_h1_baseline(
            env_class, camera_configs, n_demos, control_frequency
        )
        results["H1 Joint Actions"] = h1_sr
        success_counts["H1 Joint Actions"] = sum(1 for (s, _) in h1_seed_results.values() if s)
        
        # Collect test seeds
        test_seeds = set(demo.seed for demo in h1_demos)
        
        # Merge results
        for seed, result in h1_seed_results.items():
            if seed not in all_seed_results:
                all_seed_results[seed] = {}
            all_seed_results[seed]['h1'] = result
    else:
        # If not testing H1, we need to get seeds from RBY1 demos
        demo_dir_path = Path(rby1_demo_dir)
        if demo_dir_path.exists():
            demo_files = sorted(demo_dir_path.glob("rby1_cartesian_demo_*.safetensors"))[:n_demos]
            for demo_file in demo_files:
                demo = Demo.from_safetensors(demo_file)
                if demo:
                    test_seeds.add(demo.seed)
    
    # Test 2: RBY1 Cartesian (PD-controlled)
    if test_rby1_pd:
        rby1_pd_seed_results, rby1_pd_sr = test_rby1_cartesian(
            env_class,
            "2. RBY1 CARTESIAN (PD-controlled)",
            rby1_demo_dir,
            test_seeds,
            direct_mode=False,
            control_frequency=control_frequency
        )
        results["RBY1 Cartesian PD"] = rby1_pd_sr
        success_counts["RBY1 Cartesian PD"] = sum(1 for (s, _) in rby1_pd_seed_results.values() if s)
        
        # Merge results
        for seed, result in rby1_pd_seed_results.items():
            if seed not in all_seed_results:
                all_seed_results[seed] = {}
            all_seed_results[seed]['rby1_pd'] = result
    
    # Test 3: RBY1 Cartesian Direct
    if test_rby1_direct:
        rby1_direct_seed_results, rby1_direct_sr = test_rby1_cartesian(
            env_class,
            "3. RBY1 CARTESIAN DIRECT (direct qpos control)",
            rby1_demo_dir,
            test_seeds,
            direct_mode=True,
            control_frequency=control_frequency
        )
        results["RBY1 Cartesian Direct"] = rby1_direct_sr
        success_counts["RBY1 Cartesian Direct"] = sum(1 for (s, _) in rby1_direct_seed_results.values() if s)
        
        # Merge results
        for seed, result in rby1_direct_seed_results.items():
            if seed not in all_seed_results:
                all_seed_results[seed] = {}
            all_seed_results[seed]['rby1_direct'] = result
    
    # Calculate step statistics
    step_stats = {}
    methods = ['h1', 'rby1_pd', 'rby1_direct']
    
    for method in methods:
        steps = []
        for seed_data in all_seed_results.values():
            if method in seed_data:
                success, step = seed_data[method]
                if success and step is not None:
                    steps.append(step)
        
        if steps:
            step_stats[method] = {
                'count': len(steps),
                'mean': np.mean(steps),
                'std': np.std(steps),
                'min': min(steps),
                'max': max(steps),
                'median': np.median(steps)
            }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY (RBY1 vs H1 Performance)")
    print("="*80)
    
    # First show success counts
    print(f"\nSuccesses out of {n_demos} demos:")
    for method in ["H1 Joint Actions", "RBY1 Cartesian PD", "RBY1 Cartesian Direct"]:
        count = success_counts.get(method, 0)
        print(f"  {method}: {count}/{n_demos}")
    
    print("\n| Method               | Success Rate | Avg Steps | Min | Max | Median |")
    print("|----------------------|--------------|-----------|-----|-----|--------|")
    
    method_names = {
        "H1 Joint Actions": "h1",
        "RBY1 Cartesian PD": "rby1_pd",
        "RBY1 Cartesian Direct": "rby1_direct"
    }
    
    for method, sr in results.items():
        status = "✅" if sr >= 90 else ("⚠️" if sr >= 50 else "❌")
        key = method_names.get(method)
        if key and key in step_stats:
            stats = step_stats[key]
            print(f"| {method:20s} | {status} {sr:5.1f}% | {stats['mean']:9.1f} | "
                  f"{stats['min']:3.0f} | {stats['max']:3.0f} | {stats['median']:6.0f} |")
        else:
            print(f"| {method:20s} | {status} {sr:5.1f}% |     -     |  -  |  -  |   -    |")
    
    # Print analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    h1_sr = results.get("H1 Joint Actions", 0)
    rby1_pd_sr = results.get("RBY1 Cartesian PD", 0)
    rby1_direct_sr = results.get("RBY1 Cartesian Direct", 0)
    
    print(f"""
Expected Results:
- H1 Joint Actions: Should be ~100% (ground truth with original robot)
- RBY1 Cartesian PD: May be lower due to PD controller errors and robot differences
- RBY1 Cartesian Direct: Should be higher than PD mode (bypasses controller errors)

Actual Results:
- H1 Joint Actions: {h1_sr:.1f}%
- RBY1 Cartesian PD: {rby1_pd_sr:.1f}%
- RBY1 Cartesian Direct: {rby1_direct_sr:.1f}%""")
    
    # Diagnose issues
    if rby1_direct_sr < 50:
        print("\n⚠️ WARNING: RBY1 Direct mode has low success rate!")
        print("   Possible causes:")
        print("   - RBY1 physical constraints (shorter reach)")
        print("   - IK solver limitations for RBY1's morphology")
        print("   - Target positions outside RBY1's workspace")
    elif rby1_pd_sr < rby1_direct_sr - 20:
        print("\n⚠️ PD controller causing significant performance drop")
        print("   Consider tuning PD gains or using direct mode")
    
    # Step count analysis - always show this section
    print("\n" + "="*80)
    print("STEP COUNT STATISTICS")
    print("="*80)
    
    # Show statistics for each method that has successful demos
    stats_shown = False
    for method_name, method_key in method_names.items():
        if method_key in step_stats:
            stats = step_stats[method_key]
            print(f"\n{method_name}:")
            print(f"  Mean:   {stats['mean']:.1f} steps")
            print(f"  Std:    {stats['std']:.1f} steps")
            print(f"  Median: {stats['median']:.1f} steps")
            print(f"  Range:  {stats['min']:.0f} - {stats['max']:.0f} steps")
            stats_shown = True
    
    if not stats_shown:
        print("\nNo successful demonstrations to compute step statistics.")
    
    # Per-seed comparison
    print("\n" + "="*80)
    print("PER-SEED COMPARISON")
    print("="*80)
    
    print("\n| Seed       | H1 | RBY1 PD | RBY1 Direct |")
    print("|------------|----|---------|--------------]")
    
    for seed in sorted(test_seeds)[:10]:  # Show first 10 seeds
        if seed in all_seed_results:
            data = all_seed_results[seed]
            h1_res = "✅" if data.get('h1', (False,))[0] else "❌"
            pd_res = "✅" if data.get('rby1_pd', (False,))[0] else "❌"
            direct_res = "✅" if data.get('rby1_direct', (False,))[0] else "❌"
            print(f"| {seed:10d} | {h1_res}  |   {pd_res}     |     {direct_res}       |")
    
    if len(test_seeds) > 10:
        print(f"| ... ({len(test_seeds)-10} more seeds)")


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test RBY1 performance against H1 baseline"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MovePlate",
        help="Environment name (e.g., ReachTarget, MovePlate)"
    )
    parser.add_argument(
        "--n-demos",
        type=int,
        default=60,
        help="Number of demos to test (default: 60)"
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=50,
        help="Control frequency (default: 50)"
    )
    parser.add_argument(
        "--rby1-dir",
        type=str,
        default=None,
        help="Directory for RBY1 demos (auto-generated if not specified)"
    )
    
    # Add test selection flags
    parser.add_argument(
        "--h1",
        action="store_true",
        default=False,
        help="Test H1 baseline (default: False)"
    )
    parser.add_argument(
        "--rby1-pd",
        action="store_true",
        default=True,
        help="Test RBY1 with PD control (default: False)"
    )
    parser.add_argument(
        "--rby1-direct",
        action="store_true",
        default=False,
        help="Test RBY1 with direct control (default: False)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all tests (equivalent to --h1 --rby1-pd --rby1-direct)"
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    test_h1 = args.h1 or args.all
    test_rby1_pd = args.rby1_pd or args.all
    test_rby1_direct = args.rby1_direct or args.all
    
    # If no specific tests selected, default to all
    if not (test_h1 or test_rby1_pd or test_rby1_direct):
        print("No tests specified. Use --h1, --rby1-pd, --rby1-direct, or --all")
        print("Defaulting to --all (running all tests)")
        test_h1 = True
        test_rby1_pd = True
        test_rby1_direct = True
    
    test_rby1_vs_h1(
        env_name=args.env,
        n_demos=args.n_demos,
        control_frequency=args.control_freq,
        rby1_demo_dir=args.rby1_dir,
        test_h1=test_h1,
        test_rby1_pd=test_rby1_pd,
        test_rby1_direct=test_rby1_direct
    )


if __name__ == "__main__":
    main()