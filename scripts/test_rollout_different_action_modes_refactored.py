#!/usr/bin/env python3
"""Test that converted demos with preserved seeds can be replayed correctly.

This refactored version supports different environment types dynamically.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Type, List, Optional, Dict, Any, Tuple
import importlib

from bigym.action_modes import JointPositionActionMode
from bigym.cartesian_action_mode import CartesianActionMode
from bigym.cartesian_action_mode_direct import CartesianActionModeDirect
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo


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
    }
    
    # Return the config if known, otherwise a minimal default
    return configs.get(env_name, [
        CameraConfig(name="head", rgb=True, depth=False, resolution=(128, 128))
    ])


def test_joint_actions(
    env_class: Type,
    camera_configs: List[CameraConfig],
    n_demos: int,
    control_frequency: int = 50
) -> Tuple[Dict[int, Tuple[bool, Optional[int]]], List[Demo], float]:
    """Test original joint actions (baseline).
    
    Returns:
        Tuple of (seed_results, joint_demos, success_rate)
    """
    print("\n1. ORIGINAL JOINT ACTIONS (baseline):")
    
    # Create environment with cameras to load demos
    env = env_class(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=None,
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


def test_cartesian_mode(
    env_class: Type,
    mode_name: str,
    demo_dir: str,
    test_seeds: List[int],
    action_mode: Any,
    control_frequency: int = 50
) -> Tuple[Dict[int, Tuple[bool, Optional[int]]], float]:
    """Test a cartesian action mode.
    
    Returns:
        Tuple of (seed_results, success_rate)
    """
    print(f"\n{mode_name}:")
    
    env = env_class(
        action_mode=action_mode,
        control_frequency=control_frequency,
        render_mode=None,
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
    demo_files = sorted(demo_dir_path.glob("cartesian_demo_*.safetensors"))
    
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


def test_preserved_seeds(
    env_name: str,
    n_demos: int = 60,
    control_frequency: int = 50,
    achieved_dir: Optional[str] = None,
    target_dir: Optional[str] = None,
    test_direct: bool = True,
    ik_solver: str = "mink"
):
    """Test all conversion methods with preserved seeds.
    
    Args:
        env_name: Name of the environment
        n_demos: Number of demos to test
        control_frequency: Control frequency
        achieved_dir: Directory for achieved demos (auto-generated if None)
        target_dir: Directory for target demos (auto-generated if None)
        test_direct: Whether to test direct mode
        ik_solver: IK solver to use for direct mode
    """
    print("="*80)
    print(f"TESTING {env_name} DEMOS WITH PRESERVED SEEDS (n={n_demos})")
    print("="*80)
    print("\nConverted demos should now contain the original seed values")
    print("This allows proper replay with correct target positions")
    print("-"*80)
    
    # Get environment class and camera config
    env_class = get_environment_class(env_name)
    camera_configs = get_default_camera_config(env_name)
    
    # Auto-generate directory names if not provided
    if achieved_dir is None:
        achieved_dir = f"cartesian_demos_{env_name.lower()}"
    if target_dir is None:
        target_dir = f"cartesian_demos_target_{env_name.lower()}"
    
    results = {}
    all_seed_results = {}
    
    # Test 1: Original Joint Actions (baseline)
    joint_seed_results, joint_demos, joint_sr = test_joint_actions(
        env_class, camera_configs, n_demos, control_frequency
    )
    results["Joint Actions"] = joint_sr
    
    # Collect test seeds
    test_seeds = set(demo.seed for demo in joint_demos)
    
    # Merge results
    for seed, result in joint_seed_results.items():
        if seed not in all_seed_results:
            all_seed_results[seed] = {}
        all_seed_results[seed]['joint'] = result
    
    # Test 2: Cartesian Achieved
    achieved_seed_results, achieved_sr = test_cartesian_mode(
        env_class,
        "2. CARTESIAN ACHIEVED (with preserved seed)",
        achieved_dir,
        test_seeds,
        CartesianActionMode(floating_base=True),
        control_frequency
    )
    results["Cartesian Achieved"] = achieved_sr
    
    # Merge results
    for seed, result in achieved_seed_results.items():
        if seed not in all_seed_results:
            all_seed_results[seed] = {}
        all_seed_results[seed]['achieved'] = result
    
    # Test 3: Cartesian Target
    target_seed_results, target_sr = test_cartesian_mode(
        env_class,
        "3. CARTESIAN TARGET (with preserved seed)",
        target_dir,
        test_seeds,
        CartesianActionMode(floating_base=True),
        control_frequency
    )
    results["Cartesian Target"] = target_sr
    
    # Merge results
    for seed, result in target_seed_results.items():
        if seed not in all_seed_results:
            all_seed_results[seed] = {}
        all_seed_results[seed]['target'] = result
    
    # Test 4: Cartesian Direct (optional)
    if test_direct:
        direct_seed_results, direct_sr = test_cartesian_mode(
            env_class,
            "4. CARTESIAN DIRECT (achieved demos with direct qpos control)",
            achieved_dir,  # Use achieved demos for direct mode
            test_seeds,
            CartesianActionModeDirect(floating_base=True, ik_solver=ik_solver),
            control_frequency
        )
        results["Cartesian Direct"] = direct_sr
        
        # Merge results
        for seed, result in direct_seed_results.items():
            if seed not in all_seed_results:
                all_seed_results[seed] = {}
            all_seed_results[seed]['direct'] = result
    
    # Calculate step statistics
    step_stats = {}
    methods = ['joint', 'achieved', 'target']
    if test_direct:
        methods.append('direct')
    
    for method in methods:
        steps = []
        for seed_data in all_seed_results.values():
            if method in seed_data:
                success, step = seed_data[method]
                if success and step is not None:
                    steps.append(step)
        
        if steps:
            step_stats[method] = {
                'mean': np.mean(steps),
                'std': np.std(steps),
                'min': min(steps),
                'max': max(steps),
                'median': np.median(steps)
            }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY (With Preserved Seeds)")
    print("="*80)
    print("\n| Method               | Success Rate | Avg Steps | Min | Max | Median |")
    print("|----------------------|--------------|-----------|-----|-----|--------|")
    
    method_names = {
        "Joint Actions": "joint",
        "Cartesian Achieved": "achieved",
        "Cartesian Target": "target",
        "Cartesian Direct": "direct"
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
    
    joint_sr = results.get("Joint Actions", 0)
    achieved_sr = results.get("Cartesian Achieved", 0)
    target_sr = results.get("Cartesian Target", 0)
    
    print(f"""
Expected Results (with preserved seeds):
- Joint Actions: Should be ~100% (ground truth)
- Cartesian Target: Should closely match Joint Actions
- Cartesian Achieved: May be slightly lower due to PD controller errors""")
    
    if test_direct:
        direct_sr = results.get("Cartesian Direct", 0)
        print(f"- Cartesian Direct: Should be ~100% (bypasses PD controller)")
    
    print(f"""
Actual Results:
- Joint Actions: {joint_sr:.1f}%
- Cartesian Target: {target_sr:.1f}%
- Cartesian Achieved: {achieved_sr:.1f}%""")
    
    if test_direct:
        print(f"- Cartesian Direct: {direct_sr:.1f}%")
    
    # Step count analysis
    if step_stats:
        print("\n" + "="*80)
        print("STEP COUNT STATISTICS")
        print("="*80)
        
        for method_name, method_key in method_names.items():
            if method_key in step_stats:
                stats = step_stats[method_key]
                print(f"\n{method_name}:")
                print(f"  Mean:   {stats['mean']:.1f} steps")
                print(f"  Std:    {stats['std']:.1f} steps")
                print(f"  Median: {stats['median']:.1f} steps")
                print(f"  Range:  {stats['min']:.0f} - {stats['max']:.0f} steps")


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test demo conversion with preserved seeds for any environment"
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
        "--achieved-dir",
        type=str,
        default="cartesian_demos_move_plate",
        help="Directory for achieved demos (auto-generated if not specified)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="cartesian_demos_target_move_plate",
        help="Directory for target demos (auto-generated if not specified)"
    )
    parser.add_argument(
        "--no-direct",
        action="store_true",
        help="Skip testing direct mode"
    )
    parser.add_argument(
        "--ik-solver",
        type=str,
        default="mink",
        choices=["mink", "h1", "clean"],
        help="IK solver to use for direct mode (default: mink)"
    )
    
    args = parser.parse_args()
    
    test_preserved_seeds(
        env_name=args.env,
        n_demos=args.n_demos,
        control_frequency=args.control_freq,
        achieved_dir=args.achieved_dir,
        target_dir=args.target_dir,
        test_direct=False,
        ik_solver=args.ik_solver
    )


if __name__ == "__main__":
    main()