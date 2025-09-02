#!/usr/bin/env python3
"""Test Floating Gripper robot performance against H1 baseline on the same demonstrations.

This script compares:
1. H1 with joint actions (baseline)
2. Floating Gripper with converted Cartesian actions (perfect mocap tracking)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Type, List, Optional, Dict, Any, Tuple
import importlib

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.floating_gripper_action_mode import FloatingGripperActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo
from bigym.robots.configs.h1 import H1
from bigym.robots.configs.floating_grippers import FloatingGrippers


def detect_floating_dofs_from_demos(env_name: str) -> List[PelvisDof]:
    """Detect the correct floating DOFs for an environment based on available demos.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        List of PelvisDof enums for the floating base
    """
    # Environments that typically use 4 DOF (X, Y, Z, RZ) based on demo analysis
    four_dof_envs = {
        'FlipCup', 'FlipCutlery', 'FlipSandwich',  # Manipulation tasks need vertical movement
        'StackBlocks',  # Stacking needs vertical control
        'ToastSandwich', 'RemoveSandwich',  # Kitchen tasks with vertical elements
        'SaucepanToHob',  # Lifting saucepan
        'StoreBox', 'PickBox',  # Box manipulation
        'StoreKitchenware',  # Storing items at different heights
        'GroceriesStoreLower', 'GroceriesStoreUpper',  # Different height storage
        'TakeCups', 'PutCups',  # Cup manipulation
    }
    
    # Default to 3 DOF (X, Y, RZ) for most tasks
    # This includes ReachTarget, MovePlate, Dishwasher tasks, etc.
    if env_name in four_dof_envs:
        return [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    else:
        return [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]


def get_environment_class(env_name: str) -> Type:
    """Dynamically import and return the environment class.
    
    Args:
        env_name: Name of the environment (e.g., 'ReachTarget', 'MovePlate')
        
    Returns:
        Environment class
    """
    # Map of environment names to their module paths - based on available demos
    env_modules = {
        # Core environments
        'ReachTarget': 'bigym.envs.reach_target',
        'ReachTargetSingle': 'bigym.envs.reach_target',
        'ReachTargetDual': 'bigym.envs.reach_target',
        'MovePlate': 'bigym.envs.move_plates',
        'MovePlates': 'bigym.envs.move_plates',  # Alias
        'MoveTwoPlates': 'bigym.envs.move_plates',
        
        # Dishwasher tasks
        'DishwasherOpen': 'bigym.envs.dishwasher',
        'DishwasherClose': 'bigym.envs.dishwasher',
        'DishwasherOpenTrays': 'bigym.envs.dishwasher',
        'DishwasherCloseTrays': 'bigym.envs.dishwasher',
        'DishwasherLoadCups': 'bigym.envs.dishwasher',
        'DishwasherLoadCutlery': 'bigym.envs.dishwasher',
        'DishwasherLoadPlates': 'bigym.envs.dishwasher',
        'DishwasherUnloadCups': 'bigym.envs.dishwasher',
        'DishwasherUnloadCupsLong': 'bigym.envs.dishwasher',
        'DishwasherUnloadCutlery': 'bigym.envs.dishwasher',
        'DishwasherUnloadCutleryLong': 'bigym.envs.dishwasher',
        'DishwasherUnloadPlates': 'bigym.envs.dishwasher',
        'DishwasherUnloadPlatesLong': 'bigym.envs.dishwasher',
        
        # Manipulation tasks
        'FlipCup': 'bigym.envs.manipulation',
        'FlipCutlery': 'bigym.envs.manipulation',
        'FlipSandwich': 'bigym.envs.manipulation',
        'StackBlocks': 'bigym.envs.manipulation',
        
        # Kitchen tasks
        'ToastSandwich': 'bigym.envs.kitchen',
        'RemoveSandwich': 'bigym.envs.kitchen',
        'SaucepanToHob': 'bigym.envs.kitchen',
        
        # Storage tasks
        'StoreBox': 'bigym.envs.storage',
        'PickBox': 'bigym.envs.storage',
        'StoreKitchenware': 'bigym.envs.storage',
        'GroceriesStoreLower': 'bigym.envs.storage',
        'GroceriesStoreUpper': 'bigym.envs.storage',
        'TakeCups': 'bigym.envs.storage',
        'PutCups': 'bigym.envs.storage',
        
        # Cupboard/Drawer tasks
        'CupboardsOpenAll': 'bigym.envs.cupboards',
        'CupboardsCloseAll': 'bigym.envs.cupboards',
        'WallCupboardOpen': 'bigym.envs.cupboards',
        'WallCupboardClose': 'bigym.envs.cupboards',
        'DrawersAllOpen': 'bigym.envs.drawers',
        'DrawersAllClose': 'bigym.envs.drawers',
        'DrawerTopOpen': 'bigym.envs.drawers',
        'DrawerTopClose': 'bigym.envs.drawers',
    }
    
    # Handle special cases and determine actual class name
    if env_name == 'MovePlates':
        class_name = 'MovePlate'
    elif env_name == 'MoveTwoPlates':
        class_name = 'MovePlate'  # Might be same class with different config
    elif env_name.startswith('ReachTarget'):
        class_name = 'ReachTarget'  # All reach variants use same class
    else:
        class_name = env_name
    
    if env_name not in env_modules:
        # Try to import from bigym.envs directly
        module_name = f"bigym.envs.{env_name.lower()}"
        try:
            module = importlib.import_module(module_name)
            return getattr(module, env_name)
        except (ImportError, AttributeError):
            # Try without underscores
            module_name = f"bigym.envs.{env_name.replace('_', '').lower()}"
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
    # Default configurations for most environments
    return [
        CameraConfig("head", resolution=(84, 84)),
        CameraConfig("left_wrist", resolution=(84, 84)),
        CameraConfig("right_wrist", resolution=(84, 84)),
    ]


def test_h1_baseline(
    env_class: Type,
    env_name: str,
    camera_configs: List[CameraConfig],
    n_demos: int,
    control_frequency: int = 50
) -> Tuple[Dict[int, Tuple[bool, Optional[int]]], List[Demo], float]:
    """Test H1 with original joint actions (baseline).
    
    Returns:
        Tuple of (seed_results, joint_demos, success_rate)
    """
    print("\n1. H1 JOINT ACTIONS (baseline):")
    
    # Detect the correct floating DOFs for this environment
    floating_dofs = detect_floating_dofs_from_demos(env_name)
    dof_str = "4 DOF (X,Y,Z,RZ)" if len(floating_dofs) == 4 else "3 DOF (X,Y,RZ)"
    print(f"  Using {dof_str} floating base for {env_name}")
    
    # Create H1 environment with cameras to load demos
    env = env_class(
        action_mode=JointPositionActionMode(
            floating_base=True, 
            absolute=True,
            floating_dofs=floating_dofs
        ),
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
        action_mode=JointPositionActionMode(
            floating_base=True, 
            absolute=True,
            floating_dofs=floating_dofs
        ),
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
    return seed_results, joint_demos, success_rate


def test_floating_gripper_cartesian(
    env_class: Type,
    env_name: str,
    mode_name: str,
    demo_dir: str,
    test_seeds: List[int],
    control_frequency: int = 50
) -> Tuple[Dict[int, Tuple[bool, Optional[int]]], float]:
    """Test Floating Gripper with cartesian actions.
    
    Returns:
        Tuple of (seed_results, success_rate)
    """
    print(f"\n{mode_name}:")
    
    env = env_class(
        action_mode=FloatingGripperActionMode(control_frequency=control_frequency),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=FloatingGrippers
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
    
    if not demo_files:
        print(f"  ⚠️ No demo files found in: {demo_dir}")
        env.close()
        return seed_results, 0.0
    
    for demo_file in demo_files:
        try:
            demo = Demo.from_safetensors(demo_file)
        except Exception as e:
            print(f"  Warning: Could not load {demo_file}: {e}")
            continue
            
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


def test_fg_vs_h1(
    env_name: str,
    n_demos: int = 10,
    control_frequency: int = 50,
    fg_demo_dir: Optional[str] = None
):
    """Test Floating Gripper performance against H1 baseline.
    
    Args:
        env_name: Name of the environment
        n_demos: Number of demos to test
        control_frequency: Control frequency
        fg_demo_dir: Directory for floating gripper demos (auto-generated if None)
    """
    print("="*80)
    print(f"TESTING FLOATING GRIPPER vs H1 on {env_name} (n={n_demos})")
    print("="*80)
    print("\nComparing Floating Gripper (perfect mocap tracking) against H1 baseline")
    print("-"*80)
    
    # Get environment class and camera config
    env_class = get_environment_class(env_name)
    camera_configs = get_default_camera_config(env_name)
    
    # Auto-generate directory name if not provided
    if fg_demo_dir is None:
        fg_demo_dir = f"rby1_cartesian_demos_{env_name.lower()}"
    
    results = {}
    all_seed_results = {}
    success_counts = {}
    
    # Test 1: H1 Joint Actions (baseline)
    h1_seed_results, h1_demos, h1_sr = test_h1_baseline(
        env_class, env_name, camera_configs, n_demos, control_frequency
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
    
    # Test 2: Floating Gripper Cartesian (mocap-controlled)
    fg_seed_results, fg_sr = test_floating_gripper_cartesian(
        env_class,
        env_name,
        "2. FLOATING GRIPPER CARTESIAN (perfect mocap tracking)",
        fg_demo_dir,
        test_seeds,
        control_frequency=control_frequency
    )
    results["Floating Gripper"] = fg_sr
    success_counts["Floating Gripper"] = sum(1 for (s, _) in fg_seed_results.values() if s)
    
    # Merge results
    for seed, result in fg_seed_results.items():
        if seed not in all_seed_results:
            all_seed_results[seed] = {}
        all_seed_results[seed]['fg'] = result
    
    # Calculate step statistics
    step_stats = {}
    methods = ['h1', 'fg']
    
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
    print("SUMMARY (Floating Gripper vs H1 Performance)")
    print("="*80)
    
    # First show success counts
    print(f"\nSuccesses out of {n_demos} demos:")
    for method in ["H1 Joint Actions", "Floating Gripper"]:
        count = success_counts.get(method, 0)
        print(f"  {method}: {count}/{n_demos}")
    
    print("\n| Method               | Success Rate | Avg Steps | Min | Max | Median |")
    print("|----------------------|--------------|-----------|-----|-----|--------|")
    
    method_names = {
        "H1 Joint Actions": "h1",
        "Floating Gripper": "fg"
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
    fg_sr = results.get("Floating Gripper", 0)
    
    print(f"""
Expected Results:
- H1 Joint Actions: Should be ~100% (ground truth with original robot)
- Floating Gripper: Should achieve ~100% with perfect mocap tracking

Actual Results:
- H1 Joint Actions: {h1_sr:.1f}%
- Floating Gripper: {fg_sr:.1f}%""")
    
    # Diagnose issues
    if fg_sr < h1_sr - 10:
        print("\n⚠️ WARNING: Floating Gripper has lower success rate than H1!")
        print("   Possible causes:")
        print("   - Demo conversion issues")
        print("   - Gripper control differences")
        print("   - Missing interpolation or control frequency mismatch")
    elif fg_sr >= h1_sr - 5:
        print("\n✅ SUCCESS: Floating Gripper achieves comparable performance to H1!")
        print("   The perfect mocap tracking is working as expected.")
    
    # Step count analysis
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
    
    print("\n| Seed       | H1 | FG |")
    print("|------------|----|----|")
    
    for seed in sorted(test_seeds)[:10]:  # Show first 10 seeds
        if seed in all_seed_results:
            data = all_seed_results[seed]
            h1_res = "✅" if data.get('h1', (False,))[0] else "❌"
            fg_res = "✅" if data.get('fg', (False,))[0] else "❌"
            print(f"| {seed:10d} | {h1_res}  | {fg_res}  |")
    
    if len(test_seeds) > 10:
        print(f"| ... ({len(test_seeds)-10} more seeds)")
    
    # Additional info about floating gripper
    print("\n" + "="*80)
    print("FLOATING GRIPPER INFO")
    print("="*80)
    print("""
The Floating Gripper uses:
- Perfect mocap tracking (zero tracking error)
- Linear interpolation between control steps
- 500Hz physics with interpolation (vs 50Hz control)
- Same ROBOTIQ gripper as H1
- Compensated 7mm gripper offset

This should achieve near-perfect demo replay.""")


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Floating Gripper performance against H1 baseline"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="FlipCup",
        help="Environment name (e.g., ReachTarget, MovePlate, FlipCup, DishwasherOpen, etc.)"
    )
    parser.add_argument(
        "--n-demos",
        type=int,
        default=60,
        help="Number of demos to test (default: 10)"
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=20,
        help="Control frequency (default: 50)"
    )
    parser.add_argument(
        "--fg-dir",
        type=str,
        default=None,
        help="Directory for floating gripper demos (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Print available environments if requested
    if args.env == "list":
        print("Available environments:")
        print("\nCore environments:")
        print("  ReachTarget, MovePlate")
        print("\nManipulation tasks:")
        print("  FlipCup, FlipCutlery, FlipSandwich, StackBlocks")
        print("\nDishwasher tasks:")
        print("  DishwasherOpen, DishwasherClose, DishwasherOpenTrays, DishwasherCloseTrays")
        print("  DishwasherLoadCups, DishwasherLoadCutlery, DishwasherLoadPlates")
        print("  DishwasherUnloadCups, DishwasherUnloadCutlery, DishwasherUnloadPlates")
        print("\nKitchen tasks:")
        print("  ToastSandwich, RemoveSandwich, SaucepanToHob")
        print("\nStorage tasks:")
        print("  StoreBox, PickBox, StoreKitchenware")
        print("  GroceriesStoreLower, GroceriesStoreUpper, TakeCups, PutCups")
        print("\nCupboard/Drawer tasks:")
        print("  CupboardsOpenAll, CupboardsCloseAll, WallCupboardOpen, WallCupboardClose")
        print("  DrawersAllOpen, DrawersAllClose, DrawerTopOpen, DrawerTopClose")
        return
    
    test_fg_vs_h1(
        env_name=args.env,
        n_demos=args.n_demos,
        control_frequency=args.control_freq,
        fg_demo_dir=args.fg_dir
    )


if __name__ == "__main__":
    main()