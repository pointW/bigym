#!/usr/bin/env python3
"""Simple demo replay for floating gripper robot - follows RBY1 pattern."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from bigym.envs.reach_target import ReachTarget
from bigym.envs.move_plates import MovePlate
from bigym.floating_gripper_action_mode import FloatingGripperActionMode
from demonstrations.demo import Demo
from bigym.robots.configs.floating_grippers import FloatingGrippers
from bigym.const import HandSide


def replay_floating_gripper_demo(env_name: str = "ReachTarget", demo_dir: str = None, headless: bool = False):
    """Replay floating gripper demo step by step.
    
    Args:
        env_name: Environment name ("ReachTarget" or "MovePlate")
        demo_dir: Directory containing cartesian demos
        headless: If True, run without GUI. If False, show visualization.
    """
    
    # Set default demo directory based on environment - use RBY1 demos
    if demo_dir is None:
        demo_dir = f"rby1_cartesian_demos_{env_name.lower()}"
    
    # Load first demo
    demo_path = Path(demo_dir)
    demo_files = sorted(demo_path.glob("rby1_cartesian_demo_*.safetensors"))
    
    if not demo_files:
        print(f"No demos found in {demo_dir}!")
        print("Please run conversion script first:")
        print(f"  python scripts/convert_demos_to_cartesian.py --env {env_name}")
        return
    
    demo = Demo.from_safetensors(demo_files[0])
    print(f"Loaded demo with seed {demo.seed}, {len(demo.timesteps)} timesteps")
    
    # Select environment class
    if env_name == "ReachTarget":
        env_cls = ReachTarget
    elif env_name == "MovePlate":
        env_cls = MovePlate
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Create floating gripper environment
    env = env_cls(
        action_mode=FloatingGripperActionMode(),
        control_frequency=50,
        render_mode=None if headless else "human",
        robot_cls=FloatingGrippers
    )
    
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Expected: 20D (3+6+3+6+2)")
    
    # Reset with demo seed
    env.reset(seed=demo.seed)
    if not headless:
        env.render()
    
    print("\nStarting replay...")
    
    max_tracking_error = 0.0
    success = False
    
    for step_idx in range(len(demo.timesteps)):
        timestep = demo.timesteps[step_idx]
        
        # Get action from demo (try different storage locations)
        action = timestep.info.get('demo_action')
        if action is None:
            action = timestep.executed_action
        if action is None:
            action = timestep.action
        
        if action is None:
            print(f"Step {step_idx}: No action found!")
            continue
        
        print(f"\nStep {step_idx}:")
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        # Parse action to understand it
        if len(action) == 20:
            left_pos = action[0:3]
            left_ori = action[3:9]
            right_pos = action[9:12]
            right_ori = action[12:18]
            grippers = action[18:20]
            
            print(f"  Target left pos: {left_pos}")
            print(f"  Target right pos: {right_pos}")
            print(f"  Grippers: {grippers}")
        else:
            print(f"  WARNING: Unexpected action dimension {len(action)}")
        
        # Step environment
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            if not headless:
                env.render()
            
            # Get actual end-effector positions
            achieved_left = env.robot.get_hand_pos(HandSide.LEFT)
            achieved_right = env.robot.get_hand_pos(HandSide.RIGHT)
            
            print(f"  Achieved left pos: {achieved_left}")
            print(f"  Achieved right pos: {achieved_right}")
            
            # Calculate tracking error
            if len(action) == 20:
                left_error = np.linalg.norm(achieved_left - left_pos)
                right_error = np.linalg.norm(achieved_right - right_pos)
                max_tracking_error = max(max_tracking_error, left_error, right_error)
                
                print(f"  Tracking error: L={left_error*1000:.6f}mm, R={right_error*1000:.6f}mm")
                
                if left_error < 1e-6 and right_error < 1e-6:
                    print(f"  ✓ PERFECT tracking!")
            
            print(f"  Reward: {reward:.3f}")
            print(f"  Success: {info.get('task_success', False)}")
            
            if info.get('task_success', False):
                success = True
                print(f"✅ SUCCESS at step {step_idx}!")
                break
            
            if terminated or truncated:
                print(f"Episode ended at step {step_idx}")
                break
                
        except Exception as e:
            print(f"  ERROR: {e}")
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("REPLAY SUMMARY")
    print("=" * 80)
    print(f"Demo seed: {demo.seed}")
    print(f"Task success: {'✅ Yes' if success else '❌ No'}")
    print(f"Max tracking error: {max_tracking_error*1000:.6f}mm")
    
    if max_tracking_error < 1e-5:
        print("✨ PERFECT TRACKING achieved throughout replay!")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replay floating gripper demos")
    parser.add_argument("--env", default="MovePlate", 
                        choices=["ReachTarget", "MovePlate"],
                        help="Environment to use")
    parser.add_argument("--demo-dir", help="Directory containing demos")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    
    replay_floating_gripper_demo(args.env, args.demo_dir, args.headless)