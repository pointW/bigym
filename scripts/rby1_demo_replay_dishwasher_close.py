#!/usr/bin/env python3
"""Debug RBY1 demo replay to understand why it's failing."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from bigym.envs.reach_target import ReachTarget
from bigym.envs.move_plates import MovePlate
from bigym.envs.manipulation import FlipCup
from bigym.envs.dishwasher import DishwasherOpen, DishwasherClose
from bigym.envs.test_env import TestEnv
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from demonstrations.demo import Demo
from bigym.robots.configs.rby1 import RBY1
from bigym.const import HandSide


def debug_rby1_demo(headless: bool = False):
    """Debug RBY1 demo replay step by step.
    
    Args:
        headless: If True, run without GUI. If False, show visualization.
    """
    
    # Load first RBY1 demo
    demo_dir = Path("rby1_cartesian_demos_dishwasherclose")
    # demo_dir = Path("rby1_cartesian_demos_flipcup")
    demo_files = sorted(demo_dir.glob("rby1_cartesian_demo_*.safetensors"))
    
    if not demo_files:
        print("No RBY1 demos found!")
        return
    
    for i in range(len(demo_files)):
        demo = Demo.from_safetensors(demo_files[i])
        if demo.seed == 346512169:
            break
    # demo = Demo.from_safetensors(demo_files[0])
    print(f"Loaded demo with seed {demo.seed}, {len(demo.timesteps)} timesteps")
    
    # Create RBY1 environment
    env = DishwasherClose(
        action_mode=RBY1CartesianActionModeWholeBody(direct_mode=False, block_until_reached=False, control_frequency=50),
        control_frequency=50,
        render_mode=None if headless else "human",  # GUI when not headless
        robot_cls=RBY1
    )

    # env = FlipCup(
    #     action_mode=RBY1CartesianActionModeWholeBody(direct_mode=False, block_until_reached=False, control_frequency=20),
    #     control_frequency=20,
    #     render_mode=None if headless else "human",  # GUI when not headless
    #     robot_cls=RBY1
    # )
    
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Expected: 20D (3+6+3+6+2)")
    
    # Reset with demo seed
    env.reset(seed=demo.seed)
    if not headless:
        env.render()

    # while True:
    #     env.render()
    
    print("\nStarting replay...")
    
    for step_idx in range(len(demo.timesteps)): 
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
            
            print(f"  Left EE pos: {left_pos}")
            print(f"  Right EE pos: {right_pos}")
            print(f"  Grippers: {grippers}")
        else:
            print(f"  WARNING: Unexpected action dimension {len(action)}")
        
        # Step environment
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            if not headless:
                env.render()
            
            # Get actual end-effector positions
            left_site = env.robot._wrist_sites[HandSide.LEFT]
            right_site = env.robot._wrist_sites[HandSide.RIGHT]
            
            print(f"  Achieved left pos: {left_site.get_position()}")
            print(f"  Achieved right pos: {right_site.get_position()}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Success: {info.get('task_success', False)}")
            
            if info.get('task_success', False):
                print(f"✅ SUCCESS at step {step_idx}!")
                break
            
            if terminated or truncated:
                print(f"Episode ended at step {step_idx}")
                break
                
        except Exception as e:
            print(f"  ERROR: {e}")
            break
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug RBY1 demo replay")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    
    debug_rby1_demo(headless=args.headless)