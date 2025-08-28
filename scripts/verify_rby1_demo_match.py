#!/usr/bin/env python3
"""Verify RBY1 reset matches initial poses in RBY1 demonstrations."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bigym.envs.reach_target import ReachTarget
from bigym.action_modes import JointPositionActionMode
from bigym.robots.configs.rby1 import RBY1
from bigym.const import HandSide
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

def main():
    print("=" * 80)
    print("VERIFICATION: RBY1 RESET vs DEMO INITIAL POSES")
    print("=" * 80)
    
    # Create RBY1 environment
    rby1_env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=False, absolute=True),
        control_frequency=50,
        render_mode=None,
        robot_cls=RBY1
    )
    
    # Try to load RBY1 demos
    demo_store = DemoStore()
    
    # Get metadata for RBY1
    metadata = Metadata.from_env(rby1_env)
    print(f"\nLooking for demos with metadata:")
    print(f"  Robot: {metadata.robot_name}")
    print(f"  Action mode: {metadata.action_mode}")
    
    try:
        # Try to get some demos
        demos = demo_store.get_demos(metadata, amount=5, frequency=50)
        
        if demos:
            print(f"\n✅ Found {len(demos)} RBY1 demos")
            
            # Reset environment to get current reset pose
            rby1_env.reset(seed=42)
            
            current_left_pos = rby1_env.robot._wrist_sites[HandSide.LEFT].get_position()
            current_right_pos = rby1_env.robot._wrist_sites[HandSide.RIGHT].get_position()
            
            print(f"\nCurrent reset EE positions:")
            print(f"  Left:  {current_left_pos}")
            print(f"  Right: {current_right_pos}")
            
            # Check initial poses in demos
            print(f"\nInitial EE positions in demos:")
            print("-" * 60)
            
            for i, demo in enumerate(demos[:3]):  # Check first 3 demos
                print(f"\nDemo {i+1}:")
                
                # Apply initial action from demo
                initial_timestep = demo.timesteps[0]
                if hasattr(initial_timestep, 'action'):
                    # Reset and apply initial action
                    obs = rby1_env.reset(seed=demo.seed if hasattr(demo, 'seed') else None)
                    
                    # Apply the initial action
                    rby1_env.step(initial_timestep.action)
                    
                    demo_left_pos = rby1_env.robot._wrist_sites[HandSide.LEFT].get_position()
                    demo_right_pos = rby1_env.robot._wrist_sites[HandSide.RIGHT].get_position()
                    
                    print(f"  Left:  {demo_left_pos}")
                    print(f"  Right: {demo_right_pos}")
                    
                    # Calculate differences
                    left_diff = np.linalg.norm(demo_left_pos - current_left_pos) * 1000
                    right_diff = np.linalg.norm(demo_right_pos - current_right_pos) * 1000
                    
                    print(f"  Difference from reset:")
                    print(f"    Left:  {left_diff:.2f}mm")
                    print(f"    Right: {right_diff:.2f}mm")
        else:
            print("\n⚠️  No RBY1 demos found in the dataset")
            print("   This is expected - RBY1 demos may not exist yet")
            
    except Exception as e:
        print(f"\n⚠️  Could not load RBY1 demos: {e}")
        print("   This is expected - RBY1 demos may not exist yet")
    
    # Also check if there are H1 demos we should be compatible with
    print("\n" + "=" * 80)
    print("CHECKING H1 DEMO COMPATIBILITY:")
    print("=" * 80)
    
    from bigym.robots.configs.h1 import H1
    
    h1_env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
        robot_cls=H1
    )
    
    h1_metadata = Metadata.from_env(h1_env)
    
    try:
        h1_demos = demo_store.get_demos(h1_metadata, amount=3, frequency=50)
        
        if h1_demos:
            print(f"✅ Found {len(h1_demos)} H1 demos")
            
            # Check if RBY1 can match H1 initial poses
            h1_env.reset(seed=42)
            h1_left = h1_env.robot._wrist_sites[HandSide.LEFT].get_position()
            h1_right = h1_env.robot._wrist_sites[HandSide.RIGHT].get_position()
            
            rby1_env.reset(seed=42)
            rby1_left = rby1_env.robot._wrist_sites[HandSide.LEFT].get_position()
            rby1_right = rby1_env.robot._wrist_sites[HandSide.RIGHT].get_position()
            
            left_error = np.linalg.norm(rby1_left - h1_left) * 1000
            right_error = np.linalg.norm(rby1_right - h1_right) * 1000
            
            print(f"\nRBY1 compatibility with H1 demos:")
            print(f"  Position errors (RBY1 vs H1):")
            print(f"    Left:  {left_error:.2f}mm")
            print(f"    Right: {right_error:.2f}mm")
            
            if max(left_error, right_error) < 5.0:
                print("  ✅ RBY1 should be able to replay H1 demos!")
            else:
                print("  ⚠️  Position mismatch may prevent H1 demo replay")
    except:
        print("⚠️  Could not load H1 demos")
    
    h1_env.close()
    rby1_env.close()

if __name__ == "__main__":
    main()