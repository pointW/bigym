"""Replay Cartesian demonstrations in the environment.

Similar to replay_demo.py but for Cartesian action space demonstrations.
"""
import numpy as np
from pathlib import Path

import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from bigym.cartesian_action_mode import CartesianActionMode, Pose
from bigym.envs.reach_target import ReachTarget
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo import Demo


def replay_cartesian_demo_from_file(demo_path: str, track_ik_errors: bool = True):
    """Replay a Cartesian demo from a saved file with IK error tracking.
    
    Args:
        demo_path: Path to the saved Cartesian demonstration file
        track_ik_errors: Whether to track and display IK solver errors
    """
    print(f"🎬 Loading Cartesian demo from {demo_path}...")
    
    # Check if it's a converted demo (safetensors) or original demo format
    demo_path = Path(demo_path)
    
    if demo_path.suffix == '.safetensors' and 'cartesian_demo_' in demo_path.name:
        # This is a converted cartesian demo - load it directly
        from safetensors.numpy import load_file
        demo_data = load_file(demo_path)
        
        # Find action key
        action_key = None
        for key in demo_data.keys():
            if 'action' in key.lower():
                action_key = key
                break
        
        if not action_key:
            print(f"❌ No action key found in {demo_path}")
            return
        
        actions = demo_data[action_key]
        print(f"📊 Loaded converted demo with {len(actions)} actions")
        
    else:
        # Original demo format
        demo = Demo.from_safetensors(demo_path)
        actions = np.array([step.executed_action for step in demo.timesteps])
        print(f"📊 Loaded original demo with {len(actions)} actions")
    
    print(f"Action space shape: {actions.shape}")
    print(f"Action space breakdown:")
    print(f"  Total dimensions: {actions.shape[1]}")
    print(f"  Left EE position: [0:3]")
    print(f"  Left EE rotation (6D): [3:9]") 
    print(f"  Right EE position: [9:12]")
    print(f"  Right EE rotation (6D): [12:18]")
    print(f"  Base control: [18:21]")
    print(f"  Gripper control: [21:23]")
    
    # Create environment with matching action mode
    control_frequency = 50
    env = ReachTarget(
        action_mode=CartesianActionMode(floating_base=True),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", resolution=(84, 84)),
                CameraConfig("left_wrist", resolution=(84, 84)),
                CameraConfig("right_wrist", resolution=(84, 84)),
            ]
        ),
        render_mode="human",
    )
    
    print(f"\nEnvironment action space: {env.action_space.shape}")
    
    # Reset environment
    env.reset()
    
    if track_ik_errors:
        print("\n🔍 Starting replay with IK error tracking...")
        print("💡 Watch the visual - target poses vs achieved poses")
        ik_errors = []
        
        for step_idx, action in enumerate(actions):
            # Extract target poses from action
            target_left_pos = action[0:3]
            target_right_pos = action[9:12]
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get actual achieved poses
            from bigym.const import HandSide
            actual_left_pos = env.robot._wrist_sites[HandSide.LEFT].get_position()
            actual_right_pos = env.robot._wrist_sites[HandSide.RIGHT].get_position()
            
            # Calculate IK errors
            left_error = np.linalg.norm(actual_left_pos - target_left_pos)
            right_error = np.linalg.norm(actual_right_pos - target_right_pos)
            total_error = left_error + right_error
            ik_errors.append(total_error)
            
            # Print progress and errors every 20 steps
            if step_idx % 20 == 0 or step_idx < 5:
                print(f"  Step {step_idx:3d}: L_error={left_error*1000:5.1f}mm R_error={right_error*1000:5.1f}mm Total={total_error*1000:5.1f}mm")
            
            # Render
            env.render()
            
            # Check for completion
            if info.get('task_success', False):
                print(f"✅ Task completed at step {step_idx}")
                break
            
            if terminated or truncated:
                print(f"🏁 Episode ended at step {step_idx}")
                break
        
        # Print IK performance summary
        print(f"\n📈 IK SOLVER PERFORMANCE SUMMARY:")
        print(f"  Steps analyzed: {len(ik_errors)}")
        print(f"  Average error: {np.mean(ik_errors)*1000:.1f}mm")
        print(f"  Maximum error: {np.max(ik_errors)*1000:.1f}mm")
        print(f"  Minimum error: {np.min(ik_errors)*1000:.1f}mm")
        print(f"  Final error: {ik_errors[-1]*1000:.1f}mm")
        print(f"  Error std dev: {np.std(ik_errors)*1000:.1f}mm")
        
        # Analyze error progression
        if len(ik_errors) > 10:
            first_10_avg = np.mean(ik_errors[:10])
            last_10_avg = np.mean(ik_errors[-10:])
            error_drift = last_10_avg - first_10_avg
            
            print(f"  Error drift: {error_drift*1000:+.1f}mm (last 10 vs first 10 steps)")
            
            if error_drift > 0.01:  # 10mm
                print(f"  ⚠️  Significant error accumulation detected!")
            elif error_drift < -0.01:
                print(f"  ✅ Error decreased over time")
            else:
                print(f"  ➡️  Error remained stable")
    
    else:
        # Simple replay without tracking
        print("\n🎬 Starting simple replay...")
        env.reset()
        
        for step_idx, action in enumerate(actions):
            env.step(action)
            env.render()
            
            if step_idx % 50 == 0:
                print(f"  Step {step_idx}/{len(actions)}")
    
    env.close()
    print(f"\n🎬 Replay complete!")


def replay_converted_demos(demos_dir: str = "cartesian_demos_final", demo_index: int = None):
    """Replay converted Cartesian demos in a directory.
    
    Args:
        demos_dir: Directory containing converted Cartesian demos
        demo_index: Specific demo index to replay (None for interactive selection)
    """
    demos_path = Path(demos_dir)
    
    if not demos_path.exists():
        print(f"❌ Demos directory '{demos_dir}' not found!")
        print("Available directories:")
        for path in ["cartesian_demos", "cartesian_demos_corrected", "cartesian_demos_final"]:
            if Path(path).exists():
                print(f"  ✓ {path}/")
            else:
                print(f"  ❌ {path}/")
        print("\nPlease run convert_demos_to_cartesian.py first to create Cartesian demos.")
        return
    
    # Find all demo files
    demo_files = sorted(list(demos_path.glob("*.safetensors")))
    
    if not demo_files:
        print(f"❌ No demo files found in '{demos_dir}'!")
        return
    
    print(f"📋 Found {len(demo_files)} demo files in {demos_dir}/:")
    for i, demo_file in enumerate(demo_files):
        # Try to get demo length info
        try:
            from safetensors.numpy import load_file
            demo_data = load_file(demo_file)
            action_key = None
            for key in demo_data.keys():
                if 'action' in key.lower():
                    action_key = key
                    break
            if action_key:
                actions = demo_data[action_key]
                print(f"  {i}: {demo_file.name} ({len(actions)} steps)")
            else:
                print(f"  {i}: {demo_file.name} (no actions)")
        except Exception:
            print(f"  {i}: {demo_file.name}")
    
    # Handle specific demo index
    if demo_index is not None:
        if 0 <= demo_index < len(demo_files):
            print(f"\n🎬 Replaying demo {demo_index}: {demo_files[demo_index].name}")
            replay_cartesian_demo_from_file(str(demo_files[demo_index]))
        else:
            print(f"❌ Invalid demo index {demo_index}. Available: 0-{len(demo_files)-1}")
        return
    
    # Interactive selection
    while True:
        try:
            choice = input(f"\nEnter demo number (0-{len(demo_files)-1}) or 'q' to quit: ")
            if choice.lower() == 'q':
                break
                
            demo_idx = int(choice)
            if 0 <= demo_idx < len(demo_files):
                replay_cartesian_demo_from_file(str(demo_files[demo_idx]))
                break
            else:
                print(f"Invalid choice. Please enter 0-{len(demo_files)-1}")
                
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def create_simple_cartesian_demo():
    """Create a simple test Cartesian demo by recording basic actions."""
    print("Creating a simple test Cartesian demo...")
    
    env = ReachTarget(
        action_mode=CartesianActionMode(floating_base=True),
        render_mode=None,
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Record a few simple actions
    actions = []
    observations = []
    
    # Get initial end-effector poses
    initial_poses = env.action_mode.get_current_ee_poses()
    left_pose, right_pose = initial_poses
    
    print(f"Initial left EE position: {left_pose.position}")
    print(f"Initial right EE position: {right_pose.position}")
    
    # Create 10 actions with small movements
    for step in range(10):
        # Move end-effectors in small increments
        left_pos = left_pose.position + np.array([0.01 * step, 0.005 * step, 0.0])
        right_pos = right_pose.position + np.array([0.01 * step, -0.005 * step, 0.0])
        
        # Keep original orientations
        left_new_pose = Pose(left_pos, left_pose.orientation)
        right_new_pose = Pose(right_pos, right_pose.orientation)
        
        # Convert to action
        action = env.action_mode.poses_to_action(
            left_new_pose, right_new_pose,
            base_action=np.array([0.0, 0.0, 0.0]),  # No base movement
            gripper_action=np.array([0.5, 0.5])     # Half-open grippers
        )
        
        actions.append(action)
        
        # Execute and record observation
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        
        if terminated or truncated:
            break
    
    # Create demo with proper timesteps format
    from demonstrations.utils import Metadata
    from demonstrations.demo import DemoStep
    
    # Convert actions and observations to timesteps
    timesteps = []
    for i, (action, obs) in enumerate(zip(actions, observations)):
        timestep = DemoStep(
            observation=obs,
            reward=0.0,  # Simple demo has no reward
            termination=False,
            truncation=(i == len(actions) - 1),  # Last step is truncated
            info={},
            action=action
        )
        timesteps.append(timestep)
    
    demo = Demo(
        metadata=Metadata.from_env(env),
        timesteps=timesteps,
    )
    
    # Save demo
    output_path = Path("cartesian_demos")
    output_path.mkdir(exist_ok=True)
    demo_path = output_path / "simple_test_demo.safetensors"
    demo.save(demo_path)
    
    print(f"Created and saved simple demo to {demo_path}")
    print(f"Demo has {len(actions)} actions with shape {np.array(actions).shape}")
    
    env.close()
    return str(demo_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replay Cartesian demonstrations with IK error tracking")
    parser.add_argument(
        "--demo-file", 
        type=str, 
        help="Path to a specific Cartesian demo file to replay"
    )
    parser.add_argument(
        "--demos-dir", 
        type=str, 
        default="cartesian_demos_target",
        help="Directory containing Cartesian demos (default: cartesian_demos_final)"
    )
    parser.add_argument(
        "--demo-index",
        type=int,
        help="Specific demo index to replay (0, 1, 2, ...)"
    )
    parser.add_argument(
        "--no-ik-tracking",
        action="store_true",
        help="Disable IK error tracking for faster replay"
    )
    parser.add_argument(
        "--create-test", 
        action="store_true",
        help="Create a simple test demo first"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available demos and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        # Just list available demos without replaying
        demos_path = Path(args.demos_dir)
        if demos_path.exists():
            demo_files = sorted(list(demos_path.glob("*.safetensors")))
            print(f"📋 Available demos in {args.demos_dir}/:")
            for i, demo_file in enumerate(demo_files):
                print(f"  {i}: {demo_file.name}")
        else:
            print(f"❌ Directory {args.demos_dir}/ not found")
    elif args.create_test:
        demo_path = create_simple_cartesian_demo()
        print(f"\nNow replaying the created test demo...")
        replay_cartesian_demo_from_file(demo_path, track_ik_errors=not args.no_ik_tracking)
    elif args.demo_file:
        replay_cartesian_demo_from_file(args.demo_file, track_ik_errors=not args.no_ik_tracking)
    else:
        replay_converted_demos(args.demos_dir, args.demo_index)