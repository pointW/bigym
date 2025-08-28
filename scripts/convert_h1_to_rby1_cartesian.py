"""Convert H1 joint demonstrations to RBY1 Cartesian demonstrations.

Since RBY1 uses whole-body IK that optimizes base movement together with joints,
the Cartesian action only needs end-effector poses and gripper actions.
The base/torso movements are automatically handled by the whole-body IK solver.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import List, Type, Optional, Dict, Any
import importlib
from tqdm import tqdm
from pyquaternion import Quaternion

from bigym.action_modes import JointPositionActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.cartesian_action_mode import CartesianActionMode, Pose, rotation_matrix_to_6d
from bigym.const import HandSide
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo, DemoStep
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


def poses_to_rby1_cartesian_action(
    left_pose: Pose, 
    right_pose: Pose, 
    gripper_action: Optional[np.ndarray]
) -> np.ndarray:
    """Convert poses to RBY1 Cartesian action format.
    
    For RBY1 with whole-body IK, we only need:
    - Left end-effector pose (9D: 3 pos + 6 ori)
    - Right end-effector pose (9D: 3 pos + 6 ori)
    - Gripper actions (2D)
    
    Total: 20D action
    
    The whole-body IK solver will automatically optimize base and torso movements.
    """
    action_parts = []
    
    # Left end-effector
    action_parts.append(left_pose.position)
    action_parts.append(rotation_matrix_to_6d(left_pose.orientation.rotation_matrix))
    
    # Right end-effector
    action_parts.append(right_pose.position)
    action_parts.append(rotation_matrix_to_6d(right_pose.orientation.rotation_matrix))
    
    # Gripper actions
    action_parts.append(gripper_action if gripper_action is not None else np.zeros(2))
    
    return np.concatenate(action_parts)


def convert_h1_demo_to_rby1_cartesian(
    original_demo: Demo,
    env_class: Type,
    camera_configs: List[CameraConfig],
    control_frequency: int = 50,
    render_mode: Optional[str] = None,
    robot_type: str = "rby1"
) -> Demo:
    """Convert a single H1 joint demo to RBY1 Cartesian demo.
    
    Args:
        original_demo: Original H1 demo with joint actions
        env_class: Environment class to use
        camera_configs: Camera configurations for the environment
        control_frequency: Control frequency for the environment
        render_mode: Render mode (None for headless)
        robot_type: Target robot type (should be "rby1")
        
    Returns:
        New demo with RBY1 Cartesian actions
    """
    # Get the actions and observations from the original demo
    joint_actions = np.array([step.executed_action for step in original_demo.timesteps])
    observations = [step.observation for step in original_demo.timesteps]
    
    cartesian_actions = []
    
    print(f"Converting H1 demo with {len(joint_actions)} steps to RBY1 Cartesian format...")
    
    # Create H1 environment for replaying the demo
    h1_env = env_class(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=render_mode,
        robot_cls=H1  # Use H1 robot class to replay original demo
    )
    
    # Reset with original seed
    h1_env.reset(seed=original_demo.seed)
    
    # Get floating base DOF count for H1
    floating_base_dof = (
        h1_env.robot.floating_base.dof_amount 
        if h1_env.action_mode.floating_base else 0
    )
    
    for step_idx, joint_action in enumerate(tqdm(joint_actions, desc="Converting steps")):
        # Clip joint action to H1 bounds
        joint_action_clipped = np.clip(
            joint_action, 
            h1_env.action_space.low, 
            h1_env.action_space.high
        )
        
        # Step H1 environment to get achieved poses
        obs, reward, terminated, truncated, info = h1_env.step(joint_action_clipped)
        
        # Get end-effector poses from H1 after stepping
        left_site = h1_env.robot._wrist_sites[HandSide.LEFT]
        right_site = h1_env.robot._wrist_sites[HandSide.RIGHT]
        
        left_pose = Pose(
            left_site.get_position(),
            Quaternion(left_site.get_quaternion())
        )
        
        right_pose = Pose(
            right_site.get_position(),
            Quaternion(right_site.get_quaternion())
        )
        
        # Extract gripper actions from H1 joint action
        gripper_action = joint_action_clipped[-2:]
        
        # Convert to RBY1 Cartesian action (no base/torso needed)
        rby1_cartesian_action = poses_to_rby1_cartesian_action(
            left_pose, right_pose, gripper_action
        )
        
        cartesian_actions.append(rby1_cartesian_action)
    
    h1_env.close()
    
    # Create new demo with RBY1 Cartesian actions
    timesteps = []
    for i, (cartesian_action, obs) in enumerate(zip(cartesian_actions, observations)):
        original_step = original_demo.timesteps[i]
        timestep = DemoStep(
            observation=obs,
            reward=original_step.reward,
            termination=original_step.termination,
            truncation=original_step.truncation,
            info=original_step.info.copy(),
            action=cartesian_action
        )
        timesteps.append(timestep)
    
    # Create RBY1 Cartesian environment for metadata
    # Use empty camera config to avoid camera name conflicts
    rby1_env = env_class(
        action_mode=CartesianActionMode(floating_base=False),  # RBY1 whole-body IK handles base
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=[]),  # Empty cameras to avoid conflicts
        render_mode=None,
        robot_cls=RBY1  # Use RBY1 robot class for target metadata
    )
    
    rby1_metadata = Metadata.from_env(rby1_env)
    rby1_metadata.seed = original_demo.seed
    rby1_env.close()
    
    return Demo(metadata=rby1_metadata, timesteps=timesteps)


def convert_h1_demos_batch(
    env_name: str,
    demo_amount: int = 3,
    output_dir: str = None,
    camera_configs: Optional[List[CameraConfig]] = None,
    control_frequency: int = 50,
    render_mode: Optional[str] = None,
    robot_type: str = "rby1"
) -> List[Demo]:
    """Convert a batch of H1 demonstrations to RBY1 Cartesian format.
    
    Args:
        env_name: Name of the environment (e.g., 'ReachTarget', 'MovePlate')
        demo_amount: Number of demos to convert
        output_dir: Directory to save converted demos (auto-generated if None)
        camera_configs: Camera configurations (uses defaults if None)
        control_frequency: Control frequency
        render_mode: Render mode for conversion
        robot_type: Target robot type (default "rby1")
        
    Returns:
        List of converted RBY1 Cartesian demos
    """
    # Get environment class
    env_class = get_environment_class(env_name)
    
    # Use default camera config if not provided
    if camera_configs is None:
        camera_configs = get_default_camera_config(env_name)
    
    # Auto-generate output directory name if not provided
    if output_dir is None:
        output_dir = f"rby1_cartesian_demos_{env_name.lower()}"
    
    print(f"Converting {demo_amount} H1 {env_name} demonstrations to RBY1 Cartesian format...")
    print(f"Output directory: {output_dir}")
    
    # Create H1 environment to load demos
    h1_env = env_class(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=render_mode,
        robot_cls=H1  # Load H1 demos with H1 robot class
    )
    
    # Load original H1 demos
    print("Loading original H1 joint demonstrations...")
    demo_store = DemoStore()
    h1_metadata = Metadata.from_env(h1_env)
    original_demos = demo_store.get_demos(
        h1_metadata, 
        amount=demo_amount, 
        frequency=control_frequency
    )
    
    h1_env.close()
    
    print(f"Loaded {len(original_demos)} original H1 demos")
    
    # Convert each demo
    rby1_cartesian_demos = []
    
    for i, original_demo in enumerate(original_demos):
        print(f"\nConverting H1 demo {i+1}/{len(original_demos)} to RBY1 Cartesian...")
        
        try:
            rby1_demo = convert_h1_demo_to_rby1_cartesian(
                original_demo,
                env_class,
                camera_configs,
                control_frequency,
                render_mode,
                robot_type
            )
            rby1_cartesian_demos.append(rby1_demo)
            
            # Save converted demo
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            demo_path = output_path / f"rby1_cartesian_demo_{i:03d}.safetensors"
            print(f"Saving RBY1 demo to {demo_path}...")
            rby1_demo.save(demo_path)
            
            print(f"✓ Successfully saved RBY1 Cartesian demo")
            
        except Exception as e:
            print(f"❌ Error converting H1 demo {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nSuccessfully converted {len(rby1_cartesian_demos)}/{len(original_demos)} H1 demonstrations to RBY1")
    return rby1_cartesian_demos


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert H1 joint demos to RBY1 Cartesian demos"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default="MovePlate",
        help="Environment name (e.g., ReachTarget, MovePlate, PickCube)"
    )
    parser.add_argument(
        "--max-demos", 
        type=int, 
        default=60, 
        help="Maximum number of demos to convert"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=50,
        help="Control frequency (default: 50)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during conversion"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="rby1",
        help="Target robot type (default: rby1)"
    )
    
    args = parser.parse_args()
    
    # Convert H1 demos to RBY1 Cartesian
    converted_demos = convert_h1_demos_batch(
        env_name=args.env,
        demo_amount=args.max_demos,
        output_dir=args.output_dir,
        control_frequency=args.control_freq,
        render_mode="human" if args.render else None,
        robot_type=args.robot
    )
    
    print(f"\nConversion complete! Converted {len(converted_demos)} H1 demos to RBY1 Cartesian format.")
    if args.output_dir:
        print(f"RBY1 Cartesian demos saved in '{args.output_dir}/' directory")
    else:
        print(f"RBY1 Cartesian demos saved in 'rby1_cartesian_demos_{args.env.lower()}/' directory")


if __name__ == "__main__":
    main()