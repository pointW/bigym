"""Convert demonstration data from joint actions to Cartesian actions.

This refactored version supports different environment types dynamically.
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


def poses_to_cartesian_action_direct(
    left_pose: Pose, 
    right_pose: Pose, 
    base_action: Optional[np.ndarray], 
    gripper_action: Optional[np.ndarray]
) -> np.ndarray:
    """Convert poses directly to cartesian action format (bypassing IK).
    
    Action format: [left_pos(3), left_ori(6), right_pos(3), right_ori(6), base(3), grippers(2)]
    """
    action_parts = []
    
    # Left end-effector
    action_parts.append(left_pose.position)
    action_parts.append(rotation_matrix_to_6d(left_pose.orientation.rotation_matrix))
    
    # Right end-effector
    action_parts.append(right_pose.position)
    action_parts.append(rotation_matrix_to_6d(right_pose.orientation.rotation_matrix))
    
    # Base action
    action_parts.append(base_action if base_action is not None else np.zeros(3))
    
    # Gripper actions
    action_parts.append(gripper_action if gripper_action is not None else np.zeros(2))
    
    return np.concatenate(action_parts)


def convert_joint_demo_to_cartesian(
    original_demo: Demo,
    env_class: Type,
    camera_configs: List[CameraConfig],
    control_frequency: int = 50,
    render_mode: Optional[str] = None
) -> Demo:
    """Convert a single demo from joint actions to Cartesian actions.
    
    Args:
        original_demo: Original demo with joint actions
        env_class: Environment class to use
        camera_configs: Camera configurations for the environment
        control_frequency: Control frequency for the environment
        render_mode: Render mode (None for headless)
        
    Returns:
        New demo with Cartesian actions
    """
    # Get the actions and observations from the original demo
    joint_actions = np.array([step.executed_action for step in original_demo.timesteps])
    observations = [step.observation for step in original_demo.timesteps]
    
    cartesian_actions = []
    
    print(f"Converting demo with {len(joint_actions)} steps...")
    
    # Create isolated environment for conversion
    isolated_env = env_class(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=render_mode,
    )
    
    # Reset with original seed
    isolated_env.reset(seed=original_demo.seed)
    
    # Get floating base DOF count
    floating_base_dof = (
        isolated_env.robot.floating_base.dof_amount 
        if isolated_env.action_mode.floating_base else 0
    )
    
    for step_idx, joint_action in enumerate(tqdm(joint_actions, desc="Converting steps")):
        # Clip joint action to bounds
        joint_action_clipped = np.clip(
            joint_action, 
            isolated_env.action_space.low, 
            isolated_env.action_space.high
        )
        
        # Step to get achieved poses
        obs, reward, terminated, truncated, info = isolated_env.step(joint_action_clipped)
        
        # Get end-effector poses after stepping
        left_site = isolated_env.robot._wrist_sites[HandSide.LEFT]
        right_site = isolated_env.robot._wrist_sites[HandSide.RIGHT]
        
        left_pose = Pose(
            left_site.get_position(),
            Quaternion(left_site.get_quaternion())
        )
        
        right_pose = Pose(
            right_site.get_position(),
            Quaternion(right_site.get_quaternion())
        )
        
        # Extract base and gripper actions
        base_action = None
        gripper_action = None
        
        if floating_base_dof > 0:
            base_action = joint_action_clipped[:floating_base_dof]
            gripper_action = joint_action_clipped[-2:]
        else:
            gripper_action = joint_action_clipped[-2:]
        
        # Convert to Cartesian action
        cartesian_action = poses_to_cartesian_action_direct(
            left_pose, right_pose, base_action, gripper_action
        )
        
        cartesian_actions.append(cartesian_action)
    
    isolated_env.close()
    
    # Create new demo with Cartesian actions
    from demonstrations.demo import DemoStep
    
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
    
    # Create Cartesian environment for metadata
    cartesian_env = env_class(
        action_mode=CartesianActionMode(floating_base=True),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=None,
    )
    
    cartesian_metadata = Metadata.from_env(cartesian_env)
    cartesian_metadata.seed = original_demo.seed
    cartesian_env.close()
    
    return Demo(metadata=cartesian_metadata, timesteps=timesteps)


def convert_demos_batch(
    env_name: str,
    demo_amount: int = 3,
    output_dir: str = None,
    camera_configs: Optional[List[CameraConfig]] = None,
    control_frequency: int = 50,
    render_mode: Optional[str] = None
) -> List[Demo]:
    """Convert a batch of demonstrations to Cartesian format.
    
    Args:
        env_name: Name of the environment (e.g., 'ReachTarget', 'MovePlate')
        demo_amount: Number of demos to convert
        output_dir: Directory to save converted demos (auto-generated if None)
        camera_configs: Camera configurations (uses defaults if None)
        control_frequency: Control frequency
        render_mode: Render mode for conversion
        
    Returns:
        List of converted Cartesian demos
    """
    # Get environment class
    env_class = get_environment_class(env_name)
    
    # Use default camera config if not provided
    if camera_configs is None:
        camera_configs = get_default_camera_config(env_name)
    
    # Auto-generate output directory name if not provided
    if output_dir is None:
        output_dir = f"cartesian_demos_{env_name.lower()}"
    
    print(f"Converting {demo_amount} {env_name} demonstrations to Cartesian format...")
    print(f"Output directory: {output_dir}")
    
    # Create environment to load demos
    joint_env = env_class(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=render_mode,
    )
    
    # Load original demos
    print("Loading original joint demonstrations...")
    demo_store = DemoStore()
    joint_metadata = Metadata.from_env(joint_env)
    original_demos = demo_store.get_demos(
        joint_metadata, 
        amount=demo_amount, 
        frequency=control_frequency
    )
    
    joint_env.close()
    
    print(f"Loaded {len(original_demos)} original demos")
    
    # Convert each demo
    cartesian_demos = []
    
    for i, original_demo in enumerate(original_demos):
        print(f"\nConverting demo {i+1}/{len(original_demos)}...")
        
        try:
            cartesian_demo = convert_joint_demo_to_cartesian(
                original_demo,
                env_class,
                camera_configs,
                control_frequency,
                render_mode
            )
            cartesian_demos.append(cartesian_demo)
            
            # Save converted demo
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            demo_path = output_path / f"cartesian_demo_{i:03d}.safetensors"
            print(f"Saving demo to {demo_path}...")
            cartesian_demo.save(demo_path)
            
            print(f"✓ Successfully saved converted demo")
            
        except Exception as e:
            print(f"❌ Error converting demo {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nSuccessfully converted {len(cartesian_demos)}/{len(original_demos)} demonstrations")
    return cartesian_demos


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert joint demos to Cartesian demos for any environment"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default="ReachTarget",
        help="Environment name (e.g., ReachTarget, MovePlate)"
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
    
    args = parser.parse_args()
    
    # Convert demos
    converted_demos = convert_demos_batch(
        env_name=args.env,
        demo_amount=args.max_demos,
        output_dir=args.output_dir,
        control_frequency=args.control_freq,
        render_mode="human" if args.render else None
    )
    
    print(f"\nConversion complete! Converted {len(converted_demos)} demos.")
    if args.output_dir:
        print(f"Converted demos saved in '{args.output_dir}/' directory")
    else:
        print(f"Converted demos saved in 'cartesian_demos_{args.env.lower()}/' directory")


if __name__ == "__main__":
    main()