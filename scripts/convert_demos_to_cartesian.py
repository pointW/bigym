"""Convert demonstration data from joint actions to Cartesian actions."""
import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import List
import warnings
from tqdm import tqdm
from pyquaternion import Quaternion

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.envs.move_plates import MovePlate
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.cartesian_action_mode import CartesianActionMode, Pose, rotation_matrix_to_6d
from bigym.const import HandSide
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo


def poses_to_cartesian_action_direct(left_pose: Pose, right_pose: Pose, base_action, gripper_action) -> np.ndarray:
    """Convert poses directly to cartesian action format (bypassing IK).
    
    This creates a cartesian action vector directly from poses without any IK solving.
    Should have ZERO error when converting from FK results.
    
    Action format: [left_pos(3), left_ori(6), right_pos(3), right_ori(6), base(3), grippers(2)]
    """
    action_parts = []
    
    # Left end-effector position (3D)
    action_parts.append(left_pose.position)
    
    # Left end-effector orientation (6D rotation representation)
    left_6d = rotation_matrix_to_6d(left_pose.orientation.rotation_matrix)
    action_parts.append(left_6d)
    
    # Right end-effector position (3D)
    action_parts.append(right_pose.position)
    
    # Right end-effector orientation (6D rotation representation)
    right_6d = rotation_matrix_to_6d(right_pose.orientation.rotation_matrix)
    action_parts.append(right_6d)
    
    # Base action (3D for floating base)
    if base_action is not None:
        action_parts.append(base_action)
    else:
        action_parts.append(np.zeros(3))
    
    # Gripper actions (2D)
    if gripper_action is not None:
        action_parts.append(gripper_action)
    else:
        action_parts.append(np.zeros(2))
    
    return np.concatenate(action_parts)


def convert_joint_demo_to_cartesian(
    original_demo: Demo, 
    cartesian_env: MovePlate,
    joint_env: MovePlate = None  # Unused - we create isolated environment
) -> Demo:
    """Convert a single demo from joint actions to Cartesian actions.
    
    Args:
        original_demo: Original demo with joint actions
        cartesian_env: Environment with CartesianActionMode for target format
        joint_env: Environment with JointPositionActionMode for simulation
        
    Returns:
        New demo with Cartesian actions
    """
    # Get the actions and observations from the original demo timesteps
    joint_actions = np.array([step.executed_action for step in original_demo.timesteps])
    observations = [step.observation for step in original_demo.timesteps]
    
    cartesian_actions = []
    
    print(f"Converting demo with {len(joint_actions)} steps...")
    
    # Create isolated environment for this conversion to avoid state contamination
    isolated_env = MovePlate(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", resolution=(84, 84)),
                CameraConfig("left_wrist", resolution=(84, 84)),
                CameraConfig("right_wrist", resolution=(84, 84)),
            ]
        ),
        render_mode="human",
    )
    
    # Reset environment to initial state with original demo's seed
    # This is CRITICAL to ensure the target positions match the original demo
    isolated_env.reset(seed=original_demo.seed)
    
    # Get floating base DOF count for proper action extraction
    floating_base_dof = isolated_env.robot.floating_base.dof_amount if isolated_env.action_mode.floating_base else 0
    
    for step_idx, joint_action in enumerate(tqdm(joint_actions, desc="Converting steps")):
        # Clip joint action to ensure it's within action space bounds (demos may have slight bound violations)
        joint_action_clipped = np.clip(joint_action, isolated_env.action_space.low, isolated_env.action_space.high)
        
        # Apply joint action to get the resulting end-effector poses (target poses)
        obs, reward, terminated, truncated, info = isolated_env.step(joint_action_clipped)
        
        # Get end-effector poses AFTER stepping (these are the targets the action achieved)
        left_site = isolated_env.robot._wrist_sites[HandSide.LEFT]
        right_site = isolated_env.robot._wrist_sites[HandSide.RIGHT]
        
        left_pos = left_site.get_position()
        left_quat = Quaternion(left_site.get_quaternion())
        left_pose = Pose(left_pos, left_quat)
        
        right_pos = right_site.get_position()
        right_quat = Quaternion(right_site.get_quaternion())
        right_pose = Pose(right_pos, right_quat)
        
        # Extract base and gripper actions DIRECTLY from joint action
        base_action = None
        gripper_action = None
        
        if cartesian_env.action_mode.floating_base:
            # FIX: Copy base action directly from joint action (not pelvis movement delta)
            base_action = joint_action_clipped[:floating_base_dof]
            # Gripper actions are at the end
            gripper_action = joint_action_clipped[-2:]
        else:
            # No floating base - gripper actions are still at the end
            gripper_action = joint_action_clipped[-2:]
            
        # Convert poses to Cartesian action directly (bypassing IK)
        # This should have ZERO error since we're converting FROM FK results
        cartesian_action = poses_to_cartesian_action_direct(
            left_pose, right_pose, base_action, gripper_action
        )
        
        cartesian_actions.append(cartesian_action)
    
    # Close isolated environment
    isolated_env.close()
    
    # Create new demo with Cartesian actions using proper timesteps format
    from demonstrations.demo import DemoStep
    
    timesteps = []
    for i, (cartesian_action, obs) in enumerate(zip(cartesian_actions, observations)):
        # Get additional info from original demo
        original_step = original_demo.timesteps[i]
        timestep = DemoStep(
            observation=obs,
            reward=original_step.reward,
            termination=original_step.termination,
            truncation=original_step.truncation,
            info=original_step.info.copy(),  # Copy original info but replace action
            action=cartesian_action
        )
        timesteps.append(timestep)
    
    # Create metadata with seed from original demo
    cartesian_metadata = Metadata.from_env(cartesian_env)
    cartesian_metadata.seed = original_demo.seed  # Preserve original seed for proper replay
    
    cartesian_demo = Demo(
        metadata=cartesian_metadata,
        timesteps=timesteps,
    )
    
    return cartesian_demo


def convert_demos_batch(
    demo_amount: int = 3,
    output_dir: str = "cartesian_demos"
) -> List[Demo]:
    """Convert a batch of demonstrations to Cartesian format.
    
    Args:
        demo_amount: Number of demos to convert
        output_dir: Directory to save converted demos
        
    Returns:
        List of converted Cartesian demos
    """
    print(f"Converting {demo_amount} demonstrations to Cartesian format...")
    
    # Create environments
    joint_env = MovePlate(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", resolution=(84, 84)),
                CameraConfig("left_wrist", resolution=(84, 84)),
                CameraConfig("right_wrist", resolution=(84, 84)),
            ]
        ),
        render_mode="human",
    )
    
    # Debug: Print action space bounds
    print(f"Joint env action space bounds:")
    print(f"  Low:  {joint_env.action_space.low}")
    print(f"  High: {joint_env.action_space.high}")
    print(f"  Shape: {joint_env.action_space.shape}")
    
    cartesian_env = MovePlate(
        action_mode=CartesianActionMode(floating_base=True),
        control_frequency=50,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", resolution=(84, 84)),
                CameraConfig("left_wrist", resolution=(84, 84)),
                CameraConfig("right_wrist", resolution=(84, 84)),
            ]
        ),
        render_mode="human",
    )
    
    # Load original joint demos (using cached demos if available)
    print("Loading original joint demonstrations...")
    demo_store = DemoStore()
    joint_metadata = Metadata.from_env(joint_env)
    control_frequency = 50
    original_demos = demo_store.get_demos(joint_metadata, amount=demo_amount, frequency=control_frequency)
    
    print(f"Loaded {len(original_demos)} original demos")
    
    # Debug: Check first few actions from first demo
    if original_demos:
        first_demo_actions = np.array([step.executed_action for step in original_demos[0].timesteps])
        print(f"First demo action range:")
        print(f"  Min: {first_demo_actions.min(axis=0)}")
        print(f"  Max: {first_demo_actions.max(axis=0)}")
        print(f"  Shape: {first_demo_actions.shape}")
        
        # Check if any actions are out of bounds
        low_violations = (first_demo_actions < joint_env.action_space.low).any(axis=0)
        high_violations = (first_demo_actions > joint_env.action_space.high).any(axis=0)
        print(f"  Low bound violations: {low_violations.any()} {np.where(low_violations)[0] if low_violations.any() else 'None'}")
        print(f"  High bound violations: {high_violations.any()} {np.where(high_violations)[0] if high_violations.any() else 'None'}")
    
    # Convert each demo
    cartesian_demos = []
    
    for i, original_demo in enumerate(original_demos):
        print(f"\nConverting demo {i+1}/{len(original_demos)}...")
        
        try:
            print(f"Starting conversion of demo {i+1} with {len(original_demo.timesteps)} steps...")
            cartesian_demo = convert_joint_demo_to_cartesian(
                original_demo, cartesian_env, joint_env
            )
            cartesian_demos.append(cartesian_demo)
            
            # Save converted demo
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            demo_filename = f"cartesian_demo_{i:03d}.safetensors"
            demo_path = output_path / demo_filename
            print(f"Saving demo to {demo_path}...")
            cartesian_demo.save(demo_path)
            
            print(f"✓ Successfully saved converted demo to {demo_path}")
            print(f"  Original demo steps: {len(original_demo.timesteps)}")
            print(f"  Converted demo steps: {len(cartesian_demo.timesteps)}")
            
        except Exception as e:
            print(f"❌ Error converting demo {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    joint_env.close()
    cartesian_env.close()
    
    print(f"\nSuccessfully converted {len(cartesian_demos)}/{len(original_demos)} demonstrations")
    return cartesian_demos


def analyze_action_spaces():
    """Analyze and compare joint vs Cartesian action spaces."""
    print("Analyzing action spaces...")
    
    # Joint action mode
    joint_env = MovePlate(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    )
    
    # Cartesian action mode  
    cartesian_env = MovePlate(
        action_mode=CartesianActionMode(floating_base=True),
    )
    
    print("\nJoint Action Space:")
    print(f"  Shape: {joint_env.action_space.shape}")
    print(f"  Low bounds: {joint_env.action_space.low}")
    print(f"  High bounds: {joint_env.action_space.high}")
    
    print("\nCartesian Action Space:")
    print(f"  Shape: {cartesian_env.action_space.shape}")
    print(f"  Low bounds: {cartesian_env.action_space.low}")
    print(f"  High bounds: {cartesian_env.action_space.high}")
    
    print("\nCartesian action space breakdown:")
    print("  Left EE position: [0:3]")
    print("  Left EE rotation (6D): [3:9]") 
    print("  Right EE position: [9:12]")
    print("  Right EE rotation (6D): [12:18]")
    print("  Base control: [18:21]")
    print("  Gripper control: [21:23]")
    
    joint_env.close()
    cartesian_env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert joint demos to Cartesian demos")
    parser.add_argument("--max-demos", type=int, default=60, help="Maximum number of demos to convert")
    parser.add_argument("--output-dir", type=str, default="cartesian_demos_move_plate", help="Output directory for converted demos")
    args = parser.parse_args()
    
    # First analyze the action spaces
    analyze_action_spaces()
    
    # Convert some demos
    print("\n" + "="*50)
    converted_demos = convert_demos_batch(demo_amount=args.max_demos, output_dir=args.output_dir)
    
    print(f"\nConversion complete! Converted {len(converted_demos)} demos.")
    print(f"Converted demos saved in '{args.output_dir}/' directory")