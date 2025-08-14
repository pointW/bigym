"""Convert demonstration data from joint actions to Cartesian actions using TARGET poses.

This version uses the target end-effector poses computed from joint action targets,
rather than the actual achieved poses after stepping.
"""
import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import List
import warnings
from tqdm import tqdm
from pyquaternion import Quaternion

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.cartesian_action_mode import CartesianActionMode
from bigym.const import HandSide
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo
from vr.ik.h1_upper_body_ik import Pose


def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D rotation representation."""
    return rotation_matrix[:2, :].flatten()


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


def compute_target_poses_from_joint_action(
    joint_action: np.ndarray,
    env: ReachTarget,
    current_joint_positions: np.ndarray
) -> tuple[Pose, Pose]:
    """Compute target end-effector poses from joint action targets using forward kinematics.
    
    This computes what the end-effector poses WOULD BE if the joints instantly
    reached their target positions.
    
    Args:
        joint_action: The joint action containing target joint positions
        env: Environment for accessing robot model
        current_joint_positions: Current joint positions (for reference)
        
    Returns:
        Tuple of (left_pose, right_pose) that would result from target joints
    """
    # Extract joint targets from action (assuming absolute control)
    floating_base_dof = env.robot.floating_base.dof_amount if env.action_mode.floating_base else 0
    num_limb_actuators = len(env.robot.limb_actuators)
    
    # Get target joint positions from action
    target_joints = joint_action[floating_base_dof:floating_base_dof + num_limb_actuators]
    
    # Create a temporary physics state to compute forward kinematics
    # We need to set joints to target positions and compute resulting EE poses
    # WITHOUT actually stepping the simulation
    
    # Save current state
    saved_qpos = env.robot.qpos_actuated.copy()
    
    # Set joints to target positions directly (bypassing dynamics)
    for i, actuator in enumerate(env.robot.limb_actuators):
        if actuator.joint:
            joint = env.mojo.physics.bind(actuator.joint)
            joint.qpos = target_joints[i]
    
    # Update kinematics (forward kinematics only, no dynamics)
    env.mojo.physics.forward()
    
    # Get end-effector poses at target configuration
    left_site = env.robot._wrist_sites[HandSide.LEFT]
    right_site = env.robot._wrist_sites[HandSide.RIGHT]
    
    left_pos = left_site.get_position().copy()
    left_quat = Quaternion(left_site.get_quaternion())
    left_pose = Pose(left_pos, left_quat)
    
    right_pos = right_site.get_position().copy()
    right_quat = Quaternion(right_site.get_quaternion())
    right_pose = Pose(right_pos, right_quat)
    
    # Restore original joint positions
    for i, actuator in enumerate(env.robot.limb_actuators):
        if actuator.joint:
            joint = env.mojo.physics.bind(actuator.joint)
            joint.qpos = saved_qpos[floating_base_dof + i]
    
    # Update kinematics back to original state
    env.mojo.physics.forward()
    
    return left_pose, right_pose


def convert_joint_demo_to_cartesian_target(
    original_demo: Demo, 
    cartesian_env: ReachTarget,
    joint_env: ReachTarget = None  # Unused - we create isolated environment
) -> Demo:
    """Convert a single demo from joint actions to Cartesian actions using TARGET poses.
    
    This version uses the target end-effector poses that WOULD be achieved if
    the joints instantly reached their commanded positions, rather than the
    actual achieved poses after dynamics simulation.
    
    Args:
        original_demo: Original demo with joint actions
        cartesian_env: Environment with CartesianActionMode for target format
        joint_env: Environment with JointPositionActionMode for simulation
        
    Returns:
        New demo with Cartesian actions based on target poses
    """
    # Get the actions and observations from the original demo timesteps
    joint_actions = np.array([step.info['demo_action'] for step in original_demo.timesteps])
    observations = [step.observation for step in original_demo.timesteps]
    
    cartesian_actions = []
    
    print(f"Converting demo with {len(joint_actions)} steps using TARGET poses...")
    
    # Create isolated environment for this conversion to avoid state contamination
    isolated_env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    # Reset environment to initial state with original demo's seed
    # This is CRITICAL to ensure the target positions match the original demo
    isolated_env.reset(seed=original_demo.seed)
    
    # Track pelvis position for base action computation
    prev_pelvis_pos = isolated_env.robot.pelvis.get_position().copy()
    
    for step_idx, joint_action in enumerate(tqdm(joint_actions, desc="Converting steps")):
        # Clip joint action to ensure it's within action space bounds
        joint_action_clipped = np.clip(joint_action, isolated_env.action_space.low, isolated_env.action_space.high)
        
        # Get current joint positions before stepping
        floating_base_dof = isolated_env.robot.floating_base.dof_amount if isolated_env.action_mode.floating_base else 0
        num_limb_actuators = len(isolated_env.robot.limb_actuators)
        current_joints = np.array(isolated_env.robot.qpos_actuated[floating_base_dof:floating_base_dof+num_limb_actuators])
        
        # Compute TARGET poses from joint action (what we're commanding)
        target_left_pose, target_right_pose = compute_target_poses_from_joint_action(
            joint_action_clipped, isolated_env, current_joints
        )
        
        # Now step the environment to maintain state consistency
        # (but we use the target poses, not the achieved poses)
        obs, reward, terminated, truncated, info = isolated_env.step(joint_action_clipped)
        
        # Get current pelvis position AFTER stepping (for base action)
        current_pelvis_pos = isolated_env.robot.pelvis.get_position()
        
        # Compute base action
        base_action = None
        gripper_action = None
        
        if cartesian_env.action_mode.floating_base:
            # For base, we still use the actual movement since base doesn't have the same issues
            pelvis_movement = current_pelvis_pos - prev_pelvis_pos
            base_action = pelvis_movement
            # Gripper actions are at the end
            gripper_action = joint_action[-2:]
        else:
            # No floating base - gripper actions are still at the end
            gripper_action = joint_action[-2:]
        
        # Update previous pelvis position for next iteration
        prev_pelvis_pos = current_pelvis_pos.copy()
            
        # Convert TARGET poses to Cartesian action
        cartesian_action = poses_to_cartesian_action_direct(
            target_left_pose, target_right_pose, base_action, gripper_action
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
        
        # Create new info dict with both original and cartesian actions
        new_info = original_step.info.copy()
        new_info['demo_action'] = cartesian_action  # Store cartesian action
        new_info['original_joint_action'] = joint_actions[i]  # Keep original for reference
        
        timestep = DemoStep(
            observation=obs,
            reward=original_step.reward,
            termination=original_step.termination,
            truncation=original_step.truncation,
            info=new_info,
            action=cartesian_action  # Add the action parameter
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
    output_dir: str = "cartesian_demos_target"
) -> List[Demo]:
    """Convert a batch of demonstrations to Cartesian format using target poses.
    
    Args:
        demo_amount: Number of demos to convert
        output_dir: Directory to save converted demos
        
    Returns:
        List of converted Cartesian demos
    """
    print(f"Converting {demo_amount} demonstrations to Cartesian format (using TARGET poses)...")
    
    # Create environments
    joint_env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(
                    name="head",
                    rgb=True,
                    depth=False,
                    resolution=(128, 128),
                )
            ],
        ),
        render_mode=None,
    )
    
    cartesian_env = ReachTarget(
        action_mode=CartesianActionMode(floating_base=True),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(
                    name="head", 
                    rgb=True,
                    depth=False,
                    resolution=(128, 128),
                )
            ],
        ),
        render_mode=None,
    )
    
    # Load original joint demos
    print("Loading original joint demonstrations...")
    demo_store = DemoStore()
    joint_metadata = Metadata.from_env(joint_env)
    control_frequency = 50
    original_demos = demo_store.get_demos(joint_metadata, amount=demo_amount, frequency=control_frequency)
    
    print(f"Loaded {len(original_demos)} original demos")
    
    # Convert each demo
    cartesian_demos = []
    
    for i, original_demo in enumerate(original_demos):
        print(f"\nConverting demo {i+1}/{len(original_demos)}...")
        
        try:
            print(f"Starting conversion of demo {i+1} with {len(original_demo.timesteps)} steps...")
            cartesian_demo = convert_joint_demo_to_cartesian_target(
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


def analyze_difference():
    """Analyze the difference between achieved and target poses."""
    print("Analyzing difference between ACHIEVED and TARGET poses...")
    print("="*60)
    
    # Create environment
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    # Load a demo
    demo_store = DemoStore()
    metadata = Metadata.from_env(env)
    demos = demo_store.get_demos(metadata, amount=1, frequency=50)
    
    if not demos:
        print("No demos found!")
        return
    
    demo = demos[0]
    env.reset()
    
    differences = []
    
    print("\nAnalyzing first 20 timesteps...")
    for i in range(min(20, len(demo.timesteps))):
        joint_action = demo.timesteps[i].info['demo_action']
        
        # Get current state
        floating_base_dof = env.robot.floating_base.dof_amount if env.action_mode.floating_base else 0
        num_limb_actuators = len(env.robot.limb_actuators)
        current_joints = np.array(env.robot.qpos_actuated[floating_base_dof:floating_base_dof+num_limb_actuators])
        
        # Compute target poses
        target_left, target_right = compute_target_poses_from_joint_action(
            joint_action, env, current_joints
        )
        
        # Step to get achieved poses
        env.step(joint_action)
        
        # Get achieved poses
        left_site = env.robot._wrist_sites[HandSide.LEFT]
        right_site = env.robot._wrist_sites[HandSide.RIGHT]
        achieved_left = left_site.get_position()
        achieved_right = right_site.get_position()
        
        # Calculate differences
        left_diff = np.linalg.norm(achieved_left - target_left.position)
        right_diff = np.linalg.norm(achieved_right - target_right.position)
        avg_diff = (left_diff + right_diff) / 2
        
        differences.append(avg_diff)
        
        if i < 5 or i % 5 == 0:
            print(f"  Step {i:3d}: Target-Achieved difference = {avg_diff*1000:.1f}mm")
    
    print(f"\nAverage difference: {np.mean(differences)*1000:.1f}mm")
    print(f"Max difference: {np.max(differences)*1000:.1f}mm")
    print(f"Min difference: {np.min(differences)*1000:.1f}mm")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert joint demos to Cartesian demos using TARGET poses")
    parser.add_argument("--max-demos", type=int, default=60, help="Maximum number of demos to convert")
    parser.add_argument("--output-dir", type=str, default="cartesian_demos_target", 
                       help="Output directory for converted demos")
    parser.add_argument("--analyze", action="store_true", help="Analyze difference between achieved and target")
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze the difference between achieved and target poses
        analyze_difference()
    else:
        # Convert demos
        print("\n" + "="*50)
        converted_demos = convert_demos_batch(demo_amount=args.max_demos, output_dir=args.output_dir)
        
        print(f"\nConversion complete! Converted {len(converted_demos)} demos.")
        print(f"Converted demos saved in '{args.output_dir}/' directory")
        print(f"\nThese demos use TARGET poses (what joints are commanded to reach)")
        print(f"rather than ACHIEVED poses (what joints actually reach)")