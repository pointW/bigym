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
from typing import List, Type, Optional, Dict, Any, Tuple
import importlib
from tqdm import tqdm
from pyquaternion import Quaternion

from bigym.action_modes import JointPositionActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.rby1_cartesian_action_mode_whole_body import (
    RBY1CartesianActionModeWholeBody,
    rotation_matrix_to_6d,
    rotation_6d_to_matrix,
)
from vr.ik.h1_upper_body_ik import Pose
from bigym.const import HandSide
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo, DemoStep
from bigym.robots.configs.h1 import H1
from bigym.robots.configs.rby1 import RBY1
from bigym.action_modes import PelvisDof


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
        'DishwasherOpen', 'DishwasherClose', 'DishwasherOpenTrays', 'DishwasherCloseTrays', 'DishwasherLoadCups', 'DishwasherLoadCutlery', 'DishwasherLoadPlates', 'DishwasherUnloadCups', 'DishwasherUnloadCutlery', 'DishwasherUnloadPlates', 'DishwasherUnloadPlatesLong',
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
        'DishwasherLoadCups': 'bigym.envs.dishwasher_cups',
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
        'TakeCups': 'bigym.envs.pick_and_place',
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


def _split_rby1_cartesian_action(
    action: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a 20D RBY1 Cartesian action into components."""
    idx = 0
    left_pos = action[idx:idx + 3]
    idx += 3
    left_rot_6d = action[idx:idx + 6]
    idx += 6
    right_pos = action[idx:idx + 3]
    idx += 3
    right_rot_6d = action[idx:idx + 6]
    idx += 6
    gripper_action = action[idx:]
    return left_pos, left_rot_6d, right_pos, right_rot_6d, gripper_action


def _get_initial_ee_poses_from_obs(
    obs: Dict[str, np.ndarray]
) -> Optional[Tuple[Pose, Pose]]:
    """Extract initial end-effector poses from observation."""
    left_pos = obs.get("left_ee_pos")
    right_pos = obs.get("right_ee_pos")
    left_quat = obs.get("left_ee_quat")
    right_quat = obs.get("right_ee_quat")
    if (
        left_pos is None
        or right_pos is None
        or left_quat is None
        or right_quat is None
    ):
        return None
    left_pose = Pose(left_pos, Quaternion(left_quat))
    right_pose = Pose(right_pos, Quaternion(right_quat))
    return left_pose, right_pose


def _blend_cartesian_actions(
    cartesian_actions: List[np.ndarray],
    initial_left_pose: Pose,
    initial_right_pose: Pose,
    blend_steps: int,
    target_left_pose: Optional[Pose] = None,
    target_right_pose: Optional[Pose] = None,
) -> List[np.ndarray]:
    """Blend initial steps from current pose to a target pose."""
    if blend_steps <= 0 or not cartesian_actions:
        return cartesian_actions

    steps = min(blend_steps, len(cartesian_actions))
    blended = list(cartesian_actions)

    if target_left_pose is None or target_right_pose is None:
        (
            left_pos0,
            left_rot_6d_0,
            right_pos0,
            right_rot_6d_0,
            _,
        ) = _split_rby1_cartesian_action(blended[0])
        target_left_pose = Pose(
            left_pos0, Quaternion(matrix=rotation_6d_to_matrix(left_rot_6d_0))
        )
        target_right_pose = Pose(
            right_pos0, Quaternion(matrix=rotation_6d_to_matrix(right_rot_6d_0))
        )

    def _compute_delta(init_pose: Pose, target_pose: Pose):
        delta_quat = init_pose.orientation * target_pose.orientation.inverse
        delta_rot = delta_quat.rotation_matrix
        delta_t = init_pose.position - delta_rot @ target_pose.position
        return delta_quat, delta_rot, delta_t

    left_delta_quat, left_delta_rot, left_delta_t = _compute_delta(
        initial_left_pose, target_left_pose
    )
    right_delta_quat, right_delta_rot, right_delta_t = _compute_delta(
        initial_right_pose, target_right_pose
    )

    for i in range(steps):
        alpha = (i + 1) / steps
        (
            left_pos,
            left_rot_6d,
            right_pos,
            right_rot_6d,
            gripper_action,
        ) = _split_rby1_cartesian_action(blended[i])

        left_quat = Quaternion(matrix=rotation_6d_to_matrix(left_rot_6d))
        right_quat = Quaternion(matrix=rotation_6d_to_matrix(right_rot_6d))

        left_rel_pos = left_delta_rot @ left_pos + left_delta_t
        right_rel_pos = right_delta_rot @ right_pos + right_delta_t
        left_rel_quat = left_delta_quat * left_quat
        right_rel_quat = right_delta_quat * right_quat

        left_blend_pos = (1 - alpha) * left_rel_pos + alpha * left_pos
        right_blend_pos = (1 - alpha) * right_rel_pos + alpha * right_pos
        left_blend_quat = Quaternion.slerp(left_rel_quat, left_quat, amount=alpha)
        right_blend_quat = Quaternion.slerp(right_rel_quat, right_quat, amount=alpha)

        blended[i] = np.concatenate(
            [
                left_blend_pos,
                rotation_matrix_to_6d(left_blend_quat.rotation_matrix),
                right_blend_pos,
                rotation_matrix_to_6d(right_blend_quat.rotation_matrix),
                gripper_action,
            ]
        )

    return blended


def _blend_action_from_obs(
    obs: Dict[str, np.ndarray],
    target_action: np.ndarray,
    alpha_pos: float,
    alpha_ori: Optional[float] = None,
) -> np.ndarray:
    """Blend current observed EE pose toward target action."""
    if alpha_ori is None:
        alpha_ori = alpha_pos
    left_pos, left_rot_6d, right_pos, right_rot_6d, gripper_action = _split_rby1_cartesian_action(
        target_action
    )
    current_left_pos = obs.get("left_ee_pos")
    current_right_pos = obs.get("right_ee_pos")
    current_left_quat = obs.get("left_ee_quat")
    current_right_quat = obs.get("right_ee_quat")
    if (
        current_left_pos is None
        or current_right_pos is None
        or current_left_quat is None
        or current_right_quat is None
    ):
        return target_action

    left_quat = Quaternion(matrix=rotation_6d_to_matrix(left_rot_6d))
    right_quat = Quaternion(matrix=rotation_6d_to_matrix(right_rot_6d))
    current_left_quat = Quaternion(current_left_quat)
    current_right_quat = Quaternion(current_right_quat)

    blend_left_pos = (1 - alpha_pos) * current_left_pos + alpha_pos * left_pos
    blend_right_pos = (1 - alpha_pos) * current_right_pos + alpha_pos * right_pos
    blend_left_quat = Quaternion.slerp(
        current_left_quat, left_quat, amount=alpha_ori
    )
    blend_right_quat = Quaternion.slerp(
        current_right_quat, right_quat, amount=alpha_ori
    )

    return np.concatenate(
        [
            blend_left_pos,
            rotation_matrix_to_6d(blend_left_quat.rotation_matrix),
            blend_right_pos,
            rotation_matrix_to_6d(blend_right_quat.rotation_matrix),
            gripper_action,
        ]
    )


def convert_h1_demo_to_rby1_cartesian(
    original_demo: Demo,
    env_class: Type,
    env_name: str,
    camera_configs: List[CameraConfig],
    control_frequency: int = 50,
    render_mode: Optional[str] = None,
    robot_type: str = "rby1",
    blend_steps: int = 0,
    blend_ori_steps: Optional[int] = None,
) -> Tuple[Demo, bool]:
    """Convert a single H1 joint demo to RBY1 Cartesian demo.
    
    Args:
        original_demo: Original H1 demo with joint actions
        env_class: Environment class to use
        env_name: Name of the environment for DOF detection
        camera_configs: Camera configurations for the environment
        control_frequency: Control frequency for the environment
        render_mode: Render mode (None for headless)
        robot_type: Target robot type (should be "rby1")
        blend_steps: Number of initial steps to blend from perturbed pose
        blend_ori_steps: Steps to blend orientation (defaults to blend_steps)
        
    Returns:
        Tuple of (converted demo, success flag from RBY1 rollout)
    """
    # Get the actions from the original demo
    joint_actions = np.array([step.executed_action for step in original_demo.timesteps])
    
    cartesian_actions = []
    
    # Detect the correct floating DOFs for this environment
    floating_dofs = detect_floating_dofs_from_demos(env_name)
    dof_str = "4 DOF (X,Y,Z,RZ)" if len(floating_dofs) == 4 else "3 DOF (X,Y,RZ)"
    print(f"Converting H1 demo with {len(joint_actions)} steps to RBY1 Cartesian format...")
    print(f"Using {dof_str} floating base for {env_name}")
    
    # Create H1 environment for replaying the demo with appropriate floating DOFs
    h1_env = env_class(
        action_mode=JointPositionActionMode(
            floating_base=True, 
            absolute=True,
            floating_dofs=floating_dofs
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=[]),
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
    
    # Create RBY1 environment to actually execute the Cartesian actions
    rby1_env = env_class(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=render_mode,
        robot_cls=RBY1,
    )
    
    rby1_obs, _ = rby1_env.reset(seed=original_demo.seed)

    
    timesteps: list[DemoStep] = []
    last_info: Dict[str, Any] = {}
    for step_idx, cartesian_action in enumerate(cartesian_actions):
        if blend_steps > 0 and step_idx < blend_steps:
            alpha_pos = (step_idx + 1) / blend_steps
            ori_steps = blend_ori_steps or blend_steps
            alpha_ori = min((step_idx + 1) / ori_steps, 1.0)
            cartesian_action = _blend_action_from_obs(
                rby1_obs,
                cartesian_action,
                alpha_pos,
                alpha_ori=alpha_ori,
            )
        clipped_action = np.clip(
            cartesian_action,
            rby1_env.action_space.low,
            rby1_env.action_space.high,
        )
        obs, reward, terminated, truncated, info = rby1_env.step(clipped_action)
        rby1_obs = obs
        info = info or {}
        last_info = info
        timesteps.append(
            DemoStep(
                observation=obs,
                reward=reward,
                termination=terminated,
                truncation=truncated,
                info=info.copy(),
                action=clipped_action,
            )
        )
        if terminated or truncated:
            break
    
    rby1_metadata = Metadata.from_env(rby1_env)
    rby1_metadata.seed = original_demo.seed
    success = bool(last_info.get("task_success", False)) if last_info else bool(rby1_env.success)
    rby1_env.close()
    
    return Demo(metadata=rby1_metadata, timesteps=timesteps), success


def convert_h1_demos_batch(
    env_name: str,
    demo_amount: int = 3,
    output_dir: str = None,
    camera_configs: Optional[List[CameraConfig]] = None,
    control_frequency: int = 50,
    render_mode: Optional[str] = None,
    robot_type: str = "rby1",
    blend_steps: int = 0,
    blend_ori_steps: Optional[int] = None,
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
        blend_steps: Number of initial steps to blend from perturbed pose
        blend_ori_steps: Steps to blend orientation (defaults to blend_steps)
        
    Returns:
        List of converted RBY1 Cartesian demos
    """
    # Get environment class
    env_class = get_environment_class(env_name)
    
    # Use default camera config if not provided
    if camera_configs is None:
        camera_configs = [
            CameraConfig("head", resolution=(84, 84)),
            CameraConfig("left_wrist", resolution=(84, 84)),
            CameraConfig("right_wrist", resolution=(84, 84)),
        ]
    
    # Auto-generate output directory name if not provided
    if output_dir is None:
        output_dir = f"rby1_cartesian_demos_{env_name.lower()}"
    
    print(f"Converting {demo_amount} H1 {env_name} demonstrations to RBY1 Cartesian format...")
    print(f"Output directory: {output_dir}")
    
    # Detect the correct floating DOFs for this environment
    floating_dofs = detect_floating_dofs_from_demos(env_name)
    dof_str = "4 DOF (X,Y,Z,RZ)" if len(floating_dofs) == 4 else "3 DOF (X,Y,RZ)"
    print(f"Detected floating base configuration: {dof_str}")
    
    # Create H1 environment to load demos with appropriate floating DOFs
    h1_env = env_class(
        action_mode=JointPositionActionMode(
            floating_base=True, 
            absolute=True,
            floating_dofs=floating_dofs
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=[]),
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
    
    success_idx = 0
    for i, original_demo in enumerate(original_demos):
        print(f"\nConverting H1 demo {i+1}/{len(original_demos)} to RBY1 Cartesian...")
        
        try:
            rby1_demo, success = convert_h1_demo_to_rby1_cartesian(
                original_demo,
                env_class,
                env_name,
                camera_configs,
                control_frequency,
                render_mode,
                robot_type,
                blend_steps=blend_steps,
                blend_ori_steps=blend_ori_steps,
            )
            status_msg = "SUCCESS" if success else "FAILURE"
            print(f"RBY1 rollout result: {status_msg}")
            if not success:
                print("Skipping save because the converted demo did not succeed.")
                continue
            rby1_cartesian_demos.append(rby1_demo)
            
            # Save converted demo with contiguous success numbering
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            demo_path = output_path / f"rby1_cartesian_demo_{success_idx:03d}.safetensors"
            print(f"Saving successful RBY1 demo to {demo_path}...")
            rby1_demo.save(demo_path)
            success_idx += 1
            
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
        default="FlipCup",
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
    parser.add_argument(
        "--blend-steps",
        type=int,
        default=0,
        help="Number of initial steps to blend from perturbed pose"
    )
    parser.add_argument(
        "--blend-ori-steps",
        type=int,
        default=None,
        help="Steps to blend orientation (defaults to blend-steps)"
    )
    
    args = parser.parse_args()
    
    # Convert H1 demos to RBY1 Cartesian
    converted_demos = convert_h1_demos_batch(
        env_name=args.env,
        demo_amount=args.max_demos,
        output_dir=args.output_dir,
        control_frequency=args.control_freq,
        render_mode="human" if args.render else None,
        robot_type=args.robot,
        blend_steps=args.blend_steps,
        blend_ori_steps=args.blend_ori_steps,
    )
    
    print(f"\nConversion complete! Converted {len(converted_demos)} H1 demos to RBY1 Cartesian format.")
    if args.output_dir:
        print(f"RBY1 Cartesian demos saved in '{args.output_dir}/' directory")
    else:
        print(f"RBY1 Cartesian demos saved in 'rby1_cartesian_demos_{args.env.lower()}/' directory")


if __name__ == "__main__":
    main()
