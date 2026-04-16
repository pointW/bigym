"""Convert H1 joint demonstrations to RBY1 Cartesian demonstrations.

Since RBY1 uses whole-body IK that optimizes base movement together with joints,
the Cartesian action only needs end-effector poses and gripper actions.
The base/torso movements are automatically handled by the whole-body IK solver.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from typing import List, Type, Optional, Dict, Any, Tuple
import importlib
import multiprocessing as mp
import traceback
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
from demonstrations.demo_store import DemoStore, DemoNotFoundError
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
        'DishwasherOpen', 'DishwasherClose', 'DishwasherOpenTrays', 'DishwasherCloseTrays',
        'DishwasherLoadCups', 'DishwasherLoadCutlery', 'DishwasherLoadPlates',
        'DishwasherUnloadCups', 'DishwasherUnloadCutlery', 'DishwasherUnloadPlates',
        'DishwasherUnloadCupsLong', 'DishwasherUnloadCutleryLong', 'DishwasherUnloadPlatesLong',
    }
    
    # Default to 3 DOF (X, Y, RZ) for most tasks
    # This includes ReachTarget, MovePlate, Dishwasher tasks, etc.
    if env_name in four_dof_envs:
        return [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    else:
        return [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]


def _dof_label(floating_dofs: List[PelvisDof]) -> str:
    """Return a readable floating-DOF label."""
    return "4 DOF (X,Y,Z,RZ)" if len(floating_dofs) == 4 else "3 DOF (X,Y,RZ)"


def get_environment_class(env_name: str) -> Type:
    """Dynamically import and return the environment class.
    
    Args:
        env_name: Name of the environment (e.g., 'ReachTarget', 'MovePlate')
        
    Returns:
        Environment class
    """
    # Map of environment names to (module_path, class_name override).
    # class_name is omitted when it matches env_name.
    env_modules = {
        # Core environments
        'ReachTarget': ('bigym.envs.reach_target', None),
        'ReachTargetSingle': ('bigym.envs.reach_target', None),
        'ReachTargetDual': ('bigym.envs.reach_target', None),
        'MovePlate': ('bigym.envs.move_plates', None),
        'MovePlates': ('bigym.envs.move_plates', 'MovePlate'),  # Alias
        'MoveTwoPlates': ('bigym.envs.move_plates', None),
        
        # Dishwasher tasks
        'DishwasherOpen': ('bigym.envs.dishwasher', None),
        'DishwasherClose': ('bigym.envs.dishwasher', None),
        'DishwasherOpenTrays': ('bigym.envs.dishwasher', None),
        'DishwasherCloseTrays': ('bigym.envs.dishwasher', None),
        'DishwasherLoadCups': ('bigym.envs.dishwasher_cups', None),
        'DishwasherLoadCutlery': ('bigym.envs.dishwasher_cutlery', None),
        'DishwasherLoadPlates': ('bigym.envs.dishwasher_plates', None),
        'DishwasherUnloadCups': ('bigym.envs.dishwasher_cups', None),
        'DishwasherUnloadCupsLong': ('bigym.envs.dishwasher_cups', None),
        'DishwasherUnloadCutlery': ('bigym.envs.dishwasher_cutlery', None),
        'DishwasherUnloadCutleryLong': ('bigym.envs.dishwasher_cutlery', None),
        'DishwasherUnloadPlates': ('bigym.envs.dishwasher_plates', None),
        'DishwasherUnloadPlatesLong': ('bigym.envs.dishwasher_plates', None),
        
        # Manipulation tasks
        'FlipCup': ('bigym.envs.manipulation', None),
        'FlipCutlery': ('bigym.envs.manipulation', None),
        'FlipSandwich': ('bigym.envs.pick_and_place', None),
        'StackBlocks': ('bigym.envs.manipulation', None),
        
        # Pick and place tasks
        'ToastSandwich': ('bigym.envs.pick_and_place', None),
        'RemoveSandwich': ('bigym.envs.pick_and_place', None),
        'SaucepanToHob': ('bigym.envs.pick_and_place', None),
        'StoreBox': ('bigym.envs.pick_and_place', None),
        'PickBox': ('bigym.envs.pick_and_place', None),
        'StoreKitchenware': ('bigym.envs.pick_and_place', None),
        'TakeCups': ('bigym.envs.pick_and_place', None),
        'PutCups': ('bigym.envs.pick_and_place', None),

        # Groceries tasks
        'GroceriesStoreLower': ('bigym.envs.groceries', None),
        'GroceriesStoreUpper': ('bigym.envs.groceries', None),
        
        # Cupboard/Drawer tasks
        'CupboardsOpenAll': ('bigym.envs.cupboards', None),
        'CupboardsCloseAll': ('bigym.envs.cupboards', None),
        'WallCupboardOpen': ('bigym.envs.cupboards', None),
        'WallCupboardClose': ('bigym.envs.cupboards', None),
        'DrawersAllOpen': ('bigym.envs.cupboards', None),
        'DrawersAllClose': ('bigym.envs.cupboards', None),
        'DrawerTopOpen': ('bigym.envs.cupboards', None),
        'DrawerTopClose': ('bigym.envs.cupboards', None),
    }

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
    
    module_name, class_override = env_modules[env_name]
    class_name = class_override or env_name
    module = importlib.import_module(module_name)
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


def _build_rby1_hold_action_from_obs(obs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Build a cartesian action that holds the current EE poses."""
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

    left_rot6d = rotation_matrix_to_6d(Quaternion(left_quat).rotation_matrix)
    right_rot6d = rotation_matrix_to_6d(Quaternion(right_quat).rotation_matrix)
    gripper = np.zeros(2, dtype=np.float64)
    return np.concatenate(
        [left_pos, left_rot6d, right_pos, right_rot6d, gripper],
        axis=0,
    )


def _strip_raw_depth_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop raw depth images from an observation dict while keeping point clouds."""
    if not isinstance(obs, dict):
        return obs
    return {k: v for k, v in obs.items() if not (isinstance(k, str) and k.startswith("depth_"))}


def convert_h1_demo_to_rby1_cartesian(
    original_demo: Demo,
    env_class: Type,
    env_name: str,
    camera_configs: List[CameraConfig],
    control_frequency: int = 20,
    interpolation_frequency: int = 20,
    low_pass_freq_hz: float = 10.0,
    render_mode: Optional[str] = None,
    robot_type: str = "rby1",
    blend_steps: int = 0,
    blend_ori_steps: Optional[int] = None,
    with_pointcloud: bool = True,
    pcd_points: int = 1024,
    pcd_min_dist: Optional[float] = None,
    pcd_max_dist: Optional[float] = 3.0,
    pcd_min_world_z: Optional[float] = 0.01,
    success_settle_steps: int = 50,
) -> Tuple[Demo, bool]:
    """Convert a single H1 joint demo to RBY1 Cartesian demo.
    
    Args:
        original_demo: Original H1 demo with joint actions
        env_class: Environment class to use
        env_name: Name of the environment for DOF detection
        camera_configs: Camera configurations for the environment
        control_frequency: Control frequency for the environment
        interpolation_frequency: IK/interpolation update frequency for RBY1
        low_pass_freq_hz: Low-pass cutoff for RBY1 command smoothing
        render_mode: Render mode (None for headless)
        robot_type: Target robot type (should be "rby1")
        blend_steps: Number of initial steps to blend from perturbed pose
        blend_ori_steps: Steps to blend orientation (defaults to blend_steps)
        with_pointcloud: If True, generate point clouds from depth + rgb
        pcd_points: Number of points to sample per camera
        pcd_min_dist: Minimum camera distance (meters) to keep points
        pcd_max_dist: Maximum camera distance (meters) to keep points
        pcd_min_world_z: Minimum world-frame z (meters) to keep point-cloud points
        success_settle_steps: Number of additional hold steps before success
            judgment. Settle steps are not recorded in the output demo.
        
    Returns:
        Tuple of (converted demo, success flag from RBY1 rollout)
    """
    # Get the actions from the original demo
    joint_actions = np.array([step.executed_action for step in original_demo.timesteps])

    cartesian_actions = []

    if with_pointcloud:
        camera_configs = [CameraConfig(**vars(cam)) for cam in camera_configs]
        for cam in camera_configs:
            cam.rgb = True
            cam.depth = False
            cam.pcd = True
            cam.pcd_points = int(pcd_points)
            cam.pcd_min_dist = pcd_min_dist
            cam.pcd_max_dist = pcd_max_dist
            cam.pcd_min_world_z = pcd_min_world_z
    
    # Replay source demo with the exact action-mode / robot metadata.
    source_env_data = original_demo.metadata.environment_data
    source_floating_dofs = [PelvisDof(dof) for dof in source_env_data.floating_dofs]
    source_absolute = (
        True
        if source_env_data.action_mode_absolute is None
        else bool(source_env_data.action_mode_absolute)
    )
    source_robot_cls = original_demo.metadata.robot_cls
    dof_str = _dof_label(source_floating_dofs)
    print(f"Converting H1 demo with {len(joint_actions)} steps to RBY1 Cartesian format...")
    print(
        f"Using source metadata for replay: {dof_str}, "
        f"absolute={source_absolute}, robot={source_robot_cls.__name__}"
    )
    
    # Create H1 environment for replaying the demo with appropriate floating DOFs
    h1_env = env_class(
        action_mode=JointPositionActionMode(
            floating_base=source_env_data.floating_base,
            absolute=source_absolute,
            floating_dofs=source_floating_dofs,
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=[]),
        render_mode=render_mode,
        robot_cls=source_robot_cls,
        init_perturb=False,
    )
    h1_env.reset(seed=original_demo.seed)
    
    # Get floating base DOF count for H1
    floating_base_dof = (
        h1_env.robot.floating_base.dof_amount 
        if h1_env.action_mode.floating_base else 0
    )
    
    for step_idx, joint_action in enumerate(joint_actions):
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
            interpolation_frequency=interpolation_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=camera_configs),
        render_mode=render_mode,
        robot_cls=RBY1,
        init_perturb=False,
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
        obs = _strip_raw_depth_obs(obs)
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
    
    settle_steps = int(max(0, success_settle_steps))
    if settle_steps > 0:
        for _ in range(settle_steps):
            hold_action = _build_rby1_hold_action_from_obs(rby1_obs)
            if hold_action is None:
                hold_action = np.zeros_like(
                    rby1_env.action_space.low, dtype=np.float64
                )
            clipped_hold = np.clip(
                hold_action,
                rby1_env.action_space.low,
                rby1_env.action_space.high,
            )
            rby1_obs, _, _, _, info = rby1_env.step(clipped_hold)
            rby1_obs = _strip_raw_depth_obs(rby1_obs)
            last_info = info or {}

    rby1_metadata = Metadata.from_env(rby1_env)
    rby1_metadata.init_perturb_enabled = False
    rby1_metadata.seed = original_demo.seed
    success = bool(last_info.get("task_success", False)) if last_info else bool(rby1_env.success)
    rby1_env.close()
    
    return Demo(metadata=rby1_metadata, timesteps=timesteps), success


def _convert_demo_worker(
    payload: Tuple[
        int,
        Demo,
        str,
        List[CameraConfig],
        int,
        int,
        float,
        Optional[str],
        str,
        int,
        Optional[int],
        Optional[int],
        bool,
        bool,
        int,
        Optional[float],
        Optional[float],
        Optional[float],
        int,
        int,
    ]
) -> Tuple[int, bool, Optional[Demo], Optional[str]]:
    """Worker for multiprocessing demo conversion."""
    (
        index,
        original_demo,
        env_name,
        camera_configs,
        control_frequency,
        interpolation_frequency,
        low_pass_freq_hz,
        render_mode,
        robot_type,
        blend_steps,
        blend_ori_steps,
        with_pointcloud,
        pcd_points,
        pcd_min_dist,
        pcd_max_dist,
        pcd_min_world_z,
        success_settle_steps,
    ) = payload
    try:
        env_class = get_environment_class(env_name)
        rby1_demo, success = convert_h1_demo_to_rby1_cartesian(
            original_demo,
            env_class,
            env_name,
            camera_configs,
            control_frequency,
            interpolation_frequency,
            low_pass_freq_hz,
            render_mode,
            robot_type,
            blend_steps=blend_steps,
            blend_ori_steps=blend_ori_steps,
            with_pointcloud=with_pointcloud,
            pcd_points=pcd_points,
            pcd_min_dist=pcd_min_dist,
            pcd_max_dist=pcd_max_dist,
            pcd_min_world_z=pcd_min_world_z,
            success_settle_steps=success_settle_steps,
        )
        return index, success, rby1_demo, None
    except Exception:
        return index, False, None, traceback.format_exc()


def convert_h1_demos_batch(
    env_name: str,
    demo_amount: int = -1,
    output_dir: str = None,
    camera_configs: Optional[List[CameraConfig]] = None,
    camera_resolution: int = 224,
    control_frequency: int = 20,
    interpolation_frequency: int = 20,
    low_pass_freq_hz: float = 10.0,
    render_mode: Optional[str] = None,
    robot_type: str = "rby1",
    blend_steps: int = 0,
    blend_ori_steps: Optional[int] = None,
    processes: int = 1,
    source_seed: Optional[int] = None,
    perturb_variants: int = 1,
    save_videos: bool = False,
    video_dir: Optional[str] = None,
    with_pointcloud: bool = True,
    pcd_points: int = 1024,
    pcd_min_dist: Optional[float] = None,
    pcd_max_dist: Optional[float] = 3.0,
    pcd_min_world_z: Optional[float] = 0.01,
    success_settle_steps: int = 50,
) -> List[Demo]:
    """Convert a batch of H1 demonstrations to RBY1 Cartesian format.
    
    Args:
        env_name: Name of the environment (e.g., 'ReachTarget', 'MovePlate')
        demo_amount: Number of demos to convert
        output_dir: Directory to save converted demos (auto-generated if None)
        camera_configs: Camera configurations (uses defaults if None)
        camera_resolution: Square RGB resolution used when camera_configs is None
        control_frequency: Control frequency
        interpolation_frequency: IK/interpolation update frequency for RBY1
        low_pass_freq_hz: Low-pass cutoff for RBY1 command smoothing
        render_mode: Render mode for conversion
        robot_type: Target robot type (default "rby1")
        blend_steps: Number of initial steps to blend from perturbed pose
        blend_ori_steps: Steps to blend orientation (defaults to blend_steps)
        source_seed: Seed of the source demo to convert (optional)
        perturb_variants: Number of perturbation variants to generate per source demo
        save_videos: Whether to save RGB videos for converted demos
        video_dir: Directory to store videos (defaults to output_dir/videos)
        with_pointcloud: If True, generate point clouds from depth + rgb
        pcd_points: Number of points to sample per camera
        pcd_min_dist: Minimum camera distance (meters) to keep points
        pcd_max_dist: Maximum camera distance (meters) to keep points
        pcd_min_world_z: Minimum world-frame z (meters) to keep point-cloud points
        success_settle_steps: Number of additional hold steps before success
            judgment. Settle steps are not recorded.
        
    Returns:
        List of converted RBY1 Cartesian demos
    """
    # Get environment class
    env_class = get_environment_class(env_name)
    
    # Use default camera config if not provided
    if camera_configs is None:
        camera_configs = [
            CameraConfig("head", resolution=(camera_resolution, camera_resolution)),
            CameraConfig("left_wrist", resolution=(camera_resolution, camera_resolution)),
            CameraConfig("right_wrist", resolution=(camera_resolution, camera_resolution)),
        ]
    if with_pointcloud:
        camera_configs = [CameraConfig(**vars(cam)) for cam in camera_configs]
        for cam in camera_configs:
            cam.rgb = True
            cam.depth = False
            cam.pcd = True
            cam.pcd_points = int(pcd_points)
            cam.pcd_min_dist = pcd_min_dist
            cam.pcd_max_dist = pcd_max_dist
            cam.pcd_min_world_z = pcd_min_world_z
    
    # Auto-generate output directory name if not provided
    if output_dir is None:
        output_dir = f"rby1_cartesian_demos_{env_name.lower()}"
    if save_videos and video_dir is None:
        video_dir = str(Path(output_dir) / "videos")
    
    print(f"Converting {demo_amount} H1 {env_name} demonstrations to RBY1 Cartesian format...")
    print(f"Output directory: {output_dir}")
    
    # Detect preferred floating DOFs and fallback candidates.
    preferred_dofs = detect_floating_dofs_from_demos(env_name)
    fallback_dofs = (
        [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]
        if len(preferred_dofs) == 4
        else [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    )
    dof_candidates = [preferred_dofs]
    if fallback_dofs != preferred_dofs:
        dof_candidates.append(fallback_dofs)
    print(f"Preferred floating base configuration: {_dof_label(preferred_dofs)}")

    # Try default robot first, then H1 for legacy paths.
    robot_candidates: list[Type] = [env_class.DEFAULT_ROBOT]
    if env_class.DEFAULT_ROBOT != H1:
        robot_candidates.append(H1)
    absolute_candidates = [True, False]

    # Load original H1 demos, retrying with alternate floating DOFs when needed.
    print("Loading original H1 joint demonstrations...")
    cache_root_env = os.getenv("BIGYM_CACHE_ROOT")
    if cache_root_env:
        cache_root = Path(cache_root_env).expanduser()
        print(f"Using DemoStore cache root override: {cache_root}")
        demo_store = DemoStore(cache_root=cache_root)
    else:
        demo_store = DemoStore()
    demo_amount = -1 if source_seed is not None else demo_amount
    original_demos = None
    last_error: Optional[Exception] = None
    attempt_idx = 0
    for floating_dofs in dof_candidates:
        for absolute in absolute_candidates:
            for robot_cls in robot_candidates:
                attempt_idx += 1
                if attempt_idx > 1:
                    print(
                        "Retrying demo lookup with "
                        f"{_dof_label(floating_dofs)}, absolute={absolute}, "
                        f"robot={robot_cls.__name__}..."
                    )
                h1_env = env_class(
                    action_mode=JointPositionActionMode(
                        floating_base=True,
                        absolute=absolute,
                        floating_dofs=floating_dofs,
                    ),
                    control_frequency=control_frequency,
                    observation_config=ObservationConfig(cameras=[]),
                    render_mode=render_mode,
                    robot_cls=robot_cls,
                )
                try:
                    h1_metadata = Metadata.from_env(h1_env)
                    original_demos = demo_store.get_demos(
                        h1_metadata,
                        amount=demo_amount,
                        frequency=control_frequency,
                    )
                    print(
                        "Using source lookup config: "
                        f"{_dof_label(floating_dofs)}, absolute={absolute}, "
                        f"robot={robot_cls.__name__}"
                    )
                    break
                except DemoNotFoundError as exc:
                    last_error = exc
                    original_demos = None
                finally:
                    h1_env.close()
            if original_demos is not None:
                break
        if original_demos is not None:
            break

    if original_demos is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"Failed to load source demos for env={env_name}. No DOF configuration matched."
        )
    
    if source_seed is not None:
        original_demos = [d for d in original_demos if d.seed == source_seed]
        if not original_demos:
            raise ValueError(f"No demo found for seed {source_seed}")

    print(f"Loaded {len(original_demos)} original H1 demos")
    
    # Convert each demo
    rby1_cartesian_demos = []
    total_variants = max(1, int(perturb_variants)) * len(original_demos)
    results: list[Optional[Tuple[bool, Optional[Demo], Optional[str]]]] = [
        None for _ in range(total_variants)
    ]

    payloads = []
    for original_demo in original_demos:
        variants = max(1, int(perturb_variants))
        for _variant_idx in range(variants):
            payloads.append(
                (
                    len(payloads),
                    original_demo,
                    env_name,
                    camera_configs,
                    control_frequency,
                    interpolation_frequency,
                    low_pass_freq_hz,
                    render_mode,
                    robot_type,
                    blend_steps,
                    blend_ori_steps,
                    with_pointcloud,
                    pcd_points,
                    pcd_min_dist,
                    pcd_max_dist,
                    pcd_min_world_z,
                    success_settle_steps,
                )
            )

    def _write_demo_videos(demo: Demo, demo_idx: int):
        if not save_videos:
            return
        output_path = Path(video_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for camera_config in camera_configs:
            if not camera_config.rgb:
                continue
            key = f"rgb_{camera_config.name}"
            writer = imageio.get_writer(
                output_path / f"{env_name.lower()}_{demo_idx:03d}_{camera_config.name}.mp4",
                fps=control_frequency,
            )
            try:
                for step in demo.timesteps:
                    frame = step.observation.get(key)
                    if frame is None:
                        continue
                    if frame.shape[0] in (1, 3):
                        frame = np.moveaxis(frame, 0, -1)
                    writer.append_data(frame.astype(np.uint8))
            finally:
                writer.close()

    if processes > 1:
        print(f"Using multiprocessing with {processes} processes")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=processes) as pool:
            for index, success, demo, error in tqdm(
                pool.imap_unordered(_convert_demo_worker, payloads),
                total=len(payloads),
                desc="Converting demos",
            ):
                results[index] = (success, demo, error)
    else:
        for payload in tqdm(payloads, total=len(payloads), desc="Converting demos"):
            (
                index,
                original_demo,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                with_pointcloud,
                pcd_points,
                pcd_min_dist,
                pcd_max_dist,
                pcd_min_world_z,
                success_settle_steps,
            ) = payload
            try:
                rby1_demo, success = convert_h1_demo_to_rby1_cartesian(
                    original_demo,
                    env_class,
                    env_name,
                    camera_configs,
                    control_frequency,
                    interpolation_frequency,
                    low_pass_freq_hz,
                    render_mode,
                    robot_type,
                    blend_steps=blend_steps,
                    blend_ori_steps=blend_ori_steps,
                    with_pointcloud=with_pointcloud,
                    pcd_points=pcd_points,
                    pcd_min_dist=pcd_min_dist,
                    pcd_max_dist=pcd_max_dist,
                    pcd_min_world_z=pcd_min_world_z,
                    success_settle_steps=success_settle_steps,
                )
                results[index] = (success, rby1_demo, None)
            except Exception:
                results[index] = (False, None, traceback.format_exc())

    success_idx = 0
    failure_idx = 0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        success, rby1_demo, error = result or (False, None, "No result returned.")
        print(f"\nConverting demo {i+1}/{total_variants} to RBY1 Cartesian...")
        if error:
            print(f"❌ Error converting H1 demo {i+1}:\n{error}")
            continue
        status_msg = "SUCCESS" if success else "FAILURE"
        print(f"RBY1 rollout result: {status_msg}")
        if rby1_demo is None:
            print("Skipping save because the converted demo is missing.")
            continue

        if success:
            rby1_cartesian_demos.append(rby1_demo)
            demo_path = output_path / f"rby1_cartesian_demo_{success_idx:03d}.safetensors"
            print(f"Saving successful RBY1 demo to {demo_path}...")
            rby1_demo.save(demo_path)
            _write_demo_videos(rby1_demo, success_idx)
            success_idx += 1
            print("✓ Successfully saved RBY1 Cartesian demo")
        else:
            failure_path = output_path / "failure"
            failure_path.mkdir(parents=True, exist_ok=True)
            demo_path = failure_path / f"failed_rby1_cartesian_demo_{failure_idx:03d}.safetensors"
            print(f"Saving failed RBY1 demo to {demo_path}...")
            rby1_demo.save(demo_path)
            failure_idx += 1
            print("✓ Saved failed RBY1 Cartesian demo")
    
    print(f"\nSuccessfully converted {len(rby1_cartesian_demos)}/{total_variants} H1 demonstrations to RBY1")
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
        default=-1, 
        help="Maximum number of demos to convert (-1 for all)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--camera-resolution",
        type=int,
        default=224,
        help="Square RGB camera resolution for head/left_wrist/right_wrist (default: 224)"
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=20,
        help="Control frequency (default: 20)"
    )
    parser.add_argument(
        "--interpolation-frequency",
        type=int,
        default=20,
        help="RBY1 IK/interpolation frequency (default: 20)"
    )
    parser.add_argument(
        "--low-pass-freq-hz",
        type=float,
        default=10.0,
        help="RBY1 command low-pass cutoff in Hz (default: 10)"
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
    parser.add_argument(
        "--source-seed",
        type=int,
        default=None,
        help="Seed of the source demo to convert"
    )
    parser.add_argument(
        "--perturb-variants",
        type=int,
        default=1,
        help="Number of perturbation variants per source demo"
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save RGB videos for each converted demo"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Output directory for videos (defaults to output_dir/videos)"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes to use"
    )
    parser.add_argument(
        "--no-pointcloud",
        action="store_true",
        help="Disable point cloud generation (default is enabled)"
    )
    parser.add_argument(
        "--pcd-points",
        type=int,
        default=1024,
        help="Number of points to sample per camera for point clouds"
    )
    parser.add_argument(
        "--pcd-min-dist",
        type=float,
        default=None,
        help="Minimum camera distance (meters) to keep points"
    )
    parser.add_argument(
        "--pcd-max-dist",
        type=float,
        default=3.0,
        help="Maximum camera distance (meters) to keep points"
    )
    parser.add_argument(
        "--pcd-min-world-z",
        type=float,
        default=0.01,
        help="Minimum world-frame z (meters) to keep point-cloud points"
    )
    parser.add_argument(
        "--success-settle-steps",
        type=int,
        default=50,
        help="Number of non-recorded hold steps before success judgment (default: 50)"
    )
    args = parser.parse_args()
    
    # Convert H1 demos to RBY1 Cartesian
    converted_demos = convert_h1_demos_batch(
        env_name=args.env,
        demo_amount=args.max_demos,
        output_dir=args.output_dir,
        camera_resolution=args.camera_resolution,
        control_frequency=args.control_freq,
        interpolation_frequency=args.interpolation_frequency,
        low_pass_freq_hz=args.low_pass_freq_hz,
        render_mode="human" if args.render else None,
        robot_type=args.robot,
        blend_steps=args.blend_steps,
        blend_ori_steps=args.blend_ori_steps,
        processes=args.processes,
        source_seed=args.source_seed,
        perturb_variants=args.perturb_variants,
        save_videos=args.save_videos,
        video_dir=args.video_dir,
        with_pointcloud=not args.no_pointcloud,
        pcd_points=args.pcd_points,
        pcd_min_dist=args.pcd_min_dist,
        pcd_max_dist=args.pcd_max_dist,
        pcd_min_world_z=args.pcd_min_world_z,
        success_settle_steps=args.success_settle_steps,
    )
    
    print(f"\nConversion complete! Converted {len(converted_demos)} H1 demos to RBY1 Cartesian format.")
    if args.output_dir:
        print(f"RBY1 Cartesian demos saved in '{args.output_dir}/' directory")
    else:
        print(f"RBY1 Cartesian demos saved in 'rby1_cartesian_demos_{args.env.lower()}/' directory")


if __name__ == "__main__":
    main()
