"""RBY1 Cartesian space action mode with whole-body IK.

This version uses whole-body IK to automatically optimize base movement
along with joint positions to reach end-effector targets.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional
import numpy as np
from gymnasium import spaces
from pyquaternion import Quaternion
import mujoco
from yaml import safe_load

from bigym.action_modes import ActionMode, TargetStateNotReachedWarning
from bigym.const import HandSide, TOLERANCE_ANGULAR
from bigym.ik.rby1_whole_body_ik import RBY1WholeBodyIK
from bigym.utils.physics_utils import (
    is_target_reached,
)
from vr.ik.h1_upper_body_ik import Pose
import warnings

DEFAULT_RBY1_WBC_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "rby1_wbc.yaml"


def _load_whole_body_action_mode_config(config_path: Optional[str | Path]) -> dict:
    """Load whole-body Cartesian action mode defaults from rby1_wbc.yaml."""
    path = Path(config_path) if config_path is not None else DEFAULT_RBY1_WBC_CONFIG_PATH
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = safe_load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load RBY1 WBC config from '{path}': {exc}") from exc

    if not isinstance(cfg, dict):
        raise ValueError(f"RBY1 WBC config at '{path}' must be a mapping.")

    action_mode_cfg = cfg.get("whole_body_action_mode", {})
    if action_mode_cfg is None:
        return {}
    if not isinstance(action_mode_cfg, dict):
        raise ValueError(
            "RBY1 WBC config key 'whole_body_action_mode' must be a mapping."
        )
    return action_mode_cfg


def _parse_range(value, name: str) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} must have 2 values [min, max], got {arr.size}.")
    low = float(arr[0])
    high = float(arr[1])
    if low > high:
        raise ValueError(f"{name} must satisfy min <= max (got {value}).")
    return low, high


def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D rotation representation.
    
    The 6D representation consists of the first two rows of the rotation matrix.
    """
    # Return first two rows as a flat array
    return rotation_matrix[:2, :].flatten()


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.
    
    Reconstructs the rotation matrix from its first two rows using
    Gram-Schmidt orthogonalization.
    """
    # Reshape to get the two rows
    row1 = rotation_6d[:3]
    row2 = rotation_6d[3:6]
    
    # Normalize first row
    norm1 = np.linalg.norm(row1)
    if norm1 < 1e-6:
        # Handle degenerate case
        x = np.array([1, 0, 0])
    else:
        x = row1 / norm1
    
    # Make second row orthogonal to first and normalize
    y = row2 - np.dot(row2, x) * x
    norm2 = np.linalg.norm(y)
    if norm2 < 1e-6:
        # Handle degenerate case - create orthogonal vector
        # Find a vector not parallel to x
        if abs(x[0]) < 0.9:
            y = np.array([1, 0, 0]) - np.dot([1, 0, 0], x) * x
        else:
            y = np.array([0, 1, 0]) - np.dot([0, 1, 0], x) * x
        y = y / np.linalg.norm(y)
    else:
        y = y / norm2
    
    # Third row is cross product
    z = np.cross(x, y)
    
    # Stack rows to form matrix
    matrix = np.vstack([x, y, z])
    
    # Ensure perfect orthogonality through SVD
    U, S, Vt = np.linalg.svd(matrix)
    matrix = U @ Vt
    
    # Ensure determinant is +1 (proper rotation, not reflection)
    if np.linalg.det(matrix) < 0:
        # Flip the last column of U
        U[:, -1] = -U[:, -1]
        matrix = U @ Vt
    
    return matrix


class RBY1CartesianActionModeWholeBody(ActionMode):
    """Control RBY1 through end-effector poses using whole-body IK.
    
    The action space consists of:
    - Left end-effector position (3D): absolute position in world coordinates
    - Left end-effector orientation (6D): 6D rotation representation 
    - Right end-effector position (3D): absolute position in world coordinates
    - Right end-effector orientation (6D): 6D rotation representation
    - Gripper control (2D): left and right gripper commands
    
    Total: 20D action space (no explicit base control)
    
    This action mode:
    1. Solves whole-body IK for desired end-effector poses
    2. Applies base motion either by `mocap+weld` or by wheel/base velocity commands
    3. Applies joint positions from IK solution
    4. Controls grippers
    
    The whole-body IK automatically optimizes base movement to:
    - Reach end-effector targets (highest priority)
    - Keep base on ground (hard constraint)
    - Maintain upright posture (soft constraint)
    - Ensure stability (COM within support polygon)
    """
    
    MAX_STEPS = 50  # Maximum steps for block_until_reached
    POSITION_TOLERANCE = 0.01  # 1cm tolerance for position
    ORIENTATION_TOLERANCE = 0.1  # ~5.7 degrees for orientation
    
    def __init__(
        self,
        position_limits: Optional[tuple[float, float]] = None,
        block_until_reached: Optional[bool] = None,
        direct_mode: Optional[bool] = None,
        control_frequency: Optional[int] = None,
        interpolation_frequency: Optional[int] = None,
        low_level_frequency: Optional[int] = None,
        base_control_mode: Optional[str] = None,
        base_error_gain: Optional[tuple[float, float, float]] = None,
        base_velocity_gain: Optional[tuple[float, float, float]] = None,
        max_base_linear_speed: Optional[float] = None,
        max_base_yaw_speed: Optional[float] = None,
        wheel_radius: Optional[float] = None,
        wheel_half_length: Optional[float] = None,
        wheel_half_width: Optional[float] = None,
        wheel_speed_limit: Optional[float] = None,
        wheel_velocity_signs: Optional[tuple[float, float, float, float]] = None,
        config_path: Optional[str | Path] = None,
    ):
        """Initialize RBY1 Cartesian action mode with whole-body IK.
        
        Args:
            position_limits: End-effector position limits. If None, use YAML default.
            block_until_reached: If None, use YAML default.
            direct_mode: If None, use YAML default.
            control_frequency: If None, use YAML default.
            interpolation_frequency: If None, use YAML default.
            low_level_frequency: If None, use YAML default.
            base_control_mode: `mocap_weld` or `wheel_velocity`. If None, use YAML default.
            base_error_gain: [x, y, yaw] P gains. If None, use YAML default.
            base_velocity_gain: [vx, vy, wz] feed-forward gains. If None, use YAML default.
            max_base_linear_speed: If None, use YAML default.
            max_base_yaw_speed: If None, use YAML default.
            wheel_radius: If None, use YAML default.
            wheel_half_length: If None, use YAML default.
            wheel_half_width: If None, use YAML default.
            wheel_speed_limit: If None, use YAML default.
            wheel_velocity_signs: If None, use YAML default.
            config_path: Optional path to RBY1 WBC YAML.
                Defaults to `<bigym>/bigym/config/rby1_wbc.yaml`.
        """

        # Initialize parent with no floating DOFs (we handle base control via whole-body IK)
        super().__init__(floating_base=False, floating_dofs=None)
        self.uses_internal_substeps = True

        action_mode_cfg = _load_whole_body_action_mode_config(config_path)
        base_cfg = action_mode_cfg.get("base", {})
        if base_cfg is None:
            base_cfg = {}
        if not isinstance(base_cfg, dict):
            raise ValueError(
                "RBY1 WBC config key 'whole_body_action_mode.base' must be a mapping."
            )

        def pick_general(explicit_value, key: str, default):
            if explicit_value is not None:
                return explicit_value
            return action_mode_cfg.get(key, default)

        legacy_base_keys = {
            "control_mode": "base_control_mode",
            "error_gain": "base_error_gain",
            "velocity_gain": "base_velocity_gain",
            "max_linear_speed": "max_base_linear_speed",
            "max_yaw_speed": "max_base_yaw_speed",
            "wheel_radius": "wheel_radius",
            "wheel_half_length": "wheel_half_length",
            "wheel_half_width": "wheel_half_width",
            "wheel_speed_limit": "wheel_speed_limit",
            "wheel_velocity_signs": "wheel_velocity_signs",
        }

        def pick_base(explicit_value, base_key: str, default):
            if explicit_value is not None:
                return explicit_value
            if base_key in base_cfg:
                return base_cfg[base_key]
            legacy_key = legacy_base_keys.get(base_key)
            if legacy_key and legacy_key in action_mode_cfg:
                return action_mode_cfg[legacy_key]
            return default

        resolved_position_limits = (
            _parse_range(position_limits, "position_limits")
            if position_limits is not None
            else _parse_range(
                pick_general(None, "position_limits", (-2.0, 2.0)),
                "whole_body_action_mode.position_limits",
            )
        )
        resolved_block_until_reached = bool(
            pick_general(block_until_reached, "block_until_reached", False)
        )
        resolved_direct_mode = bool(pick_general(direct_mode, "direct_mode", False))
        resolved_control_frequency = int(
            pick_general(control_frequency, "control_frequency", 50)
        )
        resolved_interpolation_frequency = int(
            pick_general(interpolation_frequency, "interpolation_frequency", 50)
        )
        resolved_low_level_frequency = int(
            pick_general(low_level_frequency, "low_level_frequency", 1000)
        )
        resolved_base_control_mode = str(
            pick_base(base_control_mode, "control_mode", "wheel_velocity")
        )

        if resolved_control_frequency <= 0:
            raise ValueError("control_frequency must be > 0.")
        if resolved_interpolation_frequency <= 0:
            raise ValueError("interpolation_frequency must be > 0.")
        if resolved_low_level_frequency <= 0:
            raise ValueError("low_level_frequency must be > 0.")

        if resolved_base_control_mode not in {"mocap_weld", "wheel_velocity"}:
            raise ValueError(
                f"Invalid base_control_mode='{resolved_base_control_mode}'. "
                "Expected one of {'mocap_weld', 'wheel_velocity'}."
            )

        self.base_control_mode = resolved_base_control_mode
        self.position_limits = resolved_position_limits
        self.block_until_reached = resolved_block_until_reached
        self.direct_mode = resolved_direct_mode  # Direct qpos control mode
        self.control_frequency = resolved_control_frequency
        self.interpolation_frequency = (
            resolved_interpolation_frequency  # Frequency for IK waypoints
        )
        self.low_level_frequency = resolved_low_level_frequency  # Physics simulation frequency

        self.base_error_gain = np.asarray(
            pick_base(base_error_gain, "error_gain", (3.0, 3.0, 3.0)),
            dtype=float,
        ).reshape(-1)
        if self.base_error_gain.size != 3:
            raise ValueError("base_error_gain must have 3 elements [x, y, yaw].")
        self.base_velocity_gain = np.asarray(
            pick_base(base_velocity_gain, "velocity_gain", (0.1, 0.1, 0.1)),
            dtype=float,
        ).reshape(-1)
        if self.base_velocity_gain.size != 3:
            raise ValueError("base_velocity_gain must have 3 elements [vx, vy, wz].")

        self.max_base_linear_speed = float(
            pick_base(max_base_linear_speed, "max_linear_speed", 0.6)
        )
        self.max_base_yaw_speed = float(
            pick_base(max_base_yaw_speed, "max_yaw_speed", 1.2)
        )
        self.wheel_radius = float(pick_base(wheel_radius, "wheel_radius", 0.076))
        self.wheel_half_length = float(
            pick_base(wheel_half_length, "wheel_half_length", 0.245)
        )
        self.wheel_half_width = float(
            pick_base(wheel_half_width, "wheel_half_width", 0.245)
        )
        self.wheel_speed_limit = float(
            pick_base(wheel_speed_limit, "wheel_speed_limit", 20.0)
        )
        self.wheel_velocity_signs = np.asarray(
            pick_base(wheel_velocity_signs, "wheel_velocity_signs", (1.0, 1.0, 1.0, 1.0)),
            dtype=float,
        ).reshape(-1)
        if self.wheel_velocity_signs.size != 4:
            raise ValueError("wheel_velocity_signs must have 4 elements (fr, fl, rr, rl).")
        if self.wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be > 0.")

        self._ik_solver = None
        self._base_target_body_id = None
        self._base_free_joint_id = None
        self._base_qpos_adr = None
        self._base_dof_adr = None
        self._wheel_qpos_indices = None
        self._wheel_dof_indices = None
        self._base_weld_eq_id = None
        self._prev_base_target_world = None
        self._last_ik_solution = None  # Store last IK solution to avoid recomputation
        self._last_ik_info = None  # Store IK solver info for debugging
        self._base_z_ref = None
        
    def bind_robot(self, robot, mojo):
        """Bind action mode to robot."""
        super().bind_robot(robot, mojo)
        # IK solver will be initialized when first needed
        self._ik_solver = None
        # Base control handles are resolved lazily on first step/reset
        self._base_target_body_id = None
        self._base_free_joint_id = None
        self._base_qpos_adr = None
        self._base_dof_adr = None
        self._wheel_qpos_indices = None
        self._wheel_dof_indices = None
        self._base_weld_eq_id = None
        self._prev_base_target_world = None
        self._base_z_ref = None
        
    def action_space(self, action_scale: float, seed: Optional[int] = None) -> spaces.Box:
        """Create action space for Cartesian control.
        
        Action space (20D total):
        - Left EE position (3D): [x, y, z] 
        - Left EE orientation (6D): 6D rotation representation
        - Right EE position (3D): [x, y, z]
        - Right EE orientation (6D): 6D rotation representation
        - Gripper control (2D): [left_gripper, right_gripper]
        
        Note: No explicit base control - handled automatically by whole-body IK
        """
        bounds = []
        
        # Left end-effector position (3D)
        pos_bounds = np.array([self.position_limits, self.position_limits, self.position_limits])
        bounds.extend(pos_bounds)
        
        # Left end-effector orientation (6D) - normalized, so roughly [-1, 1]
        ori_bounds = np.array([(-1, 1)] * 6)
        bounds.extend(ori_bounds)
        
        # Right end-effector position (3D) 
        bounds.extend(pos_bounds)
        
        # Right end-effector orientation (6D)
        bounds.extend(ori_bounds)
        
        # Gripper control (2D)
        for _, gripper in self._robot.grippers.items():
            bounds.append(gripper.range)
            
        bounds = np.array(bounds).astype(np.float32)
        low, high = bounds.T
        
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
            seed=seed,
        )
    
    def step(self, action: np.ndarray):
        """Execute Cartesian action using whole-body IK with target interpolation.
        
        Steps:
        1. Calculate number of waypoints from interpolation_frequency and control_frequency
        2. Interpolate end-effector targets for each waypoint
        3. Solve whole-body IK for each interpolated target
        4. Apply IK solution and step physics based on low_level_frequency
        5. Control grippers
        
        Args:
            action: Cartesian action vector (20D)
        """
        # Initialize IK solver if not done yet
        if self._ik_solver is None:
            self._initialize_ik_solver()
            
        # Parse action components
        idx = 0
        
        # Left end-effector pose
        left_pos = action[idx:idx+3]
        idx += 3
        left_rot_6d = action[idx:idx+6] 
        idx += 6
        left_rot_matrix = rotation_6d_to_matrix(left_rot_6d)
        left_quat = Quaternion(matrix=left_rot_matrix, atol=1e-6, rtol=1e-6)
        
        # Right end-effector pose
        right_pos = action[idx:idx+3]
        idx += 3
        right_rot_6d = action[idx:idx+6]
        idx += 6  
        right_rot_matrix = rotation_6d_to_matrix(right_rot_6d)
        right_quat = Quaternion(matrix=right_rot_matrix, atol=1e-6, rtol=1e-6)
        
        # Gripper control
        gripper_action = action[idx:]
        
        # Convert target quaternions to numpy arrays in wxyz format
        target_left_quat_np = np.array([left_quat.w, left_quat.x, left_quat.y, left_quat.z])
        target_right_quat_np = np.array([right_quat.w, right_quat.x, right_quat.y, right_quat.z])
        
        # Store final target poses for convergence checking
        self._target_left_pos = left_pos
        self._target_left_quat = target_left_quat_np
        self._target_right_pos = right_pos
        self._target_right_quat = target_right_quat_np
        
        # Get current end-effector poses for interpolation
        current_left_pose, current_right_pose = self.get_current_ee_poses()
        current_left_pos = current_left_pose.position
        current_left_quat_np = np.array([
            current_left_pose.orientation.w,
            current_left_pose.orientation.x,
            current_left_pose.orientation.y,
            current_left_pose.orientation.z
        ])
        current_right_pos = current_right_pose.position
        current_right_quat_np = np.array([
            current_right_pose.orientation.w,
            current_right_pose.orientation.x,
            current_right_pose.orientation.y,
            current_right_pose.orientation.z
        ])
        
        # Calculate number of waypoints
        num_waypoints = int(self.interpolation_frequency // self.control_frequency)
        if num_waypoints < 1:
            num_waypoints = 1
        
        # Calculate number of physics steps per waypoint
        steps_per_waypoint = int(self.low_level_frequency // self.interpolation_frequency)
        if steps_per_waypoint < 1:
            steps_per_waypoint = 1
        
        # Get model and data references
        model = self._mojo.physics.model._model
        data = self._mojo.physics.data._data
        self._ensure_base_control_handles(model)
        self._set_base_weld_active(model, data, active=(self.base_control_mode == "mocap_weld"))
        
        # Interpolate targets and solve IK for each waypoint
        for waypoint in range(num_waypoints):
            # Calculate interpolation factor (0 to 1)
            alpha = (waypoint + 1) / num_waypoints
            
            # Interpolate end-effector positions
            interp_left_pos = (1 - alpha) * current_left_pos + alpha * left_pos
            interp_right_pos = (1 - alpha) * current_right_pos + alpha * right_pos
            
            # Interpolate quaternions using SLERP (Spherical Linear Interpolation)
            # Convert numpy arrays to Quaternion objects for SLERP
            left_quat_current = Quaternion(current_left_quat_np)
            left_quat_target = Quaternion(target_left_quat_np)
            right_quat_current = Quaternion(current_right_quat_np)
            right_quat_target = Quaternion(target_right_quat_np)
            
            # Perform SLERP interpolation
            left_quat_interp = Quaternion.slerp(left_quat_current, left_quat_target, alpha)
            right_quat_interp = Quaternion.slerp(right_quat_current, right_quat_target, alpha)
            
            # Convert back to numpy arrays in wxyz format
            interp_left_quat = np.array([left_quat_interp.w, left_quat_interp.x, 
                                         left_quat_interp.y, left_quat_interp.z])
            interp_right_quat = np.array([right_quat_interp.w, right_quat_interp.x,
                                          right_quat_interp.y, right_quat_interp.z])
            
            # Get current qpos for IK initialization
            current_qpos = self._mojo.physics.data.qpos.copy()
            
            # Solve whole-body IK for interpolated targets
            ik_solution, success, info = self._ik_solver.solve(
                left_target_pos=interp_left_pos,
                left_target_quat=interp_left_quat,
                right_target_pos=interp_right_pos,
                right_target_quat=interp_right_quat,
                current_qpos=current_qpos,
            )

            if not success:
                # IK failed for this waypoint, skip to next or continue with last solution
                if waypoint == 0:
                    # First waypoint failed, can't continue
                    self._last_ik_info = info
                    return
                # Use last successful solution and continue
                continue
            
            self._last_ik_solution = ik_solution
            self._last_ik_info = info
            
            # Extract base SE(2) target from IK result.
            base_x = float(ik_solution[0])
            base_y = float(ik_solution[1])
            quat = ik_solution[3:7]  # [w, x, y, z]
            base_rz = self._yaw_from_quat_wxyz(quat)
            target_mocap_quat = np.array(
                [np.cos(base_rz / 2.0), 0.0, 0.0, np.sin(base_rz / 2.0)],
                dtype=float,
            )
            
            # Extract joint positions from IK solution
            # Note: qpos structure for RBY1 is:
            # [0:7] base, [7:11] wheels, [11:17] torso, [17:24] right arm, [24:32] right gripper, [32:39] left arm
            torso_joints = ik_solution[11:17]  # Torso at indices 11-16
            right_arm_joints = ik_solution[17:24]  # Right arm at indices 17-23
            left_arm_joints = ik_solution[32:39]  # Left arm at indices 32-38
            
            # Target joint positions
            joint_positions = np.concatenate([torso_joints, right_arm_joints, left_arm_joints])
            
            # Apply IK base target through selected base-control backend.
            if self.base_control_mode == "mocap_weld":
                mocap_id = -1
                if self._base_target_body_id is not None and self._base_target_body_id >= 0:
                    mocap_id = int(model.body_mocapid[self._base_target_body_id])
                if mocap_id >= 0:
                    data.mocap_pos[mocap_id][0] = base_x
                    data.mocap_pos[mocap_id][1] = base_y
                    data.mocap_pos[mocap_id][2] = 0.0  # Keep Z at ground level.
                    data.mocap_quat[mocap_id] = target_mocap_quat

            if self.direct_mode:
                # ===== DIRECT MODE: Set qpos directly =====
                if self.base_control_mode == "mocap_weld":
                    # In weld mode, direct-set base pose directly.
                    base_qadr = int(self._base_qpos_adr) if self._base_qpos_adr is not None else 0
                    data.qpos[base_qadr + 0] = base_x
                    data.qpos[base_qadr + 1] = base_y
                    data.qpos[base_qadr + 3 : base_qadr + 7] = ik_solution[3:7]
                # Set wheel joints
                if self._wheel_qpos_indices is not None and len(self._wheel_qpos_indices) == 4:
                    data.qpos[self._wheel_qpos_indices] = ik_solution[7:11]
                
                # Set torso joints: qpos[11:17]
                data.qpos[11:17] = torso_joints
                # Set right arm joints: qpos[17:24]
                data.qpos[17:24] = right_arm_joints
                # Set left arm joints: qpos[32:39]
                data.qpos[32:39] = left_arm_joints
                
                # ALSO set ctrl to prevent motor drift
                for i, actuator in enumerate(self._robot.limb_actuators):
                    actuator_bound = self._mojo.physics.bind(actuator)
                    actuator_bound.ctrl = joint_positions[i]
                
                # Need to forward after direct qpos modification
                mujoco.mj_forward(model, data)
                
            else:
                # ===== STANDARD MODE: Set ctrl values =====
                for i, actuator in enumerate(self._robot.limb_actuators):
                    actuator_bound = self._mojo.physics.bind(actuator)
                    actuator_bound.ctrl = joint_positions[i]
            
            if self.base_control_mode == "wheel_velocity":
                if self.block_until_reached:
                    warnings.warn(
                        "block_until_reached is not supported in wheel_velocity mode; "
                        "using fixed substep rollout instead.",
                        UserWarning,
                    )
                dt = 1.0 / float(self.low_level_frequency)
                for _ in range(steps_per_waypoint):
                    twist_world, measured_yaw = self._compute_base_twist_world(
                        target_x=base_x,
                        target_y=base_y,
                        target_yaw=base_rz,
                        dt=dt,
                        data=data,
                    )
                    twist_body = self._world_to_body_twist(twist_world, measured_yaw)
                    wheel_vel = self._body_twist_to_wheel_joint_vel(twist_body)
                    self._apply_base_and_wheel_velocity(data, twist_world, wheel_vel)
                    self._mojo.step()
                    self._project_base_pose_to_se2(data)
            else:
                if self.block_until_reached:
                    self._step_until_reached()
                else:
                    # Step the simulation for this waypoint.
                    for _ in range(steps_per_waypoint):
                        self._mojo.step()

        # Control grippers 
        for side, action in zip(self._robot.grippers, gripper_action):
            self._robot.grippers[side].set_control(action)
        # Step 10 times to allow grippers to fully actuate.
        # In wheel-velocity mode, keep base state projected to SE(2) during these extra steps
        # too; otherwise z/roll/pitch drift can reappear after the main rollout loop.
        for _ in range(10):
            if self.base_control_mode == "wheel_velocity":
                self._zero_base_and_wheel_velocity(data)
            self._mojo.step()
            if self.base_control_mode == "wheel_velocity":
                self._project_base_pose_to_se2(data)
        
    def reset(self, reset_state: np.ndarray):
        """Reset robot state.
        
        Args:
            reset_state: Target reset state of robot actuators
        """
        # Check if robot has limb actuators (may not be initialized yet)
        if hasattr(self._robot, 'limb_actuators') and self._robot.limb_actuators:
            # Reset actuators to given state
            if len(reset_state) != len(self._robot.limb_actuators):
                raise ValueError(
                    f"Mismatch between reset_state length "
                    f"({len(reset_state)}) "
                    f"and number of actuators ({len(self._robot.limb_actuators)}). "
                    f"Ensure reset_state matches the actuators count in the model."
                )
            
            for value, actuator in zip(reset_state, self._robot.limb_actuators):
                if actuator.joint:
                    bound_joint = self._mojo.physics.bind(actuator.joint)
                    bound_joint.qpos = value
                    bound_joint.qvel *= 0
                    bound_joint.qacc *= 0
                
                # Set ctrl to match the reset position
                bound_actuator = self._mojo.physics.bind(actuator)
                bound_actuator.ctrl = value
        
        # Clear IK solver to force reinitialization
        # self._ik_solver = None
        self._prev_base_target_world = None
        self._last_ik_solution = None
        self._last_ik_info = None

        if self._mojo is not None and getattr(self._mojo, "physics", None):
            model = self._mojo.physics.model._model
            data = self._mojo.physics.data._data
            self._ensure_base_control_handles(model)
            self._set_base_weld_active(
                model, data, active=(self.base_control_mode == "mocap_weld")
            )
            if self._base_qpos_adr is not None and self._base_qpos_adr >= 0:
                self._base_z_ref = float(data.qpos[int(self._base_qpos_adr) + 2])

    def on_reset_warmup_step(self):
        """Keep wheel-velocity base state on SE(2) during passive reset warmup."""
        if self.base_control_mode != "wheel_velocity":
            return
        if self._mojo is None or getattr(self._mojo, "physics", None) is None:
            return
        data = self._mojo.physics.data._data
        self._zero_base_and_wheel_velocity(data)
        self._project_base_pose_to_se2(data)
        
    def _initialize_ik_solver(self):
        """Initialize the RBY1 whole-body IK solver."""
        # Get MuJoCo model and data from mojo
        model = self._mojo.physics.model._model
        data = self._mojo.physics.data._data
        
        # Create the RBY1 whole-body IK solver
        self._ik_solver = RBY1WholeBodyIK(model, data)

    @staticmethod
    def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
        """Extract yaw (Z rotation) from quaternion in wxyz order."""
        qw, qx, qy, qz = np.asarray(quat, dtype=float).reshape(4)
        return float(
            math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
        )

    @staticmethod
    def _angle_difference(target: float, current: float) -> float:
        """Smallest signed angle from current to target."""
        return float(math.atan2(math.sin(target - current), math.cos(target - current)))

    def _find_joint_id(self, model: mujoco.MjModel, candidates: list[str]) -> int:
        for name in candidates:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                return int(joint_id)
        return -1

    def _find_body_id(self, model: mujoco.MjModel, candidates: list[str]) -> int:
        for name in candidates:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return int(body_id)
        return -1

    def _ensure_base_control_handles(self, model: mujoco.MjModel) -> None:
        """Resolve model handles needed by mocap/wheel base control paths."""
        if self._base_target_body_id is None:
            self._base_target_body_id = self._find_body_id(model, ["base_target"])
            if self._base_target_body_id < 0 and self.base_control_mode == "mocap_weld":
                print("WARNING: base_target mocap body not found in model")

        if self._base_free_joint_id is None:
            self._base_free_joint_id = -1
            self._base_qpos_adr = -1
            self._base_dof_adr = -1
            for joint_id in range(model.njnt):
                if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    self._base_free_joint_id = int(joint_id)
                    self._base_qpos_adr = int(model.jnt_qposadr[joint_id])
                    self._base_dof_adr = int(model.jnt_dofadr[joint_id])
                    break
            if self._base_free_joint_id < 0:
                print("WARNING: could not find free base joint for wheel velocity mode")

        if self._wheel_qpos_indices is None or self._wheel_dof_indices is None:
            wheel_qpos_indices = []
            wheel_dof_indices = []
            for short_name in ("wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl"):
                joint_id = self._find_joint_id(
                    model, [f"rby1/{short_name}", short_name]
                )
                if joint_id < 0:
                    continue
                wheel_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
                wheel_dof_indices.append(int(model.jnt_dofadr[joint_id]))
            self._wheel_qpos_indices = np.asarray(wheel_qpos_indices, dtype=int)
            self._wheel_dof_indices = np.asarray(wheel_dof_indices, dtype=int)
            if self.base_control_mode == "wheel_velocity" and len(self._wheel_dof_indices) != 4:
                print(
                    "WARNING: expected 4 wheel joints for wheel_velocity mode, got "
                    f"{len(self._wheel_dof_indices)}"
                )

        if self._base_weld_eq_id is None:
            self._base_weld_eq_id = -1
            for eq_id in range(model.neq):
                if int(model.eq_type[eq_id]) != int(mujoco.mjtEq.mjEQ_WELD):
                    continue
                body_1 = int(model.eq_obj1id[eq_id])
                body_2 = int(model.eq_obj2id[eq_id])
                body_1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_1) or ""
                body_2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_2) or ""
                names = (body_1_name.lower(), body_2_name.lower())
                if "base_target" in names[0] or "base_target" in names[1]:
                    self._base_weld_eq_id = int(eq_id)
                    break

    def _set_base_weld_active(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        active: bool,
    ) -> None:
        """Enable/disable base_target weld constraint if present."""
        if self._base_weld_eq_id is None:
            self._ensure_base_control_handles(model)
        if self._base_weld_eq_id is None or self._base_weld_eq_id < 0:
            return
        desired = 1 if active else 0
        eq_id = int(self._base_weld_eq_id)
        if int(data.eq_active[eq_id]) == desired:
            return
        data.eq_active[eq_id] = desired
        mujoco.mj_forward(model, data)

    def _compute_base_twist_world(
        self,
        target_x: float,
        target_y: float,
        target_yaw: float,
        dt: float,
        data: mujoco.MjData,
    ) -> tuple[np.ndarray, float]:
        """Compute commanded world-frame base twist from target SE(2)."""
        if self._base_qpos_adr is None or self._base_qpos_adr < 0:
            return np.zeros(3, dtype=float), 0.0

        base_qadr = int(self._base_qpos_adr)
        measured_x = float(data.qpos[base_qadr + 0])
        measured_y = float(data.qpos[base_qadr + 1])
        measured_yaw = self._yaw_from_quat_wxyz(data.qpos[base_qadr + 3 : base_qadr + 7])

        error = np.array(
            [
                target_x - measured_x,
                target_y - measured_y,
                self._angle_difference(target_yaw, measured_yaw),
            ],
            dtype=float,
        )

        desired_velocity_world = np.zeros(3, dtype=float)
        if self._prev_base_target_world is not None and dt > 1e-8:
            prev = self._prev_base_target_world
            desired_velocity_world[0] = (target_x - float(prev[0])) / dt
            desired_velocity_world[1] = (target_y - float(prev[1])) / dt
            desired_velocity_world[2] = self._angle_difference(target_yaw, float(prev[2])) / dt
        self._prev_base_target_world = np.array([target_x, target_y, target_yaw], dtype=float)

        command_world = (
            self.base_error_gain * error
            + self.base_velocity_gain * desired_velocity_world
        )

        linear_norm = float(np.linalg.norm(command_world[:2]))
        if self.max_base_linear_speed > 0.0 and linear_norm > self.max_base_linear_speed:
            command_world[:2] *= self.max_base_linear_speed / max(linear_norm, 1e-8)
        if self.max_base_yaw_speed > 0.0:
            command_world[2] = float(
                np.clip(command_world[2], -self.max_base_yaw_speed, self.max_base_yaw_speed)
            )

        return command_world, measured_yaw

    def _world_to_body_twist(self, twist_world: np.ndarray, yaw: float) -> np.ndarray:
        """Convert [vx, vy, wz] world twist into body frame using current yaw."""
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        vx_world = float(twist_world[0])
        vy_world = float(twist_world[1])
        vx_body = cy * vx_world + sy * vy_world
        vy_body = -sy * vx_world + cy * vy_world
        return np.array([vx_body, vy_body, float(twist_world[2])], dtype=float)

    def _body_twist_to_wheel_joint_vel(self, twist_body: np.ndarray) -> np.ndarray:
        """Map body twist [vx, vy, wz] to wheel angular velocities [fr, fl, rr, rl]."""
        vx = float(twist_body[0])
        vy = float(twist_body[1])
        wz = float(twist_body[2])
        k = self.wheel_half_length + self.wheel_half_width
        r = self.wheel_radius

        # Standard 4-mecanum inverse kinematics (body frame: x-forward, y-left).
        wheel_vel = np.array(
            [
                (vx - vy - k * wz) / r,  # fr
                (vx + vy + k * wz) / r,  # fl
                (vx + vy - k * wz) / r,  # rr
                (vx - vy + k * wz) / r,  # rl
            ],
            dtype=float,
        )
        wheel_vel *= self.wheel_velocity_signs
        if self.wheel_speed_limit > 0.0:
            wheel_vel = np.clip(wheel_vel, -self.wheel_speed_limit, self.wheel_speed_limit)
        return wheel_vel

    def _apply_base_and_wheel_velocity(
        self,
        data: mujoco.MjData,
        twist_world: np.ndarray,
        wheel_vel: np.ndarray,
    ) -> None:
        """Apply base free-joint and wheel joint velocity commands to MuJoCo state."""
        if self._base_dof_adr is not None and self._base_dof_adr >= 0:
            base_dadr = int(self._base_dof_adr)
            # Free-joint qvel order is [vx, vy, vz, wx, wy, wz].
            data.qvel[base_dadr + 0] = float(twist_world[0])
            data.qvel[base_dadr + 1] = float(twist_world[1])
            data.qvel[base_dadr + 2] = 0.0
            data.qvel[base_dadr + 3] = 0.0
            data.qvel[base_dadr + 4] = 0.0
            data.qvel[base_dadr + 5] = float(twist_world[2])

        if self._wheel_dof_indices is not None and len(self._wheel_dof_indices) == 4:
            data.qvel[self._wheel_dof_indices] = wheel_vel

    def _zero_base_and_wheel_velocity(self, data: mujoco.MjData) -> None:
        """Clear base and wheel velocity state for post-rollout stabilization steps."""
        if self._base_dof_adr is not None and self._base_dof_adr >= 0:
            base_dadr = int(self._base_dof_adr)
            data.qvel[base_dadr : base_dadr + 6] = 0.0
        if self._wheel_dof_indices is not None and len(self._wheel_dof_indices) == 4:
            data.qvel[self._wheel_dof_indices] = 0.0

    def _project_base_pose_to_se2(self, data: mujoco.MjData) -> None:
        """
        Keep base pose on SE(2): lock z and roll/pitch while preserving current yaw.
        This suppresses vertical drift and tilt accumulation in wheel_velocity mode.
        """
        if self._base_qpos_adr is None or self._base_qpos_adr < 0:
            return
        if self._base_dof_adr is None or self._base_dof_adr < 0:
            return

        base_qadr = int(self._base_qpos_adr)
        base_dadr = int(self._base_dof_adr)

        if self._base_z_ref is None:
            self._base_z_ref = float(data.qpos[base_qadr + 2])

        quat = np.asarray(data.qpos[base_qadr + 3 : base_qadr + 7], dtype=float)
        yaw = self._yaw_from_quat_wxyz(quat)
        yaw_quat = np.array(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            dtype=float,
        )

        data.qpos[base_qadr + 2] = float(self._base_z_ref)
        data.qpos[base_qadr + 3 : base_qadr + 7] = yaw_quat

        # Keep velocity state consistent with SE(2)-only pose.
        data.qvel[base_dadr + 2] = 0.0
        data.qvel[base_dadr + 3] = 0.0
        data.qvel[base_dadr + 4] = 0.0

        model = self._mojo.physics.model._model
        mujoco.mj_forward(model, data)
    
    def get_last_ik_solution(self) -> tuple[np.ndarray, dict]:
        """Get the last IK solution and info for debugging."""
        return self._last_ik_solution, self._last_ik_info

    def get_current_ee_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current end-effector positions in world frame.
        
        Returns:
            Tuple of (left_pos, right_pos) as numpy arrays
        """
        left_pose, right_pose = self.get_current_ee_poses()
        return left_pose.position, right_pose.position
    
    def get_current_ee_poses(self) -> tuple[Pose, Pose]:
        """Get current end-effector poses.
        
        Returns:
            Tuple of (left_pose, right_pose)
        """
        # RBY1 uses different site names
        left_site = self._robot._wrist_sites[HandSide.LEFT]
        right_site = self._robot._wrist_sites[HandSide.RIGHT]
        
        left_pos = left_site.get_position()
        left_quat = Quaternion(left_site.get_quaternion())
        
        right_pos = right_site.get_position()  
        right_quat = Quaternion(right_site.get_quaternion())
        
        return Pose(left_pos, left_quat), Pose(right_pos, right_quat)
        
    def poses_to_action(
        self, 
        left_pose: Pose, 
        right_pose: Pose, 
        gripper_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert end-effector poses to Cartesian action.
        
        Note: No base position parameters as it's handled automatically by whole-body IK.
        
        Args:
            left_pose: Left end-effector pose
            right_pose: Right end-effector pose  
            gripper_action: Gripper control action
            
        Returns:
            Cartesian action vector (20D)
        """
        action_parts = []
        
        # Left end-effector
        action_parts.extend(left_pose.position)
        left_rot_6d = rotation_matrix_to_6d(left_pose.orientation.rotation_matrix)
        action_parts.extend(left_rot_6d)
        
        # Right end-effector
        action_parts.extend(right_pose.position)
        right_rot_6d = rotation_matrix_to_6d(right_pose.orientation.rotation_matrix)
        action_parts.extend(right_rot_6d)
        
        # Gripper control
        if gripper_action is None:
            gripper_action = np.zeros(len(self._robot.grippers))
        action_parts.extend(gripper_action)
        
        return np.array(action_parts, dtype=np.float32)
    
    # def _step_until_reached(self):
    #     """Step physics until target poses are reached or max steps exceeded."""
    #     steps_counter = 0
    #     while steps_counter < self.MAX_STEPS:
    #         self._mojo.step()
    #         steps_counter += 1
            
    #         if self._is_target_reached():
    #             break
                
    #     if steps_counter >= self.MAX_STEPS and not self._is_target_reached():
    #         warnings.warn(
    #             f"Failed to reach target poses in {self.MAX_STEPS} steps!",
    #             UserWarning,
    #         )
    #     print(steps_counter)
    
    # def _is_target_reached(self) -> bool:
    #     """Check if end-effectors have reached target poses."""
    #     if not hasattr(self, '_target_left_pos'):
    #         return True  # No targets set yet
            
    #     # Get current end-effector poses
    #     left_pose, right_pose = self.get_current_ee_poses()
        
    #     # Check left end-effector position
    #     left_pos_error = np.linalg.norm(left_pose.position - self._target_left_pos)
    #     if left_pos_error > self.POSITION_TOLERANCE:
    #         return False
            
    #     # Check right end-effector position
    #     right_pos_error = np.linalg.norm(right_pose.position - self._target_right_pos)
    #     if right_pos_error > self.POSITION_TOLERANCE:
    #         return False
            
    #     # Check left end-effector orientation
    #     left_quat_current = np.array([
    #         left_pose.orientation.w,
    #         left_pose.orientation.x, 
    #         left_pose.orientation.y,
    #         left_pose.orientation.z
    #     ])
    #     left_quat_diff = Quaternion(self._target_left_quat) * Quaternion(left_quat_current).inverse
    #     left_angle_error = 2 * np.arccos(np.clip(abs(left_quat_diff.w), -1, 1))
    #     if left_angle_error > self.ORIENTATION_TOLERANCE:
    #         return False
            
    #     # Check right end-effector orientation
    #     right_quat_current = np.array([
    #         right_pose.orientation.w,
    #         right_pose.orientation.x,
    #         right_pose.orientation.y,
    #         right_pose.orientation.z
    #     ])
    #     right_quat_diff = Quaternion(self._target_right_quat) * Quaternion(right_quat_current).inverse
    #     right_angle_error = 2 * np.arccos(np.clip(abs(right_quat_diff.w), -1, 1))
    #     if right_angle_error > self.ORIENTATION_TOLERANCE:
    #         return False
            
    #     return True

    def _step_until_reached(self):
        """Step physics until the target position is reached."""
        steps_counter = 0
        while True:
            self._mojo.step()
            steps_counter += 1
            if self._is_target_state_reached() or steps_counter >= self.MAX_STEPS:
                if steps_counter >= self.MAX_STEPS:
                    warnings.warn(
                        f"Failed to reach target state in " f"{self.MAX_STEPS} steps!",
                        TargetStateNotReachedWarning,
                    )
                break

    def _is_target_state_reached(self):
        if self.floating_base:
            if not self._robot.floating_base.is_target_reached:
                return False
        for actuator in self._robot.limb_actuators:
            if not is_target_reached(actuator, self._mojo.physics, TOLERANCE_ANGULAR):
                return False
        return True
