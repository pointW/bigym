"""RBY1 Cartesian space action mode with whole-body IK.

This version uses whole-body IK to automatically optimize base movement
along with joint positions to reach end-effector targets.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from gymnasium import spaces
from pyquaternion import Quaternion
import mujoco

from bigym.action_modes import ActionMode, TargetStateNotReachedWarning
from bigym.const import HandSide, TOLERANCE_ANGULAR
from bigym.ik.rby1_whole_body_ik import RBY1WholeBodyIK
from bigym.utils.physics_utils import (
    is_target_reached,
)
from vr.ik.h1_upper_body_ik import Pose
import warnings


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
    2. Moves base_target mocap body to the optimized base position
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
        position_limits: tuple[float, float] = (-2.0, 2.0),
        block_until_reached: bool = False,
        direct_mode: bool = False,
        control_frequency: int = 50,
        interpolation_frequency: int = 200,
        low_level_frequency: int = 1000,
    ):
        """Initialize RBY1 Cartesian action mode with whole-body IK.
        
        Args:
            position_limits: Min/max limits for end-effector positions
            block_until_reached: Whether to block until position is reached
            direct_mode: If True, directly set joint qpos (bypassing controllers)
            control_frequency: Control frequency in Hz (default: 50)
            interpolation_frequency: Frequency for IK waypoints in Hz (default: 100)
            low_level_frequency: Physics simulation frequency in Hz (default: 500)
        """
        
        # Initialize parent with no floating DOFs (we handle base control via whole-body IK)
        super().__init__(floating_base=False, floating_dofs=None)
        
        self.position_limits = position_limits
        self.block_until_reached = block_until_reached
        self.direct_mode = direct_mode  # Direct qpos control mode
        self.control_frequency = control_frequency
        self.interpolation_frequency = interpolation_frequency  # Frequency for IK waypoints
        self.low_level_frequency = low_level_frequency  # Physics simulation frequency
        self._ik_solver = None
        self._base_target_body_id = None
        self._last_ik_solution = None  # Store last IK solution to avoid recomputation
        self._last_ik_info = None  # Store IK solver info for debugging
        
    def bind_robot(self, robot, mojo):
        """Bind action mode to robot."""
        super().bind_robot(robot, mojo)
        # IK solver will be initialized when first needed
        self._ik_solver = None
        # Base target body ID will be set later when needed
        self._base_target_body_id = None
        
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
        
        # Find base_target mocap body if not done yet
        if self._base_target_body_id is None:
            model = self._mojo.physics.model._model
            self._base_target_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "base_target"
            )
            if self._base_target_body_id < 0:
                # Mocap body doesn't exist, we need to handle this
                print("WARNING: base_target mocap body not found in model")
                self._base_target_body_id = -1
            
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
                left_body_relative=False,  # Always use world frame
                right_body_relative=False,  # Always use world frame
                current_qpos=current_qpos,
                max_iterations=100,
                tolerance=0.001,
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
            
            # Extract base position from IK solution for mocap target
            base_x = ik_solution[0]
            base_y = ik_solution[1]
            # Extract rotation from quaternion (only Z rotation for wheeled base)
            quat = ik_solution[3:7]  # [w, x, y, z]
            # Convert quaternion to euler angles, extract Z rotation (yaw)
            base_rz = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                                1 - 2*(quat[2]**2 + quat[3]**2))
            
            # Prepare target mocap quaternion for Z rotation
            target_mocap_quat = np.array([np.cos(base_rz / 2), 0, 0, np.sin(base_rz / 2)])
            
            # Extract joint positions from IK solution
            # Note: qpos structure for RBY1 is:
            # [0:7] base, [7:11] wheels, [11:17] torso, [17:24] right arm, [24:32] right gripper, [32:39] left arm
            torso_joints = ik_solution[11:17]  # Torso at indices 11-16
            right_arm_joints = ik_solution[17:24]  # Right arm at indices 17-23
            left_arm_joints = ik_solution[32:39]  # Left arm at indices 32-38
            
            # Target joint positions
            joint_positions = np.concatenate([torso_joints, right_arm_joints, left_arm_joints])
            
            # Apply IK solution and step physics
            mocap_id = model.body_mocapid[self._base_target_body_id]
            if mocap_id >= 0:
                # Set mocap position
                data.mocap_pos[mocap_id][0] = base_x
                data.mocap_pos[mocap_id][1] = base_y
                data.mocap_pos[mocap_id][2] = 0.0  # Keep Z at ground level
                data.mocap_quat[mocap_id] = target_mocap_quat
            
            if self.direct_mode:
                # ===== DIRECT MODE: Set qpos directly =====
                # Set base position
                data.qpos[0] = base_x
                data.qpos[1] = base_y
                # data.qpos[2] is Z, keep as is (should be 0)
                
                # Set base quaternion
                data.qpos[3:7] = ik_solution[3:7]
                
                # Set wheel joints
                data.qpos[7:11] = ik_solution[7:11]
                
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
            
            if self.block_until_reached:
                self._step_until_reached()
            else:
                # Step the simulation for this waypoint
                for _ in range(steps_per_waypoint):
                    self._mojo.step()

        # Control grippers 
        for side, action in zip(self._robot.grippers, gripper_action):
            self._robot.grippers[side].set_control(action)
        # Step 10 times to allow grippers to fully actuate
        for _ in range(10):
            self._mojo.step()
        
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
        
        # Reset base_target mocap position to origin
        if hasattr(self, '_base_target_body_id') and self._base_target_body_id is not None and self._base_target_body_id >= 0:
            model = self._mojo.physics.model._model
            data = self._mojo.physics.data._data
            mocap_id = model.body_mocapid[self._base_target_body_id]
            if mocap_id >= 0:
                # Reset mocap to origin
                data.mocap_pos[mocap_id] = [0, 0, 0]
                data.mocap_quat[mocap_id] = [1, 0, 0, 0]  # Identity quaternion
        
        # Clear IK solver to force reinitialization
        self._ik_solver = None
        self._last_ik_solution = None
        self._last_ik_info = None
        
    def _initialize_ik_solver(self):
        """Initialize the RBY1 whole-body IK solver."""
        # Get MuJoCo model and data from mojo
        model = self._mojo.physics.model._model
        data = self._mojo.physics.data._data
        
        # Create the RBY1 whole-body IK solver
        self._ik_solver = RBY1WholeBodyIK(model, data)
    
    def get_last_ik_solution(self) -> tuple[np.ndarray, dict]:
        """Get the last IK solution and info for debugging.
        
        Returns:
            Tuple of (ik_solution_qpos, ik_info_dict)
            Returns (None, None) if no IK has been solved yet
        """
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