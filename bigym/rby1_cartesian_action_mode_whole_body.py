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
    ):
        """Initialize RBY1 Cartesian action mode with whole-body IK.
        
        Args:
            position_limits: Min/max limits for end-effector positions
            block_until_reached: Whether to block until position is reached
            direct_mode: If True, directly set joint qpos (bypassing controllers)
            control_frequency: Control frequency in Hz (default: 50)
        """
        
        # Initialize parent with no floating DOFs (we handle base control via whole-body IK)
        super().__init__(floating_base=False, floating_dofs=None)
        
        self.position_limits = position_limits
        self.block_until_reached = block_until_reached
        self.direct_mode = direct_mode  # Direct qpos control mode
        self.control_frequency = control_frequency
        self.low_level_frequency = 500  # Physics simulation frequency
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
        """Execute Cartesian action using whole-body IK.
        
        Steps:
        1. Solve whole-body IK for desired end-effector poses
        2. Move base_target mocap body to optimized base position
        3. Apply joint positions from IK solution
        4. Control grippers
        
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
        left_quat_np = np.array([left_quat.w, left_quat.x, left_quat.y, left_quat.z])
        right_quat_np = np.array([right_quat.w, right_quat.x, right_quat.y, right_quat.z])
        
        # Get current qpos for IK initialization
        current_qpos = self._mojo.physics.data.qpos.copy()
                
        # Store target poses for convergence checking
        self._target_left_pos = left_pos
        self._target_left_quat = left_quat_np
        self._target_right_pos = right_pos
        self._target_right_quat = right_quat_np
    
        # Step 1: Solve whole-body IK for target poses
        # The IK solver will optimize base position along with joint positions
        # Don't use body-relative mode as it causes instability
        ik_solution, success, info = self._ik_solver.solve(
            left_target_pos=left_pos,
            left_target_quat=left_quat_np,
            right_target_pos=right_pos,
            right_target_quat=right_quat_np,
            left_body_relative=False,  # Always use world frame
            right_body_relative=False,  # Always use world frame
            current_qpos=current_qpos,
            max_iterations=100,
            tolerance=0.001,
        )
        
        if not success:
            # IK failed, keep current positions but still control grippers
            for side, action in zip(self._robot.grippers, gripper_action):
                self._robot.grippers[side].set_control(action)
            self._mojo.step()
            self._last_ik_info = info  # Store even on failure
            return
        
        self._last_ik_solution = ik_solution
        self._last_ik_info = info  # Store IK solver info
        
        # Step 2: Extract base position from IK solution for mocap target
        # IK solution qpos structure: [base_x, base_y, base_z, quat_w, quat_x, quat_y, quat_z, ...]
        base_x = ik_solution[0]
        base_y = ik_solution[1]
        # Extract rotation from quaternion (only Z rotation for wheeled base)
        quat = ik_solution[3:7]  # [w, x, y, z]
        # Convert quaternion to euler angles, extract Z rotation (yaw)
        base_rz = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                            1 - 2*(quat[2]**2 + quat[3]**2))
        
        # Prepare target mocap quaternion for Z rotation
        target_mocap_quat = np.array([np.cos(base_rz / 2), 0, 0, np.sin(base_rz / 2)])
        
        # Get current mocap pose for interpolation 
        model = self._mojo.physics.model._model
        data = self._mojo.physics.data._data
        
        mocap_id = model.body_mocapid[self._base_target_body_id]
        if mocap_id >= 0:
            current_mocap_pos = data.mocap_pos[mocap_id].copy()
            current_mocap_quat = data.mocap_quat[mocap_id].copy()
        
        # Step 3: Apply joint positions from IK solution
        # RBY1 qpos structure: [base(3), quat(4), wheels(4), torso(6), right_arm(7), left_arm(7)]
        
        # Step 3: Extract joint positions from IK solution and interpolate
        # Note: qpos structure for RBY1 is:
        # [0:7] base, [7:11] wheels, [11:17] torso, [17:24] right arm, [24:32] right gripper, [32:39] left arm
        torso_joints = ik_solution[11:17]  # Torso at indices 11-16
        right_arm_joints = ik_solution[17:24]  # Right arm at indices 17-23
        left_arm_joints = ik_solution[32:39]  # Left arm at indices 32-38 (NOT 24-30!)
        
        # Target joint positions
        joint_positions = np.concatenate([torso_joints, right_arm_joints, left_arm_joints])
        
        # Get current joint positions for interpolation (ALWAYS from qpos for accuracy)
        # Get current qpos values for joints we're controlling
        current_torso = data.qpos[11:17].copy()
        current_right_arm = data.qpos[17:24].copy()
        current_left_arm = data.qpos[32:39].copy()
        current_joint_positions = np.concatenate([current_torso, current_right_arm, current_left_arm])
        
        # Also get current base position for interpolation (used in direct mode)
        current_base_x = data.qpos[0]
        current_base_y = data.qpos[1]
        current_base_quat = data.qpos[3:7].copy()
        current_wheels = data.qpos[7:11].copy()
        
        # Calculate number of interpolation steps
        num_steps = int(self.low_level_frequency // self.control_frequency)
        
        # Linear interpolation over multiple physics steps
        for step in range(num_steps):
            # Calculate interpolation factor (0 to 1)
            alpha = (step + 1) / num_steps
            
            # Interpolate mocap body
            mocap_id = model.body_mocapid[self._base_target_body_id]
            # Interpolate mocap position (only X and Y)
            data.mocap_pos[mocap_id][0] = (1 - alpha) * current_mocap_pos[0] + alpha * base_x
            data.mocap_pos[mocap_id][1] = (1 - alpha) * current_mocap_pos[1] + alpha * base_y
            data.mocap_pos[mocap_id][2] = 0.0  # Keep Z at ground level
            
            # Interpolate mocap quaternion (SLERP would be better but linear is ok for small rotations)
            interp_mocap_quat = (1 - alpha) * current_mocap_quat + alpha * target_mocap_quat
            # Normalize quaternion
            interp_mocap_quat = interp_mocap_quat / np.linalg.norm(interp_mocap_quat)
            data.mocap_quat[mocap_id] = interp_mocap_quat
            
            if self.direct_mode:
                # ===== DIRECT MODE: Interpolate and set qpos directly =====
                
                # Interpolate base position (only if direct mode controls base)
                interp_base_x = (1 - alpha) * current_base_x + alpha * base_x
                interp_base_y = (1 - alpha) * current_base_y + alpha * base_y
                
                # Set base X, Y from interpolated values
                data.qpos[0] = interp_base_x
                data.qpos[1] = interp_base_y
                # data.qpos[2] is Z, keep as is (should be 0)
                
                # Interpolate base quaternion (using SLERP would be better but linear is acceptable for small rotations)
                interp_base_quat = (1 - alpha) * current_base_quat + alpha * ik_solution[3:7]
                # Normalize quaternion
                interp_base_quat = interp_base_quat / np.linalg.norm(interp_base_quat)
                data.qpos[3:7] = interp_base_quat
                
                # Interpolate wheel joints
                interp_wheels = (1 - alpha) * current_wheels + alpha * ik_solution[7:11]
                data.qpos[7:11] = interp_wheels
                
                # Interpolate joint positions
                interp_joint_positions = (1 - alpha) * current_joint_positions + alpha * joint_positions
                
                # DEBUG: Print joint values at first and last interpolation step
                # if (step == 0 or step == num_steps - 1):
                #     print(f"\n=== Direct Mode Step {step}/{num_steps} (alpha={alpha:.3f}) ===")
                #     print(f"Before setting qpos:")
                #     print(f"  Left arm qpos[32:39]: {data.qpos[32:39]}")
                #     print(f"  Target interp positions[13:20]: {interp_joint_positions[13:20]}")
                
                # Set torso joints: qpos[11:17]
                data.qpos[11:17] = interp_joint_positions[0:6]
                # Set right arm joints: qpos[17:24]
                data.qpos[17:24] = interp_joint_positions[6:13]
                # Set left arm joints: qpos[32:39]
                data.qpos[32:39] = interp_joint_positions[13:20]
                
                # if (step == 0 or step == num_steps - 1):
                #     print(f"After setting qpos:")
                #     print(f"  Left arm qpos[32:39]: {data.qpos[32:39]}")
                
                # ALSO set ctrl to prevent motor drift
                for i, actuator in enumerate(self._robot.limb_actuators):
                    # Set ctrl to the interpolated position
                    actuator_bound = self._mojo.physics.bind(actuator)
                    actuator_bound.ctrl = interp_joint_positions[i]
                
                # if (step == 0 or step == num_steps - 1):
                #     print(f"After setting ctrl:")
                #     print(f"  ctrl[13:20]: {data.ctrl[13:20]}")
                
                # Need to forward after direct qpos modification
                mujoco.mj_forward(model, data)
                
                # if (step == 0 or step == num_steps - 1):
                #     print(f"After mj_forward:")
                #     print(f"  Left arm qpos[32:39]: {data.qpos[32:39]}")
                
            else:
                # ===== STANDARD MODE: Interpolate ctrl values =====
                
                # Interpolate joint positions
                interp_joint_positions = (1 - alpha) * current_joint_positions + alpha * joint_positions
                
                for i, actuator in enumerate(self._robot.limb_actuators):
                    # Set ctrl to the interpolated position
                    actuator_bound = self._mojo.physics.bind(actuator)
                    actuator_bound.ctrl = interp_joint_positions[i]
            
            # Step the simulation for this interpolation step
            if self.block_until_reached:
                self._step_until_reached()
            else:
                self._mojo.step()
        
        # Step 4: Control grippers at the end and step 100 times for gripper action
        # if self.direct_mode:
        #     print(f"\n=== Before gripper control steps ===")
        #     print(f"  Left arm qpos[32:39]: {data.qpos[32:39]}")
        
        for side, action in zip(self._robot.grippers, gripper_action):
            self._robot.grippers[side].set_control(action)
        
        # Step 100 times to allow grippers to fully actuate
        for gripper_step in range(100):
            self._mojo.step()
            # if self.direct_mode and (gripper_step == 0 or gripper_step == 99):
            #     print(f"\n=== After gripper step {gripper_step+1}/100 ===")
            #     print(f"  Left arm qpos[32:39]: {data.qpos[32:39]}")
        
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