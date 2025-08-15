"""Cartesian space action mode for BiGym robots."""
from __future__ import annotations

from typing import Optional
import numpy as np
from gymnasium import spaces
from pyquaternion import Quaternion

from bigym.action_modes import ActionMode, JointPositionActionMode
from bigym.const import HandSide
from vr.ik.h1_upper_body_ik import H1UpperBodyIK, Pose


def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        6D rotation vector (first two rows of rotation matrix flattened)
    """
    return rotation_matrix[:2, :].flatten()


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.
    
    Uses Gram-Schmidt process to orthonormalize the vectors.
    Based on "On the Continuity of Rotation Representations in Neural Networks"
    
    Args:
        rotation_6d: 6D rotation vector (first two columns of rotation matrix)
        
    Returns:
        3x3 rotation matrix
    """
    # Extract the two 3D vectors
    x_raw = rotation_6d[:3]
    y_raw = rotation_6d[3:6]
    
    # Normalize first vector to get x
    x = x_raw / np.linalg.norm(x_raw)
    
    # Make second vector orthogonal to first, then normalize to get y
    y = y_raw - np.dot(y_raw, x) * x
    y = y / np.linalg.norm(y)
    
    # Third vector is cross product
    z = np.cross(x, y)
    
    # Stack as columns to form rotation matrix
    return np.column_stack([x, y, z])


class CartesianActionMode(JointPositionActionMode):
    """Control robot arms through end-effector poses in Cartesian space.
    
    The action space consists of:
    - Left end-effector position (3D): absolute position in world coordinates
    - Left end-effector orientation (6D): 6D rotation representation 
    - Right end-effector position (3D): absolute position in world coordinates
    - Right end-effector orientation (6D): 6D rotation representation
    - Base control (if floating_base=True): same as JointPositionActionMode
    - Gripper control (2D): left and right gripper commands
    
    Uses inverse kinematics to convert Cartesian poses to joint positions.
    """
    
    def __init__(
        self,
        floating_base: bool = True,
        floating_dofs: Optional[list] = None,
        position_limits: tuple[float, float] = (-2.0, 2.0),
        ik_solver: str = "original",
    ):
        """Initialize Cartesian action mode.
        
        Args:
            floating_base: Whether to enable floating base control
            floating_dofs: Floating DOFs to use
            position_limits: Min/max limits for end-effector positions
            ik_solver: IK solver to use ("original" or "mink")
        """
        # Initialize as JointPositionActionMode with absolute control
        super().__init__(
            absolute=True, 
            floating_base=floating_base, 
            floating_dofs=floating_dofs
        )
        self.position_limits = position_limits
        self.ik_solver_type = ik_solver
        self._ik_solver = None
        
    def bind_robot(self, robot, mojo):
        """Bind action mode to robot."""
        super().bind_robot(robot, mojo)
        # IK solver will be initialized when first needed
        self._ik_solver = None
        
    def action_space(self, action_scale: float, seed: Optional[int] = None) -> spaces.Box:
        """Create action space for Cartesian control.
        
        Action space:
        - Left EE position (3D): [x, y, z] 
        - Left EE orientation (6D): 6D rotation representation
        - Right EE position (3D): [x, y, z]
        - Right EE orientation (6D): 6D rotation representation  
        - Base control (variable): depends on floating_base configuration
        - Gripper control (2D): [left_gripper, right_gripper]
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
        
        # Base control (if enabled)
        if self.floating_base:
            base_bounds = self._robot.floating_base.get_action_bounds()
            base_bounds = [np.array(b) * action_scale for b in base_bounds]
            bounds.extend(base_bounds)
            
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
        """Execute Cartesian action by converting to joint positions via IK.
        
        Args:
            action: Cartesian action vector
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
        left_quat = Quaternion(matrix=left_rot_matrix)
        
        # Right end-effector pose
        right_pos = action[idx:idx+3]
        idx += 3
        right_rot_6d = action[idx:idx+6]
        idx += 6  
        right_rot_matrix = rotation_6d_to_matrix(right_rot_6d)
        right_quat = Quaternion(matrix=right_rot_matrix)
        
        # Base control (if enabled)
        base_action = None
        if self.floating_base:
            base_dof = self._robot.floating_base.dof_amount
            base_action = action[idx:idx+base_dof]
            idx += base_dof
            
        # Gripper control
        gripper_action = action[idx:]
        
        # Get actual pelvis pose from the environment
        pelvis = self._robot.pelvis
        pelvis_pos = pelvis.get_position()
        pelvis_quat = Quaternion(pelvis.get_quaternion())
        pelvis_pose = Pose(pelvis_pos, pelvis_quat)
        
        # Get current arm joint positions
        start_index = self._robot.floating_base.dof_amount if self.floating_base else 0
        end_index = start_index + len(self._robot.limb_actuators)
        arms_qpos = np.array(self._robot.qpos_actuated[start_index:end_index])
        qpos_arm_left, qpos_arm_right = np.split(arms_qpos, 2)
        
        # Solve IK for target poses
        ik_solution = self._ik_solver.solve(
            pelvis_pose=pelvis_pose,
            qpos_arm_left=qpos_arm_left,
            qpos_arm_right=qpos_arm_right,
            target_pose_left=Pose(left_pos, left_quat),
            target_pose_right=Pose(right_pos, right_quat),
        )
        
        # Construct joint action
        joint_action = []
        
        # Add base control if enabled
        if self.floating_base and base_action is not None:
            joint_action.extend(base_action)
            
        # Add arm joint positions from IK
        joint_action.extend(ik_solution)
        
        # Add gripper control
        joint_action.extend(gripper_action)
        
        # Execute via parent JointPositionActionMode
        super().step(np.array(joint_action))
        
    def reset(self, reset_state: np.ndarray):
        """Reset robot state."""
        super().reset(reset_state)
        # Initialize IK solver after reset to capture correct initial state
        if self._ik_solver is None and self._robot is not None and self._mojo is not None:
            self._initialize_ik_solver()
        
    def _initialize_ik_solver(self):
        """Initialize the IK solver based on selected type."""
        # Create a minimal wrapper that provides the interface IK solvers expect
        class ActualEnvWrapper:
            def __init__(self, robot, mojo):
                self.robot = robot
                self.mojo = mojo
        
        # Use the actual robot and mojo references that were bound to this action mode
        env_wrapper = ActualEnvWrapper(self._robot, self._mojo)
        
        # Create the IK solver based on selected type
        if self.ik_solver_type == "mink":
            try:
                from bigym.ik.mink_h1_ik import MinkH1UpperBodyIK
                self._ik_solver = MinkH1UpperBodyIK(env_wrapper)
            except ImportError:
                print("Warning: Mink IK solver not available, falling back to original")
                self._ik_solver = H1UpperBodyIK(env_wrapper)
        else:  # "original"
            self._ik_solver = H1UpperBodyIK(env_wrapper)
        
        # Note: Calibration is disabled as it was making accuracy worse
        self._calibrate_ik_solver()
    
    def _calibrate_ik_solver(self):
        """Calibration is currently disabled as it was making accuracy worse."""
        # The H1UpperBodyIK solver has fundamental model differences:
        # - Expects 4 joints per arm (actual has 5)
        # - Removes all non-H1 bodies from the model
        # Now using actual pelvis pose from environment instead of hardcoded value
        pass
        
    def get_current_ee_poses(self) -> tuple[Pose, Pose]:
        """Get current end-effector poses.
        
        Returns:
            Tuple of (left_pose, right_pose)
        """
        left_site = self._robot._wrist_sites[HandSide.LEFT]
        right_site = self._robot._wrist_sites[HandSide.RIGHT]
        
        left_pos = left_site.get_position()
        left_quat = Quaternion(left_site.get_quaternion())
        
        right_pos = right_site.get_position()  
        right_quat = Quaternion(right_site.get_quaternion())
        
        return Pose(left_pos, left_quat), Pose(right_pos, right_quat)
        
    def poses_to_action(self, left_pose: Pose, right_pose: Pose, base_action: Optional[np.ndarray] = None, gripper_action: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert end-effector poses to Cartesian action.
        
        Args:
            left_pose: Left end-effector pose
            right_pose: Right end-effector pose  
            base_action: Base control action (if floating_base enabled)
            gripper_action: Gripper control action
            
        Returns:
            Cartesian action vector
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
        
        # Base action
        if self.floating_base:
            if base_action is not None:
                action_parts.extend(base_action)
            else:
                # Use zero base action as default
                base_dof = self._robot.floating_base.dof_amount if self._robot else 3
                action_parts.extend([0.0] * base_dof)
                
        # Gripper action
        if gripper_action is not None:
            action_parts.extend(gripper_action)
        else:
            # Use zero gripper action as default
            action_parts.extend([0.0, 0.0])
            
        return np.array(action_parts)