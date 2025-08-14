"""Cartesian action mode with direct qpos control (mink-style)."""
from __future__ import annotations

from typing import Optional
import numpy as np
from gymnasium import spaces

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.const import HandSide
from vr.ik.h1_upper_body_ik import H1UpperBodyIK, Pose
from pyquaternion import Quaternion


def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D rotation representation."""
    return rotation_matrix[:2, :].flatten()


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    x_raw = rotation_6d[:3]
    y_raw = rotation_6d[3:6]
    
    x = x_raw / np.linalg.norm(x_raw)
    y = y_raw - np.dot(y_raw, x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    
    return np.column_stack([x, y, z])


class CartesianActionModeDirect(JointPositionActionMode):
    """Cartesian control with direct qpos manipulation (mink-style).
    
    This mode achieves excellent accuracy by:
    1. Computing IK to get target joint positions
    2. DIRECTLY setting joint qpos values (bypassing motor PD controllers)
    3. Using the same approach as mink library
    
    This is fundamentally different from sending commands through motor
    actuators, which have low implicit PD gains that limit accuracy.
    """
    
    def __init__(
        self,
        floating_base: bool = True,
        floating_dofs: Optional[list] = None,
        position_limits: tuple[float, float] = (-2.0, 2.0),
    ):
        """Initialize direct Cartesian action mode.
        
        Args:
            floating_base: Whether to enable floating base control
            floating_dofs: Floating DOFs (defaults to 4 DOF as per paper)
            position_limits: Min/max limits for end-effector positions
        """
        # Default to 4 DOF for base as specified in paper
        if floating_dofs is None and floating_base:
            floating_dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        
        # Initialize parent - Note: we don't use block_until_reached
        # since we're directly setting positions
        super().__init__(
            absolute=True,
            block_until_reached=False,
            floating_base=floating_base,
            floating_dofs=floating_dofs
        )
        
        self.position_limits = position_limits
        self._ik_solver = None
        
    def bind_robot(self, robot, mojo):
        """Bind action mode to robot."""
        super().bind_robot(robot, mojo)
        self._ik_solver = None
        
    def action_space(self, action_scale: float, seed: Optional[int] = None) -> spaces.Box:
        """Create action space for Cartesian control."""
        bounds = []
        
        # Left end-effector position (3D) and orientation (6D)
        pos_bounds = np.array([self.position_limits, self.position_limits, self.position_limits])
        bounds.extend(pos_bounds)
        ori_bounds = np.array([(-1, 1)] * 6)
        bounds.extend(ori_bounds)
        
        # Right end-effector position (3D) and orientation (6D)
        bounds.extend(pos_bounds)
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
        
        return spaces.Box(low=low, high=high, dtype=np.float32, seed=seed)
    
    def step(self, action: np.ndarray):
        """Execute Cartesian action with direct qpos control.
        
        This is the KEY DIFFERENCE from standard CartesianActionMode:
        We directly set joint positions (qpos) instead of sending
        commands through motor actuators.
        """
        # Initialize IK solver if needed
        if self._ik_solver is None:
            self._initialize_ik_solver()
        
        # Parse action components
        idx = 0
        
        # Left end-effector
        left_pos = action[idx:idx+3]
        idx += 3
        left_rot_6d = action[idx:idx+6]
        idx += 6
        left_quat = Quaternion(matrix=rotation_6d_to_matrix(left_rot_6d))
        
        # Right end-effector
        right_pos = action[idx:idx+3]
        idx += 3
        right_rot_6d = action[idx:idx+6]
        idx += 6
        right_quat = Quaternion(matrix=rotation_6d_to_matrix(right_rot_6d))
        
        # Base and gripper control
        base_action = None
        if self.floating_base:
            base_dof = self._robot.floating_base.dof_amount
            base_action = action[idx:idx+base_dof]
            idx += base_dof
        gripper_action = action[idx:]
        
        # Get current pelvis pose from actual environment
        pelvis = self._robot.pelvis
        pelvis_pos = pelvis.get_position()
        pelvis_quat = Quaternion(pelvis.get_quaternion())
        pelvis_pose = Pose(pelvis_pos, pelvis_quat)
        
        # Get current arm joints
        start_idx = self._robot.floating_base.dof_amount if self.floating_base else 0
        end_idx = start_idx + len(self._robot.limb_actuators)
        arms_qpos = np.array(self._robot.qpos_actuated[start_idx:end_idx])
        qpos_arm_left, qpos_arm_right = np.split(arms_qpos, 2)
        
        # Solve IK
        ik_solution = self._ik_solver.solve(
            pelvis_pose=pelvis_pose,
            qpos_arm_left=qpos_arm_left[:5],
            qpos_arm_right=qpos_arm_right[:5],
            target_pose_left=Pose(left_pos, left_quat),
            target_pose_right=Pose(right_pos, right_quat),
        )
        
        # ===== CRITICAL DIFFERENCE: DIRECT QPOS CONTROL =====
        # Instead of setting actuator.ctrl (which goes through motor PD controller),
        # we directly set joint.qpos (bypassing the controller entirely)
        
        # Set base control normally (it already uses direct control)
        if self.floating_base and base_action is not None:
            self._robot.floating_base.set_control(base_action)
        
        # DIRECTLY SET ARM JOINT POSITIONS (mink-style)
        # CRITICAL FIX: Also set ctrl to prevent motor drift
        for i, actuator in enumerate(self._robot.limb_actuators):
            joint = actuator.joint
            if joint:
                bound_joint = self._mojo.physics.bind(joint)
                bound_actuator = self._mojo.physics.bind(actuator)
                
                # Direct position assignment - no PD controller involved!
                bound_joint.qpos = ik_solution[i]
                
                # ALSO set ctrl to the same value to prevent the motor
                # from trying to drive the joint back to ctrl=0
                bound_actuator.ctrl = ik_solution[i]
        
        # Set gripper control normally
        for side, grip_action in zip(self._robot.grippers, gripper_action):
            self._robot.grippers[side].set_control(grip_action)
        
        # Step physics once
        self._mojo.step()
        
    def _initialize_ik_solver(self):
        """Initialize the IK solver."""
        class ActualEnvWrapper:
            def __init__(self, robot, mojo):
                self.robot = robot
                self.mojo = mojo
        
        env_wrapper = ActualEnvWrapper(self._robot, self._mojo)
        self._ik_solver = H1UpperBodyIK(env_wrapper)
        
    def reset(self, reset_state: np.ndarray):
        """Reset robot state."""
        super().reset(reset_state)
        if self._ik_solver is None and self._robot is not None and self._mojo is not None:
            self._initialize_ik_solver()
        
    def get_current_ee_poses(self) -> tuple[Pose, Pose]:
        """Get current end-effector poses."""
        left_site = self._robot._wrist_sites[HandSide.LEFT]
        right_site = self._robot._wrist_sites[HandSide.RIGHT]
        
        left_pos = left_site.get_position()
        left_quat = Quaternion(left_site.get_quaternion())
        
        right_pos = right_site.get_position()
        right_quat = Quaternion(right_site.get_quaternion())
        
        return Pose(left_pos, left_quat), Pose(right_pos, right_quat)
        
    def poses_to_action(self, left_pose: Pose, right_pose: Pose, 
                       base_action: Optional[np.ndarray] = None, 
                       gripper_action: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert end-effector poses to Cartesian action."""
        action_parts = []
        
        # Left end-effector
        action_parts.extend(left_pose.position)
        action_parts.extend(rotation_matrix_to_6d(left_pose.orientation.rotation_matrix))
        
        # Right end-effector
        action_parts.extend(right_pose.position)
        action_parts.extend(rotation_matrix_to_6d(right_pose.orientation.rotation_matrix))
        
        # Base action
        if self.floating_base:
            if base_action is not None:
                action_parts.extend(base_action)
            else:
                base_dof = self._robot.floating_base.dof_amount if self._robot else 4
                action_parts.extend([0.0] * base_dof)
                
        # Gripper action
        if gripper_action is not None:
            action_parts.extend(gripper_action)
        else:
            action_parts.extend([0.0, 0.0])
            
        return np.array(action_parts)