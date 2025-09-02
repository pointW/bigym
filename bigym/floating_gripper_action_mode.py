"""Floating Gripper Action Mode - Perfect tracking via mocap bodies."""
from __future__ import annotations
from typing import Optional
import numpy as np
from gymnasium import spaces
from pyquaternion import Quaternion

from bigym.action_modes import ActionMode
from bigym.const import HandSide


def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D rotation representation.
    
    The 6D representation consists of the first two rows of the rotation matrix.
    This representation is continuous and more suitable for learning.
    """
    return rotation_matrix[:2, :].flatten()


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation back to rotation matrix.
    
    Reconstructs the full 3x3 rotation matrix from the 6D representation.
    """
    row1 = rotation_6d[:3]
    row2 = rotation_6d[3:6]
    
    # Normalize first row
    x = row1 / np.linalg.norm(row1)
    
    # Make second row orthogonal to first and normalize
    y = row2 - np.dot(row2, x) * x
    y = y / np.linalg.norm(y)
    
    # Third row is cross product
    z = np.cross(x, y)
    
    return np.vstack([x, y, z])


class FloatingGripperActionMode(ActionMode):
    """Action mode that directly sets gripper poses with zero tracking error.
    
    This action mode uses mocap bodies to achieve perfect end-effector tracking,
    useful for verifying if demonstrations are feasible with perfect control.
    
    Action space (20D):
    - Left gripper position (3D): absolute position in world coordinates
    - Left gripper orientation (6D): 6D rotation representation
    - Right gripper position (3D): absolute position in world coordinates
    - Right gripper orientation (6D): 6D rotation representation
    - Gripper control (2D): [-1, 1] for each gripper
    """
    
    def __init__(self, control_frequency=50):
        """Initialize floating gripper action mode."""
        super().__init__(floating_base=False)
        self._left_mocap_idx = None
        self._right_mocap_idx = None
        self._mojo = None
        self._robot = None
        self._initialized = False
        self.absolute = True  # This is an absolute action mode
        self.control_frequency = control_frequency
        self.low_level_frequency = 500
    
    def action_space(self, action_scale: float, seed: Optional[int] = None) -> spaces.Box:
        """Return the action space.
        
        Args:
            action_scale: Scaling factor for actions (not used for mocap)
            seed: Random seed (not used)
            
        Returns:
            Gymnasium Box space with 20 dimensions
        """
        return self.get_action_space()
    
    def step(self, action: np.ndarray):
        """Apply action (alias for apply method)."""
        return self.apply(action)
    
    def initialize(self, robot, mojo):
        """Initialize with robot and mojo references.
        
        Args:
            robot: Robot instance with grippers
            mojo: Mojo physics instance
        """
        self._robot = robot
        self._mojo = mojo
        
        # Find mocap bodies for grippers
        physics = self._mojo.physics
        if physics is None:
            # Physics not ready yet, will be initialized later
            self._initialized = False
            return
        
        model = physics.model
        
        # Look for mocap bodies by name
        # Mocap bodies have a special index in MuJoCo
        self._left_mocap_idx = None
        self._right_mocap_idx = None
        
        # Find which mocap index corresponds to each gripper
        mocap_bodies = []
        for body_id in range(model.nbody):
            body_name = model.id2name(body_id, 'body')
            if body_name and 'gripper_mocap' in body_name:
                # Check if this body is a mocap body
                if model.body_mocapid[body_id] >= 0:
                    mocap_idx = model.body_mocapid[body_id]
                    if 'left' in body_name:
                        self._left_mocap_idx = mocap_idx
                    elif 'right' in body_name:
                        self._right_mocap_idx = mocap_idx
        
        if self._left_mocap_idx is None or self._right_mocap_idx is None:
            raise ValueError(
                f"Mocap bodies for grippers not found in model. "
                f"Found left={self._left_mocap_idx}, right={self._right_mocap_idx}"
            )
        
        self._initialized = True
    
    def get_action_space(self) -> spaces.Box:
        """Get action space for floating grippers.
        
        Returns:
            Gymnasium Box space with 20 dimensions
        """
        # Position bounds: reasonable workspace
        pos_low = np.array([-1.0, -1.0, 0.0])
        pos_high = np.array([2.0, 1.0, 2.0])
        
        # Rotation bounds: normalized 6D representation
        rot_low = np.array([-1.0] * 6)
        rot_high = np.array([1.0] * 6)
        
        # Gripper bounds
        gripper_low = np.array([-1.0, -1.0])
        gripper_high = np.array([1.0, 1.0])
        
        # Combine all bounds
        low = np.concatenate([pos_low, rot_low, pos_low, rot_low, gripper_low])
        high = np.concatenate([pos_high, rot_high, pos_high, rot_high, gripper_high])
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def action_to_target_state(self, action: np.ndarray, current_state=None):
        """Convert action to target state.
        
        For mocap control, the action directly represents the target state.
        
        Args:
            action: 20D action vector
            current_state: Not used for absolute mocap control
            
        Returns:
            Target state (same as action for mocap)
        """
        return action
    
    def apply(self, action: np.ndarray):
        """Apply action by directly setting mocap body poses.
        
        This achieves perfect tracking with zero error using linear interpolation
        for smooth motion between control steps.
        
        Args:
            action: 20D action vector
        """
        if not self._initialized:
            # Try lazy initialization
            if self._robot and self._mojo and self._mojo.physics:
                self.initialize(self._robot, self._mojo)
            
            if not self._initialized:
                raise RuntimeError("FloatingGripperActionMode not initialized")
        
        # Parse action
        left_pos = action[0:3]
        left_ori_6d = action[3:9]
        right_pos = action[9:12]
        right_ori_6d = action[12:18]
        gripper_control = action[18:20]
        
        # Convert 6D rotation to quaternion for target
        left_rot_matrix = rotation_6d_to_matrix(left_ori_6d)
        right_rot_matrix = rotation_6d_to_matrix(right_ori_6d)
        
        left_quat_target = Quaternion(matrix=left_rot_matrix)
        right_quat_target = Quaternion(matrix=right_rot_matrix)
        
        physics = self._mojo.physics
        
        # Get current mocap poses for interpolation
        left_pos_current = physics.data.mocap_pos[self._left_mocap_idx].copy()
        left_quat_current = Quaternion(physics.data.mocap_quat[self._left_mocap_idx])
        
        right_pos_current = physics.data.mocap_pos[self._right_mocap_idx].copy()
        right_quat_current = Quaternion(physics.data.mocap_quat[self._right_mocap_idx])
        
        # Control grippers (set once at the beginning)
        if hasattr(self._robot, 'grippers') and self._robot.grippers:
            for side, ctrl in zip(self._robot.grippers, gripper_control):
                self._robot.grippers[side].set_control(ctrl)
        
        # Linear interpolation over multiple physics steps
        # This provides smooth motion and better gripper control
        num_steps = int(self.low_level_frequency // self.control_frequency)
        
        for i in range(num_steps):
            # Calculate interpolation factor (0 to 1)
            alpha = (i + 1) / num_steps
            
            # Interpolate positions (linear)
            left_pos_interp = (1 - alpha) * left_pos_current + alpha * left_pos
            right_pos_interp = (1 - alpha) * right_pos_current + alpha * right_pos
            
            # Interpolate quaternions (spherical linear interpolation)
            left_quat_interp = Quaternion.slerp(left_quat_current, left_quat_target, alpha)
            right_quat_interp = Quaternion.slerp(right_quat_current, right_quat_target, alpha)
            
            # Set interpolated mocap poses
            physics.data.mocap_pos[self._left_mocap_idx] = left_pos_interp
            physics.data.mocap_quat[self._left_mocap_idx] = [
                left_quat_interp.w, left_quat_interp.x, left_quat_interp.y, left_quat_interp.z
            ]
            
            physics.data.mocap_pos[self._right_mocap_idx] = right_pos_interp
            physics.data.mocap_quat[self._right_mocap_idx] = [
                right_quat_interp.w, right_quat_interp.x, right_quat_interp.y, right_quat_interp.z
            ]
            
            # Step physics
            self._mojo.step()
    
    def reset(self, reset_state: np.ndarray):
        """Reset action mode.
        
        For mocap control, reset doesn't require special handling.
        
        Args:
            reset_state: Reset state (not used for mocap)
        """
        pass  # Mocap bodies will be set on first action
    
    def get_current_poses(self):
        """Get current gripper poses.
        
        Returns:
            Tuple of (left_pos, left_quat, right_pos, right_quat)
        """
        if not self._initialized:
            return None, None, None, None
        
        physics = self._mojo.physics
        
        left_pos = physics.data.mocap_pos[self._left_mocap_idx].copy()
        left_quat = physics.data.mocap_quat[self._left_mocap_idx].copy()
        
        right_pos = physics.data.mocap_pos[self._right_mocap_idx].copy()
        right_quat = physics.data.mocap_quat[self._right_mocap_idx].copy()
        
        return left_pos, left_quat, right_pos, right_quat