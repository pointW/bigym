"""RBY1 Cartesian space action mode."""
from __future__ import annotations

from typing import Optional
import numpy as np
from gymnasium import spaces
from pyquaternion import Quaternion
import mujoco

from bigym.action_modes import ActionMode
from bigym.const import HandSide
from bigym.ik.rby1_ik import RBY1IK
from vr.ik.h1_upper_body_ik import Pose


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
    x = row1 / np.linalg.norm(row1)
    
    # Make second row orthogonal to first and normalize
    y = row2 - np.dot(row2, x) * x
    y = y / np.linalg.norm(y)
    
    # Third row is cross product
    z = np.cross(x, y)
    
    # Stack rows to form matrix
    return np.vstack([x, y, z])


class RBY1CartesianActionMode(ActionMode):
    """Control RBY1 arms through end-effector poses in Cartesian space.
    
    The action space consists of:
    - Left end-effector position (3D): absolute position in world coordinates
    - Left end-effector orientation (6D): 6D rotation representation 
    - Right end-effector position (3D): absolute position in world coordinates
    - Right end-effector orientation (6D): 6D rotation representation
    - Base control (3D): X, Y, RZ for wheeled base
    - Gripper control (2D): left and right gripper commands
    
    This action mode:
    1. First moves the base_target mocap body to control base position
    2. Then solves IK for the desired end-effector poses
    3. Finally applies joint positions and gripper controls
    
    Uses RBY1IK to convert Cartesian poses to joint positions.
    """
    
    def __init__(
        self,
        floating_base: bool = True,
        position_limits: tuple[float, float] = (-2.0, 2.0),
        block_until_reached: bool = False,
    ):
        """Initialize RBY1 Cartesian action mode.
        
        Args:
            floating_base: Must be True for RBY1 (wheeled robot always controls base).
                           Raises error if False.
            position_limits: Min/max limits for end-effector positions
            block_until_reached: Whether to block until position is reached
        """
        # RBY1 is a wheeled robot - must always control the base
        if not floating_base:
            raise ValueError(
                "RBY1CartesianActionMode requires floating_base=True since RBY1 is a wheeled robot. "
                "The base must always be controlled."
            )
        
        # Initialize parent with no floating DOFs (we handle base control manually)
        super().__init__(floating_base=False, floating_dofs=None)
        
        self.position_limits = position_limits
        self.block_until_reached = block_until_reached
        self._ik_solver = None
        self._base_target_body_id = None
        
    def bind_robot(self, robot, mojo):
        """Bind action mode to robot."""
        super().bind_robot(robot, mojo)
        # IK solver will be initialized when first needed
        self._ik_solver = None
        # Base target body ID will be set later when needed
        self._base_target_body_id = None
        
    def action_space(self, action_scale: float, seed: Optional[int] = None) -> spaces.Box:
        """Create action space for Cartesian control.
        
        Action space (23D total):
        - Left EE position (3D): [x, y, z] 
        - Left EE orientation (6D): 6D rotation representation
        - Right EE position (3D): [x, y, z]
        - Right EE orientation (6D): 6D rotation representation  
        - Base control (3D): X, Y, RZ for wheeled base
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
        
        # Base control (3D): X, Y, RZ
        # Position bounds for X, Y
        base_pos_bounds = [(-2.0, 2.0), (-2.0, 2.0)]  # X, Y limits
        base_rot_bounds = [(-np.pi, np.pi)]  # RZ limits
        bounds.extend(base_pos_bounds + base_rot_bounds)
        
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
        """Execute Cartesian action by controlling base, solving IK, and applying controls.
        
        Steps:
        1. Move base_target mocap body to desired base position
        2. Update robot base position in IK solver
        3. Solve IK for desired end-effector poses
        4. Apply joint positions from IK solution
        5. Control grippers
        
        Args:
            action: Cartesian action vector (23D)
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
                raise RuntimeError("base_target mocap body not found in model")
            
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
        
        # Base control (X, Y, RZ)
        base_x = action[idx]
        base_y = action[idx+1]
        base_rz = action[idx+2]
        idx += 3
        
        # Gripper control
        gripper_action = action[idx:]
        
        # Step 1: Move base_target mocap body to desired position
        # The base_target is connected to the robot base via weld constraint
        data = self._mojo.physics.data._data
        model = self._mojo.physics.model._model
        
        # Set mocap body position and orientation
        mocap_id = model.body_mocapid[self._base_target_body_id]
        if mocap_id >= 0:
            # Set position
            data.mocap_pos[mocap_id][0] = base_x
            data.mocap_pos[mocap_id][1] = base_y
            data.mocap_pos[mocap_id][2] = 0.0  # Keep Z at ground level
            
            # Set orientation (quaternion from RZ rotation)
            # Convert RZ angle to quaternion (rotation around Z axis)
            qw = np.cos(base_rz / 2)
            qx = 0
            qy = 0
            qz = np.sin(base_rz / 2)
            data.mocap_quat[mocap_id] = [qw, qx, qy, qz]
        
        # Step 2: Forward kinematics to update positions after base movement
        mujoco.mj_forward(model, data)
        
        # Get updated base position and orientation after mocap movement
        base_body = self._robot.pelvis  # This maps to the "base" body
        base_pos_current = base_body.get_position()
        base_quat_obj = Quaternion(base_body.get_quaternion())
        base_quat_current = np.array([base_quat_obj.w, base_quat_obj.x, base_quat_obj.y, base_quat_obj.z])
        
        # Convert target quaternions to numpy arrays in wxyz format
        left_quat_np = np.array([left_quat.w, left_quat.x, left_quat.y, left_quat.z])
        right_quat_np = np.array([right_quat.w, right_quat.x, right_quat.y, right_quat.z])
        
        # Get current qpos for IK initialization
        current_qpos = self._mojo.physics.data.qpos.copy()
        
        # Step 3: Solve IK for target poses with updated base position
        ik_solution, success, info = self._ik_solver.solve(
            base_pos=base_pos_current,
            base_quat=base_quat_current,
            left_target_pos=left_pos,
            left_target_quat=left_quat_np,
            right_target_pos=right_pos,
            right_target_quat=right_quat_np,
            current_qpos=current_qpos,
        )
        
        if not success:
            # IK failed, keep current positions but still control grippers
            for side, action in zip(self._robot.grippers, gripper_action):
                self._robot.grippers[side].set_control(action)
            self._mojo.step()
            return
        
        # Step 4: Apply joint positions from IK solution
        # IK solution contains full qpos, extract actuated joints
        # RBY1 qpos structure: [base_x, base_y, base_rz, wheel_joints(4), torso_joints(6), right_arm(7), left_arm(7)]
        
        # Skip base (3) and wheels (4) to get actuated joints
        actuated_start = 7  # After base(3) + wheels(4)
        
        # Extract torso and arm joints from IK solution
        torso_joints = ik_solution[actuated_start:actuated_start+6]
        right_arm_joints = ik_solution[actuated_start+6:actuated_start+13]
        left_arm_joints = ik_solution[actuated_start+13:actuated_start+20]
        
        # Apply joint positions to actuators
        joint_positions = np.concatenate([torso_joints, right_arm_joints, left_arm_joints])
        
        for i, actuator in enumerate(self._robot.limb_actuators):
            actuator_bound = self._mojo.physics.bind(actuator)
            actuator_bound.ctrl = joint_positions[i]
        
        # Step 5: Control grippers
        for side, action in zip(self._robot.grippers, gripper_action):
            self._robot.grippers[side].set_control(action)
        
        # Step the simulation
        if self.block_until_reached:
            # TODO: Implement blocking behavior if needed
            self._mojo.step()
        else:
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
                
                bound_actuator = self._mojo.physics.bind(actuator)
                bound_actuator.ctrl = value
        
        # Initialize IK solver after reset to capture correct initial state
        if self._ik_solver is None and self._robot is not None and self._mojo is not None:
            self._initialize_ik_solver()
        
    def _initialize_ik_solver(self):
        """Initialize the RBY1 IK solver."""
        # Get MuJoCo model and data from mojo
        model = self._mojo.physics.model._model
        data = self._mojo.physics.data._data
        
        # Create the RBY1 IK solver
        self._ik_solver = RBY1IK(model, data)
    
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
        base_x: float = 0.0,
        base_y: float = 0.0,
        base_rz: float = 0.0,
        gripper_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert end-effector poses and base position to Cartesian action.
        
        Args:
            left_pose: Left end-effector pose
            right_pose: Right end-effector pose  
            base_x: Base X position
            base_y: Base Y position
            base_rz: Base rotation around Z
            gripper_action: Gripper control action
            
        Returns:
            Cartesian action vector (23D)
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
        
        # Base control
        action_parts.extend([base_x, base_y, base_rz])
        
        # Gripper control
        if gripper_action is None:
            gripper_action = np.zeros(len(self._robot.grippers))
        action_parts.extend(gripper_action)
        
        return np.array(action_parts, dtype=np.float32)