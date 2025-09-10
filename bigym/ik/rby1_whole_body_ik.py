"""RBY1 Whole-Body IK solver using Mink library.

This solver optimizes both base movement and joint positions to reach end-effector targets,
while maintaining stability and upright posture constraints.
"""
import numpy as np
from typing import Optional, Tuple, Dict
import mujoco

try:
    import mink
    from mink import SE3
    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False
    print("Warning: Mink library not available. RBY1 whole-body IK solver will not work.")


class RBY1WholeBodyIK:
    """Whole-body IK solver for RBY1 robot using Mink optimization library.
    
    This solver handles:
    - Base movement (X, Y, theta) - optimized together with joints
    - 6 DOF torso chain
    - Dual 7 DOF arms
    
    Optimization priorities:
    1. End-effector target positions and orientations (highest priority)
    2. Base Z stays on ground (hard constraint)
    3. Upper body upright orientation (weak regularization)
    4. COM stability within base support polygon (medium regularization)
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Initialize RBY1 whole-body IK solver.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        if not MINK_AVAILABLE:
            raise ImportError("Mink library is required for RBY1 whole-body IK solver")
        
        self.model = model
        self.data = data
        
        # Store joint indices for different parts
        self._setup_joint_indices()
        
        # Check if we're in environment context (with namespace)
        self.has_namespace = any("rby1/" in self.model.body(i).name for i in range(self.model.nbody))
        
        # Set body/site names based on namespace
        if self.has_namespace:
            self.base_name = "rby1/base"
            self.torso5_name = "rby1/link_torso_5"
            self.left_ee_name = "rby1/end_effector_l"
            self.right_ee_name = "rby1/end_effector_r"
            # Wheel link names for stability check
            self.wheel_names = ["rby1/link_wheel_fr", "rby1/link_wheel_fl", 
                               "rby1/link_wheel_rr", "rby1/link_wheel_rl"]
        else:
            self.base_name = "base"
            self.torso5_name = "link_torso_5"
            self.left_ee_name = "end_effector_l"
            self.right_ee_name = "end_effector_r"
            # Wheel link names for stability check
            self.wheel_names = ["link_wheel_fr", "link_wheel_fl", 
                               "link_wheel_rr", "link_wheel_rl"]
    
    def _setup_joint_indices(self):
        """Setup joint indices for different robot parts."""
        # Base joint (now controlled by IK)
        self.base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_j")
        self.base_qpos_indices = [0, 1, 2]  # X, Y, Z positions
        self.base_quat_indices = [3, 4, 5, 6]  # Quaternion (w, x, y, z)
        
        # Wheel joints (not modified by IK)
        self.wheel_joint_names = ["wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl"]
        self.wheel_qpos_indices = []
        for name in self.wheel_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.wheel_qpos_indices.append(qpos_adr)
        
        # Torso joints (controlled by IK)
        self.torso_joint_names = [f"torso_{i}" for i in range(6)]
        self.torso_qpos_indices = []
        for name in self.torso_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.torso_qpos_indices.append(qpos_adr)
        
        # Left arm joints (controlled by IK)
        self.left_arm_joint_names = [f"left_arm_{i}" for i in range(7)]
        self.left_arm_qpos_indices = []
        for name in self.left_arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.left_arm_qpos_indices.append(qpos_adr)
        
        # Right arm joints (controlled by IK)
        self.right_arm_joint_names = [f"right_arm_{i}" for i in range(7)]
        self.right_arm_qpos_indices = []
        for name in self.right_arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.right_arm_qpos_indices.append(qpos_adr)
        
        # All IK-controlled indices (including base now)
        self.ik_controlled_indices = (
            self.base_qpos_indices +  # Base X, Y, Z
            self.torso_qpos_indices + 
            self.left_arm_qpos_indices + 
            self.right_arm_qpos_indices
        )
    
    def solve(
        self,
        left_target_pos: Optional[np.ndarray] = None,
        left_target_quat: Optional[np.ndarray] = None,
        right_target_pos: Optional[np.ndarray] = None,
        right_target_quat: Optional[np.ndarray] = None,
        current_qpos: Optional[np.ndarray] = None,
        max_iterations: int = 100,  # Not used in current implementation but kept for API compatibility
        tolerance: float = 1e-3,  # Not used in current implementation but kept for API compatibility
        left_body_relative: bool = False,
        right_body_relative: bool = False,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """Solve whole-body IK for given end-effector targets.
        
        The solver optimizes base position (X, Y, theta) along with joint positions
        to reach the targets while maintaining stability constraints.
        
        Args:
            left_target_pos: Left end effector target position (3D)
            left_target_quat: Left end effector target orientation (quaternion wxyz)
            right_target_pos: Right end effector target position (3D)
            right_target_quat: Right end effector target orientation (quaternion wxyz)
            current_qpos: Current joint positions (if None, uses data.qpos)
            max_iterations: Maximum IK iterations
            tolerance: Convergence tolerance
            left_body_relative: If True, left target is relative to body frame
            right_body_relative: If True, right target is relative to body frame
            
        Returns:
            Tuple of (solution_qpos, success, info_dict)
            Note: solution_qpos contains optimized base position and joint positions
        """
        # Use current configuration if not provided
        if current_qpos is None:
            current_qpos = self.data.qpos.copy()
        
        # Update MuJoCo data with initial configuration
        self.data.qpos[:] = current_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial base pose for body-relative constraints
        initial_base_pos = current_qpos[:3].copy()  # [x, y, z]
        initial_base_quat = current_qpos[3:7].copy()  # [w, x, y, z]
        
        # Store initial EE positions in body frame if needed (currently not used in simplified solver)
        # if left_body_relative and left_target_pos is not None:
        #     # For body-relative constraints, we want to maintain the hand's position
        #     # relative to the body. Since we're dealing with a wheeled robot that 
        #     # primarily moves in X-Y and rotates around Z, we'll use a simplified
        #     # approach: maintain the offset from the base in world frame.
        #     # This works well for small rotations.
        #     left_body_offset = left_target_pos - initial_base_pos
        #     
        # if right_body_relative and right_target_pos is not None:
        #     # Same for right hand
        #     right_body_offset = right_target_pos - initial_base_pos
        
        # Create configuration from current state
        configuration = mink.Configuration(self.model, current_qpos.copy())
        
        # Create task list
        tasks = []
        
        # 1. End-effector tasks (highest priority)
        if left_target_pos is not None:
            left_ee_task = mink.FrameTask(
                frame_name=self.left_ee_name,
                frame_type="site",
                position_cost=10000.0,  # Highest priority
                orientation_cost=10000.0 if left_target_quat is not None else 0.0,
                lm_damping=1e-5,
            )
            
            if left_target_quat is not None:
                target_matrix = self._pose_to_matrix(left_target_pos, left_target_quat)
            else:
                target_matrix = self._pose_to_matrix(left_target_pos, np.array([1, 0, 0, 0]))
            
            left_ee_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(left_ee_task)
        
        if right_target_pos is not None:
            right_ee_task = mink.FrameTask(
                frame_name=self.right_ee_name,
                frame_type="site",
                position_cost=10000.0,  # Highest priority
                orientation_cost=10000.0 if right_target_quat is not None else 0.0,
                lm_damping=1e-5,
            )
            
            if right_target_quat is not None:
                target_matrix = self._pose_to_matrix(right_target_pos, right_target_quat)
            else:
                target_matrix = self._pose_to_matrix(right_target_pos, np.array([1, 0, 0, 0]))
            
            right_ee_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(right_ee_task)
        
        # 2. Base ground constraint (very high priority - base must stay on ground)
        # Constrain base Z position to 0 and only allow yaw rotation
        base_ground_task = mink.FrameTask(
            frame_name=self.base_name,
            frame_type="body",
            position_cost=[100.0, 100.0, 100000.0],  # Allow X,Y movement, strongly constrain Z
            orientation_cost=[100000.0, 100000.0, 100.0],  # Constrain roll/pitch, allow yaw
            lm_damping=1e-6,
        )
        # Set target to current X,Y but Z=0 and upright orientation with current yaw
        base_target_matrix = np.eye(4)
        base_target_matrix[0, 3] = current_qpos[0]  # Current X
        base_target_matrix[1, 3] = current_qpos[1]  # Current Y  
        base_target_matrix[2, 3] = 0.0  # Z must be 0 (ground)
        # Extract yaw from current quaternion and create upright rotation with that yaw
        current_quat = current_qpos[3:7]
        yaw = np.arctan2(2*(current_quat[0]*current_quat[3] + current_quat[1]*current_quat[2]),
                         1 - 2*(current_quat[2]**2 + current_quat[3]**2))
        # Create rotation matrix for yaw-only rotation
        c_yaw = np.cos(yaw)
        s_yaw = np.sin(yaw)
        base_target_matrix[:3, :3] = np.array([
            [c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]
        ])
        base_ground_task.set_target(mink.SE3.from_matrix(base_target_matrix))
        tasks.append(base_ground_task)
        
        # 3. Upper body upright orientation (STRONG constraint for stability)
        # Constrain torso_5 link to point upward - CRITICAL for preventing falls
        torso_upright_task = mink.FrameTask(
            frame_name=self.torso5_name,
            frame_type="body",
            position_cost=0.0,  # Don't constrain position
            orientation_cost=1000.0,  # STRONG constraint to maintain upright posture
            lm_damping=1e-4,
        )
        # Set target to upright orientation (identity rotation)
        upright_matrix = np.eye(4)
        upright_matrix[:3, 3] = [0, 0, 1.0]  # Dummy position (not used due to position_cost=0)
        torso_upright_task.set_target(mink.SE3.from_matrix(upright_matrix))
        tasks.append(torso_upright_task)
        
        # 4. COM stability constraint (medium regularization)
        # This is approximated by keeping torso_5 position within base support polygon
        # We use a relative position task between torso and base
        com_stability_task = mink.RelativeFrameTask(
            frame_name=self.torso5_name,
            frame_type="body",
            root_name=self.base_name,
            root_type="body",
            position_cost=100.0,  # Medium cost for stability
            orientation_cost=0.0,  # Don't constrain relative orientation
            lm_damping=1e-4,
        )
        # Target: torso should be above base center with some tolerance
        relative_matrix = np.eye(4)
        relative_matrix[:3, 3] = [0, 0, 0.8]  # Torso approximately 0.8m above base
        com_stability_task.set_target(mink.SE3.from_matrix(relative_matrix))
        tasks.append(com_stability_task)
        
        # # 4b. Torso height relative to grippers task
        # # Encourage torso to be 10cm higher than average gripper height
        # # Calculate average gripper z position
        # gripper_z_sum = left_target_pos[2] + right_target_pos[2]        
        # avg_gripper_z = gripper_z_sum / 2
        # target_torso_z = avg_gripper_z + 0.3  # 10cm higher
        
        # # Create a task for torso height
        # torso_height_task = mink.FrameTask(
        #     frame_name=self.torso5_name,
        #     frame_type="body",
        #     position_cost=[0.0, 0.0, 100.0],  # Only constrain Z with medium cost
        #     orientation_cost=0.0,  # Don't constrain orientation
        #     lm_damping=1e-4,
        # )
        # # Set target with desired Z height
        # torso_height_matrix = np.eye(4)
        # torso_height_matrix[2, 3] = target_torso_z
        # torso_height_task.set_target(mink.SE3.from_matrix(torso_height_matrix))
        # tasks.append(torso_height_task)

        # 5. Wheel joint constraints (wheels should not move - passive)
        # Create individual joint tasks for each wheel to keep them fixed
        # for wheel_name in self.wheel_joint_names:
        #     wheel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
        #     if wheel_joint_id >= 0:
        #         # Use a posture task with very high cost for this specific joint
        #         # This effectively locks the wheel in place
        #         wheel_task = mink.PostureTask(
        #             model=self.model,
        #             cost=100000.0,  # Very high cost to prevent wheel movement
        #         )
        #         wheel_task.set_target(current_qpos)
        #         # Note: This will affect all joints but with the high cost only on wheels
        #         # The effect on other joints is negligible compared to other tasks
        #         tasks.append(wheel_task)
        #         break  # One task is sufficient for all wheels due to implementation
        
        posture_task = mink.PostureTask(
            model=self.model,
            cost=100.0  
        )
        # Set reference posture
        reference_qpos = current_qpos.copy()   # torso_5: stay near 0
        posture_task.set_target(reference_qpos)
        tasks.append(posture_task)

        # 6. Torso movement penalty task
        # Define preferred ranges for torso joints (in radians)
        # torso_1: -10° to 45° = -0.175 to 0.785 rad
        # torso_2: -90° to 10° = -1.571 to 0.175 rad  
        # torso_3: -10° to 45° = -0.175 to 0.785 rad
        # torso_0, torso_4, torso_5: keep small range around 0

        posture_task = mink.PostureTask(
            model=self.model,
            cost=50.0  # Lower cost - mainly for arm redundancy resolution
        )
        # Set reference posture
        reference_qpos = current_qpos.copy()
        reference_qpos[11] = 0.0      # torso_0: stay near 0
        reference_qpos[12] = 0.305    # torso_1: middle of [-0.175, 0.785]
        reference_qpos[13] = -0.698   # torso_2: middle of [-1.571, 0.175]
        reference_qpos[14] = 0.305    # torso_3: middle of [-0.175, 0.785]
        reference_qpos[15] = 0.0      # torso_4: stay near 0
        reference_qpos[16] = 0.0      # torso_5: stay near 0
        posture_task.set_target(reference_qpos)
        tasks.append(posture_task)

        # limits = []
        # max_velocities = {
        #     'rby1/': 0.5
        # }
        # velocity_limit = mink.VelocityLimit(self.model, max_velocities)
        # limits.append(velocity_limit)
        
        # Solver parameters
        dt = 1e-3  # Integration timestep
        solver = "daqp"
        damping = 1e-6
        
        vel = mink.solve_ik(configuration, tasks, dt, solver, damping)
        configuration.integrate_inplace(vel, dt)
        
        # Get solution
        solution_qpos = configuration.q.copy()
        
        # Final error check
        final_errors = {}
        if left_target_pos is not None:
            current_left_pos = self._get_site_position(self.left_ee_name, solution_qpos)
            final_errors["left_position_error"] = np.linalg.norm(current_left_pos - left_target_pos)
        if right_target_pos is not None:
            current_right_pos = self._get_site_position(self.right_ee_name, solution_qpos)
            final_errors["right_position_error"] = np.linalg.norm(current_right_pos - right_target_pos)
        
        # Check stability (COM within support polygon)
        torso5_pos = self._get_body_position(self.torso5_name, solution_qpos)
        base_pos = solution_qpos[:3]
        relative_pos = torso5_pos - base_pos
        stability_margin = np.linalg.norm(relative_pos[:2])  # XY distance from base center
        
        info = {
            "errors": final_errors,
            "iterations": 0,
            "success": True,
            "base_position": solution_qpos[:3].copy(),
            "stability_margin": stability_margin,
        }
        
        # Restore original qpos to prevent direct joint position changes
        # The IK solution will be applied through actuators, not directly
        # self.data.qpos[:] = current_qpos
        # mujoco.mj_forward(self.model, self.data)
        
        return solution_qpos, True, info
    
    def _get_site_position(self, site_name: str, qpos: np.ndarray) -> np.ndarray:
        """Get site position for given joint configuration.
        
        Args:
            site_name: Name of the site
            qpos: Joint positions
            
        Returns:
            3D position of the site
        """
        # Temporarily set qpos and compute forward kinematics
        old_qpos = self.data.qpos.copy()
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = self.data.site_xpos[site_id].copy()
        
        # Restore original qpos
        self.data.qpos[:] = old_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return pos
    
    def _get_body_position(self, body_name: str, qpos: np.ndarray) -> np.ndarray:
        """Get body position for given joint configuration.
        
        Args:
            body_name: Name of the body
            qpos: Joint positions
            
        Returns:
            3D position of the body
        """
        # Temporarily set qpos and compute forward kinematics
        old_qpos = self.data.qpos.copy()
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        pos = self.data.xpos[body_id].copy()
        
        # Restore original qpos
        self.data.qpos[:] = old_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return pos
    
    def _pose_to_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix.
        
        Args:
            position: 3D position
            quaternion: Quaternion [w, x, y, z]
            
        Returns:
            4x4 transformation matrix
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self._quat_to_rotmat(quaternion)
        matrix[:3, 3] = position
        return matrix
    
    def _quat_to_rotmat(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R