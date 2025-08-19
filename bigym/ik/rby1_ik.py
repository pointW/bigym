"""RBY1 IK solver using Mink library."""
import numpy as np
from typing import Optional, Tuple, Dict
import mujoco

try:
    import mink
    from mink import SE3
    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False
    print("Warning: Mink library not available. RBY1 IK solver will not work.")


class RBY1IK:
    """IK solver for RBY1 robot using Mink optimization library.
    
    This solver handles:
    - 6 DOF torso chain
    - Dual 7 DOF arms
    
    Note: Base motion (X, Y, theta) is controlled separately by the action mode,
    not by the IK solver. The IK solver only handles torso and arm joints.
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Initialize RBY1 IK solver.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        if not MINK_AVAILABLE:
            raise ImportError("Mink library is required for RBY1 IK solver")
        
        self.model = model
        self.data = data
        
        # Store joint indices for different parts
        self._setup_joint_indices()
    
    def _setup_joint_indices(self):
        """Setup joint indices for different robot parts."""
        # Get joint IDs and qpos indices
        
        # Base joint (for reference, but not modified by IK)
        self.base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_j")
        self.base_qpos_start = 0  # X, Y, Z positions (0-2) and quaternion (3-6)
        
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
        
        # Create list of IK-controlled joint indices
        self.ik_controlled_indices = (
            self.torso_qpos_indices + 
            self.left_arm_qpos_indices + 
            self.right_arm_qpos_indices
        )
    
    def solve(
        self,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        left_target_pos: Optional[np.ndarray] = None,
        left_target_quat: Optional[np.ndarray] = None,
        right_target_pos: Optional[np.ndarray] = None,
        right_target_quat: Optional[np.ndarray] = None,
        current_qpos: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """Solve IK for given targets.
        
        Args:
            base_pos: Base position (3D: X, Y, Z) - not modified by IK
            base_quat: Base orientation quaternion (wxyz) - not modified by IK
            left_target_pos: Left end effector target position (3D)
            left_target_quat: Left end effector target orientation (quaternion wxyz)
            right_target_pos: Right end effector target position (3D)
            right_target_quat: Right end effector target orientation (quaternion wxyz)
            current_qpos: Current joint positions (if None, uses data.qpos)
            max_iterations: Maximum IK iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (solution_qpos, success, info_dict)
            Note: solution_qpos contains full qpos with base unchanged
        """
        # Use current configuration if not provided
        if current_qpos is None:
            current_qpos = self.data.qpos.copy()
        
        # Set base position and orientation (not modified by IK)
        current_qpos[0:3] = base_pos
        current_qpos[3:7] = base_quat / np.linalg.norm(base_quat)  # Normalize quaternion
        
        # Update MuJoCo data with initial configuration
        self.data.qpos[:] = current_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # Create configuration from current state
        configuration = mink.Configuration(self.model, current_qpos.copy())
        
        # Create task list
        tasks = []
        
        # Add base pose task to constrain base movement
        # This is more effective than high posture cost alone
        # Check if we're in an environment context (bodies have namespace)
        base_name = "rby1/base" if any("rby1/" in self.model.body(i).name for i in range(self.model.nbody)) else "base"
        base_task = mink.FrameTask(
            frame_name=base_name,  # RBY1 base body (with or without namespace)
            frame_type="body",
            position_cost=10000.0,  # Very high cost to keep base fixed
            orientation_cost=10000.0,  # Very high cost to keep orientation fixed
            lm_damping=1e-6,
        )
        # Set target to current base pose
        base_matrix = np.eye(4)
        base_matrix[:3, 3] = base_pos
        # Convert quaternion to rotation matrix
        w, x, y, z = base_quat
        base_matrix[:3, :3] = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
        base_task.set_target(mink.SE3.from_matrix(base_matrix))
        tasks.append(base_task)
        
        # Add posture task for joint stability
        posture_task = mink.PostureTask(
            model=self.model,
            cost=1.0  # Low cost, just for redundancy resolution
        )
        posture_task.set_target(current_qpos)
        tasks.append(posture_task)
        
        # Add limits for IK-controlled joints (skip base and wheels)
        # Create a mask for IK-controlled joints
        joint_mask = np.zeros(self.model.nv, dtype=bool)
        for idx in self.ik_controlled_indices:
            if idx < len(joint_mask):
                joint_mask[idx] = True
        
        # Add left arm task if target provided
        if left_target_pos is not None:
            # Check for namespace
            left_ee_name = "rby1/end_effector_l" if "rby1/" in base_name else "end_effector_l"
            left_ee_task = mink.FrameTask(
                frame_name=left_ee_name,
                frame_type="site",
                position_cost=10000.0,  # Extremely high cost for sub-mm accuracy
                orientation_cost=1000.0 if left_target_quat is not None else 0.0,
                lm_damping=1e-5,  # Minimal damping for best accuracy
            )
            
            if left_target_quat is not None:
                # Full pose target
                target_matrix = self._pose_to_matrix(left_target_pos, left_target_quat)
            else:
                # Position only - use identity rotation
                target_matrix = self._pose_to_matrix(left_target_pos, np.array([1, 0, 0, 0]))
            
            left_ee_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(left_ee_task)
        
        # Add right arm task if target provided
        if right_target_pos is not None:
            # Check for namespace
            right_ee_name = "rby1/end_effector_r" if "rby1/" in base_name else "end_effector_r"
            right_ee_task = mink.FrameTask(
                frame_name=right_ee_name,
                frame_type="site",
                position_cost=10000.0,  # Extremely high cost for sub-mm accuracy
                orientation_cost=1000.0 if right_target_quat is not None else 0.0,
                lm_damping=1e-5,  # Minimal damping for best accuracy
            )
            
            if right_target_quat is not None:
                # Full pose target
                target_matrix = self._pose_to_matrix(right_target_pos, right_target_quat)
            else:
                # Position only - use identity rotation
                target_matrix = self._pose_to_matrix(right_target_pos, np.array([1, 0, 0, 0]))
            
            right_ee_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(right_ee_task)
        
        # Solver parameters
        dt = 0.1  # Larger timestep for faster convergence
        solver = "daqp"
        damping = 1e-6  # Extremely low damping for best accuracy
        
        # IK solving loop
        success = False
        for iteration in range(max_iterations):
            # Compute velocity
            vel = mink.solve_ik(configuration, tasks, dt, solver, damping)
            
            # Base is controlled via mocap body - zero base velocities
            # The base movement is constrained by the base_task with high cost
            vel[0:6] = 0  # Base velocities (controlled via mocap)
            
            # Integrate velocity
            configuration.integrate_inplace(vel, dt)
            
            # Update MuJoCo data with new configuration
            self.data.qpos[:] = configuration.q
            mujoco.mj_forward(self.model, self.data)
            
            # Check convergence using actual position difference
            converged = True
            errors = {}
            
            if left_target_pos is not None:
                # Get actual end effector position
                current_left_pos = self._get_site_position(left_ee_name, configuration.q)
                pos_error_left = np.linalg.norm(current_left_pos - left_target_pos)
                errors["left_position_error"] = pos_error_left
                if pos_error_left > tolerance:
                    converged = False
            
            if right_target_pos is not None:
                # Get actual end effector position
                current_right_pos = self._get_site_position(right_ee_name, configuration.q)
                pos_error_right = np.linalg.norm(current_right_pos - right_target_pos)
                errors["right_position_error"] = pos_error_right
                if pos_error_right > tolerance:
                    converged = False
            
            # Check if velocity is small (additional convergence criterion)
            vel_norm = np.linalg.norm(vel)
            if vel_norm < 1e-6 and iteration > 10:  # Stuck, not making progress
                break
            
            if converged:
                success = True
                break
        
        # Get solution
        solution_qpos = configuration.q.copy()
        
        # Final check of actual errors
        final_errors = {}
        if left_target_pos is not None:
            current_left_pos = self._get_site_position(left_ee_name, solution_qpos)
            final_errors["left_position_error"] = np.linalg.norm(current_left_pos - left_target_pos)
        if right_target_pos is not None:
            current_right_pos = self._get_site_position(right_ee_name, solution_qpos)
            final_errors["right_position_error"] = np.linalg.norm(current_right_pos - right_target_pos)
        
        info = {
            "errors": final_errors,
            "iterations": iteration + 1,
            "success": success,
        }
        
        return solution_qpos, success, info
    
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
    
    def _euler_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion.
        
        Args:
            roll: Roll angle (rotation around X)
            pitch: Pitch angle (rotation around Y)
            yaw: Yaw angle (rotation around Z)
            
        Returns:
            Quaternion [w, x, y, z]
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])