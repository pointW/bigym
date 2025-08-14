"""Mink-based IK solver for H1 robot following the official example."""
import numpy as np
from pyquaternion import Quaternion

try:
    import mink
    import mujoco
    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False
    print("Warning: mink library not installed. Install with: pip install mink")


class Pose:
    """Simple pose class matching the original implementation."""
    def __init__(self, position: np.ndarray, orientation: Quaternion):
        self.position = position
        self.orientation = orientation


class MinkH1UpperBodyIK:
    """Mink-based IK solver for H1 robot upper body following official example.
    
    This implementation closely follows the mink arm_iiwa.py example,
    with proper configuration updates and MuJoCo synchronization.
    """
    
    def __init__(self, env):
        """Initialize the Mink IK solver.
        
        Args:
            env: Environment containing robot and mojo physics
        """
        if not MINK_AVAILABLE:
            raise ImportError("mink and mujoco libraries are required but not installed")
            
        self.env = env
        self.robot = env.robot
        self.mojo = env.mojo
        
        # Get the raw MuJoCo model and data
        self.model = self.mojo.physics.model.ptr if hasattr(self.mojo.physics.model, 'ptr') else self.mojo.physics.model._ptr
        self.data = self.mojo.physics.data.ptr if hasattr(self.mojo.physics.data, 'ptr') else self.mojo.physics.data._ptr
        
        # Get joint indices for arms
        self._setup_joint_mapping()
        
    def _setup_joint_mapping(self):
        """Setup mapping between arm actuators and configuration indices."""
        # Find the qpos indices for arm joints
        model = self.env.mojo.physics.model
        
        self.left_indices = []
        self.right_indices = []
        
        for i in range(5):
            # Left arm
            actuator = self.robot.limb_actuators[i]
            if actuator.joint:
                joint_name = actuator.joint.name
                for j in range(self.model.njnt):
                    if model.joint(j).name == joint_name or joint_name in model.joint(j).name:
                        qpos_idx = model.joint(j).qposadr[0]
                        self.left_indices.append(qpos_idx)
                        break
            
            # Right arm
            actuator = self.robot.limb_actuators[5 + i]
            if actuator.joint:
                joint_name = actuator.joint.name
                for j in range(self.model.njnt):
                    if model.joint(j).name == joint_name or joint_name in model.joint(j).name:
                        qpos_idx = model.joint(j).qposadr[0]
                        self.right_indices.append(qpos_idx)
                        break
        
    def solve(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        target_pose_left: Pose,
        target_pose_right: Pose,
    ) -> np.ndarray:
        """Solve IK for target end-effector poses.
        
        Args:
            pelvis_pose: Current pelvis pose (pelvis controlled separately)
            qpos_arm_left: Current left arm joint positions (5 DOF)
            qpos_arm_right: Current right arm joint positions (5 DOF)
            target_pose_left: Target pose for left end-effector
            target_pose_right: Target pose for right end-effector
            
        Returns:
            Array of joint positions [left_arm(5), right_arm(5)]
        """
        
        # Update MuJoCo data with current arm positions
        for i, idx in enumerate(self.left_indices):
            if i < len(qpos_arm_left) and idx < self.model.nq:
                self.data.qpos[idx] = qpos_arm_left[i]
        
        for i, idx in enumerate(self.right_indices):
            if i < len(qpos_arm_right) and idx < self.model.nq:
                self.data.qpos[idx] = qpos_arm_right[i]
        
        # Update forward kinematics
        mujoco.mj_fwdPosition(self.model, self.data)
        
        # Create configuration from current state
        configuration = mink.Configuration(self.model, self.data.qpos.copy())
        
        # Setup tasks following arm_iiwa.py example
        left_hand_task = mink.FrameTask(
            frame_name="h1/left_end_effector",
            frame_type="site",
            position_cost=1.0,  # From arm example
            orientation_cost=0.0,  # No orientation for hands
            lm_damping=1.0,
        )
        
        right_hand_task = mink.FrameTask(
            frame_name="h1/right_end_effector",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        
        # Posture task with low cost
        posture_task = mink.PostureTask(
            model=self.model,
            cost=1e-2  # From arm example
        )
        
        # Set posture target to current configuration
        posture_task.set_target(configuration.q)
        
        # Set hand targets
        left_target_matrix = self._pose_to_matrix(target_pose_left)
        right_target_matrix = self._pose_to_matrix(target_pose_right)
        
        left_hand_task.set_target(mink.SE3.from_matrix(left_target_matrix))
        right_hand_task.set_target(mink.SE3.from_matrix(right_target_matrix))
        
        tasks = [posture_task, left_hand_task, right_hand_task]
        
        # Solver parameters from arm example
        dt = 0.01  # 100 Hz
        solver = "daqp"
        damping = 1e-3  # From arm example
        pos_threshold = 1e-4
        max_iters = 50
        
        try:
            # IK solving loop following arm example pattern
            for i in range(max_iters):
                # Compute velocity
                vel = mink.solve_ik(configuration, tasks, dt, solver, damping)
                
                # Integrate velocity
                configuration.integrate_inplace(vel, dt)
                
                # Update MuJoCo data with new configuration
                self.data.qpos[:] = configuration.q
                mujoco.mj_fwdPosition(self.model, self.data)
                
                # Check convergence based on task error
                err_left = left_hand_task.compute_error(configuration)
                err_right = right_hand_task.compute_error(configuration)
                
                pos_error_left = np.linalg.norm(err_left[:3])
                pos_error_right = np.linalg.norm(err_right[:3])
                
                # Check convergence
                if pos_error_left <= pos_threshold and pos_error_right <= pos_threshold:
                    break
            
            # Extract final arm joint positions
            final_q = configuration.q
            result = np.zeros(10)
            
            # Get left arm joints
            for i, idx in enumerate(self.left_indices):
                if i < 5 and idx < len(final_q):
                    result[i] = final_q[idx]
            
            # Get right arm joints  
            for i, idx in enumerate(self.right_indices):
                if i < 5 and idx < len(final_q):
                    result[5 + i] = final_q[idx]
            
            return result
            
        except Exception as e:
            print(f"Mink IK solve failed: {e}")
            # Return current positions as fallback
            result = np.zeros(10)
            result[:5] = qpos_arm_left[:5]
            result[5:] = qpos_arm_right[:5]
            return result
    
    def _pose_to_matrix(self, pose: Pose) -> np.ndarray:
        """Convert Pose to 4x4 transformation matrix.
        
        Args:
            pose: Pose object with position and orientation
            
        Returns:
            4x4 transformation matrix
        """
        matrix = np.eye(4)
        matrix[:3, :3] = pose.orientation.rotation_matrix
        matrix[:3, 3] = pose.position
        return matrix