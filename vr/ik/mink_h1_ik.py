"""Mink-based H1 upper body IK solver.

This implementation uses the Mink library for differential inverse kinematics
to provide improved accuracy and robustness compared to the original solver.
"""
import numpy as np
import mujoco
from pyquaternion import Quaternion

import mink

from bigym.bigym_env import BiGymEnv
from bigym.const import HandSide
from bigym.robots.configs.h1 import H1_CONFIG
from vr.ik.h1_upper_body_ik import Pose


class MinkH1IK:
    """Mink-based H1 upper body IK solver.
    
    Uses differential inverse kinematics with quadratic programming
    for improved accuracy and convergence.
    """
    
    def __init__(self, env: BiGymEnv):
        """Initialize Mink-based IK solver.
        
        Args:
            env: BiGym environment containing the H1 robot
        """
        self.env = env
        self._physics = env.mojo.physics
        self._model = self._physics.model.ptr  # Get raw MuJoCo model
        self._data = self._physics.data.ptr    # Get raw MuJoCo data
        
        # Get robot configuration
        self._setup_robot_configuration()
        
        # Initialize Mink configuration
        self._configuration = mink.Configuration(self._model)
        
        # Set up tasks and limits
        self._setup_tasks()
        self._setup_limits()
        
        # Solver parameters
        self._dt = 0.005  # Even smaller timestep for more stable integration
        self._max_iters = 200   # More iterations with smaller steps
        self._tolerance = 0.001  # 1mm tolerance
        
        # Calibration offsets (initialized to zero, updated during calibration)
        self._left_offset = np.zeros(3)
        self._right_offset = np.zeros(3)
        
        print(f"Mink H1 IK solver initialized with {len(self._arm_joint_ids)} arm joints")
    
    def _setup_robot_configuration(self):
        """Set up robot joint and site configuration."""
        # Define arm joint names and indices based on the H1 model structure
        # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist  
        # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist
        
        self._arm_joint_names = [
            "h1/left_shoulder_pitch", "h1/left_shoulder_roll", "h1/left_shoulder_yaw", 
            "h1/left_elbow", "h1/left_wrist",
            "h1/right_shoulder_pitch", "h1/right_shoulder_roll", "h1/right_shoulder_yaw",
            "h1/right_elbow", "h1/right_wrist"
        ]
        
        self._arm_joint_ids = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20]  # Based on qposadr output
        
        # Find end-effector sites using physics named model
        named = self._physics.named.model
        site_names = [name for name in dir(named) if 'site' in name.lower()]
        
        # Look for end-effector sites
        left_site_name = f"h1/{H1_CONFIG.arms[HandSide.LEFT].site}"
        right_site_name = f"h1/{H1_CONFIG.arms[HandSide.RIGHT].site}"
        
        # Find site indices by iterating through sites
        self._left_site_id = None
        self._right_site_id = None
        
        for i in range(self._model.nsite):
            # Use dm_control's site access
            site_name = None
            try:
                # Try to get site name from physics
                if hasattr(self._physics.named, 'site'):
                    if hasattr(self._physics.named.site, 'name'):
                        all_site_names = list(self._physics.named.site.name)
                        if i < len(all_site_names):
                            site_name = all_site_names[i]
            except:
                pass
                
            if site_name == left_site_name:
                self._left_site_id = i
            elif site_name == right_site_name:
                self._right_site_id = i
        
        if self._left_site_id is None or self._right_site_id is None:
            # Fallback: use known site IDs for H1 model
            print("Warning: Could not auto-detect site IDs, using known H1 structure")
            self._left_site_id = 2   # h1/left_end_effector
            self._right_site_id = 4  # h1/right_end_effector
        
        print(f"Found {len(self._arm_joint_ids)} arm joints: {self._arm_joint_names}")
        print(f"End-effector sites: left={self._left_site_id}, right={self._right_site_id}")
    
    def _setup_tasks(self):
        """Set up IK tasks for end-effector pose control."""
        self._tasks = []
        
        # Left end-effector frame task (very high position priority)
        self._left_frame_task = mink.FrameTask(
            frame_name=f"h1/{H1_CONFIG.arms[HandSide.LEFT].site}",
            frame_type="site",
            position_cost=100.0,  # Very high priority on position accuracy
            orientation_cost=1.0,  # Moderate orientation cost 
        )
        self._tasks.append(self._left_frame_task)
        
        # Right end-effector frame task
        self._right_frame_task = mink.FrameTask(
            frame_name=f"h1/{H1_CONFIG.arms[HandSide.RIGHT].site}",
            frame_type="site", 
            position_cost=100.0,  # Very high priority on position accuracy
            orientation_cost=1.0,   # Moderate orientation cost
        )
        self._tasks.append(self._right_frame_task)
        
        # Add posture task to prevent drift and maintain reasonable configuration
        self._posture_task = mink.PostureTask(
            model=self._model,
            cost=10.0,  # Higher cost to strongly prefer initial configuration
        )
        self._tasks.append(self._posture_task)
        
        print(f"Set up {len(self._tasks)} tasks for IK solving")
    
    def _setup_limits(self):
        """Set up joint limits and velocity limits."""
        # Configuration limits for joint range enforcement
        self._configuration_limit = mink.ConfigurationLimit(
            model=self._model,
            gain=0.8,  # Conservative movement towards limits
            min_distance_from_limits=0.02  # 2cm margin from limits for safety
        )
        
        # Velocity limits for stability
        max_velocity = 2.0  # rad/s - conservative limit for stability
        velocity_limits = {}
        for joint_name in self._arm_joint_names:
            velocity_limits[joint_name] = max_velocity
        
        self._velocity_limit = mink.VelocityLimit(
            model=self._model,
            velocities=velocity_limits,
        )
        
        # Enable both limits
        self._limits = [self._configuration_limit, self._velocity_limit]
        
        print(f"Set up configuration and velocity limits for {len(self._arm_joint_names)} joints")
    
    def solve(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        target_pose_left: Pose,
        target_pose_right: Pose,
    ) -> np.ndarray:
        """Solve inverse kinematics for target end-effector poses.
        
        Args:
            pelvis_pose: Current pelvis pose
            qpos_arm_left: Current left arm joint positions
            qpos_arm_right: Current right arm joint positions
            target_pose_left: Target left end-effector pose
            target_pose_right: Target right end-effector pose
            
        Returns:
            Joint positions for both arms to achieve target poses
        """
        # Set robot configuration
        self._set_robot_state(pelvis_pose, qpos_arm_left, qpos_arm_right)
        
        # Ensure arm poses have correct joint count (5 joints each)
        if len(qpos_arm_left) == 4:  # Original solver format (no wrist)
            qpos_arm_left = np.append(qpos_arm_left, 0.0)  # Add wrist joint
        elif len(qpos_arm_left) > 5:  # Test format might have extra joints
            qpos_arm_left = qpos_arm_left[:5]  # Take first 5 joints
        
        if len(qpos_arm_right) == 4:
            qpos_arm_right = np.append(qpos_arm_right, 0.0)
        elif len(qpos_arm_right) > 5:
            qpos_arm_right = qpos_arm_right[:5]
            
        # Ensure 5 joints per arm
        assert len(qpos_arm_left) == 5 and len(qpos_arm_right) == 5, f"Expected 5 joints per arm, got {len(qpos_arm_left)}, {len(qpos_arm_right)}"
        
        # Set task targets  
        self._set_task_targets(target_pose_left, target_pose_right, qpos_arm_left, qpos_arm_right)
        
        # Solve differential IK iteratively with progressive damping
        for iteration in range(self._max_iters):
            # Higher damping for stability with manual integration
            damping = 0.5 * (1.0 - iteration / self._max_iters) + 0.1
            
            # Solve for joint velocities
            try:
                velocity = mink.solve_ik(
                    configuration=self._configuration,
                    tasks=self._tasks,
                    dt=self._dt,
                    solver="daqp",
                    damping=damping,
                    limits=self._limits,
                )
                # Debug: check velocity magnitude
                if iteration < 3:  # Only for first few iterations
                    vel_norm = np.linalg.norm(velocity)
                    print(f"Iteration {iteration}: velocity norm = {vel_norm:.6f}")
            except Exception as e:
                # If QP fails, try without limits as fallback
                try:
                    velocity = mink.solve_ik(
                        configuration=self._configuration,
                        tasks=self._tasks,
                        dt=self._dt,
                        solver="daqp",
                        damping=damping * 2.0,  # Higher damping for stability
                    )
                    if iteration < 3:
                        vel_norm = np.linalg.norm(velocity)
                        print(f"Iteration {iteration}: fallback velocity norm = {vel_norm:.6f}")
                except Exception as e2:
                    print(f"Mink solver failed at iteration {iteration}: {e}")
                    break
            
            # Manual velocity integration (Mink's integrate() seems broken)
            # Store current configuration
            q_current = self._configuration.q.copy()
            
            # Manual integration: q_new = q_current + velocity * dt
            q_new = q_current + velocity * self._dt
            
            # Update configuration and MuJoCo data
            self._configuration.update(q_new)
            self._data.qpos[:] = q_new
            mujoco.mj_fwdPosition(self._model, self._data)
            
            # Check convergence more frequently for better responsiveness
            if iteration % 3 == 2:  # Check every 3 iterations
                if self._check_convergence(target_pose_left, target_pose_right):
                    print(f"Converged after {iteration + 1} iterations")
                    # Extract solution immediately upon convergence to preserve state
                    arm_qpos = self._extract_arm_joint_positions()
                    # Store the full converged state for accurate final pose calculation
                    self._converged_full_state = self._data.qpos.copy()
                    # Split into left and right arms (5 joints each)
                    n_left_joints = 5
                    left_solution = arm_qpos[:n_left_joints]
                    right_solution = arm_qpos[n_left_joints:]
                    return np.concatenate((left_solution, right_solution))
                else:
                    # Debug: show current error
                    mujoco.mj_fwdPosition(self._model, self._data)
                    left_pos = self._data.site_xpos[self._left_site_id]
                    right_pos = self._data.site_xpos[self._right_site_id]
                    left_error = np.linalg.norm(left_pos - target_pose_left.position)
                    right_error = np.linalg.norm(right_pos - target_pose_right.position)
                    
                    # Show progress less frequently to reduce noise
                    if iteration % 15 == 14:  # Every 15 iterations
                        print(f"Iteration {iteration + 1}: L={left_error*1000:.1f}mm R={right_error*1000:.1f}mm")
                    
                    # Early stopping if error is increasing consistently
                    if iteration > 30 and (left_error > 0.5 or right_error > 0.5):  # 500mm threshold
                        print(f"Early stopping at iteration {iteration + 1} due to large errors")
                        break
        
        # If we didn't converge, extract the best solution we have
        arm_qpos = self._extract_arm_joint_positions()
        
        # Split into left and right arms (5 joints each)
        n_left_joints = 5
        left_solution = arm_qpos[:n_left_joints]
        right_solution = arm_qpos[n_left_joints:]
        
        return np.concatenate((left_solution, right_solution))
    
    def _set_robot_state(self, pelvis_pose: Pose, qpos_arm_left: np.ndarray, qpos_arm_right: np.ndarray):
        """Set the robot state in Mink configuration."""
        # Set pelvis pose
        pelvis_pos = pelvis_pose.position
        pelvis_quat = pelvis_pose.orientation.elements  # [w, x, y, z]
        
        # Set pelvis position and orientation in the physics (base position + quaternion)
        self._data.qpos[0:3] = pelvis_pos
        self._data.qpos[3:7] = pelvis_quat
        
        # Set arm joint positions using the correct joint indices
        arm_qpos = np.concatenate((qpos_arm_left, qpos_arm_right))
        
        # Set arm joints directly using known indices: [3,4,5,6,7,16,17,18,19,20]
        for i, joint_id in enumerate(self._arm_joint_ids):
            if i < len(arm_qpos):
                self._data.qpos[joint_id] = arm_qpos[i]
        
        # Set velocities to zero
        self._data.qvel[:] = 0.0
        
        # Update Mink configuration
        self._configuration.update(self._data.qpos)
        
        # Forward kinematics to update site positions
        mujoco.mj_fwdPosition(self._model, self._data)
        
        # Debug output removed for cleaner logs
    
    def _set_task_targets(self, target_pose_left: Pose, target_pose_right: Pose,
                         qpos_arm_left: np.ndarray, qpos_arm_right: np.ndarray):
        """Set target poses for IK tasks."""
        # Apply calibration offsets to targets
        # The targets from CartesianActionMode are in "real robot" space
        # We need to convert them to "IK solver" space by subtracting offsets
        calibrated_left_pos = target_pose_left.position - self._left_offset
        calibrated_right_pos = target_pose_right.position - self._right_offset
        
        # Left end-effector target
        left_quat = target_pose_left.orientation.elements  # [w, x, y, z]
        # SE3 expects (qw, qx, qy, qz, x, y, z) format
        left_wxyz_xyz = np.array([left_quat[0], left_quat[1], left_quat[2], left_quat[3], 
                                 calibrated_left_pos[0], calibrated_left_pos[1], calibrated_left_pos[2]])
        left_transform = mink.SE3(left_wxyz_xyz)
        self._left_frame_task.set_target(left_transform)
        
        # Right end-effector target
        right_quat = target_pose_right.orientation.elements
        right_wxyz_xyz = np.array([right_quat[0], right_quat[1], right_quat[2], right_quat[3],
                                  calibrated_right_pos[0], calibrated_right_pos[1], calibrated_right_pos[2]])
        right_transform = mink.SE3(right_wxyz_xyz)
        self._right_frame_task.set_target(right_transform)
        
        # Set posture task target to maintain the INITIAL configuration
        # This prevents the solver from drifting to alternative IK solutions
        # We want to preserve the current joint configuration as much as possible
        initial_qpos = self._data.qpos.copy()
        self._posture_task.set_target(initial_qpos)
    
    def _check_convergence(self, target_pose_left: Pose, target_pose_right: Pose) -> bool:
        """Check if IK has converged to target poses."""
        # Get current end-effector positions  
        mujoco.mj_fwdPosition(self._model, self._data)
        
        left_pos = self._data.site_xpos[self._left_site_id]
        right_pos = self._data.site_xpos[self._right_site_id]
        
        # Calculate position errors
        left_error = np.linalg.norm(left_pos - target_pose_left.position)
        right_error = np.linalg.norm(right_pos - target_pose_right.position)
        
        # Debug convergence check
        converged = left_error < self._tolerance and right_error < self._tolerance
        if converged:
            print(f"Convergence check: L={left_error*1000:.1f}mm R={right_error*1000:.1f}mm (tolerance={self._tolerance*1000:.1f}mm)")
        
        # Check if both errors are below tolerance
        return converged
    
    def _extract_arm_joint_positions(self) -> np.ndarray:
        """Extract arm joint positions from current configuration."""
        arm_qpos = []
        
        # Use the actual joint IDs directly instead of sequential indexing
        for joint_id in self._arm_joint_ids:
            if joint_id < len(self._data.qpos):
                arm_qpos.append(self._data.qpos[joint_id])
            else:
                arm_qpos.append(0.0)  # Fallback
        
        return np.array(arm_qpos)
    
    def get_converged_end_effector_positions(self):
        """Get end-effector positions using the stored converged full state.
        
        This avoids the inconsistency caused by _set_robot_state() which only
        sets arm joints but resets other DOFs that affect forward kinematics.
        
        Returns:
            Tuple of (left_position, right_position) using converged state
        """
        if not hasattr(self, '_converged_full_state'):
            # Fallback to current state if no converged state stored
            mujoco.mj_fwdPosition(self._model, self._data)
            left_pos = self._data.site_xpos[self._left_site_id].copy()
            right_pos = self._data.site_xpos[self._right_site_id].copy()
            return left_pos, right_pos
        
        # Temporarily apply the full converged state
        original_qpos = self._data.qpos.copy()
        self._data.qpos[:] = self._converged_full_state
        mujoco.mj_fwdPosition(self._model, self._data)
        
        # Get positions with converged state
        left_pos = self._data.site_xpos[self._left_site_id].copy()
        right_pos = self._data.site_xpos[self._right_site_id].copy()
        
        # Restore original state
        self._data.qpos[:] = original_qpos
        mujoco.mj_fwdPosition(self._model, self._data)
        
        return left_pos, right_pos

    def calibrate_with_real_robot(self, pelvis_pose: Pose, qpos_arm_left: np.ndarray, 
                                  qpos_arm_right: np.ndarray, real_left_pose: Pose, 
                                  real_right_pose: Pose):
        """Calibrate the IK solver with real robot offsets.
        
        This is critical for matching the actual robot's end-effector positions
        to the IK solver's computed positions.
        
        Args:
            pelvis_pose: Current pelvis pose
            qpos_arm_left: Current left arm joint positions
            qpos_arm_right: Current right arm joint positions
            real_left_pose: Actual left end-effector pose from robot
            real_right_pose: Actual right end-effector pose from robot
        """
        # Set the robot state to current configuration
        self._set_robot_state(pelvis_pose, qpos_arm_left, qpos_arm_right)
        
        # Get IK solver's computed end-effector positions
        computed_left_pos = self._data.site_xpos[self._left_site_id].copy()
        computed_right_pos = self._data.site_xpos[self._right_site_id].copy()
        
        # Calculate offsets between real and computed positions
        self._left_offset = real_left_pose.position - computed_left_pos
        self._right_offset = real_right_pose.position - computed_right_pos
        
        print(f"IK solver calibrated:")
        print(f"  Left offset: {self._left_offset}")
        print(f"  Right offset: {self._right_offset}")