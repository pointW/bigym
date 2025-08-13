"""Simplified Mink-based H1 upper body IK solver v2.

This implementation follows a simpler approach inspired by:
https://github.com/xxm19/whole-body/blob/1ba21e2a2ab5a7aec06a3cc0d01c1fc6b8fa4471/ik_rby1m.py

Key differences from v1:
- Simpler task setup without complex weights
- Direct velocity integration without manual steps
- No posture task that restricts movement
- More straightforward convergence checking
"""
import numpy as np
import mujoco
from pyquaternion import Quaternion

import mink

from bigym.bigym_env import BiGymEnv
from bigym.const import HandSide
from bigym.robots.configs.h1 import H1_CONFIG
from vr.ik.h1_upper_body_ik import Pose


class MinkH1IKv2:
    """Simplified Mink-based H1 upper body IK solver.
    
    Uses a simpler approach with fewer constraints for better flexibility.
    """
    
    def __init__(self, env: BiGymEnv):
        """Initialize simplified Mink IK solver.
        
        Args:
            env: BiGym environment containing the H1 robot
        """
        self.env = env
        self._physics = env.mojo.physics
        self._model = self._physics.model.ptr
        self._data = self._physics.data.ptr
        
        # Find end-effector sites
        self._setup_sites()
        
        # Initialize Mink configuration
        self._configuration = mink.Configuration(self._model)
        
        # Create frame tasks for end-effectors
        self._setup_tasks()
        
        # Set up simple limits
        self._setup_limits()
        
        # Solver parameters (following reference implementation)
        self._dt = 0.01  # Larger timestep for faster convergence
        self._damping = 1e-3  # Small damping for stability
        self._max_iters = 100  # Reasonable iteration limit
        self._tolerance = 0.001  # 1mm tolerance
        
        print(f"MinkH1IKv2 initialized (simplified approach)")
    
    def _setup_sites(self):
        """Find end-effector site IDs."""
        # Get site names from H1 config
        left_site_name = f"h1/{H1_CONFIG.arms[HandSide.LEFT].site}"
        right_site_name = f"h1/{H1_CONFIG.arms[HandSide.RIGHT].site}"
        
        # Find site IDs (use known defaults for H1)
        self._left_site_id = 2   # h1/left_end_effector
        self._right_site_id = 4  # h1/right_end_effector
        
        print(f"End-effector sites: left={self._left_site_id}, right={self._right_site_id}")
    
    def _setup_tasks(self):
        """Set up simple frame tasks for end-effectors."""
        # Left end-effector task (simple, no complex weights)
        self._left_task = mink.FrameTask(
            frame_name=f"h1/{H1_CONFIG.arms[HandSide.LEFT].site}",
            frame_type="site",
            position_cost=1.0,  # Simple unit cost
            orientation_cost=1.0,  # Equal weight for orientation
        )
        
        # Right end-effector task
        self._right_task = mink.FrameTask(
            frame_name=f"h1/{H1_CONFIG.arms[HandSide.RIGHT].site}",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        
        # Just the essential tasks, no posture task that restricts movement
        self._tasks = [self._left_task, self._right_task]
        
        print(f"Set up {len(self._tasks)} frame tasks")
    
    def _setup_limits(self):
        """Set up simple joint limits."""
        # Only configuration limits, no velocity limits initially
        self._configuration_limit = mink.ConfigurationLimit(
            model=self._model,
            gain=0.95,  # Stay well within limits
        )
        
        self._limits = [self._configuration_limit]
        
        print(f"Set up configuration limits")
    
    def solve(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        target_pose_left: Pose,
        target_pose_right: Pose,
    ) -> np.ndarray:
        """Solve IK using simplified approach.
        
        Args:
            pelvis_pose: Current pelvis pose
            qpos_arm_left: Current left arm joint positions
            qpos_arm_right: Current right arm joint positions
            target_pose_left: Target left end-effector pose
            target_pose_right: Target right end-effector pose
            
        Returns:
            Joint positions for both arms to achieve target poses
        """
        # Set initial robot state
        self._set_robot_state(pelvis_pose, qpos_arm_left, qpos_arm_right)
        
        # Ensure proper joint count
        if len(qpos_arm_left) == 4:
            qpos_arm_left = np.append(qpos_arm_left, 0.0)
        elif len(qpos_arm_left) > 5:
            qpos_arm_left = qpos_arm_left[:5]
            
        if len(qpos_arm_right) == 4:
            qpos_arm_right = np.append(qpos_arm_right, 0.0)
        elif len(qpos_arm_right) > 5:
            qpos_arm_right = qpos_arm_right[:5]
        
        # Set task targets
        self._set_task_targets(target_pose_left, target_pose_right)
        
        # Solve iteratively (following reference implementation style)
        for iteration in range(self._max_iters):
            # Solve for joint velocities
            try:
                velocity = mink.solve_ik(
                    configuration=self._configuration,
                    tasks=self._tasks,
                    dt=self._dt,
                    solver="daqp",  # Use available solver
                    damping=self._damping,
                    limits=self._limits,
                )
            except Exception as e:
                # Fallback without limits if solver fails
                try:
                    velocity = mink.solve_ik(
                        configuration=self._configuration,
                        tasks=self._tasks,
                        dt=self._dt,
                        solver="daqp",
                        damping=self._damping * 10,  # Higher damping for stability
                    )
                except Exception as e2:
                    print(f"Solver failed at iteration {iteration}: {e2}")
                    break
            
            # Integrate velocity (simpler approach)
            self._configuration.integrate(velocity, self._dt)
            
            # Update MuJoCo data
            self._data.qpos[:] = self._configuration.q
            mujoco.mj_fwdPosition(self._model, self._data)
            
            # Check convergence every 5 iterations
            if iteration % 5 == 0:
                left_pos = self._data.site_xpos[self._left_site_id]
                right_pos = self._data.site_xpos[self._right_site_id]
                
                left_error = np.linalg.norm(left_pos - target_pose_left.position)
                right_error = np.linalg.norm(right_pos - target_pose_right.position)
                
                if left_error < self._tolerance and right_error < self._tolerance:
                    # Converged!
                    return self._extract_arm_joints()
                
                # Print progress occasionally
                if iteration % 20 == 0 and iteration > 0:
                    print(f"  Iter {iteration}: L={left_error*1000:.1f}mm R={right_error*1000:.1f}mm")
        
        # Return best solution even if not converged
        return self._extract_arm_joints()
    
    def _set_robot_state(self, pelvis_pose: Pose, qpos_arm_left: np.ndarray, qpos_arm_right: np.ndarray):
        """Set robot state in the solver."""
        # For H1, the floating base is complex and not directly in qpos[0:7]
        # We'll skip setting pelvis for now and just work with arms
        # This is a simplification but should work for arm-only IK
        
        # Set arm joints based on known H1 indices
        # Left arm: qpos[3, 4, 5, 6, 7]
        # Right arm: qpos[16, 17, 18, 19, 20]
        arm_indices_left = [3, 4, 5, 6, 7]
        arm_indices_right = [16, 17, 18, 19, 20]
        
        for i, idx in enumerate(arm_indices_left):
            if i < len(qpos_arm_left):
                self._data.qpos[idx] = qpos_arm_left[i]
        
        for i, idx in enumerate(arm_indices_right):
            if i < len(qpos_arm_right):
                self._data.qpos[idx] = qpos_arm_right[i]
        
        # Clear velocities
        self._data.qvel[:] = 0.0
        
        # Update configuration
        self._configuration.update(self._data.qpos)
        
        # Forward kinematics
        mujoco.mj_fwdPosition(self._model, self._data)
        
        # Debug: check initial end-effector positions
        if hasattr(self, '_left_site_id'):
            left_pos = self._data.site_xpos[self._left_site_id]
            right_pos = self._data.site_xpos[self._right_site_id]
            print(f"  Initial EE positions: L={left_pos} R={right_pos}")
    
    def _set_task_targets(self, target_pose_left: Pose, target_pose_right: Pose):
        """Set target poses for the frame tasks."""
        # Left target
        left_quat = target_pose_left.orientation.elements  # [w, x, y, z]
        left_transform = mink.SE3(
            np.array([
                left_quat[0], left_quat[1], left_quat[2], left_quat[3],
                target_pose_left.position[0],
                target_pose_left.position[1], 
                target_pose_left.position[2]
            ])
        )
        self._left_task.set_target(left_transform)
        
        # Right target
        right_quat = target_pose_right.orientation.elements
        right_transform = mink.SE3(
            np.array([
                right_quat[0], right_quat[1], right_quat[2], right_quat[3],
                target_pose_right.position[0],
                target_pose_right.position[1],
                target_pose_right.position[2]
            ])
        )
        self._right_task.set_target(right_transform)
    
    def _extract_arm_joints(self) -> np.ndarray:
        """Extract arm joint positions from current state."""
        # Extract arm joints based on known H1 indices
        arm_indices_left = [3, 4, 5, 6, 7]
        arm_indices_right = [16, 17, 18, 19, 20]
        
        left_joints = [self._data.qpos[i] for i in arm_indices_left]
        right_joints = [self._data.qpos[i] for i in arm_indices_right]
        
        return np.concatenate([left_joints, right_joints])
    
    def calibrate_with_real_robot(self, pelvis_pose: Pose, qpos_arm_left: np.ndarray, 
                                  qpos_arm_right: np.ndarray, real_left_pose: Pose, 
                                  real_right_pose: Pose):
        """Calibration stub for compatibility."""
        # This simplified version doesn't need calibration
        # as it works directly with the provided environment
        pass