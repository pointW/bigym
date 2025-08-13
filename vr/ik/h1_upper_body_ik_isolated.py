"""Isolated wrapper for H1UpperBodyIK solver with dummy environment.

This wrapper ensures the original IK solver doesn't modify the actual simulation state
by providing it with a separate dummy environment that gets synchronized before each solve.
"""
import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from vr.ik.h1_upper_body_ik import H1UpperBodyIK, Pose


class H1UpperBodyIKIsolated:
    """Isolated H1 upper body IK solver with dummy environment.
    
    This wrapper creates a separate dummy environment for the original H1UpperBodyIK solver
    to prevent it from modifying the actual simulation state during IK computation.
    """
    
    def __init__(self, actual_env: BiGymEnv, enable_full_6d_control: bool = False):
        """Initialize isolated IK solver with dummy environment.
        
        Args:
            actual_env: The actual BiGym environment (used for reference only)
            enable_full_6d_control: Whether to enable full 6D orientation control
        """
        self.actual_env = actual_env
        self.enable_full_6d_control = enable_full_6d_control
        
        # Create a separate dummy environment for IK solving
        # This prevents the IK solver from modifying the actual simulation
        dummy_joint_mode = JointPositionActionMode(
            absolute=True,
            floating_base=True,  # Assume floating base for now
        )
        self._dummy_env = ReachTarget(action_mode=dummy_joint_mode, render_mode=None)
        
        # Reset dummy environment to get its initial state
        self._dummy_env.reset()
        
        # Create the original solver with the dummy environment
        self._original_solver = H1UpperBodyIK(self._dummy_env, enable_full_6d_control)
        
        # Store reference to actual robot for state synchronization
        self._actual_robot = actual_env.robot
        
        # Calibration offsets (difference between actual and dummy environments)
        self._left_offset = np.zeros(3)
        self._right_offset = np.zeros(3)
        self._calibrated = False
        
        print(f"H1UpperBodyIKIsolated initialized with dummy environment")
        print(f"  Full 6D control: {enable_full_6d_control}")
    
    def _sync_dummy_state(self, pelvis_pose: Pose, qpos_arm_left: np.ndarray, qpos_arm_right: np.ndarray):
        """Synchronize dummy environment state with provided configuration.
        
        This ensures the dummy environment matches the desired state before solving.
        
        Args:
            pelvis_pose: Pelvis pose to set
            qpos_arm_left: Left arm joint positions
            qpos_arm_right: Right arm joint positions
        """
        # Get current state from actual robot
        actual_qpos = self._actual_robot._mojo.physics.data.qpos.copy()
        actual_qvel = self._actual_robot._mojo.physics.data.qvel.copy()
        
        # Sync the dummy environment's full state
        self._dummy_env.robot._mojo.physics.data.qpos[:] = actual_qpos
        self._dummy_env.robot._mojo.physics.data.qvel[:] = actual_qvel
        
        # Also sync control state if it exists
        if hasattr(self._actual_robot._mojo.physics.data, 'ctrl'):
            actual_ctrl = self._actual_robot._mojo.physics.data.ctrl.copy()
            if hasattr(self._dummy_env.robot._mojo.physics.data, 'ctrl'):
                self._dummy_env.robot._mojo.physics.data.ctrl[:] = actual_ctrl
        
        # Forward kinematics to update positions
        import mujoco
        mujoco.mj_fwdPosition(
            self._dummy_env.robot._mojo.physics.model.ptr,
            self._dummy_env.robot._mojo.physics.data.ptr
        )
        
        # Also update the solver's physics view
        mujoco.mj_fwdPosition(
            self._original_solver._physics.model.ptr,
            self._original_solver._physics.data.ptr
        )
    
    def solve(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        target_pose_left: Pose,
        target_pose_right: Pose,
    ) -> np.ndarray:
        """Solve IK using isolated dummy environment.
        
        Args:
            pelvis_pose: Current pelvis pose
            qpos_arm_left: Current left arm joint positions
            qpos_arm_right: Current right arm joint positions
            target_pose_left: Target left end-effector pose
            target_pose_right: Target right end-effector pose
            
        Returns:
            Joint positions for both arms to achieve target poses
        """
        # Sync dummy environment with current state before solving
        self._sync_dummy_state(pelvis_pose, qpos_arm_left, qpos_arm_right)
        
        # Apply calibration offsets to targets if calibrated
        if self._calibrated:
            # Adjust targets to account for environment differences
            calibrated_left = Pose(
                target_pose_left.position - self._left_offset,
                target_pose_left.orientation
            )
            calibrated_right = Pose(
                target_pose_right.position - self._right_offset,
                target_pose_right.orientation
            )
        else:
            calibrated_left = target_pose_left
            calibrated_right = target_pose_right
        
        # Solve using the original solver with dummy environment
        # The original solver will modify the dummy environment, not the actual one
        solution = self._original_solver.solve(
            pelvis_pose,
            qpos_arm_left,
            qpos_arm_right,
            calibrated_left,
            calibrated_right
        )
        
        return solution
    
    def calibrate_with_real_robot(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        real_left_pose: Pose,
        real_right_pose: Pose,
    ):
        """Calibrate solver with real robot offsets.
        
        This accounts for any differences between the actual and dummy environments.
        
        Args:
            pelvis_pose: Current pelvis pose
            qpos_arm_left: Current left arm joint positions
            qpos_arm_right: Current right arm joint positions
            real_left_pose: Actual left end-effector pose from robot
            real_right_pose: Actual right end-effector pose from robot
        """
        # Sync dummy environment first
        self._sync_dummy_state(pelvis_pose, qpos_arm_left, qpos_arm_right)
        
        # Solve with dummy targets to get dummy environment's end-effector positions
        dummy_solution = self._original_solver.solve(
            pelvis_pose,
            qpos_arm_left,
            qpos_arm_right,
            Pose(np.array([0.3, 0.2, 1.0]), Quaternion(w=1, x=0, y=0, z=0)),
            Pose(np.array([0.3, -0.2, 1.0]), Quaternion(w=1, x=0, y=0, z=0))
        )
        
        # Get computed positions from dummy environment
        # The original solver has already modified the dummy environment during solve
        # The solver's physics is the same as dummy env's physics
        physics = self._original_solver._physics
        left_site = physics.bind(self._original_solver._left_arm_site)
        right_site = physics.bind(self._original_solver._right_arm_site)
        
        computed_left_pos = left_site.xpos.copy()
        computed_right_pos = right_site.xpos.copy()
        
        # Calculate offsets between real and computed positions
        self._left_offset = real_left_pose.position - computed_left_pos
        self._right_offset = real_right_pose.position - computed_right_pos
        
        self._calibrated = True
        
        print(f"IK solver calibrated:")
        print(f"  Left offset: {self._left_offset} (magnitude: {np.linalg.norm(self._left_offset)*1000:.1f}mm)")
        print(f"  Right offset: {self._right_offset} (magnitude: {np.linalg.norm(self._right_offset)*1000:.1f}mm)")
    
    def get_end_effector_positions(self):
        """Get current end-effector positions from dummy environment.
        
        Returns:
            Tuple of (left_position, right_position)
        """
        # Bind sites using the solver's physics
        physics = self._original_solver._physics
        left_site = physics.bind(self._original_solver._left_arm_site)
        right_site = physics.bind(self._original_solver._right_arm_site)
        
        left_pos = left_site.xpos.copy()
        right_pos = right_site.xpos.copy()
        
        # Apply calibration offsets if calibrated
        if self._calibrated:
            left_pos = left_pos + self._left_offset
            right_pos = right_pos + self._right_offset
        
        return left_pos, right_pos
    
    def __del__(self):
        """Clean up dummy environment on deletion."""
        if hasattr(self, '_dummy_env'):
            self._dummy_env.close()