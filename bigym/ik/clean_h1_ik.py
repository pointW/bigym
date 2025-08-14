"""Clean H1 IK solver with optimization-based approach."""
import numpy as np
from typing import Optional, Tuple
from pyquaternion import Quaternion
from scipy.optimize import minimize


class Pose:
    """Simple pose class matching the original implementation."""
    def __init__(self, position: np.ndarray, orientation: Quaternion):
        self.position = position
        self.orientation = orientation


class CleanH1UpperBodyIK:
    """Clean IK solver for H1 robot upper body using optimization.
    
    This solver uses scipy.optimize instead of physics simulation for better accuracy.
    """
    
    def __init__(self, env):
        """Initialize the clean IK solver.
        
        Args:
            env: Environment containing robot and mojo physics
        """
        self.env = env
        self.robot = env.robot
        self.mojo = env.mojo
        
        # Store joint limits
        self.joint_limits = self._get_joint_limits()
        
    def _get_joint_limits(self):
        """Get joint limits for arm joints."""
        limits = []
        for i in range(10):  # 5 joints per arm
            actuator = self.robot.limb_actuators[i]
            if actuator.joint:
                joint = actuator.joint
                # Extract range from joint element
                joint_range = joint.range if hasattr(joint, 'range') else None
                if joint_range is not None:
                    # Parse the range string or attribute
                    if isinstance(joint_range, str):
                        low, high = map(float, joint_range.split())
                    else:
                        low = float(joint_range[0]) if hasattr(joint_range, '__getitem__') else -np.pi
                        high = float(joint_range[1]) if hasattr(joint_range, '__getitem__') else np.pi
                    limits.append((low, high))
                else:
                    limits.append((-np.pi, np.pi))
            else:
                limits.append((-np.pi, np.pi))
        return limits
    
    def solve(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        target_pose_left: Pose,
        target_pose_right: Pose,
    ) -> np.ndarray:
        """Solve IK for target end-effector poses using optimization.
        
        Args:
            pelvis_pose: Current pelvis pose
            qpos_arm_left: Current left arm joint positions (5 DOF)
            qpos_arm_right: Current right arm joint positions (5 DOF)
            target_pose_left: Target pose for left end-effector
            target_pose_right: Target pose for right end-effector
            
        Returns:
            Array of joint positions [left_arm(5), right_arm(5)]
        """
        # Initial guess from current positions
        x0 = np.concatenate([qpos_arm_left[:5], qpos_arm_right[:5]])
        
        # Objective function: minimize distance to targets
        def objective(x):
            # Set joint positions in simulation
            for i in range(5):
                actuator = self.robot.limb_actuators[i]
                if actuator.joint:
                    joint = self.mojo.physics.bind(actuator.joint)
                    joint.qpos = x[i]
            
            for i in range(5):
                actuator = self.robot.limb_actuators[5 + i]
                if actuator.joint:
                    joint = self.mojo.physics.bind(actuator.joint)
                    joint.qpos = x[5 + i]
            
            # Forward kinematics
            self.mojo.physics.forward()
            
            # Get achieved positions
            from bigym.const import HandSide
            left_site = self.robot._wrist_sites[HandSide.LEFT]
            right_site = self.robot._wrist_sites[HandSide.RIGHT]
            
            left_pos = left_site.get_position()
            right_pos = right_site.get_position()
            
            # Compute position errors
            left_pos_error = np.linalg.norm(left_pos - target_pose_left.position)
            right_pos_error = np.linalg.norm(right_pos - target_pose_right.position)
            
            # Optionally add orientation error
            left_quat = Quaternion(left_site.get_quaternion())
            right_quat = Quaternion(right_site.get_quaternion())
            
            left_ori_error = Quaternion.distance(left_quat, target_pose_left.orientation)
            right_ori_error = Quaternion.distance(right_quat, target_pose_right.orientation)
            
            # Total error (position weighted more than orientation)
            total_error = (left_pos_error + right_pos_error) * 100 + (left_ori_error + right_ori_error) * 10
            
            return total_error
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=self.joint_limits,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        return result.x