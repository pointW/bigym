#!/usr/bin/env python3
"""Find RBY1 arm joint positions with torso truly fixed at zero."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bigym.envs.reach_target import ReachTarget
from bigym.action_modes import JointPositionActionMode
from bigym.robots.configs.h1 import H1
from bigym.robots.configs.rby1 import RBY1
from bigym.const import HandSide
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import mujoco

def main():
    print("=" * 80)
    print("FINDING RBY1 ARM JOINTS WITH TORSO FIXED AT 0")
    print("=" * 80)
    
    # Step 1: Get H1's target poses
    print("\n1. Getting H1 target poses...")
    print("-" * 60)
    
    h1_env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
        robot_cls=H1
    )
    
    h1_env.reset(seed=42)
    
    h1_left_pos = h1_env.robot._wrist_sites[HandSide.LEFT].get_position()
    h1_right_pos = h1_env.robot._wrist_sites[HandSide.RIGHT].get_position()
    h1_left_quat = h1_env.robot._wrist_sites[HandSide.LEFT].get_quaternion()
    h1_right_quat = h1_env.robot._wrist_sites[HandSide.RIGHT].get_quaternion()
    
    print(f"H1 Left EE:  pos={h1_left_pos}")
    print(f"H1 Right EE: pos={h1_right_pos}")
    
    h1_env.close()
    
    # Step 2: Use H1 positions directly (RBY1 model is already scaled)
    target_left_pos = h1_left_pos
    target_right_pos = h1_right_pos
    target_left_quat = h1_left_quat
    target_right_quat = h1_right_quat
    
    print(f"\nRBY1 Targets:")
    print(f"  Left:  {target_left_pos}")
    print(f"  Right: {target_right_pos}")
    
    # Step 3: Setup RBY1 with optimization
    print("\n2. Setting up arm-only optimization...")
    print("-" * 60)
    
    rby1_env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=False, absolute=True),
        control_frequency=50,
        render_mode=None,
        robot_cls=RBY1
    )
    
    rby1_env.reset(seed=42)
    
    # Get physics references
    physics = rby1_env.robot._mojo.physics
    model = physics.model.ptr
    data = physics.data.ptr
    
    # Get site IDs
    left_site_name = "rby1/end_effector_l"
    right_site_name = "rby1/end_effector_r"
    left_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_site_name)
    right_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, right_site_name)
    
    # Set torso to 0 and keep it fixed
    initial_qpos = physics.data.qpos.copy()
    torso_indices = list(range(11, 17))
    for idx in torso_indices:
        initial_qpos[idx] = 0.0
    
    # Arm joint indices
    right_arm_indices = list(range(17, 24))
    left_arm_indices = list(range(32, 39))
    arm_indices = right_arm_indices + left_arm_indices
    
    print(f"Torso joints (11-16): fixed at 0")
    print(f"Right arm joints (17-23): optimizing")
    print(f"Left arm joints (32-38): optimizing")
    
    # Optimization function
    def objective(arm_values):
        # Create full qpos with fixed torso
        qpos = initial_qpos.copy()
        
        # Set arm values
        for i, idx in enumerate(arm_indices):
            qpos[idx] = arm_values[i]
        
        # Apply and compute forward kinematics
        physics.data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        # Get current end-effector positions and orientations
        left_pos = data.site_xpos[left_site_id].copy()
        right_pos = data.site_xpos[right_site_id].copy()
        
        # Get current orientations as rotation matrices
        left_mat = data.site_xmat[left_site_id].reshape(3, 3).copy()
        right_mat = data.site_xmat[right_site_id].reshape(3, 3).copy()
        
        # Convert target quaternions to rotation matrices
        left_rot_target = Rotation.from_quat([target_left_quat[1], target_left_quat[2], target_left_quat[3], target_left_quat[0]])
        right_rot_target = Rotation.from_quat([target_right_quat[1], target_right_quat[2], target_right_quat[3], target_right_quat[0]])
        left_mat_target = left_rot_target.as_matrix()
        right_mat_target = right_rot_target.as_matrix()
        
        # Compute position errors
        left_pos_error = np.linalg.norm(left_pos - target_left_pos)
        right_pos_error = np.linalg.norm(right_pos - target_right_pos)
        
        # Compute orientation errors (Frobenius norm of rotation matrix difference)
        left_ori_error = np.linalg.norm(left_mat - left_mat_target, 'fro') * 0.1
        right_ori_error = np.linalg.norm(right_mat - right_mat_target, 'fro') * 0.1
        
        # Add regularization for joint limits and smoothness
        joint_regularization = 0.001 * np.sum(arm_values**2)
        
        return left_pos_error + right_pos_error + left_ori_error + right_ori_error + joint_regularization
    
    # Get initial arm values
    initial_arms = [initial_qpos[idx] for idx in arm_indices]
    
    # Set bounds for arm joints
    bounds = []
    for idx in arm_indices:
        # Find joint ID for this qpos index
        for j in range(model.njnt):
            if model.jnt_qposadr[j] == idx:
                joint_range = model.jnt_range[j]
                bounds.append((joint_range[0], joint_range[1]))
                break
    
    print("\n3. Running optimization...")
    print("-" * 60)
    
    # Run optimization
    result = minimize(
        objective,
        initial_arms,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    
    if result.success:
        print(f"✅ Optimization converged!")
    else:
        print(f"⚠️  Optimization did not fully converge")
    
    print(f"  Iterations: {result.nit}")
    print(f"  Final error: {result.fun:.6f}")
    
    # Apply solution
    solution_qpos = initial_qpos.copy()
    for i, idx in enumerate(arm_indices):
        solution_qpos[idx] = result.x[i]
    
    physics.data.qpos[:] = solution_qpos
    mujoco.mj_forward(model, data)
    
    # Verify results
    print("\n4. Verification:")
    print("-" * 60)
    
    # Get achieved positions
    left_site = rby1_env.robot._wrist_sites[HandSide.LEFT]
    right_site = rby1_env.robot._wrist_sites[HandSide.RIGHT]
    
    achieved_left_pos = left_site.get_position()
    achieved_right_pos = right_site.get_position()
    achieved_left_quat = left_site.get_quaternion()
    achieved_right_quat = right_site.get_quaternion()
    
    left_pos_error = np.linalg.norm(achieved_left_pos - target_left_pos) * 1000
    right_pos_error = np.linalg.norm(achieved_right_pos - target_right_pos) * 1000
    
    # Calculate rotation differences
    left_rot_target = Rotation.from_quat([target_left_quat[1], target_left_quat[2], target_left_quat[3], target_left_quat[0]])
    right_rot_target = Rotation.from_quat([target_right_quat[1], target_right_quat[2], target_right_quat[3], target_right_quat[0]])
    left_rot_achieved = Rotation.from_quat([achieved_left_quat[1], achieved_left_quat[2], achieved_left_quat[3], achieved_left_quat[0]])
    right_rot_achieved = Rotation.from_quat([achieved_right_quat[1], achieved_right_quat[2], achieved_right_quat[3], achieved_right_quat[0]])
    
    left_angle_diff = np.linalg.norm((left_rot_target.inv() * left_rot_achieved).as_rotvec()) * 180 / np.pi
    right_angle_diff = np.linalg.norm((right_rot_target.inv() * right_rot_achieved).as_rotvec()) * 180 / np.pi
    
    print(f"Position errors:")
    print(f"  Left:  {left_pos_error:.2f}mm")
    print(f"  Right: {right_pos_error:.2f}mm")
    
    print(f"Orientation errors:")
    print(f"  Left:  {left_angle_diff:.1f}°")
    print(f"  Right: {right_angle_diff:.1f}°")
    
    # Verify torso is at 0
    print("\n5. Joint values:")
    print("-" * 60)
    
    torso_joints = solution_qpos[11:17]
    right_arm_joints = solution_qpos[17:24]
    left_arm_joints = solution_qpos[32:39]
    
    print("Torso (should be all 0):")
    for i, val in enumerate(torso_joints):
        status = "✓" if abs(val) < 0.001 else "✗"
        print(f"  torso_{i}: {val:7.4f} {status}")
    
    print("\nRight arm:")
    for i, val in enumerate(right_arm_joints):
        print(f"  right_{i}: {val:7.4f}")
    
    print("\nLeft arm:")
    for i, val in enumerate(left_arm_joints):
        print(f"  left_{i}: {val:7.4f}")
    
    # Output reset_state
    reset_state = np.concatenate([torso_joints, right_arm_joints, left_arm_joints])
    
    print("\n6. New reset_state for RBY1:")
    print("-" * 60)
    print("reset_state=np.array([")
    print(f"    # Torso (6 DOF) - fixed at zero")
    for i in range(6):
        print(f"    {reset_state[i]:.4f},")
    print(f"    # Right arm (7 DOF)")
    for i in range(6, 13):
        print(f"    {reset_state[i]:.4f},")
    print(f"    # Left arm (7 DOF)")
    for i in range(13, 20):
        if i < 19:
            print(f"    {reset_state[i]:.4f},")
        else:
            print(f"    {reset_state[i]:.4f}")
    print("])")
    
    # Assessment
    print("\n7. Assessment:")
    print("-" * 60)
    
    if max(left_pos_error, right_pos_error) < 5.0:
        print(f"✅ Position match: {max(left_pos_error, right_pos_error):.2f}mm < 5mm")
    else:
        print(f"⚠️  Position error: {max(left_pos_error, right_pos_error):.2f}mm")
    
    if max(left_angle_diff, right_angle_diff) < 5.0:
        print(f"✅ Orientation match: {max(left_angle_diff, right_angle_diff):.1f}° < 5°")
    else:
        print(f"⚠️  Orientation error: {max(left_angle_diff, right_angle_diff):.1f}°")
    
    if all(abs(v) < 0.001 for v in torso_joints):
        print("✅ Torso joints all at zero - clean posture!")
    
    rby1_env.close()

if __name__ == "__main__":
    main()