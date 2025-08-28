#!/usr/bin/env python3
"""Verify RBY1 reset matches H1 end-effector pose."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bigym.envs.reach_target import ReachTarget
from bigym.action_modes import JointPositionActionMode
from bigym.robots.configs.rby1 import RBY1
from bigym.robots.configs.h1 import H1
from bigym.const import HandSide
from scipy.spatial.transform import Rotation

def main():
    print("=" * 80)
    print("VERIFICATION: RBY1 RESET vs H1 END-EFFECTOR POSE")
    print("=" * 80)
    
    # Get H1 reference pose
    print("\n1. H1 REFERENCE POSE:")
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
    
    h1_left_rot = Rotation.from_quat([h1_left_quat[1], h1_left_quat[2], h1_left_quat[3], h1_left_quat[0]])
    h1_right_rot = Rotation.from_quat([h1_right_quat[1], h1_right_quat[2], h1_right_quat[3], h1_right_quat[0]])
    
    print(f"Left hand:")
    print(f"  Position: {h1_left_pos}")
    print(f"  Euler: {h1_left_rot.as_euler('xyz', degrees=True)}")
    
    print(f"Right hand:")
    print(f"  Position: {h1_right_pos}")
    print(f"  Euler: {h1_right_rot.as_euler('xyz', degrees=True)}")
    
    h1_env.close()
    
    # Get RBY1 pose
    print("\n2. RBY1 CURRENT RESET POSE:")
    print("-" * 60)
    rby1_env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=False, absolute=True),
        control_frequency=50,
        render_mode=None,
        robot_cls=RBY1
    )
    rby1_env.reset(seed=42)
    
    rby1_left_pos = rby1_env.robot._wrist_sites[HandSide.LEFT].get_position()
    rby1_right_pos = rby1_env.robot._wrist_sites[HandSide.RIGHT].get_position()
    rby1_left_quat = rby1_env.robot._wrist_sites[HandSide.LEFT].get_quaternion()
    rby1_right_quat = rby1_env.robot._wrist_sites[HandSide.RIGHT].get_quaternion()
    
    rby1_left_rot = Rotation.from_quat([rby1_left_quat[1], rby1_left_quat[2], rby1_left_quat[3], rby1_left_quat[0]])
    rby1_right_rot = Rotation.from_quat([rby1_right_quat[1], rby1_right_quat[2], rby1_right_quat[3], rby1_right_quat[0]])
    
    print(f"Left hand:")
    print(f"  Position: {rby1_left_pos}")
    print(f"  Euler: {rby1_left_rot.as_euler('xyz', degrees=True)}")
    
    print(f"Right hand:")
    print(f"  Position: {rby1_right_pos}")
    print(f"  Euler: {rby1_right_rot.as_euler('xyz', degrees=True)}")
    
    # Check joint values
    qpos = rby1_env.robot._mojo.physics.data.qpos
    torso = qpos[11:17]
    print(f"\nTorso joints: {[f'{x:.4f}' for x in torso]}")
    
    # Calculate errors
    print("\n3. POSITION COMPARISON:")
    print("-" * 60)
    left_pos_error = np.linalg.norm(rby1_left_pos - h1_left_pos) * 1000
    right_pos_error = np.linalg.norm(rby1_right_pos - h1_right_pos) * 1000
    
    print(f"Position errors:")
    print(f"  Left:  {left_pos_error:.2f}mm")
    print(f"  Right: {right_pos_error:.2f}mm")
    
    # Calculate orientation errors
    print("\n4. ORIENTATION COMPARISON:")
    print("-" * 60)
    
    left_relative_rot = h1_left_rot.inv() * rby1_left_rot
    right_relative_rot = h1_right_rot.inv() * rby1_right_rot
    
    left_angle_diff = np.linalg.norm(left_relative_rot.as_rotvec()) * 180 / np.pi
    right_angle_diff = np.linalg.norm(right_relative_rot.as_rotvec()) * 180 / np.pi
    
    print(f"Total rotation difference:")
    print(f"  Left:  {left_angle_diff:.1f}°")
    print(f"  Right: {right_angle_diff:.1f}°")
    
    # Assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT:")
    print("=" * 80)
    
    pos_match = max(left_pos_error, right_pos_error) < 5.0
    orient_match = max(left_angle_diff, right_angle_diff) < 5.0
    
    if pos_match and orient_match:
        print("✅ EXCELLENT! Both position and orientation match H1.")
        print(f"   Position: max {max(left_pos_error, right_pos_error):.2f}mm < 5mm")
        print(f"   Orientation: max {max(left_angle_diff, right_angle_diff):.1f}° < 5°")
    elif pos_match:
        print("✅ Position matches H1 well")
        print(f"⚠️  Orientation difference: max {max(left_angle_diff, right_angle_diff):.1f}°")
    elif orient_match:
        print("✅ Orientation matches H1 well")
        print(f"⚠️  Position difference: max {max(left_pos_error, right_pos_error):.2f}mm")
    else:
        print("⚠️  Differences in both position and orientation")
        print(f"   Position: max {max(left_pos_error, right_pos_error):.2f}mm")
        print(f"   Orientation: max {max(left_angle_diff, right_angle_diff):.1f}°")
    
    rby1_env.close()

if __name__ == "__main__":
    main()