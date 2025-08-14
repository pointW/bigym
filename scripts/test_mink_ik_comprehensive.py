#!/usr/bin/env python3
"""Comprehensive test suite for Mink H1 IK solver.

This test covers:
1. IK solver accuracy in isolation (no Cartesian action mode)
2. Integration with CartesianActionModeDirect
3. Movement in different directions
4. Pelvis movement handling
5. Comparison with original H1UpperBodyIK solver
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pyquaternion import Quaternion
import time

from bigym.action_modes import JointPositionActionMode
from bigym.cartesian_action_mode_direct import CartesianActionModeDirect
from bigym.envs.reach_target import ReachTarget
from bigym.const import HandSide
from vr.ik.h1_upper_body_ik import H1UpperBodyIK, Pose

# Import the mink IK solver
try:
    from bigym.ik.mink_h1_ik import MinkH1UpperBodyIK
    MINK_IK_AVAILABLE = True
except ImportError:
    print("Warning: MinkH1UpperBodyIK not available")
    MINK_IK_AVAILABLE = False


class TestMinkIKSolver:
    """Test suite for Mink H1 IK solver."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {}
        
    def test_ik_solver_isolation(self):
        """Test 1: IK solver accuracy in isolation (no Cartesian action mode)."""
        print("="*80)
        print("TEST 1: IK SOLVER IN ISOLATION")
        print("="*80)
        print("Testing IK solver accuracy without Cartesian action mode integration.\n")
        
        if not MINK_IK_AVAILABLE:
            print("⚠️ Mink IK solver not available, skipping test")
            return None
        
        # Create SEPARATE environments for each solver to avoid state contamination
        env_original = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=50,
            render_mode=None,
        )
        
        env_mink = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=50,
            render_mode=None,
        )
        
        # Create solvers with their own environments
        original_solver = H1UpperBodyIK(env_original)
        mink_solver = MinkH1UpperBodyIK(env_mink)
        
        # Reset both environments and get pelvis positions
        env_original.reset()
        env_mink.reset()
        
        # Get pelvis from original environment
        pelvis_original = env_original.robot.pelvis
        actual_pelvis_pos_orig = pelvis_original.get_position()
        actual_pelvis_quat_orig = Quaternion(pelvis_original.get_quaternion())
        
        # Get pelvis from mink environment  
        pelvis_mink = env_mink.robot.pelvis
        actual_pelvis_pos_mink = pelvis_mink.get_position()
        actual_pelvis_quat_mink = Quaternion(pelvis_mink.get_quaternion())
        
        print(f"Using actual robot pelvis position (original env): {actual_pelvis_pos_orig}")
        print(f"Pelvis height (original): {actual_pelvis_pos_orig[2]:.3f}m")
        print(f"Pelvis height (mink): {actual_pelvis_pos_mink[2]:.3f}m")
        
        # Test configuration with CORRECT pelvis for each environment
        pelvis_pose_orig = Pose(actual_pelvis_pos_orig, actual_pelvis_quat_orig)
        pelvis_pose_mink = Pose(actual_pelvis_pos_mink, actual_pelvis_quat_mink)
        
        initial_left = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
        initial_right = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
        
        # Set initial joint configuration in BOTH robots
        # Original environment
        for i in range(5):
            actuator = env_original.robot.limb_actuators[i]
            if actuator.joint:
                joint = env_original.mojo.physics.bind(actuator.joint)
                joint.qpos = initial_left[i]
        
        for i in range(5):
            actuator = env_original.robot.limb_actuators[5 + i]
            if actuator.joint:
                joint = env_original.mojo.physics.bind(actuator.joint)
                joint.qpos = initial_right[i]
        
        env_original.mojo.physics.forward()
        
        # Mink environment
        for i in range(5):
            actuator = env_mink.robot.limb_actuators[i]
            if actuator.joint:
                joint = env_mink.mojo.physics.bind(actuator.joint)
                joint.qpos = initial_left[i]
        
        for i in range(5):
            actuator = env_mink.robot.limb_actuators[5 + i]
            if actuator.joint:
                joint = env_mink.mojo.physics.bind(actuator.joint)
                joint.qpos = initial_right[i]
        
        env_mink.mojo.physics.forward()
        
        # Get initial end-effector positions from both robots
        left_site_orig = env_original.robot._wrist_sites[HandSide.LEFT]
        right_site_orig = env_original.robot._wrist_sites[HandSide.RIGHT]
        initial_left_pos_orig = left_site_orig.get_position().copy()
        initial_right_pos_orig = right_site_orig.get_position().copy()
        
        left_site_mink = env_mink.robot._wrist_sites[HandSide.LEFT]
        right_site_mink = env_mink.robot._wrist_sites[HandSide.RIGHT]
        initial_left_pos_mink = left_site_mink.get_position().copy()
        initial_right_pos_mink = right_site_mink.get_position().copy()
        
        print(f"\nInitial EE positions (original env):")
        print(f"  Left:  {initial_left_pos_orig}")
        print(f"  Right: {initial_right_pos_orig}")
        print(f"\nInitial EE positions (mink env):")
        print(f"  Left:  {initial_left_pos_mink}")
        print(f"  Right: {initial_right_pos_mink}")
        
        # Test different target distances
        test_distances = [0.01, 0.02, 0.05, 0.10, 0.15]
        
        results = {"original": [], "mink": []}
        
        for distance in test_distances:
            print(f"\n{'='*60}")
            print(f"Testing {distance*1000:.0f}mm forward reach")
            print(f"{'='*60}")
            
            # Set targets for both environments (using their respective initial positions)
            target_left_pos_orig = initial_left_pos_orig + np.array([distance, 0, 0])
            target_right_pos_orig = initial_right_pos_orig + np.array([distance, 0, 0])
            
            target_left_pos_mink = initial_left_pos_mink + np.array([distance, 0, 0])
            target_right_pos_mink = initial_right_pos_mink + np.array([distance, 0, 0])
            
            target_left_orig = Pose(target_left_pos_orig, Quaternion(w=1, x=0, y=0, z=0))
            target_right_orig = Pose(target_right_pos_orig, Quaternion(w=1, x=0, y=0, z=0))
            
            target_left_mink = Pose(target_left_pos_mink, Quaternion(w=1, x=0, y=0, z=0))
            target_right_mink = Pose(target_right_pos_mink, Quaternion(w=1, x=0, y=0, z=0))
            
            # Reset joints in Original environment
            for i in range(5):
                actuator = env_original.robot.limb_actuators[i]
                if actuator.joint:
                    joint = env_original.mojo.physics.bind(actuator.joint)
                    joint.qpos = initial_left[i]
            
            for i in range(5):
                actuator = env_original.robot.limb_actuators[5 + i]
                if actuator.joint:
                    joint = env_original.mojo.physics.bind(actuator.joint)
                    joint.qpos = initial_right[i]
            
            env_original.mojo.physics.forward()
            
            # Test Original solver
            try:
                original_solution = original_solver.solve(
                    pelvis_pose_orig, initial_left, initial_right,
                    target_left_orig, target_right_orig
                )
                
                # Apply to original environment robot
                for i in range(10):
                    if i < len(original_solution):
                        actuator = env_original.robot.limb_actuators[i]
                        if actuator.joint:
                            joint = env_original.mojo.physics.bind(actuator.joint)
                            joint.qpos = original_solution[i]
                
                env_original.mojo.physics.forward()
                
                orig_left_achieved = left_site_orig.get_position().copy()
                orig_right_achieved = right_site_orig.get_position().copy()
                
                orig_left_error = np.linalg.norm(orig_left_achieved - target_left_pos_orig)
                orig_right_error = np.linalg.norm(orig_right_achieved - target_right_pos_orig)
                orig_avg_error = (orig_left_error + orig_right_error) / 2
                
            except Exception as e:
                print(f"  Original solver exception: {e}")
                orig_avg_error = float('inf')
            
            # Reset joints in Mink environment
            for i in range(5):
                actuator = env_mink.robot.limb_actuators[i]
                if actuator.joint:
                    joint = env_mink.mojo.physics.bind(actuator.joint)
                    joint.qpos = initial_left[i]
            
            for i in range(5):
                actuator = env_mink.robot.limb_actuators[5 + i]
                if actuator.joint:
                    joint = env_mink.mojo.physics.bind(actuator.joint)
                    joint.qpos = initial_right[i]
            
            env_mink.mojo.physics.forward()
            
            # Test Mink solver
            try:
                mink_solution = mink_solver.solve(
                    pelvis_pose_mink, initial_left, initial_right,
                    target_left_mink, target_right_mink
                )
                
                # Apply to mink environment robot
                for i in range(10):
                    if i < len(mink_solution):
                        actuator = env_mink.robot.limb_actuators[i]
                        if actuator.joint:
                            joint = env_mink.mojo.physics.bind(actuator.joint)
                            joint.qpos = mink_solution[i]
                
                env_mink.mojo.physics.forward()
                
                mink_left_achieved = left_site_mink.get_position().copy()
                mink_right_achieved = right_site_mink.get_position().copy()
                
                mink_left_error = np.linalg.norm(mink_left_achieved - target_left_pos_mink)
                mink_right_error = np.linalg.norm(mink_right_achieved - target_right_pos_mink)
                mink_avg_error = (mink_left_error + mink_right_error) / 2
                
            except Exception as e:
                print(f"  Mink solver exception: {e}")
                mink_avg_error = float('inf')
            
            # Store results
            results["original"].append(orig_avg_error)
            results["mink"].append(mink_avg_error)
            
            # Print comparison
            print(f"  Original: {orig_avg_error*1000:.2f}mm")
            print(f"  Mink:     {mink_avg_error*1000:.2f}mm")
            
            if mink_avg_error < orig_avg_error:
                improvement = (orig_avg_error - mink_avg_error) / orig_avg_error * 100
                print(f"  ✅ Mink is {improvement:.1f}% better!")
            elif mink_avg_error > orig_avg_error * 1.5:
                print(f"  ❌ Mink is {(mink_avg_error/orig_avg_error - 1)*100:.1f}% worse")
            else:
                print(f"  ⚠️ Comparable performance")
        
        # Close both environments
        env_original.close()
        env_mink.close()
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - IK Solver in Isolation")
        print(f"{'='*60}")
        orig_avg = np.mean([e for e in results["original"] if e < float('inf')])
        mink_avg = np.mean([e for e in results["mink"] if e < float('inf')])
        
        print(f"Average error - Original: {orig_avg*1000:.2f}mm")
        print(f"Average error - Mink:     {mink_avg*1000:.2f}mm")
        
        if mink_avg < orig_avg:
            print(f"\n🎉 Mink solver is {(1 - mink_avg/orig_avg)*100:.0f}% better on average!")
        
        self.test_results["isolation"] = results
        return results
    
    def test_cartesian_direct_integration(self):
        """Test 2: Integration with CartesianActionModeDirect."""
        print("\n" + "="*80)
        print("TEST 2: CARTESIAN ACTION MODE DIRECT INTEGRATION")
        print("="*80)
        print("Testing IK solvers integrated with CartesianActionModeDirect.")
        print("Direct mode bypasses PD controller for pure IK performance.\n")
        
        if not MINK_IK_AVAILABLE:
            print("⚠️ Mink IK solver not available, skipping integration test")
            return None
        
        # Create custom Direct action modes using different IK solvers
        class CartesianActionModeDirectMink(CartesianActionModeDirect):
            """Direct Cartesian action mode using Mink IK solver."""
            
            def _initialize_ik_solver(self):
                """Initialize the Mink IK solver."""
                class MockEnv:
                    def __init__(self, robot, mojo):
                        self.robot = robot
                        self.mojo = mojo
                
                mock_env = MockEnv(self._robot, self._mojo)
                self._ik_solver = MinkH1UpperBodyIK(mock_env)
        
        # Test both solvers with Direct mode
        modes = [
            ("Original IK (Direct)", CartesianActionModeDirect(floating_base=True)),
            ("Mink IK (Direct)", CartesianActionModeDirectMink(floating_base=True)),
        ]
        
        test_distances = [0.01, 0.02, 0.05, 0.10]
        results = {}
        
        for mode_name, action_mode in modes:
            print(f"\n{'='*60}")
            print(f"Testing: {mode_name}")
            print(f"{'='*60}")
            
            env = ReachTarget(
                action_mode=action_mode,
                control_frequency=50,
                render_mode=None,
            )
            
            errors = []
            execution_times = []
            
            for distance in test_distances:
                env.reset()
                
                # Get initial positions
                left_site = env.robot._wrist_sites[HandSide.LEFT]
                right_site = env.robot._wrist_sites[HandSide.RIGHT]
                initial_left = left_site.get_position()
                initial_right = right_site.get_position()
                
                # Set targets
                target_left = initial_left + np.array([distance, 0, 0])
                target_right = initial_right + np.array([distance, 0, 0])
                
                # Create action
                base_dof = env.robot.floating_base.dof_amount if action_mode.floating_base else 0
                
                # Time the execution
                start_time = time.time()
                
                action = env.action_mode.poses_to_action(
                    Pose(target_left, Quaternion(w=1, x=0, y=0, z=0)),
                    Pose(target_right, Quaternion(w=1, x=0, y=0, z=0)),
                    base_action=np.zeros(base_dof),
                    gripper_action=np.zeros(2)
                )
                
                # Execute
                env.step(action)
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Measure error
                achieved_left = left_site.get_position()
                achieved_right = right_site.get_position()
                
                left_error = np.linalg.norm(achieved_left - target_left)
                right_error = np.linalg.norm(achieved_right - target_right)
                avg_error = (left_error + right_error) / 2
                
                errors.append(avg_error)
                
                status = "✅" if avg_error < 0.001 else "⚠️" if avg_error < 0.01 else "❌"
                print(f"  {distance*1000:3.0f}mm reach: {status} Error={avg_error*1000:.3f}mm, Time={execution_time*1000:.1f}ms")
            
            env.close()
            results[mode_name] = {
                "errors": errors,
                "times": execution_times,
                "avg_error": np.mean(errors),
                "avg_time": np.mean(execution_times)
            }
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - Cartesian Direct Mode Integration")
        print(f"{'='*60}")
        
        for mode_name, data in results.items():
            print(f"\n{mode_name}:")
            print(f"  Average error: {data['avg_error']*1000:.3f}mm")
            print(f"  Average time:  {data['avg_time']*1000:.1f}ms")
        
        # Compare performance
        if "Mink IK (Direct)" in results and "Original IK (Direct)" in results:
            mink_data = results["Mink IK (Direct)"]
            orig_data = results["Original IK (Direct)"]
            
            error_improvement = (1 - mink_data['avg_error']/orig_data['avg_error']) * 100
            time_diff = (mink_data['avg_time'] - orig_data['avg_time']) / orig_data['avg_time'] * 100
            
            print(f"\nComparison:")
            if error_improvement > 0:
                print(f"  ✅ Mink IK is {error_improvement:.0f}% more accurate")
            else:
                print(f"  ❌ Mink IK is {-error_improvement:.0f}% less accurate")
            
            if abs(time_diff) < 20:
                print(f"  ✅ Similar execution time ({time_diff:+.0f}%)")
            elif time_diff < 0:
                print(f"  ✅ Mink IK is {-time_diff:.0f}% faster")
            else:
                print(f"  ⚠️ Mink IK is {time_diff:.0f}% slower")
        
        self.test_results["cartesian_direct"] = results
        return results
    
    def test_movement_directions(self):
        """Test 3: Accuracy in different movement directions."""
        print("\n" + "="*80)
        print("TEST 3: MOVEMENT DIRECTIONS")
        print("="*80)
        print("Testing IK solver accuracy in different movement directions.\n")
        
        if not MINK_IK_AVAILABLE:
            print("⚠️ Mink IK solver not available, skipping test")
            return None
        
        env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=50,
            render_mode=None,
        )
        
        original_solver = H1UpperBodyIK(env)
        mink_solver = MinkH1UpperBodyIK(env)
        
        env.reset()
        
        # Use actual pelvis
        pelvis = env.robot.pelvis
        pelvis_pose = Pose(pelvis.get_position(), Quaternion(pelvis.get_quaternion()))
        
        # Initial configuration
        initial_joints = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
        
        # Set initial joints
        for i in range(5):
            for j, actuator in enumerate([env.robot.limb_actuators[i], 
                                         env.robot.limb_actuators[5+i]]):
                if actuator.joint:
                    joint = env.mojo.physics.bind(actuator.joint)
                    joint.qpos = initial_joints[i]
        
        env.mojo.physics.forward()
        
        # Get initial positions
        left_site = env.robot._wrist_sites[HandSide.LEFT]
        initial_left_pos = left_site.get_position().copy()
        initial_right_pos = env.robot._wrist_sites[HandSide.RIGHT].get_position().copy()
        
        # Test directions
        test_vectors = [
            ("Forward (X)", np.array([0.05, 0, 0])),
            ("Left (Y)", np.array([0, 0.05, 0])),
            ("Up (Z)", np.array([0, 0, 0.05])),
            ("Backward (-X)", np.array([-0.05, 0, 0])),
            ("Right (-Y)", np.array([0, -0.05, 0])),
            ("Down (-Z)", np.array([0, 0, -0.05])),
            ("Diagonal XY", np.array([0.035, 0.035, 0])),
            ("Diagonal XZ", np.array([0.035, 0, 0.035])),
            ("Diagonal YZ", np.array([0, 0.035, 0.035])),
            ("Diagonal XYZ", np.array([0.029, 0.029, 0.029])),
        ]
        
        results = {"original": {}, "mink": {}}
        
        for direction_name, movement in test_vectors:
            print(f"\n{direction_name}:")
            
            # Set targets
            target_left_pos = initial_left_pos + movement
            target_right_pos = initial_right_pos + movement
            
            target_left = Pose(target_left_pos, Quaternion(w=1, x=0, y=0, z=0))
            target_right = Pose(target_right_pos, Quaternion(w=1, x=0, y=0, z=0))
            
            # Test Original solver
            try:
                original_solution = original_solver.solve(
                    pelvis_pose, initial_joints, initial_joints,
                    target_left, target_right
                )
                
                # Apply to robot
                for i in range(5):
                    actuator = env.robot.limb_actuators[i]
                    if actuator.joint:
                        joint = env.mojo.physics.bind(actuator.joint)
                        joint.qpos = original_solution[i]
                
                env.mojo.physics.forward()
                orig_achieved = left_site.get_position().copy()
                orig_error = np.linalg.norm(orig_achieved - target_left_pos)
                
            except Exception:
                orig_error = float('inf')
            
            # Reset for Mink test
            for i in range(5):
                for j, actuator in enumerate([env.robot.limb_actuators[i], 
                                             env.robot.limb_actuators[5+i]]):
                    if actuator.joint:
                        joint = env.mojo.physics.bind(actuator.joint)
                        joint.qpos = initial_joints[i]
            
            env.mojo.physics.forward()
            
            # Test Mink solver
            try:
                mink_solution = mink_solver.solve(
                    pelvis_pose, initial_joints, initial_joints,
                    target_left, target_right
                )
                
                # Apply to robot
                for i in range(5):
                    actuator = env.robot.limb_actuators[i]
                    if actuator.joint:
                        joint = env.mojo.physics.bind(actuator.joint)
                        joint.qpos = mink_solution[i]
                
                env.mojo.physics.forward()
                mink_achieved = left_site.get_position().copy()
                mink_error = np.linalg.norm(mink_achieved - target_left_pos)
                
            except Exception:
                mink_error = float('inf')
            
            results["original"][direction_name] = orig_error
            results["mink"][direction_name] = mink_error
            
            print(f"  Original: {orig_error*1000:.2f}mm")
            print(f"  Mink:     {mink_error*1000:.2f}mm", end="")
            
            if mink_error < orig_error * 0.5:
                print(" ✅")
            elif mink_error < orig_error * 1.5:
                print(" ⚠️")
            else:
                print(" ❌")
        
        env.close()
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - Movement Directions")
        print(f"{'='*60}")
        
        orig_errors = list(results["original"].values())
        mink_errors = list(results["mink"].values())
        
        orig_avg = np.mean([e for e in orig_errors if e < float('inf')])
        mink_avg = np.mean([e for e in mink_errors if e < float('inf')])
        
        print(f"Average error - Original: {orig_avg*1000:.2f}mm")
        print(f"Average error - Mink:     {mink_avg*1000:.2f}mm")
        
        if mink_avg < orig_avg:
            print(f"\n✅ Mink solver is {(1 - mink_avg/orig_avg)*100:.0f}% better on average")
        
        # Find best and worst directions for each solver
        best_orig = min(results["original"].items(), key=lambda x: x[1])
        worst_orig = max(results["original"].items(), key=lambda x: x[1] if x[1] < float('inf') else 0)
        best_mink = min(results["mink"].items(), key=lambda x: x[1])
        worst_mink = max(results["mink"].items(), key=lambda x: x[1] if x[1] < float('inf') else 0)
        
        print(f"\nBest directions:")
        print(f"  Original: {best_orig[0]} ({best_orig[1]*1000:.2f}mm)")
        print(f"  Mink:     {best_mink[0]} ({best_mink[1]*1000:.2f}mm)")
        
        print(f"\nWorst directions:")
        print(f"  Original: {worst_orig[0]} ({worst_orig[1]*1000:.2f}mm)")
        print(f"  Mink:     {worst_mink[0]} ({worst_mink[1]*1000:.2f}mm)")
        
        self.test_results["directions"] = results
        return results
    
    def test_pelvis_movement(self):
        """Test 4: IK accuracy with different pelvis poses."""
        print("\n" + "="*80)
        print("TEST 4: PELVIS MOVEMENT HANDLING")
        print("="*80)
        print("Testing IK solver accuracy with different pelvis positions/orientations.\n")
        
        if not MINK_IK_AVAILABLE:
            print("⚠️ Mink IK solver not available, skipping test")
            return None
        
        env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=50,
            render_mode=None,
        )
        
        original_solver = H1UpperBodyIK(env)
        mink_solver = MinkH1UpperBodyIK(env)
        
        env.reset()
        
        # Base pelvis configuration
        base_pelvis_pos = env.robot.pelvis.get_position()
        base_pelvis_quat = Quaternion(env.robot.pelvis.get_quaternion())
        
        initial_joints = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
        
        # Test different pelvis configurations
        pelvis_configs = [
            ("Nominal", np.array([0, 0, 0]), Quaternion(w=1, x=0, y=0, z=0)),
            ("Forward 5cm", np.array([0.05, 0, 0]), Quaternion(w=1, x=0, y=0, z=0)),
            ("Left 5cm", np.array([0, 0.05, 0]), Quaternion(w=1, x=0, y=0, z=0)),
            ("Up 3cm", np.array([0, 0, 0.03]), Quaternion(w=1, x=0, y=0, z=0)),
            ("Yaw 10°", np.array([0, 0, 0]), Quaternion(axis=[0, 0, 1], degrees=10)),
            ("Pitch 5°", np.array([0, 0, 0]), Quaternion(axis=[0, 1, 0], degrees=5)),
            ("Roll 5°", np.array([0, 0, 0]), Quaternion(axis=[1, 0, 0], degrees=5)),
            ("Combined", np.array([0.03, 0.02, 0.01]), Quaternion(axis=[0, 0, 1], degrees=5)),
        ]
        
        results = {"original": {}, "mink": {}}
        
        for config_name, pelvis_offset, pelvis_rotation in pelvis_configs:
            print(f"\n{config_name}:")
            
            # Create pelvis pose
            pelvis_pos = base_pelvis_pos + pelvis_offset
            pelvis_quat = base_pelvis_quat * pelvis_rotation
            pelvis_pose = Pose(pelvis_pos, pelvis_quat)
            
            # Set initial joints
            for i in range(5):
                for j, actuator in enumerate([env.robot.limb_actuators[i], 
                                             env.robot.limb_actuators[5+i]]):
                    if actuator.joint:
                        joint = env.mojo.physics.bind(actuator.joint)
                        joint.qpos = initial_joints[i]
            
            env.mojo.physics.forward()
            
            # Get current end-effector positions
            left_site = env.robot._wrist_sites[HandSide.LEFT]
            right_site = env.robot._wrist_sites[HandSide.RIGHT]
            current_left = left_site.get_position()
            current_right = right_site.get_position()
            
            # Set targets (50mm forward from current position)
            target_left_pos = current_left + np.array([0.05, 0, 0])
            target_right_pos = current_right + np.array([0.05, 0, 0])
            
            target_left = Pose(target_left_pos, Quaternion(w=1, x=0, y=0, z=0))
            target_right = Pose(target_right_pos, Quaternion(w=1, x=0, y=0, z=0))
            
            # Test Original solver
            try:
                original_solution = original_solver.solve(
                    pelvis_pose, initial_joints, initial_joints,
                    target_left, target_right
                )
                
                for i in range(5):
                    actuator = env.robot.limb_actuators[i]
                    if actuator.joint:
                        joint = env.mojo.physics.bind(actuator.joint)
                        joint.qpos = original_solution[i]
                
                env.mojo.physics.forward()
                orig_achieved = left_site.get_position()
                orig_error = np.linalg.norm(orig_achieved - target_left_pos)
                
            except Exception:
                orig_error = float('inf')
            
            # Reset and test Mink
            for i in range(5):
                for j, actuator in enumerate([env.robot.limb_actuators[i], 
                                             env.robot.limb_actuators[5+i]]):
                    if actuator.joint:
                        joint = env.mojo.physics.bind(actuator.joint)
                        joint.qpos = initial_joints[i]
            
            env.mojo.physics.forward()
            
            try:
                mink_solution = mink_solver.solve(
                    pelvis_pose, initial_joints, initial_joints,
                    target_left, target_right
                )
                
                for i in range(5):
                    actuator = env.robot.limb_actuators[i]
                    if actuator.joint:
                        joint = env.mojo.physics.bind(actuator.joint)
                        joint.qpos = mink_solution[i]
                
                env.mojo.physics.forward()
                mink_achieved = left_site.get_position()
                mink_error = np.linalg.norm(mink_achieved - target_left_pos)
                
            except Exception:
                mink_error = float('inf')
            
            results["original"][config_name] = orig_error
            results["mink"][config_name] = mink_error
            
            print(f"  Original: {orig_error*1000:.2f}mm")
            print(f"  Mink:     {mink_error*1000:.2f}mm", end="")
            
            if mink_error < orig_error * 0.5:
                print(" ✅")
            elif mink_error < orig_error * 1.5:
                print(" ⚠️")
            else:
                print(" ❌")
        
        env.close()
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - Pelvis Movement Handling")
        print(f"{'='*60}")
        
        orig_avg = np.mean([e for e in results["original"].values() if e < float('inf')])
        mink_avg = np.mean([e for e in results["mink"].values() if e < float('inf')])
        
        print(f"Average error - Original: {orig_avg*1000:.2f}mm")
        print(f"Average error - Mink:     {mink_avg*1000:.2f}mm")
        
        if mink_avg < orig_avg:
            print(f"\n✅ Mink handles pelvis movement {(1 - mink_avg/orig_avg)*100:.0f}% better")
        
        self.test_results["pelvis"] = results
        return results
    
    def run_all_tests(self):
        """Run all tests and provide summary."""
        print("="*80)
        print("COMPREHENSIVE TEST SUITE FOR MINK H1 IK SOLVER")
        print("="*80)
        print("This test suite covers:")
        print("1. IK solver accuracy in isolation")
        print("2. Integration with CartesianActionModeDirect")
        print("3. Movement in different directions")
        print("4. Pelvis movement handling")
        print("\n")
        
        test_results = {}
        
        # Run tests
        try:
            test_results["isolation"] = self.test_ik_solver_isolation()
        except Exception as e:
            print(f"\n❌ Test 1 failed with exception: {e}")
            test_results["isolation"] = None
        
        try:
            test_results["cartesian_direct"] = self.test_cartesian_direct_integration()
        except Exception as e:
            print(f"\n❌ Test 2 failed with exception: {e}")
            test_results["cartesian_direct"] = None
        
        try:
            test_results["directions"] = self.test_movement_directions()
        except Exception as e:
            print(f"\n❌ Test 3 failed with exception: {e}")
            test_results["directions"] = None
        
        try:
            test_results["pelvis"] = self.test_pelvis_movement()
        except Exception as e:
            print(f"\n❌ Test 4 failed with exception: {e}")
            test_results["pelvis"] = None
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        if not MINK_IK_AVAILABLE:
            print("\n⚠️ Mink IK solver not available for testing")
        else:
            passed = sum(1 for v in test_results.values() if v is not None)
            total = len(test_results)
            
            print(f"\nTests completed: {passed}/{total}")
            
            # Calculate overall performance
            if "isolation" in self.test_results and self.test_results["isolation"]:
                orig_errors = self.test_results["isolation"].get("original", [])
                mink_errors = self.test_results["isolation"].get("mink", [])
                
                if orig_errors and mink_errors:
                    orig_avg = np.mean([e for e in orig_errors if e < float('inf')])
                    mink_avg = np.mean([e for e in mink_errors if e < float('inf')])
                    
                    if mink_avg < orig_avg:
                        print(f"\n🎉 Mink IK solver shows {(1 - mink_avg/orig_avg)*100:.0f}% improvement!")
                    elif mink_avg == orig_avg:
                        print("\n⚠️ Mink IK solver performs similarly to original")
                    else:
                        print("\n❌ Mink IK solver performs worse than original")
            
            # Cartesian Direct mode performance
            if "cartesian_direct" in self.test_results and self.test_results["cartesian_direct"]:
                mink_direct = self.test_results["cartesian_direct"].get("Mink IK (Direct)")
                orig_direct = self.test_results["cartesian_direct"].get("Original IK (Direct)")
                
                if mink_direct and orig_direct:
                    mink_error = mink_direct["avg_error"]
                    orig_error = orig_direct["avg_error"]
                    
                    print(f"\nDirect Mode Integration:")
                    print(f"  Original: {orig_error*1000:.3f}mm average error")
                    print(f"  Mink:     {mink_error*1000:.3f}mm average error")
                    
                    if mink_error < orig_error:
                        print(f"  ✅ Mink is {(1 - mink_error/orig_error)*100:.0f}% better in Direct mode")
        
        print("\nKey insights:")
        print("  - Mink uses optimization-based IK with proper MuJoCo synchronization")
        print("  - Direct mode bypasses PD controllers showing pure IK performance")
        print("  - Sub-millimeter accuracy is achievable with proper configuration")
        print("  - Performance depends on proper task weights and solver parameters")
        
        return test_results


def main():
    """Run the comprehensive test suite."""
    tester = TestMinkIKSolver()
    results = tester.run_all_tests()
    
    if not MINK_IK_AVAILABLE:
        print("\n" + "="*80)
        print("SETUP REQUIRED")
        print("="*80)
        print("\n1. Install mink: pip install mink")
        print("2. Ensure MinkH1UpperBodyIK is properly imported")
        print("3. Run this test again to verify performance")


if __name__ == "__main__":
    main()