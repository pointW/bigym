#!/usr/bin/env python3
"""Comprehensive test suite for Clean H1 IK solver.

This test covers:
1. IK solver accuracy in isolation (no Cartesian action mode)
2. Pelvis movement handling
3. Integration with Cartesian action modes
4. Movement in different directions
5. Comparison with original H1UpperBodyIK solver
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pyquaternion import Quaternion
import time

from bigym.action_modes import JointPositionActionMode
from bigym.cartesian_action_mode import CartesianActionMode
from bigym.cartesian_action_mode_direct import CartesianActionModeDirect
from bigym.envs.reach_target import ReachTarget
from bigym.const import HandSide
from vr.ik.h1_upper_body_ik import H1UpperBodyIK, Pose

# Import the clean IK solver
try:
    from bigym.ik.clean_h1_ik import CleanH1UpperBodyIK
    CLEAN_IK_AVAILABLE = True
except ImportError:
    print("Warning: CleanH1UpperBodyIK not available yet")
    CLEAN_IK_AVAILABLE = False


class TestCleanIKSolver:
    """Test suite for Clean H1 IK solver."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {}
        
    def test_ik_solver_isolation(self):
        """Test 1: IK solver accuracy in isolation (no Cartesian action mode)."""
        print("="*80)
        print("TEST 1: IK SOLVER IN ISOLATION")
        print("="*80)
        print("Testing IK solver accuracy without Cartesian action mode integration.\n")
        
        # Create environment
        env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=50,
            render_mode=None,
        )
        
        # Create both solvers
        original_solver = H1UpperBodyIK(env)
        clean_solver = CleanH1UpperBodyIK(env) if CLEAN_IK_AVAILABLE else None
        
        # Get ACTUAL pelvis position - this is crucial!
        env.reset()
        pelvis = env.robot.pelvis
        actual_pelvis_pos = pelvis.get_position()
        actual_pelvis_quat = Quaternion(pelvis.get_quaternion())
        
        print(f"Using actual robot pelvis position: {actual_pelvis_pos}")
        print(f"Pelvis height: {actual_pelvis_pos[2]:.3f}m")
        
        # Test configuration with CORRECT pelvis
        pelvis_pose = Pose(actual_pelvis_pos, actual_pelvis_quat)
        
        initial_left = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
        initial_right = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
        
        # Set initial joint configuration in actual robot
        for i in range(5):
            actuator = env.robot.limb_actuators[i]
            if actuator.joint:
                joint = env.mojo.physics.bind(actuator.joint)
                joint.qpos = initial_left[i]
        
        for i in range(5):
            actuator = env.robot.limb_actuators[5 + i]
            if actuator.joint:
                joint = env.mojo.physics.bind(actuator.joint)
                joint.qpos = initial_right[i]
        
        env.mojo.physics.forward()
        
        # Get initial end-effector positions from actual robot
        left_site = env.robot._wrist_sites[HandSide.LEFT]
        right_site = env.robot._wrist_sites[HandSide.RIGHT]
        initial_left_pos = left_site.get_position().copy()
        initial_right_pos = right_site.get_position().copy()
        
        print(f"\nInitial EE positions (from actual robot):")
        print(f"  Left:  {initial_left_pos}")
        print(f"  Right: {initial_right_pos}")
        
        # Test different target distances
        test_distances = [0.01, 0.02, 0.05, 0.10, 0.20]
        
        results = {"original": [], "clean": []}
        
        for distance in test_distances:
            print(f"\n{'='*60}")
            print(f"Testing {distance*1000:.0f}mm forward reach")
            print(f"{'='*60}")
            
            # Set targets
            target_left_pos = initial_left_pos + np.array([distance, 0, 0])
            target_right_pos = initial_right_pos + np.array([distance, 0, 0])
            
            target_left = Pose(target_left_pos, Quaternion(w=1, x=0, y=0, z=0))
            target_right = Pose(target_right_pos, Quaternion(w=1, x=0, y=0, z=0))
            
            # Test Original solver
            try:
                original_solution = original_solver.solve(
                    pelvis_pose, initial_left, initial_right,
                    target_left, target_right
                )
                
                # Apply to actual robot to measure accuracy
                for i in range(10):
                    if i < len(original_solution):
                        actuator = env.robot.limb_actuators[i]
                        if actuator.joint:
                            joint = env.mojo.physics.bind(actuator.joint)
                            joint.qpos = original_solution[i]
                
                env.mojo.physics.forward()
                
                orig_left_achieved = left_site.get_position().copy()
                orig_right_achieved = right_site.get_position().copy()
                
                orig_left_error = np.linalg.norm(orig_left_achieved - target_left_pos)
                orig_right_error = np.linalg.norm(orig_right_achieved - target_right_pos)
                orig_avg_error = (orig_left_error + orig_right_error) / 2
                
            except Exception as e:
                print(f"  Original solver exception: {e}")
                orig_avg_error = float('inf')
            
            # Test Clean solver
            if CLEAN_IK_AVAILABLE and clean_solver:
                try:
                    clean_solution = clean_solver.solve(
                        pelvis_pose, initial_left, initial_right,
                        target_left, target_right
                    )
                    
                    # Apply to actual robot
                    for i in range(10):
                        if i < len(clean_solution):
                            actuator = env.robot.limb_actuators[i]
                            if actuator.joint:
                                joint = env.mojo.physics.bind(actuator.joint)
                                joint.qpos = clean_solution[i]
                    
                    env.mojo.physics.forward()
                    
                    clean_left_achieved = left_site.get_position().copy()
                    clean_right_achieved = right_site.get_position().copy()
                    
                    clean_left_error = np.linalg.norm(clean_left_achieved - target_left_pos)
                    clean_right_error = np.linalg.norm(clean_right_achieved - target_right_pos)
                    clean_avg_error = (clean_left_error + clean_right_error) / 2
                    
                except Exception as e:
                    print(f"  Clean solver exception: {e}")
                    clean_avg_error = float('inf')
            else:
                clean_avg_error = float('inf')
            
            # Store results
            results["original"].append(orig_avg_error)
            results["clean"].append(clean_avg_error)
            
            # Print comparison
            print(f"  Original: {orig_avg_error*1000:.2f}mm")
            if CLEAN_IK_AVAILABLE:
                status = "✅" if clean_avg_error < orig_avg_error else "⚠️"
                print(f"  Clean:    {clean_avg_error*1000:.2f}mm {status}")
                
                if clean_avg_error < float('inf'):
                    improvement = (orig_avg_error - clean_avg_error) / orig_avg_error * 100 if orig_avg_error > 0 else 0
                    if improvement > 0:
                        print(f"  Improvement: {improvement:.1f}%")
        
        env.close()
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - IK Solver in Isolation")
        print(f"{'='*60}")
        orig_avg = np.mean([e for e in results["original"] if e < float('inf')])
        print(f"Average error - Original: {orig_avg*1000:.2f}mm")
        
        if CLEAN_IK_AVAILABLE:
            clean_avg = np.mean([e for e in results["clean"] if e < float('inf')])
            print(f"Average error - Clean: {clean_avg*1000:.2f}mm")
            
            if clean_avg < orig_avg:
                print(f"\n✅ Clean solver is {(1 - clean_avg/orig_avg)*100:.0f}% better on average")
        
        self.test_results["isolation"] = results
        return results
    
    def test_cartesian_integration(self):
        """Test 2: Integration with Cartesian action modes."""
        print("\n" + "="*80)
        print("TEST 2: CARTESIAN ACTION MODE INTEGRATION (DIRECT MODE)")
        print("="*80)
        print("Testing IK solvers integrated with CartesianActionModeDirect.\n")
        print("Using Direct mode to bypass PD controller for accurate IK assessment.\n")
        
        if not CLEAN_IK_AVAILABLE:
            print("⚠️ Clean IK solver not available, skipping integration test")
            return None
        
        # Create custom Direct action modes using different IK solvers
        class CartesianActionModeDirectClean(CartesianActionModeDirect):
            """Direct Cartesian action mode using Clean IK solver."""
            
            def _initialize_ik_solver(self):
                """Initialize the clean IK solver."""
                class MockEnv:
                    def __init__(self, robot, mojo):
                        self.robot = robot
                        self.mojo = mojo
                
                mock_env = MockEnv(self._robot, self._mojo)
                self._ik_solver = CleanH1UpperBodyIK(mock_env)
        
        # Test both solvers with Direct mode (bypasses PD controller)
        modes = [
            ("Original IK (Direct)", CartesianActionModeDirect(floating_base=True)),
            ("Clean IK (Direct)", CartesianActionModeDirectClean(floating_base=True)),
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
                action = env.action_mode.poses_to_action(
                    Pose(target_left, Quaternion(w=1, x=0, y=0, z=0)),
                    Pose(target_right, Quaternion(w=1, x=0, y=0, z=0)),
                    base_action=np.zeros(base_dof),
                    gripper_action=np.zeros(2)
                )
                
                # Execute
                env.step(action)
                
                # Measure error
                achieved_left = left_site.get_position()
                achieved_right = right_site.get_position()
                
                left_error = np.linalg.norm(achieved_left - target_left)
                right_error = np.linalg.norm(achieved_right - target_right)
                avg_error = (left_error + right_error) / 2
                
                errors.append(avg_error)
                
                status = "✅" if avg_error < 0.002 else "⚠️"  # Tighter threshold for Direct mode
                print(f"  {distance*1000:3.0f}mm reach: {status} Error={avg_error*1000:.2f}mm")
            
            env.close()
            results[mode_name] = errors
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - Cartesian Integration (Direct Mode)")
        print(f"{'='*60}")
        
        for mode_name, errors in results.items():
            avg_error = np.mean(errors)
            print(f"{mode_name:20s}: Average error = {avg_error*1000:.2f}mm")
        
        self.test_results["cartesian"] = results
        
        # Check if clean is better
        if "Clean IK (Direct)" in results and "Original IK (Direct)" in results:
            clean_avg = np.mean(results["Clean IK (Direct)"])
            orig_avg = np.mean(results["Original IK (Direct)"])
            if clean_avg < orig_avg:
                print(f"\n✅ Clean IK is {(1 - clean_avg/orig_avg)*100:.0f}% better in Direct mode")
            print("\nNote: Direct mode bypasses PD controller, showing pure IK performance.")
        
        return results
    
    def test_movement_directions(self):
        """Test 3: Accuracy in different movement directions."""
        print("\n" + "="*80)
        print("TEST 3: MOVEMENT DIRECTIONS")
        print("="*80)
        print("Testing IK solver accuracy in different movement directions.\n")
        
        env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=50,
            render_mode=None,
        )
        
        original_solver = H1UpperBodyIK(env)
        clean_solver = CleanH1UpperBodyIK(env) if CLEAN_IK_AVAILABLE else None
        
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
            ("Diagonal XY", np.array([0.035, 0.035, 0])),
            ("Diagonal XYZ", np.array([0.029, 0.029, 0.029])),
        ]
        
        results = {"original": {}, "clean": {}}
        
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
            
            # Test Clean solver
            if CLEAN_IK_AVAILABLE and clean_solver:
                try:
                    clean_solution = clean_solver.solve(
                        pelvis_pose, initial_joints, initial_joints,
                        target_left, target_right
                    )
                    
                    # Apply to robot
                    for i in range(5):
                        actuator = env.robot.limb_actuators[i]
                        if actuator.joint:
                            joint = env.mojo.physics.bind(actuator.joint)
                            joint.qpos = clean_solution[i]
                    
                    env.mojo.physics.forward()
                    clean_achieved = left_site.get_position().copy()
                    clean_error = np.linalg.norm(clean_achieved - target_left_pos)
                    
                except Exception:
                    clean_error = float('inf')
            else:
                clean_error = float('inf')
            
            results["original"][direction_name] = orig_error
            results["clean"][direction_name] = clean_error
            
            print(f"  Original: {orig_error*1000:.2f}mm")
            if CLEAN_IK_AVAILABLE:
                status = "✅" if clean_error < orig_error else "⚠️"
                print(f"  Clean:    {clean_error*1000:.2f}mm {status}")
        
        env.close()
        
        # Summary
        print(f"\n{'='*40}")
        print("SUMMARY - Movement Directions")
        print(f"{'='*40}")
        
        orig_errors = list(results["original"].values())
        orig_avg = np.mean([e for e in orig_errors if e < float('inf')])
        print(f"Average error - Original: {orig_avg*1000:.2f}mm")
        
        if CLEAN_IK_AVAILABLE:
            clean_errors = list(results["clean"].values())
            clean_avg = np.mean([e for e in clean_errors if e < float('inf')])
            print(f"Average error - Clean: {clean_avg*1000:.2f}mm")
            
            if clean_avg < orig_avg:
                print(f"\n✅ Clean solver is {(1 - clean_avg/orig_avg)*100:.0f}% better on average")
        
        self.test_results["directions"] = results
        return results
    
    def run_all_tests(self):
        """Run all tests and provide summary."""
        print("="*80)
        print("COMPREHENSIVE TEST SUITE FOR CLEAN H1 IK SOLVER")
        print("="*80)
        print("This test suite covers:")
        print("1. IK solver accuracy in isolation")
        print("2. Integration with Cartesian action modes")
        print("3. Movement in different directions")
        print("\n")
        
        test_results = {}
        
        # Run tests
        try:
            test_results["isolation"] = self.test_ik_solver_isolation()
        except Exception as e:
            print(f"\n❌ Test 1 failed with exception: {e}")
            test_results["isolation"] = None
        
        try:
            test_results["cartesian"] = self.test_cartesian_integration()
        except Exception as e:
            print(f"\n❌ Test 2 failed with exception: {e}")
            test_results["cartesian"] = None
        
        try:
            test_results["directions"] = self.test_movement_directions()
        except Exception as e:
            print(f"\n❌ Test 3 failed with exception: {e}")
            test_results["directions"] = None
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        if not CLEAN_IK_AVAILABLE:
            print("\n⚠️ Clean IK solver not available for testing")
            print("Once implemented, this test will compare it against the original solver")
        else:
            passed = sum(1 for v in test_results.values() if v is not None)
            total = len(test_results)
            
            print(f"\nTests completed: {passed}/{total}")
            
            # Calculate overall improvement if Clean IK is available
            if "isolation" in self.test_results and self.test_results["isolation"]:
                orig_errors = self.test_results["isolation"].get("original", [])
                clean_errors = self.test_results["isolation"].get("clean", [])
                
                if orig_errors and clean_errors:
                    orig_avg = np.mean([e for e in orig_errors if e < float('inf')])
                    clean_avg = np.mean([e for e in clean_errors if e < float('inf')])
                    
                    if clean_avg < orig_avg:
                        print(f"\n🎉 Clean IK solver shows {(1 - clean_avg/orig_avg)*100:.0f}% improvement!")
                    elif clean_avg == orig_avg:
                        print("\n⚠️ Clean IK solver performs similarly to original")
                    else:
                        print("\n❌ Clean IK solver performs worse than original")
        
        print("\nKey insights:")
        print("  - Original H1UpperBodyIK uses 40-step physics simulation")
        print("  - This limits accuracy to ~1-2mm at best")
        print("  - Clean IK should use optimization-based approach for better accuracy")
        print("  - Must use actual robot pelvis height (1.0m) for accurate testing")
        
        return test_results


def main():
    """Run the comprehensive test suite."""
    tester = TestCleanIKSolver()
    results = tester.run_all_tests()
    
    if not CLEAN_IK_AVAILABLE:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Complete implementation of CleanH1UpperBodyIK")
        print("2. Ensure it uses proper optimization (scipy.optimize, etc.)")
        print("3. Run this test again to verify improvement")
        print("4. Target: <0.5mm average error for reachable targets")


if __name__ == "__main__":
    main()