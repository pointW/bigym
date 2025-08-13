#!/usr/bin/env python3
"""Systematic failure analysis for Mink IK solver with pelvis movement.

Test 1: Gradual target progression - Start from initial pose, gradually move target away
Test 2: Distance scaling - Always reset to initial, gradually increase target distance  
Test 3: Pelvis movement - Test IK solving with different pelvis positions/orientations

This helps identify exactly where and why the Mink solver starts to fail.
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
from pyquaternion import Quaternion

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from vr.ik.mink_h1_ik import MinkH1IK
from vr.ik.h1_upper_body_ik import H1UpperBodyIK, Pose


def test_gradual_target_progression():
    """Test 1: Start from initial pose, gradually move target away (cumulative movement)."""
    print("=" * 80)
    print("TEST 1: GRADUAL TARGET PROGRESSION (Cumulative Movement)")
    print("=" * 80)
    print("Starting from initial pose, gradually moving target away step by step.")
    print("This tests cumulative error buildup and solver stability over sequential poses.\n")
    
    # Create environment and solvers
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    mink_solver = MinkH1IK(env)
    original_solver = H1UpperBodyIK(env, enable_full_6d_control=False)
    
    # Initial configuration
    pelvis_pose = Pose(
        position=np.array([0.0, 0.0, 0.98]),
        orientation=Quaternion(w=1, x=0, y=0, z=0)
    )
    
    initial_left = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
    initial_right = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
    
    # Get initial end-effector positions
    mink_solver._set_robot_state(pelvis_pose, initial_left, initial_right)
    mujoco.mj_fwdPosition(mink_solver._model, mink_solver._data)
    
    initial_left_pos = mink_solver._data.site_xpos[mink_solver._left_site_id].copy()
    initial_right_pos = mink_solver._data.site_xpos[mink_solver._right_site_id].copy()
    
    print(f"Initial positions:")
    print(f"  Left: {initial_left_pos}")
    print(f"  Right: {initial_right_pos}")
    
    # Define movement directions and step sizes
    test_directions = [
        ("Forward", np.array([1.0, 0.0, 0.0])),
        ("Up", np.array([0.0, 0.0, 1.0])),
        ("Left", np.array([0.0, 1.0, 0.0])),
        ("Forward-Up", np.array([1.0, 0.0, 1.0]) / np.sqrt(2)),
    ]
    
    step_size = 0.02  # 2cm steps
    max_steps = 20    # Up to 40cm movement
    
    for direction_name, direction in test_directions:
        print(f"\n{'='*60}")
        print(f"TESTING {direction_name.upper()} MOVEMENT")
        print(f"{'='*60}")
        
        # Reset to initial state for each direction
        current_left_joints = initial_left.copy()
        current_right_joints = initial_right.copy()
        current_left_pos = initial_left_pos.copy()
        current_right_pos = initial_right_pos.copy()
        
        failure_step = None
        
        for step in range(1, max_steps + 1):
            # Calculate new target positions (gradual progression)
            target_left_pos = initial_left_pos + direction * step * step_size
            target_right_pos = initial_right_pos + direction * step * step_size
            
            target_left = Pose(target_left_pos, Quaternion(w=1, x=0, y=0, z=0))
            target_right = Pose(target_right_pos, Quaternion(w=1, x=0, y=0, z=0))
            
            distance = step * step_size * 1000  # Convert to mm
            
            # Test Mink solver (using current joint state, not initial)
            try:
                mink_solution = mink_solver.solve(
                    pelvis_pose, current_left_joints, current_right_joints, 
                    target_left, target_right
                )
                
                # Calculate error using the fixed measurement system
                if hasattr(mink_solver, '_converged_full_state'):
                    left_pos, right_pos = mink_solver.get_converged_end_effector_positions()
                    left_error = np.linalg.norm(left_pos - target_left_pos)
                    right_error = np.linalg.norm(right_pos - target_right_pos)
                    converged = True
                else:
                    # Fallback if no convergence achieved
                    left_error = float('inf')
                    right_error = float('inf')
                    converged = False
                
                combined_error = (left_error + right_error) / 2
                
                # Test original solver for comparison
                try:
                    original_solution = original_solver.solve(
                        pelvis_pose, current_left_joints, current_right_joints[:4],  # Original uses 4 joints
                        target_left, target_right
                    )
                    
                    # Calculate original solver error
                    original_solver._set_robot_state(pelvis_pose, original_solution[:4], original_solution[4:])
                    orig_left_pos = original_solver._physics.named.data.site_xpos[f"h1/{original_solver._left_site_name}"]
                    orig_right_pos = original_solver._physics.named.data.site_xpos[f"h1/{original_solver._right_site_name}"]
                    
                    orig_left_error = np.linalg.norm(orig_left_pos - target_left_pos)
                    orig_right_error = np.linalg.norm(orig_right_pos - target_right_pos)
                    orig_combined_error = (orig_left_error + orig_right_error) / 2
                    
                except Exception as e:
                    orig_combined_error = float('inf')
                
                # Check for failure
                if not converged or combined_error > 0.05:  # 50mm threshold
                    failure_step = step
                    print(f"  Step {step:2d} ({distance:3.0f}mm): ❌ FAILURE - Mink error {combined_error*1000:.1f}mm")
                    break
                else:
                    # Update current joint state for next iteration
                    current_left_joints = mink_solution[:5]
                    current_right_joints = mink_solution[5:]
                    
                    success_indicator = "✅" if combined_error < orig_combined_error else "⚠️"
                    print(f"  Step {step:2d} ({distance:3.0f}mm): {success_indicator} Mink {combined_error*1000:.1f}mm vs Orig {orig_combined_error*1000:.1f}mm")
                
            except Exception as e:
                failure_step = step
                print(f"  Step {step:2d} ({distance:3.0f}mm): ❌ EXCEPTION - {str(e)[:50]}...")
                break
        
        if failure_step is None:
            print(f"  ✅ SUCCESS: Completed all {max_steps} steps ({max_steps * step_size * 1000:.0f}mm total)")
        else:
            failure_distance = failure_step * step_size * 1000
            print(f"  ❌ FAILED at step {failure_step} ({failure_distance:.0f}mm distance)")
    
    env.close()
    return


def test_distance_scaling():
    """Test 2: Always reset to initial, gradually increase target distance."""
    print(f"\n{'='*80}")
    print("TEST 2: DISTANCE SCALING (Always Reset to Initial)")
    print("=" * 80)
    print("Always reset to initial pose, gradually increase target distance.")
    print("This tests solver capability vs target distance without cumulative effects.\n")
    
    # Create environment and solvers
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    mink_solver = MinkH1IK(env)
    original_solver = H1UpperBodyIK(env, enable_full_6d_control=False)
    
    # Initial configuration
    pelvis_pose = Pose(
        position=np.array([0.0, 0.0, 0.98]),
        orientation=Quaternion(w=1, x=0, y=0, z=0)
    )
    
    initial_left = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
    initial_right = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
    
    # Get initial end-effector positions
    mink_solver._set_robot_state(pelvis_pose, initial_left, initial_right)
    mujoco.mj_fwdPosition(mink_solver._model, mink_solver._data)
    
    initial_left_pos = mink_solver._data.site_xpos[mink_solver._left_site_id].copy()
    initial_right_pos = mink_solver._data.site_xpos[mink_solver._right_site_id].copy()
    
    # Define test directions
    test_directions = [
        ("Forward", np.array([1.0, 0.0, 0.0])),
        ("Up", np.array([0.0, 0.0, 1.0])),
        ("Left", np.array([0.0, 1.0, 0.0])),
        ("Forward-Up", np.array([1.0, 0.0, 1.0]) / np.sqrt(2)),
        ("Complex", np.array([1.0, 0.5, 0.3]) / np.linalg.norm([1.0, 0.5, 0.3])),
    ]
    
    distances = np.linspace(0.01, 0.5, 25)  # 1cm to 50cm
    
    for direction_name, direction in test_directions:
        print(f"\n{'='*60}")
        print(f"TESTING {direction_name.upper()} DIRECTION")
        print(f"{'='*60}")
        
        failure_distance = None
        
        for distance in distances:
            # Calculate target positions (always from initial)
            target_left_pos = initial_left_pos + direction * distance
            target_right_pos = initial_right_pos + direction * distance
            
            target_left = Pose(target_left_pos, Quaternion(w=1, x=0, y=0, z=0))
            target_right = Pose(target_right_pos, Quaternion(w=1, x=0, y=0, z=0))
            
            distance_mm = distance * 1000
            
            # Test Mink solver (always from initial state)
            try:
                mink_solution = mink_solver.solve(
                    pelvis_pose, initial_left, initial_right, 
                    target_left, target_right
                )
                
                # Calculate error using the fixed measurement system
                if hasattr(mink_solver, '_converged_full_state'):
                    left_pos, right_pos = mink_solver.get_converged_end_effector_positions()
                    left_error = np.linalg.norm(left_pos - target_left_pos)
                    right_error = np.linalg.norm(right_pos - target_right_pos)
                    converged = True
                else:
                    left_error = float('inf')
                    right_error = float('inf')
                    converged = False
                
                combined_error = (left_error + right_error) / 2
                
                # Test original solver for comparison
                try:
                    original_solution = original_solver.solve(
                        pelvis_pose, initial_left, initial_right[:4],
                        target_left, target_right
                    )
                    
                    original_solver._set_robot_state(pelvis_pose, original_solution[:4], original_solution[4:])
                    orig_left_pos = original_solver._physics.named.data.site_xpos[f"h1/{original_solver._left_site_name}"]
                    orig_right_pos = original_solver._physics.named.data.site_xpos[f"h1/{original_solver._right_site_name}"]
                    
                    orig_left_error = np.linalg.norm(orig_left_pos - target_left_pos)
                    orig_right_error = np.linalg.norm(orig_right_pos - target_right_pos)
                    orig_combined_error = (orig_left_error + orig_right_error) / 2
                    
                except Exception:
                    orig_combined_error = float('inf')
                
                # Check for failure
                if not converged or combined_error > 0.05:  # 50mm threshold
                    failure_distance = distance
                    print(f"  {distance_mm:3.0f}mm: ❌ FAILURE - Mink error {combined_error*1000:.1f}mm")
                    break
                else:
                    success_indicator = "✅" if combined_error < orig_combined_error else "⚠️"
                    if distance_mm % 50 < 1 or combined_error > 0.01:  # Print every 50mm or if error > 10mm
                        print(f"  {distance_mm:3.0f}mm: {success_indicator} Mink {combined_error*1000:.1f}mm vs Orig {orig_combined_error*1000:.1f}mm")
                
            except Exception as e:
                failure_distance = distance
                print(f"  {distance_mm:3.0f}mm: ❌ EXCEPTION - {str(e)[:50]}...")
                break
        
        if failure_distance is None:
            print(f"  ✅ SUCCESS: Completed all distances up to {distances[-1]*1000:.0f}mm")
        else:
            failure_distance_mm = failure_distance * 1000
            print(f"  ❌ FAILED at {failure_distance_mm:.0f}mm distance")
    
    env.close()
    return


def test_pelvis_movement():
    """Test 3: Test IK solving with different pelvis positions and orientations."""
    print(f"\n{'='*80}")
    print("TEST 3: PELVIS MOVEMENT ANALYSIS")
    print("=" * 80)
    print("Testing IK solver performance with different pelvis poses.")
    print("This tests how well the solver handles base movement.\n")
    
    # Create environment and solvers
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode=None,
    )
    
    mink_solver = MinkH1IK(env)
    original_solver = H1UpperBodyIK(env, enable_full_6d_control=False)
    
    # Base pelvis configuration
    base_pelvis_pos = np.array([0.0, 0.0, 0.98])
    base_pelvis_quat = Quaternion(w=1, x=0, y=0, z=0)
    
    initial_left = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
    initial_right = np.array([0.0, 0.0, 0.0, -0.5, 0.0])
    
    # Test scenarios for pelvis movement
    pelvis_test_cases = [
        ("Forward 10cm", np.array([0.1, 0.0, 0.0]), Quaternion(w=1, x=0, y=0, z=0)),
        ("Forward 20cm", np.array([0.2, 0.0, 0.0]), Quaternion(w=1, x=0, y=0, z=0)),
        ("Backward 10cm", np.array([-0.1, 0.0, 0.0]), Quaternion(w=1, x=0, y=0, z=0)),
        ("Left 10cm", np.array([0.0, 0.1, 0.0]), Quaternion(w=1, x=0, y=0, z=0)),
        ("Right 10cm", np.array([0.0, -0.1, 0.0]), Quaternion(w=1, x=0, y=0, z=0)),
        ("Up 5cm", np.array([0.0, 0.0, 0.05]), Quaternion(w=1, x=0, y=0, z=0)),
        ("Down 5cm", np.array([0.0, 0.0, -0.05]), Quaternion(w=1, x=0, y=0, z=0)),
        ("Yaw 15°", np.array([0.0, 0.0, 0.0]), Quaternion(axis=[0, 0, 1], degrees=15)),
        ("Yaw -15°", np.array([0.0, 0.0, 0.0]), Quaternion(axis=[0, 0, 1], degrees=-15)),
        ("Forward + Yaw", np.array([0.1, 0.0, 0.0]), Quaternion(axis=[0, 0, 1], degrees=10)),
        ("Complex", np.array([0.1, 0.05, 0.02]), Quaternion(axis=[0, 0, 1], degrees=10)),
    ]
    
    # Test each pelvis configuration
    for case_name, pelvis_offset, pelvis_rotation in pelvis_test_cases:
        print(f"\n{'='*60}")
        print(f"TESTING: {case_name}")
        print(f"{'='*60}")
        
        # Create pelvis pose
        pelvis_pos = base_pelvis_pos + pelvis_offset
        pelvis_quat = base_pelvis_quat * pelvis_rotation
        pelvis_pose = Pose(pelvis_pos, pelvis_quat)
        
        print(f"  Pelvis position: {pelvis_pos}")
        print(f"  Pelvis rotation: {pelvis_rotation.degrees}° around {pelvis_rotation.axis if hasattr(pelvis_rotation, 'axis') else 'N/A'}")
        
        # Get target positions with this pelvis pose
        mink_solver._set_robot_state(pelvis_pose, initial_left, initial_right)
        mujoco.mj_fwdPosition(mink_solver._model, mink_solver._data)
        
        current_left_pos = mink_solver._data.site_xpos[mink_solver._left_site_id].copy()
        current_right_pos = mink_solver._data.site_xpos[mink_solver._right_site_id].copy()
        
        # Test maintaining current position (should be easy)
        target_left = Pose(current_left_pos, Quaternion(w=1, x=0, y=0, z=0))
        target_right = Pose(current_right_pos, Quaternion(w=1, x=0, y=0, z=0))
        
        # Test Mink solver
        try:
            mink_solution = mink_solver.solve(
                pelvis_pose, initial_left, initial_right,
                target_left, target_right
            )
            
            if hasattr(mink_solver, '_converged_full_state'):
                left_pos, right_pos = mink_solver.get_converged_end_effector_positions()
                left_error = np.linalg.norm(left_pos - current_left_pos)
                right_error = np.linalg.norm(right_pos - current_right_pos)
                mink_error = (left_error + right_error) / 2
            else:
                mink_error = float('inf')
            
        except Exception as e:
            mink_error = float('inf')
            print(f"  Mink exception: {str(e)[:50]}...")
        
        # Test original solver
        try:
            original_solution = original_solver.solve(
                pelvis_pose, initial_left, initial_right[:4],
                target_left, target_right
            )
            
            original_solver._set_robot_state(pelvis_pose, original_solution[:4], original_solution[4:])
            orig_left_pos = original_solver._physics.named.data.site_xpos[f"h1/{original_solver._left_site_name}"]
            orig_right_pos = original_solver._physics.named.data.site_xpos[f"h1/{original_solver._right_site_name}"]
            
            orig_left_error = np.linalg.norm(orig_left_pos - current_left_pos)
            orig_right_error = np.linalg.norm(orig_right_pos - current_right_pos)
            orig_error = (orig_left_error + orig_right_error) / 2
            
        except Exception as e:
            orig_error = float('inf')
            print(f"  Original exception: {str(e)[:50]}...")
        
        # Print results
        if mink_error < float('inf'):
            success_indicator = "✅" if mink_error < 0.01 else "⚠️" if mink_error < 0.05 else "❌"
            comparison = "BETTER" if mink_error < orig_error else "WORSE"
            print(f"  {success_indicator} Mink error: {mink_error*1000:.1f}mm")
            print(f"     Original error: {orig_error*1000:.1f}mm")
            print(f"     Mink is {comparison} by {abs(mink_error - orig_error)*1000:.1f}mm")
        else:
            print(f"  ❌ Mink solver failed")
            if orig_error < float('inf'):
                print(f"     Original error: {orig_error*1000:.1f}mm")
        
        # Now test with arm movement from this pelvis pose
        print(f"\n  Testing arm reach from this pelvis pose:")
        
        # Test reaching forward 10cm
        target_left_reach = Pose(current_left_pos + np.array([0.1, 0.0, 0.0]), Quaternion(w=1, x=0, y=0, z=0))
        target_right_reach = Pose(current_right_pos + np.array([0.1, 0.0, 0.0]), Quaternion(w=1, x=0, y=0, z=0))
        
        try:
            mink_solution = mink_solver.solve(
                pelvis_pose, initial_left, initial_right,
                target_left_reach, target_right_reach
            )
            
            if hasattr(mink_solver, '_converged_full_state'):
                left_pos, right_pos = mink_solver.get_converged_end_effector_positions()
                left_error = np.linalg.norm(left_pos - (current_left_pos + np.array([0.1, 0.0, 0.0])))
                right_error = np.linalg.norm(right_pos - (current_right_pos + np.array([0.1, 0.0, 0.0])))
                mink_reach_error = (left_error + right_error) / 2
                
                success_indicator = "✅" if mink_reach_error < 0.01 else "⚠️" if mink_reach_error < 0.05 else "❌"
                print(f"    {success_indicator} Forward 10cm reach: {mink_reach_error*1000:.1f}mm error")
            else:
                print(f"    ❌ Forward 10cm reach: Failed to converge")
                
        except Exception as e:
            print(f"    ❌ Forward 10cm reach: Exception - {str(e)[:50]}...")
    
    env.close()
    return


def main():
    """Run systematic failure analysis with pelvis movement."""
    print("SYSTEMATIC FAILURE ANALYSIS FOR MINK IK SOLVER")
    print("=" * 80)
    print("This analysis will help identify exactly where and why the Mink solver fails.")
    print("Three complementary tests:")
    print("1. Gradual progression - cumulative movement effects")
    print("2. Distance scaling - pure distance limitations")
    print("3. Pelvis movement - base pose variations")
    print("\nUsing the FIXED measurement system for accurate error calculation.\n")
    
    try:
        # Test 1: Gradual target progression
        test_gradual_target_progression()
        
        # Test 2: Distance scaling
        test_distance_scaling()
        
        # Test 3: Pelvis movement
        test_pelvis_movement()
        
        print(f"\n{'='*80}")
        print("SYSTEMATIC FAILURE ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print("Key insights to analyze:")
        print("1. At what distance does the solver start failing?")
        print("2. Are certain directions more problematic than others?")
        print("3. Does cumulative movement (Test 1) cause earlier failure than pure distance (Test 2)?")
        print("4. How does pelvis movement affect solver performance?")
        print("5. Where does Mink outperform the original solver, and where does it fail?")
        print("6. Are failures due to convergence issues or fundamental limitations?")
        
    except Exception as e:
        print(f"Error during systematic analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()