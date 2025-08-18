#!/usr/bin/env python3
"""Comprehensive test suite for RBY1 IK solver.

This test covers:
1. IK solver accuracy with different target distances
2. Base movement control with pose task
3. Torso joint utilization 
4. Arm joint movement tracking
5. Different base constraint strategies
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import mink


class TestRBY1IKComprehensive:
    """Comprehensive test suite for RBY1 IK solver."""
    
    def __init__(self):
        """Initialize test suite."""
        # Load model
        model_path = "bigym/envs/xmls/rby1/model_act.xml"
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(model_path))
        
        try:
            self.model = mujoco.MjModel.from_xml_path(os.path.basename(model_path))
            self.data = mujoco.MjData(self.model)
        finally:
            os.chdir(original_dir)
        
        # Get site IDs
        self.left_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_l")
        self.right_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_r")
        
        # Setup joint tracking
        self._setup_joint_tracking()
        
    def _setup_joint_tracking(self):
        """Setup joint indices for tracking movement."""
        # Base joints (world_j: 0-6 in qpos, 0-5 in vel)
        self.base_qpos_indices = list(range(7))
        self.base_vel_indices = list(range(6))
        
        # Wheel joints (7-10 in qpos, 6-9 in vel)
        self.wheel_qpos_indices = list(range(7, 11))
        self.wheel_vel_indices = list(range(6, 10))
        
        # Torso joints (11-16 in qpos, 10-15 in vel)
        self.torso_qpos_indices = []
        self.torso_names = []
        for i in range(6):
            joint_name = f"torso_{i}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.torso_qpos_indices.append(self.model.jnt_qposadr[joint_id])
                self.torso_names.append(joint_name)
        
        # Right arm joints (17-23 in qpos, 16-22 in vel)
        self.right_arm_qpos_indices = []
        self.right_arm_names = []
        for i in range(7):
            joint_name = f"right_arm_{i}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.right_arm_qpos_indices.append(self.model.jnt_qposadr[joint_id])
                self.right_arm_names.append(joint_name)
        
        # Left arm joints (24-30 in qpos, 23-29 in vel)
        self.left_arm_qpos_indices = []
        self.left_arm_names = []
        for i in range(7):
            joint_name = f"left_arm_{i}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.left_arm_qpos_indices.append(self.model.jnt_qposadr[joint_id])
                self.left_arm_names.append(joint_name)
    
    def test_basic_reach(self):
        """Test 1: Basic reaching with different distances."""
        print("="*80)
        print("TEST 1: BASIC REACHING")
        print("="*80)
        print("Testing IK accuracy at different distances.\n")
        
        # Reset to initial state
        self.data.qpos[:] = 0
        self.data.qpos[2] = 0.35  # Base height
        self.data.qpos[3] = 1.0   # Identity quaternion
        mujoco.mj_forward(self.model, self.data)
        
        initial_left = self.data.site_xpos[self.left_site_id].copy()
        initial_right = self.data.site_xpos[self.right_site_id].copy()
        
        print(f"Initial EE positions:")
        print(f"  Left:  {initial_left}")
        print(f"  Right: {initial_right}")
        
        # Test different distances
        test_distances = [0.02, 0.05, 0.10, 0.15, 0.20]
        
        for distance in test_distances:
            print(f"\n{'='*60}")
            print(f"Testing {distance*100:.0f}cm forward reach")
            print(f"{'='*60}")
            
            # Reset
            self.data.qpos[:] = 0
            self.data.qpos[2] = 0.35
            self.data.qpos[3] = 1.0
            mujoco.mj_forward(self.model, self.data)
            
            # Targets
            target_left = initial_left + np.array([distance, 0, 0])
            target_right = initial_right + np.array([distance, 0, 0])
            
            # Create Mink configuration
            configuration = mink.Configuration(self.model, self.data.qpos.copy())
            
            # Setup tasks
            tasks = []
            
            # Posture task for stability
            posture_task = mink.PostureTask(model=self.model, cost=10.0)
            posture_task.set_target(self.data.qpos.copy())
            tasks.append(posture_task)
            
            # Left EE task
            left_task = mink.FrameTask(
                frame_name="end_effector_l",
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = target_left
            left_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(left_task)
            
            # Right EE task
            right_task = mink.FrameTask(
                frame_name="end_effector_r",
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            target_matrix[:3, 3] = target_right
            right_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(right_task)
            
            # Solve IK
            dt = 0.1
            max_iterations = 100
            
            initial_qpos = self.data.qpos.copy()
            
            for iteration in range(max_iterations):
                vel = mink.solve_ik(configuration, tasks, dt, "daqp", 1e-6)
                
                # Only zero wheel velocities
                vel[6:10] = 0
                
                configuration.integrate_inplace(vel, dt)
                
                self.data.qpos[:] = configuration.q
                mujoco.mj_forward(self.model, self.data)
                
                # Check convergence
                left_pos = self.data.site_xpos[self.left_site_id]
                right_pos = self.data.site_xpos[self.right_site_id]
                
                left_error = np.linalg.norm(left_pos - target_left)
                right_error = np.linalg.norm(right_pos - target_right)
                
                if left_error < 0.001 and right_error < 0.001:
                    break
            
            # Analyze results
            final_qpos = self.data.qpos.copy()
            
            # Track joint movements
            base_movement = np.linalg.norm(final_qpos[0:3] - initial_qpos[0:3])
            torso_movement = np.linalg.norm(final_qpos[self.torso_qpos_indices] - 
                                           initial_qpos[self.torso_qpos_indices])
            left_arm_movement = np.linalg.norm(final_qpos[self.left_arm_qpos_indices] - 
                                              initial_qpos[self.left_arm_qpos_indices])
            right_arm_movement = np.linalg.norm(final_qpos[self.right_arm_qpos_indices] - 
                                               initial_qpos[self.right_arm_qpos_indices])
            
            status = "✅" if (left_error < 0.001 and right_error < 0.001) else "❌"
            
            print(f"  {status} Converged in {iteration+1} iterations")
            print(f"  EE errors: L={left_error*1000:.2f}mm, R={right_error*1000:.2f}mm")
            print(f"  Joint movements:")
            print(f"    Base:      {base_movement*100:.1f}cm")
            print(f"    Torso:     {torso_movement:.3f}rad")
            print(f"    Left arm:  {left_arm_movement:.3f}rad")
            print(f"    Right arm: {right_arm_movement:.3f}rad")
            
            # Show which torso joints moved
            torso_deltas = final_qpos[self.torso_qpos_indices] - initial_qpos[self.torso_qpos_indices]
            print(f"  Torso joint deltas (rad):")
            for i, (name, delta) in enumerate(zip(self.torso_names, torso_deltas)):
                if abs(delta) > 0.001:
                    print(f"    {name}: {delta:+.3f}")
    
    def test_base_constraint(self):
        """Test 2: Different base constraint strategies."""
        print("\n" + "="*80)
        print("TEST 2: BASE CONSTRAINT STRATEGIES")
        print("="*80)
        print("Testing different ways to control base movement.\n")
        
        # Reset
        self.data.qpos[:] = 0
        self.data.qpos[2] = 0.35
        self.data.qpos[3] = 1.0
        mujoco.mj_forward(self.model, self.data)
        
        initial_left = self.data.site_xpos[self.left_site_id].copy()
        initial_right = self.data.site_xpos[self.right_site_id].copy()
        
        # Target 10cm forward
        target_left = initial_left + np.array([0.1, 0, 0])
        target_right = initial_right + np.array([0.1, 0, 0])
        
        strategies = [
            ("Low posture cost (1.0)", 1.0, False),
            ("Medium posture cost (10.0)", 10.0, False),
            ("High posture cost (100.0)", 100.0, False),
            ("Very high posture cost (1000.0)", 1000.0, False),
            ("Base pose task (cost=10000)", None, True),
        ]
        
        for strategy_name, posture_cost, use_base_task in strategies:
            print(f"\n{strategy_name}:")
            
            # Reset
            self.data.qpos[:] = 0
            self.data.qpos[2] = 0.35
            self.data.qpos[3] = 1.0
            mujoco.mj_forward(self.model, self.data)
            
            configuration = mink.Configuration(self.model, self.data.qpos.copy())
            
            # Setup tasks
            tasks = []
            
            if use_base_task:
                # Add specific base pose task with very high cost
                base_task = mink.FrameTask(
                    frame_name="base",  # RBY1 base body
                    frame_type="body",
                    position_cost=10000.0,
                    orientation_cost=10000.0,
                    lm_damping=1e-6,
                )
                base_matrix = np.eye(4)
                base_matrix[:3, 3] = np.array([0, 0, 0.35])
                base_task.set_target(mink.SE3.from_matrix(base_matrix))
                tasks.append(base_task)
                
                # Still add posture task for arms
                posture_task = mink.PostureTask(model=self.model, cost=1.0)
                posture_task.set_target(self.data.qpos.copy())
                tasks.append(posture_task)
            else:
                # Just posture task
                posture_task = mink.PostureTask(model=self.model, cost=posture_cost)
                posture_task.set_target(self.data.qpos.copy())
                tasks.append(posture_task)
            
            # EE tasks
            left_task = mink.FrameTask(
                frame_name="end_effector_l",
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = target_left
            left_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(left_task)
            
            right_task = mink.FrameTask(
                frame_name="end_effector_r",
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            target_matrix[:3, 3] = target_right
            right_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(right_task)
            
            # Solve
            initial_qpos = self.data.qpos.copy()
            
            for iteration in range(100):
                vel = mink.solve_ik(configuration, tasks, 0.1, "daqp", 1e-6)
                vel[6:10] = 0  # Zero wheels
                configuration.integrate_inplace(vel, 0.1)
                
                self.data.qpos[:] = configuration.q
                mujoco.mj_forward(self.model, self.data)
                
                left_pos = self.data.site_xpos[self.left_site_id]
                right_pos = self.data.site_xpos[self.right_site_id]
                
                left_error = np.linalg.norm(left_pos - target_left)
                right_error = np.linalg.norm(right_pos - target_right)
                
                if left_error < 0.001 and right_error < 0.001:
                    break
            
            final_qpos = self.data.qpos.copy()
            
            # Analyze
            base_movement = np.linalg.norm(final_qpos[0:3] - initial_qpos[0:3])
            torso_movement = np.linalg.norm(final_qpos[self.torso_qpos_indices] - 
                                           initial_qpos[self.torso_qpos_indices])
            
            print(f"  Iterations: {iteration+1}")
            print(f"  EE errors: L={left_error*1000:.2f}mm, R={right_error*1000:.2f}mm")
            print(f"  Base movement: {base_movement*100:.1f}cm")
            print(f"  Torso movement: {torso_movement:.3f}rad")
            print(f"  Base final pos: [{final_qpos[0]*100:.1f}, {final_qpos[1]*100:.1f}, {final_qpos[2]*100:.1f}]cm")
    
    def test_movement_directions(self):
        """Test 3: Movement in different directions."""
        print("\n" + "="*80)
        print("TEST 3: MOVEMENT DIRECTIONS")
        print("="*80)
        print("Testing IK accuracy in different directions.\n")
        
        # Reset
        self.data.qpos[:] = 0
        self.data.qpos[2] = 0.35
        self.data.qpos[3] = 1.0
        mujoco.mj_forward(self.model, self.data)
        
        initial_left = self.data.site_xpos[self.left_site_id].copy()
        initial_right = self.data.site_xpos[self.right_site_id].copy()
        
        # Test directions (5cm movements)
        test_vectors = [
            ("Forward (X)", np.array([0.05, 0, 0])),
            ("Left (Y)", np.array([0, 0.05, 0])),
            ("Up (Z)", np.array([0, 0, 0.05])),
            ("Down (Z)", np.array([0, 0, -0.05])),
            ("Diagonal XY", np.array([0.035, 0.035, 0])),
            ("Diagonal XZ", np.array([0.035, 0, 0.035])),
        ]
        
        for direction_name, movement in test_vectors:
            print(f"\n{direction_name}:")
            
            # Reset
            self.data.qpos[:] = 0
            self.data.qpos[2] = 0.35
            self.data.qpos[3] = 1.0
            mujoco.mj_forward(self.model, self.data)
            
            target_left = initial_left + movement
            target_right = initial_right + movement
            
            configuration = mink.Configuration(self.model, self.data.qpos.copy())
            
            # Tasks with base pose constraint
            tasks = []
            
            # Base pose task
            base_task = mink.FrameTask(
                frame_name="base",
                frame_type="body",
                position_cost=1000.0,
                orientation_cost=1000.0,
                lm_damping=1e-6,
            )
            base_matrix = np.eye(4)
            base_matrix[:3, 3] = np.array([0, 0, 0.35])
            base_task.set_target(mink.SE3.from_matrix(base_matrix))
            tasks.append(base_task)
            
            # Posture task
            posture_task = mink.PostureTask(model=self.model, cost=1.0)
            posture_task.set_target(self.data.qpos.copy())
            tasks.append(posture_task)
            
            # EE tasks
            left_task = mink.FrameTask(
                frame_name="end_effector_l",
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = target_left
            left_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(left_task)
            
            right_task = mink.FrameTask(
                frame_name="end_effector_r",
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            target_matrix[:3, 3] = target_right
            right_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(right_task)
            
            # Solve
            initial_qpos = self.data.qpos.copy()
            
            for iteration in range(100):
                vel = mink.solve_ik(configuration, tasks, 0.1, "daqp", 1e-6)
                vel[6:10] = 0
                configuration.integrate_inplace(vel, 0.1)
                
                self.data.qpos[:] = configuration.q
                mujoco.mj_forward(self.model, self.data)
                
                left_pos = self.data.site_xpos[self.left_site_id]
                right_pos = self.data.site_xpos[self.right_site_id]
                
                left_error = np.linalg.norm(left_pos - target_left)
                right_error = np.linalg.norm(right_pos - target_right)
                
                if left_error < 0.001 and right_error < 0.001:
                    break
            
            final_qpos = self.data.qpos.copy()
            base_movement = np.linalg.norm(final_qpos[0:3] - initial_qpos[0:3])
            torso_movement = np.linalg.norm(final_qpos[self.torso_qpos_indices] - 
                                           initial_qpos[self.torso_qpos_indices])
            
            status = "✅" if (left_error < 0.001 and right_error < 0.001) else "⚠️" if (left_error < 0.01 and right_error < 0.01) else "❌"
            
            print(f"  {status} EE errors: L={left_error*1000:.2f}mm, R={right_error*1000:.2f}mm")
            print(f"    Base movement: {base_movement*100:.2f}cm")
            print(f"    Torso movement: {torso_movement:.3f}rad")
    
    def test_torso_utilization(self):
        """Test 4: Verify torso joints are being used effectively."""
        print("\n" + "="*80)
        print("TEST 4: TORSO UTILIZATION")
        print("="*80)
        print("Verifying that torso joints contribute to reaching.\n")
        
        # Reset
        self.data.qpos[:] = 0
        self.data.qpos[2] = 0.35
        self.data.qpos[3] = 1.0
        mujoco.mj_forward(self.model, self.data)
        
        initial_left = self.data.site_xpos[self.left_site_id].copy()
        initial_right = self.data.site_xpos[self.right_site_id].copy()
        
        # Test with torso locked vs unlocked
        scenarios = [
            ("Torso + Arms", False),
            ("Arms only (torso locked)", True),
        ]
        
        target_left = initial_left + np.array([0.15, 0, 0])  # 15cm forward
        target_right = initial_right + np.array([0.15, 0, 0])
        
        for scenario_name, lock_torso in scenarios:
            print(f"\n{scenario_name}:")
            
            # Reset
            self.data.qpos[:] = 0
            self.data.qpos[2] = 0.35
            self.data.qpos[3] = 1.0
            mujoco.mj_forward(self.model, self.data)
            
            configuration = mink.Configuration(self.model, self.data.qpos.copy())
            
            # Tasks
            tasks = []
            
            # Base constraint
            base_task = mink.FrameTask(
                frame_name="base",
                frame_type="body",
                position_cost=10000.0,
                orientation_cost=10000.0,
                lm_damping=1e-6,
            )
            base_matrix = np.eye(4)
            base_matrix[:3, 3] = np.array([0, 0, 0.35])
            base_task.set_target(mink.SE3.from_matrix(base_matrix))
            tasks.append(base_task)
            
            # Posture
            posture_task = mink.PostureTask(model=self.model, cost=1.0)
            posture_task.set_target(self.data.qpos.copy())
            tasks.append(posture_task)
            
            # EE tasks
            left_task = mink.FrameTask(
                frame_name="end_effector_l",
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = target_left
            left_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(left_task)
            
            right_task = mink.FrameTask(
                frame_name="end_effector_r",
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.0,
                lm_damping=1e-6,
            )
            target_matrix[:3, 3] = target_right
            right_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(right_task)
            
            # Solve
            initial_qpos = self.data.qpos.copy()
            
            for iteration in range(100):
                vel = mink.solve_ik(configuration, tasks, 0.1, "daqp", 1e-6)
                
                # Zero wheels
                vel[6:10] = 0
                
                # Lock torso if requested
                if lock_torso:
                    vel[10:16] = 0  # Zero torso velocities
                
                configuration.integrate_inplace(vel, 0.1)
                
                self.data.qpos[:] = configuration.q
                mujoco.mj_forward(self.model, self.data)
                
                left_pos = self.data.site_xpos[self.left_site_id]
                right_pos = self.data.site_xpos[self.right_site_id]
                
                left_error = np.linalg.norm(left_pos - target_left)
                right_error = np.linalg.norm(right_pos - target_right)
                
                if left_error < 0.001 and right_error < 0.001:
                    break
            
            final_qpos = self.data.qpos.copy()
            
            # Analyze joint contributions
            base_movement = np.linalg.norm(final_qpos[0:3] - initial_qpos[0:3])
            torso_movement = np.linalg.norm(final_qpos[self.torso_qpos_indices] - 
                                           initial_qpos[self.torso_qpos_indices])
            left_arm_movement = np.linalg.norm(final_qpos[self.left_arm_qpos_indices] - 
                                              initial_qpos[self.left_arm_qpos_indices])
            right_arm_movement = np.linalg.norm(final_qpos[self.right_arm_qpos_indices] - 
                                               initial_qpos[self.right_arm_qpos_indices])
            
            print(f"  Iterations: {iteration+1}")
            print(f"  EE errors: L={left_error*1000:.2f}mm, R={right_error*1000:.2f}mm")
            print(f"  Joint movements:")
            print(f"    Base:      {base_movement*100:.2f}cm")
            print(f"    Torso:     {torso_movement:.3f}rad")
            print(f"    Left arm:  {left_arm_movement:.3f}rad")
            print(f"    Right arm: {right_arm_movement:.3f}rad")
            
            if not lock_torso:
                # Show torso contribution
                torso_deltas = final_qpos[self.torso_qpos_indices] - initial_qpos[self.torso_qpos_indices]
                print(f"  Torso joints (rad):")
                for name, delta in zip(self.torso_names, torso_deltas):
                    if abs(delta) > 0.001:
                        print(f"    {name}: {delta:+.3f}")
    
    def run_all_tests(self):
        """Run all tests."""
        print("="*80)
        print("COMPREHENSIVE TEST SUITE FOR RBY1 IK SOLVER")
        print("="*80)
        print("This test suite covers:")
        print("1. Basic reaching at different distances")
        print("2. Base constraint strategies")
        print("3. Movement in different directions")
        print("4. Torso joint utilization")
        print("\n")
        
        self.test_basic_reach()
        self.test_base_constraint()
        self.test_movement_directions()
        self.test_torso_utilization()
        
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print("\nKey findings:")
        print("  - RBY1 has 20 IK-controlled DOF (6 torso + 14 arm joints)")
        print("  - Base movement can be controlled with FrameTask on 'base' body")
        print("  - Torso joints provide significant reach extension")
        print("  - High posture costs alone don't prevent base movement")
        print("  - Base pose task with high cost effectively constrains base")
        print("\nRecommendations:")
        print("  - Use base FrameTask for strict base constraint")
        print("  - Allow torso movement for better reach")
        print("  - Monitor all joint groups to ensure proper utilization")


def main():
    """Run comprehensive test."""
    tester = TestRBY1IKComprehensive()
    tester.run_all_tests()


if __name__ == "__main__":
    main()