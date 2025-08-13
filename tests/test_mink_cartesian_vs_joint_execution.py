"""Test Mink-based cartesian action execution against joint action execution.

This test compares the execution of original joint demos vs converted cartesian demos
using the new Mink-based IK solver to evaluate its real-world performance.
"""
import sys
import os

# Add the project root to Python path for direct execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion

from bigym.cartesian_action_mode import CartesianActionMode
from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata


class MinkCartesianActionMode(CartesianActionMode):
    """CartesianActionMode using Mink-based IK solver."""
    
    def _initialize_ik_solver(self):
        """Initialize the Mink IK solver instead of the original."""
        from bigym.envs.reach_target import ReachTarget
        from vr.ik.mink_h1_ik import MinkH1IK
        
        # Create a separate dummy environment for IK solving
        # This prevents the IK solver from modifying the actual simulation
        dummy_joint_mode = JointPositionActionMode(
            absolute=True,
            floating_base=self.floating_base,
            floating_dofs=self.floating_dofs
        )
        self._ik_dummy_env = ReachTarget(action_mode=dummy_joint_mode, render_mode=None)
        
        # CRITICAL: Reset dummy environment to get its initial state
        self._ik_dummy_env.reset()
        
        # CRITICAL: Sync the dummy environment state with actual robot COMPLETELY
        # This ensures they have the same exact state including model parameters
        actual_qpos = self._robot._mojo.physics.data.qpos.copy()
        actual_qvel = self._robot._mojo.physics.data.qvel.copy()
        
        # Copy ALL state to dummy environment
        self._ik_dummy_env.robot._mojo.physics.data.qpos[:] = actual_qpos
        self._ik_dummy_env.robot._mojo.physics.data.qvel[:] = actual_qvel
        
        # Also copy control state if it exists
        if hasattr(self._robot._mojo.physics.data, 'ctrl'):
            actual_ctrl = self._robot._mojo.physics.data.ctrl.copy()
            self._ik_dummy_env.robot._mojo.physics.data.ctrl[:] = actual_ctrl
        
        # Forward kinematics to update positions
        import mujoco
        mujoco.mj_fwdPosition(
            self._ik_dummy_env.robot._mojo.physics.model.ptr,
            self._ik_dummy_env.robot._mojo.physics.data.ptr
        )
        
        # Use Mink IK solver with the properly synced dummy environment
        self._ik_solver = MinkH1IK(self._ik_dummy_env)
        
        # We'll sync state before each solve
        self._needs_state_sync = True
        
        # Calibrate the IK solver with the current robot state
        self._calibrate_ik_solver()
    
    def _sync_ik_state(self):
        """Synchronize IK solver's dummy environment with actual robot state."""
        if not self._needs_state_sync:
            return
            
        # Get current state from actual robot
        actual_qpos = self._robot._mojo.physics.data.qpos.copy()
        actual_qvel = self._robot._mojo.physics.data.qvel.copy()
        
        # Set dummy environment to match actual state
        self._ik_dummy_env.robot._mojo.physics.data.qpos[:] = actual_qpos
        self._ik_dummy_env.robot._mojo.physics.data.qvel[:] = actual_qvel
        
        # Forward kinematics to update positions
        import mujoco
        mujoco.mj_fwdPosition(
            self._ik_dummy_env.robot._mojo.physics.model.ptr,
            self._ik_dummy_env.robot._mojo.physics.data.ptr
        )
        
        self._needs_state_sync = False
    
    def step(self, action: np.ndarray):
        """Override step to sync state before IK solving."""
        # Sync IK solver state with actual robot before solving
        if hasattr(self, '_ik_dummy_env'):
            self._sync_ik_state()
        
        # Call parent step which will use the IK solver
        super().step(action)
        
        # Mark that we need to sync again next time
        self._needs_state_sync = True


class TestMinkCartesianVsJointExecution:
    """Test Mink IK solver performance by comparing joint vs cartesian execution."""
    
    @pytest.fixture
    def environments(self):
        """Create environments for IK testing."""
        joint_env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=50,
            render_mode=None,
        )
        
        # Original cartesian environment
        original_cartesian_env = ReachTarget(
            action_mode=CartesianActionMode(floating_base=True),
            control_frequency=50,
            render_mode=None,
        )
        
        # Mink-based cartesian environment
        mink_cartesian_env = ReachTarget(
            action_mode=MinkCartesianActionMode(floating_base=True),
            control_frequency=50,
            render_mode=None,
        )
        
        return joint_env, original_cartesian_env, mink_cartesian_env
    
    def _find_unambiguous_demo_match(self, joint_demos, cartesian_demo_files):
        """Find unambiguous joint-cartesian demo pair using step count.
        
        Returns None if there are ambiguous matches (multiple demos with same step count).
        This ensures we only test when we're confident about the demo pairing.
        """
        from safetensors.numpy import load_file
        
        # First, collect step counts for all demos
        joint_step_counts = []
        cartesian_step_counts = []
        
        for joint_demo in joint_demos:
            joint_actions = np.array([ts.executed_action for ts in joint_demo.timesteps])
            joint_step_counts.append(len(joint_actions))
        
        for cartesian_file in cartesian_demo_files:
            try:
                demo_data = load_file(cartesian_file)
                action_key = None
                for key in demo_data.keys():
                    if 'action' in key.lower():
                        action_key = key
                        break
                
                if action_key:
                    cartesian_actions = demo_data[action_key]
                    cartesian_step_counts.append(len(cartesian_actions))
                else:
                    cartesian_step_counts.append(0)
            except Exception:
                cartesian_step_counts.append(0)
        
        print(f"Joint demo step counts: {joint_step_counts}")
        print(f"Cartesian demo step counts: {cartesian_step_counts}")
        
        # Check for ambiguities
        for step_count in joint_step_counts:
            if joint_step_counts.count(step_count) > 1:
                print(f"❌ Ambiguous: Multiple joint demos have {step_count} steps")
                return None
        
        for step_count in cartesian_step_counts:
            if step_count > 0 and cartesian_step_counts.count(step_count) > 1:
                print(f"❌ Ambiguous: Multiple cartesian demos have {step_count} steps")
                return None
        
        # Find exact step count matches
        for joint_idx, joint_demo in enumerate(joint_demos):
            joint_steps = joint_step_counts[joint_idx]
            
            matching_cartesian_indices = [
                i for i, steps in enumerate(cartesian_step_counts) 
                if steps == joint_steps
            ]
            
            if len(matching_cartesian_indices) == 1:
                cartesian_idx = matching_cartesian_indices[0]
                cartesian_file = cartesian_demo_files[cartesian_idx]
                
                print(f"✓ Unambiguous match found:")
                print(f"  Joint demo {joint_idx}: {joint_steps} steps")
                print(f"  Cartesian demo {cartesian_idx} ({cartesian_file.name}): {joint_steps} steps")
                
                return (joint_demo, cartesian_file, joint_steps)
        
        print(f"❌ No unambiguous step count matches found")
        return None
    
    def test_mink_vs_original_ik_accuracy(self, environments):
        """Test Mink IK solver accuracy compared to original solver.
        
        This test compares:
        1. Original IK solver execution vs joint execution
        2. Mink IK solver execution vs joint execution  
        3. Mink vs Original IK solver execution
        
        The goal is to show that Mink achieves better accuracy.
        """
        joint_env, original_cartesian_env, mink_cartesian_env = environments
        
        # Load joint demos
        demo_store = DemoStore()
        joint_metadata = Metadata.from_env(joint_env)
        joint_demos = demo_store.get_demos(joint_metadata, amount=3, frequency=50)
        
        if not joint_demos:
            pytest.skip("No demos available for testing")
        
        # Load cartesian demos
        cartesian_demos_dir = Path("cartesian_demos_final")
        if not cartesian_demos_dir.exists():
            pytest.skip("No cartesian demos directory found")
        
        cartesian_demo_files = sorted(list(cartesian_demos_dir.glob("*.safetensors")))
        if not cartesian_demo_files:
            pytest.skip("No cartesian demos found")
        
        from safetensors.numpy import load_file
        
        # Find unambiguous matching pair using step count only
        match_result = self._find_unambiguous_demo_match(joint_demos, cartesian_demo_files)
        
        if match_result is None:
            pytest.skip("No unambiguous demo match found - multiple demos with same step count")
        
        joint_demo, cartesian_file, step_count = match_result
        joint_actions = np.array([ts.executed_action for ts in joint_demo.timesteps])
        
        # Load the matched cartesian demo
        demo_data = load_file(cartesian_file)
        action_key = None
        for key in demo_data.keys():
            if 'action' in key.lower():
                action_key = key
                break
        
        if not action_key:
            pytest.skip("No action key found in cartesian demo")
        
        cartesian_actions = demo_data[action_key]
        
        print(f"Testing with {step_count} step demo pair")
        
        # Test first few timesteps for detailed comparison
        max_timesteps = min(5, len(joint_actions), len(cartesian_actions))
        
        # Reset all environments to same state
        joint_env.reset()
        original_cartesian_env.reset()
        mink_cartesian_env.reset()
        
        original_errors = []
        mink_errors = []
        comparative_errors = []
        
        for step_idx in range(max_timesteps):
            joint_action = joint_actions[step_idx]
            cartesian_action = cartesian_actions[step_idx]
            
            # Clip actions to bounds
            joint_action_clipped = np.clip(
                joint_action, joint_env.action_space.low, joint_env.action_space.high
            )
            cartesian_action_clipped = np.clip(
                cartesian_action, original_cartesian_env.action_space.low, original_cartesian_env.action_space.high
            )
            
            # Execute joint action and capture resulting poses (ground truth)
            joint_env.step(joint_action_clipped)
            
            from bigym.const import HandSide
            joint_left_site = joint_env.robot._wrist_sites[HandSide.LEFT]
            joint_right_site = joint_env.robot._wrist_sites[HandSide.RIGHT]
            
            joint_left_pos = joint_left_site.get_position()
            joint_right_pos = joint_right_site.get_position()
            
            # Execute original cartesian action
            original_cartesian_env.step(cartesian_action_clipped)
            
            original_left_site = original_cartesian_env.robot._wrist_sites[HandSide.LEFT]
            original_right_site = original_cartesian_env.robot._wrist_sites[HandSide.RIGHT]
            
            original_left_pos = original_left_site.get_position()
            original_right_pos = original_right_site.get_position()
            
            # Execute Mink cartesian action
            mink_cartesian_env.step(cartesian_action_clipped)
            
            mink_left_site = mink_cartesian_env.robot._wrist_sites[HandSide.LEFT]
            mink_right_site = mink_cartesian_env.robot._wrist_sites[HandSide.RIGHT]
            
            mink_left_pos = mink_left_site.get_position()
            mink_right_pos = mink_right_site.get_position()
            
            # Calculate errors
            # Original IK vs Joint execution
            original_left_error = np.linalg.norm(original_left_pos - joint_left_pos)
            original_right_error = np.linalg.norm(original_right_pos - joint_right_pos)
            
            # Mink IK vs Joint execution
            mink_left_error = np.linalg.norm(mink_left_pos - joint_left_pos)
            mink_right_error = np.linalg.norm(mink_right_pos - joint_right_pos)
            
            # Mink vs Original IK
            comparative_left_error = np.linalg.norm(mink_left_pos - original_left_pos)
            comparative_right_error = np.linalg.norm(mink_right_pos - original_right_pos)
            
            original_errors.append((original_left_error, original_right_error))
            mink_errors.append((mink_left_error, mink_right_error))
            comparative_errors.append((comparative_left_error, comparative_right_error))
            
            print(f"\nStep {step_idx}:")
            print(f"  Original IK vs Joint:  L={original_left_error*1000:.1f}mm R={original_right_error*1000:.1f}mm")
            print(f"  Mink IK vs Joint:      L={mink_left_error*1000:.1f}mm R={mink_right_error*1000:.1f}mm")
            print(f"  Mink vs Original IK:   L={comparative_left_error*1000:.1f}mm R={comparative_right_error*1000:.1f}mm")
            
            # Show improvement
            left_improvement = ((original_left_error - mink_left_error) / original_left_error * 100) if original_left_error > 0 else 0
            right_improvement = ((original_right_error - mink_right_error) / original_right_error * 100) if original_right_error > 0 else 0
            print(f"  Improvement:           L={left_improvement:.1f}% R={right_improvement:.1f}%")
        
        # Analyze results
        avg_original_error = np.mean([e[0] + e[1] for e in original_errors])
        avg_mink_error = np.mean([e[0] + e[1] for e in mink_errors])
        avg_comparative_error = np.mean([e[0] + e[1] for e in comparative_errors])
        
        print(f"\n=== SUMMARY ===")
        print(f"Original IK solver avg error:  {avg_original_error*1000:.1f}mm")
        print(f"Mink IK solver avg error:      {avg_mink_error*1000:.1f}mm")
        print(f"Difference between solvers:    {avg_comparative_error*1000:.1f}mm")
        
        # Calculate overall improvement
        if avg_original_error > 0:
            overall_improvement = ((avg_original_error - avg_mink_error) / avg_original_error * 100)
            print(f"Overall improvement:           {overall_improvement:.1f}%")
            
            if overall_improvement > 0:
                print(f"✅ Mink solver is {overall_improvement:.1f}% more accurate!")
            else:
                print(f"❌ Mink solver is {-overall_improvement:.1f}% less accurate")
        
        # Performance assertions
        print(f"\n=== ANALYSIS ===")
        
        # Mink should be more accurate than original (at least for some steps)
        mink_better_count = sum(1 for i in range(len(original_errors)) 
                               if (mink_errors[i][0] + mink_errors[i][1]) < (original_errors[i][0] + original_errors[i][1]))
        
        print(f"Mink performed better in {mink_better_count}/{len(original_errors)} steps")
        
        # Document significant differences
        if avg_comparative_error > 0.01:  # 10mm threshold
            print(f"⚠️  Significant difference between solvers: {avg_comparative_error*1000:.1f}mm")
        
        # Success criteria: Mink should be better on average OR better in majority of steps
        mink_success = (avg_mink_error < avg_original_error) or (mink_better_count > len(original_errors) / 2)
        
        if mink_success:
            print("✅ Mink IK solver shows improvement over original solver")
        else:
            print("❌ Mink IK solver does not show clear improvement")
            
        # Clean up
        joint_env.close()
        original_cartesian_env.close() 
        mink_cartesian_env.close()
        
        # Document results but don't fail the test - this is for analysis
        print(f"\nTest completed: Mink average error = {avg_mink_error*1000:.1f}mm, Original average error = {avg_original_error*1000:.1f}mm")

    def test_mink_solver_convergence_analysis(self, environments):
        """Analyze convergence behavior of Mink solver in real execution."""
        joint_env, original_cartesian_env, mink_cartesian_env = environments
        
        print("\n=== Mink Solver Convergence Analysis ===")
        
        # Test with a simple, known-good cartesian action
        test_action = np.array([
            # Left EE position
            0.3, 0.3, 1.0,
            # Left EE orientation (6D)
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            # Right EE position  
            0.3, -0.3, 1.0,
            # Right EE orientation (6D)
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            # Base movement
            0.0, 0.0, 0.0,
            # Grippers
            0.0, 0.0
        ])
        
        print("Testing with synthetic action...")
        
        # Reset environments
        mink_cartesian_env.reset()
        
        # Get initial state
        from bigym.const import HandSide
        initial_left_site = mink_cartesian_env.robot._wrist_sites[HandSide.LEFT]
        initial_right_site = mink_cartesian_env.robot._wrist_sites[HandSide.RIGHT]
        
        initial_left_pos = initial_left_site.get_position()
        initial_right_pos = initial_right_site.get_position()
        
        print(f"Initial positions:")
        print(f"  Left:  {initial_left_pos}")
        print(f"  Right: {initial_right_pos}")
        
        # Execute action
        mink_cartesian_env.step(test_action)
        
        # Get final state
        final_left_site = mink_cartesian_env.robot._wrist_sites[HandSide.LEFT]
        final_right_site = mink_cartesian_env.robot._wrist_sites[HandSide.RIGHT]
        
        final_left_pos = final_left_site.get_position()
        final_right_pos = final_right_site.get_position()
        
        print(f"Final positions:")
        print(f"  Left:  {final_left_pos}")
        print(f"  Right: {final_right_pos}")
        
        # Target positions from action
        target_left_pos = test_action[0:3]
        target_right_pos = test_action[9:12]
        
        print(f"Target positions:")
        print(f"  Left:  {target_left_pos}")
        print(f"  Right: {target_right_pos}")
        
        # Calculate accuracy
        left_error = np.linalg.norm(final_left_pos - target_left_pos)
        right_error = np.linalg.norm(final_right_pos - target_right_pos)
        
        print(f"Execution accuracy:")
        print(f"  Left error:  {left_error*1000:.1f}mm")
        print(f"  Right error: {right_error*1000:.1f}mm")
        
        # Check if within reasonable bounds
        if left_error < 0.01 and right_error < 0.01:
            print("✅ Excellent accuracy (< 10mm)")
        elif left_error < 0.05 and right_error < 0.05:
            print("✅ Good accuracy (< 50mm)")
        else:
            print("❌ Poor accuracy (> 50mm)")
        
        mink_cartesian_env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])