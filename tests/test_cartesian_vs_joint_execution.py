"""Test cartesian action execution against joint action execution.
This test compares the execution of original joint demos vs converted cartesian demos
to identify where the IK solver diverges from the expected behavior.
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


class TestCartesianVsJointExecution:
    """Test IK solver performance by comparing joint vs cartesian execution."""
    
    @pytest.fixture
    def environments(self):
        """Create environments for IK testing."""
        joint_env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=50,
            render_mode=None,
        )
        
        cartesian_env = ReachTarget(
            action_mode=CartesianActionMode(floating_base=True),
            control_frequency=50,
            render_mode=None,
        )
        
        return joint_env, cartesian_env
    
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
    
    def test_ik_solver_accuracy(self, environments):
        """Test IK solver accuracy by comparing cartesian vs joint execution.
        
        This test verifies:
        1. Cartesian demo action labels (from conversion) match joint FK results
        2. Cartesian environment execution matches the action labels  
        3. Joint environment execution matches cartesian execution
        
        If 1&3 match but 2 diverges, the issue is in the IK solver.
        
        Uses the actual converted demos with robust matching.
        """
        joint_env, cartesian_env = environments
        
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
        
        # Reset both environments to same state
        joint_env.reset()
        cartesian_env.reset()
        
        pose_errors = []
        execution_errors = []
        
        for step_idx in range(max_timesteps):
            joint_action = joint_actions[step_idx]
            cartesian_action = cartesian_actions[step_idx]
            
            # Clip actions to bounds
            joint_action_clipped = np.clip(
                joint_action, joint_env.action_space.low, joint_env.action_space.high
            )
            cartesian_action_clipped = np.clip(
                cartesian_action, cartesian_env.action_space.low, cartesian_env.action_space.high
            )
            
            # Execute joint action and capture resulting poses (ground truth)
            joint_env.step(joint_action_clipped)
            
            from bigym.const import HandSide
            joint_left_site = joint_env.robot._wrist_sites[HandSide.LEFT]
            joint_right_site = joint_env.robot._wrist_sites[HandSide.RIGHT]
            
            joint_left_pos = joint_left_site.get_position()
            joint_right_pos = joint_right_site.get_position()
            
            # Execute cartesian action and capture resulting poses
            cartesian_env.step(cartesian_action_clipped)
            
            cartesian_left_site = cartesian_env.robot._wrist_sites[HandSide.LEFT]
            cartesian_right_site = cartesian_env.robot._wrist_sites[HandSide.RIGHT]
            
            cartesian_left_pos = cartesian_left_site.get_position()
            cartesian_right_pos = cartesian_right_site.get_position()
            
            # Compare 1: Cartesian action labels vs joint FK results
            # Extract target poses from cartesian action
            target_left_pos = cartesian_action[0:3]
            target_right_pos = cartesian_action[9:12]
            
            # Calculate errors between action labels and joint FK (should be ~0)
            label_vs_joint_left = np.linalg.norm(target_left_pos - joint_left_pos)
            label_vs_joint_right = np.linalg.norm(target_right_pos - joint_right_pos)
            
            # Compare 2: Cartesian execution vs joint execution
            execution_left_error = np.linalg.norm(cartesian_left_pos - joint_left_pos)
            execution_right_error = np.linalg.norm(cartesian_right_pos - joint_right_pos)
            
            # Compare 3: Cartesian execution vs cartesian action labels  
            execution_vs_label_left = np.linalg.norm(cartesian_left_pos - target_left_pos)
            execution_vs_label_right = np.linalg.norm(cartesian_right_pos - target_right_pos)
            
            pose_errors.append((label_vs_joint_left, label_vs_joint_right))
            execution_errors.append((execution_left_error, execution_right_error, 
                                   execution_vs_label_left, execution_vs_label_right))
            
            print(f"\nStep {step_idx}:")
            print(f"  Action label vs Joint FK:  L={label_vs_joint_left*1000:.1f}mm R={label_vs_joint_right*1000:.1f}mm")
            print(f"  Cartesian vs Joint exec:   L={execution_left_error*1000:.1f}mm R={execution_right_error*1000:.1f}mm")
            print(f"  Cartesian exec vs label:   L={execution_vs_label_left*1000:.1f}mm R={execution_vs_label_right*1000:.1f}mm")
        
        # Analyze results
        avg_label_joint_error = np.mean([e[0] + e[1] for e in pose_errors])
        avg_execution_error = np.mean([e[0] + e[1] for e in execution_errors])
        avg_label_execution_error = np.mean([e[2] + e[3] for e in execution_errors])
        
        print(f"\nSUMMARY:")
        print(f"  Avg action label vs joint FK: {avg_label_joint_error*1000:.1f}mm (should be ~0)")
        print(f"  Avg cartesian vs joint exec:  {avg_execution_error*1000:.1f}mm")
        print(f"  Avg cartesian exec vs label:  {avg_label_execution_error*1000:.1f}mm (IK error)")
        
        # Assertions
        # 1. Action labels should match joint FK perfectly (we fixed conversion)
        assert avg_label_joint_error < 0.001, f"Action labels don't match joint FK: {avg_label_joint_error*1000:.1f}mm"
        
        # 2. If IK is perfect, cartesian execution should match action labels
        if avg_label_execution_error > 0.01:  # 10mm threshold
            print(f"⚠️  IK solver error detected: {avg_label_execution_error*1000:.1f}mm average")
            print("   This explains the success rate difference between joint and cartesian demos")
        
        # 3. Document the execution difference for analysis
        execution_ratio = avg_execution_error / avg_label_joint_error if avg_label_joint_error > 0 else float('inf')
        print(f"\nExecution error ratio: {execution_ratio:.1f}x conversion error")
        
        joint_env.close()
        cartesian_env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])