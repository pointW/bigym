"""Unit tests for demo conversion scripts.

Tests verify that:
1. Joint achieved FK == achieved data
2. Joint target FK == target data  
3. Pelvis actions are identical across three datasets
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pathlib import Path
from pyquaternion import Quaternion

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.cartesian_action_mode import CartesianActionMode
from bigym.const import HandSide
from demonstrations.demo import Demo, DemoStep
from demonstrations.utils import Metadata

# Import conversion functions
from scripts.convert_demos_to_cartesian import convert_joint_demo_to_cartesian
from scripts.convert_demos_to_cartesian_target import convert_joint_demo_to_cartesian_target


class TestDemoConversion:
    """Test suite for demo conversion functions."""
    
    @pytest.fixture
    def environments(self):
        """Create test environments."""
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
        
        yield joint_env, cartesian_env
        
        joint_env.close()
        cartesian_env.close()
    
    def create_fake_demo(self, env, num_steps=3, seed=42):
        """Create a fake demo with simple movements.
        
        Args:
            env: Environment to create demo for
            num_steps: Number of timesteps in the demo
            seed: Random seed for reproducibility
            
        Returns:
            Demo object with simple movements
        """
        # Reset environment
        obs, info = env.reset(seed=seed)
        
        timesteps = []
        
        # Get action space info
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high
        
        for i in range(num_steps):
            # Create simple action that moves slightly from neutral
            # Start from middle of action space and add small perturbation
            action = (action_low + action_high) / 2.0
            
            # Add small movement to make it interesting
            # Move base slightly forward and rotate arms
            if env.action_mode.floating_base:
                # Base movement (X, Y, RZ)
                action[0] += 0.01 * i  # Move forward slightly
                action[1] += 0.005 * i  # Move sideways slightly
                action[2] += 0.002 * i  # Rotate slightly
            
            # Arm movements (simple pattern)
            arm_start = 3 if env.action_mode.floating_base else 0
            for j in range(arm_start, arm_start + 10):  # 10 arm actuators
                action[j] += 0.01 * np.sin(i * 0.5 + j * 0.2)
            
            # Clip to bounds
            action = np.clip(action, action_low, action_high)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Create timestep with proper info
            info_dict = {
                'demo_action': action.copy(),
                'task_success': False,
                'step': i
            }
            
            timestep = DemoStep(
                observation=obs,
                reward=reward,
                termination=terminated,
                truncation=truncated,
                info=info_dict,
                action=action
            )
            timesteps.append(timestep)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Create metadata
        metadata = Metadata.from_env(env)
        metadata.seed = seed
        
        # Create demo
        demo = Demo(
            metadata=metadata,
            timesteps=timesteps
        )
        
        return demo
    
    def compute_fk_poses(self, env, joint_positions):
        """Compute forward kinematics for given joint positions.
        
        Args:
            env: Environment with robot model
            joint_positions: Joint positions to compute FK for
            
        Returns:
            Tuple of (left_pose, right_pose) as numpy arrays
        """
        # Save current state
        saved_qpos = env.robot.qpos_actuated.copy()
        
        # Get floating base DOF
        floating_base_dof = env.robot.floating_base.dof_amount if env.action_mode.floating_base else 0
        
        # Set joint positions
        for i, actuator in enumerate(env.robot.limb_actuators):
            if actuator.joint:
                joint = env.mojo.physics.bind(actuator.joint)
                joint.qpos = joint_positions[floating_base_dof + i]
        
        # Update kinematics
        env.mojo.physics.forward()
        
        # Get end-effector poses
        left_site = env.robot._wrist_sites[HandSide.LEFT]
        right_site = env.robot._wrist_sites[HandSide.RIGHT]
        
        left_pos = left_site.get_position().copy()
        left_quat = left_site.get_quaternion().copy()
        
        right_pos = right_site.get_position().copy()
        right_quat = right_site.get_quaternion().copy()
        
        # Restore original state
        for i, actuator in enumerate(env.robot.limb_actuators):
            if actuator.joint:
                joint = env.mojo.physics.bind(actuator.joint)
                joint.qpos = saved_qpos[floating_base_dof + i]
        
        env.mojo.physics.forward()
        
        return (left_pos, left_quat), (right_pos, right_quat)
    
    def test_achieved_conversion_fk_match(self, environments):
        """Test that joint achieved FK matches achieved data after conversion."""
        joint_env, cartesian_env = environments
        
        # Create fake joint demo
        fake_demo = self.create_fake_demo(joint_env, num_steps=3, seed=42)
        
        # Convert to cartesian (achieved)
        cartesian_demo = convert_joint_demo_to_cartesian(fake_demo, cartesian_env, joint_env)
        
        # Verify conversion by replaying and checking FK
        joint_env.reset(seed=fake_demo.seed)
        
        floating_base_dof = joint_env.robot.floating_base.dof_amount if joint_env.action_mode.floating_base else 0
        
        errors = []
        for i, timestep in enumerate(fake_demo.timesteps):
            # Step joint environment
            joint_action = timestep.executed_action
            joint_env.step(joint_action)
            
            # Get achieved EE poses after stepping
            left_site = joint_env.robot._wrist_sites[HandSide.LEFT]
            right_site = joint_env.robot._wrist_sites[HandSide.RIGHT]
            
            achieved_left_pos = left_site.get_position()
            achieved_left_quat = Quaternion(left_site.get_quaternion())
            
            achieved_right_pos = right_site.get_position()
            achieved_right_quat = Quaternion(right_site.get_quaternion())
            
            # Extract EE poses from cartesian action
            cartesian_action = cartesian_demo.timesteps[i].executed_action
            
            # Cartesian action format: [left_pos(3), left_ori(6), right_pos(3), right_ori(6), base(3), grippers(2)]
            converted_left_pos = cartesian_action[0:3]
            converted_right_pos = cartesian_action[9:12]
            
            # Compare positions
            left_error = np.linalg.norm(achieved_left_pos - converted_left_pos)
            right_error = np.linalg.norm(achieved_right_pos - converted_right_pos)
            
            errors.append((left_error, right_error))
            
            # Should be very close (< 1mm error)
            assert left_error < 0.001, f"Step {i}: Left EE position mismatch: {left_error*1000:.2f}mm"
            assert right_error < 0.001, f"Step {i}: Right EE position mismatch: {right_error*1000:.2f}mm"
        
        avg_error = np.mean([e[0] + e[1] for e in errors]) / 2
        print(f"✅ Achieved conversion FK match test passed. Avg error: {avg_error*1000:.3f}mm")
    
    def test_target_conversion_fk_match(self, environments):
        """Test that joint target FK matches target data after conversion."""
        joint_env, cartesian_env = environments
        
        # Create fake joint demo
        fake_demo = self.create_fake_demo(joint_env, num_steps=3, seed=42)
        
        # Convert to cartesian (target)
        cartesian_demo = convert_joint_demo_to_cartesian_target(fake_demo, cartesian_env, joint_env)
        
        # Verify conversion by computing target FK
        joint_env.reset(seed=fake_demo.seed)
        
        floating_base_dof = joint_env.robot.floating_base.dof_amount if joint_env.action_mode.floating_base else 0
        num_limb_actuators = len(joint_env.robot.limb_actuators)
        
        errors = []
        for i, timestep in enumerate(fake_demo.timesteps):
            joint_action = timestep.executed_action
            
            # Extract target joint positions from action
            target_joints = joint_action[floating_base_dof:floating_base_dof + num_limb_actuators]
            
            # Compute FK for target joints
            # Save current state
            saved_qpos = joint_env.robot.qpos_actuated.copy()
            
            # Set joints to target positions
            for j, actuator in enumerate(joint_env.robot.limb_actuators):
                if actuator.joint:
                    joint = joint_env.mojo.physics.bind(actuator.joint)
                    joint.qpos = target_joints[j]
            
            # Update kinematics
            joint_env.mojo.physics.forward()
            
            # Get target EE poses
            left_site = joint_env.robot._wrist_sites[HandSide.LEFT]
            right_site = joint_env.robot._wrist_sites[HandSide.RIGHT]
            
            target_left_pos = left_site.get_position().copy()
            target_right_pos = right_site.get_position().copy()
            
            # Restore state
            for j, actuator in enumerate(joint_env.robot.limb_actuators):
                if actuator.joint:
                    joint = joint_env.mojo.physics.bind(actuator.joint)
                    joint.qpos = saved_qpos[floating_base_dof + j]
            
            joint_env.mojo.physics.forward()
            
            # Now step to maintain state consistency
            joint_env.step(joint_action)
            
            # Extract EE poses from cartesian action
            cartesian_action = cartesian_demo.timesteps[i].executed_action
            
            # Cartesian action format: [left_pos(3), left_ori(6), right_pos(3), right_ori(6), base(3), grippers(2)]
            converted_left_pos = cartesian_action[0:3]
            converted_right_pos = cartesian_action[9:12]
            
            # Compare positions
            left_error = np.linalg.norm(target_left_pos - converted_left_pos)
            right_error = np.linalg.norm(target_right_pos - converted_right_pos)
            
            errors.append((left_error, right_error))
            
            # Should be very close (< 1mm error)
            assert left_error < 0.001, f"Step {i}: Left EE position mismatch: {left_error*1000:.2f}mm"
            assert right_error < 0.001, f"Step {i}: Right EE position mismatch: {right_error*1000:.2f}mm"
        
        avg_error = np.mean([e[0] + e[1] for e in errors]) / 2
        print(f"✅ Target conversion FK match test passed. Avg error: {avg_error*1000:.3f}mm")
    
    def test_base_action_preservation(self, environments):
        """Test that pelvis/base actions are identical across all three datasets."""
        joint_env, cartesian_env = environments
        
        # Create fake joint demo
        fake_demo = self.create_fake_demo(joint_env, num_steps=3, seed=42)
        
        # Convert to both cartesian formats
        cartesian_achieved_demo = convert_joint_demo_to_cartesian(fake_demo, cartesian_env, joint_env)
        cartesian_target_demo = convert_joint_demo_to_cartesian_target(fake_demo, cartesian_env, joint_env)
        
        # Check base actions are preserved
        floating_base_dof = 3  # X, Y, RZ for floating base
        
        for i in range(len(fake_demo.timesteps)):
            # Get base actions from each dataset
            joint_action = fake_demo.timesteps[i].executed_action
            joint_base = joint_action[:floating_base_dof]
            
            achieved_action = cartesian_achieved_demo.timesteps[i].executed_action
            achieved_base = achieved_action[18:21]  # Base is at [18:21] in cartesian
            
            target_action = cartesian_target_demo.timesteps[i].executed_action
            target_base = target_action[18:21]  # Base is at [18:21] in cartesian
            
            # All three should be identical
            joint_achieved_diff = np.linalg.norm(joint_base - achieved_base)
            joint_target_diff = np.linalg.norm(joint_base - target_base)
            achieved_target_diff = np.linalg.norm(achieved_base - target_base)
            
            assert joint_achieved_diff < 1e-10, f"Step {i}: Joint-Achieved base mismatch: {joint_achieved_diff}"
            assert joint_target_diff < 1e-10, f"Step {i}: Joint-Target base mismatch: {joint_target_diff}"
            assert achieved_target_diff < 1e-10, f"Step {i}: Achieved-Target base mismatch: {achieved_target_diff}"
        
        print(f"✅ Base action preservation test passed. All base actions are identical!")
    
    def test_gripper_action_preservation(self, environments):
        """Test that gripper actions are preserved across conversions."""
        joint_env, cartesian_env = environments
        
        # Create fake joint demo
        fake_demo = self.create_fake_demo(joint_env, num_steps=3, seed=42)
        
        # Convert to both cartesian formats
        cartesian_achieved_demo = convert_joint_demo_to_cartesian(fake_demo, cartesian_env, joint_env)
        cartesian_target_demo = convert_joint_demo_to_cartesian_target(fake_demo, cartesian_env, joint_env)
        
        for i in range(len(fake_demo.timesteps)):
            # Get gripper actions from each dataset
            joint_action = fake_demo.timesteps[i].executed_action
            joint_grippers = joint_action[-2:]  # Last 2 elements
            
            achieved_action = cartesian_achieved_demo.timesteps[i].executed_action
            achieved_grippers = achieved_action[-2:]  # Last 2 elements
            
            target_action = cartesian_target_demo.timesteps[i].executed_action
            target_grippers = target_action[-2:]  # Last 2 elements
            
            # All three should be identical
            joint_achieved_diff = np.linalg.norm(joint_grippers - achieved_grippers)
            joint_target_diff = np.linalg.norm(joint_grippers - target_grippers)
            
            assert joint_achieved_diff < 1e-10, f"Step {i}: Joint-Achieved gripper mismatch: {joint_achieved_diff}"
            assert joint_target_diff < 1e-10, f"Step {i}: Joint-Target gripper mismatch: {joint_target_diff}"
        
        print(f"✅ Gripper action preservation test passed!")
    
    def test_demo_metadata_preservation(self, environments):
        """Test that demo metadata is properly preserved during conversion."""
        joint_env, cartesian_env = environments
        
        # Create fake joint demo with specific seed
        seed = 12345
        fake_demo = self.create_fake_demo(joint_env, num_steps=3, seed=seed)
        
        # Convert to both formats
        cartesian_achieved_demo = convert_joint_demo_to_cartesian(fake_demo, cartesian_env, joint_env)
        cartesian_target_demo = convert_joint_demo_to_cartesian_target(fake_demo, cartesian_env, joint_env)
        
        # Check seed preservation
        assert fake_demo.seed == seed
        assert cartesian_achieved_demo.seed == seed
        assert cartesian_target_demo.seed == seed
        
        # Check timestep count
        assert len(fake_demo.timesteps) == len(cartesian_achieved_demo.timesteps)
        assert len(fake_demo.timesteps) == len(cartesian_target_demo.timesteps)
        
        print(f"✅ Demo metadata preservation test passed!")
    
    def test_conversion_consistency(self, environments):
        """Test that multiple conversions of the same demo produce identical results."""
        joint_env, cartesian_env = environments
        
        # Create fake joint demo
        fake_demo = self.create_fake_demo(joint_env, num_steps=3, seed=42)
        
        # Convert twice
        conversion1 = convert_joint_demo_to_cartesian(fake_demo, cartesian_env, joint_env)
        conversion2 = convert_joint_demo_to_cartesian(fake_demo, cartesian_env, joint_env)
        
        # Check that both conversions are identical
        for i in range(len(conversion1.timesteps)):
            action1 = conversion1.timesteps[i].executed_action
            action2 = conversion2.timesteps[i].executed_action
            
            diff = np.linalg.norm(action1 - action2)
            assert diff < 1e-10, f"Step {i}: Inconsistent conversion: {diff}"
        
        print(f"✅ Conversion consistency test passed!")


if __name__ == "__main__":
    # Run tests manually for debugging
    import sys
    
    test = TestDemoConversion()
    
    # Create environments
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
    
    environments = (joint_env, cartesian_env)
    
    print("Running Demo Conversion Tests...")
    print("="*60)
    
    try:
        print("\n1. Testing achieved conversion FK match...")
        test.test_achieved_conversion_fk_match(environments)
        
        print("\n2. Testing target conversion FK match...")
        test.test_target_conversion_fk_match(environments)
        
        print("\n3. Testing base action preservation...")
        test.test_base_action_preservation(environments)
        
        print("\n4. Testing gripper action preservation...")
        test.test_gripper_action_preservation(environments)
        
        print("\n5. Testing demo metadata preservation...")
        test.test_demo_metadata_preservation(environments)
        
        print("\n6. Testing conversion consistency...")
        test.test_conversion_consistency(environments)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    finally:
        joint_env.close()
        cartesian_env.close()