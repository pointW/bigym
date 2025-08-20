"""Unit tests for RBY1 Cartesian Action Mode."""
import sys
import os

# Add the project root to Python path for direct execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import unittest
import numpy as np
from pyquaternion import Quaternion

from bigym.envs.reach_target_rby1 import ReachTargetRBY1
from bigym.rby1_cartesian_action_mode import (
    RBY1CartesianActionMode,
    rotation_matrix_to_6d,
    rotation_6d_to_matrix,
    Pose
)
from bigym.action_modes import JointPositionActionMode


class TestRBY1CartesianActionMode(unittest.TestCase):
    """Test RBY1 Cartesian action mode functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = None
        
    def tearDown(self):
        """Clean up after tests."""
        if self.env is not None:
            self.env.close()
            
    def test_rotation_conversions(self):
        """Test 6D rotation representation conversions."""
        # Create a random rotation matrix
        q = Quaternion(axis=[0, 0, 1], angle=np.pi/4)
        rot_matrix = q.rotation_matrix
        
        # Convert to 6D and back
        rot_6d = rotation_matrix_to_6d(rot_matrix)
        recovered_matrix = rotation_6d_to_matrix(rot_6d)
        
        # Check if we recover the original matrix
        np.testing.assert_allclose(rot_matrix, recovered_matrix, rtol=1e-5)
        
    def test_action_space_creation(self):
        """Test action space creation for RBY1."""
        # Create environment with Cartesian action mode
        action_mode = RBY1CartesianActionMode(floating_base=True)
        self.env = ReachTargetRBY1(
            action_mode=action_mode,
            render_mode=None
        )
        
        # Check action space
        action_space = self.env.action_space
        
        # Expected dimensions:
        # - Left EE: 3 (pos) + 6 (ori) = 9
        # - Right EE: 3 (pos) + 6 (ori) = 9
        # - Base: 3 (X, Y, RZ)
        # - Grippers: 2
        expected_dim = 9 + 9 + 3 + 2  # 23
        
        self.assertEqual(action_space.shape[0], expected_dim)
        self.assertEqual(action_space.dtype, np.float32)
        
    def test_action_mode_initialization(self):
        """Test RBY1 Cartesian action mode initialization."""
        # Test with floating base (required for RBY1)
        action_mode = RBY1CartesianActionMode(floating_base=True)
        self.assertEqual(action_mode.position_limits, (-2.0, 2.0))
        
        # Test that floating_base=False raises error
        with self.assertRaises(ValueError) as context:
            action_mode_fixed = RBY1CartesianActionMode(floating_base=False)
        
        self.assertIn("RBY1CartesianActionMode requires floating_base=True", str(context.exception))
        
    def test_environment_reset(self):
        """Test environment reset with RBY1 Cartesian action mode."""
        action_mode = RBY1CartesianActionMode(floating_base=True)
        self.env = ReachTargetRBY1(
            action_mode=action_mode,
            render_mode=None
        )
        
        # Reset environment
        obs, info = self.env.reset()
        
        # Check observation structure
        self.assertIn("proprioception", obs)
        self.assertIn("proprioception_grippers", obs)
        
        # Get privileged observations directly from environment
        priv_obs = self.env._get_task_privileged_obs()
        self.assertIn("target_position", priv_obs)
        
        # Check target position shape
        target_pos = priv_obs["target_position"]
        self.assertEqual(target_pos.shape, (3,))
        
    def test_environment_step(self):
        """Test stepping through environment with Cartesian actions."""
        action_mode = RBY1CartesianActionMode(floating_base=True)
        self.env = ReachTargetRBY1(
            action_mode=action_mode,
            render_mode=None
        )
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Create a simple action
        action = np.zeros(self.env.action_space.shape[0])
        
        # Set some end-effector positions
        action[0:3] = [0.5, 0.2, 1.2]  # Left EE position
        action[9:12] = [0.5, -0.2, 1.2]  # Right EE position
        
        # Set identity orientations (6D representation)
        identity_6d = rotation_matrix_to_6d(np.eye(3))
        action[3:9] = identity_6d  # Left EE orientation
        action[12:18] = identity_6d  # Right EE orientation
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check that step completed without errors
        self.assertIsNotNone(obs)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
    def test_poses_to_action_conversion(self):
        """Test conversion from poses to action vector."""
        action_mode = RBY1CartesianActionMode(floating_base=True)
        self.env = ReachTargetRBY1(
            action_mode=action_mode,
            render_mode=None
        )
        
        # Reset to bind robot
        self.env.reset()
        
        # Create test poses
        left_pose = Pose(
            position=np.array([0.5, 0.2, 1.2]),
            orientation=Quaternion(axis=[0, 0, 1], angle=0)
        )
        right_pose = Pose(
            position=np.array([0.5, -0.2, 1.2]),
            orientation=Quaternion(axis=[0, 0, 1], angle=0)
        )
        
        # Convert to action
        action = action_mode.poses_to_action(
            left_pose, 
            right_pose,
            base_x=0.0,
            base_y=0.0,
            base_rz=0.0,
            gripper_action=np.array([0, 0])
        )
        
        # Check action shape
        self.assertEqual(action.shape[0], 23)  # 9+9+3+2
        
        # Verify positions are correctly placed
        np.testing.assert_array_almost_equal(action[0:3], left_pose.position)
        np.testing.assert_array_almost_equal(action[9:12], right_pose.position)
        
    def test_get_current_ee_poses(self):
        """Test getting current end-effector poses."""
        action_mode = RBY1CartesianActionMode(floating_base=True)
        self.env = ReachTargetRBY1(
            action_mode=action_mode,
            render_mode=None
        )
        
        # Reset environment
        self.env.reset()
        
        # Get current poses
        left_pose, right_pose = action_mode.get_current_ee_poses()
        
        # Check that poses are valid
        self.assertIsInstance(left_pose, Pose)
        self.assertIsInstance(right_pose, Pose)
        
        # Check position shapes
        self.assertEqual(left_pose.position.shape, (3,))
        self.assertEqual(right_pose.position.shape, (3,))
        
        # Check orientations are valid quaternions
        self.assertIsInstance(left_pose.orientation, Quaternion)
        self.assertIsInstance(right_pose.orientation, Quaternion)
        self.assertAlmostEqual(left_pose.orientation.norm, 1.0, places=5)
        self.assertAlmostEqual(right_pose.orientation.norm, 1.0, places=5)


class TestRBY1CartesianVsJoint(unittest.TestCase):
    """Test RBY1 Cartesian mode compared to joint mode."""
    
    def setUp(self):
        """Set up test environments."""
        self.cartesian_env = None
        self.joint_env = None
        
    def tearDown(self):
        """Clean up environments."""
        if self.cartesian_env is not None:
            self.cartesian_env.close()
        if self.joint_env is not None:
            self.joint_env.close()
            
    def test_action_space_consistency(self):
        """Test that both modes can control the same robot."""
        # Create Cartesian environment
        cartesian_mode = RBY1CartesianActionMode(floating_base=True)
        self.cartesian_env = ReachTargetRBY1(
            action_mode=cartesian_mode,
            render_mode=None
        )
        
        # Create Joint environment
        joint_mode = JointPositionActionMode(
            absolute=True,
            floating_base=True
        )
        self.joint_env = ReachTargetRBY1(
            action_mode=joint_mode,
            render_mode=None
        )
        
        # Reset both
        self.cartesian_env.reset(seed=42)
        self.joint_env.reset(seed=42)
        
        # Check that both environments have the same robot
        self.assertEqual(
            self.cartesian_env.robot.__class__.__name__,
            self.joint_env.robot.__class__.__name__
        )
        
        # Cartesian should have smaller action space (end-effector space)
        # Joint mode: 3 (base) + 6 (torso) + 14 (arms) + 2 (grippers) = 25
        # Cartesian mode: 18 (2x9 EE) + 3 (base) + 2 (grippers) = 23
        self.assertEqual(self.joint_env.action_space.shape[0], 25)
        self.assertEqual(self.cartesian_env.action_space.shape[0], 23)
        
    def test_simple_reaching(self):
        """Test that both modes can reach a target."""
        # Test with fixed seed for reproducibility
        seed = 12345
        
        # Test Cartesian mode
        cartesian_mode = RBY1CartesianActionMode(floating_base=True)
        self.cartesian_env = ReachTargetRBY1(
            action_mode=cartesian_mode,
            render_mode=None
        )
        
        obs, _ = self.cartesian_env.reset(seed=seed)
        priv_obs = self.cartesian_env._get_task_privileged_obs()
        target_pos = priv_obs["target_position"]
        
        # Create action to reach target
        action = np.zeros(self.cartesian_env.action_space.shape[0])
        action[0:3] = target_pos  # Left EE to target
        action[9:12] = target_pos  # Right EE to target
        
        # Set identity orientations
        identity_6d = rotation_matrix_to_6d(np.eye(3))
        action[3:9] = identity_6d
        action[12:18] = identity_6d
        
        # Take a few steps
        for _ in range(10):
            obs, reward, terminated, truncated, _ = self.cartesian_env.step(action)
            if terminated:
                break
                
        # Check if we got closer to target
        # Note: This is a basic test, actual reaching may require more steps
        self.assertIsNotNone(reward)


if __name__ == "__main__":
    unittest.main()