"""Unit tests for RBY1 ReachTarget environment."""
import sys
import os

# Add the project root to Python path for direct execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import unittest
import numpy as np

from bigym.envs.reach_target_rby1 import (
    ReachTargetRBY1,
    ReachTargetSingleRBY1,
    ReachTargetDualRBY1
)
from bigym.action_modes import JointPositionActionMode
from bigym.rby1_cartesian_action_mode import RBY1CartesianActionMode
from bigym.const import HandSide


class TestRBY1ReachTarget(unittest.TestCase):
    """Test RBY1 ReachTarget environment."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = None
        
    def tearDown(self):
        """Clean up after tests."""
        if self.env is not None:
            self.env.close()
            
    def test_environment_creation(self):
        """Test creating RBY1 ReachTarget environment."""
        # Create with joint position mode
        self.env = ReachTargetRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None
        )
        
        # Check robot type
        self.assertEqual(self.env.robot.__class__.__name__, "RBY1")
        
        # Check targets exist
        self.assertEqual(len(self.env.targets), 1)
        
    def test_environment_reset(self):
        """Test environment reset."""
        self.env = ReachTargetRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None
        )
        
        # Reset environment
        obs, info = self.env.reset(seed=42)
        
        # Check observation structure
        self.assertIn("proprioception", obs)
        self.assertIn("proprioception_grippers", obs)
        
        # Get privileged obs directly from environment
        priv_obs = self.env._get_task_privileged_obs()
        self.assertIn("target_position", priv_obs)
        
        # Check target position is valid
        target_pos = priv_obs["target_position"]
        self.assertEqual(target_pos.shape, (3,))
        
        # Target should be near expected position with some randomization
        expected_base = np.array([0.6, 0, 1.2])
        distance = np.linalg.norm(target_pos - expected_base)
        self.assertLess(distance, 0.3)  # Should be within bounds
        
    def test_environment_step(self):
        """Test stepping through environment."""
        self.env = ReachTargetRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None
        )
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Create zero action
        action = np.zeros(self.env.action_space.shape[0])
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check outputs
        self.assertIsNotNone(obs)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
    def test_target_reaching(self):
        """Test target reaching detection."""
        self.env = ReachTargetRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=False),
            render_mode=None
        )
        
        # Reset environment
        self.env.reset()
        
        # Check initial success (should be False)
        success = self.env._success()
        self.assertFalse(success)
        
        # Get target position
        target = self.env.targets[0]
        target_pos = target.body.get_position()
        
        # Check distance calculation
        left_hand_pos = self.env.robot.get_hand_pos(HandSide.LEFT)
        distance = target.distance(left_hand_pos)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
        
    def test_single_target_variant(self):
        """Test single target variant."""
        self.env = ReachTargetSingleRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None
        )
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Check only one target
        self.assertEqual(len(self.env.targets), 1)
        
        # Check target is for left hand only
        target = self.env.targets[0]
        self.assertEqual(target._config.target_hands, [HandSide.LEFT])
        
    def test_dual_target_variant(self):
        """Test dual target variant."""
        self.env = ReachTargetDualRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None
        )
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Check two targets
        self.assertEqual(len(self.env.targets), 2)
        
        # Check targets are for different hands
        left_target = self.env.targets[0]
        right_target = self.env.targets[1]
        
        self.assertEqual(left_target._config.target_hands, [HandSide.LEFT])
        self.assertEqual(right_target._config.target_hands, [HandSide.RIGHT])
        
        # Check targets have different positions
        left_pos = left_target.body.get_position()
        right_pos = right_target.body.get_position()
        
        # Y positions should be different (left positive, right negative)
        self.assertGreater(left_pos[1], 0)
        self.assertLess(right_pos[1], 0)
        
    def test_with_cartesian_action_mode(self):
        """Test environment with Cartesian action mode."""
        action_mode = RBY1CartesianActionMode(floating_base=True)
        self.env = ReachTargetRBY1(
            action_mode=action_mode,
            render_mode=None
        )
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Check action space is Cartesian
        # Expected: 18 (EE) + 3 (base) + 2 (grippers) = 23
        self.assertEqual(self.env.action_space.shape[0], 23)
        
        # Create Cartesian action
        action = np.zeros(23)
        
        # Get target position from environment
        priv_obs = self.env._get_task_privileged_obs()
        target_pos = priv_obs["target_position"]
        action[0:3] = target_pos  # Left EE
        action[9:12] = target_pos  # Right EE
        
        # Set identity orientations (6D)
        action[3:9] = [1, 0, 0, 0, 1, 0]  # Left
        action[12:18] = [1, 0, 0, 0, 1, 0]  # Right
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check step completed
        self.assertIsNotNone(obs)
        
    def test_workspace_bounds(self):
        """Test that targets are within RBY1's workspace."""
        self.env = ReachTargetRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=False),
            render_mode=None
        )
        
        # Test multiple resets to check target positions
        for _ in range(10):
            obs, _ = self.env.reset()
            # Get privileged observations directly from environment
            priv_obs = self.env._get_task_privileged_obs()
            target_pos = priv_obs["target_position"]
            
            # Check position is reasonable for RBY1
            # X should be positive (in front)
            self.assertGreater(target_pos[0], 0.3)
            self.assertLess(target_pos[0], 1.0)
            
            # Y should be within arm reach
            self.assertGreater(target_pos[1], -0.5)
            self.assertLess(target_pos[1], 0.5)
            
            # Z should be at reasonable height
            self.assertGreater(target_pos[2], 0.8)
            self.assertLess(target_pos[2], 1.6)
            
    def test_gripper_integration(self):
        """Test that grippers are properly integrated."""
        self.env = ReachTargetRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None
        )
        
        # Reset environment
        self.env.reset()
        
        # Check robot has grippers
        self.assertGreater(len(self.env.robot.grippers), 0)
        
        # Check gripper control in action space
        action = np.zeros(self.env.action_space.shape[0])
        
        # Set gripper actions (last 2 dimensions)
        action[-2] = 1.0  # Left gripper close
        action[-1] = 0.0  # Right gripper open
        
        # Step should work with gripper actions
        obs, _, _, _, _ = self.env.step(action)
        self.assertIsNotNone(obs)


class TestRBY1EnvironmentConsistency(unittest.TestCase):
    """Test consistency between RBY1 and H1 environments."""
    
    def test_similar_structure(self):
        """Test that RBY1 environment has similar structure to H1."""
        from bigym.envs.reach_target import ReachTarget as H1ReachTarget
        
        # Create both environments
        h1_env = H1ReachTarget(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None
        )
        
        rby1_env = ReachTargetRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None
        )
        
        try:
            # Reset both
            h1_obs, _ = h1_env.reset()
            rby1_obs, _ = rby1_env.reset()
            
            # Check core observation keys are present in both
            # Note: H1 includes privileged_obs in observation, RBY1 doesn't
            core_keys = {"proprioception", "proprioception_grippers"}
            self.assertTrue(core_keys.issubset(set(h1_obs.keys())))
            self.assertTrue(core_keys.issubset(set(rby1_obs.keys())))
            
            # Get privileged observations from both environments
            h1_priv = h1_env._get_task_privileged_obs()
            rby1_priv = rby1_env._get_task_privileged_obs()
            
            # Both should have target position in privileged obs
            self.assertIn("target_position", h1_priv)
            self.assertIn("target_position", rby1_priv)
            
            # Both should have targets
            self.assertEqual(len(h1_env.targets), len(rby1_env.targets))
            
            # Both should have similar tolerance
            self.assertEqual(h1_env.TOLERANCE, rby1_env.TOLERANCE)
            
        finally:
            h1_env.close()
            rby1_env.close()


if __name__ == "__main__":
    unittest.main()