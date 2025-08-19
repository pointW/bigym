"""RBY1 ReachTarget environment."""
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces

from bigym.bigym_env import BiGymEnv
from bigym.const import HandSide
from bigym.envs.reach_target import Target, TargetConfig
from bigym.robots.configs.rby1 import RBY1


class ReachTargetRBY1(BiGymEnv):
    """Reach target environment for RBY1 robot.
    
    This environment is specifically designed for RBY1:
    - Uses RBY1 robot configuration with Robotiq grippers
    - Adapted target positions for RBY1's workspace
    - Wheeled base can move to reach targets
    """
    
    # Override default robot to use RBY1
    DEFAULT_ROBOT = RBY1
    
    # Target configuration adapted for RBY1's workspace
    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT, HandSide.RIGHT],
            reset_position=np.array([0.6, 0, 1.2]),  # Adjusted for RBY1 height
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]
    
    POSITION_BOUNDS = np.array([0.15, 0.15, 0.15])  # Larger bounds for RBY1
    TOLERANCE = 0.1  # Same tolerance as H1
    
    def _initialize_env(self):
        """Initialize the environment."""
        self.targets: list[Target] = []
        for config in self.TARGET_CONFIGS:
            self.targets.append(Target(self._mojo, self.robot, config))
    
    def _on_reset(self):
        """Reset target positions."""
        for target in self.targets:
            offset = np.random.uniform(-self.POSITION_BOUNDS, self.POSITION_BOUNDS)
            target.reset_position(offset)
            target.set_highlight(False)
    
    def _success(self) -> bool:
        """Check if task is successful."""
        for target in self.targets:
            if not target.is_reached(self.TOLERANCE):
                return False
        return True
    
    def _on_step(self):
        """Highlight spheres even in fast mode."""
        self._success()
    
    def _get_task_privileged_obs_space(self):
        """Get observation space for privileged observations."""
        return {
            "target_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        }
    
    def _get_task_privileged_obs(self):
        """Get privileged observations."""
        return {
            "target_position": np.array(
                self.targets[0].body.get_position(), np.float32
            ).copy()
        }


class ReachTargetSingleRBY1(ReachTargetRBY1):
    """Reach the target with specific wrist (RBY1 version)."""
    
    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT],
            reset_position=np.array([0.6, 0.2, 1.2]),  # Adjusted for RBY1
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]


class ReachTargetDualRBY1(ReachTargetRBY1):
    """Reach 2 targets, one with each arm (RBY1 version)."""
    
    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT],
            reset_position=np.array([0.6, 0.3, 1.2]),  # Left target
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        ),
        TargetConfig(
            target_hands=[HandSide.RIGHT],
            reset_position=np.array([0.6, -0.3, 1.2]),  # Right target
            color_default=np.array([0, 0, 0.3, 1]),
            color_highlight=np.array([0, 0, 1, 1]),
        ),
    ]