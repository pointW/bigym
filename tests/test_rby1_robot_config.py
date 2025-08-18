"""Unit tests for RBY1 robot configuration."""
import pytest
import numpy as np
from pathlib import Path

from bigym.robots.configs.rby1 import (
    RBY1_CONFIG,
    RBY1_FINE_MANIPULATION_CONFIG,
    RBY1_LEFT_ARM,
    RBY1_RIGHT_ARM,
    RBY1_ACTUATORS,
    RBY1_FLOATING_BASE,
    RBY1_FULL_BODY,
)
from bigym.action_modes import PelvisDof
from bigym.const import HandSide
from mojo.elements.consts import JointType


class TestRBY1Configuration:
    """Test RBY1 robot configuration."""
    
    def test_rby1_basic_config(self):
        """Test basic RBY1 configuration properties."""
        # Check model file path
        assert RBY1_CONFIG.model.name == "rby1.xml"
        assert "xmls" in str(RBY1_CONFIG.model.parent)
        
        # Check pelvis body
        assert RBY1_CONFIG.pelvis_body == "base"
        
        # Check delta range and position kp
        assert RBY1_CONFIG.delta_range == (-0.1, 0.1)
        assert RBY1_CONFIG.position_kp == 300
    
    def test_arm_configurations(self):
        """Test arm configurations for RBY1."""
        # Left arm
        assert RBY1_LEFT_ARM.site == "end_effector_l"
        assert len(RBY1_LEFT_ARM.links) == 7  # 7 DOF arm
        assert RBY1_LEFT_ARM.links[0] == "link_left_arm_0"
        assert RBY1_LEFT_ARM.links[-1] == "link_left_arm_6"
        assert RBY1_LEFT_ARM.wrist_dof is None  # No separate wrist
        
        # Right arm
        assert RBY1_RIGHT_ARM.site == "end_effector_r"
        assert len(RBY1_RIGHT_ARM.links) == 7  # 7 DOF arm
        assert RBY1_RIGHT_ARM.links[0] == "link_right_arm_0"
        assert RBY1_RIGHT_ARM.links[-1] == "link_right_arm_6"
        assert RBY1_RIGHT_ARM.wrist_dof is None
    
    def test_actuator_mapping(self):
        """Test actuator configuration."""
        # Check wheel actuators (should be False - velocity controlled)
        assert RBY1_ACTUATORS["wheel_fr"] == False
        assert RBY1_ACTUATORS["wheel_fl"] == False
        assert RBY1_ACTUATORS["wheel_rr"] == False
        assert RBY1_ACTUATORS["wheel_rl"] == False
        
        # Check torso actuators (should be True - position controlled)
        for i in range(6):
            assert RBY1_ACTUATORS[f"torso_{i}"] == True
        
        # Check arm actuators (should be True - position controlled)
        for i in range(7):
            assert RBY1_ACTUATORS[f"left_arm_{i}"] == True
            assert RBY1_ACTUATORS[f"right_arm_{i}"] == True
        
        # Total number of actuators
        assert len(RBY1_ACTUATORS) == 4 + 6 + 7 + 7  # wheels + torso + 2 arms
    
    def test_floating_base_config(self):
        """Test floating base configuration for wheeled robot."""
        # Check DOFs
        assert PelvisDof.X in RBY1_FLOATING_BASE.dofs
        assert PelvisDof.Y in RBY1_FLOATING_BASE.dofs
        assert PelvisDof.RZ in RBY1_FLOATING_BASE.dofs
        
        # Check X DOF
        x_dof = RBY1_FLOATING_BASE.dofs[PelvisDof.X]
        assert x_dof.joint_type == JointType.SLIDE
        assert np.allclose(x_dof.axis, (1, 0, 0))
        assert x_dof.stiffness == 0
        
        # Check Y DOF
        y_dof = RBY1_FLOATING_BASE.dofs[PelvisDof.Y]
        assert y_dof.joint_type == JointType.SLIDE
        assert np.allclose(y_dof.axis, (0, 1, 0))
        assert y_dof.stiffness == 0
        
        # Check RZ DOF
        rz_dof = RBY1_FLOATING_BASE.dofs[PelvisDof.RZ]
        assert rz_dof.joint_type == JointType.HINGE
        assert np.allclose(rz_dof.axis, (0, 0, 1))
        assert rz_dof.stiffness == 0
        
        # Check delta ranges
        assert RBY1_FLOATING_BASE.delta_range_position == (-0.2, 0.2)
        assert RBY1_FLOATING_BASE.delta_range_rotation == (-0.5, 0.5)
    
    def test_full_body_config(self):
        """Test full body configuration."""
        # Check offset position
        assert np.allclose(RBY1_FULL_BODY.offset_position, [0, 0, 0.3])
        
        # Check reset state dimensions
        reset_state = RBY1_FULL_BODY.reset_state
        assert len(reset_state) == 4 + 6 + 7 + 7  # wheels + torso + 2 arms
        
        # All joints should start at 0 (neutral position)
        assert np.allclose(reset_state, 0)
    
    def test_gripper_configuration(self):
        """Test gripper configurations."""
        # Standard RBY1 uses ROBOTIQ_2F85 grippers
        assert RBY1_CONFIG.gripper is not None
        # The actual gripper name depends on ROBOTIQ_2F85 configuration
        
        # Fine manipulation variant
        assert RBY1_FINE_MANIPULATION_CONFIG.gripper is not None
    
    def test_config_consistency(self):
        """Test configuration consistency."""
        # Both configs should use same arms
        assert RBY1_CONFIG.arms[HandSide.LEFT] == RBY1_LEFT_ARM
        assert RBY1_CONFIG.arms[HandSide.RIGHT] == RBY1_RIGHT_ARM
        assert RBY1_FINE_MANIPULATION_CONFIG.arms[HandSide.LEFT] == RBY1_LEFT_ARM
        assert RBY1_FINE_MANIPULATION_CONFIG.arms[HandSide.RIGHT] == RBY1_RIGHT_ARM
        
        # Both should use same floating base
        assert RBY1_CONFIG.floating_base == RBY1_FLOATING_BASE
        assert RBY1_FINE_MANIPULATION_CONFIG.floating_base == RBY1_FLOATING_BASE
        
        # Both should use same actuators
        assert RBY1_CONFIG.actuators == RBY1_ACTUATORS
        assert RBY1_FINE_MANIPULATION_CONFIG.actuators == RBY1_ACTUATORS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])