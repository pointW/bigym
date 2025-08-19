"""RBY1 Robot Configuration."""
import numpy as np
from mojo.elements.consts import JointType

from bigym.action_modes import PelvisDof
from bigym.const import ASSETS_PATH, HandSide

from bigym.robots.config import (
    ArmConfig,
    FloatingBaseConfig,
    RobotConfig,
    FullBodyConfig,
)
from bigym.robots.configs.robotiq import ROBOTIQ_2F85, ROBOTIQ_2F85_FINE_MANIPULATION
from bigym.robots.robot import Robot
from bigym.utils.dof import Dof


# RBY1 has 7 DOF arms without explicit wrist joint at the end
# The 7th DOF is the last arm joint (arm_6)
RBY1_LEFT_ARM = ArmConfig(
    site="end_effector_l",  # Site name from the XML
    links=[
        "link_left_arm_0",
        "link_left_arm_1",
        "link_left_arm_2",
        "link_left_arm_3",
        "link_left_arm_4",
        "link_left_arm_5",
        "link_left_arm_6",
    ],
    wrist_dof=None,  # No separate wrist joint, arm_6 serves as wrist rotation
    offset_euler=np.array([np.pi, np.pi/2, 0]),  # Same as H1 for gripper orientation
    offset_position=np.array([0, 0, 0]),  # Gripper attachment offset
)

RBY1_RIGHT_ARM = ArmConfig(
    site="end_effector_r",  # Site name from the XML
    links=[
        "link_right_arm_0",
        "link_right_arm_1",
        "link_right_arm_2",
        "link_right_arm_3",
        "link_right_arm_4",
        "link_right_arm_5",
        "link_right_arm_6",
    ],
    wrist_dof=None,  # No separate wrist joint
    offset_euler=np.array([np.pi / 2, np.pi / 2, 0]),  # Same as H1 for gripper orientation
    offset_position=np.array([0, 0, 0]),  # Gripper attachment offset
)

# Actuator mapping for RBY1
# Note: Wheels are no longer actuated - base is controlled via mocap
RBY1_ACTUATORS = {
    # Torso actuators
    "torso_0": True,  # Torso joint 0 (roll)
    "torso_1": True,  # Torso joint 1 (pitch)
    "torso_2": True,  # Torso joint 2 (pitch)
    "torso_3": True,  # Torso joint 3 (pitch)
    "torso_4": True,  # Torso joint 4 (roll)
    "torso_5": True,  # Torso joint 5 (yaw)
    # Right arm actuators
    "right_arm_0": True,
    "right_arm_1": True,
    "right_arm_2": True,
    "right_arm_3": True,
    "right_arm_4": True,
    "right_arm_5": True,
    "right_arm_6": True,
    # Left arm actuators
    "left_arm_0": True,
    "left_arm_1": True,
    "left_arm_2": True,
    "left_arm_3": True,
    "left_arm_4": True,
    "left_arm_5": True,
    "left_arm_6": True,
}

# Floating base configuration for wheeled robot
# X, Y movement and rotation around Z (theta)
# RBY1 wheels provide holonomic base movement
STIFFNESS_XY = 0  # Free movement for wheeled base
RBY1_FLOATING_BASE = FloatingBaseConfig(
    dofs={
        PelvisDof.X: Dof(
            joint_type=JointType.SLIDE,
            axis=(1, 0, 0),
            stiffness=STIFFNESS_XY,
        ),
        PelvisDof.Y: Dof(
            joint_type=JointType.SLIDE,
            axis=(0, 1, 0),
            stiffness=STIFFNESS_XY,
        ),
        PelvisDof.RZ: Dof(
            joint_type=JointType.HINGE,
            axis=(0, 0, 1),
            stiffness=0,  # Free rotation
        ),
    },
    delta_range_position=(-0.2, 0.2),  # Delta range for position control
    delta_range_rotation=(-0.5, 0.5),  # Delta range for rotation control
    animated_legs_class=None,  # No animated legs for wheeled robot
)

# Full body configuration (for completeness, though we mainly use upper body)
RBY1_FULL_BODY = FullBodyConfig(
    offset_position=np.array([0, 0, 0.3]),  # Base height is approximately 0.3m
    reset_state=np.array([
        # No wheel actuators anymore - base controlled via mocap
        # Torso (6 DOF) - neutral position
        0, 0, 0, 0, 0, 0,
        # Right arm (7 DOF) - neutral/rest position
        0, 0, 0, 0, 0, 0, 0,
        # Left arm (7 DOF) - neutral/rest position
        0, 0, 0, 0, 0, 0, 0,
    ]),
)

# Main robot configuration
RBY1_CONFIG = RobotConfig(
    model=ASSETS_PATH / "rby1" / "model_act_consolidated.xml",
    delta_range=(-0.1, 0.1),
    position_kp=300,
    pelvis_body="base",  # RBY1 base body
    full_body=RBY1_FULL_BODY,
    floating_base=RBY1_FLOATING_BASE,
    gripper=ROBOTIQ_2F85,  # Using H1 grippers for consistency
    arms={HandSide.LEFT: RBY1_LEFT_ARM, HandSide.RIGHT: RBY1_RIGHT_ARM},
    actuators=RBY1_ACTUATORS,
    cameras=[],  # Will add cameras as needed
    namespaces_to_remove=[],
)

# Fine manipulation variant with different gripper settings
RBY1_FINE_MANIPULATION_CONFIG = RobotConfig(
    model=ASSETS_PATH / "rby1" / "model_act_consolidated.xml",
    delta_range=(-0.1, 0.1),
    position_kp=300,
    pelvis_body="base",
    full_body=RBY1_FULL_BODY,
    floating_base=RBY1_FLOATING_BASE,
    gripper=ROBOTIQ_2F85_FINE_MANIPULATION,
    arms={HandSide.LEFT: RBY1_LEFT_ARM, HandSide.RIGHT: RBY1_RIGHT_ARM},
    actuators=RBY1_ACTUATORS,
    cameras=[],
    namespaces_to_remove=[],
)


class RBY1(Robot):
    """RBY1 Robot with Robotiq grippers."""

    def __init__(self, action_mode, mojo=None):
        """Initialize RBY1 robot with mocap base control."""
        super().__init__(action_mode, mojo)
        
        # Add mocap body for base control after robot is loaded
        if self._mojo and self._mojo.root_element:
            # Check if base_target doesn't already exist
            existing_target = None
            try:
                existing_target = self._mojo.root_element.mjcf.find("body", "base_target")
            except:
                pass
            
            if not existing_target:
                # Add mocap body at world level
                worldbody = self._mojo.root_element.mjcf.worldbody
                base_target = worldbody.add("body", name="base_target", mocap=True)
                base_target.add("geom", type="box", size=[0.1, 0.1, 0.05], 
                               contype=0, conaffinity=0, rgba=[0.8, 0.2, 0.2, 0.5])
                
                # Add weld equality constraint
                # Get or create equality section
                if not hasattr(self._mojo.root_element.mjcf, "equality"):
                    self._mojo.root_element.mjcf.add("equality")
                
                # Reference the base body with namespace
                # The body gets prefixed with "rby1/" when included in environment
                base_body_name = "rby1/base"
                self._mojo.root_element.mjcf.equality.add("weld", body1="base_target", body2=base_body_name)

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return RBY1_CONFIG


class RBY1FineManipulation(Robot):
    """RBY1 Robot with Robotiq gripper for fine manipulations."""

    def __init__(self, action_mode, mojo=None):
        """Initialize RBY1 robot with mocap base control."""
        super().__init__(action_mode, mojo)
        
        # Add mocap body for base control after robot is loaded
        if self._mojo and self._mojo.root_element:
            # Check if base_target doesn't already exist
            existing_target = None
            try:
                existing_target = self._mojo.root_element.mjcf.find("body", "base_target")
            except:
                pass
            
            if not existing_target:
                # Add mocap body at world level
                worldbody = self._mojo.root_element.mjcf.worldbody
                base_target = worldbody.add("body", name="base_target", mocap=True)
                base_target.add("geom", type="box", size=[0.1, 0.1, 0.05], 
                               contype=0, conaffinity=0, rgba=[0.8, 0.2, 0.2, 0.5])
                
                # Add weld equality constraint
                # Get or create equality section
                if not hasattr(self._mojo.root_element.mjcf, "equality"):
                    self._mojo.root_element.mjcf.add("equality")
                
                # Reference the base body with namespace
                # The body gets prefixed with "rby1/" when included in environment
                base_body_name = "rby1/base"
                self._mojo.root_element.mjcf.equality.add("weld", body1="base_target", body2=base_body_name)

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return RBY1_FINE_MANIPULATION_CONFIG