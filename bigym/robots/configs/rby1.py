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
    offset_euler=np.array([0, 0, 0]),  # Will adjust if needed
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
    offset_euler=np.array([0, 0, 0]),  # Will adjust if needed
)

# Actuator mapping for RBY1
# Wheels are velocity controlled, everything else is position controlled
RBY1_ACTUATORS = {
    # Wheel actuators (velocity controlled)
    "wheel_fr": False,  # Front right wheel
    "wheel_fl": False,  # Front left wheel
    "wheel_rr": False,  # Rear right wheel
    "wheel_rl": False,  # Rear left wheel
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
        # Wheels (4 DOF) - not directly controlled in position mode
        0, 0, 0, 0,
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
    model=ASSETS_PATH / "xmls" / "rby1" / "model_act.xml",
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
    model=ASSETS_PATH / "xmls" / "rby1" / "model_act.xml",
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