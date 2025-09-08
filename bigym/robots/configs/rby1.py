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
    offset_euler=np.array([np.pi / 2, np.pi / 2, 0]),
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
    offset_euler=np.array([np.pi / 2, np.pi / 2, 0]),
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

# Full body configuration
RBY1_FULL_BODY = FullBodyConfig(
    offset_position=np.array([0, 0, 0]),  # Keep base at ground level for wheeled robot
    # reset_state=np.array([
    #     # Torso (6 DOF) - minimal movement for H1 pose match (position+orientation)
    #     0.0296, 0.0177, 0.0000, -0.0177, -0.0296, -0.0000,
    #     # Right arm (7 DOF) - matched to H1 pose (position+orientation) with scale=1.3
    #     0.5381, 0.0500, 0.2096, -2.0186, 0.3451, -0.0468, -0.2460,
    #     # 0.5100, -0.0967, 0.1544, -2.0225, -0.3253, 0.0430, 0.2704,
    #     # Left arm (7 DOF) - matched to H1 pose (position+orientation) with scale=1.3
    #     0.5100, 0.0967, -0.1544, -2.0225, 0.3253, -0.0430, -0.2704,
    # ]),
#     reset_state=np.array([
#     # Torso (6 DOF) - fixed at zero
#     0.0000,
#     0.0000,
#     0.0000,
#     0.0000,
#     0.0000,
#     0.0000,
#     # Right arm (7 DOF)
#     0.3392,
#     0.0284,
#     0.2281,
#     -1.2886,
#     0.3424,
#     0.2172,
#     -0.2441,
#     # Left arm (7 DOF)
#     0.3329,
#     -0.0017,
#     -0.2026,
#     -1.2914,
#     0.3109,
#     0.2232,
#     -0.2684
# ])
reset_state=np.array([
    # Torso (6 DOF) - fixed at zero
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    # Right arm (7 DOF)
    0.4768,
    -0.0089,
    0.1617,
    -1.9939,
    0.3179,
    -0.0256,
    -0.2916,
    # Left arm (7 DOF)
    0.4706,
    0.0405,
    -0.1765,
    -1.9902,
    0.2878,
    -0.0346,
    -0.2912
])
)

# Main robot configuration
RBY1_CONFIG = RobotConfig(
    model=ASSETS_PATH / "rby1" / "model_act_consolidated.xml",
    delta_range=(-0.1, 0.1),
    position_kp=300,
    pelvis_body="base",  # RBY1 base body
    full_body=RBY1_FULL_BODY,
    floating_base=None,  # RBY1 doesn't use floating base
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
    floating_base=None,  # RBY1 doesn't use floating base
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
        # Set desired scale before loading
        self._model_scale = 1.0
        super().__init__(action_mode, mojo)
        
        # Fix limb_actuators for RBY1 with namespace
        # This is needed for custom action modes that don't populate limb_actuators
        self._fix_limb_actuators()
        
        # Add mocap body for base control after robot is loaded
        if self._mojo and self._mojo.root_element:
            # Check if base_target doesn't already exist
            existing_target = None
            try:
                existing_target = self._mojo.root_element.mjcf.find("body", "base_target")
            except:
                pass
            
            if not existing_target:
                # Add mocap body at world level, at ground level
                worldbody = self._mojo.root_element.mjcf.worldbody
                base_target = worldbody.add("body", name="base_target", mocap=True, 
                                           pos=[0, 0, 0])  # At ground level
                base_target.add("geom", type="box", size=[0.1, 0.1, 0.05], 
                               contype=0, conaffinity=0, rgba=[0.8, 0.2, 0.2, 0.5])
                
                # Add weld constraint to connect mocap to base
                # This is needed for the mocap to actually control the robot base
                # Add weld constraint between base_target and robot base
                # The robot base is namespaced as "rby1/base"
                self._mojo.root_element.mjcf.equality.add(
                    "weld", 
                    body1="base_target", 
                    body2="rby1/base",
                    solimp=[0.99, 0.999, 0.001, 0.5, 2],
                    solref=[0.001, 1],
                )

    def _fix_limb_actuators(self):
        """Fix limb actuators for RBY1 with namespace."""
        if not hasattr(self, '_mojo') or not self._mojo:
            return
        
        # RBY1 actuators have "rby1/" prefix in environment context
        # but config doesn't have the prefix, so we need to manually populate limb_actuators
        if not hasattr(self, '_limb_actuators'):
            self._limb_actuators = []
        
        # If limb_actuators is already populated (by parent), don't override
        if self._limb_actuators:
            return
        
        # Get all actuators from root element MJCF  
        all_actuators = []
        if hasattr(self._mojo, 'root_element'):
            root = self._mojo.root_element
            # Use mjcf attribute if available
            if hasattr(root, 'mjcf'):
                root = root.mjcf
            # Now find all actuators
            if hasattr(root, 'find_all'):
                all_actuators = root.find_all("actuator")
        
        # Expected actuator names (WITHOUT namespace - actuators don't have rby1/ prefix)
        expected_names = [
            "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
            "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", 
            "right_arm_4", "right_arm_5", "right_arm_6",
            "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
            "left_arm_4", "left_arm_5", "left_arm_6"
        ]
        
        for actuator in all_actuators:
            if hasattr(actuator, 'name') and actuator.name in expected_names:
                self._limb_actuators.append(actuator)
        
        # Sort by expected order
        self._limb_actuators.sort(key=lambda a: expected_names.index(a.name) if a.name in expected_names else 999)
    
    def _on_loaded(self, model):
        """Override to apply scaling before model compilation."""
        # Apply scaling to the MJCF model before it's compiled
        if hasattr(self, '_model_scale') and self._model_scale != 1.0:
            # Scale all body positions
            for body in model.find_all('body'):
                if body.pos is not None:
                    body.pos = [p * self._model_scale for p in body.pos]
            
            # Scale all geom sizes (collision geometry)
            for geom in model.find_all('geom'):
                if geom.size is not None:
                    geom.size = [s * self._model_scale for s in geom.size]
                if hasattr(geom, 'fromto') and geom.fromto is not None:
                    geom.fromto = [f * self._model_scale for f in geom.fromto]
                # Also scale mesh if this geom uses one
                if hasattr(geom, 'mesh') and geom.mesh is not None:
                    # Geoms with meshes need their mesh scaled
                    mesh_name = geom.mesh
                    # Find the corresponding mesh asset and scale it
                    for mesh in model.find_all('mesh'):
                        if hasattr(mesh, 'name') and mesh.name == mesh_name:
                            if mesh.scale is not None:
                                mesh.scale = [s * self._model_scale for s in mesh.scale]
                            else:
                                mesh.scale = [self._model_scale] * 3
            
            # Scale all mesh assets directly 
            for mesh in model.find_all('mesh'):
                if mesh.scale is not None:
                    mesh.scale = [s * self._model_scale for s in mesh.scale]
                else:
                    mesh.scale = [self._model_scale] * 3
            
            # Scale all site positions  
            for site in model.find_all('site'):
                if site.pos is not None:
                    site.pos = [p * self._model_scale for p in site.pos]
                if site.size is not None:
                    site.size = [s * self._model_scale for s in site.size]
            
            # Scale joint ranges (for position limits)
            for joint in model.find_all('joint'):
                if joint.range is not None:
                    # Only scale positional joints, not angular ones
                    if hasattr(joint, 'type') and joint.type in ['slide', 'free']:
                        joint.range = [r * self._model_scale for r in joint.range]
                if joint.pos is not None:
                    joint.pos = [p * self._model_scale for p in joint.pos]
            
            # Scale inertial properties (optional - MuJoCo often handles this)
            for body in model.find_all('body'):
                if hasattr(body, 'inertial') and body.inertial is not None:
                    inertial = body.inertial
                    if inertial.pos is not None:
                        inertial.pos = [p * self._model_scale for p in inertial.pos]
                    # Mass scales with volume (scale^3)
                    if inertial.mass is not None:
                        inertial.mass = inertial.mass * (self._model_scale ** 3)
                    # Inertia scales with mass * length^2, so scale^5 total
                    if inertial.diaginertia is not None:
                        inertial.diaginertia = [i * (self._model_scale ** 5) for i in inertial.diaginertia]
        
        # Call parent's _on_loaded to continue normal initialization
        super()._on_loaded(model)
    
    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return RBY1_CONFIG


class RBY1FineManipulation(Robot):
    """RBY1 Robot with Robotiq gripper for fine manipulations."""

    def __init__(self, action_mode, mojo=None):
        """Initialize RBY1 robot with mocap base control."""
        # Set desired scale before loading
        self._model_scale = 1.3 
        super().__init__(action_mode, mojo)
        
        # Fix limb_actuators for RBY1 with namespace
        self._fix_limb_actuators()
        
        # Add mocap body for base control after robot is loaded
        if self._mojo and self._mojo.root_element:
            # Check if base_target doesn't already exist
            existing_target = None
            try:
                existing_target = self._mojo.root_element.mjcf.find("body", "base_target")
            except:
                pass
            
            if not existing_target:
                # Add mocap body at world level, at ground level
                worldbody = self._mojo.root_element.mjcf.worldbody
                base_target = worldbody.add("body", name="base_target", mocap=True, 
                                           pos=[0, 0, 0])  # At ground level
                base_target.add("geom", type="box", size=[0.1, 0.1, 0.05], 
                               contype=0, conaffinity=0, rgba=[0.8, 0.2, 0.2, 0.5])
                
                # Add weld constraint to connect mocap to base
                # This is needed for the mocap to actually control the robot base
                # Add weld constraint between base_target and robot base
                # The robot base is namespaced as "rby1/base"
                self._mojo.root_element.mjcf.equality.add(
                    "weld", 
                    body1="base_target", 
                    body2="rby1/base",
                    solimp=[0.95, 0.99, 0.001], 
                    solref=[0.02, 1]
                )

    def _fix_limb_actuators(self):
        """Fix limb actuators for RBY1 with namespace."""
        if not hasattr(self, '_mojo') or not self._mojo:
            return
        
        # RBY1 actuators have "rby1/" prefix in environment context
        # but config doesn't have the prefix, so we need to manually populate limb_actuators
        if not hasattr(self, '_limb_actuators'):
            self._limb_actuators = []
        
        # If limb_actuators is already populated (by parent), don't override
        if self._limb_actuators:
            return
        
        # Get all actuators from root element MJCF  
        all_actuators = []
        if hasattr(self._mojo, 'root_element'):
            root = self._mojo.root_element
            # Use mjcf attribute if available
            if hasattr(root, 'mjcf'):
                root = root.mjcf
            # Now find all actuators
            if hasattr(root, 'find_all'):
                all_actuators = root.find_all("actuator")
        
        # Expected actuator names (WITHOUT namespace - actuators don't have rby1/ prefix)
        expected_names = [
            "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
            "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", 
            "right_arm_4", "right_arm_5", "right_arm_6",
            "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
            "left_arm_4", "left_arm_5", "left_arm_6"
        ]
        
        for actuator in all_actuators:
            if hasattr(actuator, 'name') and actuator.name in expected_names:
                self._limb_actuators.append(actuator)
        
        # Sort by expected order
        self._limb_actuators.sort(key=lambda a: expected_names.index(a.name) if a.name in expected_names else 999)
    
    def _on_loaded(self, model):
        """Override to apply scaling before model compilation."""
        # Apply scaling to the MJCF model before it's compiled
        if hasattr(self, '_model_scale') and self._model_scale != 1.0:
            # Scale all body positions
            for body in model.find_all('body'):
                if body.pos is not None:
                    body.pos = [p * self._model_scale for p in body.pos]
            
            # Scale all geom sizes (collision geometry)
            for geom in model.find_all('geom'):
                if geom.size is not None:
                    geom.size = [s * self._model_scale for s in geom.size]
                if hasattr(geom, 'fromto') and geom.fromto is not None:
                    geom.fromto = [f * self._model_scale for f in geom.fromto]
                # Also scale mesh if this geom uses one
                if hasattr(geom, 'mesh') and geom.mesh is not None:
                    # Geoms with meshes need their mesh scaled
                    mesh_name = geom.mesh
                    # Find the corresponding mesh asset and scale it
                    for mesh in model.find_all('mesh'):
                        if hasattr(mesh, 'name') and mesh.name == mesh_name:
                            if mesh.scale is not None:
                                mesh.scale = [s * self._model_scale for s in mesh.scale]
                            else:
                                mesh.scale = [self._model_scale] * 3
            
            # Scale all mesh assets directly 
            for mesh in model.find_all('mesh'):
                if mesh.scale is not None:
                    mesh.scale = [s * self._model_scale for s in mesh.scale]
                else:
                    mesh.scale = [self._model_scale] * 3
            
            # Scale all site positions  
            for site in model.find_all('site'):
                if site.pos is not None:
                    site.pos = [p * self._model_scale for p in site.pos]
                if site.size is not None:
                    site.size = [s * self._model_scale for s in site.size]
            
            # Scale joint ranges (for position limits)
            for joint in model.find_all('joint'):
                if joint.range is not None:
                    # Only scale positional joints, not angular ones
                    if hasattr(joint, 'type') and joint.type in ['slide', 'free']:
                        joint.range = [r * self._model_scale for r in joint.range]
                if joint.pos is not None:
                    joint.pos = [p * self._model_scale for p in joint.pos]
            
            # Scale inertial properties (optional - MuJoCo often handles this)
            for body in model.find_all('body'):
                if hasattr(body, 'inertial') and body.inertial is not None:
                    inertial = body.inertial
                    if inertial.pos is not None:
                        inertial.pos = [p * self._model_scale for p in inertial.pos]
                    # Mass scales with volume (scale^3)
                    if inertial.mass is not None:
                        inertial.mass = inertial.mass * (self._model_scale ** 3)
                    # Inertia scales with mass * length^2, so scale^5 total
                    if inertial.diaginertia is not None:
                        inertial.diaginertia = [i * (self._model_scale ** 5) for i in inertial.diaginertia]
        
        # Call parent's _on_loaded to continue normal initialization
        super()._on_loaded(model)
    
    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return RBY1_FINE_MANIPULATION_CONFIG