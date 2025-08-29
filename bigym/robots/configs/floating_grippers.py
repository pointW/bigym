"""Floating Grippers Robot Configuration."""
from typing import Optional, Dict
import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Site, MujocoElement
from mujoco_utils import mjcf_utils

from bigym.const import ASSETS_PATH, HandSide
from bigym.robots.robot import Robot
from bigym.robots.config import RobotConfig, ArmConfig
from bigym.robots.configs.robotiq import ROBOTIQ_2F85
from bigym.robots.gripper import Gripper
from bigym.action_modes import ActionMode
from bigym.floating_gripper_action_mode import FloatingGripperActionMode


# Arm configurations for floating grippers
FLOATING_LEFT_ARM = ArmConfig(
    site="left_gripper_site",  # Site name from the XML
    links=[],  # No arm links - just floating grippers
    wrist_dof=None,  # No wrist joint
    offset_euler=np.array([np.pi / 2, 0, np.pi / 2]),  # Correct offset from RBY1
    offset_position=np.array([0, 0, 0]),  # No position offset
)

FLOATING_RIGHT_ARM = ArmConfig(
    site="right_gripper_site",  # Site name from the XML
    links=[],  # No arm links
    wrist_dof=None,
    offset_euler=np.array([np.pi / 2, 0, np.pi / 2]),  # Correct offset from RBY1
    offset_position=np.array([0, 0, 0]),
)

# Floating gripper configuration
FLOATING_GRIPPERS = RobotConfig(
    model=ASSETS_PATH / "floating_grippers" / "floating_grippers.xml",
    delta_range=(-0.1, 0.1),
    position_kp=300,
    pelvis_body="dummy_base",  # Use dummy base as pelvis
    full_body=None,  # No full body config
    floating_base=None,  # No floating base - grippers are controlled via mocap
    gripper=ROBOTIQ_2F85,  # Use proper Robotiq grippers
    arms={
        HandSide.LEFT: FLOATING_LEFT_ARM,
        HandSide.RIGHT: FLOATING_RIGHT_ARM,
    },
    actuators={
        "dummy_actuator1": False,  # Not used
        "dummy_actuator2": False,  # Not used
    },
    cameras=[],  # No cameras
)


class FloatingGrippers(Robot):
    """Robot with floating grippers controlled via mocap bodies.
    
    This robot uses a dummy structure to satisfy the Robot class requirements,
    while the actual grippers are controlled via mocap bodies for perfect tracking.
    """
    
    def __init__(
        self,
        action_mode: ActionMode,
        mojo: Optional[Mojo] = None,
    ):
        """Initialize floating grippers robot.
        
        Args:
            action_mode: Must be FloatingGripperActionMode
            mojo: Optional Mojo physics instance
        """
        # Verify action mode is correct type
        if not isinstance(action_mode, FloatingGripperActionMode):
            raise ValueError(
                f"FloatingGrippers requires FloatingGripperActionMode, "
                f"got {type(action_mode)}"
            )
        
        # Store action mode for later initialization
        self._pending_action_mode = action_mode
        self._mocap_bodies = {}  # Will store mocap body references
        
        # Initialize parent - this will load the model and call _on_loaded
        super().__init__(action_mode, mojo)
        
        # Remove the freejoint that was automatically added to the floating_grippers model
        # This prevents the dummy_base (pelvis) from falling due to gravity
        if self._body and hasattr(self._body.mjcf, 'freejoint'):
            self._body.mjcf.freejoint.remove()
            # Force physics recompilation
            self._mojo._physics = None
    
    @property
    def config(self) -> RobotConfig:
        """Get robot configuration."""
        return FLOATING_GRIPPERS
    
    def _get_grippers(self) -> Dict[HandSide, Gripper]:
        """Override to load grippers as root bodies with weld constraints to mocap.
        
        This keeps grippers as dynamic bodies constrained to mocap bodies,
        allowing proper physics interaction while maintaining near-perfect tracking.
        """
        
        # Create mocap bodies for gripper control
        worldbody = self._mojo.root_element.mjcf.worldbody
        
        # Use H1's default end-effector positions for initial mocap positions
        left_mocap_pos = [0.3185, 0.21353, 1.08661]
        right_mocap_pos = [0.3185, -0.21353, 1.08661]
        
        # Create mocap bodies (these control the position)
        left_mocap = worldbody.add(
            'body',
            name='left_gripper_mocap',
            mocap=True,
            pos=left_mocap_pos
        )
        # Visual indicator for debugging
        left_mocap.add(
            'geom',
            name='left_mocap_viz',
            type='sphere',
            size=[0.02],
            rgba=[1, 0, 0, 0.3],
            contype=0,
            conaffinity=0
        )
        
        right_mocap = worldbody.add(
            'body',
            name='right_gripper_mocap',
            mocap=True,
            pos=right_mocap_pos
        )
        # Visual indicator for debugging
        right_mocap.add(
            'geom',
            name='right_mocap_viz',
            type='sphere',
            size=[0.02],
            rgba=[0, 0, 1, 0.3],
            contype=0,
            conaffinity=0
        )
        
        # Store mocap references
        self._mocap_bodies[HandSide.LEFT] = left_mocap
        self._mocap_bodies[HandSide.RIGHT] = right_mocap
        
        # Force physics recompilation to include mocap bodies
        self._mojo._physics = None
        
        # Load gripper models directly at the root level (not attached to sites)
        from mojo.elements import Body
        from bigym.robots.gripper import Gripper
        
        # Load left gripper at root level
        left_gripper_body = self._mojo.load_model(
            str(self.config.gripper.model),
            None,  # Load at world level (None means root)
            on_loaded=lambda model: self._configure_gripper(model, HandSide.LEFT)
        )
        left_gripper_body.mjcf.pos = left_mocap_pos
        # Don't set euler - let weld constraint handle orientation
        
        # Load right gripper at root level
        right_gripper_body = self._mojo.load_model(
            str(self.config.gripper.model),
            None,  # Load at world level (None means root)
            on_loaded=lambda model: self._configure_gripper(model, HandSide.RIGHT)
        )
        right_gripper_body.mjcf.pos = right_mocap_pos
        # Don't set euler - let weld constraint handle orientation
        
        # Add free joints to the loaded gripper bodies so they can move
        # The loaded grippers created wrapper bodies named "robotiq_2f85_left/" and "robotiq_2f85_right/"
        left_gripper_body.mjcf.add('freejoint', name='left_gripper_free')
        right_gripper_body.mjcf.add('freejoint', name='right_gripper_free')
        
        # Force physics recompilation after loading grippers
        self._mojo._physics = None
        
        # Add weld constraints with rotation offset
        # The gripper base is at euler [π/2, π/2, 0] relative to the EE site orientation
        # We need to include this offset in the weld constraint
        from scipy.spatial.transform import Rotation
        
        # Convert offset euler to quaternion for relpose
        offset_euler = self.config.arms[HandSide.LEFT].offset_euler  # [π/2, π/2, 0]
        offset_rot = Rotation.from_euler('xyz', offset_euler)
        offset_quat = offset_rot.as_quat()  # [x, y, z, w]
        
        # Create relpose string: "x y z qw qx qy qz"
        # The offset means: gripper_base = mocap * offset
        left_relpose = f"0 0 0 {offset_quat[3]} {offset_quat[0]} {offset_quat[1]} {offset_quat[2]}"
        right_relpose = f"0 0 0 {offset_quat[3]} {offset_quat[0]} {offset_quat[1]} {offset_quat[2]}"
        
        self._mojo.root_element.mjcf.equality.add(
            'weld',
            name='left_gripper_weld',
            body1='left_gripper_mocap',
            body2='robotiq_2f85_left/base_mount',  # Attachment point
            relpose=left_relpose,  # Include rotation offset
            solimp=[0.9999, 0.9999, 0.0001],  # CFM, damping, impedance (all very stiff)
            solref=[0.0001, 1]  # Time constant and damping ratio
        )
        
        self._mojo.root_element.mjcf.equality.add(
            'weld',
            name='right_gripper_weld',
            body1='right_gripper_mocap',
            body2='robotiq_2f85_right/base_mount',  # Attachment point
            relpose=right_relpose,  # Include rotation offset
            solimp=[0.9999, 0.9999, 0.0001],  # CFM, damping, impedance (all very stiff)
            solref=[0.0001, 1]  # Time constant and damping ratio
        )
        
        # Force final physics recompilation
        self._mojo._physics = None
        
        # Initialize mocap quaternions to identity (matching H1/RBY1 EE site orientation)
        # The demos control the EE site, which starts near identity, NOT the gripper base
        physics = self._mojo.physics
        if physics:
            # Find mocap indices
            model = physics.model
            left_mocap_idx = None
            right_mocap_idx = None
            for body_id in range(model.nbody):
                body_name = model.id2name(body_id, 'body')
                if body_name and 'gripper_mocap' in body_name:
                    if model.body_mocapid[body_id] >= 0:
                        mocap_idx = model.body_mocapid[body_id]
                        if 'left' in body_name:
                            left_mocap_idx = mocap_idx
                        elif 'right' in body_name:
                            right_mocap_idx = mocap_idx
            
            # Set mocap quaternions to identity [w, x, y, z] = [1, 0, 0, 0]
            # This matches the EE site orientation in H1/RBY1
            if left_mocap_idx is not None:
                physics.data.mocap_quat[left_mocap_idx] = [1.0, 0.0, 0.0, 0.0]
            if right_mocap_idx is not None:
                physics.data.mocap_quat[right_mocap_idx] = [1.0, 0.0, 0.0, 0.0]
        
        # Create gripper objects with the loaded bodies
        # We need to find the wrist sites on the loaded grippers
        from mojo.elements import Site, MujocoElement
        mojo_model = MujocoElement(self._mojo, self._mojo.root_element.mjcf)
        
        # Create Gripper objects manually since we loaded them differently
        grippers: Dict[HandSide, Gripper] = {}
        
        # For left gripper
        left_gripper = Gripper.__new__(Gripper)
        left_gripper._side = HandSide.LEFT
        left_gripper._mojo = self._mojo
        left_gripper._config = self.config.gripper
        left_gripper._body = left_gripper_body
        # Find and set actuators and other gripper components
        self._process_gripper_internals(left_gripper, 'robotiq_2f85_left')
        grippers[HandSide.LEFT] = left_gripper
        
        # For right gripper
        right_gripper = Gripper.__new__(Gripper)
        right_gripper._side = HandSide.RIGHT
        right_gripper._mojo = self._mojo
        right_gripper._config = self.config.gripper
        right_gripper._body = right_gripper_body
        # Find and set actuators and other gripper components
        self._process_gripper_internals(right_gripper, 'robotiq_2f85_right')
        grippers[HandSide.RIGHT] = right_gripper
        
        # Store gripper references for get_hand_pos
        self._grippers = grippers
        
        # Initialize action mode with mocap bodies ready (after physics is compiled)
        if hasattr(self, '_pending_action_mode'):
            self._pending_action_mode.initialize(self, self._mojo)
        
        return grippers
    
    def _configure_gripper(self, model: mjcf.RootElement, side: HandSide):
        """Configure gripper model after loading."""
        model.model += f"_{side.value.lower()}"
        # Don't add free joint here - we'll add it after loading when we have the full structure
    
    def _process_gripper_internals(self, gripper: Gripper, namespace: str):
        """Process gripper internals like actuators and sites."""
        from dm_control import mjcf
        
        # Find actuators for this gripper
        gripper._actuators = []
        all_actuators = mjcf_utils.safe_find_all(self._mojo.root_element.mjcf, "actuator")
        for actuator in all_actuators:
            if namespace in str(actuator.full_identifier):
                gripper._actuators.append(actuator)
        
        # Find actuated joints
        gripper._actuated_joints = []
        for actuator in gripper._actuators:
            if actuator.joint:
                gripper._actuated_joints.append(actuator.joint)
        
        # Find pinch site for grasping detection
        try:
            gripper._pinch_site = Site.get(
                self._mojo, 
                f"{namespace}/pinch",
                MujocoElement(self._mojo, self._mojo.root_element.mjcf)
            )
        except:
            gripper._pinch_site = None
        
        # For wrist site, we need to use the base_mount position
        # This matches where H1 would measure its end-effector position
        # The base_mount is the attachment point, equivalent to H1's end-effector site
        try:
            # Create a site reference at the base_mount body
            base_mount_body = self._mojo.root_element.mjcf.find('body', f'{namespace}/base_mount')
            if base_mount_body:
                # Add a site at the base_mount origin (this is where H1 attaches)
                ee_site = base_mount_body.add('site', name=f'{namespace}_ee', pos=[0, 0, 0])
                self._mojo._physics = None  # Force recompilation
                # Now get the site reference - it gets namespaced as namespace/namespace_ee
                gripper._wrist_site = Site.get(
                    self._mojo,
                    f'{namespace}/{namespace}_ee',
                    MujocoElement(self._mojo, self._mojo.root_element.mjcf)
                )
            else:
                # Fallback to pinch site if we can't create ee site
                gripper._wrist_site = gripper._pinch_site
        except Exception as e:
            print(f"Warning: Could not create end-effector site for {namespace}: {e}")
            gripper._wrist_site = gripper._pinch_site
        
        # Get pad geoms for collision detection
        gripper._pad_geoms = []  # Will be populated when gripper is used
    
    
    def get_hand_pos(self, side: HandSide) -> np.ndarray:
        """Get position of gripper.
        
        Returns the actual gripper position (from the gripper's wrist site),
        not the mocap position. This allows us to measure tracking error.
        
        Args:
            side: Which gripper (LEFT or RIGHT)
            
        Returns:
            3D position of the gripper's actual end-effector
        """
        # Return the actual gripper position from its wrist site
        # The grippers have end-effector sites that move with them
        if side in self._grippers:
            return self._grippers[side].wrist_position
        
        # Fallback if gripper not found
        return np.zeros(3)
    
    def reset(self, position: np.ndarray, orientation: np.ndarray):
        """Reset robot.
        
        For floating grippers, we reset mocap bodies to H1's default end-effector positions.
        The position and orientation arguments are ignored since we match H1's exact pose.
        """
        # Skip parent reset since we don't have a full_body config
        # Just reset the action mode if needed
        if self._action_mode:
            reset_state = self.config.floating_base.reset_state if self.config.floating_base else None
            if reset_state is not None:
                self._action_mode.reset(reset_state)
        
        # Reset mocap positions and orientations to H1 defaults
        physics = self._mojo.physics
        if physics:
            # Find mocap indices
            model = physics.model
            left_mocap_idx = None
            right_mocap_idx = None
            for body_id in range(model.nbody):
                body_name = model.id2name(body_id, 'body')
                if body_name and 'gripper_mocap' in body_name:
                    if model.body_mocapid[body_id] >= 0:
                        mocap_idx = model.body_mocapid[body_id]
                        if 'left' in body_name:
                            left_mocap_idx = mocap_idx
                        elif 'right' in body_name:
                            right_mocap_idx = mocap_idx
            
            # Reset positions
            if left_mocap_idx is not None:
                physics.data.mocap_pos[left_mocap_idx] = [0.3185, 0.21353, 1.08661]
            if right_mocap_idx is not None:
                physics.data.mocap_pos[right_mocap_idx] = [0.3185, -0.21353, 1.08661]
            
            # Reset to identity orientation (matching H1/RBY1 EE site)
            # Set mocap quaternions to identity [w, x, y, z] = [1, 0, 0, 0]
            if left_mocap_idx is not None:
                physics.data.mocap_quat[left_mocap_idx] = [1.0, 0.0, 0.0, 0.0]
            if right_mocap_idx is not None:
                physics.data.mocap_quat[right_mocap_idx] = [1.0, 0.0, 0.0, 0.0]