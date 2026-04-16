"""RBY1 Robot Configuration."""
import logging
import mujoco
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

# Default neck pose used when the head is not actively controlled
RBY1_HEAD_DEFAULT_JOINT_POSITIONS = {
    "rby1/head_0": 0.0,    # keep head centered in yaw
    # "rby1/head_1": 1.0,  
    "rby1/head_1": 0.785398,  
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
    cameras=["head", "left_wrist", "right_wrist"],
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
    cameras=["head", "left_wrist", "right_wrist"],
    namespaces_to_remove=[],
)


# Small perturbations applied after reset (meters / radians)
# Ranges are specified as (min, max).
_BASE_PERTURB_X_RANGE = (-0.1, 0.1)  # base X jitter
_BASE_PERTURB_Y_RANGE = (-0.1, 0.1)  # base Y jitter
_BASE_PERTURB_YAW_RANGE = (-np.deg2rad(20.0), np.deg2rad(20.0))  # base yaw jitter
_EE_PERTURB_POS_RANGE = (-0.1, 0.1)  # end-effector translational jitter
_EE_PERTURB_ROT_RANGE = (-np.deg2rad(20.0), np.deg2rad(20.0))  # end-effector rotational jitter


def _parse_range(value, name: str) -> tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a (min, max) tuple, got {value}.")
        low, high = float(value[0]), float(value[1])
    else:
        # Fallback for legacy scalar values: treat as symmetric range.
        bound = float(value)
        low, high = -abs(bound), abs(bound)
    if low > high:
        raise ValueError(f"{name} must satisfy min <= max (got {value}).")
    return low, high


def _rand_unit_vec(rng: np.random.RandomState):
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def _small_random_quat(
    angle_range: tuple[float, float] | float, rng: np.random.RandomState
) -> np.ndarray:
    """Sample a small random quaternion with angle bounded by angle_range."""
    axis = _rand_unit_vec(rng)
    min_angle, max_angle = _parse_range(angle_range, "rot_range")
    angle = rng.uniform(min_angle, max_angle)
    half = angle / 2.0
    sin_half = np.sin(half)
    return np.array([np.cos(half), *(axis * sin_half)], dtype=np.float64)


def _make_rby1_perturb_rng():
    rng_state = np.random.get_state()
    rng = np.random.RandomState()
    rng.set_state(rng_state)
    return rng, rng_state


def _restore_rby1_rng_state(rng_state):
    # Restore RNG state so task randomization stays consistent.
    np.random.set_state(rng_state)


def _perturb_rby1_base(
    mojo,
    x_range: tuple[float, float] | float,
    y_range: tuple[float, float] | float,
    yaw_range: tuple[float, float] | float,
    rng: np.random.RandomState,
):
    """Apply a small XY/yaw perturbation to the mobile base."""

    if mojo is None or not getattr(mojo, "physics", None):
        return

    physics = mojo.physics
    model = physics.model._model
    data = physics.data._data

    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rby1/")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rby1/base")
    if base_body_id < 0:
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if root_body_id < 0 and base_body_id < 0:
        logging.debug("Skipping RBY1 base perturbation (base body not found)")
        return

    free_qpos_adr = None
    for body_id in (root_body_id, base_body_id):
        if body_id < 0:
            continue
        jnt_adr = model.body_jntadr[body_id]
        jnt_num = model.body_jntnum[body_id]
        for j in range(jnt_adr, jnt_adr + jnt_num):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                free_qpos_adr = model.jnt_qposadr[j]
                break
        if free_qpos_adr is not None:
            break

    if free_qpos_adr is not None:
        base_pos = np.array(data.qpos[free_qpos_adr : free_qpos_adr + 3], dtype=np.float64)
        base_quat = np.array(
            data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7], dtype=np.float64
        )
    else:
        base_pos = np.array(data.xpos[base_body_id], dtype=np.float64)
        base_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(base_quat, data.xmat[base_body_id])

    x_min, x_max = _parse_range(x_range, "base_perturb_x_range")
    y_min, y_max = _parse_range(y_range, "base_perturb_y_range")
    yaw_min, yaw_max = _parse_range(yaw_range, "base_perturb_yaw_range")

    delta_xy = np.array(
        [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float64
    )
    delta_yaw = rng.uniform(yaw_min, yaw_max)

    new_pos = base_pos.copy()
    new_pos[0] += delta_xy[0]
    new_pos[1] += delta_xy[1]

    delta_quat = np.array(
        [np.cos(delta_yaw / 2.0), 0.0, 0.0, np.sin(delta_yaw / 2.0)],
        dtype=np.float64,
    )
    new_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(new_quat, delta_quat, base_quat)

    if free_qpos_adr is not None:
        data.qpos[free_qpos_adr : free_qpos_adr + 3] = new_pos
        data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = new_quat

    base_target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_target"
    )
    if base_target_body_id >= 0:
        mocap_id = model.body_mocapid[base_target_body_id]
        if mocap_id >= 0:
            data.mocap_pos[mocap_id][:] = new_pos
            data.mocap_quat[mocap_id][:] = new_quat

    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def _perturb_rby1_end_effectors(
    mojo,
    pos_range: tuple[float, float] | float,
    rot_range: tuple[float, float] | float,
    rng: np.random.RandomState | None = None,
    restore_state: bool = True,
):
    """Apply a small SE(3) perturbation to both end effectors via IK.

    Keeps the robot in a valid configuration while slightly moving wrists.
    """

    if mojo is None or not getattr(mojo, "physics", None):
        return

    try:
        from bigym.ik.rby1_whole_body_ik import RBY1WholeBodyIK
    except Exception as exc:  # noqa: BLE001 - we want to swallow any import issues
        logging.debug("Skipping RBY1 EE perturbation (IK unavailable): %s", exc)
        return

    physics = mojo.physics
    model = physics.model._model
    data = physics.data._data

    def _site_pose(site_name: str):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found")
        pos = np.array(data.site_xpos[site_id], dtype=np.float64)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        return pos, quat

    try:
        left_pos, left_quat = _site_pose("rby1/end_effector_l")
        right_pos, right_quat = _site_pose("rby1/end_effector_r")
    except ValueError as exc:  # noqa: BLE001
        logging.debug("Skipping RBY1 EE perturbation (site lookup failed): %s", exc)
        return

    if rng is None:
        rng, rng_state = _make_rby1_perturb_rng()
    else:
        rng_state = None

    try:
        def _perturb_pose(pos: np.ndarray, quat: np.ndarray):
            pos_min, pos_max = _parse_range(pos_range, "ee_perturb_pos_range")
            delta_pos = rng.uniform(pos_min, pos_max, size=3)
            delta_quat = _small_random_quat(rot_range, rng)
            new_pos = pos + delta_pos
            new_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(new_quat, delta_quat, quat)
            return new_pos, new_quat

        left_target_pos, left_target_quat = _perturb_pose(left_pos, left_quat)
        right_target_pos, right_target_quat = _perturb_pose(right_pos, right_quat)

        ik_solver = RBY1WholeBodyIK(model, data)
        try:
            solution_qpos, success, _ = ik_solver.solve(
                left_target_pos=left_target_pos,
                left_target_quat=left_target_quat,
                right_target_pos=right_target_pos,
                right_target_quat=right_target_quat,
                current_qpos=data.qpos.copy(),
            )
        except Exception as exc:  # noqa: BLE001
            logging.debug("RBY1 EE perturbation solve failed: %s", exc)
            return

        if not success or solution_qpos is None:
            logging.debug("RBY1 EE perturbation IK did not converge; leaving init pose unchanged")
            return

        data.qpos[:] = solution_qpos
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
    finally:
        if restore_state and rng_state is not None:
            _restore_rby1_rng_state(rng_state)

    # Keep base_target mocap aligned with the updated base pose.
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rby1/base")
    base_target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_target")
    if base_body_id >= 0 and base_target_body_id >= 0:
        mocap_id = model.body_mocapid[base_target_body_id]
        if mocap_id >= 0:
            data.mocap_pos[mocap_id][:] = data.xpos[base_body_id]
            base_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(base_quat, data.xmat[base_body_id])
            data.mocap_quat[mocap_id][:] = base_quat
            mujoco.mj_forward(model, data)


def _apply_head_default_posture(mojo):
    """Set head joints to default fixed angles and align actuator targets."""
    if not mojo or not getattr(mojo, "physics", None):
        return

    model = mojo.physics.model._model
    data = mojo.physics.data._data

    def _resolve_id(obj_type, base_name):
        """Find object id by trying bare and namespaced joint names."""
        for candidate in (base_name, f"rby1/{base_name}"):
            try:
                return mujoco.mj_name2id(model, obj_type, candidate)
            except ValueError:
                continue
        return -1

    needs_forward = False
    for joint_name, target in RBY1_HEAD_DEFAULT_JOINT_POSITIONS.items():
        joint_id = _resolve_id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue

        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = target
        dof_adr = model.jnt_dofadr[joint_id]
        data.qvel[dof_adr] = 0.0
        data.qacc[dof_adr] = 0.0

        actuator_id = _resolve_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_act"
        )
        if actuator_id >= 0:
            data.ctrl[actuator_id] = target

        needs_forward = True

    if needs_forward:
        mujoco.mj_forward(model, data)


class RBY1(Robot):
    """RBY1 Robot with Robotiq grippers."""

    def __init__(
        self,
        action_mode,
        mojo=None,
        base_perturb_x_range: tuple[float, float] | float | None = None,
        base_perturb_y_range: tuple[float, float] | float | None = None,
        base_perturb_yaw_range: tuple[float, float] | float | None = None,
        ee_perturb_pos_range: tuple[float, float] | float | None = None,
        ee_perturb_rot_range: tuple[float, float] | float | None = None,
        init_perturb: bool = False,
    ):
        """Initialize RBY1 robot with mocap base control."""
        self._base_perturb_x_range = _parse_range(
            _BASE_PERTURB_X_RANGE if base_perturb_x_range is None else base_perturb_x_range,
            "base_perturb_x_range",
        )
        self._base_perturb_y_range = _parse_range(
            _BASE_PERTURB_Y_RANGE if base_perturb_y_range is None else base_perturb_y_range,
            "base_perturb_y_range",
        )
        self._base_perturb_yaw_range = _parse_range(
            _BASE_PERTURB_YAW_RANGE if base_perturb_yaw_range is None else base_perturb_yaw_range,
            "base_perturb_yaw_range",
        )
        self._ee_perturb_pos_range = _parse_range(
            _EE_PERTURB_POS_RANGE if ee_perturb_pos_range is None else ee_perturb_pos_range,
            "ee_perturb_pos_range",
        )
        self._ee_perturb_rot_range = _parse_range(
            _EE_PERTURB_ROT_RANGE if ee_perturb_rot_range is None else ee_perturb_rot_range,
            "ee_perturb_rot_range",
        )

        super().__init__(action_mode, mojo, init_perturb=init_perturb)
        
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
        
        # Find base_target mocap body if not done yet
        model = self._mojo.physics.model._model
        self._base_target_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base_target"
        )
        if self._base_target_body_id < 0:
            # Mocap body doesn't exist, we need to handle this
            print("WARNING: base_target mocap body not found in model")
            self._base_target_body_id = -1

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

    def _set_pose(self, position: np.ndarray, orientation: np.ndarray):
        data = self._mojo.physics.data._data
        model = self._mojo.physics.model._model
        mocap_id = model.body_mocapid[self._base_target_body_id]
        data.qpos[0] = position[0]
        data.qpos[1] = position[1]
        data.qpos[3:7] = orientation
        data.mocap_pos[mocap_id][0] = position[0]
        data.mocap_pos[mocap_id][1] = position[1]
        data.mocap_quat[mocap_id] = orientation

    def reset(self, position: np.ndarray, orientation: np.ndarray):
        super().reset(position, orientation)
        _apply_head_default_posture(self._mojo)
        if self._init_perturb:
            rng, rng_state = _make_rby1_perturb_rng()
            try:
                _perturb_rby1_base(
                    self._mojo,
                    x_range=self._base_perturb_x_range,
                    y_range=self._base_perturb_y_range,
                    yaw_range=self._base_perturb_yaw_range,
                    rng=rng,
                )
                _perturb_rby1_end_effectors(
                    self._mojo,
                    pos_range=self._ee_perturb_pos_range,
                    rot_range=self._ee_perturb_rot_range,
                    rng=rng,
                    restore_state=False,
                )
            finally:
                _restore_rby1_rng_state(rng_state)

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return RBY1_CONFIG


class RBY1FineManipulation(Robot):
    """RBY1 Robot with Robotiq gripper for fine manipulations."""

    def __init__(
        self,
        action_mode,
        mojo=None,
        base_perturb_x_range: tuple[float, float] | float | None = None,
        base_perturb_y_range: tuple[float, float] | float | None = None,
        base_perturb_yaw_range: tuple[float, float] | float | None = None,
        ee_perturb_pos_range: tuple[float, float] | float | None = None,
        ee_perturb_rot_range: tuple[float, float] | float | None = None,
        init_perturb: bool = False,
    ):
        """Initialize RBY1 robot with mocap base control."""
        self._base_perturb_x_range = _parse_range(
            _BASE_PERTURB_X_RANGE if base_perturb_x_range is None else base_perturb_x_range,
            "base_perturb_x_range",
        )
        self._base_perturb_y_range = _parse_range(
            _BASE_PERTURB_Y_RANGE if base_perturb_y_range is None else base_perturb_y_range,
            "base_perturb_y_range",
        )
        self._base_perturb_yaw_range = _parse_range(
            _BASE_PERTURB_YAW_RANGE if base_perturb_yaw_range is None else base_perturb_yaw_range,
            "base_perturb_yaw_range",
        )
        self._ee_perturb_pos_range = _parse_range(
            _EE_PERTURB_POS_RANGE if ee_perturb_pos_range is None else ee_perturb_pos_range,
            "ee_perturb_pos_range",
        )
        self._ee_perturb_rot_range = _parse_range(
            _EE_PERTURB_ROT_RANGE if ee_perturb_rot_range is None else ee_perturb_rot_range,
            "ee_perturb_rot_range",
        )

        # Set desired scale before loading
        self._model_scale = 1.3
        super().__init__(action_mode, mojo, init_perturb=init_perturb)
        
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

    def reset(self, position: np.ndarray, orientation: np.ndarray):
        super().reset(position, orientation)
        _apply_head_default_posture(self._mojo)
        if self._init_perturb:
            rng, rng_state = _make_rby1_perturb_rng()
            try:
                _perturb_rby1_base(
                    self._mojo,
                    x_range=self._base_perturb_x_range,
                    y_range=self._base_perturb_y_range,
                    yaw_range=self._base_perturb_yaw_range,
                    rng=rng,
                )
                _perturb_rby1_end_effectors(
                    self._mojo,
                    pos_range=self._ee_perturb_pos_range,
                    rot_range=self._ee_perturb_rot_range,
                    rng=rng,
                    restore_state=False,
                )
            finally:
                _restore_rby1_rng_state(rng_state)
    
    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return RBY1_FINE_MANIPULATION_CONFIG
