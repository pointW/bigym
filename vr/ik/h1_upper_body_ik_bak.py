"""H1 upper body IK solver."""
from dataclasses import dataclass, field

import mujoco
import numpy as np
from dm_control import mjcf
from lxml import etree
from mujoco_utils import mjcf_utils, physics_utils, collision_utils
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import (
    HandSide,
)
from bigym.robots.configs.h1 import H1_CONFIG

WORLDBODY = "worldbody"
FEATURES_TO_REMOVE = {"key", "actuator", "tendon", "contact", "equality"}
GRIPPER_NAME_PREFIX = "robotiq"
EE_ORIGIN_SITE_NAME = "ee_origin"
H1_PREFIX = "h1"
PELVIS_NAME = f"{H1_PREFIX}\\pelvis"
TORSO_NAME = f"{H1_PREFIX}\\torso_link"
KP = 1000
KV = 2 * np.sqrt(KP)
JOINT_DAMPING = KP / 200
RANGE_EE_POSITION = (-5, 5)
SOLVER_MAX_STEPS = 200
WRIST_ANGLE_SCALE = 2
TIMESTEP_FACTOR = 10

JOINT_LIMITS = {
    f"{H1_PREFIX}\\left_elbow": (-1.25, np.pi / 2),
    f"{H1_PREFIX}\\right_elbow": (-1.25, np.pi / 2),
}


@dataclass
class Pose:
    """Pose represented by np.ndarray and Quaternion."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: Quaternion = Quaternion()


# ToDO: add abstract IK class
class H1UpperBodyIK:
    """H1 upper body IK solver.

    Notes:
     - Position is controlled by refsite actuators. See documentation for more details:
       https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general-refsite
     - Wrist rotation is controlled by target quaternion directly.
    """

    def __init__(self, env: BiGymEnv, enable_full_6d_control: bool = False):
        """Init.
        
        Args:
            env: BiGym environment
            enable_full_6d_control: If True, add rotation actuators for full 6D pose control.
                                   If False, use original limited orientation control.
        """
        self.enable_full_6d_control = enable_full_6d_control
        self._calibration_offset_left = None
        self._calibration_offset_right = None
        base_model = env.mojo.root_element.mjcf
        base_xml = base_model.to_xml()

        # Removing all except H1
        for elem in base_xml.find(WORLDBODY):
            if not elem.attrib["name"].startswith(H1_PREFIX):
                elem.getparent().remove(elem)

        self._model = mjcf.from_xml_string(
            xml_string=etree.tostring(base_xml),
            escape_separators=True,
            assets=base_model.get_assets(),
        )

        for feature in FEATURES_TO_REMOVE:
            try:
                elements = mjcf_utils.safe_find_all(self._model, feature)
                for element in elements:
                    element.remove()
            except ValueError:
                pass

        # Fix pelvis
        _all = mjcf_utils.safe_find_all(self._model, "body")
        self._pelvis = mjcf_utils.safe_find(self._model, "body", PELVIS_NAME)
        if self._pelvis.freejoint:
            self._pelvis.freejoint.remove()

        # Find limb roots
        torso = mjcf_utils.safe_find(self._model, "body", TORSO_NAME)
        left_shoulder = mjcf_utils.safe_find(
            self._model,
            "body",
            f"{H1_PREFIX}\\{H1_CONFIG.arms[HandSide.LEFT].links[0]}",
        )
        right_shoulder = mjcf_utils.safe_find(
            self._model,
            "body",
            f"{H1_PREFIX}\\{H1_CONFIG.arms[HandSide.RIGHT].links[0]}",
        )

        # Find sites
        self._left_arm_site = mjcf_utils.safe_find(
            self._model,
            "site",
            f"{H1_PREFIX}\\{H1_CONFIG.arms[HandSide.LEFT].site}",
        )
        self._right_arm_site = mjcf_utils.safe_find(
            self._model,
            "site",
            f"{H1_PREFIX}\\{H1_CONFIG.arms[HandSide.RIGHT].site}",
        )

        # Find arm joints
        all_joints = mjcf_utils.safe_find_all(self._pelvis, "joint")
        if self.enable_full_6d_control:
            # Include all arm joints including wrist joints for full 6D control
            arm_joints = {
                *mjcf_utils.safe_find_all(left_shoulder, "joint"),
                *mjcf_utils.safe_find_all(right_shoulder, "joint"),
            }
        else:
            # Original behavior: exclude wrist joints (last joint in each arm)
            arm_joints = {
                *mjcf_utils.safe_find_all(left_shoulder, "joint")[:-1],
                *mjcf_utils.safe_find_all(right_shoulder, "joint")[:-1],
            }

        for joint in all_joints:
            if joint not in arm_joints or GRIPPER_NAME_PREFIX in joint.name.lower():
                joint.remove()
            else:
                joint.damping = JOINT_DAMPING
                if joint.name in JOINT_LIMITS:
                    joint.range = JOINT_LIMITS[joint.name]

        self._arm_joints = mjcf_utils.safe_find_all(self._pelvis, "joint")

        origin_site = self._model.worldbody.add("site", name=EE_ORIGIN_SITE_NAME)

        self._actuators_left = self._generate_ee_actuators(
            self._left_arm_site.name,
            origin_site.name,
        )
        self._actuators_right = self._generate_ee_actuators(
            self._right_arm_site.name,
            origin_site.name,
        )
        
        # Add rotation actuators for full 6D control
        if self.enable_full_6d_control:
            self._rotation_actuators_left = self._generate_rotation_actuators(
                self._left_arm_site.name,
                origin_site.name,
            )
            self._rotation_actuators_right = self._generate_rotation_actuators(
                self._right_arm_site.name,
                origin_site.name,
            )

        # Disable collisions between arms and torso
        self._model.contact.add("exclude", body1=torso.name, body2=left_shoulder.name)
        self._model.contact.add("exclude", body1=torso.name, body2=right_shoulder.name)

        # Enable gravity compensation
        physics_utils.compensate_gravity(self._model)

        # Disable collisions
        for body in mjcf_utils.safe_find_all(self._model, "body"):
            collision_utils.disable_body_collisions(body)

        self._physics = mjcf.Physics.from_mjcf_model(self._model)
        self._physics.model.opt.timestep *= TIMESTEP_FACTOR

        for body in mjcf_utils.safe_find_all(self._model, "body"):
            body = self._physics.bind(body)
            body.inertia *= 0

    def calibrate_with_real_robot(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        real_left_pose: Pose,
        real_right_pose: Pose,
    ):
        """Calibrate IK solver by computing offset between internal model and real robot.
        
        Args:
            pelvis_pose: Current pelvis pose
            qpos_arm_left: Current left arm joint positions
            qpos_arm_right: Current right arm joint positions
            real_left_pose: Real robot's current left end-effector pose
            real_right_pose: Real robot's current right end-effector pose
        """
        # Set IK solver to current robot state
        arm_joints = self._physics.bind(self._arm_joints)
        if self.enable_full_6d_control:
            qpos = np.concatenate((qpos_arm_left, qpos_arm_right))
        else:
            qpos = np.concatenate((qpos_arm_left[:-1], qpos_arm_right[:-1]))
        
        arm_joints.qpos = qpos
        arm_joints.qvel = np.zeros_like(qpos)
        arm_joints.qacc = np.zeros_like(qpos)
        
        self._physics.bind(self._pelvis).pos = pelvis_pose.position
        self._physics.bind(self._pelvis).quat = pelvis_pose.orientation.elements
        
        # Step once to update poses
        self._physics.step(1)
        
        # Get IK solver's predicted poses
        ik_left_pos = self._physics.bind(self._left_arm_site).xpos
        ik_right_pos = self._physics.bind(self._right_arm_site).xpos
        
        # Compute calibration offsets
        self._calibration_offset_left = real_left_pose.position - ik_left_pos
        self._calibration_offset_right = real_right_pose.position - ik_right_pos
        
        print(f"IK solver calibrated:")
        print(f"  Left offset: {self._calibration_offset_left}")
        print(f"  Right offset: {self._calibration_offset_right}")

    def solve(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        target_pose_left: Pose,
        target_pose_right: Pose,
    ) -> np.ndarray:
        """Solve IK."""
        arm_joints = self._physics.bind(self._arm_joints)
        if self.enable_full_6d_control:
            # Include all joints including wrist joints
            qpos = np.concatenate((qpos_arm_left, qpos_arm_right))
        else:
            # Original behavior: exclude wrist joints
            qpos = np.concatenate((qpos_arm_left[:-1], qpos_arm_right[:-1]))
        arm_joints.qpos = qpos
        arm_joints.qvel = np.zeros_like(qpos)
        arm_joints.qacc = np.zeros_like(qpos)

        # Solve position
        self._physics.bind(self._pelvis).pos = pelvis_pose.position
        self._physics.bind(self._pelvis).quat = pelvis_pose.orientation.elements

        # Apply calibration offsets to compensate for model mismatch
        if self._calibration_offset_left is not None and self._calibration_offset_right is not None:
            # Subtract the calibration offset to correct for model mismatch
            corrected_left_target = target_pose_left.position - self._calibration_offset_left
            corrected_right_target = target_pose_right.position - self._calibration_offset_right
        else:
            # No calibration available, use targets as-is
            corrected_left_target = target_pose_left.position
            corrected_right_target = target_pose_right.position
            
        self._physics.bind(self._actuators_left).ctrl = corrected_left_target
        self._physics.bind(self._actuators_right).ctrl = corrected_right_target

        if self.enable_full_6d_control:
            # Full 6D control: use rotation actuators for orientation
            left_euler = self._quaternion_to_euler(target_pose_left.orientation)
            right_euler = self._quaternion_to_euler(target_pose_right.orientation)
            
            self._physics.bind(self._rotation_actuators_left).ctrl = left_euler
            self._physics.bind(self._rotation_actuators_right).ctrl = right_euler

        # Iterative solving with convergence checking
        for step in range(SOLVER_MAX_STEPS):
            self._physics.step(1)
            
            # Check convergence every 10 steps to avoid overhead
            if step % 10 == 9:
                # Check position convergence
                left_site = self._physics.bind(self._left_arm_site)
                right_site = self._physics.bind(self._right_arm_site)
                
                left_pos_error = np.linalg.norm(left_site.xpos - target_pose_left.position)
                right_pos_error = np.linalg.norm(right_site.xpos - target_pose_right.position)
                
                # Convergence criteria: position error < 1mm
                if left_pos_error < 0.001 and right_pos_error < 0.001:
                    break

        if self.enable_full_6d_control:
            # Return joint solution directly (including wrist joints)
            solution = np.array(self._physics.bind(self._arm_joints).qpos)
            return solution
        else:
            # Original limited orientation control using cosine distance
            # Cache orientation
            left_site_quat = self._get_site_quaternion(self._left_arm_site)
            right_site_quat = self._get_site_quaternion(self._right_arm_site)

            # Solve orientation
            y = np.array([0, 1, 0])
            z = np.array([0, 0, 1])
            left_up = target_pose_left.orientation.rotate(z)
            right_up = target_pose_right.orientation.rotate(z)
            left_site_up = left_site_quat.rotate(y)
            right_site_up = right_site_quat.rotate(y)

            left_wrist = np.arccos(np.dot(left_site_up, left_up)) - np.pi / 2
            left_wrist = np.clip(left_wrist * WRIST_ANGLE_SCALE, -np.pi / 2, np.pi / 2)
            right_wrist = np.arccos(np.dot(right_site_up, right_up)) - np.pi / 2
            right_wrist = np.clip(right_wrist * WRIST_ANGLE_SCALE, -np.pi / 2, np.pi / 2)

            solution = np.array(self._physics.bind(self._arm_joints).qpos)
            left_solution, right_solution = np.split(solution, 2)
            left_solution = np.append(left_solution, left_wrist)
            right_solution = np.append(right_solution, right_wrist)

        return np.concatenate((left_solution, right_solution))

    def _generate_ee_actuators(self, site: str, origin: str):
        x = self._model.actuator.add(
            "position",
            kp=KP,
            kv=KV,
            ctrlrange=RANGE_EE_POSITION,
            name=f"{site}_ee_x",
        )

        y = self._model.actuator.add(
            "position",
            kp=KP,
            kv=KV,
            ctrlrange=RANGE_EE_POSITION,
            name=f"{site}_ee_y",
        )
        z = self._model.actuator.add(
            "position",
            kp=KP,
            kv=KV,
            ctrlrange=RANGE_EE_POSITION,
            name=f"{site}_ee_z",
        )

        actuators = [x, y, z]
        for index, actuator in enumerate(actuators):
            actuator.gear = np.zeros(6)
            actuator.gear[index] = 1
            actuator.site = site
            actuator.refsite = origin
        return actuators

    def _generate_rotation_actuators(self, site: str, origin: str):
        """Generate rotation actuators for full 6D orientation control."""
        RANGE_EE_ROTATION = (-np.pi, np.pi)
        
        rx = self._model.actuator.add(
            "position",
            kp=KP,
            kv=KV,
            ctrlrange=RANGE_EE_ROTATION,
            name=f"{site}_ee_rx",
        )

        ry = self._model.actuator.add(
            "position",
            kp=KP,
            kv=KV,
            ctrlrange=RANGE_EE_ROTATION,
            name=f"{site}_ee_ry",
        )
        
        rz = self._model.actuator.add(
            "position",
            kp=KP,
            kv=KV,
            ctrlrange=RANGE_EE_ROTATION,
            name=f"{site}_ee_rz",
        )

        actuators = [rx, ry, rz]
        for index, actuator in enumerate(actuators):
            actuator.gear = np.zeros(6)
            actuator.gear[3 + index] = 1  # Rotation around x, y, z axes
            actuator.site = site
            actuator.refsite = origin
        return actuators

    def _get_site_quaternion(self, site: mjcf.Element) -> Quaternion:
        bound_site = self._physics.bind(site)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, bound_site.xmat)
        return Quaternion(quat)

    def _quaternion_to_euler(self, quaternion: Quaternion) -> np.ndarray:
        """Convert quaternion to Euler angles for rotation actuator control."""
        # Convert to rotation matrix first
        rot_matrix = quaternion.rotation_matrix
        
        # Extract Euler angles (XYZ convention)
        # This is a simple extraction - may need adjustment based on MuJoCo's convention
        sy = np.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
            y = np.arctan2(-rot_matrix[2, 0], sy)
            z = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
        else:
            x = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
            y = np.arctan2(-rot_matrix[2, 0], sy)
            z = 0
            
        return np.array([x, y, z])
