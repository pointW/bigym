"""Set of plate moving tasks."""
from abc import ABC
import os

import numpy as np
from gymnasium import spaces
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.const import PRESETS_PATH
from bigym.envs.props.holders import DishDrainer
from bigym.envs.props.kitchenware import Plate
from bigym.envs.props.tables import Table
from bigym.utils.physics_utils import distance
from bigym.utils.observation_config import ObservationConfig


# Legacy reset distribution (used before commit e624aa8).
LEGACY_RACK_BOUNDS = np.array([0.05, 0.05, 0.0])
LEGACY_RACK_POSITION_LEFT = np.array([0.7, 0.3, 0.95])
LEGACY_RACK_POSITION_RIGHT = np.array([0.7, -0.3, 0.95])

# Enhanced reset distribution (default).
# Values chosen to increase variation while staying within a reasonable range for asset sizes.
RACK_BOUNDS = np.array([0.12, 0.12, 0.0])  # XY jitter for rack placement (meters)
RACK_YAW_BOUNDS = np.deg2rad(25)  # Rack yaw jitter (radians)
TABLE_Z_BOUNDS = 0.1  # Table height jitter (meters)

PLATE_OFFSET_POS = np.array([0, 0.01, 0])
PLATE_OFFSET_ROT = Quaternion(axis=[1, 0, 0], degrees=-5).elements


def _bigym_perturb_enabled() -> bool:
    """Return True if Bigym task reset perturbation is enabled."""
    value = os.getenv("BIGYM_DISABLE_PERTURB", "0").strip().lower()
    return value not in {"1", "true", "yes", "on"}


class _MovePlatesEnv(BiGymEnv, ABC):
    """Base plates environment."""

    _PRESET_PATH = PRESETS_PATH / "move_plates.yaml"

    _SUCCESSFUL_DIST = 0.05
    _SUCCESS_ROT = np.deg2rad(20)

    _PLATES_COUNT = 1
    _PCD_SURFACE_KEEP_EPS = 3.0e-3

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]
        self.rack_start = self._preset.get_props(DishDrainer)[0]
        self.rack_target = self._preset.get_props(DishDrainer)[1]
        self.plates = [Plate(self._mojo) for _ in range(self._PLATES_COUNT)]
        # Cache base poses from preset for reproducible randomized resets.
        self._table_base_pos = self.table.body.get_position().copy()
        self._rack_start_base_pos = self.rack_start.body.get_position().copy()
        self._rack_target_base_pos = self.rack_target.body.get_position().copy()
        self._rack_start_base_quat = self.rack_start.body.get_quaternion().copy()
        self._rack_target_base_quat = self.rack_target.body.get_quaternion().copy()

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])
        for plate in self.plates:
            if np.all(
                [
                    distance(plate.body, site) > self._SUCCESSFUL_DIST
                    for site in self.rack_target.sites
                ]
            ):
                return False
            plate_up = Quaternion(plate.body.get_quaternion()).rotate(up)
            angle = np.arccos(np.clip(np.dot(plate_up, right), -1.0, 1.0))
            if angle > self._SUCCESS_ROT:
                return False
            if not plate.is_colliding(self.rack_target):
                return False
            if plate.is_colliding(self.table):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(plate, side):
                    return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for plate in self.plates:
            if plate.is_colliding(self.floor):
                return True
        return False

    def _get_reset_pcd_min_world_z(self):
        """Keep table-surface and above using current table collider geometry."""
        max_z = self._max_world_z_from_colliders(self.table.colliders)
        if max_z is not None:
            return float(max_z) - self._PCD_SURFACE_KEEP_EPS
        return float(self.table.body.get_position()[2]) - self._PCD_SURFACE_KEEP_EPS

    def _on_reset(self):
        if not _bigym_perturb_enabled():
            offset = np.random.uniform(-LEGACY_RACK_BOUNDS, LEGACY_RACK_BOUNDS)
            self.rack_start.body.set_position(LEGACY_RACK_POSITION_LEFT + offset)
            offset = np.random.uniform(-LEGACY_RACK_BOUNDS, LEGACY_RACK_BOUNDS)
            self.rack_target.body.set_position(LEGACY_RACK_POSITION_RIGHT + offset)
        else:
            table_z_offset = np.random.uniform(-TABLE_Z_BOUNDS, TABLE_Z_BOUNDS)
            table_pos = self._table_base_pos.copy()
            table_pos[2] += table_z_offset
            self.table.body.set_position(table_pos, True)

            def _sample_rack_pose(base_pos, base_quat):
                offset = np.random.uniform(-RACK_BOUNDS, RACK_BOUNDS)
                pos = base_pos + offset
                pos[2] += table_z_offset
                yaw = np.random.uniform(-RACK_YAW_BOUNDS, RACK_YAW_BOUNDS)
                quat = Quaternion(axis=[0, 0, 1], angle=yaw) * Quaternion(base_quat)
                return pos, quat

            pos, quat = _sample_rack_pose(self._rack_start_base_pos, self._rack_start_base_quat)
            self.rack_start.body.set_position(pos, True)
            self.rack_start.body.set_quaternion(quat.elements, True)

            pos, quat = _sample_rack_pose(self._rack_target_base_pos, self._rack_target_base_quat)
            self.rack_target.body.set_position(pos, True)
            self.rack_target.body.set_quaternion(quat.elements, True)

        sites = np.array(self.rack_start.sites)
        sites = np.random.choice(sites, size=len(self.plates), replace=False)

        for site, plate in zip(sites, self.plates):
            plate.body.set_position(site.get_position() + PLATE_OFFSET_POS, True)
            quat = Quaternion(site.get_quaternion())
            quat *= PLATE_OFFSET_ROT
            plate.body.set_quaternion(quat.elements, True)


class MovePlate(_MovePlatesEnv):
    """Move one plate from one rack to another."""

    def __init__(
        self,
        action_mode,
        observation_config: ObservationConfig = ObservationConfig(),
        render_mode=None,
        start_seed=None,
        control_frequency: int = CONTROL_FREQUENCY_MAX,
        robot_cls=None,
        robot_kwargs=None,
    ):
        resolved_robot_cls = robot_cls or self.DEFAULT_ROBOT
        if robot_kwargs is None and getattr(resolved_robot_cls, "__name__", None) in {
            "RBY1",
            "RBY1FineManipulation",
        }:
            # Freeze MovePlate defaults to current RBY1 init perturb ranges.
            robot_kwargs = {
                "base_perturb_x_range": (-0.1, 0.1),
                "base_perturb_y_range": (-0.1, 0.1),
                "base_perturb_yaw_range": (
                    -np.deg2rad(20.0),
                    np.deg2rad(20.0),
                ),
                "ee_perturb_pos_range": (-0.1, 0.1),
                "ee_perturb_rot_range": (
                    -np.deg2rad(20.0),
                    np.deg2rad(20.0),
                ),
            }

        super().__init__(
            action_mode=action_mode,
            observation_config=observation_config,
            render_mode=render_mode,
            start_seed=start_seed,
            control_frequency=control_frequency,
            robot_cls=robot_cls,
            robot_kwargs=robot_kwargs,
        )

    def _get_task_privileged_obs_space(self):
        return {
            "rack_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),
            "plate_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        }

    def _get_task_privileged_obs(self):
        return {
            "rack_pose": np.array(self.rack_target.get_pose(), np.float32).flatten(),
            "plate_pose": np.array(self.plates[0].get_pose(), np.float32).flatten(),
        }


class MoveTwoPlates(_MovePlatesEnv):
    """Move two plates from one rack to another."""

    _PLATES_COUNT = 2

    def _get_task_privileged_obs_space(self):
        return {}

    def _get_task_privileged_obs(self):
        return {}
