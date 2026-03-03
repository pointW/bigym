"""Manipulation tasks."""
from abc import ABC
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import ASSETS_PATH, PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet
from bigym.envs.props.cutlery import Spoon
from bigym.envs.props.items import Cube
from bigym.envs.props.kitchenware import Mug
from bigym.robots.configs.h1 import H1FineManipulation
from bigym.utils.env_utils import get_random_points_on_plane


def _bigym_perturb_enabled() -> bool:
    """Return True if task reset perturbation is enabled."""
    value = os.getenv("BIGYM_DISABLE_PERTURB", "0").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _rotate_point_xy(point: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a 3D point around `center` in XY plane by yaw radians."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rel_x, rel_y = point[0] - center[0], point[1] - center[1]
    rotated = point.copy()
    rotated[0] = center[0] + (cos_yaw * rel_x - sin_yaw * rel_y)
    rotated[1] = center[1] + (sin_yaw * rel_x + cos_yaw * rel_y)
    return rotated


def _max_obj_axis(obj_path: Path, axis: int) -> float:
    """Return max vertex coordinate along an OBJ axis."""
    max_value = -np.inf
    with obj_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            value = float(parts[axis + 1])
            if value > max_value:
                max_value = value
    if not np.isfinite(max_value):
        raise ValueError(f"Could not read vertices from OBJ: {obj_path}")
    return float(max_value)


@lru_cache(maxsize=1)
def _upside_down_mug_support_depth() -> float:
    """Return support depth from mug root to contact point when upside down."""
    mug_assets = ASSETS_PATH / "props/mug/assets"
    collision_meshes = sorted(mug_assets.glob("mug_collision_*.obj"))
    if not collision_meshes:
        collision_meshes = [mug_assets / "mug.obj"]
    return max(_max_obj_axis(mesh_path, axis=1) for mesh_path in collision_meshes)


class _ManipulationEnv(BiGymEnv, ABC):
    """Base manipulation environment."""

    _PRESET_PATH = PRESETS_PATH / "cabinet.yaml"

    def _initialize_env(self):
        self.cabinet = self._preset.get_props(BaseCabinet)[0]


class FlipCup(_ManipulationEnv):
    """Flip cup upside-up task."""

    _CUP_POS = np.array([0.8, 0, 1])
    _CUP_ROT_X = np.deg2rad(180)
    _CUP_ROT_Z = np.deg2rad(180)
    _CUP_STEP = 0.1
    _CUP_POS_EXTENTS = np.array([0.1, 0.25])
    _CUP_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUP_ROT_BOUNDS = np.deg2rad(30)
    # Enhanced reset distribution for perturb mode:
    # shift center toward the counter center and cover most of the tabletop
    # while keeping margin from edges to avoid frequent falls.
    _CUP_POS_EXTENTS_ENHANCED = np.array([0.12, 0.12])
    _CUP_POS_BOUNDS_ENHANCED = np.array([0.03, 0.03, 0.0])
    _CUP_ROT_BOUNDS_ENHANCED = np.pi
    _TABLE_Z_BOUNDS = 0.1
    _TABLE_YAW_BOUNDS = np.pi
    _CUP_TABLE_CONTACT_EPS = 1.0e-4
    _PCD_SURFACE_KEEP_EPS = 3.0e-3

    _TOLERANCE = np.deg2rad(5)

    def _initialize_env(self):
        super()._initialize_env()
        self.cup = Mug(self._mojo)
        # Cache base pose so resets are deterministic when perturb is disabled.
        self._cabinet_base_pos = self.cabinet.body.get_position().copy()
        self._cabinet_base_quat = self.cabinet.body.get_quaternion().copy()
        self._counter_base_pos = self.cabinet.counter.get_position().copy()
        base_rot = Quaternion(self._cabinet_base_quat).rotation_matrix
        # Counter pose expressed in cabinet-local frame at reset.
        self._counter_local_pos = base_rot.T @ (self._counter_base_pos - self._cabinet_base_pos)
        # Use the point right below counter center (z=0 in cabinet-local frame)
        # as a practical "cabinet lower center" pivot during perturbation yaw.
        self._cabinet_pivot_local = self._counter_local_pos.copy()
        self._cabinet_pivot_local[2] = 0.0
        self._cabinet_pivot_world_base = (
            self._cabinet_base_pos + base_rot @ self._cabinet_pivot_local
        )
        counter_size = np.array(self.cabinet.counter.mjcf.size, dtype=np.float64)
        if counter_size.size < 3:
            raise ValueError(
                "Expected counter geom to have 3D size for z surface offset."
            )
        counter_half_height = float(counter_size[2])
        self._cup_z_from_counter = (
            counter_half_height
            + _upside_down_mug_support_depth()
            - self._CUP_TABLE_CONTACT_EPS
        )

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        cup_up = Quaternion(self.cup.body.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(cup_up, up), -1.0, 1.0))
        if angle_to_up > self._TOLERANCE:
            return False
        if not self.cup.is_colliding(self.cabinet.counter):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.cup, side):
                return False
        return True

    def _get_reset_pcd_min_world_z(self):
        """Keep table-surface and above by using current counter top height."""
        counter_size = np.asarray(self.cabinet.counter.mjcf.size, dtype=np.float64)
        if counter_size.size < 3:
            return None
        top_z = float(self.cabinet.counter.get_position()[2] + counter_size[2])
        return top_z - self._PCD_SURFACE_KEEP_EPS

    def _on_reset(self):
        perturb_enabled = _bigym_perturb_enabled()
        table_yaw_offset = 0.0
        # Keep original reset distribution when perturb is disabled.
        if not perturb_enabled:
            self.cabinet.body.set_position(self._cabinet_base_pos, True)
            self.cabinet.body.set_quaternion(self._cabinet_base_quat, True)
            cup_pos_center = self._CUP_POS
            cup_pos_extents = self._CUP_POS_EXTENTS
            cup_pos_bounds = self._CUP_POS_BOUNDS
            cup_rot_bounds = self._CUP_ROT_BOUNDS
        else:
            # Domain randomization: jitter cabinet/table pose and expand cup spawn jitter.
            table_z_offset = np.random.uniform(-self._TABLE_Z_BOUNDS, self._TABLE_Z_BOUNDS)
            table_yaw_offset = np.random.uniform(
                -self._TABLE_YAW_BOUNDS, self._TABLE_YAW_BOUNDS
            )
            cabinet_quat = Quaternion(axis=[0, 0, 1], angle=table_yaw_offset) * Quaternion(
                self._cabinet_base_quat
            )
            rot = cabinet_quat.rotation_matrix
            # Keep yaw pivot around cabinet lower center (computed above),
            # and apply z offset only after pivot alignment.
            pivot_world = self._cabinet_pivot_world_base.copy()
            pivot_world[2] += table_z_offset
            cabinet_pos = pivot_world - rot @ self._cabinet_pivot_local
            self.cabinet.body.set_position(cabinet_pos, True)
            self.cabinet.body.set_quaternion(cabinet_quat.elements, True)

            counter_pos = cabinet_pos + rot @ self._counter_local_pos
            cup_pos_center = counter_pos.copy()
            cup_pos_center[2] += self._cup_z_from_counter
            cup_pos_extents = self._CUP_POS_EXTENTS_ENHANCED
            cup_pos_bounds = self._CUP_POS_BOUNDS_ENHANCED
            cup_rot_bounds = self._CUP_ROT_BOUNDS_ENHANCED

        spawn_point = get_random_points_on_plane(
            1,
            cup_pos_center,
            cup_pos_extents,
            self._CUP_STEP,
            cup_pos_bounds,
        )[0]
        if perturb_enabled:
            spawn_point = _rotate_point_xy(spawn_point, cup_pos_center, table_yaw_offset)
        self.cup.body.set_position(spawn_point, True)
        quat = Quaternion(axis=[1, 0, 0], angle=self._CUP_ROT_X)
        angle = np.random.uniform(-cup_rot_bounds, cup_rot_bounds)
        quat *= Quaternion(axis=[0, 0, 1], angle=self._CUP_ROT_Z + angle)
        self.cup.body.set_quaternion(quat.elements, True)


class FlipCutlery(_ManipulationEnv):
    """Flip cutlery item task."""

    DEFAULT_ROBOT = H1FineManipulation

    _CUP_POS = np.array([0.8, 0, 0.86])
    _CUP_ROT_Z = np.deg2rad(180)
    _CUP_STEP = 0.1
    _CUP_POS_EXTENTS = np.array([0.1, 0.25])
    _CUP_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUP_ROT_BOUNDS = np.deg2rad(180)

    _SPOON_OFFSET = np.array([0, 0, 0.15])
    _SPOON_QUAT = Quaternion(axis=[1, 0, 0], degrees=90)

    _TOLERANCE = np.deg2rad(50)

    def _initialize_env(self):
        super()._initialize_env()
        self.cup = Mug(self._mojo, kinematic=False)
        self.spoon = Spoon(self._mojo)

    def _success(self) -> bool:
        down = np.array([0, 0, -1])
        fwd = np.array([0, 1, 0])
        spoon_fwd = Quaternion(self.spoon.body.get_quaternion()).rotate(fwd)
        angle_to_up = np.arccos(np.clip(np.dot(spoon_fwd, down), -1.0, 1.0))
        if angle_to_up > self._TOLERANCE:
            return False
        if not self.spoon.is_colliding(self.cup):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.cup, side):
                return False
        return True

    def _on_reset(self):
        spawn_point = get_random_points_on_plane(
            1,
            self._CUP_POS,
            self._CUP_POS_EXTENTS,
            self._CUP_STEP,
            self._CUP_POS_BOUNDS,
        )[0]
        self.cup.body.set_position(spawn_point)
        angle = np.random.uniform(-self._CUP_ROT_BOUNDS, self._CUP_ROT_BOUNDS)
        quat = Quaternion(axis=[0, 0, 1], angle=self._CUP_ROT_Z + angle)
        self.cup.body.set_quaternion(quat.elements)

        self.spoon.body.set_position(spawn_point + self._SPOON_OFFSET, True)
        self.spoon.body.set_quaternion(self._SPOON_QUAT.elements, True)


class StackBlocks(BiGymEnv):
    """Stack blocks in the correct area of the table."""

    _PRESET_PATH = PRESETS_PATH / "stack_blocks.yaml"

    _NUM_BLOCKS = 3
    _BLOCKS_POS = np.array([0.7, 0, 1])
    _BLOCKS_POS_EXTENTS = np.array([0.2, 0.5])
    _BLOCKS_STEP = 0.15
    _BLOCKS_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    _TARGET_SIZE = np.array([0.05, 0.05, 0.001])
    _TARGET_COLOR = np.array([0.3, 0.8, 0.3, 1.0])
    _TARGET_POS = np.array([1.4, 0, 0.95])
    _TARGET_POS_BOUNDS = np.array([0.05, 0.2, 0.0])

    def _initialize_env(self):
        self.blocks = [Cube(self._mojo) for _ in range(self._NUM_BLOCKS)]
        self.target: Body = Body.create(self._mojo)
        self.target_collider = Geom.create(
            self._mojo,
            parent=self.target,
            geom_type=GeomType.BOX,
            size=self._TARGET_SIZE,
            color=self._TARGET_COLOR,
        )

    def _on_reset(self):
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        for block, point in zip(self.blocks, points):
            block.set_pose(
                point,
                position_bounds=self._BLOCKS_POS_BOUNDS,
                rotation_bounds=self._BLOCKS_ROT_BOUNDS,
            )
        offset = np.random.uniform(-self._TARGET_POS_BOUNDS, self._TARGET_POS_BOUNDS)
        self.target.set_position(self._TARGET_POS + offset)

    def _success(self) -> bool:
        blocks_sorted = sorted(self.blocks, key=lambda b: b.body.get_position()[2])
        if not blocks_sorted[0].is_colliding(self.target_collider):
            return False
        if not blocks_sorted[1].is_colliding(blocks_sorted[0]):
            return False
        if not blocks_sorted[2].is_colliding(blocks_sorted[1]):
            return False
        for block in self.blocks:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(block, side):
                    return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for block in self.blocks:
            if block.is_colliding(self.floor):
                return True
        return False
