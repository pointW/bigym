"""Load/unload cups to/from dishwasher."""
from abc import ABC
import os

import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet
from bigym.envs.props.dishwasher import Dishwasher
from bigym.envs.props.kitchenware import Mug
from bigym.utils.env_utils import get_random_sites
from bigym.utils.observation_config import ObservationConfig


def _bigym_perturb_enabled() -> bool:
    """Return True if task reset perturbation is enabled."""
    value = os.getenv("BIGYM_DISABLE_PERTURB", "0").strip().lower()
    return value not in {"1", "true", "yes", "on"}


class _DishwasherCupsEnv(BiGymEnv, ABC):
    """Base cups environment."""

    RESET_ROBOT_POS = np.array([0, -0.6, 0])

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher.yaml"
    _CUPS_COUNT = 2

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Dishwasher)[0]
        self.cabinets = self._preset.get_props(BaseCabinet)
        self.cups = [Mug(self._mojo) for _ in range(self._CUPS_COUNT)]

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for cup in self.cups:
            if cup.is_colliding(self.floor):
                return True
        return False

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=0, middle_tray=1)


class DishwasherUnloadCups(_DishwasherCupsEnv):
    """Unload cups from dishwasher task."""

    _SITES_STEP = 3
    _SITES_SLICE = 3

    _CUPS_ROT_X = np.deg2rad(180)
    _CUPS_ROT_Z = np.deg2rad(90)
    _CUPS_ROT_BOUNDS = np.deg2rad(5)
    _CUPS_POS = np.array([0, -0.05, 0.05])
    _CUPS_STEP = np.array([0.115, 0, 0])

    def _success(self) -> bool:
        for cup in self.cups:
            if not any([cup.is_colliding(cabinet) for cabinet in self.cabinets]):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True

    def _on_reset(self):
        super()._on_reset()
        sites = self.dishwasher.tray_middle.site_sets[0]
        sites = get_random_sites(
            sites, len(self.cups), self._SITES_STEP, self._SITES_SLICE
        )
        for site, cup in zip(sites, self.cups):
            quat = Quaternion(axis=[1, 0, 0], angle=self._CUPS_ROT_X)
            angle = np.random.uniform(-self._CUPS_ROT_BOUNDS, self._CUPS_ROT_BOUNDS)
            quat *= Quaternion(axis=[0, 0, 1], angle=self._CUPS_ROT_Z + angle)
            cup.body.set_quaternion(quat.elements, True)
            pos = site.get_position()
            pos += self._CUPS_POS
            cup.body.set_position(pos, True)


class DishwasherUnloadCupsLong(DishwasherUnloadCups):
    """Unload cup from dishwasher in wall cabinet task."""

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher_wall_cabinet.yaml"
    _CUPS_COUNT = 1
    _SITE_PREFIX_COUNT = 4
    _CUPS_ROT_BOUNDS = np.pi

    _COUNTERTOP_Z_BOUNDS = 0.1
    _WALL_CABINET_Z_BOUNDS = 0.2
    _WALL_CABINET_YAW_RANGE = (0.0, np.pi)
    _TOLERANCE = 0.1

    def __init__(
        self,
        action_mode,
        observation_config: ObservationConfig = ObservationConfig(),
        render_mode=None,
        start_seed=None,
        control_frequency: int = CONTROL_FREQUENCY_MAX,
        robot_cls=None,
        robot_kwargs=None,
        reset_warmup_steps=None,
    ):
        resolved_robot_cls = robot_cls or self.DEFAULT_ROBOT
        if robot_kwargs is None and getattr(resolved_robot_cls, "__name__", None) in {
            "RBY1",
            "RBY1FineManipulation",
        }:
            robot_kwargs = {
                "base_perturb_x_range": (-0.5, 0.0),
                "base_perturb_y_range": (-0.5, 0.0),
                "base_perturb_yaw_range": (0.0, np.deg2rad(90.0)),
            }

        super().__init__(
            action_mode=action_mode,
            observation_config=observation_config,
            render_mode=render_mode,
            start_seed=start_seed,
            control_frequency=control_frequency,
            robot_cls=robot_cls,
            robot_kwargs=robot_kwargs,
            reset_warmup_steps=reset_warmup_steps,
        )

    def _initialize_env(self):
        super()._initialize_env()
        self.wall_cabinet = self._preset.get_props(WallCabinet)[0]
        self._countertop_base_poses = [
            (
                prop.body,
                prop.body.get_position().copy(),
                prop.body.get_quaternion().copy(),
            )
            for prop in [self.dishwasher, *self.cabinets]
        ]

        counter_centers = [
            cabinet.counter.get_position().copy()
            for cabinet in self.cabinets
            if getattr(cabinet, "counter", None) is not None
        ]
        if counter_centers:
            self._countertop_pivot_world_base = np.mean(
                np.stack(counter_centers, axis=0), axis=0
            )
        else:
            self._countertop_pivot_world_base = self.dishwasher.body.get_position().copy()

        self._wall_cabinet_base_pos = self.wall_cabinet.body.get_position().copy()
        self._wall_cabinet_base_quat = self.wall_cabinet.body.get_quaternion().copy()

    def _set_countertop_pose(self, z_offset: float, yaw_offset: float):
        yaw_quat = Quaternion(axis=[0, 0, 1], angle=yaw_offset)
        rot = yaw_quat.rotation_matrix
        pivot = self._countertop_pivot_world_base

        for body, base_pos, base_quat in self._countertop_base_poses:
            rel_pos = base_pos - pivot
            new_pos = pivot + rot @ rel_pos
            new_pos[2] += z_offset
            new_quat = (yaw_quat * Quaternion(base_quat)).elements
            body.set_position(new_pos, True)
            body.set_quaternion(new_quat, True)

    def _set_wall_cabinet_pose(self, z_offset: float, yaw_offset: float):
        yaw_quat = Quaternion(axis=[0, 0, 1], angle=yaw_offset)
        new_pos = self._wall_cabinet_base_pos.copy()
        new_pos[2] += z_offset
        new_quat = (yaw_quat * Quaternion(self._wall_cabinet_base_quat)).elements
        self.wall_cabinet.body.set_position(new_pos, True)
        self.wall_cabinet.body.set_quaternion(new_quat, True)

    def _sample_cup_sites(self):
        candidates = []
        for site_set in self.dishwasher.tray_middle.site_sets:
            candidates.extend(site_set[: self._SITE_PREFIX_COUNT])

        if len(candidates) < len(self.cups):
            raise ValueError(
                "Not enough cup sites for DishwasherUnloadCupsLong reset."
            )

        candidates = np.array(candidates, dtype=object)
        return np.random.choice(candidates, size=len(self.cups), replace=False).tolist()

    def _on_reset(self):
        if not _bigym_perturb_enabled():
            self._set_countertop_pose(z_offset=0.0, yaw_offset=0.0)
            self._set_wall_cabinet_pose(z_offset=0.0, yaw_offset=0.0)
        else:
            countertop_z_offset = np.random.uniform(
                -self._COUNTERTOP_Z_BOUNDS, self._COUNTERTOP_Z_BOUNDS
            )
            self._set_countertop_pose(
                z_offset=countertop_z_offset, yaw_offset=0.0
            )

            cabinet_z_offset = np.random.uniform(
                -self._WALL_CABINET_Z_BOUNDS, self._WALL_CABINET_Z_BOUNDS
            )
            cabinet_yaw_offset = np.random.uniform(*self._WALL_CABINET_YAW_RANGE)
            self._set_wall_cabinet_pose(
                z_offset=cabinet_z_offset, yaw_offset=cabinet_yaw_offset
            )

        _DishwasherCupsEnv._on_reset(self)
        sites = self._sample_cup_sites()
        for site, cup in zip(sites, self.cups):
            quat = Quaternion(axis=[1, 0, 0], angle=self._CUPS_ROT_X)
            angle = np.random.uniform(-self._CUPS_ROT_BOUNDS, self._CUPS_ROT_BOUNDS)
            quat *= Quaternion(axis=[0, 0, 1], angle=self._CUPS_ROT_Z + angle)
            cup.body.set_quaternion(quat.elements, True)
            pos = site.get_position().copy()
            pos += self._CUPS_POS
            cup.body.set_position(pos, True)

    def _success(self) -> bool:
        if not np.allclose(self.dishwasher.get_state(), 0, atol=self._TOLERANCE):
            return False
        if not np.allclose(self.wall_cabinet.get_state(), 0, atol=self._TOLERANCE):
            return False
        for cup in self.cups:
            if not cup.is_colliding(self.wall_cabinet.shelf_bottom):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True


class DishwasherLoadCups(_DishwasherCupsEnv):
    """Load cups to dishwasher task."""

    _CUPS_POS = np.array([0.6, -0.6, 1])
    _CUPS_POS_STEP = np.array([0, 0.15, 0])
    _CUPS_POS_BOUNDS = np.array([0.05, 0.02, 0])
    _CUPS_ROT_X = np.deg2rad(180)
    _CUPS_ROT_Z = np.deg2rad(180)
    _CUPS_ROT_BOUNDS = np.deg2rad(30)

    def _success(self) -> bool:
        for cup in self.cups:
            if not cup.is_colliding(self.dishwasher.tray_middle.colliders):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True

    def _on_reset(self):
        super()._on_reset()
        for i, cup in enumerate(self.cups):
            quat = Quaternion(axis=[1, 0, 0], angle=self._CUPS_ROT_X)
            angle = np.random.uniform(-self._CUPS_ROT_BOUNDS, self._CUPS_ROT_BOUNDS)
            quat *= Quaternion(axis=[0, 0, 1], angle=self._CUPS_ROT_Z + angle)
            cup.body.set_quaternion(quat.elements, True)
            pos = self._CUPS_POS + i * self._CUPS_POS_STEP
            pos += np.random.uniform(-self._CUPS_POS_BOUNDS, self._CUPS_POS_BOUNDS)
            cup.body.set_position(pos, True)
