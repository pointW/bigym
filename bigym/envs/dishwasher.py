"""Dishwasher interaction tasks."""
from abc import ABC
import os

import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.const import PRESETS_PATH
from bigym.envs.props.dishwasher import Dishwasher
from bigym.utils.observation_config import ObservationConfig


def _bigym_perturb_enabled() -> bool:
    """Return True if task reset perturbation is enabled."""
    value = os.getenv("BIGYM_DISABLE_PERTURB", "0").strip().lower()
    return value not in {"1", "true", "yes", "on"}


class _DishwasherEnv(BiGymEnv, ABC):
    """Base dishwasher environment."""

    RESET_ROBOT_POS = np.array([0, -0.8, 0])

    _PRESET_PATH = PRESETS_PATH / "dishwasher.yaml"
    _TOLERANCE = 0.05
    _DISHWASHER_Z_BOUNDS = 0.1
    _DISHWASHER_YAW_BOUNDS = np.deg2rad(45.0)
    _TRAY_PARTIAL_CLOSED = 0.5

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Dishwasher)[0]
        self._dishwasher_base_pos = self.dishwasher.body.get_position().copy()
        self._dishwasher_base_quat = self.dishwasher.body.get_quaternion().copy()
        base_rot = Quaternion(self._dishwasher_base_quat).rotation_matrix
        # Pivot around the bottom-center of the dishwasher shell in the closed pose.
        self._dishwasher_pivot_local = self._compute_dishwasher_pivot_local()
        self._dishwasher_pivot_world_base = (
            self._dishwasher_base_pos + base_rot @ self._dishwasher_pivot_local
        )

    def _compute_dishwasher_pivot_local(self) -> np.ndarray:
        geoms = [g for g in self.dishwasher.body.geoms if g.is_collidable()]
        if not geoms:
            return np.zeros(3, dtype=np.float64)

        bounds_min = []
        bounds_max = []
        for geom in geoms:
            mjcf = geom.mjcf
            size = getattr(mjcf, "size", None)
            if size is None:
                continue
            size = np.array(size, dtype=np.float64)
            geom_type = getattr(mjcf, "type", "box")
            if geom_type == "sphere":
                extents = np.array([size[0], size[0], size[0]], dtype=np.float64)
            elif geom_type in {"cylinder", "capsule"}:
                extents = np.array([size[0], size[0], size[1]], dtype=np.float64)
            else:
                extents = size
            pos = np.array(getattr(mjcf, "pos", [0.0, 0.0, 0.0]), dtype=np.float64)
            bounds_min.append(pos - extents)
            bounds_max.append(pos + extents)

        if not bounds_min:
            return np.zeros(3, dtype=np.float64)

        mins = np.min(np.stack(bounds_min, axis=0), axis=0)
        maxs = np.max(np.stack(bounds_max, axis=0), axis=0)
        return np.array(
            [(mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5, mins[2]],
            dtype=np.float64,
        )


class DishwasherOpen(_DishwasherEnv):
    """Open the dishwasher door and pull out all trays."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=0, bottom_tray=0, middle_tray=0)


class DishwasherClose(_DishwasherEnv):
    """Push back all trays and close the door of the dishwasher."""

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
            robot_kwargs = {
                "base_perturb_x_range": (-0.2, 0.0),
                "base_perturb_y_range": (-0.2, 0.0),
                "base_perturb_yaw_range": (0.0, np.deg2rad(45.0)),
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

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 0, atol=self._TOLERANCE)

    def _on_reset(self):
        if not _bigym_perturb_enabled():
            self.dishwasher.body.set_position(self._dishwasher_base_pos, True)
            self.dishwasher.body.set_quaternion(self._dishwasher_base_quat, True)
            self.dishwasher.set_state(door=1, bottom_tray=1, middle_tray=1)
            return

        z_offset = np.random.uniform(
            -self._DISHWASHER_Z_BOUNDS, self._DISHWASHER_Z_BOUNDS
        )
        yaw_offset = np.random.uniform(-self._DISHWASHER_YAW_BOUNDS, 0.0)

        base_quat = (
            Quaternion(axis=[0, 0, 1], angle=yaw_offset)
            * Quaternion(self._dishwasher_base_quat)
        )
        rot = base_quat.rotation_matrix
        pivot_world = self._dishwasher_pivot_world_base.copy()
        pivot_world[2] += z_offset
        new_pos = pivot_world - rot @ self._dishwasher_pivot_local

        self.dishwasher.body.set_position(new_pos, True)
        self.dishwasher.body.set_quaternion(base_quat.elements, True)

        self.dishwasher.set_state(door=1, bottom_tray=1, middle_tray=1)


class DishwasherCloseTrays(DishwasherClose):
    """Push the dishwasher’s trays back with the door initially open."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state()[1:], 0, atol=self._TOLERANCE)


class DishwasherOpenTrays(DishwasherClose):
    """Pull out the dishwasher’s trays with the door initially open."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state()[1:], 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=0, middle_tray=0)
