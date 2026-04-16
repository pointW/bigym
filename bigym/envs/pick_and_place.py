"""Pick and place tasks."""

import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet
from bigym.envs.props.cutlery import Spatula
from bigym.envs.props.items import Box, Sandwich
from bigym.envs.props.prop import Prop
from bigym.envs.props.kitchenware import Mug, Saucepan, Pan, ChoppingBoard
from bigym.robots.configs.h1 import H1FineManipulation
from bigym.utils.env_utils import get_random_points_on_plane
def _quat_yaw(quat) -> float:
    quat = Quaternion(quat)
    rot = quat.rotation_matrix
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


class PutCups(BiGymEnv):
    """Put cups in the wall cabinet."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_1x1.yaml"

    _CUPS_COUNT = 2

    _CUPS_POS = np.array([0.8, 0, 1])
    _CUPS_ROT = np.deg2rad(180)
    _CUPS_STEP = 0.15
    _CUPS_POS_EXTENTS = np.array([0.1, 0.25])
    _CUPS_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUPS_ROT_BOUNDS = np.deg2rad(30)

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.cabinet_wall = self._preset.get_props(WallCabinet)[0]
        self.cups = [Mug(self._mojo) for _ in range(self._CUPS_COUNT)]

    def _success(self) -> bool:
        for cup in self.cups:
            if not cup.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True

    def _on_reset(self):
        points = get_random_points_on_plane(
            len(self.cups),
            self._CUPS_POS,
            self._CUPS_POS_EXTENTS,
            self._CUPS_STEP,
            self._CUPS_POS_BOUNDS,
        )
        for cup, point in zip(self.cups, points):
            cup.body.set_position(point)
            angle = np.random.uniform(-self._CUPS_ROT_BOUNDS, self._CUPS_ROT_BOUNDS)
            cup.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._CUPS_ROT + angle).elements
            )


class TakeCups(PutCups):
    """Take cups from the wall cupboard."""

    _CUPS_POS = np.array([1.05, 0, 1.5])

    def _success(self) -> bool:
        for cup in self.cups:
            if not cup.is_colliding(self.cabinet_base.counter):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True


class StoreBox(BiGymEnv):
    """Put box in the cupboard task."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_door.yaml"

    _BOX_POS = np.array([0.8, 0, 1])
    _BOX_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _BOX_ROT_BOUNDS = np.deg2rad(180)

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.box = Box(self._mojo, True)

    def _on_reset(self):
        offset = np.random.uniform(-self._BOX_POS_BOUNDS, self._BOX_POS_BOUNDS)
        self.box.body.set_position(self._BOX_POS + offset, True)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        self.box.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=angle).elements)

    def _success(self) -> bool:
        if not self.box.is_colliding(self.cabinet_base.shelf):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.box, side):
                return False
        return True


class PickBox(StoreBox):
    """Pick up box from and place it on the counter task."""

    _BOX_POS = np.array([0.8, -1, 0.2])
    _BOX_QUAT = Quaternion(axis=[0, 1, 0], degrees=90)

    def _success(self) -> bool:
        if not self.box.is_colliding(self.cabinet_base.counter):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.box, side):
                return False
        return True

    def _on_reset(self):
        offset = np.random.uniform(-self._BOX_POS_BOUNDS, self._BOX_POS_BOUNDS)
        self.box.body.set_position(self._BOX_POS + offset, True)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        quat = self._BOX_QUAT
        quat *= Quaternion(axis=[1, 0, 0], angle=angle)
        self.box.body.set_quaternion(quat.elements)


class SaucepanToHob(BiGymEnv):
    """Take saucepan from cabinet and place it to hob."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_hob.yaml"

    _SAUCEPAN_POS = np.array([0.85, 0.1, 0.5])
    _SAUCEPAN_QUAT = Quaternion(axis=[0, 0, 1], degrees=90)
    _SAUCEPAN_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _SAUCEPAN_ROT_BOUNDS = np.deg2rad([0, 0, 20])

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.saucepan = Saucepan(self._mojo)

    def _success(self) -> bool:
        if not self.saucepan.is_colliding(self.cabinet_base.hob):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.saucepan, side):
                return False
        return True

    def _on_reset(self):
        self.saucepan.set_pose(
            self._SAUCEPAN_POS,
            self._SAUCEPAN_QUAT.elements,
            self._SAUCEPAN_POS_BOUNDS,
            self._SAUCEPAN_ROT_BOUNDS,
        )


class StoreKitchenware(BiGymEnv):
    """Put all kitchenware to cupboard."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_hob.yaml"

    _ITEMS = [Saucepan, Pan]
    _ITEMS_QUAT = Quaternion(axis=[0, 0, 1], degrees=15)
    _ITEMS_POS_BOUNDS = np.array([0.02, 0.02, 0])
    _ITEMS_ROT_BOUNDS = np.deg2rad([0, 0, 30])
    _CABINET_Z_BOUNDS = np.array([-0.1, 0.1])
    _CABINET_YAW_BOUNDS = np.deg2rad(30)
    _SHELF_Z_DELTA_BOUNDS = np.array([-0.10, 0.04])

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.items: list[Prop] = [item(self._mojo) for item in self._ITEMS]
        self._cabinet_base_pos = self.cabinet_base.body.get_position().copy()
        self._cabinet_base_quat = self.cabinet_base.body.get_quaternion().copy()
        base_rot = Quaternion(self._cabinet_base_quat).rotation_matrix
        self._item_site_local_positions = []
        for site in [self.cabinet_base.sites[0], self.cabinet_base.sites[2]]:
            local_pos = base_rot.T @ (site.get_position() - self._cabinet_base_pos)
            self._item_site_local_positions.append(local_pos)
        self._shelf_local_pos = np.array(
            self.cabinet_base.shelf_body.mjcf.pos,
            dtype=np.float64,
        ).copy()
        # Preserve the original "world yaw = 15 deg" spawn orientation, but
        # express it in the cabinet's local frame so it stays semantically
        # identical when the cabinet itself is yaw-perturbed.
        self._item_nominal_local_quat = (
            Quaternion(self._cabinet_base_quat).inverse * self._ITEMS_QUAT
        )

    def _success(self) -> bool:
        for item in self.items:
            if not item.is_colliding(self.cabinet_base.shelf):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(item, side):
                    return False
        return True

    def _on_reset(self):
        perturb_enabled = self.init_perturb_enabled()

        cabinet_pos = self._cabinet_base_pos.copy()
        cabinet_quat = self._cabinet_base_quat.copy()
        shelf_local_pos = self._shelf_local_pos.copy()

        if perturb_enabled:
            cabinet_pos[2] += np.random.uniform(*self._CABINET_Z_BOUNDS)
            yaw_offset = np.random.uniform(
                -self._CABINET_YAW_BOUNDS,
                self._CABINET_YAW_BOUNDS,
            )
            cabinet_quat = (
                Quaternion(axis=[0, 0, 1], angle=yaw_offset)
                * Quaternion(self._cabinet_base_quat)
            ).elements
            shelf_local_pos[2] += np.random.uniform(*self._SHELF_Z_DELTA_BOUNDS)

        self.cabinet_base.body.set_position(cabinet_pos, True)
        self.cabinet_base.body.set_quaternion(cabinet_quat, True)
        self.cabinet_base.shelf_body.set_position(shelf_local_pos, True)

        rot = Quaternion(cabinet_quat).rotation_matrix
        site_local_positions = [pos.copy() for pos in self._item_site_local_positions]
        np.random.shuffle(site_local_positions)
        for item, local_pos in zip(self.items, site_local_positions):
            item_quat = (
                Quaternion(cabinet_quat) * self._item_nominal_local_quat
            ).elements
            if perturb_enabled:
                local_spawn = local_pos.copy()
                local_spawn += np.random.uniform(
                    -self._ITEMS_POS_BOUNDS,
                    self._ITEMS_POS_BOUNDS,
                )
                spawn_pos = cabinet_pos + rot @ local_spawn
                item.set_pose(
                    spawn_pos,
                    item_quat,
                    np.zeros(3),
                    self._ITEMS_ROT_BOUNDS,
                )
            else:
                spawn_pos = cabinet_pos + rot @ local_pos
                item.set_pose(
                    spawn_pos,
                    item_quat,
                    self._ITEMS_POS_BOUNDS,
                    self._ITEMS_ROT_BOUNDS,
                )


class ToastSandwich(BiGymEnv):
    """Move sandwich on the frying pan."""

    DEFAULT_ROBOT = H1FineManipulation

    _PRESET_PATH = PRESETS_PATH / "counter_base_2_hob.yaml"

    _PAN_QUAT = Quaternion(axis=[0, 0, 1], degrees=-90)
    _PAN_POS_BOUNDS = np.array([0.02, 0.02, 0])
    _PAN_ROT_BOUNDS = np.deg2rad([0, 0, 30])

    _SPATULA_OFFSET = np.array([-0.02, 0.02, 0.08])
    _SPATULA_QUAT = Quaternion(axis=[0, 0, 1], degrees=90)

    _BOARD_POS = np.array([0.7, -0.6, 0.88])
    _BOARD_ROT_BOUNDS = np.deg2rad([0, 0, 5])

    _TOLERANCE = np.deg2rad(10)
    _TOASTED = False
    _ROUNDED = False

    _SANDWICH_OFFSET = np.array([0, 0, 0.05])
    _SANDWICH_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _SANDWICH_ROT_BOUNDS = np.deg2rad([0, 0, 45])

    @property
    def _sandwich_anchor(self) -> Prop:
        return self.board

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.pan = Pan(self._mojo)
        self.spatula = Spatula(self._mojo)
        self.board = ChoppingBoard(self._mojo)
        self.sandwich = Sandwich(
            self._mojo, toasted=self._TOASTED, rounded_collider=self._ROUNDED
        )

    def _on_reset(self):
        site = self.cabinet_base.sites[0]
        self.pan.set_pose(
            site.get_position(),
            self._PAN_QUAT.elements,
            self._PAN_POS_BOUNDS,
            self._PAN_ROT_BOUNDS,
        )
        self.spatula.set_pose(
            self.pan.body.get_position() + self._SPATULA_OFFSET,
            self._SPATULA_QUAT.elements,
        )
        self.board.set_pose(self._BOARD_POS, rotation_bounds=self._BOARD_ROT_BOUNDS)
        self.sandwich.set_pose(
            self._sandwich_anchor.body.get_position() + self._SANDWICH_OFFSET,
            position_bounds=self._SANDWICH_POS_BOUNDS,
            rotation_bounds=self._SANDWICH_ROT_BOUNDS,
        )

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.body.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(sandwich_up, up), -1.0, 1.0))
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if not (angle_to_up <= self._TOLERANCE or angle_to_down <= self._TOLERANCE):
            return False
        if not self.pan.is_colliding(self.cabinet_base.hob):
            return False
        if not self.sandwich.is_colliding(self.pan):
            return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.sandwich, side):
                return True
        return False


class FlipSandwich(ToastSandwich):
    """Flip sandwich using spatula."""

    _PAN_QUAT_PERTURB = Quaternion(axis=[0, 0, 1], degrees=-100)
    _PAN_ROT_BOUNDS_PERTURB = np.deg2rad([0, 0, 15])
    _SANDWICH_OFFSET = np.array([0, 0, 0.04])
    _SANDWICH_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _SANDWICH_REL_YAW_RANGE_PERTURB = np.deg2rad([0, 20])
    _SANDWICH_ROT_BOUNDS_NO_PERTURB = np.deg2rad([0, 0, 180])

    _ROUNDED = True

    @property
    def _sandwich_anchor(self) -> Prop:
        return self.pan

    def _on_reset(self):
        perturb_enabled = self.init_perturb_enabled()
        site = self.cabinet_base.sites[0]
        self.pan.set_pose(
            site.get_position(),
            (
                self._PAN_QUAT_PERTURB.elements
                if perturb_enabled
                else self._PAN_QUAT.elements
            ),
            self._PAN_POS_BOUNDS,
            (
                self._PAN_ROT_BOUNDS_PERTURB
                if perturb_enabled
                else self._PAN_ROT_BOUNDS
            ),
        )
        self.spatula.set_pose(
            self.pan.body.get_position() + self._SPATULA_OFFSET,
            self._SPATULA_QUAT.elements,
        )
        self.board.set_pose(self._BOARD_POS, rotation_bounds=self._BOARD_ROT_BOUNDS)
        if perturb_enabled:
            pan_yaw = _quat_yaw(self.pan.body.get_quaternion())
            sandwich_rel_yaw = np.random.uniform(
                self._SANDWICH_REL_YAW_RANGE_PERTURB[0],
                self._SANDWICH_REL_YAW_RANGE_PERTURB[1],
            )
            sandwich_quat = Quaternion(axis=[0, 0, 1], angle=pan_yaw + sandwich_rel_yaw)
            self.sandwich.set_pose(
                self._sandwich_anchor.body.get_position() + self._SANDWICH_OFFSET,
                quat=sandwich_quat.elements,
                position_bounds=self._SANDWICH_POS_BOUNDS,
                rotation_bounds=np.zeros(3),
            )
        else:
            self.sandwich.set_pose(
                self._sandwich_anchor.body.get_position() + self._SANDWICH_OFFSET,
                position_bounds=self._SANDWICH_POS_BOUNDS,
                rotation_bounds=self._SANDWICH_ROT_BOUNDS_NO_PERTURB,
            )

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.body.get_quaternion()).rotate(up)
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if angle_to_down > self._TOLERANCE:
            return False
        if not self.pan.is_colliding(self.cabinet_base.hob):
            return False
        if not self.sandwich.is_colliding(self.pan):
            return False
        return True


class RemoveSandwich(FlipSandwich):
    """Remove sandwich from the frying pan."""

    _TOASTED = True

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.body.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(sandwich_up, up), -1.0, 1.0))
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if not (angle_to_up <= self._TOLERANCE or angle_to_down <= self._TOLERANCE):
            return False
        if not self.sandwich.is_colliding(self.board):
            return False
        return True
