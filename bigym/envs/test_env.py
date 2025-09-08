from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces
from mojo import Mojo
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType

from bigym.bigym_env import BiGymEnv
from bigym.const import HandSide
from bigym.robots.robot import Robot

class TestEnv(BiGymEnv, ABC):
    def _success(self) -> bool:
        return False

    def _on_step(self):
        """Highlight spheres even in fast mode."""
        self._success()