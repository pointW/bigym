"""Configuration classes for environment observations."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraConfig:
    """Configuration for camera observations."""
    name: str
    rgb: bool = True
    depth: bool = False
    pcd: bool = False
    pcd_points: int = 1024
    pcd_min_dist: Optional[float] = None
    pcd_max_dist: Optional[float] = 3.0
    pcd_min_world_z: Optional[float] = None
    resolution: tuple[int, int] = (128, 128)
    pos: Optional[tuple[float, float, float]] = None
    quat: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self):
        """Validation."""
        assert len(self.resolution) == 2
        if not isinstance(self.resolution, tuple):
            self.resolution = tuple(self.resolution)
        if self.pos is not None:
            assert len(self.pos) == 3
            if not isinstance(self.pos, tuple):
                self.pos = tuple(self.pos)
        if self.quat is not None:
            assert len(self.quat) == 4
            if not isinstance(self.quat, tuple):
                self.quat = tuple(self.quat)
        self.pcd_points = int(self.pcd_points)
        if self.pcd_points <= 0:
            raise ValueError("pcd_points must be a positive integer")

    @classmethod
    def from_safetensors_metadata(cls, metadata: dict):
        """Get metadata from a safetensor metadata dict."""
        metadata = dict(metadata)
        metadata.pop("pcd_keep_depth", None)  # backward compatibility
        camera_config = cls(**metadata)
        camera_config.resolution = tuple(camera_config.resolution)
        return camera_config

    def to_string(self):
        """Get a string representation of the camera configuration."""
        s = self.name
        if self.rgb:
            s += "-rgb"
        if self.depth:
            s += "-depth"
        if self.pcd:
            s += "-pcd"
        s += "-" + "x".join(map(str, self.resolution))
        return s


@dataclass
class ObservationConfig:
    """Configuration for environment observations."""

    cameras: list[CameraConfig] = field(default_factory=list)
    proprioception: bool = True
    privileged_information: bool = False

    @classmethod
    def from_safetensors_metadata(cls, metadata: dict):
        """Get metadata from a safetensor file."""
        metadata["cameras"] = [
            CameraConfig.from_safetensors_metadata(camera)
            for camera in metadata["cameras"]
        ]
        return cls(**metadata)
