"""Depth-to-pointcloud utilities shared across Bigym codepaths."""
from __future__ import annotations

from typing import Any, Optional

import mujoco
import numpy as np


def get_camera_fovy_deg(model: Any, cam_id: int) -> float:
    """Return camera vertical FoV in degrees."""
    fovy = float(model.cam_fovy[cam_id])
    if fovy <= 0:
        fovy = float(model.vis.global_.fovy)
    return fovy


def resolve_camera_id(
    model: Any,
    camera_name: str,
    cameras_map: Optional[dict[str, tuple[int, Any]]] = None,
) -> int:
    """Resolve camera id robustly across namespaced / non-namespaced names."""
    if cameras_map:
        for key in (camera_name, f"rby1/{camera_name}", f"h1/{camera_name}"):
            entry = cameras_map.get(key)
            if entry is None:
                continue
            try:
                cam_id = int(entry[0])
            except Exception:
                cam_id = -1
            if cam_id >= 0:
                return cam_id

    for key in (camera_name, f"rby1/{camera_name}", f"h1/{camera_name}"):
        try:
            cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, key))
        except Exception:
            cam_id = -1
        if cam_id >= 0:
            return cam_id

    for cam_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        if name == camera_name or (name and name.endswith(f"/{camera_name}")):
            return int(cam_id)

    return -1


def camera_intrinsics_from_fovy(
    fovy_deg: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Compute pinhole intrinsics from FoV and image size."""
    fovy_rad = np.deg2rad(fovy_deg)
    fy = 0.5 * float(height) / np.tan(fovy_rad / 2.0)
    fovx = 2.0 * np.arctan(np.tan(fovy_rad / 2.0) * (float(width) / float(height)))
    fx = 0.5 * float(width) / np.tan(fovx / 2.0)
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return fx, fy, cx, cy


def depth_rgb_to_world_pcd(
    depth: np.ndarray,
    rgb: np.ndarray,
    cam_xpos: np.ndarray,
    cam_xmat: np.ndarray,
    fovy_deg: float,
    n_points: int,
    rng: np.random.Generator,
    min_dist: float | None = None,
    max_dist: float | None = None,
    min_world_z: float | None = None,
) -> np.ndarray:
    """Create an XYZRGB point cloud in world frame from camera depth / rgb."""
    if depth is None or rgb is None:
        return np.zeros((n_points, 6), dtype=np.float32)

    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth image (H,W), got {depth.shape}")

    if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
        rgb_img = np.moveaxis(rgb, 0, -1)
    else:
        rgb_img = rgb

    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics_from_fovy(fovy_deg, width, height)

    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    mask = np.isfinite(depth) & (depth > 0)
    if not np.any(mask):
        return np.zeros((n_points, 6), dtype=np.float32)

    z = depth[mask].astype(np.float32)
    x = (xs[mask].astype(np.float32) - cx) * z / fx
    y = (ys[mask].astype(np.float32) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    # MuJoCo camera frame correction (flip Y and Z)
    pts_cam[:, 1] *= -1.0
    pts_cam[:, 2] *= -1.0

    if min_dist is not None or max_dist is not None:
        dist = np.linalg.norm(pts_cam, axis=1)
        keep = np.ones(dist.shape[0], dtype=bool)
        if min_dist is not None:
            keep &= dist >= float(min_dist)
        if max_dist is not None:
            keep &= dist <= float(max_dist)
        if not np.any(keep):
            return np.zeros((n_points, 6), dtype=np.float32)
        pts_cam = pts_cam[keep]
        mask_idx = np.flatnonzero(mask)[keep]
    else:
        mask_idx = np.flatnonzero(mask)

    R = cam_xmat.reshape(3, 3)
    pts_world = pts_cam @ R.T + cam_xpos.reshape(1, 3)

    colors = rgb_img.reshape(-1, rgb_img.shape[-1])[mask_idx].astype(np.float32)
    if colors.size == 0:
        colors = np.zeros((pts_world.shape[0], 3), dtype=np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    if colors.shape[1] != 3:
        colors = colors[:, :3]

    if min_world_z is not None:
        keep = pts_world[:, 2] >= float(min_world_z)
        if not np.any(keep):
            return np.zeros((n_points, 6), dtype=np.float32)
        pts_world = pts_world[keep]
        colors = colors[keep]

    total = pts_world.shape[0]
    replace = total < n_points
    idx = rng.choice(total, size=n_points, replace=replace)
    pts_world = pts_world[idx]
    colors = colors[idx]

    return np.concatenate([pts_world, colors], axis=1).astype(np.float32)


def get_or_resolve_cam_params(
    *,
    model: Any,
    camera_name: str,
    cameras_map: Optional[dict[str, tuple[int, Any]]] = None,
    cam_params_cache: Optional[dict[str, tuple[int, float]]] = None,
) -> tuple[int, float]:
    """Get (cam_id, fovy_deg), resolving and caching if needed."""
    if cam_params_cache is not None and camera_name in cam_params_cache:
        return cam_params_cache[camera_name]

    cam_id = resolve_camera_id(
        model=model,
        camera_name=camera_name,
        cameras_map=cameras_map,
    )
    if cam_id < 0:
        model_cam_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]
        raise ValueError(
            f"Failed to resolve camera '{camera_name}' for pointcloud generation. "
            f"Available model cameras: {model_cam_names}"
        )

    params = (cam_id, get_camera_fovy_deg(model, cam_id))
    if cam_params_cache is not None:
        cam_params_cache[camera_name] = params
    return params


def generate(
    *,
    depth: np.ndarray,
    rgb: np.ndarray,
    camera_name: str,
    model: Any,
    data: Any,
    cameras_map: Optional[dict[str, tuple[int, Any]]] = None,
    cam_params_cache: Optional[dict[str, tuple[int, float]]] = None,
    n_points: int = 1024,
    rng: Optional[np.random.Generator] = None,
    min_dist: float | None = None,
    max_dist: float | None = None,
    min_world_z: float | None = None,
) -> np.ndarray:
    """Generate world-frame XYZRGB pointcloud for one camera."""
    if depth is None or rgb is None:
        raise RuntimeError(
            f"camera '{camera_name}' is configured for point cloud "
            "but missing rgb/depth buffers"
        )

    if rng is None:
        rng = np.random.default_rng()

    cam_id, fovy_deg = get_or_resolve_cam_params(
        model=model,
        camera_name=camera_name,
        cameras_map=cameras_map,
        cam_params_cache=cam_params_cache,
    )

    cam_xpos = np.array(data.cam_xpos[cam_id])
    cam_xmat = np.array(data.cam_xmat[cam_id])
    return depth_rgb_to_world_pcd(
        depth=depth,
        rgb=rgb,
        cam_xpos=cam_xpos,
        cam_xmat=cam_xmat,
        fovy_deg=fovy_deg,
        n_points=n_points,
        rng=rng,
        min_dist=min_dist,
        max_dist=max_dist,
        min_world_z=min_world_z,
    )


class PointCloudGenerator:
    """Stateful pointcloud generator with cached camera params and RNG."""

    def __init__(
        self,
        *,
        model: Any,
        cameras_map: Optional[dict[str, tuple[int, Any]]] = None,
    ):
        self._model = model
        self._cameras_map = cameras_map
        self._cam_params_cache: dict[str, tuple[int, float]] = {}
        self._rng = np.random.default_rng()

    def set_seed(self, seed: Optional[int]) -> None:
        """Seed internal RNG; if None use nondeterministic seed."""
        if seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(int(seed))

    def generate(
        self,
        *,
        depth: np.ndarray,
        rgb: np.ndarray,
        camera_name: str,
        data: Any,
        n_points: int,
        min_dist: float | None = None,
        max_dist: float | None = None,
        min_world_z: float | None = None,
    ) -> np.ndarray:
        """Generate world-frame XYZRGB pointcloud for one camera."""
        return generate(
            depth=depth,
            rgb=rgb,
            camera_name=camera_name,
            model=self._model,
            data=data,
            cameras_map=self._cameras_map,
            cam_params_cache=self._cam_params_cache,
            n_points=n_points,
            rng=self._rng,
            min_dist=min_dist,
            max_dist=max_dist,
            min_world_z=min_world_z,
        )
