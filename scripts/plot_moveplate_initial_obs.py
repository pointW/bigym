#!/usr/bin/env python3
"""Reset MovePlate with RBY1 and dump the initial RGB observations."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bigym.envs.move_plates import MovePlate
from bigym.rby1_cartesian_action_mode_whole_body import (
    RBY1CartesianActionModeWholeBody,
)
from bigym.robots.configs.rby1 import RBY1
from bigym.utils.observation_config import ObservationConfig, CameraConfig

CAMERA_ORDER = [
    ("rgb_head", "Head"),
    ("rgb_left_wrist", "Left wrist"),
    ("rgb_right_wrist", "Right wrist"),
]


def _prepare_image(rgb: np.ndarray) -> np.ndarray:
    if rgb is None:
        raise ValueError("Missing RGB observation.")
    arr = np.asarray(rgb)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {arr.shape}")
    if arr.shape[0] in {3, 4}:
        arr = arr[:3]
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] in {3, 4}:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")
    arr = arr.astype(np.float32)
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def capture_initial_obs(output: Path, resolution: int, seed: int | None) -> Path:
    cameras = [
        CameraConfig("head", resolution=(resolution, resolution)),
        CameraConfig("left_wrist", resolution=(resolution, resolution)),
        CameraConfig("right_wrist", resolution=(resolution, resolution)),
    ]
    observation_config = ObservationConfig(cameras=cameras)

    action_mode = RBY1CartesianActionModeWholeBody(
        direct_mode=False, block_until_reached=False, control_frequency=50
    )
    env = MovePlate(
        action_mode=action_mode,
        control_frequency=50,
        observation_config=observation_config,
        robot_cls=RBY1,
    )
    try:
        obs, *_ = env.reset(seed=seed)
    finally:
        env.close()

    fig, axes = plt.subplots(
        1, len(CAMERA_ORDER), figsize=(4 * len(CAMERA_ORDER), 4), squeeze=False
    )
    for idx, (key, title) in enumerate(CAMERA_ORDER):
        ax = axes[0, idx]
        ax.imshow(_prepare_image(obs[key]))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("MovePlate initial RGB observations")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grab the initial MovePlate RGB observations for camera debugging."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("moveplate_initial_obs.png"),
        help="Where to store the resulting PNG (default: moveplate_initial_obs.png).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=84,
        help="Square resolution for each camera stream (default: 84).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional env seed to make resets reproducible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    png_path = capture_initial_obs(args.output, args.resolution, args.seed)
    print(f"Saved initial observations to {png_path}")


if __name__ == "__main__":
    main()
