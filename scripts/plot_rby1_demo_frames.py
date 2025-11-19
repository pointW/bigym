#!/usr/bin/env python3
"""Plot RBY1 end-effector and head frames from proprioception demos."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # Ensure headless rendering works everywhere
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from demonstrations.demo import Demo  # noqa: E402


FRAME_SPECS = {
    "head": {
        "pos_key": "head_site_pos",
        "quat_key": "head_site_quat",
        "color": "#9467bd",
        "label": "Head",
    },
    "left": {
        "pos_key": "left_ee_pos",
        "quat_key": "left_ee_quat",
        "color": "#1f77b4",
        "label": "Left EE",
    },
    "right": {
        "pos_key": "right_ee_pos",
        "quat_key": "right_ee_quat",
        "color": "#d62728",
        "label": "Right EE",
    },
}

AXIS_COLORS = ("#ff7f0e", "#2ca02c", "#17becf")  # X, Y, Z


def _load_demo(path: Path) -> Demo:
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")
    demo = Demo.from_safetensors(path)
    if demo is None or not demo.timesteps:
        raise ValueError(f"No timesteps found in demo {path}")
    return demo


def _downsample_indices(length: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("Downsample factor must be a positive integer.")
    indices = list(range(0, length, stride))
    return indices if indices else [0]


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo-format quaternion (wxyz) to 3x3 rotation matrix."""
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must have 4 components, got {quat}")
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float64,
    )


def _plot_frame_axes(
    ax: plt.Axes,
    origin: np.ndarray,
    rotation: np.ndarray,
    length: float,
    alpha: float,
):
    """Draw a triad at `origin` with axes given by `rotation` columns."""
    for axis_idx, color in enumerate(AXIS_COLORS):
        direction = rotation[:, axis_idx]
        segment = np.stack([origin, origin + direction * length], axis=0)
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=1.5,
            alpha=alpha,
        )


def _set_axes_equal(ax: plt.Axes, points: np.ndarray):
    """Make the 3D plot axes have equal scale."""
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    max_range = (maxs - mins).max()
    if max_range <= 0:
        max_range = 1.0
    half = max_range / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def plot_demo_frames(
    demo_path: Path,
    output_path: Path,
    downsample: int,
    axis_length: float,
    alpha: float = 0.7,
) -> Path:
    demo = _load_demo(demo_path)
    indices = _downsample_indices(len(demo.timesteps), downsample)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    trajectories: dict[str, list[np.ndarray]] = {key: [] for key in FRAME_SPECS}
    all_points: list[np.ndarray] = []

    for step_idx in indices:
        timestep = demo.timesteps[step_idx]
        obs = timestep.observation
        for key, spec in FRAME_SPECS.items():
            if spec["pos_key"] not in obs or spec["quat_key"] not in obs:
                raise KeyError(
                    f"Step {step_idx} missing '{spec['pos_key']}' or '{spec['quat_key']}'."
                )
            pos = np.asarray(obs[spec["pos_key"]], dtype=np.float64).reshape(-1)[:3]
            quat = np.asarray(obs[spec["quat_key"]], dtype=np.float64).reshape(-1)[:4]
            rot = _quat_to_matrix(quat)
            _plot_frame_axes(ax, pos, rot, axis_length, alpha=alpha)
            trajectories[key].append(pos)
            all_points.append(pos)

    for key, points in trajectories.items():
        if not points:
            continue
        traj = np.vstack(points)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=FRAME_SPECS[key]["color"],
            linewidth=2.0,
            label=FRAME_SPECS[key]["label"],
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        f"{demo_path.stem} frames (downsample x{downsample})",
        pad=12,
    )
    _set_axes_equal(ax, np.vstack(all_points) if all_points else np.zeros((1, 3)))

    legend_handles = [
        Line2D([0], [0], color=spec["color"], lw=2, label=spec["label"])
        for spec in FRAME_SPECS.values()
    ]
    ax.legend(handles=legend_handles, loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RBY1 end-effector and head frames from proprioception data."
    )
    parser.add_argument(
        "--demo-path",
        type=Path,
        default=Path("rby1_cartesian_demos_moveplate/rby1_cartesian_demo_000.safetensors"),
        help="Path to the demo safetensors file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rby1_demo_frames.png"),
        help="Where to save the rendered figure.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Plot every Nth timestep (default: 10).",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Length of each coordinate axis drawn at a frame origin (in meters).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = plot_demo_frames(
        args.demo_path,
        args.output,
        args.downsample,
        args.axis_length,
    )
    print(f"Saved frame plot to {output_path}")


if __name__ == "__main__":
    main()
