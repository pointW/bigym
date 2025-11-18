#!/usr/bin/env python3
"""Plot RGB observations from an RBY1 Cartesian demo."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # Ensure headless rendering works everywhere
import matplotlib.pyplot as plt
import numpy as np

from demonstrations.demo import Demo  # noqa: E402


CAMERA_KEYS = [
    ("rgb_head", "Head"),
    ("rgb_left_wrist", "Left wrist"),
    ("rgb_right_wrist", "Right wrist"),
]


def _load_demo(path: Path) -> Demo:
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")
    demo = Demo.from_safetensors(path)
    if demo is None or not demo.timesteps:
        raise ValueError(f"No timesteps found in demo {path}")
    return demo


def _prepare_image(array: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) observations to (H, W, 3) float images."""
    if array is None:
        raise ValueError("Missing RGB observation.")
    np_array = np.asarray(array)
    if np_array.ndim != 3:
        raise ValueError(f"Unexpected RGB array shape: {np_array.shape}")

    # Move channels to the last dimension if needed
    if np_array.shape[0] in {3, 4}:
        np_array = np_array[:3, ...]  # Allow accidental RGBA inputs
        np_array = np.transpose(np_array, (1, 2, 0))
    elif np_array.shape[-1] in {3, 4}:
        np_array = np_array[..., :3]
    else:
        raise ValueError(f"Unable to infer channel dimension for shape {np_array.shape}")

    np_array = np_array.astype(np.float32)
    max_val = float(np_array.max()) if np_array.size else 0.0
    if max_val > 1.0:
        np_array /= 255.0
    np_array = np.clip(np_array, 0.0, 1.0)
    return np_array


def _downsample_indices(length: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("Downsample factor must be a positive integer.")
    indices = list(range(0, length, stride))
    return indices if indices else [0]


def plot_demo_rgb(
    demo_path: Path,
    output_path: Path,
    downsample: int,
    dpi: int = 200,
) -> Path:
    demo = _load_demo(demo_path)
    indices = _downsample_indices(len(demo.timesteps), downsample)

    n_rows = len(CAMERA_KEYS)
    n_cols = len(indices)

    col_width = 1.6
    fig_height = 3.2 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(6.0, col_width * n_cols), fig_height),
        squeeze=False,
    )

    for row, (camera_key, row_label) in enumerate(CAMERA_KEYS):
        for col, step_idx in enumerate(indices):
            ax = axes[row, col]
            timestep = demo.timesteps[step_idx]
            rgb = timestep.observation.get(camera_key)
            if rgb is None:
                raise KeyError(
                    f"Observation '{camera_key}' missing at step {step_idx}."
                )
            ax.imshow(_prepare_image(rgb))
            if row == 0:
                ax.set_title(f"Step {step_idx}", fontsize=8)
            ax.axis("off")
        axes[row, 0].set_ylabel(
            row_label,
            rotation=90,
            fontsize=10,
            labelpad=10,
            va="center",
        )

    fig.suptitle(
        f"{demo_path.stem} RGB observations (downsample x{downsample})",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot head and wrist RGB observations from an RBY1 demo."
    )
    parser.add_argument(
        "--demo-path",
        type=Path,
        required=True,
        help="Path to the rby1_cartesian_demo_XXX.safetensors file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rby1_demo_rgb.png"),
        help="Where to save the combined PNG grid.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Plot every Nth frame (default: 10).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI used when writing the PNG.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = plot_demo_rgb(args.demo_path, args.output, args.downsample, args.dpi)
    print(f"Saved RGB grid to {output_path}")


if __name__ == "__main__":
    main()
