#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from safetensors import safe_open


DEFAULT_RGB_KEYS = ("obs_rgb_head", "obs_rgb_left_wrist", "obs_rgb_right_wrist")


def load_rgb_streams_safetensors(
    path: Path, rgb_keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    streams: dict[str, np.ndarray] = {}
    with safe_open(path, framework="np", device="cpu") as handle:
        for key in rgb_keys:
            if key in handle.keys():
                streams[key] = handle.get_tensor(key)
    if not streams:
        raise ValueError(f"No RGB observations found in {path}")
    return streams


def load_rgb_streams_hdf5(path: Path, rgb_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    streams: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        obs_group = handle["data"]["demo_0"]["obs"]
        for key in rgb_keys:
            hdf5_key = key.removeprefix("obs_")
            if hdf5_key in obs_group:
                streams[key] = obs_group[hdf5_key][()]
    if not streams:
        raise ValueError(f"No RGB observations found in {path}")
    return streams


def load_rgb_streams(path: Path, rgb_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if path.suffix == ".safetensors":
        return load_rgb_streams_safetensors(path, rgb_keys)
    if path.suffix == ".hdf5":
        return load_rgb_streams_hdf5(path, rgb_keys)
    raise ValueError(f"Unsupported demo format: {path}")


def chw_to_hwc(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {frame.shape}")
    return np.transpose(frame, (1, 2, 0))


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 14), fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(255, 255, 255))
    return np.asarray(img)


def stack_stream_frames(
    streams: dict[str, np.ndarray], step: int, rgb_keys: tuple[str, ...]
) -> np.ndarray:
    ordered = []
    for key in rgb_keys:
        if key not in streams:
            continue
        frame = chw_to_hwc(streams[key][step]).astype(np.uint8, copy=False)
        if len(rgb_keys) == 1:
            ordered.append(frame)
        else:
            ordered.append(add_label(frame, key.removeprefix("obs_rgb_")))
    return ordered[0] if len(ordered) == 1 else np.concatenate(ordered, axis=1)


def save_demo_gif(
    path: Path, out_path: Path, stride: int, fps: int, rgb_keys: tuple[str, ...]
) -> None:
    streams = load_rgb_streams(path, rgb_keys)
    demo_len = min(stream.shape[0] for stream in streams.values())
    frames = [
        stack_stream_frames(streams, step, rgb_keys) for step in range(0, demo_len, stride)
    ]
    if (demo_len - 1) % stride != 0:
        frames.append(stack_stream_frames(streams, demo_len - 1, rgb_keys))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)


def iter_demo_files(source_dir: Path) -> list[tuple[str, Path]]:
    success = sorted(source_dir.glob("*.safetensors"))
    failure = sorted((source_dir / "failure").glob("*.safetensors"))
    return [("success", path) for path in success] + [("failure", path) for path in failure]


def iter_hdf5_success_files(source_dir: Path) -> list[tuple[str, Path]]:
    success = sorted(source_dir.glob("shard_*/bigym_flip_cup_demo/tmp/*.hdf5"))
    return [("success", path) for path in success]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GIFs from BiGym safetensors demos.")
    parser.add_argument("source_dir", type=Path, help="Directory containing success demos and failure/ subdir")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <source_dir>/gifs",
    )
    parser.add_argument("--stride", type=int, default=4, help="Frame stride for GIF generation")
    parser.add_argument("--fps", type=int, default=10, help="GIF fps")
    parser.add_argument(
        "--camera",
        choices=("head", "left_wrist", "right_wrist", "all"),
        default="all",
        help="Which RGB camera stream to export",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "safetensors", "hdf5"),
        default="auto",
        help="Input dataset format",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = (args.output_dir or (source_dir / "gifs")).resolve()
    if args.camera == "all":
        rgb_keys = DEFAULT_RGB_KEYS
    else:
        rgb_keys = (f"obs_rgb_{args.camera}",)

    if args.format == "safetensors":
        demo_entries = iter_demo_files(source_dir)
    elif args.format == "hdf5":
        demo_entries = iter_hdf5_success_files(source_dir)
    else:
        demo_entries = iter_hdf5_success_files(source_dir)
        if not demo_entries:
            demo_entries = iter_demo_files(source_dir)

    for status, demo_path in demo_entries:
        out_name = demo_path.name.replace(".safetensors", ".gif").replace(".hdf5", ".gif")
        out_path = output_dir / status / out_name
        save_demo_gif(
            demo_path,
            out_path,
            stride=max(1, args.stride),
            fps=max(1, args.fps),
            rgb_keys=rgb_keys,
        )
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
