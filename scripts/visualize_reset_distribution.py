#!/usr/bin/env python3
"""Visualize reset distribution by resetting an env at a fixed interval."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from bigym.robots.configs.rby1 import RBY1


ENV_MODULES = {
    "FlipCup": "bigym.envs.manipulation",
    "FlipCutlery": "bigym.envs.manipulation",
    "StackBlocks": "bigym.envs.manipulation",
    "MovePlate": "bigym.envs.move_plates",
    "MoveTwoPlates": "bigym.envs.move_plates",
    "StoreKitchenware": "bigym.envs.pick_and_place",
}


def _resolve_env_class(env: str):
    # Accept "module.path:ClassName" for custom environments.
    if ":" in env:
        module_name, class_name = env.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    module_name = ENV_MODULES.get(env)
    if module_name is None:
        raise ValueError(
            f"Unknown env '{env}'. Use one of {sorted(ENV_MODULES.keys())} "
            f"or pass module path as 'module.path:ClassName'."
        )
    module = importlib.import_module(module_name)
    return getattr(module, env)


def _configure_perturb_flags(enable_env_perturb: bool, enable_robot_perturb: bool):
    if enable_env_perturb:
        os.environ.pop("BIGYM_DISABLE_PERTURB", None)
    else:
        os.environ["BIGYM_DISABLE_PERTURB"] = "1"

    if enable_robot_perturb:
        os.environ.pop("RBY1_DISABLE_PERTURB", None)
    else:
        os.environ["RBY1_DISABLE_PERTURB"] = "1"

    # If this stays set, robot perturbation can be identical every reset.
    os.environ.pop("RBY1_PERTURB_SEED", None)


def _to_rgb_frame(frame):
    import numpy as np

    img = np.asarray(frame)
    if img.ndim != 3:
        raise RuntimeError(f"Unexpected render frame shape: {img.shape}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _capture_camera_state(cam):
    return {
        "lookat": [float(cam.lookat[i]) for i in range(3)],
        "distance": float(cam.distance),
        "azimuth": float(cam.azimuth),
        "elevation": float(cam.elevation),
    }


def _apply_camera_overrides(cam, base_state, args: argparse.Namespace):
    for i, delta in enumerate(
        (args.camera_lookat_dx, args.camera_lookat_dy, args.camera_lookat_dz)
    ):
        cam.lookat[i] = base_state["lookat"][i] + float(delta)
    cam.distance = base_state["distance"] + float(args.camera_distance_delta)
    cam.azimuth = (
        base_state["azimuth"] + float(args.camera_azimuth_delta)
        if args.camera_azimuth is None
        else float(args.camera_azimuth)
    )
    cam.elevation = (
        base_state["elevation"] + float(args.camera_elevation_delta)
        if args.camera_elevation is None
        else float(args.camera_elevation)
    )


def _maybe_prepare_camera(env, render_mode: str, args: argparse.Namespace, base_state):
    if render_mode is None:
        return base_state

    needs_override = any(
        value != 0.0
        for value in (
            args.camera_lookat_dx,
            args.camera_lookat_dy,
            args.camera_lookat_dz,
            args.camera_distance_delta,
            args.camera_azimuth_delta,
            args.camera_elevation_delta,
        )
    ) or args.camera_azimuth is not None or args.camera_elevation is not None
    if not needs_override:
        return base_state

    # Lazily initialize the viewer and cache its default free-camera pose.
    if base_state is None:
        env.render()
        viewer = env.mujoco_renderer.get_viewer(render_mode)
        base_state = _capture_camera_state(viewer.cam)
        print(f"[reset-viz] base camera: {base_state}")
    viewer = env.mujoco_renderer.get_viewer(render_mode)
    _apply_camera_overrides(viewer.cam, base_state, args)
    return base_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset an env at fixed intervals to visualize reset distribution."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="FlipCup",
        help=(
            "Env class name (e.g. FlipCup, MovePlate) or module path "
            "in format module.path:ClassName."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between resets (default: 1.0).",
    )
    parser.add_argument(
        "--num-resets",
        type=int,
        default=0,
        help="Number of resets to run. 0 means infinite until Ctrl+C.",
    )
    parser.add_argument(
        "--control-frequency",
        type=int,
        default=20,
        help="Control frequency for env/action mode (default: 20).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="If set, use seed=seed_base+i for reset i.",
    )
    parser.add_argument(
        "--disable-env-perturb",
        action="store_true",
        help="Disable environment perturbation (BIGYM).",
    )
    parser.add_argument(
        "--disable-robot-perturb",
        action="store_true",
        help="Disable robot perturbation (RBY1).",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=None,
        help="If set, save reset distribution as GIF at this path.",
    )
    parser.add_argument(
        "--gif-frame-interval",
        type=float,
        default=0.1,
        help="Seconds per GIF frame (default: 0.1).",
    )
    parser.add_argument(
        "--gif-duration",
        type=float,
        default=10.0,
        help="GIF duration in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--gif-loop",
        type=int,
        default=0,
        help="GIF loop count: 0 means infinite (default: 0).",
    )
    parser.add_argument(
        "--camera-lookat-dx",
        type=float,
        default=0.0,
        help="Additive x offset for the free camera lookat point.",
    )
    parser.add_argument(
        "--camera-lookat-dy",
        type=float,
        default=0.0,
        help="Additive y offset for the free camera lookat point.",
    )
    parser.add_argument(
        "--camera-lookat-dz",
        type=float,
        default=0.0,
        help="Additive z offset for the free camera lookat point.",
    )
    parser.add_argument(
        "--camera-distance-delta",
        type=float,
        default=0.0,
        help="Additive offset for free camera distance.",
    )
    parser.add_argument(
        "--camera-azimuth-delta",
        type=float,
        default=0.0,
        help="Additive offset for free camera azimuth in degrees.",
    )
    parser.add_argument(
        "--camera-elevation-delta",
        type=float,
        default=0.0,
        help="Additive offset for free camera elevation in degrees.",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=None,
        help="Absolute free camera azimuth in degrees. Overrides delta if set.",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=None,
        help="Absolute free camera elevation in degrees. Overrides delta if set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _configure_perturb_flags(
        enable_env_perturb=not args.disable_env_perturb,
        enable_robot_perturb=not args.disable_robot_perturb,
    )

    env_cls = _resolve_env_class(args.env)
    action_mode = RBY1CartesianActionModeWholeBody(
        direct_mode=False,
        block_until_reached=False,
        control_frequency=args.control_frequency,
    )
    save_gif = args.output_gif is not None
    render_mode = "rgb_array" if save_gif else (None if args.headless else "human")
    env = env_cls(
        action_mode=action_mode,
        control_frequency=args.control_frequency,
        render_mode=render_mode,
        robot_cls=RBY1,
    )

    print(
        f"[reset-viz] env={env_cls.__name__}, interval={args.interval}s, "
        f"BIGYM_DISABLE_PERTURB={os.getenv('BIGYM_DISABLE_PERTURB', '0')}, "
        f"RBY1_DISABLE_PERTURB={os.getenv('RBY1_DISABLE_PERTURB', '0')}, "
        f"RBY1_PERTURB_SEED={os.getenv('RBY1_PERTURB_SEED', '<unset>')}"
    )

    reset_idx = 0
    camera_base_state = None
    try:
        if save_gif:
            import imageio.v2 as imageio

            num_frames = max(1, int(round(args.gif_duration / args.gif_frame_interval)))
            frames = []
            for frame_idx in range(num_frames):
                seed = None if args.seed_base is None else args.seed_base + frame_idx
                env.reset(seed=seed)
                camera_base_state = _maybe_prepare_camera(
                    env=env,
                    render_mode=render_mode,
                    args=args,
                    base_state=camera_base_state,
                )
                frame = env.render()
                frames.append(_to_rgb_frame(frame))
                print(f"[reset-viz] gif frame {frame_idx + 1}/{num_frames} seed={seed}")

            args.output_gif.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                args.output_gif,
                frames,
                duration=args.gif_frame_interval,
                loop=args.gif_loop,
            )
            print(
                f"[reset-viz] saved GIF: {args.output_gif} "
                f"(frames={num_frames}, duration={args.gif_duration}s, "
                f"frame_interval={args.gif_frame_interval}s)"
            )
        else:
            while args.num_resets <= 0 or reset_idx < args.num_resets:
                seed = None if args.seed_base is None else args.seed_base + reset_idx
                env.reset(seed=seed)
                camera_base_state = _maybe_prepare_camera(
                    env=env,
                    render_mode=render_mode,
                    args=args,
                    base_state=camera_base_state,
                )
                reset_idx += 1
                print(f"[reset-viz] reset #{reset_idx} seed={seed}")

                deadline = time.monotonic() + args.interval
                while True:
                    now = time.monotonic()
                    if now >= deadline:
                        break
                    if not args.headless:
                        env.render()
                    time.sleep(min(0.02, deadline - now))
    except KeyboardInterrupt:
        print("\n[reset-viz] interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
