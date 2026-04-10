#!/usr/bin/env python3
"""Export one successful H1 demo replay GIF per task."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BIGYM_REPO = REPO_ROOT / "bigym"
if str(BIGYM_REPO) not in sys.path:
    sys.path.insert(0, str(BIGYM_REPO))

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.robots.configs.h1 import H1
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from demonstrations.demo import Demo
from demonstrations.demo_store import DemoNotFoundError, DemoStore
from demonstrations.utils import Metadata


ENV_MODULES: dict[str, tuple[str, str | None]] = {
    "ReachTarget": ("bigym.envs.reach_target", None),
    "ReachTargetSingle": ("bigym.envs.reach_target", None),
    "ReachTargetDual": ("bigym.envs.reach_target", None),
    "MovePlate": ("bigym.envs.move_plates", None),
    "MovePlates": ("bigym.envs.move_plates", "MovePlate"),
    "MoveTwoPlates": ("bigym.envs.move_plates", None),
    "DishwasherOpen": ("bigym.envs.dishwasher", None),
    "DishwasherClose": ("bigym.envs.dishwasher", None),
    "DishwasherOpenTrays": ("bigym.envs.dishwasher", None),
    "DishwasherCloseTrays": ("bigym.envs.dishwasher", None),
    "DishwasherLoadCups": ("bigym.envs.dishwasher_cups", None),
    "DishwasherUnloadCups": ("bigym.envs.dishwasher_cups", None),
    "DishwasherUnloadCupsLong": ("bigym.envs.dishwasher_cups", None),
    "DishwasherLoadCutlery": ("bigym.envs.dishwasher_cutlery", None),
    "DishwasherUnloadCutlery": ("bigym.envs.dishwasher_cutlery", None),
    "DishwasherUnloadCutleryLong": ("bigym.envs.dishwasher_cutlery", None),
    "DishwasherLoadPlates": ("bigym.envs.dishwasher_plates", None),
    "DishwasherUnloadPlates": ("bigym.envs.dishwasher_plates", None),
    "DishwasherUnloadPlatesLong": ("bigym.envs.dishwasher_plates", None),
    "FlipCup": ("bigym.envs.manipulation", None),
    "FlipCutlery": ("bigym.envs.manipulation", None),
    "StackBlocks": ("bigym.envs.manipulation", None),
    "PutCups": ("bigym.envs.pick_and_place", None),
    "TakeCups": ("bigym.envs.pick_and_place", None),
    "StoreBox": ("bigym.envs.pick_and_place", None),
    "PickBox": ("bigym.envs.pick_and_place", None),
    "SaucepanToHob": ("bigym.envs.pick_and_place", None),
    "StoreKitchenware": ("bigym.envs.pick_and_place", None),
    "ToastSandwich": ("bigym.envs.pick_and_place", None),
    "FlipSandwich": ("bigym.envs.pick_and_place", None),
    "RemoveSandwich": ("bigym.envs.pick_and_place", None),
    "GroceriesStoreLower": ("bigym.envs.groceries", None),
    "GroceriesStoreUpper": ("bigym.envs.groceries", None),
    "DrawerTopOpen": ("bigym.envs.cupboards", None),
    "DrawerTopClose": ("bigym.envs.cupboards", None),
    "DrawersAllOpen": ("bigym.envs.cupboards", None),
    "DrawersAllClose": ("bigym.envs.cupboards", None),
    "WallCupboardOpen": ("bigym.envs.cupboards", None),
    "WallCupboardClose": ("bigym.envs.cupboards", None),
    "CupboardsOpenAll": ("bigym.envs.cupboards", None),
    "CupboardsCloseAll": ("bigym.envs.cupboards", None),
}


def get_environment_class(env_name: str):
    if env_name not in ENV_MODULES:
        raise ValueError(f"Unknown environment: {env_name}")
    module_name, class_name = ENV_MODULES[env_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name or env_name)


def detect_floating_dofs_from_name(env_name: str) -> list[PelvisDof]:
    four_dof_envs = {
        "FlipCup",
        "FlipCutlery",
        "FlipSandwich",
        "StackBlocks",
        "ToastSandwich",
        "RemoveSandwich",
        "SaucepanToHob",
        "StoreBox",
        "PickBox",
        "StoreKitchenware",
        "GroceriesStoreLower",
        "GroceriesStoreUpper",
        "TakeCups",
        "PutCups",
        "DishwasherOpen",
        "DishwasherClose",
        "DishwasherOpenTrays",
        "DishwasherCloseTrays",
        "DishwasherLoadCups",
        "DishwasherLoadCutlery",
        "DishwasherLoadPlates",
        "DishwasherUnloadCups",
        "DishwasherUnloadCutlery",
        "DishwasherUnloadPlates",
        "DishwasherUnloadPlatesLong",
        "DishwasherUnloadCutleryLong",
        "DishwasherUnloadCupsLong",
    }
    if env_name in four_dof_envs:
        return [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    return [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]


def load_source_demos(
    env_name: str,
    control_frequency: int,
    demo_store: DemoStore,
) -> list[Demo]:
    env_class = get_environment_class(env_name)
    preferred_dofs = detect_floating_dofs_from_name(env_name)
    fallback_dofs = (
        [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]
        if len(preferred_dofs) == 4
        else [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    )
    dof_candidates = [preferred_dofs]
    if fallback_dofs != preferred_dofs:
        dof_candidates.append(fallback_dofs)

    robot_candidates = [env_class.DEFAULT_ROBOT]
    if env_class.DEFAULT_ROBOT != H1:
        robot_candidates.append(H1)
    absolute_candidates = [True, False]

    last_error: Exception | None = None
    demos: list[Demo] | None = None
    chosen: tuple[list[str], bool, str] | None = None

    for floating_dofs in dof_candidates:
        for absolute in absolute_candidates:
            for robot_cls in robot_candidates:
                env = env_class(
                    action_mode=JointPositionActionMode(
                        floating_base=True,
                        absolute=absolute,
                        floating_dofs=floating_dofs,
                    ),
                    control_frequency=control_frequency,
                    observation_config=ObservationConfig(cameras=[]),
                    render_mode=None,
                    robot_cls=robot_cls,
                )
                try:
                    metadata = Metadata.from_env(env)
                    demos = demo_store.get_demos(
                        metadata,
                        amount=-1,
                        frequency=control_frequency,
                    )
                    chosen = (
                        [dof.value for dof in floating_dofs],
                        absolute,
                        robot_cls.__name__,
                    )
                    break
                except DemoNotFoundError as exc:
                    last_error = exc
                    demos = None
                finally:
                    env.close()
            if demos is not None:
                break
        if demos is not None:
            break

    if demos is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to load demos for {env_name}")

    print(
        f"[{env_name}] source lookup:"
        f" dofs={chosen[0]} absolute={chosen[1]} robot={chosen[2]} demos={len(demos)}"
    )
    return sorted(demos, key=lambda demo: int(demo.seed))


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    return frame.astype(np.uint8)


def extract_frame(obs: dict[str, Any] | None, key: str) -> np.ndarray | None:
    if obs is None:
        return None
    frame = obs.get(key)
    if frame is None:
        return None
    return normalize_frame(frame)


def save_gif(frames: list[np.ndarray | None], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, mode="I", duration=1.0 / max(fps, 1e-6)) as writer:
        for frame in frames:
            if frame is not None:
                writer.append_data(frame)


def extract_action(timestep) -> np.ndarray | None:
    action = getattr(timestep, "executed_action", None)
    if action is None:
        action = timestep.info.get("demo_action")
    if action is None:
        action = getattr(timestep, "action", None)
    return action


def make_free_camera(preset: str, env) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    if preset == "task_preview":
        cam.distance = 3.8
        cam.azimuth = -65.0
        cam.elevation = -25.0
        cam.lookat[:] = np.asarray([0.5, 0.0, 1.0], dtype=np.float64)
        return cam

    if preset == "third_person":
        center = np.asarray([0.8, 0.0, 1.0], dtype=np.float64)
        if hasattr(env, "rack_start") and hasattr(env, "rack_target"):
            try:
                start_pos = np.asarray(env.rack_start.body.get_position(), dtype=np.float64)
                target_pos = np.asarray(env.rack_target.body.get_position(), dtype=np.float64)
                center = 0.5 * (start_pos + target_pos)
                center[2] += 0.2
            except Exception:
                pass
        elif hasattr(env, "cabinet") and hasattr(env.cabinet, "counter"):
            try:
                center = np.asarray(env.cabinet.counter.get_position(), dtype=np.float64)
            except Exception:
                pass
        cam.lookat[:] = center
        cam.distance = 1.5
        cam.azimuth = 135.0
        cam.elevation = -20.0
        return cam

    raise ValueError(f"Unknown free-camera preset: {preset}")


def replay_until_success(
    demo: Demo,
    control_frequency: int,
    camera_name: str,
    resolution: tuple[int, int],
    gif_stride: int,
    capture_frames: bool,
    camera_mode: str,
    camera_preset: str,
):
    env_class = demo.metadata.env_cls
    render_mode = None
    if capture_frames and camera_mode == "obs":
        observation_config = ObservationConfig(
            cameras=[
                CameraConfig(
                    name=camera_name,
                    rgb=True,
                    depth=False,
                    resolution=resolution,
                )
            ]
        )
    elif capture_frames and camera_mode == "free":
        observation_config = ObservationConfig(cameras=[])
        render_mode = "rgb_array"
    else:
        observation_config = ObservationConfig(cameras=[])
    env = env_class(
        action_mode=demo.metadata.get_action_mode(),
        control_frequency=control_frequency,
        observation_config=observation_config,
        render_mode=render_mode,
        robot_cls=demo.metadata.robot_cls,
    )

    gif_key = f"rgb_{camera_name}"
    frames: list[np.ndarray | None] = [] if capture_frames else []
    success = False
    success_step: int | None = None
    max_reward = 0.0

    try:
        reset_out = env.reset(seed=int(demo.seed))
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        if capture_frames:
            if camera_mode == "obs":
                frames.append(extract_frame(obs, gif_key))
            else:
                env.mojo.physics.model.vis.global_.offwidth = resolution[0]
                env.mojo.physics.model.vis.global_.offheight = resolution[1]
                env.camera_id = -1
                viewer = env.mujoco_renderer.get_viewer("rgb_array")
                viewer.cam = make_free_camera(camera_preset, env)
                frames.append(normalize_frame(env.render()))

        for step_idx, timestep in enumerate(demo.timesteps):
            action = extract_action(timestep)
            if action is None:
                continue

            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            max_reward = max(max_reward, float(reward))

            if capture_frames and step_idx % max(1, gif_stride) == 0:
                if camera_mode == "obs":
                    frames.append(extract_frame(obs, gif_key))
                else:
                    frames.append(normalize_frame(env.render()))

            if info.get("task_success", False):
                success = True
                success_step = step_idx + 1
                break

            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "success": success,
        "success_step": success_step,
        "max_reward": max_reward,
        "frames": frames,
    }


def export_first_success(
    env_name: str,
    control_frequency: int,
    output_dir: Path,
    camera_name: str,
    resolution: tuple[int, int],
    gif_stride: int,
    disable_perturb: bool,
    demo_store: DemoStore,
    camera_mode: str,
    camera_preset: str,
) -> dict[str, Any]:
    prev_disable_perturb = os.getenv("BIGYM_DISABLE_PERTURB")
    if disable_perturb:
        os.environ["BIGYM_DISABLE_PERTURB"] = "1"
    else:
        os.environ.pop("BIGYM_DISABLE_PERTURB", None)

    try:
        demos = load_source_demos(env_name, control_frequency, demo_store)
        for demo_idx, demo in enumerate(demos):
            result = replay_until_success(
                demo=demo,
                control_frequency=control_frequency,
                camera_name=camera_name,
                resolution=resolution,
                gif_stride=gif_stride,
                capture_frames=False,
                camera_mode="obs",
                camera_preset=camera_preset,
            )
            print(
                f"[{env_name}] demo {demo_idx + 1}/{len(demos)} seed={int(demo.seed)}:"
                f" success={result['success']}"
                + (
                    f" step={result['success_step']}"
                    if result["success_step"] is not None
                    else f" max_reward={result['max_reward']:.3f}"
                )
            )
            if not result["success"]:
                continue

            render_result = replay_until_success(
                demo=demo,
                control_frequency=control_frequency,
                camera_name=camera_name,
                resolution=resolution,
                gif_stride=gif_stride,
                capture_frames=True,
                camera_mode=camera_mode,
                camera_preset=camera_preset,
            )
            if not render_result["success"]:
                raise RuntimeError(
                    f"Success replay became inconsistent while rendering {env_name}"
                )

            view_tag = camera_name if camera_mode == "obs" else camera_preset
            file_name = f"{env_name}_seed{int(demo.seed)}_{view_tag}.gif"
            gif_path = output_dir / file_name
            gif_fps = control_frequency / float(max(1, gif_stride))
            save_gif(render_result["frames"], gif_path, fps=gif_fps)
            return {
                "env": env_name,
                "seed": int(demo.seed),
                "demo_index": demo_idx,
                "success_step": int(render_result["success_step"]),
                "camera": camera_name if camera_mode == "obs" else camera_preset,
                "camera_mode": camera_mode,
                "resolution": list(resolution),
                "gif_stride": int(gif_stride),
                "gif_path": str(gif_path),
                "disable_perturb": bool(disable_perturb),
            }

        raise RuntimeError(f"No successful H1 demo found for {env_name}")
    finally:
        if prev_disable_perturb is None:
            os.environ.pop("BIGYM_DISABLE_PERTURB", None)
        else:
            os.environ["BIGYM_DISABLE_PERTURB"] = prev_disable_perturb


def export_seed_record(
    env_name: str,
    seed: int,
    control_frequency: int,
    output_dir: Path,
    camera_name: str,
    resolution: tuple[int, int],
    gif_stride: int,
    disable_perturb: bool,
    demo_store: DemoStore,
    camera_mode: str,
    camera_preset: str,
) -> dict[str, Any]:
    prev_disable_perturb = os.getenv("BIGYM_DISABLE_PERTURB")
    if disable_perturb:
        os.environ["BIGYM_DISABLE_PERTURB"] = "1"
    else:
        os.environ.pop("BIGYM_DISABLE_PERTURB", None)

    try:
        demos = load_source_demos(env_name, control_frequency, demo_store)
        target_demo_idx = None
        target_demo = None
        for demo_idx, demo in enumerate(demos):
            if int(demo.seed) == int(seed):
                target_demo_idx = demo_idx
                target_demo = demo
                break
        if target_demo is None:
            raise RuntimeError(f"Seed {seed} not found for {env_name}")

        render_result = replay_until_success(
            demo=target_demo,
            control_frequency=control_frequency,
            camera_name=camera_name,
            resolution=resolution,
            gif_stride=gif_stride,
            capture_frames=True,
            camera_mode=camera_mode,
            camera_preset=camera_preset,
        )
        if not render_result["success"]:
            raise RuntimeError(f"Seed {seed} did not replay successfully for {env_name}")

        view_tag = camera_name if camera_mode == "obs" else camera_preset
        gif_path = output_dir / f"{env_name}_seed{int(seed)}_{view_tag}.gif"
        gif_fps = control_frequency / float(max(1, gif_stride))
        save_gif(render_result["frames"], gif_path, fps=gif_fps)
        return {
            "env": env_name,
            "seed": int(seed),
            "demo_index": int(target_demo_idx),
            "success_step": int(render_result["success_step"]),
            "camera": camera_name if camera_mode == "obs" else camera_preset,
            "camera_mode": camera_mode,
            "resolution": list(resolution),
            "gif_stride": int(gif_stride),
            "gif_path": str(gif_path),
            "disable_perturb": bool(disable_perturb),
        }
    finally:
        if prev_disable_perturb is None:
            os.environ.pop("BIGYM_DISABLE_PERTURB", None)
        else:
            os.environ["BIGYM_DISABLE_PERTURB"] = prev_disable_perturb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Task names to export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "h1_success_gifs",
        help="Directory for GIF outputs",
    )
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--camera", type=str, default="external")
    parser.add_argument(
        "--camera-mode",
        choices=("obs", "free"),
        default="obs",
        help="obs: named BiGym camera, free: custom viewer camera",
    )
    parser.add_argument(
        "--camera-preset",
        choices=("task_preview", "third_person"),
        default="task_preview",
        help="Preset for --camera-mode free",
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--gif-stride", type=int, default=5)
    parser.add_argument(
        "--disable-perturb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay with BIGYM_DISABLE_PERTURB=1 by default",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for JSON manifest",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=None,
        help="If set, rerender the listed env/seed pairs instead of searching successes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolution = (args.width, args.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_root_env = os.getenv("BIGYM_CACHE_ROOT")
    demo_store = (
        DemoStore(cache_root=Path(cache_root_env).expanduser())
        if cache_root_env
        else DemoStore()
    )

    manifest: list[dict[str, Any]] = []
    if args.input_manifest is not None:
        input_records = json.loads(args.input_manifest.read_text(encoding="utf-8"))
        for item in input_records:
            env_name = str(item["env"])
            if args.envs and env_name not in args.envs:
                continue
            record = export_seed_record(
                env_name=env_name,
                seed=int(item["seed"]),
                control_frequency=args.control_freq,
                output_dir=args.output_dir,
                camera_name=args.camera,
                resolution=resolution,
                gif_stride=args.gif_stride,
                disable_perturb=bool(item.get("disable_perturb", args.disable_perturb)),
                demo_store=demo_store,
                camera_mode=args.camera_mode,
                camera_preset=args.camera_preset,
            )
            manifest.append(record)
    else:
        if not args.envs:
            raise ValueError("Provide --envs or --input-manifest")
        for env_name in args.envs:
            record = export_first_success(
                env_name=env_name,
                control_frequency=args.control_freq,
                output_dir=args.output_dir,
                camera_name=args.camera,
                resolution=resolution,
                gif_stride=args.gif_stride,
                disable_perturb=args.disable_perturb,
                demo_store=demo_store,
                camera_mode=args.camera_mode,
                camera_preset=args.camera_preset,
            )
            manifest.append(record)

    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
