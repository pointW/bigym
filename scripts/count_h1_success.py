#!/usr/bin/env python3
"""Count H1 demo success rate for a BiGym task without conversion."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib
import multiprocessing as mp
from pathlib import Path
from typing import List, Type, Optional, Tuple, Dict, Any

import numpy as np
import imageio.v2 as imageio

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from bigym.robots.configs.h1 import H1


def detect_floating_dofs_from_demos(env_name: str) -> List[PelvisDof]:
    """Detect the correct floating DOFs for an environment based on available demos."""
    # Environments that typically use 4 DOF (X, Y, Z, RZ) based on demo analysis
    four_dof_envs = {
        'FlipCup', 'FlipCutlery', 'FlipSandwich',
        'StackBlocks',
        'ToastSandwich', 'RemoveSandwich',
        'SaucepanToHob',
        'StoreBox', 'PickBox',
        'StoreKitchenware',
        'GroceriesStoreLower', 'GroceriesStoreUpper',
        'TakeCups', 'PutCups',
        'DishwasherOpen', 'DishwasherClose', 'DishwasherOpenTrays', 'DishwasherCloseTrays',
        'DishwasherLoadCups', 'DishwasherLoadCutlery', 'DishwasherLoadPlates',
        'DishwasherUnloadCups', 'DishwasherUnloadCutlery', 'DishwasherUnloadPlates',
        'DishwasherUnloadPlatesLong', 'DishwasherUnloadCutleryLong', 'DishwasherUnloadCupsLong',
    }

    if env_name in four_dof_envs:
        return [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    return [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]


def get_environment_class(env_name: str) -> Type:
    """Dynamically import and return the environment class."""
    env_modules = {
        # Core environments
        'ReachTarget': 'bigym.envs.reach_target',
        'ReachTargetSingle': 'bigym.envs.reach_target',
        'ReachTargetDual': 'bigym.envs.reach_target',
        'MovePlate': 'bigym.envs.move_plates',
        'MovePlates': 'bigym.envs.move_plates',
        'MoveTwoPlates': 'bigym.envs.move_plates',

        # Dishwasher tasks
        'DishwasherOpen': 'bigym.envs.dishwasher',
        'DishwasherClose': 'bigym.envs.dishwasher',
        'DishwasherOpenTrays': 'bigym.envs.dishwasher',
        'DishwasherCloseTrays': 'bigym.envs.dishwasher',
        'DishwasherLoadCups': 'bigym.envs.dishwasher_cups',
        'DishwasherLoadCutlery': 'bigym.envs.dishwasher',
        'DishwasherLoadPlates': 'bigym.envs.dishwasher',
        'DishwasherUnloadCups': 'bigym.envs.dishwasher',
        'DishwasherUnloadCupsLong': 'bigym.envs.dishwasher',
        'DishwasherUnloadCutlery': 'bigym.envs.dishwasher',
        'DishwasherUnloadCutleryLong': 'bigym.envs.dishwasher',
        'DishwasherUnloadPlates': 'bigym.envs.dishwasher',
        'DishwasherUnloadPlatesLong': 'bigym.envs.dishwasher',

        # Manipulation tasks
        'FlipCup': 'bigym.envs.manipulation',
        'FlipCutlery': 'bigym.envs.manipulation',
        'FlipSandwich': 'bigym.envs.manipulation',
        'StackBlocks': 'bigym.envs.manipulation',

        # Kitchen tasks
        'ToastSandwich': 'bigym.envs.kitchen',
        'RemoveSandwich': 'bigym.envs.kitchen',
        'SaucepanToHob': 'bigym.envs.kitchen',

        # Storage tasks
        'StoreBox': 'bigym.envs.storage',
        'PickBox': 'bigym.envs.storage',
        'StoreKitchenware': 'bigym.envs.storage',
        'GroceriesStoreLower': 'bigym.envs.storage',
        'GroceriesStoreUpper': 'bigym.envs.storage',
        'TakeCups': 'bigym.envs.pick_and_place',
        'PutCups': 'bigym.envs.storage',

        # Cupboard/Drawer tasks
        'CupboardsOpenAll': 'bigym.envs.cupboards',
        'CupboardsCloseAll': 'bigym.envs.cupboards',
        'WallCupboardOpen': 'bigym.envs.cupboards',
        'WallCupboardClose': 'bigym.envs.cupboards',
        'DrawersAllOpen': 'bigym.envs.drawers',
        'DrawersAllClose': 'bigym.envs.drawers',
        'DrawerTopOpen': 'bigym.envs.drawers',
        'DrawerTopClose': 'bigym.envs.drawers',
    }

    if env_name == 'MovePlates':
        class_name = 'MovePlate'
    elif env_name == 'MoveTwoPlates':
        class_name = 'MovePlate'
    elif env_name.startswith('ReachTarget'):
        class_name = 'ReachTarget'
    else:
        class_name = env_name

    if env_name not in env_modules:
        module_name = f"bigym.envs.{env_name.lower()}"
        try:
            module = importlib.import_module(module_name)
            return getattr(module, env_name)
        except (ImportError, AttributeError):
            module_name = f"bigym.envs.{env_name.replace('_', '').lower()}"
            try:
                module = importlib.import_module(module_name)
                return getattr(module, env_name)
            except (ImportError, AttributeError):
                raise ValueError(f"Unknown environment: {env_name}")

    module = importlib.import_module(env_modules[env_name])
    return getattr(module, class_name)

def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.shape[0] in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    return frame.astype(np.uint8)


def _extract_frame(obs: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    if obs is None:
        return None
    frame = obs.get(key)
    if frame is None:
        return None
    return _normalize_frame(frame)


def _write_gif(
    frames_iter,
    output_path: Path,
    control_frequency: int,
    gif_stride: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(gif_stride))
    gif_fps = control_frequency / float(stride) if control_frequency > 0 else 1.0
    frame_duration = 1.0 / max(gif_fps, 1e-6)
    writer = imageio.get_writer(output_path, mode="I", duration=frame_duration)
    try:
        for frame in frames_iter:
            if frame is None:
                continue
            writer.append_data(frame)
    finally:
        writer.close()


def _run_demo_worker(payload: Tuple) -> Tuple[int, int, bool, Optional[int], float]:
    (
        demo_idx,
        demo,
        env_name,
        control_frequency,
        floating_dofs,
        save_gifs,
        gif_dir,
        gif_stride,
        gif_camera,
        gif_resolution,
    ) = payload

    env_class = get_environment_class(env_name)
    if save_gifs:
        camera_configs = [
            CameraConfig(
                name=gif_camera,
                rgb=True,
                depth=False,
                resolution=gif_resolution,
            )
        ]
        observation_config = ObservationConfig(cameras=camera_configs)
    else:
        observation_config = ObservationConfig(cameras=[])

    env = env_class(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=floating_dofs,
        ),
        control_frequency=control_frequency,
        observation_config=observation_config,
        render_mode=None,
        robot_cls=H1,
    )

    gif_key = f"rgb_{gif_camera}"
    frames = []
    reset_out = env.reset(seed=demo.seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    if save_gifs:
        frames.append(_extract_frame(obs, gif_key))

    success = False
    max_reward = 0.0
    success_step = None

    for step_idx, timestep in enumerate(demo.timesteps):
        action = timestep.info.get('demo_action')
        if action is None:
            action = getattr(timestep, "executed_action", None)
        if action is None:
            continue

        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        max_reward = max(max_reward, float(reward))

        if save_gifs and (step_idx % max(1, int(gif_stride)) == 0):
            frames.append(_extract_frame(obs, gif_key))

        if info.get('task_success', False):
            success = True
            success_step = step_idx + 1
            break

        if terminated or truncated:
            break

    if save_gifs and any(frame is not None for frame in frames):
        gif_path = Path(gif_dir) / f"h1_{env_name.lower()}_{demo_idx:03d}_seed{demo.seed}.gif"
        _write_gif(frames, gif_path, control_frequency, gif_stride)

    env.close()
    return demo_idx, demo.seed, success, success_step, max_reward


def count_h1_success(
    env_name: str,
    n_demos: int,
    control_frequency: int,
    save_gifs: bool = False,
    gif_dir: Optional[str] = None,
    gif_stride: int = 30,
    gif_camera: str = "head",
    gif_resolution: Tuple[int, int] = (84, 84),
    processes: int = 1,
) -> int:
    env_class = get_environment_class(env_name)
    floating_dofs = detect_floating_dofs_from_demos(env_name)
    dof_str = "4 DOF (X,Y,Z,RZ)" if len(floating_dofs) == 4 else "3 DOF (X,Y,RZ)"
    print(f"Using {dof_str} floating base for {env_name}")

    # Create H1 environment to load demos
    env = env_class(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=floating_dofs,
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=[]),
        render_mode=None,
        robot_cls=H1,
    )

    demo_store = DemoStore()
    metadata = Metadata.from_env(env)
    demos = demo_store.get_demos(metadata, amount=n_demos, frequency=control_frequency)

    env.close()

    if save_gifs and gif_dir is None:
        gif_dir = f"h1_gifs_{env_name.lower()}"

    payloads = [
        (
            i,
            demo,
            env_name,
            control_frequency,
            floating_dofs,
            save_gifs,
            gif_dir,
            gif_stride,
            gif_camera,
            gif_resolution,
        )
        for i, demo in enumerate(demos)
    ]

    results: List[Tuple[int, int, bool, Optional[int], float]] = []
    if processes > 1:
        print(f"Using multiprocessing with {processes} processes")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=processes) as pool:
            for result in pool.imap_unordered(_run_demo_worker, payloads):
                results.append(result)
    else:
        for payload in payloads:
            results.append(_run_demo_worker(payload))

    results.sort(key=lambda x: x[0])
    success_count = 0
    total = len(results)
    for demo_idx, seed, success, success_step, max_reward in results:
        print(f"Demo {demo_idx+1}/{total} (seed={seed}):", end=" ")
        if success:
            print(f"success at step {success_step}")
            success_count += 1
        else:
            print(f"failed (max_reward={max_reward:.3f})")

    print(
        f"\nH1 success count: {success_count}/{total} "
        f"({(success_count/total*100.0) if total else 0.0:.1f}%)"
    )
    return success_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Count H1 demo successes for a BiGym task.")
    parser.add_argument("--env", type=str, default="DishwasherClose", help="Environment name")
    parser.add_argument("--n-demos", type=int, default=-1, help="Number of demos to test (-1 for all)")
    parser.add_argument("--control-freq", type=int, default=50, help="Control frequency")
    parser.add_argument("--processes", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--save-gifs", action="store_true", help="Save rollout GIFs")
    parser.add_argument(
        "--gif-dir",
        type=str,
        default=None,
        help="Directory to save GIFs (default: h1_gifs_<env>)",
    )
    parser.add_argument("--gif-stride", type=int, default=30, help="Frame stride for GIFs")
    parser.add_argument("--gif-camera", type=str, default="head", help="Camera name for GIFs")
    args = parser.parse_args()

    count_h1_success(
        args.env,
        args.n_demos,
        args.control_freq,
        save_gifs=args.save_gifs,
        gif_dir=args.gif_dir,
        gif_stride=args.gif_stride,
        gif_camera=args.gif_camera,
        processes=args.processes,
    )


if __name__ == "__main__":
    main()
