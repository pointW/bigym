#!/usr/bin/env python3
"""Replay the exact H1 source demos used by Step1 conversion."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Type

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BIGYM_REPO = REPO_ROOT / "bigym"
if str(BIGYM_REPO) not in sys.path:
    sys.path.insert(0, str(BIGYM_REPO))

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.utils.observation_config import ObservationConfig
from demonstrations.demo import Demo
from demonstrations.demo_store import DemoNotFoundError, DemoStore
from demonstrations.utils import Metadata
from bigym.robots.configs.h1 import H1


def detect_floating_dofs_from_demos(env_name: str) -> list[PelvisDof]:
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


def get_environment_class(env_name: str) -> Type:
    env_modules = {
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
        "DishwasherLoadCutlery": ("bigym.envs.dishwasher_cutlery", None),
        "DishwasherLoadPlates": ("bigym.envs.dishwasher_plates", None),
        "DishwasherUnloadCups": ("bigym.envs.dishwasher_cups", None),
        "DishwasherUnloadCupsLong": ("bigym.envs.dishwasher_cups", None),
        "DishwasherUnloadCutlery": ("bigym.envs.dishwasher_cutlery", None),
        "DishwasherUnloadCutleryLong": ("bigym.envs.dishwasher_cutlery", None),
        "DishwasherUnloadPlates": ("bigym.envs.dishwasher_plates", None),
        "DishwasherUnloadPlatesLong": ("bigym.envs.dishwasher_plates", None),
        "FlipCup": ("bigym.envs.manipulation", None),
        "FlipCutlery": ("bigym.envs.manipulation", None),
        "FlipSandwich": ("bigym.envs.manipulation", None),
        "StackBlocks": ("bigym.envs.manipulation", None),
        "ToastSandwich": ("bigym.envs.kitchen", None),
        "RemoveSandwich": ("bigym.envs.kitchen", None),
        "SaucepanToHob": ("bigym.envs.kitchen", None),
        "StoreBox": ("bigym.envs.storage", None),
        "PickBox": ("bigym.envs.storage", None),
        "StoreKitchenware": ("bigym.envs.storage", None),
        "GroceriesStoreLower": ("bigym.envs.storage", None),
        "GroceriesStoreUpper": ("bigym.envs.storage", None),
        "TakeCups": ("bigym.envs.pick_and_place", None),
        "PutCups": ("bigym.envs.storage", None),
        "CupboardsOpenAll": ("bigym.envs.cupboards", None),
        "CupboardsCloseAll": ("bigym.envs.cupboards", None),
        "WallCupboardOpen": ("bigym.envs.cupboards", None),
        "WallCupboardClose": ("bigym.envs.cupboards", None),
        "DrawersAllOpen": ("bigym.envs.drawers", None),
        "DrawersAllClose": ("bigym.envs.drawers", None),
        "DrawerTopOpen": ("bigym.envs.drawers", None),
        "DrawerTopClose": ("bigym.envs.drawers", None),
    }
    if env_name not in env_modules:
        raise ValueError(f"Unknown environment: {env_name}")
    module_name, class_name = env_modules[env_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name or env_name)


def load_step1_source_demos(env_name: str, control_frequency: int) -> tuple[type, list[Demo]]:
    env_class = get_environment_class(env_name)
    preferred_dofs = detect_floating_dofs_from_demos(env_name)
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

    cache_root_env = os.getenv("BIGYM_CACHE_ROOT")
    demo_store = (
        DemoStore(cache_root=Path(cache_root_env).expanduser())
        if cache_root_env
        else DemoStore()
    )
    original_demos = None
    last_error: Exception | None = None
    chosen = None

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
                    original_demos = demo_store.get_demos(
                        metadata, amount=-1, frequency=control_frequency
                    )
                    chosen = (
                        [dof.value for dof in floating_dofs],
                        absolute,
                        robot_cls.__name__,
                    )
                    break
                except DemoNotFoundError as exc:
                    last_error = exc
                    original_demos = None
                finally:
                    env.close()
            if original_demos is not None:
                break
        if original_demos is not None:
            break

    if original_demos is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to load demos for env={env_name}")

    print(
        "Using source lookup config:",
        f"dofs={chosen[0]}",
        f"absolute={chosen[1]}",
        f"robot={chosen[2]}",
        f"count={len(original_demos)}",
    )
    original_demos = sorted(original_demos, key=lambda demo: int(demo.seed))
    return env_class, original_demos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MoveTwoPlates")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--disable-perturb", action="store_true")
    args = parser.parse_args()

    env_class, demos = load_step1_source_demos(args.env, args.control_freq)
    demo0 = demos[0]
    source_env_data = demo0.metadata.environment_data
    floating_dofs = [PelvisDof(dof) for dof in source_env_data.floating_dofs]
    source_absolute = (
        True
        if source_env_data.action_mode_absolute is None
        else bool(source_env_data.action_mode_absolute)
    )
    source_robot_cls = demo0.metadata.robot_cls

    prev_bigym_disable_perturb = os.getenv("BIGYM_DISABLE_PERTURB")
    if args.disable_perturb:
        os.environ["BIGYM_DISABLE_PERTURB"] = "1"
    else:
        os.environ.pop("BIGYM_DISABLE_PERTURB", None)

    env = env_class(
        action_mode=JointPositionActionMode(
            floating_base=source_env_data.floating_base,
            absolute=source_absolute,
            floating_dofs=floating_dofs,
        ),
        control_frequency=args.control_freq,
        observation_config=ObservationConfig(cameras=[]),
        render_mode=None,
        robot_cls=source_robot_cls,
    )

    successes: list[tuple[int, int]] = []
    failures: list[tuple[int, float]] = []
    try:
        for demo in demos:
            env.reset(seed=demo.seed)
            success = False
            max_reward = 0.0
            for step_idx, timestep in enumerate(demo.timesteps):
                action = getattr(timestep, "executed_action", None)
                if action is None:
                    action = timestep.info.get("demo_action")
                if action is None:
                    continue
                action = np.clip(action, env.action_space.low, env.action_space.high)
                _, reward, terminated, truncated, info = env.step(action)
                max_reward = max(max_reward, float(reward))
                if info.get("task_success", False):
                    success = True
                    successes.append((int(demo.seed), step_idx + 1))
                    break
                if terminated or truncated:
                    break
            if not success:
                failures.append((int(demo.seed), max_reward))
    finally:
        env.close()
        if prev_bigym_disable_perturb is None:
            os.environ.pop("BIGYM_DISABLE_PERTURB", None)
        else:
            os.environ["BIGYM_DISABLE_PERTURB"] = prev_bigym_disable_perturb

    mode = "no_perturb" if args.disable_perturb else "default"
    print(f"\nMODE {mode}")
    print(
        f"total={len(demos)} success={len(successes)} "
        f"rate={(len(successes) / len(demos)) if demos else 0.0:.6f}"
    )
    print("success_seeds", [seed for seed, _ in successes])
    print("failed_seeds", [seed for seed, _ in failures])


if __name__ == "__main__":
    main()
