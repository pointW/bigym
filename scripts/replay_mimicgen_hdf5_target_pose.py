#!/usr/bin/env python3
"""
Replay robomimic-style Bigym hdf5 demos by directly stepping recorded absolute
target end-effector poses from datagen_info/target_pose.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for repo_path in (REPO_ROOT / "mimicgen", REPO_ROOT / "bigym"):
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


from robomimic.envs.env_base import EnvType

from mimicgen.env_interfaces.base import make_interface
from mimicgen.env_interfaces.bigym import (
    MG_DishwasherClose,
    MG_DishwasherLoadCups,
    MG_DishwasherLoadPlates,
    MG_DishwasherUnloadCups,
    MG_DishwasherUnloadPlates,
    MG_FlipCup,
    MG_FlipSandwich,
    MG_GroceriesStoreLower,
    MG_MovePlate,
    MG_StoreKitchenware,
)
from mimicgen.envs.bigym.bigym_env import BigymEnvWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to robomimic-style hdf5 demo file")
    parser.add_argument(
        "--demo",
        default="0",
        help="Demo index (e.g., 0) or key (e.g., demo_0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Replay all demos sequentially",
    )
    parser.add_argument(
        "--check_success",
        action="store_true",
        help="Report success/failure after replaying each demo",
    )
    parser.add_argument(
        "--success_any",
        action="store_true",
        help="If set with --check_success, report success if any step is successful",
    )
    parser.add_argument(
        "--clip_action",
        action="store_true",
        help="Clip reconstructed absolute target-pose actions to the env action space",
    )
    parser.add_argument("--render", action="store_true", help="Render with GUI (requires display)")
    parser.add_argument("--start", type=int, default=0, help="Start timestep index")
    parser.add_argument("--end", type=int, default=None, help="End timestep index (exclusive)")
    parser.add_argument("--env_name", default="FlipSandwich", help="Bigym env name")
    parser.add_argument(
        "--action_mode_name",
        default="RBY1CartesianActionModeWholeBody",
        help="Bigym action mode class name",
    )
    parser.add_argument("--robot_name", default="RBY1", help="Bigym robot config class name")
    parser.add_argument("--control_frequency", type=int, default=50)
    parser.add_argument(
        "--interpolation_frequency",
        type=int,
        default=None,
        help="Optional explicit interpolation frequency override for the action mode",
    )
    parser.add_argument(
        "--low_pass_freq_hz",
        type=float,
        default=None,
        help="Optional explicit low-pass cutoff override for the action mode",
    )
    parser.add_argument(
        "--runtime_hold_steps",
        type=int,
        default=None,
        help="Optional explicit post-action physics hold steps override for the action mode",
    )
    parser.add_argument(
        "--reset_mode",
        choices=("auto", "seed_only", "seed_then_reset_to", "reset_to_only"),
        default="auto",
        help=(
            "How to initialize each demo before replay. "
            "'auto' prefers plain seed reset when the stored initial_state already "
            "matches a fresh reset, which avoids clobbering action-mode internal state."
        ),
    )
    return parser.parse_args()


def build_env_meta(args):
    action_mode_kwargs = dict(
        control_frequency=args.control_frequency,
        block_until_reached=False,
        direct_mode=False,
    )
    if args.interpolation_frequency is not None:
        action_mode_kwargs["interpolation_frequency"] = int(args.interpolation_frequency)
    if args.low_pass_freq_hz is not None:
        action_mode_kwargs["low_pass_freq_hz"] = float(args.low_pass_freq_hz)
    if args.runtime_hold_steps is not None:
        action_mode_kwargs["runtime_hold_steps"] = int(args.runtime_hold_steps)
    return dict(
        env_name=args.env_name,
        type=EnvType.GYM_TYPE,
        env_kwargs=dict(
            action_mode_name=args.action_mode_name,
            action_mode_kwargs=action_mode_kwargs,
            control_frequency=args.control_frequency,
            robot_cls=args.robot_name,
            observation_config=dict(
                cameras=[],
                proprioception=True,
                privileged_information=False,
            ),
            render_mode="human" if args.render else None,
        ),
    )


def resolve_demo_key(demo_arg: str, data_group: h5py.Group) -> str:
    if demo_arg.startswith("demo_"):
        if demo_arg not in data_group:
            raise KeyError(f"{demo_arg} not found in dataset")
        return demo_arg
    try:
        index = int(demo_arg)
    except ValueError as exc:
        raise ValueError(f"Invalid demo selector: {demo_arg}") from exc
    key = f"demo_{index}"
    if key not in data_group:
        raise KeyError(f"{key} not found in dataset")
    return key


def make_env_interface(args, demo_group: h5py.Group, env: BigymEnvWrapper):
    if "datagen_info" in demo_group:
        datagen_info_attrs = demo_group["datagen_info"].attrs
        env_interface_name = datagen_info_attrs.get("env_interface_name", None)
        env_interface_type = datagen_info_attrs.get("env_interface_type", None)
        if isinstance(env_interface_name, bytes):
            env_interface_name = env_interface_name.decode("utf-8")
        if isinstance(env_interface_type, bytes):
            env_interface_type = env_interface_type.decode("utf-8")
        if env_interface_name is not None and env_interface_type is not None:
            return make_interface(
                name=str(env_interface_name),
                interface_type=str(env_interface_type),
                env=env.env,
            )

    if args.env_name in {"MovePlate", "MovePlates"}:
        return MG_MovePlate(env=env.env)
    if args.env_name == "MoveTwoPlates":
        from mimicgen.env_interfaces.bigym import MG_MoveTwoPlates

        return MG_MoveTwoPlates(env=env.env)
    if args.env_name == "FlipCup":
        return MG_FlipCup(env=env.env)
    if args.env_name == "FlipSandwich":
        return MG_FlipSandwich(env=env.env)
    if args.env_name == "DishwasherClose":
        return MG_DishwasherClose(env=env.env)
    if args.env_name == "DishwasherLoadCups":
        return MG_DishwasherLoadCups(env=env.env)
    if args.env_name == "DishwasherLoadPlates":
        return MG_DishwasherLoadPlates(env=env.env)
    if args.env_name == "DishwasherUnloadPlates":
        return MG_DishwasherUnloadPlates(env=env.env)
    if args.env_name == "DishwasherUnloadCups":
        return MG_DishwasherUnloadCups(env=env.env)
    if args.env_name == "StoreKitchenware":
        return MG_StoreKitchenware(env=env.env)
    if args.env_name == "GroceriesStoreLower":
        return MG_GroceriesStoreLower(env=env.env)
    raise ValueError(
        "Unable to resolve env interface from datagen_info metadata "
        f"or env_name={args.env_name}"
    )


def parse_demo_seed(demo_group: h5py.Group):
    demo_seed = demo_group.attrs.get("seed", None)
    if isinstance(demo_seed, np.generic):
        demo_seed = demo_seed.item()
    try:
        return int(demo_seed) if demo_seed is not None else None
    except Exception:
        return None


def load_initial_state(demo_group: h5py.Group, states: np.ndarray):
    if "initial_state" in demo_group:
        if "states" in demo_group["initial_state"]:
            return np.array(demo_group["initial_state"]["states"], dtype=np.float32)
        return np.array(demo_group["initial_state"], dtype=np.float32)
    return np.array(states[0], dtype=np.float32)


def initialize_demo_env(
    args,
    env: BigymEnvWrapper,
    demo_seed: int | None,
    initial_state: np.ndarray,
):
    """
    Initialize replay start state.

    For source HDF5s produced by convert_bigym_safetensors_to_hdf5.py, the
    stored initial_state is captured immediately after a fresh env.reset(seed).
    Re-applying that state via reset_to() can skip action-mode reset hooks
    (filtered joint / mocap state), so prefer the plain seed reset whenever the
    saved state already matches the fresh-reset state.
    """
    if args.reset_mode == "reset_to_only":
        env.reset()
        env.reset_to({"states": initial_state})
        return "reset_to_only"

    if demo_seed is None:
        env.reset()
        if args.reset_mode == "seed_only":
            return "reset_without_seed"
        env.reset_to({"states": initial_state})
        return "reset_then_reset_to_without_seed"

    env.reset(seed=demo_seed)
    if args.reset_mode == "seed_only":
        return "seed_only"
    if args.reset_mode == "seed_then_reset_to":
        env.reset_to({"states": initial_state})
        return "seed_then_reset_to"

    fresh_state = np.asarray(env.get_state()["states"], dtype=np.float32)
    if fresh_state.shape == initial_state.shape and np.allclose(
        fresh_state,
        initial_state,
        rtol=1e-6,
        atol=1e-6,
    ):
        return "seed_only(auto)"

    env.reset_to({"states": initial_state})
    return "seed_then_reset_to(auto)"


def compose_action(
    env: BigymEnvWrapper,
    env_interface,
    target_pose: np.ndarray,
    gripper_action: np.ndarray | None,
    recorded_tail: np.ndarray | None,
    clip_action: bool,
):
    absolute_arm_action = np.asarray(
        env_interface.target_pose_to_action(
            target_pose=np.asarray(target_pose, dtype=np.float64),
            relative=False,
        ),
        dtype=np.float64,
    ).reshape(-1)

    tail = None
    if gripper_action is not None:
        tail = np.asarray(gripper_action, dtype=np.float64).reshape(-1)
    elif recorded_tail is not None:
        tail = np.asarray(recorded_tail, dtype=np.float64).reshape(-1)

    if tail is None:
        full_action = absolute_arm_action
    else:
        full_action = np.concatenate([absolute_arm_action, tail], axis=0)

    action_space = env.env.action_space
    expected_dim = int(action_space.shape[0])
    if full_action.size < expected_dim:
        full_action = np.concatenate(
            [full_action, np.zeros(expected_dim - full_action.size, dtype=np.float64)],
            axis=0,
        )
    elif full_action.size > expected_dim:
        full_action = full_action[:expected_dim]

    clipped = False
    if clip_action:
        clipped_action = np.clip(full_action, action_space.low, action_space.high)
        clipped = not np.allclose(clipped_action, full_action)
        full_action = clipped_action
    return full_action, clipped


def main():
    args = parse_args()
    dataset_path = os.path.expanduser(args.dataset)

    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        if args.all:
            demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
            demo_keys = sorted(demo_keys, key=lambda x: int(x.split("_")[1]))
        else:
            demo_keys = [resolve_demo_key(args.demo, data_group)]

        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            states = np.array(demo_group["states"], dtype=np.float32)
            initial_state = load_initial_state(demo_group=demo_group, states=states)
            demo_seed = parse_demo_seed(demo_group)

            if "datagen_info" not in demo_group or "target_pose" not in demo_group["datagen_info"]:
                raise ValueError(f"{demo_key}: missing datagen_info/target_pose")
            target_pose = np.asarray(demo_group["datagen_info"]["target_pose"], dtype=np.float64)

            gripper_action = None
            if "gripper_action" in demo_group["datagen_info"]:
                gripper_action = np.asarray(
                    demo_group["datagen_info"]["gripper_action"], dtype=np.float64
                )

            recorded_actions = None
            if "actions" in demo_group:
                recorded_actions = np.asarray(demo_group["actions"], dtype=np.float64)

            total_steps = int(target_pose.shape[0])
            start = max(0, args.start)
            end = total_steps if args.end is None else min(args.end, total_steps)
            if states.shape[0] < end:
                raise ValueError(f"{demo_key}: states length is shorter than target_pose length")

            env_meta = build_env_meta(args)
            env = BigymEnvWrapper(
                env_meta=env_meta,
                render=args.render,
                render_offscreen=False,
                use_image_obs=False,
            )
            env_interface = make_env_interface(args=args, demo_group=demo_group, env=env)
            reset_strategy = initialize_demo_env(
                args=args,
                env=env,
                demo_seed=demo_seed,
                initial_state=initial_state,
            )

            success_any = False
            final_success = False
            clip_count = 0
            try:
                for t in range(start, end):
                    tail = None
                    if recorded_actions is not None and recorded_actions.shape[1] > 18:
                        tail = recorded_actions[t, 18:]
                    grip = None if gripper_action is None else gripper_action[t]
                    env_action, clipped = compose_action(
                        env=env,
                        env_interface=env_interface,
                        target_pose=target_pose[t],
                        gripper_action=grip,
                        recorded_tail=tail,
                        clip_action=args.clip_action,
                    )
                    clip_count += int(clipped)
                    _, _, done, info = env.step(env_action)
                    step_success = bool(info.get("task_success", False)) or bool(env.env.success)
                    success_any = success_any or step_success
                    if args.render:
                        env.env.render()
                    if done:
                        break

                final_success = bool(env.env.success)
            finally:
                env.close()

            if args.check_success:
                is_success = final_success
                if args.success_any:
                    is_success = is_success or success_any
                print(
                    f"Replayed {demo_key} via target_pose steps [{start}:{end}) "
                    f"from {dataset_path} seed={demo_seed} success={is_success} "
                    f"final_success={final_success} success_any={success_any} clipped={clip_count} "
                    f"reset={reset_strategy}"
                )
            else:
                print(
                    f"Replayed {demo_key} via target_pose steps [{start}:{end}) "
                    f"from {dataset_path} seed={demo_seed} clipped={clip_count} "
                    f"reset={reset_strategy}"
                )


if __name__ == "__main__":
    main()
