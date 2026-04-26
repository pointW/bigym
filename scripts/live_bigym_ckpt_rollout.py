#!/usr/bin/env python3
"""Run a BiGym checkpoint rollout with a live viewer."""

from __future__ import annotations

import argparse
import os
import site
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable


def _bootstrap_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rby1_root = repo_root / "rby1_equidiff"
    user_site = ""
    try:
        user_site = site.getusersitepackages()
    except Exception:
        pass

    new_path = []
    for entry in sys.path:
        if user_site and os.path.abspath(entry) == os.path.abspath(user_site):
            continue
        new_path.append(entry)
    sys.path = new_path

    for entry in (str(repo_root), str(rby1_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.chdir(repo_root)


_bootstrap_paths()

import dill
import hydra
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.env.bigym.bigym_image_wrapper import BigymImageWrapper


def parse_bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    if lowered in {"none", "checkpoint", "default"}:
        return None
    raise argparse.ArgumentTypeError(f"Unsupported bool value: {value}")


def resolve_callable(path_or_callable: str | Callable):
    if callable(path_or_callable):
        return path_or_callable
    if not isinstance(path_or_callable, str):
        raise TypeError(f"Unsupported callable spec: {type(path_or_callable)}")
    if ":" in path_or_callable:
        mod_name, fn_name = path_or_callable.split(":", 1)
    else:
        parts = path_or_callable.split(".")
        mod_name, fn_name = ".".join(parts[:-1]), parts[-1]
    module = __import__(mod_name, fromlist=[fn_name])
    return getattr(module, fn_name)


def load_policy_from_checkpoint(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy_cfg = cfg.policy
    policy_target = str(getattr(policy_cfg, "_target_", ""))
    if policy_target == "equi_diffpo.policy.diffusion_unet_pretrained_policy_2view.DiffusionUnetPretrainedPolicy2View":
        policy_cfg = OmegaConf.create(OmegaConf.to_container(policy_cfg, resolve=False))
        policy_cfg._target_ = "equi_diffpo.policy.diffusion_unet_policy_abs.DiffusionUnetPolicyAbs"
        policy_cfg.fa_policy = False
        print(
            "[compat] policy target remap: "
            "DiffusionUnetPretrainedPolicy2View -> DiffusionUnetPolicyAbs(fa_policy=False)"
        )

    policy = hydra.utils.instantiate(policy_cfg)
    state_key = "ema_model" if bool(cfg.training.use_ema) and "ema_model" in payload["state_dicts"] else "model"
    load_result = policy.load_state_dict(payload["state_dicts"][state_key], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            f"[warn] checkpoint={checkpoint_path.name} state_key={state_key} "
            f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
        )
    policy.to(device)
    policy.eval()
    return cfg, policy


def stack_last_n_obs(all_obs: deque[dict[str, np.ndarray]], n_steps: int) -> dict[str, np.ndarray]:
    obs_seq = list(all_obs)
    result = {}
    for key in obs_seq[-1].keys():
        seq = [obs[key] for obs in obs_seq]
        arr = np.zeros((n_steps,) + seq[-1].shape, dtype=seq[-1].dtype)
        start_idx = -min(n_steps, len(seq))
        arr[start_idx:] = np.array(seq[start_idx:])
        if n_steps > len(seq):
            arr[:start_idx] = arr[start_idx]
        result[key] = arr
    return result


def setup_camera(raw_env, render_mode: str, camera: str) -> None:
    if camera == "default":
        return
    _ = raw_env.render()
    viewer = raw_env.mujoco_renderer.get_viewer(render_mode)
    if camera == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        return
    if camera == "external":
        external_id = None
        if hasattr(raw_env, "_cameras_map") and "external" in raw_env._cameras_map:
            external_id = int(raw_env._cameras_map["external"][0])
        if external_id is None:
            print("[warn] external camera id not found, keeping default viewer camera")
            return
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = external_id
        return
    raise ValueError(f"Unsupported camera={camera}")


def summarize_state(raw_env) -> dict[str, Any]:
    info = {}
    if hasattr(raw_env, "get_info"):
        try:
            info = dict(raw_env.get_info())
        except Exception:
            info = {}

    summary = {
        "task_success": float(info.get("task_success", 0.0)),
        "success": bool(getattr(raw_env, "success", False)),
        "fail": bool(getattr(raw_env, "fail", False)),
        "truncate": bool(getattr(raw_env, "truncate", False)),
    }

    if hasattr(raw_env, "cups") and hasattr(raw_env, "cabinets") and hasattr(raw_env, "robot"):
        cup_cabinet = []
        cup_floor = []
        cup_held = []
        for cup in raw_env.cups:
            cup_cabinet.append(any(cup.is_colliding(cabinet) for cabinet in raw_env.cabinets))
            cup_floor.append(bool(cup.is_colliding(raw_env.floor)))
            cup_held.append(any(raw_env.robot.is_gripper_holding_object(cup, side) for side in raw_env.robot.grippers))
        summary.update(
            {
                "cabinet_count": int(sum(cup_cabinet)),
                "floor_count": int(sum(cup_floor)),
                "held_count": int(sum(cup_held)),
                "cup_cabinet": cup_cabinet,
                "cup_floor": cup_floor,
                "cup_held": cup_held,
            }
        )
    return summary


def print_status(prefix: str, primitive_steps: int, max_steps: int, summary: dict[str, Any]) -> None:
    msg = (
        f"{prefix} step={primitive_steps}/{max_steps} "
        f"success={int(summary.get('success', False))} "
        f"fail={int(summary.get('fail', False))} "
        f"truncate={int(summary.get('truncate', False))} "
        f"task_success={summary.get('task_success', 0.0):.1f}"
    )
    if "cabinet_count" in summary:
        msg += (
            f" cabinet={summary['cabinet_count']}"
            f" held={summary['held_count']}"
            f" floor={summary['floor_count']}"
        )
    print(msg, flush=True)


def rollout_live(args) -> None:
    device = torch.device(args.device)
    cfg, policy = load_policy_from_checkpoint(args.checkpoint, device=device)

    shape_meta = cfg.task.shape_meta
    n_obs_steps = int(cfg.n_obs_steps)
    max_steps = int(args.max_steps if args.max_steps is not None else cfg.task.env_runner.max_steps)
    make_env_fn = resolve_callable(cfg.task.env_runner.make_env_fn)
    control_frequency = getattr(cfg.task.env_runner, "control_frequency", None)
    if control_frequency is not None:
        control_frequency = int(control_frequency)
    use_pointcloud_obs = any(k.startswith("pcd_") for k in shape_meta["obs"].keys())
    init_perturb = bool(cfg.task.env_runner.init_perturb)
    if args.init_perturb is not None:
        init_perturb = bool(args.init_perturb)
    render_obs_key = str(getattr(cfg.task.env_runner, "render_obs_key", "head_image"))

    env_kwargs = dict(
        use_pointcloud_obs=use_pointcloud_obs,
        init_perturb=init_perturb,
    )
    if control_frequency is not None:
        env_kwargs["control_frequency"] = control_frequency
    raw_env = make_env_fn(**env_kwargs)
    raw_env.render_mode = args.render_mode
    if args.render_mode in {"human", "rgb_array"}:
        setup_camera(raw_env, args.render_mode, args.camera)

    obs_wrapper = BigymImageWrapper(
        env=raw_env,
        shape_meta=shape_meta,
        render_obs_key=render_obs_key,
    )

    try:
        policy.reset()
        obs_wrapper.seed(args.seed)
        obs = obs_wrapper.reset()
        obs_hist = deque([obs], maxlen=n_obs_steps + 1)

        primitive_steps = 0
        macro_steps = 0
        final_summary = summarize_state(raw_env)
        print_status("[start]", primitive_steps, max_steps, final_summary)

        while primitive_steps < max_steps and not final_summary["success"] and not final_summary["fail"] and not final_summary["truncate"]:
            stacked_obs = stack_last_n_obs(obs_hist, n_obs_steps)
            obs_torch = dict_apply(stacked_obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device))
            with torch.no_grad():
                action_dict = policy.predict_action(obs_torch)
            action_seq = action_dict["action"].detach().to("cpu").numpy()[0]
            if not np.all(np.isfinite(action_seq)):
                raise RuntimeError("Non-finite action encountered.")

            macro_steps += 1
            for act in action_seq:
                if primitive_steps >= max_steps:
                    break
                if final_summary["success"] or final_summary["fail"] or final_summary["truncate"]:
                    break

                step_start = time.perf_counter()
                obs, _, _, _ = obs_wrapper.step(act)
                obs_hist.append(obs)
                primitive_steps += 1
                final_summary = summarize_state(raw_env)

                if args.render_mode in {"human", "rgb_array"} and (primitive_steps % args.render_every == 0):
                    raw_env.render()
                    setup_camera(raw_env, args.render_mode, args.camera)

                if args.status_every > 0 and (
                    primitive_steps % args.status_every == 0
                    or final_summary["success"]
                    or final_summary["fail"]
                    or final_summary["truncate"]
                ):
                    print_status("[live]", primitive_steps, max_steps, final_summary)

                if args.render_mode == "human" and args.realtime:
                    target_dt = (1.0 / control_frequency) / max(args.speed, 1e-6)
                    elapsed = time.perf_counter() - step_start
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)

        timed_out = (
            primitive_steps >= max_steps
            and not final_summary["success"]
            and not final_summary["fail"]
            and not final_summary["truncate"]
        )
        reason = "success" if final_summary["success"] else "fail" if final_summary["fail"] else "truncate" if final_summary["truncate"] else "timeout" if timed_out else "stopped"
        print_status("[done]", primitive_steps, max_steps, final_summary)
        print(
            f"[done] seed={args.seed} macro_steps={macro_steps} primitive_steps={primitive_steps} "
            f"reason={reason}",
            flush=True,
        )

        if args.pause_on_finish and sys.stdin.isatty():
            input("Press Enter to close the viewer...")
        elif args.hold_seconds > 0 and args.render_mode == "human":
            time.sleep(args.hold_seconds)
    finally:
        raw_env.close()


def main():
    parser = argparse.ArgumentParser(description="Run a BiGym checkpoint rollout with a live viewer.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path (.ckpt)")
    parser.add_argument("--seed", type=int, default=100000, help="Deterministic env seed to replay")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=("human", "rgb_array"),
        help="Use human for a live viewer. rgb_array is useful for smoke tests.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="external",
        choices=("external", "free", "default"),
        help="Viewer camera selection for live render.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override rollout max primitive steps")
    parser.add_argument("--render-every", type=int, default=1, help="Render every N primitive steps")
    parser.add_argument("--status-every", type=int, default=25, help="Print status every N primitive steps")
    parser.add_argument("--speed", type=float, default=1.0, help="Realtime speed factor. 1.0 is wall-clock.")
    parser.add_argument("--realtime", action="store_true", help="Sleep between steps to preserve realtime playback")
    parser.add_argument("--pause-on-finish", action="store_true", help="Wait for Enter before closing viewer")
    parser.add_argument("--hold-seconds", type=float, default=1.0, help="Hold the viewer open for N seconds at the end")
    parser.add_argument(
        "--init-perturb",
        type=parse_bool_or_none,
        default=None,
        help="Override checkpoint env perturbation. Use true / false / checkpoint.",
    )
    args = parser.parse_args()
    rollout_live(args)


if __name__ == "__main__":
    main()
