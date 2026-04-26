#!/usr/bin/env python3
"""Dump and compare exact RBY1 FlipCup replay traces for parity checks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REQUIRED_KEYS = (
    "demo_action",
    "limb_ctrl",
    "gripper_ctrl",
    "mocap_pos",
    "mocap_quat",
)

DEFAULT_DEMO = Path(
    "rby1_equidiff/data/bigym/rby1_flipcup_20Hz_noperturb_source/rby1_cartesian_demo_000.safetensors"
)


def _ensure_npz_path(path: Path) -> Path:
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def _action_from_timestep(timestep) -> np.ndarray:
    action = timestep.info.get("demo_action")
    if action is None:
        action = timestep.executed_action
    if action is None:
        action = getattr(timestep, "action", None)
    if action is None:
        raise ValueError("Demo timestep does not contain a replayable action")
    return np.asarray(action, dtype=np.float64)


def _scalar_ctrl(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _make_env():
    from bigym.envs.manipulation import FlipCup
    from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
    from bigym.robots.configs.rby1 import RBY1

    return FlipCup(
        action_mode=RBY1CartesianActionModeWholeBody(
            direct_mode=False,
            block_until_reached=False,
            control_frequency=20,
        ),
        control_frequency=20,
        render_mode=None,
        robot_cls=RBY1,
    )


def dump_trace(demo_path: Path, output_path: Path, max_steps: int | None) -> int:
    import mujoco
    from bigym.const import HandSide
    from demonstrations.demo import Demo

    demo = Demo.from_safetensors(demo_path)
    if demo is None:
        raise FileNotFoundError(f"Demo not found: {demo_path}")

    env = _make_env()
    try:
        env.reset(seed=demo.seed)
        physics = env.robot._mojo.physics
        model = physics.model._model
        data = physics.data._data

        base_target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_target")
        if base_target_body_id < 0:
            raise RuntimeError("base_target mocap body not found in the model")
        mocap_id = model.body_mocapid[base_target_body_id]
        if mocap_id < 0:
            raise RuntimeError("base_target body does not own a mocap id")

        traces = {key: [] for key in REQUIRED_KEYS}
        num_steps = len(demo.timesteps) if max_steps is None else min(max_steps, len(demo.timesteps))

        for step_idx in range(num_steps):
            action = _action_from_timestep(demo.timesteps[step_idx])
            _, _, terminated, truncated, _ = env.step(action)

            limb_ctrl = np.asarray(
                [_scalar_ctrl(physics.bind(actuator).ctrl) for actuator in env.robot.limb_actuators],
                dtype=np.float64,
            )
            gripper_ctrl = np.asarray(
                [
                    _scalar_ctrl(physics.bind(env.robot._grippers[side]._actuators[0]).ctrl)
                    for side in (HandSide.LEFT, HandSide.RIGHT)
                ],
                dtype=np.float64,
            )

            traces["demo_action"].append(action.copy())
            traces["limb_ctrl"].append(limb_ctrl)
            traces["gripper_ctrl"].append(gripper_ctrl)
            traces["mocap_pos"].append(np.array(data.mocap_pos[mocap_id], dtype=np.float64))
            traces["mocap_quat"].append(np.array(data.mocap_quat[mocap_id], dtype=np.float64))

            if terminated or truncated:
                break
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        **{key: np.asarray(values, dtype=np.float64) for key, values in traces.items()},
    )
    print(f"wrote {output_path}")
    for key in REQUIRED_KEYS:
        print(f"{key}: shape={np.asarray(traces[key]).shape}")
    return 0


def _first_mismatch(reference: np.ndarray, candidate: np.ndarray) -> str:
    if reference.shape != candidate.shape:
        return f"shape mismatch {reference.shape} != {candidate.shape}"
    mismatch = np.argwhere(reference != candidate)
    if mismatch.size == 0:
        return "values differ, but no index was identified"
    index = tuple(int(i) for i in mismatch[0])
    return (
        f"first mismatch at {index}: "
        f"reference={reference[index]!r}, candidate={candidate[index]!r}"
    )


def compare_traces(reference_path: Path, candidate_path: Path) -> int:
    reference = np.load(reference_path)
    candidate = np.load(candidate_path)

    ok = True
    for key in REQUIRED_KEYS:
        if key not in reference or key not in candidate:
            print(f"{key}: missing from reference or candidate")
            ok = False
            continue

        ref_arr = np.asarray(reference[key])
        cand_arr = np.asarray(candidate[key])
        equal = np.array_equal(ref_arr, cand_arr)
        print(f"{key}: array_equal={equal} shape_ref={ref_arr.shape} shape_cand={cand_arr.shape}")
        if not equal:
            print(f"  {_first_mismatch(ref_arr, cand_arr)}")
            ok = False

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser("dump", help="Replay a FlipCup demo and save the exact trace to .npz")
    dump_parser.add_argument("--demo", type=Path, default=DEFAULT_DEMO, help="Path to the source safetensors demo")
    dump_parser.add_argument("--output", type=Path, required=True, help="Trace output path (.npz appended if omitted)")
    dump_parser.add_argument("--max-steps", type=int, default=None, help="Optional max number of demo steps to replay")

    compare_parser = subparsers.add_parser("compare", help="Compare two saved traces with exact equality")
    compare_parser.add_argument("--reference", type=Path, required=True, help="Reference .npz trace")
    compare_parser.add_argument("--candidate", type=Path, required=True, help="Candidate .npz trace")

    args = parser.parse_args()

    if args.command == "dump":
        return dump_trace(args.demo, _ensure_npz_path(args.output), args.max_steps)
    if args.command == "compare":
        return compare_traces(_ensure_npz_path(args.reference), _ensure_npz_path(args.candidate))
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
