from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BIGYM_PATH = Path(__file__).resolve().parents[1]
if str(BIGYM_PATH) not in sys.path:
    sys.path.insert(0, str(BIGYM_PATH))

from demonstrations.demo import Demo

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "open3d is required. Activate the rby1 conda env "
        "(`conda activate rby1`) before running this script."
    ) from exc


DEFAULT_PCD_KEYS = ("pcd_head", "pcd_left_wrist", "pcd_right_wrist")
SEGMENT_COLORS = {
    "interp": np.array([1.0, 0.65, 0.0], dtype=np.float64),  # orange
    "core": np.array([0.20, 0.80, 1.0], dtype=np.float64),    # cyan
    "rest": np.array([0.65, 0.65, 0.65], dtype=np.float64),   # gray
}
LEFT_TRAJ_COLOR = np.array([1.0, 0.25, 0.25], dtype=np.float64)
RIGHT_TRAJ_COLOR = np.array([0.25, 0.45, 1.0], dtype=np.float64)
TARGET_LEFT_TRAJ_COLOR = np.array([0.20, 0.95, 0.20], dtype=np.float64)   # green
TARGET_RIGHT_TRAJ_COLOR = np.array([0.95, 0.20, 0.95], dtype=np.float64)  # magenta


def _load_demo(path: Path) -> Demo:
    demo = Demo.from_safetensors(path)
    if demo is None:
        raise FileNotFoundError(f"Failed to load demo from {path}")
    return demo


def _collect_pcd(step, keys) -> np.ndarray | None:
    pcs = []
    for key in keys:
        arr = step.observation.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        pcs.append(arr)
    if not pcs:
        return None
    return np.concatenate(pcs, axis=0)


def _collect_ee_pos(step, key: str) -> np.ndarray | None:
    arr = step.observation.get(key)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        return None
    return arr


def _pose_from_pos_quat_wxyz(pos: np.ndarray, quat: np.ndarray) -> np.ndarray | None:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if pos.shape[0] != 3 or quat.shape[0] != 4:
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rot(quat)
    T[:3, 3] = pos
    return T


def _quat_wxyz_to_rot(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _collect_base_axes_points(
    step,
    pos_key: str,
    quat_key: str,
    axis_length: float,
) -> np.ndarray | None:
    pos = step.observation.get(pos_key)
    quat = step.observation.get(quat_key)
    if pos is None or quat is None:
        return None
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if pos.shape[0] != 3 or quat.shape[0] != 4:
        return None

    rot = _quat_wxyz_to_rot(quat)
    points = np.zeros((4, 3), dtype=np.float64)
    points[0] = pos
    points[1] = pos + axis_length * rot[:, 0]  # +X
    points[2] = pos + axis_length * rot[:, 1]  # +Y
    points[3] = pos + axis_length * rot[:, 2]  # +Z
    return points


def interpolation_segment_len(num_interp_steps: int, num_fixed_steps: int) -> int:
    need_interp = int(num_interp_steps) > 0
    need_fixed = int(num_fixed_steps) > 0
    length = 0
    if need_interp:
        # add_waypoint_sequence_for_target_pose with interpolation contributes num_interp + 1 actions.
        length += int(num_interp_steps) + 1
    if need_fixed:
        # with interpolation we add exactly num_fixed; without interpolation it is num_fixed + 1.
        length += int(num_fixed_steps) if need_interp else (int(num_fixed_steps) + 1)
    return int(length)


def _parse_demo_index_from_name(path: Path) -> int | None:
    m = re.search(r"demo_(\d+)\.safetensors$", path.name)
    if m is not None:
        return int(m.group(1))
    m = re.search(r"(\d+)\.safetensors$", path.name)
    if m is not None:
        return int(m.group(1))
    return None


def _resolve_demo_key(data_group, demo_index: int) -> str | None:
    idx = int(demo_index)
    direct_candidates = (
        f"demo_{idx}",
        f"demo_{idx:03d}",
        f"demo_{idx:04d}",
    )
    for key in direct_candidates:
        if key in data_group:
            return key
    for key in data_group.keys():
        if not str(key).startswith("demo_"):
            continue
        try:
            kidx = int(str(key).split("_", 1)[1])
        except Exception:
            continue
        if kidx == idx:
            return str(key)
    return None


def _parse_first_subtask_from_config(config_path: Path) -> dict:
    with config_path.open("r") as f:
        cfg = json.load(f)
    task_spec = cfg["task"]["task_spec"]
    source_dataset_path = (
        cfg.get("experiment", {})
        .get("source", {})
        .get("dataset_path", None)
    )
    items = sorted(task_spec.items(), key=lambda kv: kv[0])
    if len(items) == 0:
        return {
            "signal_name": None,
            "num_interp_steps_cfg": 0,
            "num_fixed_steps_cfg": 0,
            "source_dataset_path": source_dataset_path,
        }
    _, first_spec = items[0]
    signal_name = first_spec.get("subtask_term_signal", None)
    return {
        "signal_name": signal_name,
        "num_interp_steps_cfg": int(first_spec.get("num_interpolation_steps", 0) or 0),
        "num_fixed_steps_cfg": int(first_spec.get("num_fixed_steps", 0) or 0),
        "source_dataset_path": source_dataset_path,
    }


def _infer_first_subtask_end_from_hdf5(
    hdf5_path: Path,
    demo_index: int,
    signal_name: str,
) -> int | None:
    if h5py is None:
        print("[WARN] h5py not available; cannot infer subtask end from hdf5.")
        return None
    if not hdf5_path.exists():
        print(f"[WARN] hdf5 not found: {hdf5_path}")
        return None

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            print(f"[WARN] missing 'data' group in {hdf5_path}")
            return None
        demo_key = _resolve_demo_key(f["data"], int(demo_index))
        if demo_key is None:
            print(f"[WARN] demo index '{int(demo_index)}' not found in {hdf5_path}")
            return None
        g = f["data"][demo_key]
        if "datagen_info" not in g:
            print(f"[WARN] datagen_info missing for {demo_key}")
            return None
        dgi = g["datagen_info"]
        if "subtask_term_signals" not in dgi:
            print(f"[WARN] subtask_term_signals missing for {demo_key}")
            return None
        if signal_name not in dgi["subtask_term_signals"]:
            print(f"[WARN] signal '{signal_name}' missing for {demo_key}")
            return None
        sig = np.asarray(dgi["subtask_term_signals"][signal_name], dtype=np.float64).reshape(-1)

    total_steps = int(sig.shape[0])
    if total_steps <= 1:
        return int(total_steps)

    diffs = sig[1:] - sig[:-1]
    nz = np.nonzero(diffs)[0]
    if len(nz) == 0:
        return int(total_steps)
    cand = int(nz[0]) + 1
    end = cand + 1
    return int(max(0, min(end, total_steps)))


def _load_target_pose_positions_from_hdf5(
    hdf5_path: Path,
    demo_index: int,
    target_frame: str = "wrist",
) -> np.ndarray | None:
    if h5py is None:
        return None
    if not hdf5_path.exists():
        return None
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return None
        demo_key = _resolve_demo_key(f["data"], int(demo_index))
        if demo_key is None:
            return None
        g = f["data"][demo_key]
        if "datagen_info" not in g or "target_pose" not in g["datagen_info"]:
            return None
        tp = np.asarray(g["datagen_info"]["target_pose"], dtype=np.float64)  # [T, A, 4, 4]
    if tp.ndim != 4 or tp.shape[-2:] != (4, 4):
        return None
    if tp.shape[1] < 2:
        return None
    if str(target_frame).lower() == "eef":
        return tp[:, :2, :3, 3]  # [T, 2, 3]

    # Convert target eef(pinch) trajectory to wrist trajectory so it matches obs_*_ee_pos frame.
    with h5py.File(hdf5_path, "r") as f:
        g = f["data"][demo_key]
        dgi = g.get("datagen_info", None)
        obs = g.get("obs", None)
        if dgi is None or ("eef_pose" not in dgi) or (obs is None):
            print("[WARN] missing datagen_info/eef_pose or obs. Falling back to target eef frame.")
            return tp[:, :2, :3, 3]
        try:
            eef = np.asarray(dgi["eef_pose"], dtype=np.float64)  # [T, A, 4, 4]
            l_pos = np.asarray(obs["left_ee_pos"], dtype=np.float64)
            r_pos = np.asarray(obs["right_ee_pos"], dtype=np.float64)
            l_quat = np.asarray(obs["left_ee_quat"], dtype=np.float64)
            r_quat = np.asarray(obs["right_ee_quat"], dtype=np.float64)
        except KeyError:
            print("[WARN] missing wrist obs keys. Falling back to target eef frame.")
            return tp[:, :2, :3, 3]

    if eef.ndim != 4 or eef.shape[-2:] != (4, 4) or eef.shape[1] < 2:
        print("[WARN] invalid datagen_info/eef_pose shape. Falling back to target eef frame.")
        return tp[:, :2, :3, 3]

    T = int(
        min(
            tp.shape[0],
            eef.shape[0],
            l_pos.shape[0],
            r_pos.shape[0],
            l_quat.shape[0],
            r_quat.shape[0],
        )
    )
    if T <= 0:
        return None

    tp = tp[:T, :2]
    eef = eef[:T, :2]
    l_pos = l_pos[:T]
    r_pos = r_pos[:T]
    l_quat = l_quat[:T]
    r_quat = r_quat[:T]

    out = np.zeros((T, 2, 3), dtype=np.float64)
    for t in range(T):
        # left arm
        T_w_wrist_obs_l = _pose_from_pos_quat_wxyz(l_pos[t], l_quat[t])
        if T_w_wrist_obs_l is None:
            print("[WARN] invalid left wrist pose in obs. Falling back to target eef frame.")
            return tp[:, :2, :3, 3]
        T_w_eef_cur_l = np.asarray(eef[t, 0], dtype=np.float64)
        T_w_eef_tgt_l = np.asarray(tp[t, 0], dtype=np.float64)
        T_eef_wrist_l = np.linalg.inv(T_w_eef_cur_l) @ T_w_wrist_obs_l
        T_w_wrist_tgt_l = T_w_eef_tgt_l @ T_eef_wrist_l
        out[t, 0] = T_w_wrist_tgt_l[:3, 3]

        # right arm
        T_w_wrist_obs_r = _pose_from_pos_quat_wxyz(r_pos[t], r_quat[t])
        if T_w_wrist_obs_r is None:
            print("[WARN] invalid right wrist pose in obs. Falling back to target eef frame.")
            return tp[:, :2, :3, 3]
        T_w_eef_cur_r = np.asarray(eef[t, 1], dtype=np.float64)
        T_w_eef_tgt_r = np.asarray(tp[t, 1], dtype=np.float64)
        T_eef_wrist_r = np.linalg.inv(T_w_eef_cur_r) @ T_w_wrist_obs_r
        T_w_wrist_tgt_r = T_w_eef_tgt_r @ T_eef_wrist_r
        out[t, 1] = T_w_wrist_tgt_r[:3, 3]
    return out


def _first_subtask_len_from_src_labels(
    hdf5_path: Path,
    demo_index: int,
) -> int | None:
    if h5py is None:
        return None
    if not hdf5_path.exists():
        return None
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return None
        demo_key = _resolve_demo_key(f["data"], int(demo_index))
        if demo_key is None:
            return None
        g = f["data"][demo_key]
        if "src_demo_labels" not in g:
            return None
        labels = np.asarray(g["src_demo_labels"], dtype=np.int64)
    if labels.ndim == 2:
        labels = labels[:, -1]
    elif labels.ndim != 1:
        return None
    if labels.size == 0:
        return None
    nz = np.nonzero(labels[1:] != labels[:-1])[0]
    if nz.size == 0:
        return int(labels.size)
    return int(nz[0] + 1)


def _source_first_subtask_target_len(
    source_hdf5_path: Path,
    src_demo_ind: int,
    signal_name: str | None,
) -> int | None:
    if h5py is None:
        return None
    if not source_hdf5_path.exists():
        return None
    with h5py.File(source_hdf5_path, "r") as f:
        if "data" not in f:
            return None
        demo_key = _resolve_demo_key(f["data"], int(src_demo_ind))
        if demo_key is None:
            return None
        g = f["data"][demo_key]
        total_steps = int(g["actions"].shape[0])
        if signal_name is None:
            return total_steps
        if "datagen_info" not in g or "subtask_term_signals" not in g["datagen_info"]:
            return None
        sig_grp = g["datagen_info"]["subtask_term_signals"]
        if signal_name not in sig_grp:
            return None
        sig = np.asarray(sig_grp[signal_name], dtype=np.float64).reshape(-1)

    if sig.size <= 1:
        return int(sig.size)
    diffs = sig[1:] - sig[:-1]
    nz = np.nonzero(diffs)[0]
    if nz.size == 0:
        return total_steps
    # Same convention as MimicGen source-subtask parsing.
    return int(min(total_steps, int(nz[0]) + 2))


def _adaptive_interp_end_from_hdf5_exact(
    mg_hdf5: Path,
    source_hdf5: Path,
    demo_index: int,
    signal_name: str | None,
) -> int | None:
    if h5py is None:
        return None
    if (not mg_hdf5.exists()) or (not source_hdf5.exists()):
        return None

    with h5py.File(mg_hdf5, "r") as f:
        if "data" not in f:
            return None
        demo_key = _resolve_demo_key(f["data"], int(demo_index))
        if demo_key is None:
            return None
        g = f["data"][demo_key]
        if "src_demo_inds" not in g:
            return None
        src_demo_inds = np.asarray(g["src_demo_inds"], dtype=np.int64).reshape(-1)
        if src_demo_inds.size == 0:
            return None
        src_demo_ind = int(src_demo_inds[0])

    first_subtask_len = _first_subtask_len_from_src_labels(
        hdf5_path=mg_hdf5,
        demo_index=int(demo_index),
    )
    if first_subtask_len is None:
        return None

    source_first_len = _source_first_subtask_target_len(
        source_hdf5_path=source_hdf5,
        src_demo_ind=src_demo_ind,
        signal_name=signal_name,
    )
    if source_first_len is None:
        return None

    interp_len = int(max(0, first_subtask_len - int(source_first_len)))
    if interp_len <= 0:
        return None
    return int(interp_len)


def _adaptive_interp_end_from_hdf5_heuristic(
    mg_hdf5: Path,
    demo_index: int,
    subtask_end_step: int | None,
) -> int | None:
    if h5py is None:
        return None
    if not mg_hdf5.exists():
        return None
    with h5py.File(mg_hdf5, "r") as f:
        if "data" not in f:
            return None
        demo_key = _resolve_demo_key(f["data"], int(demo_index))
        if demo_key is None:
            return None
        g = f["data"][demo_key]
        if "datagen_info" not in g or "target_pose" not in g["datagen_info"]:
            return None
        target_pose = np.asarray(g["datagen_info"]["target_pose"], dtype=np.float64)

    total_steps = int(target_pose.shape[0])
    if subtask_end_step is None:
        subtask_end = total_steps
    else:
        subtask_end = int(max(0, min(int(subtask_end_step), total_steps)))
    if subtask_end < 4:
        return None

    pos = target_pose[:subtask_end, :, :3, 3]  # [T, A, 3]
    deltas = np.linalg.norm(pos[1:] - pos[:-1], axis=-1).mean(axis=-1)  # [T-1]
    if deltas.size < 4:
        return None

    rel_jump = np.abs(deltas[1:] - deltas[:-1]) / np.maximum(deltas[:-1], 1e-6)
    search_n = int(min(rel_jump.size, 120))
    if search_n <= 1:
        return None

    # Find an early sharp change after a stable prefix.
    for j in range(1, search_n):
        if rel_jump[j] < 0.35:
            continue
        prefix = deltas[: j + 1]
        if prefix.size < 3:
            continue
        cv = float(np.std(prefix) / max(float(np.mean(prefix)), 1e-6))
        if cv > 0.20:
            continue
        interp_end = int(j + 2)  # jump at delta j => boundary at pose index j+2
        if subtask_end_step is not None:
            interp_end = min(interp_end, int(subtask_end_step) - 1)
        if interp_end >= 1:
            return int(interp_end)

    # Fallback: strongest early jump.
    j = int(np.argmax(rel_jump[:search_n]))
    interp_end = int(j + 2)
    if subtask_end_step is not None:
        interp_end = min(interp_end, int(subtask_end_step) - 1)
    if interp_end >= 1:
        return int(interp_end)
    return None


def _resolve_segment_bounds(
    demo_path: Path,
    interp_end_step: int | None,
    subtask_end_step: int | None,
    mg_config: Path | None,
    mg_hdf5: Path | None,
    subtask_term_signal: str | None,
    demo_index: int | None,
) -> tuple[int | None, int | None]:
    interp_end = interp_end_step
    subtask_end = subtask_end_step
    signal_name = subtask_term_signal
    source_dataset_path = None
    interp_mode = "manual"

    if mg_config is not None:
        first = _parse_first_subtask_from_config(mg_config)
        cfg_signal = first["signal_name"]
        cfg_num_interp = int(first["num_interp_steps_cfg"])
        cfg_num_fixed = int(first["num_fixed_steps_cfg"])
        source_dataset_path = first["source_dataset_path"]
        if signal_name is None:
            signal_name = cfg_signal
        if interp_end is None and cfg_num_interp >= 0:
            interp_end = interpolation_segment_len(
                num_interp_steps=cfg_num_interp,
                num_fixed_steps=cfg_num_fixed,
            )
            interp_mode = "fixed_from_config"
        elif interp_end is None and cfg_num_interp == -1:
            interp_mode = "adaptive"
        elif interp_end is not None:
            interp_mode = "manual"

    if mg_hdf5 is not None and subtask_end is None:
        if signal_name is None:
            print("[WARN] signal name unknown. pass --subtask-term-signal or --mg-config.")
        else:
            idx = demo_index if demo_index is not None else _parse_demo_index_from_name(demo_path)
            if idx is None:
                print("[WARN] failed to infer demo index from filename. pass --demo-index.")
            else:
                subtask_end = _infer_first_subtask_end_from_hdf5(
                    hdf5_path=mg_hdf5,
                    demo_index=int(idx),
                    signal_name=signal_name,
                )

    if mg_hdf5 is not None and interp_end is None and interp_mode == "adaptive":
        idx = demo_index if demo_index is not None else _parse_demo_index_from_name(demo_path)
        if idx is None:
            print(
                "[WARN] adaptive interpolation boundary needs demo index. "
                "pass --demo-index if filename parsing fails."
            )
        else:
            exact_interp = None
            if source_dataset_path:
                source_path = Path(source_dataset_path).expanduser()
                exact_interp = _adaptive_interp_end_from_hdf5_exact(
                    mg_hdf5=mg_hdf5,
                    source_hdf5=source_path,
                    demo_index=int(idx),
                    signal_name=signal_name,
                )
                if exact_interp is not None:
                    interp_end = int(exact_interp)
                    interp_mode = "adaptive_exact"
            if interp_end is None:
                heur_interp = _adaptive_interp_end_from_hdf5_heuristic(
                    mg_hdf5=mg_hdf5,
                    demo_index=int(idx),
                    subtask_end_step=subtask_end,
                )
                if heur_interp is not None:
                    interp_end = int(heur_interp)
                    interp_mode = "adaptive_heuristic"

    if interp_end is not None and interp_end < 0:
        interp_end = None
    if subtask_end is not None and subtask_end < 0:
        subtask_end = None
    if interp_end is not None and subtask_end is not None and subtask_end < interp_end:
        print(
            "[WARN] subtask_end_step < interp_end_step. "
            "clamping subtask_end_step to interp_end_step."
        )
        subtask_end = int(interp_end)

    print(
        f"[segment] interp_end_step={interp_end} "
        f"subtask_end_step={subtask_end} signal={signal_name} mode={interp_mode}"
    )
    return interp_end, subtask_end


def _segment_name(step_index: int, interp_end: int | None, subtask_end: int | None) -> str:
    if interp_end is not None and step_index < int(interp_end):
        return "interp"
    if subtask_end is not None and step_index < int(subtask_end):
        return "core"
    if subtask_end is not None:
        return "rest"
    return "traj"


def _traj_color(side: str, segment: str) -> np.ndarray:
    if segment in SEGMENT_COLORS:
        base = SEGMENT_COLORS[segment].copy()
        if side == "right":
            # Slightly darker for right-arm trajectory to disambiguate overlap.
            return np.clip(base * 0.8, 0.0, 1.0)
        return base
    return RIGHT_TRAJ_COLOR if side == "right" else LEFT_TRAJ_COLOR


def _build_traj_lines(
    points: list[np.ndarray],
    step_ids: list[int],
    side: str,
    interp_end_step: int | None,
    subtask_end_step: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
        )
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return (
            pts,
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
        )
    lines = np.column_stack(
        [np.arange(0, pts.shape[0] - 1), np.arange(1, pts.shape[0])]
    ).astype(np.int32)
    colors = np.zeros((lines.shape[0], 3), dtype=np.float64)
    for i in range(1, pts.shape[0]):
        seg_name = _segment_name(
            step_index=int(step_ids[i]),
            interp_end=interp_end_step,
            subtask_end=subtask_end_step,
        )
        colors[i - 1] = _traj_color(side=side, segment=seg_name)
    return pts, lines, colors


def _build_traj_lines_constant_color(
    points: list[np.ndarray],
    color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
        )
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return (
            pts,
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
        )
    lines = np.column_stack(
        [np.arange(0, pts.shape[0] - 1), np.arange(1, pts.shape[0])]
    ).astype(np.int32)
    colors = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (lines.shape[0], 1))
    return pts, lines, colors


def replay(
    demo: Demo,
    keys: tuple[str, ...],
    fps: float,
    stride: int,
    max_steps: int | None,
    save_ply: Path | None,
    pause_first: bool,
    show_base_axes: bool,
    base_pos_key: str,
    base_quat_key: str,
    base_axis_length: float,
    show_ee_traj: bool,
    left_ee_key: str,
    right_ee_key: str,
    interp_end_step: int | None,
    subtask_end_step: int | None,
    target_pos_seq: np.ndarray | None,
):
    if fps <= 0:
        dt = 0.0
    else:
        dt = 1.0 / fps

    paused = pause_first
    step_once_forward = False
    step_once_backward = False

    def _toggle_pause(vis):
        nonlocal paused
        paused = not paused
        return False

    def _step_forward(vis):
        nonlocal step_once_forward, step_once_backward, paused
        step_once_forward = True
        step_once_backward = False
        paused = True
        return False

    def _step_backward(vis):
        nonlocal step_once_forward, step_once_backward, paused
        step_once_backward = True
        step_once_forward = False
        paused = True
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PointCloud Replay", width=960, height=720)
    vis.register_key_callback(ord(" "), _toggle_pause)
    vis.register_key_callback(ord("N"), _step_forward)
    vis.register_key_callback(ord("B"), _step_backward)
    # GLFW key codes for arrow keys in Open3D callbacks.
    vis.register_key_callback(262, _step_forward)  # Right Arrow
    vis.register_key_callback(263, _step_backward)  # Left Arrow
    vis.register_key_callback(ord("."), _step_forward)
    vis.register_key_callback(ord(","), _step_backward)
    pcd_geom = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_geom)
    print("[keys] Space: pause/resume | N or Right or . : next | B or Left or , : prev")

    base_axes_geom = None
    if show_base_axes:
        base_axes_geom = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.zeros((4, 3), dtype=np.float64)),
            lines=o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)),
        )
        base_axes_geom.colors = o3d.utility.Vector3dVector(
            np.array(
                [
                    [1.0, 0.0, 0.0],  # X: red
                    [0.0, 1.0, 0.0],  # Y: green
                    [0.0, 0.0, 1.0],  # Z: blue
                ],
                dtype=np.float64,
            )
        )
        vis.add_geometry(base_axes_geom)

    left_traj_geom = None
    right_traj_geom = None
    left_cur_geom = None
    right_cur_geom = None
    if show_ee_traj:
        left_traj_geom = o3d.geometry.LineSet()
        right_traj_geom = o3d.geometry.LineSet()
        left_cur_geom = o3d.geometry.PointCloud()
        right_cur_geom = o3d.geometry.PointCloud()
        vis.add_geometry(left_traj_geom)
        vis.add_geometry(right_traj_geom)
        vis.add_geometry(left_cur_geom)
        vis.add_geometry(right_cur_geom)

    left_target_traj_geom = None
    right_target_traj_geom = None
    left_target_cur_geom = None
    right_target_cur_geom = None
    if target_pos_seq is not None:
        left_target_traj_geom = o3d.geometry.LineSet()
        right_target_traj_geom = o3d.geometry.LineSet()
        left_target_cur_geom = o3d.geometry.PointCloud()
        right_target_cur_geom = o3d.geometry.PointCloud()
        vis.add_geometry(left_target_traj_geom)
        vis.add_geometry(right_target_traj_geom)
        vis.add_geometry(left_target_cur_geom)
        vis.add_geometry(right_target_cur_geom)

    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([0.02, 0.02, 0.02])

    try:
        view_initialized = False
        warned_missing_base_keys = False
        warned_missing_left_key = False
        warned_missing_right_key = False
        stride_eff = max(1, stride)
        frame_step_ids = list(range(0, len(demo.timesteps), stride_eff))
        if max_steps is not None:
            frame_step_ids = frame_step_ids[: max(0, int(max_steps))]
        if len(frame_step_ids) == 0:
            print("[WARN] no frames to replay after stride/max-steps filtering.")
            return

        left_pos_seq: list[np.ndarray | None] = []
        right_pos_seq: list[np.ndarray | None] = []
        if show_ee_traj:
            for step_id in frame_step_ids:
                step = demo.timesteps[step_id]
                l = _collect_ee_pos(step, left_ee_key)
                r = _collect_ee_pos(step, right_ee_key)
                left_pos_seq.append(l)
                right_pos_seq.append(r)
                if l is None and not warned_missing_left_key:
                    print(f"[WARN] missing/invalid left EE key: '{left_ee_key}'")
                    warned_missing_left_key = True
                if r is None and not warned_missing_right_key:
                    print(f"[WARN] missing/invalid right EE key: '{right_ee_key}'")
                    warned_missing_right_key = True

        target_left_seq: list[np.ndarray | None] = []
        target_right_seq: list[np.ndarray | None] = []
        if target_pos_seq is not None:
            max_t = int(target_pos_seq.shape[0])
            for step_id in frame_step_ids:
                if 0 <= int(step_id) < max_t:
                    target_left_seq.append(np.asarray(target_pos_seq[step_id, 0], dtype=np.float64))
                    target_right_seq.append(np.asarray(target_pos_seq[step_id, 1], dtype=np.float64))
                else:
                    target_left_seq.append(None)
                    target_right_seq.append(None)

        write_idx = 0
        cursor = 0
        n_frames = len(frame_step_ids)

        while 0 <= cursor < n_frames:
            has_any_geometry = False
            step_id = frame_step_ids[cursor]
            step = demo.timesteps[step_id]

            pc = _collect_pcd(step, keys)
            if pc is not None:
                pcd_geom.points = o3d.utility.Vector3dVector(pc[:, :3])
                if pc.shape[1] >= 6:
                    colors = np.clip(pc[:, 3:6], 0.0, 1.0)
                    pcd_geom.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(pcd_geom)
                has_any_geometry = True

            if show_ee_traj:
                left_points = [p for p in left_pos_seq[: cursor + 1] if p is not None]
                right_points = [p for p in right_pos_seq[: cursor + 1] if p is not None]
                left_steps = [
                    int(frame_step_ids[i])
                    for i, p in enumerate(left_pos_seq[: cursor + 1])
                    if p is not None
                ]
                right_steps = [
                    int(frame_step_ids[i])
                    for i, p in enumerate(right_pos_seq[: cursor + 1])
                    if p is not None
                ]

                l_pts, l_lines, l_cols = _build_traj_lines(
                    points=left_points,
                    step_ids=left_steps,
                    side="left",
                    interp_end_step=interp_end_step,
                    subtask_end_step=subtask_end_step,
                )
                left_traj_geom.points = o3d.utility.Vector3dVector(l_pts)
                left_traj_geom.lines = o3d.utility.Vector2iVector(l_lines)
                left_traj_geom.colors = o3d.utility.Vector3dVector(l_cols)
                vis.update_geometry(left_traj_geom)
                if len(left_points) > 0:
                    left_cur_geom.points = o3d.utility.Vector3dVector(
                        np.asarray(left_points[-1], dtype=np.float64).reshape(1, 3)
                    )
                    left_cur_geom.colors = o3d.utility.Vector3dVector(
                        np.asarray([LEFT_TRAJ_COLOR], dtype=np.float64)
                    )
                    vis.update_geometry(left_cur_geom)
                    has_any_geometry = True
                else:
                    left_cur_geom.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
                    vis.update_geometry(left_cur_geom)

                r_pts, r_lines, r_cols = _build_traj_lines(
                    points=right_points,
                    step_ids=right_steps,
                    side="right",
                    interp_end_step=interp_end_step,
                    subtask_end_step=subtask_end_step,
                )
                right_traj_geom.points = o3d.utility.Vector3dVector(r_pts)
                right_traj_geom.lines = o3d.utility.Vector2iVector(r_lines)
                right_traj_geom.colors = o3d.utility.Vector3dVector(r_cols)
                vis.update_geometry(right_traj_geom)
                if len(right_points) > 0:
                    right_cur_geom.points = o3d.utility.Vector3dVector(
                        np.asarray(right_points[-1], dtype=np.float64).reshape(1, 3)
                    )
                    right_cur_geom.colors = o3d.utility.Vector3dVector(
                        np.asarray([RIGHT_TRAJ_COLOR], dtype=np.float64)
                    )
                    vis.update_geometry(right_cur_geom)
                    has_any_geometry = True
                else:
                    right_cur_geom.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
                    vis.update_geometry(right_cur_geom)

            if target_pos_seq is not None:
                target_left_points = [p for p in target_left_seq[: cursor + 1] if p is not None]
                target_right_points = [p for p in target_right_seq[: cursor + 1] if p is not None]

                tl_pts, tl_lines, tl_cols = _build_traj_lines_constant_color(
                    points=target_left_points,
                    color=TARGET_LEFT_TRAJ_COLOR,
                )
                left_target_traj_geom.points = o3d.utility.Vector3dVector(tl_pts)
                left_target_traj_geom.lines = o3d.utility.Vector2iVector(tl_lines)
                left_target_traj_geom.colors = o3d.utility.Vector3dVector(tl_cols)
                vis.update_geometry(left_target_traj_geom)
                if len(target_left_points) > 0:
                    left_target_cur_geom.points = o3d.utility.Vector3dVector(
                        np.asarray(target_left_points[-1], dtype=np.float64).reshape(1, 3)
                    )
                    left_target_cur_geom.colors = o3d.utility.Vector3dVector(
                        np.asarray([TARGET_LEFT_TRAJ_COLOR], dtype=np.float64)
                    )
                    vis.update_geometry(left_target_cur_geom)
                    has_any_geometry = True
                else:
                    left_target_cur_geom.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
                    vis.update_geometry(left_target_cur_geom)

                tr_pts, tr_lines, tr_cols = _build_traj_lines_constant_color(
                    points=target_right_points,
                    color=TARGET_RIGHT_TRAJ_COLOR,
                )
                right_target_traj_geom.points = o3d.utility.Vector3dVector(tr_pts)
                right_target_traj_geom.lines = o3d.utility.Vector2iVector(tr_lines)
                right_target_traj_geom.colors = o3d.utility.Vector3dVector(tr_cols)
                vis.update_geometry(right_target_traj_geom)
                if len(target_right_points) > 0:
                    right_target_cur_geom.points = o3d.utility.Vector3dVector(
                        np.asarray(target_right_points[-1], dtype=np.float64).reshape(1, 3)
                    )
                    right_target_cur_geom.colors = o3d.utility.Vector3dVector(
                        np.asarray([TARGET_RIGHT_TRAJ_COLOR], dtype=np.float64)
                    )
                    vis.update_geometry(right_target_cur_geom)
                    has_any_geometry = True
                else:
                    right_target_cur_geom.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
                    vis.update_geometry(right_target_cur_geom)

            if base_axes_geom is not None:
                base_points = _collect_base_axes_points(
                    step=step,
                    pos_key=base_pos_key,
                    quat_key=base_quat_key,
                    axis_length=base_axis_length,
                )
                if base_points is not None:
                    base_axes_geom.points = o3d.utility.Vector3dVector(base_points)
                    vis.update_geometry(base_axes_geom)
                elif not warned_missing_base_keys:
                    print(
                        f"[WARN] Could not find valid base pose keys "
                        f"('{base_pos_key}', '{base_quat_key}') in observations."
                    )
                    warned_missing_base_keys = True

            if not has_any_geometry:
                if cursor >= n_frames - 1:
                    break
                cursor += 1
                continue

            if not view_initialized:
                vis.reset_view_point(True)
                view_initialized = True
            vis.poll_events()
            vis.update_renderer()

            if save_ply is not None:
                if save_ply.is_dir():
                    out_path = save_ply / f"frame_{write_idx:04d}.ply"
                else:
                    out_path = save_ply
                o3d.io.write_point_cloud(str(out_path), pcd_geom, write_ascii=False)
                write_idx += 1

            if paused:
                while paused and (not step_once_forward) and (not step_once_backward):
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.03)
                if step_once_forward:
                    step_once_forward = False
                    cursor = min(n_frames - 1, cursor + 1)
                    continue
                if step_once_backward:
                    step_once_backward = False
                    cursor = max(0, cursor - 1)
                    continue

            if step_once_forward:
                step_once_forward = False
                paused = True
                cursor = min(n_frames - 1, cursor + 1)
                continue
            if step_once_backward:
                step_once_backward = False
                paused = True
                cursor = max(0, cursor - 1)
                continue

            if dt > 0:
                time.sleep(dt)
            if cursor >= n_frames - 1:
                break
            cursor += 1
    finally:
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Replay a Bigym safetensors demo and visualize point clouds."
    )
    parser.add_argument("--demo", type=str, required=True, help="Path to .safetensors demo")
    parser.add_argument(
        "--keys",
        type=str,
        default=",".join(DEFAULT_PCD_KEYS),
        help="Comma-separated list of point cloud keys to visualize",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Playback fps")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit frames")
    parser.add_argument(
        "--pause-first",
        action="store_true",
        help="Pause after first frame (Space toggle, N next, B previous)",
    )
    parser.add_argument(
        "--save-ply",
        type=str,
        default=None,
        help="Optional output .ply file or directory for saving frames",
    )
    parser.add_argument(
        "--hide-base-axes",
        action="store_true",
        help="Disable base pose axis visualization",
    )
    parser.add_argument(
        "--base-pos-key",
        type=str,
        default="base_pos",
        help="Observation key for base position",
    )
    parser.add_argument(
        "--base-quat-key",
        type=str,
        default="base_quat",
        help="Observation key for base quaternion (wxyz)",
    )
    parser.add_argument(
        "--base-axis-length",
        type=float,
        default=0.15,
        help="Axis length in meters for base pose visualization",
    )
    parser.add_argument(
        "--show-ee-traj",
        action="store_true",
        help="Overlay left/right EE trajectories as line strips.",
    )
    parser.add_argument(
        "--left-ee-key",
        type=str,
        default="left_ee_pos",
        help="Observation key for left EE position (xyz).",
    )
    parser.add_argument(
        "--right-ee-key",
        type=str,
        default="right_ee_pos",
        help="Observation key for right EE position (xyz).",
    )
    parser.add_argument(
        "--interp-end-step",
        type=int,
        default=None,
        help="Exclusive end step for interpolation segment color.",
    )
    parser.add_argument(
        "--subtask-end-step",
        type=int,
        default=None,
        help="Exclusive end step for first-subtask core segment color.",
    )
    parser.add_argument(
        "--mg-config",
        type=str,
        default=None,
        help="Optional MimicGen config json; used to infer interpolation length.",
    )
    parser.add_argument(
        "--mg-hdf5",
        type=str,
        default=None,
        help="Optional MimicGen combined demo.hdf5; used to infer first-subtask end.",
    )
    parser.add_argument(
        "--subtask-term-signal",
        type=str,
        default=None,
        help="Override subtask term signal name for hdf5 boundary inference.",
    )
    parser.add_argument(
        "--demo-index",
        type=int,
        default=None,
        help="Override demo index for hdf5 lookup (demo_<index>).",
    )
    parser.add_argument(
        "--show-target-traj",
        action="store_true",
        help="Overlay target EE trajectory from MimicGen hdf5 datagen_info/target_pose.",
    )
    parser.add_argument(
        "--target-frame",
        type=str,
        choices=("wrist", "eef"),
        default="wrist",
        help="Frame for target trajectory overlay. 'wrist' matches obs_*_ee_pos; 'eef' uses raw target_pose.",
    )
    args = parser.parse_args()

    demo_path = Path(args.demo).expanduser()
    demo = _load_demo(demo_path)
    keys = tuple(k.strip() for k in args.keys.split(",") if k.strip())
    save_ply = Path(args.save_ply).expanduser() if args.save_ply else None
    mg_config = Path(args.mg_config).expanduser() if args.mg_config else None
    mg_hdf5 = Path(args.mg_hdf5).expanduser() if args.mg_hdf5 else None
    interp_end_step, subtask_end_step = _resolve_segment_bounds(
        demo_path=demo_path,
        interp_end_step=args.interp_end_step,
        subtask_end_step=args.subtask_end_step,
        mg_config=mg_config,
        mg_hdf5=mg_hdf5,
        subtask_term_signal=args.subtask_term_signal,
        demo_index=args.demo_index,
    )
    target_pos_seq = None
    if args.show_target_traj:
        if mg_hdf5 is None:
            print("[WARN] --show-target-traj needs --mg-hdf5. Skipping target overlay.")
        else:
            idx = args.demo_index if args.demo_index is not None else _parse_demo_index_from_name(demo_path)
            if idx is None:
                print("[WARN] target overlay needs demo index. pass --demo-index.")
            else:
                target_pos_seq = _load_target_pose_positions_from_hdf5(
                    hdf5_path=mg_hdf5,
                    demo_index=int(idx),
                    target_frame=args.target_frame,
                )
                if target_pos_seq is None:
                    print("[WARN] failed to load target_pose for target overlay.")
                else:
                    print(
                        f"[target] loaded target_pose for demo_{int(idx)} "
                        f"with {int(target_pos_seq.shape[0])} steps "
                        f"(frame={args.target_frame})"
                    )
    replay(
        demo=demo,
        keys=keys or DEFAULT_PCD_KEYS,
        fps=args.fps,
        stride=args.stride,
        max_steps=args.max_steps,
        save_ply=save_ply,
        pause_first=args.pause_first,
        show_base_axes=not args.hide_base_axes,
        base_pos_key=args.base_pos_key,
        base_quat_key=args.base_quat_key,
        base_axis_length=args.base_axis_length,
        show_ee_traj=args.show_ee_traj,
        left_ee_key=args.left_ee_key,
        right_ee_key=args.right_ee_key,
        interp_end_step=interp_end_step,
        subtask_end_step=subtask_end_step,
        target_pos_seq=target_pos_seq,
    )


if __name__ == "__main__":
    main()
