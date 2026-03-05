#!/usr/bin/env python3
"""Step-through viewer for RBY1 Cartesian demos.

Features:
- Step one action at a time (terminal controls).
- Visualize EE targets and achieved EE positions with markers.
- Print per-step tracking error and base-table contacts.

Example:
RBY1_DISABLE_PERTURB=1 BIGYM_DISABLE_PERTURB=1 python scripts/step_through_rby1_cartesian_demo.py  --demo /juno/u/jisangp/dev/dian/rby1_equidiff/data/bigym/rby1_flipcup_source/failure/failed_rby1_cartesian_demo_001.safetensors   --control-frequency 20   --camera third_person   --viewer-backend passive
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

# Add project root to path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bigym.const import HandSide
from bigym.utils.observation_config import ObservationConfig
from demonstrations.demo import Demo

_MARKERS_ENABLED = True
_MARKERS_WARNED = False


_BASE_BODY_KEYWORDS = (
    "rby1/base",
    "base",
    "wheel",
    "chassis",
)
_SURFACE_BODY_KEYWORDS = (
    "table",
    "counter",
    "cabinet",
    "desk",
    "rack",
    "shelf",
)


def _extract_action(timestep) -> Optional[np.ndarray]:
    """Extract action from a demo timestep."""
    action = timestep.info.get("demo_action")
    if action is None:
        action = timestep.executed_action
    if action is None:
        action = timestep.info.get("action")
    if action is None:
        return None
    return np.asarray(action, dtype=np.float64)


def _parse_ee_targets(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parse left/right EE target positions from Cartesian action."""
    if action.shape[0] < 12:
        raise ValueError(f"Expected action dim >= 12, got {action.shape[0]}")
    left_target = action[0:3]
    right_target = action[9:12]
    return left_target, right_target


def _get_ee_positions(env) -> tuple[np.ndarray, np.ndarray]:
    """Read current left/right EE positions from robot wrist sites."""
    left_site = env.robot._wrist_sites[HandSide.LEFT]
    right_site = env.robot._wrist_sites[HandSide.RIGHT]
    return left_site.get_position().copy(), right_site.get_position().copy()


def _safe_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else f"obj_{int(obj_type)}_{obj_id}"


def _build_contact_classifier(env) -> dict[str, set[int]]:
    """Build body-id sets for base/wheel and support-surface classification."""
    model = env._mojo.physics.model._model
    base_body_ids: set[int] = set()
    surface_body_ids: set[int] = set()

    for bid in range(model.nbody):
        name = _safe_name(model, mujoco.mjtObj.mjOBJ_BODY, bid).lower()

        if "base_target" in name:
            continue

        if any(k in name for k in _BASE_BODY_KEYWORDS):
            base_body_ids.add(bid)
        if any(k in name for k in _SURFACE_BODY_KEYWORDS):
            surface_body_ids.add(bid)

    return {
        "base_body_ids": base_body_ids,
        "surface_body_ids": surface_body_ids,
    }


def _find_base_table_contacts(env, classifier: Optional[dict[str, set[int]]] = None) -> list[tuple[str, str, str, str, float]]:
    """Return contacts where one side is base/wheel and the other is support surface."""
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    results: list[tuple[str, str, str, str, float]] = []
    base_ids = set() if classifier is None else classifier.get("base_body_ids", set())
    surface_ids = set() if classifier is None else classifier.get("surface_body_ids", set())

    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        body1 = _safe_name(model, mujoco.mjtObj.mjOBJ_BODY, b1).lower()
        body2 = _safe_name(model, mujoco.mjtObj.mjOBJ_BODY, b2).lower()
        geom1 = _safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        geom2 = _safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)

        cond_ids = ((b1 in base_ids) and (b2 in surface_ids)) or ((b2 in base_ids) and (b1 in surface_ids))

        # String fallback for unexpected naming.
        body1_is_base = any(k in body1 for k in _BASE_BODY_KEYWORDS)
        body2_is_base = any(k in body2 for k in _BASE_BODY_KEYWORDS)
        body1_is_surface = any(k in body1 for k in _SURFACE_BODY_KEYWORDS)
        body2_is_surface = any(k in body2 for k in _SURFACE_BODY_KEYWORDS)
        cond_names = (body1_is_base and body2_is_surface) or (body2_is_base and body1_is_surface)

        # Geom fallback.
        g1l = geom1.lower()
        g2l = geom2.lower()
        g1_is_base = any(k in g1l for k in _BASE_BODY_KEYWORDS)
        g2_is_base = any(k in g2l for k in _BASE_BODY_KEYWORDS)
        g1_is_surface = any(k in g1l for k in _SURFACE_BODY_KEYWORDS)
        g2_is_surface = any(k in g2l for k in _SURFACE_BODY_KEYWORDS)
        cond_geoms = (g1_is_base and g2_is_surface) or (g2_is_base and g1_is_surface)

        if cond_ids or cond_names or cond_geoms:
            results.append((body1, body2, geom1, geom2, float(c.dist)))

    return results


def _get_all_contacts(env, max_count: int = 10) -> list[tuple[str, str, str, str, float]]:
    """Return up to max_count contact pairs for debugging classification misses."""
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    out: list[tuple[str, str, str, str, float]] = []
    for i in range(min(int(data.ncon), max_count)):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        body1 = _safe_name(model, mujoco.mjtObj.mjOBJ_BODY, b1).lower()
        body2 = _safe_name(model, mujoco.mjtObj.mjOBJ_BODY, b2).lower()
        geom1 = _safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        geom2 = _safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        out.append((body1, body2, geom1, geom2, float(c.dist)))
    return out


def _add_sphere_marker(viewer, pos: np.ndarray, rgba: tuple[float, float, float, float], size: float) -> None:
    """Add a sphere marker to the viewer."""
    if not hasattr(viewer, "add_marker"):
        return
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=np.asarray(pos, dtype=float),
        size=np.array([size, size, size], dtype=float),
        rgba=np.array(rgba, dtype=float),
    )


def _patch_marker_compat(viewer) -> None:
    """Patch viewer marker insertion for MuJoCo versions lacking some mjvGeom fields."""
    if getattr(viewer, "_marker_compat_patched", False):
        return
    if not hasattr(viewer, "_add_marker_to_scene"):
        return

    def _add_marker_to_scene_compat(self, marker: dict):
        if self.scn.ngeom >= self.scn.maxgeom:
            return

        g = self.scn.geoms[self.scn.ngeom]

        scalar_defaults = {
            "dataid": -1,
            "objtype": mujoco.mjtObj.mjOBJ_UNKNOWN,
            "objid": -1,
            "category": mujoco.mjtCatBit.mjCAT_DECOR,
            "texid": -1,
            "texuniform": 0,
            "emission": 0,
            "specular": 0.5,
            "shininess": 0.5,
            "reflectance": 0,
            "type": mujoco.mjtGeom.mjGEOM_LABEL,
        }
        for key, value in scalar_defaults.items():
            if hasattr(g, key):
                setattr(g, key, value)

        if hasattr(g, "texrepeat"):
            g.texrepeat[:] = np.array([1, 1], dtype=float)
        if hasattr(g, "size"):
            g.size[:] = np.ones(3) * 0.1
        if hasattr(g, "mat"):
            g.mat[:] = np.eye(3)
        if hasattr(g, "rgba"):
            g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if not hasattr(g, key):
                continue
            if isinstance(value, str):
                if key == "label":
                    setattr(g, key, value)
                continue

            attr = getattr(g, key)
            if np.isscalar(value) or isinstance(value, mujoco.mjtGeom):
                setattr(g, key, value)
                continue
            try:
                arr = np.asarray(value)
                attr[:] = arr.reshape(attr.shape)
            except Exception:
                try:
                    setattr(g, key, value)
                except Exception:
                    continue

        self.scn.ngeom += 1

    viewer._add_marker_to_scene = types.MethodType(_add_marker_to_scene_compat, viewer)
    viewer._marker_compat_patched = True


def _disable_markers(viewer, reason: str) -> None:
    global _MARKERS_ENABLED, _MARKERS_WARNED
    _MARKERS_ENABLED = False
    if hasattr(viewer, "_markers"):
        viewer._markers.clear()
    if not _MARKERS_WARNED:
        print(f"[warn] Marker rendering disabled: {reason}")
        _MARKERS_WARNED = True


def _render_with_markers(
    env,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_achieved: np.ndarray,
    right_achieved: np.ndarray,
    viewer_backend: str = "passive",
    passive_viewer=None,
    enable_markers: bool = True,
) -> None:
    if viewer_backend == "passive":
        if enable_markers:
            _render_passive_with_markers(
                passive_viewer, left_target, right_target, left_achieved, right_achieved
            )
        elif passive_viewer is not None and passive_viewer.is_running():
            with passive_viewer.lock():
                passive_viewer.user_scn.ngeom = 0
            passive_viewer.sync()
        return

    global _MARKERS_ENABLED
    if not enable_markers:
        env.render()
        return
    viewer = env.mujoco_renderer.get_viewer("human")
    if _MARKERS_ENABLED:
        _patch_marker_compat(viewer)
        # Target markers
        _add_sphere_marker(viewer, left_target, (1.0, 0.2, 0.2, 0.9), size=0.03)
        _add_sphere_marker(viewer, right_target, (0.2, 0.4, 1.0, 0.9), size=0.03)
        # Achieved markers
        _add_sphere_marker(viewer, left_achieved, (1.0, 0.8, 0.2, 0.9), size=0.022)
        _add_sphere_marker(viewer, right_achieved, (0.2, 1.0, 0.6, 0.9), size=0.022)

    try:
        env.render()
    except AttributeError as exc:
        if _MARKERS_ENABLED and "MjvGeom" in str(exc):
            _disable_markers(viewer, str(exc))
            env.render()
            return
        raise


def _set_third_person_camera(env) -> None:
    viewer = env.mujoco_renderer.get_viewer("human")
    if hasattr(viewer, "cam"):
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22
        viewer.cam.distance = 2.8
        viewer.cam.lookat[:] = [0.1, 0.0, 0.75]


def _set_third_person_camera_passive(viewer) -> None:
    if viewer is None or not hasattr(viewer, "cam"):
        return
    with viewer.lock():
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22
        viewer.cam.distance = 2.8
        viewer.cam.lookat[:] = [0.1, 0.0, 0.75]


def _configure_collision_view_passive(viewer, mode: str) -> None:
    """Control collision geom visibility in passive viewer."""
    if viewer is None:
        return
    if mode == "off":
        return
    with viewer.lock():
        # In this codebase, visual geoms usually group=2, collision geoms group=3.
        if mode == "only":
            if len(viewer.opt.geomgroup) > 2:
                viewer.opt.geomgroup[2] = 0
            if len(viewer.opt.geomgroup) > 3:
                viewer.opt.geomgroup[3] = 1
        elif mode == "on":
            if len(viewer.opt.geomgroup) > 3:
                viewer.opt.geomgroup[3] = 1
        try:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        except Exception:
            pass


def _configure_collision_view_env_human(env, mode: str) -> None:
    """Control collision geom visibility in env human viewer."""
    if mode == "off":
        return
    viewer = env.mujoco_renderer.get_viewer("human")
    if not hasattr(viewer, "opt"):
        return
    if mode == "only":
        if len(viewer.opt.geomgroup) > 2:
            viewer.opt.geomgroup[2] = 0
        if len(viewer.opt.geomgroup) > 3:
            viewer.opt.geomgroup[3] = 1
    elif mode == "on":
        if len(viewer.opt.geomgroup) > 3:
            viewer.opt.geomgroup[3] = 1
    try:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    except Exception:
        pass


def _add_passive_marker_sphere(scn, pos: np.ndarray, rgba: np.ndarray, size: float) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, size, size], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _render_passive_with_markers(
    viewer,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_achieved: np.ndarray,
    right_achieved: np.ndarray,
) -> None:
    if viewer is None or not viewer.is_running():
        return
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        _add_passive_marker_sphere(
            viewer.user_scn, left_target, np.array([1.0, 0.2, 0.2, 0.9]), size=0.03
        )
        _add_passive_marker_sphere(
            viewer.user_scn, right_target, np.array([0.2, 0.4, 1.0, 0.9]), size=0.03
        )
        _add_passive_marker_sphere(
            viewer.user_scn, left_achieved, np.array([1.0, 0.8, 0.2, 0.9]), size=0.022
        )
        _add_passive_marker_sphere(
            viewer.user_scn, right_achieved, np.array([0.2, 1.0, 0.6, 0.9]), size=0.022
        )
    viewer.sync()


def _build_env_from_demo(demo: Demo, control_frequency: int, render_mode: Optional[str]):
    metadata = demo.metadata
    action_mode = _build_action_mode_from_metadata(
        metadata=metadata, control_frequency=control_frequency
    )
    return metadata.env_cls(
        action_mode=action_mode,
        observation_config=ObservationConfig(cameras=[]),
        render_mode=render_mode,
        control_frequency=control_frequency,
        robot_cls=metadata.robot_cls,
    )


def _build_action_mode_from_metadata(metadata: Any, control_frequency: int):
    """Build action mode from metadata, with fallback for custom modes."""
    try:
        return metadata.get_action_mode()
    except ValueError as exc:
        msg = str(exc)
        if "Invalid action mode name" not in msg:
            raise

        env_data = metadata.environment_data
        action_mode_name = env_data.action_mode_name

        # Fallback modules where custom action modes live in this codebase.
        candidate_modules = [
            "bigym.rby1_cartesian_action_mode_whole_body",
            "bigym.rby1_cartesian_action_mode",
            "bigym.cartesian_action_mode_direct",
            "bigym.cartesian_action_mode",
        ]

        action_mode_cls = None
        for module_name in candidate_modules:
            module = importlib.import_module(module_name)
            cls = getattr(module, action_mode_name, None)
            if cls is not None:
                action_mode_cls = cls
                break

        if action_mode_cls is None:
            raise ValueError(
                f"Invalid action mode name: {action_mode_name} "
                f"(fallback lookup failed in {candidate_modules})"
            ) from exc

        params = inspect.signature(action_mode_cls.__init__).parameters
        kwargs = {}

        if "floating_base" in params:
            kwargs["floating_base"] = env_data.floating_base
        if "floating_dofs" in params and env_data.floating_dofs is not None:
            from bigym.action_modes import PelvisDof

            kwargs["floating_dofs"] = [PelvisDof(d) for d in env_data.floating_dofs]
        if "absolute" in params and env_data.action_mode_absolute is not None:
            kwargs["absolute"] = bool(env_data.action_mode_absolute)
        if "control_frequency" in params:
            kwargs["control_frequency"] = int(control_frequency)
        if "block_until_reached" in params:
            kwargs["block_until_reached"] = False

        print(
            f"[fallback] action mode '{action_mode_name}' loaded from "
            f"{action_mode_cls.__module__}.{action_mode_cls.__name__}"
        )
        return action_mode_cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step through an RBY1 Cartesian demo with EE target markers."
    )
    parser.add_argument(
        "--demo",
        type=Path,
        required=True,
        help="Path to rby1_cartesian_demo_*.safetensors (or failed_*.safetensors).",
    )
    parser.add_argument(
        "--control-frequency",
        type=int,
        default=20,
        help="Control frequency to recreate environment (default: 20).",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Replay up to this index first, then enter interactive stepping.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max number of interactive steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override reset seed. Default uses demo metadata seed.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between steps in auto mode.",
    )
    parser.add_argument(
        "--camera",
        choices=["third_person", "default"],
        default="third_person",
        help="Viewer camera preset.",
    )
    parser.add_argument(
        "--viewer-backend",
        choices=["passive", "env_human"],
        default="passive",
        help=(
            "Viewer backend. 'passive' uses mujoco.viewer.launch_passive (recommended). "
            "'env_human' uses gymnasium human renderer."
        ),
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Disable target/achieved 3D markers (for stability debugging).",
    )
    parser.add_argument(
        "--collision-view",
        choices=["off", "on", "only"],
        default="on",
        help=(
            "Collision mesh visualization: off=default, on=show collisions too, "
            "only=hide visual mesh and show only collision mesh."
        ),
    )
    parser.add_argument(
        "--dump-all-contacts",
        action="store_true",
        help="Print raw contact pairs each step (useful when filtered contact count stays 0).",
    )
    args = parser.parse_args()

    demo = Demo.from_safetensors(args.demo)
    if demo is None:
        raise FileNotFoundError(f"Failed to load demo: {args.demo}")

    total_steps = len(demo.timesteps)
    if total_steps == 0:
        raise RuntimeError("Demo has zero timesteps.")

    render_mode = None if args.viewer_backend == "passive" else "human"
    env = _build_env_from_demo(
        demo, control_frequency=args.control_frequency, render_mode=render_mode
    )
    seed = demo.seed if args.seed is None else int(args.seed)
    env.reset(seed=seed)

    print(f"Loaded demo: {args.demo}")
    print(f"Environment: {demo.metadata.env_name}")
    print(f"Seed: {seed}")
    print(f"Steps: {total_steps}")
    print("Controls: [Enter/n]=next, a=auto, p=pause auto, q=quit")
    print(f"Viewer backend: {args.viewer_backend}")
    print(f"Markers: {'off' if args.no_markers else 'on'}")
    print(f"Collision view: {args.collision_view}")

    env_dim = int(env.action_space.shape[0])
    contact_classifier = _build_contact_classifier(env)
    print(
        "Contact classifier: "
        f"base/wheel bodies={len(contact_classifier['base_body_ids'])}, "
        f"surface bodies={len(contact_classifier['surface_body_ids'])}"
    )

    passive_viewer = None
    viewer_stack = contextlib.ExitStack()
    if args.viewer_backend == "passive":
        import mujoco.viewer

        model = env._mojo.physics.model._model
        data = env._mojo.physics.data._data
        passive_viewer = viewer_stack.enter_context(
            mujoco.viewer.launch_passive(model, data)
        )
        if args.camera == "third_person":
            _set_third_person_camera_passive(passive_viewer)
        _configure_collision_view_passive(passive_viewer, args.collision_view)
        # Show initial frame.
        passive_viewer.sync()
    else:
        env.render()
        if args.camera == "third_person":
            _set_third_person_camera(env)
        _configure_collision_view_env_human(env, args.collision_view)

    # Optional fast-forward before entering interactive mode.
    start_idx = int(np.clip(args.start_step, 0, total_steps - 1))
    if start_idx > 0:
        print(f"Fast-forwarding to step {start_idx} ...")
        for i in range(start_idx):
            action = _extract_action(demo.timesteps[i])
            if action is None:
                continue
            if action.shape[0] != env_dim:
                if action.shape[0] > env_dim:
                    action = action[:env_dim]
                else:
                    action = np.pad(action, (0, env_dim - action.shape[0]), constant_values=0.0)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if args.viewer_backend == "env_human":
            env.render()

    auto = False
    steps_run = 0
    step_idx = start_idx

    try:
        while step_idx < total_steps:
            if args.max_steps is not None and steps_run >= args.max_steps:
                break

            timestep = demo.timesteps[step_idx]
            action = _extract_action(timestep)
            if action is None:
                print(f"[step {step_idx}] action missing, skipping")
                step_idx += 1
                continue

            if action.shape[0] != env_dim:
                print(
                    f"[step {step_idx}] action dim mismatch: demo={action.shape[0]} env={env_dim}. "
                    "Adjusting action size."
                )
                if action.shape[0] > env_dim:
                    action = action[:env_dim]
                else:
                    action = np.pad(action, (0, env_dim - action.shape[0]), constant_values=0.0)

            action = np.clip(action, env.action_space.low, env.action_space.high)
            left_target, right_target = _parse_ee_targets(action)
            left_pre, right_pre = _get_ee_positions(env)

            _render_with_markers(
                env,
                left_target,
                right_target,
                left_pre,
                right_pre,
                viewer_backend=args.viewer_backend,
                passive_viewer=passive_viewer,
                enable_markers=not args.no_markers,
            )

            if not auto:
                cmd = input(f"[step {step_idx}/{total_steps-1}] ").strip().lower()
                if cmd in {"q", "quit"}:
                    break
                if cmd in {"a", "auto"}:
                    auto = True

            _, reward, terminated, truncated, info = env.step(action)
            left_post, right_post = _get_ee_positions(env)
            left_err = np.linalg.norm(left_post - left_target)
            right_err = np.linalg.norm(right_post - right_target)
            contacts = _find_base_table_contacts(env, classifier=contact_classifier)

            print(
                f"step={step_idx:04d} "
                f"L_err={left_err*1000:6.1f}mm "
                f"R_err={right_err*1000:6.1f}mm "
                f"reward={reward:.3f} "
                f"success={bool(info.get('task_success', False))} "
                f"base_table_contacts={len(contacts)}"
            )
            if contacts:
                min_dist = min(c[-1] for c in contacts)
                print(f"  min_contact_dist={min_dist:.5f}m (negative means penetration)")
                for body1, body2, geom1, geom2, dist in contacts[:3]:
                    print(f"  contact: {geom1}({body1}) <-> {geom2}({body2}) dist={dist:.5f}")
            elif args.dump_all_contacts:
                raw = _get_all_contacts(env, max_count=8)
                if raw:
                    print(f"  raw_contacts={len(raw)} (first {len(raw)})")
                    for body1, body2, geom1, geom2, dist in raw:
                        print(
                            f"  raw: {geom1}({body1}) <-> {geom2}({body2}) dist={dist:.5f}"
                        )

            _render_with_markers(
                env,
                left_target,
                right_target,
                left_post,
                right_post,
                viewer_backend=args.viewer_backend,
                passive_viewer=passive_viewer,
                enable_markers=not args.no_markers,
            )

            if terminated or truncated:
                print(f"Episode ended at step {step_idx} (terminated={terminated}, truncated={truncated})")
                break

            step_idx += 1
            steps_run += 1

            if auto and args.sleep > 0.0:
                time.sleep(args.sleep)

            if auto and sys.stdin in select_inputs_ready():
                cmd = sys.stdin.readline().strip().lower()
                if cmd in {"p", "pause"}:
                    auto = False
                elif cmd in {"q", "quit"}:
                    break
            if (
                args.viewer_backend == "passive"
                and passive_viewer is not None
                and not passive_viewer.is_running()
            ):
                print("Viewer closed by user.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        viewer_stack.close()
        print("Closing environment.")
        env.close()


def select_inputs_ready() -> list:
    """Return stdin if available for non-blocking command polling."""
    try:
        import select

        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        return readable
    except Exception:
        return []


if __name__ == "__main__":
    main()
