#!/usr/bin/env python3
"""Run smoke + full conversion for all unconverted BiGym tasks."""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigym.const import CACHE_PATH, DEMO_VERSION
from demonstrations.demo_store import DemoStore


TASK_PATTERN = re.compile(r"\|\s+\[([A-Za-z0-9_]+)\]\(bigym/envs/[^)]+\)\s+\|")
SOURCE_DIR_PATTERN = re.compile(r"rby1_(.+)_source$")


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    convert_script: Path
    readme: Path
    output_root: Path
    smoke_root: Path
    python_bin: Path
    cache_root: Path


def _normalize_task_name(task: str) -> str:
    return task.lower()


def _task_output_dir(root: Path, task: str) -> Path:
    return root / f"rby1_{_normalize_task_name(task)}_source"


def _parse_readme_tasks(readme_path: Path) -> List[str]:
    text = readme_path.read_text(encoding="utf-8")
    tasks: List[str] = []
    seen = set()
    for task in TASK_PATTERN.findall(text):
        if task not in seen:
            tasks.append(task)
            seen.add(task)
    if not tasks:
        raise RuntimeError(f"No BiGym tasks found in {readme_path}")
    return tasks


def _split_task_args(raw_tasks: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in raw_tasks:
        for token in raw.split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _resolve_requested_tasks(
    all_tasks: Sequence[str], requested: Sequence[str]
) -> Tuple[List[str], List[str]]:
    if not requested:
        return list(all_tasks), []

    task_lookup: Dict[str, str] = {}
    for task in all_tasks:
        task_lookup[task.lower()] = task
        task_lookup[_normalize_task_name(task)] = task

    selected: List[str] = []
    selected_set = set()
    unknown: List[str] = []
    for token in requested:
        key = token.lower()
        task = task_lookup.get(key)
        if task is None:
            unknown.append(token)
            continue
        if task not in selected_set:
            selected.append(task)
            selected_set.add(task)
    return selected, unknown


def _find_existing_source_tasks(output_root: Path) -> set[str]:
    existing = set()
    if not output_root.exists():
        return existing
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        match = SOURCE_DIR_PATTERN.fullmatch(child.name)
        if match:
            existing.add(match.group(1))
    return existing


def _cache_version_root(cache_root: Path) -> Path:
    return cache_root / "demonstrations" / DEMO_VERSION


def _missing_local_cache_tasks(cache_root: Path, tasks: Iterable[str]) -> List[str]:
    version_root = _cache_version_root(cache_root)
    return sorted([task for task in tasks if not (version_root / task).exists()])


def _stage_minimal_cache_from_default(cache_root: Path, tasks: Sequence[str]) -> None:
    if cache_root.resolve() == CACHE_PATH.resolve():
        return

    src_version_root = _cache_version_root(CACHE_PATH)
    dst_version_root = _cache_version_root(cache_root)
    dst_version_root.mkdir(parents=True, exist_ok=True)

    staged = 0
    for task in tasks:
        src_task_root = src_version_root / task
        if not src_task_root.exists():
            continue
        for action_mode_dir in src_task_root.iterdir():
            if not action_mode_dir.is_dir():
                continue
            candidates = sorted(action_mode_dir.glob("lightweight/**/*.safetensors"))
            if not candidates:
                candidates = sorted(action_mode_dir.glob("state/50hz/*.safetensors"))
            if not candidates:
                continue
            src_demo = candidates[0]
            rel_path = src_demo.relative_to(src_version_root)
            dst_demo = dst_version_root / rel_path
            dst_demo.parent.mkdir(parents=True, exist_ok=True)
            if not dst_demo.exists():
                shutil.copy2(src_demo, dst_demo)
                staged += 1

    # Mark local cache as "available" to prevent unnecessary pull attempts.
    lock_path = dst_version_root / ".lock"
    lock_path.touch(exist_ok=True)
    print(
        f"[cache] Staged {staged} demo file(s) into local cache root: {dst_version_root}"
    )


def _ensure_demo_cache(cache_root: Path, tasks: Sequence[str]) -> List[str]:
    missing = _missing_local_cache_tasks(cache_root, tasks)
    if not missing:
        return []

    print(
        f"[cache] Local cache missing for {len(missing)} task(s). "
        "Attempting demo download..."
    )
    demo_store = DemoStore(cache_root=cache_root)
    try:
        demo_store.pull_demos()
    except Exception as exc:
        print(f"[cache] pull_demos() failed: {exc}")

    missing_after = _missing_local_cache_tasks(cache_root, tasks)
    if not missing_after:
        return []

    # If cache is marked complete but task folders are still missing, retry once.
    if demo_store.cached:
        print("[cache] Incomplete cache detected. Forcing one refresh attempt...")
        demo_store.cached = False
        try:
            demo_store.pull_demos()
        except Exception as exc:
            print(f"[cache] Forced pull_demos() failed: {exc}")
        missing_after = _missing_local_cache_tasks(cache_root, tasks)

    return missing_after


def _build_convert_cmd(
    paths: Paths,
    env_name: str,
    output_dir: Path,
    max_demos: int,
    processes: int,
    pcd_max_dist: float,
    pcd_min_world_z: float,
    no_pointcloud: bool,
) -> List[str]:
    cmd = [
        str(paths.python_bin),
        str(paths.convert_script),
        "--env",
        env_name,
        "--output-dir",
        str(output_dir),
        "--processes",
        str(int(processes)),
        "--pcd-max-dist",
        str(float(pcd_max_dist)),
        "--pcd-min-world-z",
        str(float(pcd_min_world_z)),
    ]
    if max_demos is not None:
        cmd.extend(["--max-demos", str(int(max_demos))])
    if no_pointcloud:
        cmd.append("--no-pointcloud")
    return cmd


def _count_generated_demos(output_dir: Path) -> Tuple[int, int]:
    success = len(list(output_dir.glob("rby1_cartesian_demo_*.safetensors")))
    failure = len(
        list(output_dir.glob("failure/failed_rby1_cartesian_demo_*.safetensors"))
    )
    return success, failure


def _run_convert(
    paths: Paths,
    cmd: Sequence[str],
    log_path: Path,
) -> Tuple[int, float]:
    env = os.environ.copy()
    env["RBY1_DISABLE_PERTURB"] = "1"
    env["BIGYM_DISABLE_PERTURB"] = "1"
    env["MUJOCO_GL"] = "osmesa"
    env["BIGYM_CACHE_ROOT"] = str(paths.cache_root)

    start = time.time()
    proc = subprocess.run(
        list(cmd),
        cwd=paths.repo_root,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - start

    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_str = " ".join(shlex.quote(part) for part in cmd)
    log_path.write_text(
        f"$ {command_str}\n\n{proc.stdout}\n{proc.stderr}",
        encoding="utf-8",
    )
    return proc.returncode, elapsed


def _to_jsonable_report(report: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        return value

    return _convert(report)


def main() -> int:
    default_repo_root = Path(__file__).resolve().parents[2]
    default_python = default_repo_root / "bigym" / "venv" / "bin" / "python"
    if not default_python.exists():
        default_python = Path(sys.executable)

    parser = argparse.ArgumentParser(
        description=(
            "Run smoke tests (1 demo per task) then convert all remaining "
            "BiGym tasks to rby1_{task}_source format."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_repo_root / "rby1_equidiff" / "data" / "bigym",
        help="Root directory where rby1_{task}_source folders are saved",
    )
    parser.add_argument(
        "--smoke-root",
        type=Path,
        default=Path("/tmp/mimicbigym_step1_smoke"),
        help="Root directory to store smoke-test outputs and logs",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=default_python,
        help="Python executable to run convert_h1_to_rby1_cartesian.py",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.getenv("BIGYM_CACHE_ROOT", str(CACHE_PATH))),
        help=(
            "Demo cache root used by DemoStore. "
            "Set this to a writable path when ~/.bigym is read-only."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of tasks (env names, comma-separated or space-separated)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=16,
        help="Process count for full conversion",
    )
    parser.add_argument(
        "--smoke-processes",
        type=int,
        default=1,
        help="Process count for smoke conversion",
    )
    parser.add_argument(
        "--pcd-max-dist",
        type=float,
        default=3.0,
        help="Pointcloud max distance for conversion",
    )
    parser.add_argument(
        "--pcd-min-world-z",
        type=float,
        default=0.01,
        help="Minimum world-frame z to keep pointcloud points",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=-1,
        help="Max demos for full conversion (-1 for all demos)",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip smoke conversion and run full conversion directly",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only smoke conversion (no full conversion)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute commands; print what would run",
    )
    parser.add_argument(
        "--smoke-no-pointcloud",
        action="store_true",
        help="Disable pointcloud only for smoke conversion",
    )
    parser.add_argument(
        "--full-no-pointcloud",
        action="store_true",
        help="Disable pointcloud for full conversion",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    paths = Paths(
        repo_root=default_repo_root,
        convert_script=default_repo_root
        / "bigym"
        / "scripts"
        / "convert_h1_to_rby1_cartesian.py",
        readme=default_repo_root / "bigym" / "README.md",
        output_root=args.output_root,
        smoke_root=args.smoke_root,
        python_bin=args.python_bin,
        cache_root=args.cache_root.expanduser(),
    )

    all_tasks = _parse_readme_tasks(paths.readme)
    requested_tokens = _split_task_args(args.tasks or [])
    selected_tasks, unknown_tasks = _resolve_requested_tasks(all_tasks, requested_tokens)
    if unknown_tasks:
        print(f"[error] Unknown tasks: {', '.join(unknown_tasks)}")
        return 2

    existing_sources = _find_existing_source_tasks(paths.output_root)
    skipped_existing: List[str] = []
    pending_tasks: List[str] = []
    for task in selected_tasks:
        if _normalize_task_name(task) in existing_sources:
            skipped_existing.append(task)
        else:
            pending_tasks.append(task)

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "all_tasks_count": len(all_tasks),
        "selected_tasks": selected_tasks,
        "skipped_existing": skipped_existing,
        "pending_tasks": pending_tasks,
        "cache_root": paths.cache_root,
        "missing_tasks": [],
        "smoke_passed": [],
        "smoke_failed": [],
        "convert_passed": [],
        "convert_failed": [],
        "logs": {},
    }

    print(
        f"[plan] selected={len(selected_tasks)}, "
        f"skipped_existing={len(skipped_existing)}, pending={len(pending_tasks)}"
    )

    runnable_tasks = list(pending_tasks)
    if not args.dry_run and pending_tasks:
        _stage_minimal_cache_from_default(paths.cache_root, pending_tasks)
        missing_tasks = _ensure_demo_cache(paths.cache_root, pending_tasks)
        report["missing_tasks"] = missing_tasks
        runnable_tasks = [task for task in pending_tasks if task not in set(missing_tasks)]
        if missing_tasks:
            print(
                "[cache] Missing tasks after download attempt: "
                + ", ".join(missing_tasks)
            )

    smoke_candidates = runnable_tasks
    smoke_passed: List[str] = []
    smoke_failed: List[Dict[str, Any]] = []
    smoke_run_dir = paths.smoke_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_log_dir = smoke_run_dir / "logs"

    if args.skip_smoke:
        smoke_passed = list(smoke_candidates)
    else:
        print(f"[smoke] running smoke conversions for {len(smoke_candidates)} task(s)")
        for idx, task in enumerate(smoke_candidates, start=1):
            task_name = _normalize_task_name(task)
            smoke_output_dir = _task_output_dir(smoke_run_dir, task)
            smoke_log_path = smoke_log_dir / f"{task_name}.log"
            report["logs"][f"smoke:{task}"] = smoke_log_path

            cmd = _build_convert_cmd(
                paths=paths,
                env_name=task,
                output_dir=smoke_output_dir,
                max_demos=1,
                processes=args.smoke_processes,
                pcd_max_dist=args.pcd_max_dist,
                pcd_min_world_z=args.pcd_min_world_z,
                no_pointcloud=args.smoke_no_pointcloud,
            )

            print(f"[smoke {idx}/{len(smoke_candidates)}] {task}")
            if args.dry_run:
                print("  " + " ".join(shlex.quote(part) for part in cmd))
                smoke_passed.append(task)
                continue

            if smoke_output_dir.exists():
                shutil.rmtree(smoke_output_dir)

            rc, elapsed = _run_convert(paths, cmd, smoke_log_path)
            success_count, failure_count = _count_generated_demos(smoke_output_dir)
            generated = success_count + failure_count
            smoke_ok = rc == 0 and generated > 0
            if smoke_ok:
                smoke_passed.append(task)
                print(
                    f"  pass (elapsed={elapsed:.1f}s, success={success_count}, "
                    f"failure={failure_count})"
                )
            else:
                failure = {
                    "task": task,
                    "returncode": rc,
                    "elapsed_sec": elapsed,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "log": smoke_log_path,
                }
                smoke_failed.append(failure)
                print(
                    f"  fail (rc={rc}, elapsed={elapsed:.1f}s, generated={generated})"
                )

    report["smoke_passed"] = smoke_passed
    report["smoke_failed"] = smoke_failed

    run_full = not args.smoke_only and (args.skip_smoke or not smoke_failed)
    if not run_full and smoke_failed and not args.skip_smoke:
        print("[stop] smoke failures detected. Full conversion skipped.")

    if run_full:
        convert_log_dir = smoke_run_dir / "convert_logs"
        print(f"[convert] running full conversion for {len(smoke_passed)} task(s)")
        for idx, task in enumerate(smoke_passed, start=1):
            task_name = _normalize_task_name(task)
            output_dir = _task_output_dir(paths.output_root, task)
            log_path = convert_log_dir / f"{task_name}.log"
            report["logs"][f"convert:{task}"] = log_path

            cmd = _build_convert_cmd(
                paths=paths,
                env_name=task,
                output_dir=output_dir,
                max_demos=args.max_demos,
                processes=args.processes,
                pcd_max_dist=args.pcd_max_dist,
                pcd_min_world_z=args.pcd_min_world_z,
                no_pointcloud=args.full_no_pointcloud,
            )

            print(f"[convert {idx}/{len(smoke_passed)}] {task}")
            if args.dry_run:
                print("  " + " ".join(shlex.quote(part) for part in cmd))
                report["convert_passed"].append(task)
                continue

            rc, elapsed = _run_convert(paths, cmd, log_path)
            if rc == 0:
                report["convert_passed"].append(task)
                print(f"  pass (elapsed={elapsed:.1f}s)")
            else:
                report["convert_failed"].append(
                    {
                        "task": task,
                        "returncode": rc,
                        "elapsed_sec": elapsed,
                        "log": log_path,
                    }
                )
                print(f"  fail (rc={rc}, elapsed={elapsed:.1f}s)")

    report_path = args.report_path or (smoke_run_dir / "report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_to_jsonable_report(report), indent=2),
        encoding="utf-8",
    )

    print("\nSummary")
    print(f"- selected tasks: {len(selected_tasks)}")
    print(f"- skipped existing: {len(skipped_existing)}")
    print(f"- missing tasks: {len(report['missing_tasks'])}")
    print(f"- smoke passed: {len(report['smoke_passed'])}")
    print(f"- smoke failed: {len(report['smoke_failed'])}")
    print(f"- full convert passed: {len(report['convert_passed'])}")
    print(f"- full convert failed: {len(report['convert_failed'])}")
    print(f"- report: {report_path}")

    if unknown_tasks:
        return 2
    if report["missing_tasks"] or report["smoke_failed"] or report["convert_failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
