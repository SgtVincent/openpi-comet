#!/usr/bin/env python3
"""Warm up BEHAVIOR-1K challenge caches and package them into a tar.gz archive.

Driver mode:
- Reads the 50 challenge task ids from `docs/challenge/task_data.json`
- Spawns a fresh subprocess per task to avoid long-lived OmniGibson state buildup
- Reuses the same local appdata directory across tasks so caches accumulate
- Optionally packages the warmed cache into `cache.tar.gz`

Single-task mode:
- Loads one task with OmniGibson + R1Pro + RGB observations
- Resets once and steps a few frames to populate shader / texture / sensor caches
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEHAVIOR_DIR = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K")
DEFAULT_APPDATA_BASE = Path("/tmp/omnigibson-appdata")
DEFAULT_CACHE_TAR = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/behavior1k_cache_all_gpus.tar.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-dir", type=Path, default=DEFAULT_BEHAVIOR_DIR)
    parser.add_argument("--appdata-base", type=Path, default=DEFAULT_APPDATA_BASE)
    parser.add_argument("--cache-tar", type=Path, default=DEFAULT_CACHE_TAR)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "warmup_logs")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for --single-task mode.")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma separated gpu ids for driver mode. Empty means auto-detect all GPUs.",
    )
    parser.add_argument("--steps", type=int, default=2, help="Rendered env steps per task in single-task mode.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma separated task ids. Empty means all 50 challenge tasks from task_data.json.",
    )
    parser.add_argument(
        "--partial-scene-load",
        action="store_true",
        help="Use task-relevant room loading instead of full scene loading.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after individual task failures in driver mode.",
    )
    parser.add_argument(
        "--skip-tar",
        action="store_true",
        help="Skip packaging the warmed appdata into a tar.gz archive.",
    )
    parser.add_argument(
        "--single-task",
        action="store_true",
        help="Internal mode: warm only one task in the current process.",
    )
    parser.add_argument("--task", type=str, default="", help="Task id for --single-task mode.")
    return parser.parse_args()


def read_challenge_tasks(behavior_dir: Path) -> list[str]:
    task_data_path = behavior_dir / "docs" / "challenge" / "task_data.json"
    payload = json.loads(task_data_path.read_text())
    return [item["id"] for item in payload["tasks"]]


def get_task_list(args: argparse.Namespace) -> list[str]:
    if args.tasks:
        return [task.strip() for task in args.tasks.split(",") if task.strip()]
    return read_challenge_tasks(args.behavior_dir)


def get_cache_dir(appdata_base: Path, gpu_id: int) -> Path:
    user = os.environ.get("USER", "default_user")
    return appdata_base / user / f"gpu{gpu_id}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    path = Path(tarinfo.name)
    # Skip transient screenshots and keep the archive focused on reusable caches.
    if "screenshots" in path.parts:
        return None
    return tarinfo


def package_cache(appdata_base: Path, gpu_id: int, output_tar: Path) -> Path:
    cache_dir = get_cache_dir(appdata_base, gpu_id)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache directory does not exist: {cache_dir}")

    ensure_parent(output_tar)
    arcname = cache_dir.relative_to(appdata_base.parent)
    with tarfile.open(output_tar, "w:gz") as tar:
        for root, dirs, files in os.walk(cache_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(cache_dir)
            rel_parts = rel_root.parts
            if "screenshots" in rel_parts:
                dirs[:] = []
                continue

            tar.add(root_path, arcname=str(arcname / rel_root), recursive=False, filter=_tar_filter)
            for filename in files:
                file_path = root_path / filename
                if not os.access(file_path, os.R_OK):
                    continue
                tar.add(file_path, arcname=str(arcname / rel_root / filename), recursive=False, filter=_tar_filter)
    return output_tar


def _detect_gpu_ids() -> list[int]:
    # Best-effort detection. If it fails, fall back to GPU 0 only.
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            return [0]
        count = len([ln for ln in proc.stdout.splitlines() if ln.strip().startswith("GPU ")])
        return list(range(max(count, 1)))
    except Exception:
        return [0]


def _parse_gpu_ids(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return _detect_gpu_ids()
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out or _detect_gpu_ids()


def package_cache_bundle(appdata_base: Path, gpu_ids: list[int], output_tar: Path) -> Path:
    # Bundle multiple GPU caches under the same user directory into a single tarball.
    user = os.environ.get("USER", "default_user")
    user_dir = appdata_base / user
    if not user_dir.exists():
        raise FileNotFoundError(f"user cache directory does not exist: {user_dir}")

    ensure_parent(output_tar)
    arc_root = user_dir.relative_to(appdata_base.parent)
    gpu_set = {f"gpu{gid}" for gid in gpu_ids}
    with tarfile.open(output_tar, "w:gz") as tar:
        for root, dirs, files in os.walk(user_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(user_dir)
            rel_parts = rel_root.parts
            if rel_parts and rel_parts[0].startswith("gpu") and rel_parts[0] not in gpu_set:
                dirs[:] = []
                continue
            if "screenshots" in rel_parts:
                dirs[:] = []
                continue

            tar.add(root_path, arcname=str(arc_root / rel_root), recursive=False, filter=_tar_filter)
            for filename in files:
                file_path = root_path / filename
                if not os.access(file_path, os.R_OK):
                    continue
                tar.add(file_path, arcname=str(arc_root / rel_root / filename), recursive=False, filter=_tar_filter)
    return output_tar


def _atomic_pack(pack_fn, output_tar: Path) -> Path:
    """
    Pack to a temp file then atomically rename to the final path.
    This avoids consumers reading a partially-written .tar.gz and getting EOF errors.
    """
    ensure_parent(output_tar)
    tmp_tar = output_tar.parent / (output_tar.name + ".tmp")
    if tmp_tar.exists():
        try:
            tmp_tar.unlink()
        except Exception:
            pass
    pack_fn(tmp_tar)
    os.replace(tmp_tar, output_tar)
    return output_tar


def driver_main(args: argparse.Namespace) -> int:
    tasks = get_task_list(args)
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.log_dir / f"warmup_manifest_gpu{args.gpu_id}_{int(time.time())}.json"

    print("=" * 72)
    print("BEHAVIOR-1K Cache Warmup")
    print("=" * 72)
    print(f"behavior_dir : {args.behavior_dir}")
    print(f"gpu_ids      : {gpu_ids}")
    print(f"appdata_base : {args.appdata_base}")
    print(f"cache_dirs   : {[str(get_cache_dir(args.appdata_base, gid)) for gid in gpu_ids]}")
    print(f"log_dir      : {args.log_dir}")
    print(f"cache_tar    : {args.cache_tar}")
    print(f"tasks        : {len(tasks)}")
    print(f"partial_load : {args.partial_scene_load}")
    print("=" * 72)

    results: list[dict[str, object]] = []

    def run_one(task_idx: int, task_name: str, gpu_id: int) -> dict[str, object]:
        task_log = args.log_dir / f"{task_idx:02d}_{task_name}_gpu{gpu_id}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-task",
            "--task",
            task_name,
            "--behavior-dir",
            str(args.behavior_dir),
            "--appdata-base",
            str(args.appdata_base),
            "--gpu-id",
            str(gpu_id),
            "--steps",
            str(args.steps),
        ]
        if args.partial_scene_load:
            cmd.append("--partial-scene-load")

        start = time.time()
        with task_log.open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=args.behavior_dir)
        elapsed_s = time.time() - start
        return {
            "task": task_name,
            "task_idx": task_idx,
            "gpu_id": gpu_id,
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed_s, 2),
            "log_path": str(task_log),
        }

    # Schedule tasks one-at-a-time per GPU (max concurrency = num_gpus).
    next_task_idx = 1
    task_iter = iter(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        inflight: dict[int, concurrent.futures.Future] = {}

        # Prime one task per GPU
        for gpu_id in gpu_ids:
            try:
                task = next(task_iter)
            except StopIteration:
                break
            print(f"[{next_task_idx:02d}/{len(tasks):02d}] warm {task} on gpu{gpu_id} ...")
            inflight[gpu_id] = pool.submit(run_one, next_task_idx, task, gpu_id)
            next_task_idx += 1

        while inflight:
            done, _ = concurrent.futures.wait(inflight.values(), return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                gpu_id = None
                for k, v in inflight.items():
                    if v is fut:
                        gpu_id = k
                        break
                if gpu_id is None:
                    continue
                inflight.pop(gpu_id, None)

                result = fut.result()
                results.append(result)
                print(
                    f"  done task={result['task']} gpu{result['gpu_id']} rc={result['returncode']} "
                    f"elapsed={result['elapsed_s']}s log={result['log_path']}"
                )

                if result["returncode"] != 0 and not args.keep_going:
                    manifest_path.write_text(json.dumps(results, indent=2))
                    print(f"[Error] task warmup failed: {result['task']}")
                    print(f"[Info] partial manifest written to: {manifest_path}")
                    return int(result["returncode"])

                try:
                    task = next(task_iter)
                except StopIteration:
                    continue
                print(f"[{next_task_idx:02d}/{len(tasks):02d}] warm {task} on gpu{gpu_id} ...")
                inflight[gpu_id] = pool.submit(run_one, next_task_idx, task, gpu_id)
                next_task_idx += 1

    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"[Info] manifest written to: {manifest_path}")

    if not args.skip_tar:
        if len(gpu_ids) > 1:
            output_tar = _atomic_pack(
                lambda p: package_cache_bundle(args.appdata_base, gpu_ids, p),
                args.cache_tar,
            )
        else:
            output_tar = _atomic_pack(
                lambda p: package_cache(args.appdata_base, gpu_ids[0], p),
                args.cache_tar,
            )
        size_mb = output_tar.stat().st_size / (1024 * 1024)
        print(f"[Info] packed cache tar: {output_tar} ({size_mb:.1f} MiB)")

    failures = [item for item in results if item["returncode"] != 0]
    if failures:
        print(f"[Warn] warmup completed with {len(failures)} failures")
        return 1

    print("[Info] warmup completed successfully")
    return 0


def single_task_main(args: argparse.Namespace) -> int:
    if not args.task:
        raise ValueError("--task is required in --single-task mode")

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "true")
    os.environ.setdefault("OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK", "1")
    os.environ["OMNIGIBSON_GPU_ID"] = str(args.gpu_id)

    cache_dir = get_cache_dir(args.appdata_base, args.gpu_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OMNIGIBSON_APPDATA_PATH"] = str(cache_dir)

    # Ensure the repo-local modules are importable when the script is launched directly.
    sys.path.insert(0, str(args.behavior_dir / "joylo"))
    sys.path.insert(0, str(args.behavior_dir / "OmniGibson"))
    sys.path.insert(0, str(args.behavior_dir / "bddl3"))

    import torch as th
    import omnigibson as og
    from gello.robots.sim_robot.og_teleop_utils import (
        augment_rooms,
        generate_robot_config,
        get_task_relevant_room_types,
        load_available_tasks,
    )
    from omnigibson.learning.utils.eval_utils import (
        PROPRIOCEPTION_INDICES,
        generate_basic_environment_config,
    )
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.GUI_VIEWPORT_ONLY = True
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.DEFAULT_VIEWER_WIDTH = 64
    gm.DEFAULT_VIEWER_HEIGHT = 64

    available_tasks = load_available_tasks()
    if args.task not in available_tasks:
        raise KeyError(f"task not found in available_tasks.yaml: {args.task}")

    task_cfg = available_tasks[args.task][0]
    cfg = generate_basic_environment_config(task_name=args.task, task_cfg=task_cfg)
    if args.partial_scene_load:
        relevant_rooms = get_task_relevant_room_types(activity_name=args.task)
        relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], args.task)
        cfg["scene"]["load_room_types"] = relevant_rooms

    cfg["robots"] = [generate_robot_config(task_name=args.task, task_cfg=task_cfg)]
    cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
    cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())

    start = time.time()
    env = None
    try:
        env = og.Environment(configs=cfg)
        env.reset()
        robot = env.scene.object_registry("name", "robot_r1")
        action = th.zeros(robot.action_dim, dtype=th.float32)
        for _ in range(max(args.steps, 1)):
            env.step(action, n_render_iterations=1)

        elapsed_s = time.time() - start
        print(
            json.dumps(
                {
                    "task": args.task,
                    "scene_model": task_cfg["scene_model"],
                    "partial_scene_load": args.partial_scene_load,
                    "objects": env.scene.n_objects,
                    "elapsed_s": round(elapsed_s, 2),
                    "cache_dir": str(cache_dir),
                },
                indent=2,
            )
        )
        return 0
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            og.shutdown()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    if args.single_task:
        return single_task_main(args)
    return driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
