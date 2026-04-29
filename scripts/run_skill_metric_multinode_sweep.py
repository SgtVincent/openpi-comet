#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_DIR_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K")
DEMO_DATA_PATH_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos")
RAWDATA_PATH_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata")
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "openpi_comet" / "pi05-b1kpt50-cs32"
DEFAULT_CONFIG = "pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep"
REGISTRY_REL_PATH = Path("OmniGibson/omnigibson/learning/utils/segment_skill_metric_registry.py")
CONDA_SH = "/mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh"


def q(text: Any) -> str:
    return shlex.quote(str(text))


def parse_csv_ints(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip().lower() for chunk in text.split(",") if chunk.strip()]


def flatten_skill_desc(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip().lower()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(flatten_skill_desc(item))
        return out
    s = str(value).strip().lower()
    return [s] if s else []


def load_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("skill_metric_registry", registry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load registry from {registry_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.SKILL_METRIC_REGISTRY)


def parse_frame_duration(frame_duration: Any) -> Optional[Tuple[int, int]]:
    try:
        if isinstance(frame_duration, (list, tuple)) and len(frame_duration) == 2:
            return int(frame_duration[0]), int(frame_duration[1])
        if isinstance(frame_duration, str):
            import ast

            parsed = ast.literal_eval(frame_duration)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return int(parsed[0]), int(parsed[1])
    except Exception:
        return None
    return None


def get_dynamic_max_steps(frame_duration: Any, fallback: int) -> int:
    parsed = parse_frame_duration(frame_duration)
    if parsed is None:
        return fallback
    start, end = parsed
    duration = end - start
    if duration <= 0:
        return fallback
    return duration * 2


def resolve_checkpoint_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "params").is_dir() or (path / "model.safetensors").exists():
        return path
    step_dirs = []
    for child in path.iterdir():
        if child.is_dir() and child.name.isdigit():
            if (child / "params").is_dir() or (child / "model.safetensors").exists():
                step_dirs.append(child)
    if step_dirs:
        return sorted(step_dirs, key=lambda p: int(p.name))[-1]
    return path


def wait_for_server(port: int, timeout_s: int) -> None:
    start = time.time()
    last_log = start
    while time.time() - start < timeout_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect(("127.0.0.1", port))
                return
            except OSError:
                now = time.time()
                # Avoid silent long waits in worker logs.
                if now - last_log >= 30:
                    print(f"[worker] waiting for server port {port}... elapsed={int(now-start)}s", flush=True)
                    last_log = now
                time.sleep(1.0)
    raise TimeoutError(f"server not ready on port {port} after {timeout_s}s")


def wait_for_path(path: Path, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if path.exists():
            return
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {path}")


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return f"[launcher] log file does not exist yet: {path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"[launcher] failed to read log tail from {path}: {exc}"
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(["..."] + lines[-max_lines:])


def stop_process(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def build_job_key(sample: Dict[str, Any]) -> str:
    return f"{sample['skill']}|{sample['task_name']}|{sample['demo_id']}|{int(sample['skill_idx']):03d}"


def collect_skill_jobs(
    *,
    registry: Dict[str, Dict[str, Any]],
    demo_data_path: Path,
    skills_filter: Sequence[str],
    max_samples_per_skill: int,
    max_samples_per_skill_task: int,
    max_total_jobs: int,
) -> List[Dict[str, Any]]:
    targets = set(registry)
    if skills_filter:
        wanted = set(skills_filter)
        missing = sorted(wanted - targets)
        if missing:
            raise RuntimeError(f"unknown skills requested: {missing}")
        targets = wanted

    jobs: List[Dict[str, Any]] = []
    per_skill = Counter()
    per_skill_task = Counter()
    ann_root = demo_data_path / "annotations"
    for ann_path in sorted(ann_root.glob("task-*/episode_*.json")):
        with ann_path.open() as f:
            ann = json.load(f)
        task_name = str(ann.get("task_name", "")).strip().replace(" ", "_")
        instance_id = ann.get("instance_id")
        for seg in ann.get("skill_annotation", []) or []:
            desc = " ".join(flatten_skill_desc(seg.get("skill_description")))
            if desc not in targets:
                continue
            if max_samples_per_skill > 0 and per_skill[desc] >= max_samples_per_skill:
                continue
            skill_task_key = (desc, task_name)
            if max_samples_per_skill_task > 0 and per_skill_task[skill_task_key] >= max_samples_per_skill_task:
                continue

            demo_id = ann_path.stem.replace("episode_", "")
            sample = {
                "skill": desc,
                "task_dir": ann_path.parent.name,
                "task_name": task_name,
                "ann_path": str(ann_path),
                "demo_id": demo_id,
                "instance_id": instance_id,
                "skill_idx": int(seg.get("skill_idx", 0)),
                "frame_duration": seg.get("frame_duration"),
                "object_id": seg.get("object_id"),
                "manipulating_object_id": seg.get("manipulating_object_id"),
                "spatial_prefix": seg.get("spatial_prefix"),
            }
            sample["job_key"] = build_job_key(sample)
            jobs.append(sample)
            per_skill[desc] += 1
            per_skill_task[skill_task_key] += 1

            if max_samples_per_skill > 0 and all(per_skill[skill] >= max_samples_per_skill for skill in targets):
                return sorted(jobs, key=lambda row: (row["skill"], row["task_name"], row["demo_id"], row["skill_idx"]))
            if max_total_jobs > 0 and len(jobs) >= max_total_jobs:
                return sorted(jobs, key=lambda row: (row["skill"], row["task_name"], row["demo_id"], row["skill_idx"]))
    return sorted(jobs, key=lambda row: (row["skill"], row["task_name"], row["demo_id"], row["skill_idx"]))


def build_worker_assignments(jobs: List[Dict[str, Any]], total_workers: int) -> List[List[Dict[str, Any]]]:
    if total_workers <= 0:
        raise ValueError("total_workers must be positive")
    worker_jobs: List[List[Dict[str, Any]]] = [[] for _ in range(total_workers)]
    if not jobs:
        return worker_jobs

    target_chunk = max(1, math.ceil(len(jobs) / total_workers))
    jobs_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        jobs_by_task[job["task_name"]].append(job)

    task_chunks: List[List[Dict[str, Any]]] = []
    for task_jobs in [jobs_by_task[name] for name in sorted(jobs_by_task)]:
        task_jobs = sorted(task_jobs, key=lambda row: (row["skill"], row["demo_id"], row["skill_idx"]))
        if len(task_jobs) > max(1, int(target_chunk * 1.5)):
            num_slices = math.ceil(len(task_jobs) / target_chunk)
        else:
            num_slices = 1
        slice_size = math.ceil(len(task_jobs) / num_slices)
        for idx in range(num_slices):
            chunk = task_jobs[idx * slice_size : (idx + 1) * slice_size]
            if chunk:
                task_chunks.append(chunk)

    loads = [0 for _ in range(total_workers)]
    for chunk in sorted(task_chunks, key=len, reverse=True):
        worker_idx = min(range(total_workers), key=lambda idx: (loads[idx], idx))
        worker_jobs[worker_idx].extend(chunk)
        loads[worker_idx] += len(chunk)

    for idx in range(total_workers):
        worker_jobs[idx] = sorted(
            worker_jobs[idx], key=lambda row: (row["task_name"], row["skill"], row["demo_id"], row["skill_idx"])
        )
    return worker_jobs


def render_planned_coverage_md(summary: Dict[str, Any]) -> str:
    lines = ["# Planned Multinode Skill Sweep", ""]
    lines.append(f"- out_dir: `{summary['out_dir']}`")
    lines.append(f"- total_jobs: `{summary['total_jobs']}`")
    lines.append(f"- unique_skills: `{summary['unique_skills']}`")
    lines.append(f"- unique_tasks: `{summary['unique_tasks']}`")
    lines.append(f"- unique_demos: `{summary['unique_demos']}`")
    lines.append(f"- total_workers: `{summary['total_workers']}`")
    lines.append("")
    lines.append("| Skill | Tasks | Demos | Segments |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["skill_rows"]:
        lines.append(f"| {row['skill']} | {row['task_count']} | {row['demo_count']} | {row['segment_count']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_manifest(args: argparse.Namespace, jobs: List[Dict[str, Any]]) -> None:
    total_workers = args.num_nodes * args.gpus_per_node
    worker_jobs = build_worker_assignments(jobs, total_workers)
    jobs_dir = args.out_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    worker_rows = []
    for worker_rank, worker_job_list in enumerate(worker_jobs):
        worker_path = jobs_dir / f"worker_{worker_rank:03d}.json"
        json_dump(worker_path, worker_job_list)
        worker_rows.append(
            {
                "worker_rank": worker_rank,
                "job_count": len(worker_job_list),
                "task_count": len({row["task_name"] for row in worker_job_list}),
                "skills": len({row["skill"] for row in worker_job_list}),
                "job_file": str(worker_path),
            }
        )

    grouped = defaultdict(list)
    for job in jobs:
        grouped[job["skill"]].append(job)
    skill_rows = []
    for skill, skill_jobs in sorted(grouped.items()):
        skill_rows.append(
            {
                "skill": skill,
                "task_count": len({row["task_name"] for row in skill_jobs}),
                "demo_count": len({row["demo_id"] for row in skill_jobs}),
                "segment_count": len(skill_jobs),
            }
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(args.out_dir),
        "ckpt_dir": str(args.ckpt_dir),
        "resolved_ckpt_dir": str(resolve_checkpoint_dir(args.ckpt_dir)),
        "config_name": args.config_name,
        "num_nodes": args.num_nodes,
        "gpus_per_node": args.gpus_per_node,
        "total_workers": total_workers,
        "max_steps_fallback": args.max_steps,
        "dry_run": args.dry_run,
        "write_video": args.write_video,
        "segment_predicate_dump_trace": args.segment_predicate_dump_trace,
        "skills_filter": list(args.skills_filter),
        "max_samples_per_skill": args.max_samples_per_skill,
        "max_samples_per_skill_task": args.max_samples_per_skill_task,
        "max_total_jobs": args.max_total_jobs,
        "jobs": jobs,
        "workers": worker_rows,
    }
    json_dump(args.out_dir / "manifest.json", manifest)
    write_csv(
        args.out_dir / "worker_plan.csv",
        worker_rows,
        fieldnames=["worker_rank", "job_count", "task_count", "skills", "job_file"],
    )

    planned_summary = {
        "out_dir": str(args.out_dir),
        "total_jobs": len(jobs),
        "unique_skills": len(skill_rows),
        "unique_tasks": len({row["task_name"] for row in jobs}),
        "unique_demos": len({row["demo_id"] for row in jobs}),
        "total_workers": total_workers,
        "skill_rows": skill_rows,
    }
    json_dump(args.out_dir / "planned_coverage.json", planned_summary)
    write_csv(
        args.out_dir / "planned_skill_coverage.csv",
        skill_rows,
        fieldnames=["skill", "task_count", "demo_count", "segment_count"],
    )
    (args.out_dir / "planned_skill_coverage.md").write_text(render_planned_coverage_md(planned_summary))


def start_server(
    *,
    task_name: str,
    port: int,
    gpu_id: int,
    ckpt_dir: Path,
    config_name: str,
    openpi_env: str,
    behavior_dir: Path,
    out_dir: Path,
) -> subprocess.Popen[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"server_{task_name}_gpu{gpu_id}_p{port}.log"
    cmd = f"""
set -euo pipefail
source {q(CONDA_SH)}
conda activate {q(openpi_env)}
cd {q(REPO_ROOT)}
export CUDA_VISIBLE_DEVICES={q(gpu_id)}
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH={q(str(REPO_ROOT / 'src'))}:{q(str(behavior_dir / 'joylo'))}:{q(str(behavior_dir / 'OmniGibson'))}:{q(str(behavior_dir / 'bddl3'))}${{PYTHONPATH:+:$PYTHONPATH}}
python scripts/serve_b1k.py --task_name={q(task_name)} --control_mode=receeding_horizon --max_len=32 --port={q(port)} policy:checkpoint --policy.config={q(config_name)} --policy.dir={q(ckpt_dir)}
"""
    with log_file.open("w") as f:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
    return proc


def load_metrics_row(metrics_path: Path, sample: Dict[str, Any], runtime_ok: bool, returncode: int, segment_log: Path) -> Dict[str, Any]:
    with metrics_path.open() as f:
        metrics = json.load(f)
    return {
        "job_key": sample["job_key"],
        "skill": sample["skill"],
        "task_name": sample["task_name"],
        "demo_id": sample["demo_id"],
        "instance_id": sample.get("instance_id"),
        "skill_idx": sample["skill_idx"],
        "frame_duration": sample.get("frame_duration"),
        "dynamic_max_steps": get_dynamic_max_steps(sample.get("frame_duration"), fallback=0),
        "runtime_ok": runtime_ok,
        "returncode": returncode,
        "success": metrics.get("success"),
        "result_type": metrics.get("result_type"),
        "metric_family": metrics.get("predicate_debug", {}).get("metric_family"),
        "start_all_satisfied": metrics.get("predicate_debug", {}).get("start_all_satisfied"),
        "final_step": metrics.get("rollout", {}).get("final_step"),
        "metrics_path": str(metrics_path),
        "segment_log": str(segment_log),
    }


def run_segment_eval(
    *,
    sample: Dict[str, Any],
    port: int,
    gpu_id: int,
    behavior_env: str,
    behavior_dir: Path,
    demo_data_path: Path,
    rawdata_path: Path,
    out_dir: Path,
    default_max_steps: int,
    dry_run: bool,
    write_video: bool,
    segment_predicate_dump_trace: bool,
) -> Dict[str, Any]:
    task_name = sample["task_name"]
    demo_id = sample["demo_id"]
    skill_idx = int(sample["skill_idx"])
    dynamic_max_steps = get_dynamic_max_steps(sample.get("frame_duration"), fallback=default_max_steps)

    skill_out = out_dir / "raw" / task_name / f"demo_{demo_id}" / f"skill_{skill_idx:03d}"
    skill_out.mkdir(parents=True, exist_ok=True)
    segment_log = skill_out / "segment_eval.log"
    # OmniGibson/Isaac can conflict across concurrent workers if they share the same app data/cache.
    # Also, putting appdata on NAS is extremely slow. Default to local disk (/tmp) for appdata, and
    # isolate per-(run, gpu, port).
    og_appdata_base = Path(os.environ.get("OMNIGIBSON_APPDATA_PATH_BASE", "/tmp/omnigibson-appdata"))
    og_user = os.environ.get("USER", "user")
    og_appdata = og_appdata_base / og_user / out_dir.name / f"gpu{gpu_id}_p{port}"
    og_appdata.mkdir(parents=True, exist_ok=True)

    existing_metrics = sorted((skill_out / "metrics").glob("*.json"))
    if existing_metrics:
        row = load_metrics_row(existing_metrics[0], sample, True, 0, segment_log)
        row["resume_hit"] = True
        row["dynamic_max_steps"] = dynamic_max_steps
        return row

    cmd = f"""
set -euo pipefail
source {q(CONDA_SH)}
conda activate {q(behavior_env)}
cd {q(behavior_dir)}
export PYTHONUNBUFFERED=1
export PYTHONPATH={q(str(behavior_dir / 'joylo'))}:{q(str(behavior_dir / 'OmniGibson'))}:{q(str(behavior_dir / 'bddl3'))}${{PYTHONPATH:+:$PYTHONPATH}}
export NO_PROXY="localhost,127.0.0.1,::1${{NO_PROXY:+,$NO_PROXY}}"
export no_proxy="localhost,127.0.0.1,::1${{no_proxy:+,$no_proxy}}"
export OMNIGIBSON_GPU_ID={q(gpu_id)}
export OMNIGIBSON_DATA_PATH={q(str(behavior_dir / 'datasets'))}
export OMNIGIBSON_APPDATA_PATH={q(str(og_appdata))}
export MPLBACKEND="${{MPLBACKEND:-Agg}}"
export TORCHDYNAMO_DISABLE="${{TORCHDYNAMO_DISABLE:-1}}"
export TORCHINDUCTOR_DISABLE="${{TORCHINDUCTOR_DISABLE:-1}}"
export OMNIGIBSON_HEADLESS=true
export OMNIGIBSON_DISABLE_EXTENSION_REGISTRY=0
export OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK=1
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name={q(task_name)} \
  demo_data_path={q(demo_data_path)} \
  rawdata_path={q(rawdata_path)} \
  segment_level=skill \
  segment_idx={q(skill_idx)} \
  success_mode=segment_predicates \
  grounding_topk=3 \
  dry_run={q(str(dry_run).lower())} \
  log_path={q(skill_out)} \
  demo_id={q(demo_id)} \
  headless=true \
  write_video={q(str(write_video).lower())} \
  segment_max_steps={q(dynamic_max_steps)} \
  model.host=127.0.0.1 \
  model.port={q(port)} \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
  partial_scene_load=true \
  segment_predicate_window_mode=anytime \
  segment_predicate_dump_trace={q(str(segment_predicate_dump_trace).lower())}
"""
    with segment_log.open("w") as f:
        proc = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, text=True)

    metrics_glob = sorted((skill_out / "metrics").glob("*.json"))
    result: Dict[str, Any] = {
        "job_key": sample["job_key"],
        "skill": sample["skill"],
        "task_name": task_name,
        "demo_id": demo_id,
        "instance_id": sample.get("instance_id"),
        "skill_idx": skill_idx,
        "frame_duration": sample.get("frame_duration"),
        "dynamic_max_steps": dynamic_max_steps,
        "runtime_ok": proc.returncode == 0 and len(metrics_glob) > 0,
        "returncode": proc.returncode,
        "metrics_path": str(metrics_glob[0]) if metrics_glob else None,
        "segment_log": str(segment_log),
    }
    if metrics_glob:
        result.update(load_metrics_row(metrics_glob[0], sample, result["runtime_ok"], proc.returncode, segment_log))
        result["dynamic_max_steps"] = dynamic_max_steps
    return result


def run_worker(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    wait_for_path(manifest_path, args.prepare_timeout)
    worker_jobs_path = args.out_dir / "jobs" / f"worker_{args.worker_rank:03d}.json"
    if not worker_jobs_path.exists():
        raise RuntimeError(f"missing worker job file: {worker_jobs_path}")
    with worker_jobs_path.open() as f:
        worker_jobs = json.load(f)

    results_path = args.out_dir / "worker_results" / f"worker_{args.worker_rank:03d}.jsonl"
    done_keys = set()
    if args.resume and results_path.exists():
        for row in load_jsonl_rows([results_path]):
            done_keys.add(row["job_key"])

    pending_jobs = [job for job in worker_jobs if job["job_key"] not in done_keys]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for job in pending_jobs:
        grouped[job["task_name"]].append(job)

    ckpt_dir = resolve_checkpoint_dir(args.ckpt_dir)
    status = {
        "worker_rank": args.worker_rank,
        "node_rank": args.node_rank,
        "gpu_id": args.gpu_id,
        "port": args.port,
        "planned_jobs": len(worker_jobs),
        "pending_jobs": len(pending_jobs),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(args.out_dir / "worker_status" / f"worker_{args.worker_rank:03d}.started.json", status)

    total_runtime_ok = 0
    total_success = 0
    for task_name in sorted(grouped):
        server_proc: Optional[subprocess.Popen[str]] = None
        try:
            if args.server_start_stagger_s > 0 and args.worker_rank >= 0:
                # Stagger heavy model init/JIT to avoid 8-way compile storms on one node.
                delay = args.server_start_stagger_s * int(args.worker_rank)
                if delay > 0:
                    print(f"[worker {args.worker_rank:03d}] staggering server start by {delay}s", flush=True)
                    time.sleep(delay)
            server_proc = start_server(
                task_name=task_name,
                port=args.port,
                gpu_id=args.gpu_id,
                ckpt_dir=ckpt_dir,
                config_name=args.config_name,
                openpi_env=args.openpi_env,
                behavior_dir=args.behavior_dir,
                out_dir=args.out_dir / "server_logs",
            )
            print(
                f"[worker {args.worker_rank:03d}] started server for task={task_name} gpu={args.gpu_id} port={args.port}",
                flush=True,
            )
            wait_for_server(args.port, args.server_ready_timeout)
            print(f"[worker {args.worker_rank:03d}] server ready on port {args.port}", flush=True)
            for sample in grouped[task_name]:
                print(
                    f"[worker {args.worker_rank:03d}] {sample['skill']} | {sample['task_name']} | "
                    f"demo={sample['demo_id']} | skill_idx={sample['skill_idx']}"
                )
                row = run_segment_eval(
                    sample=sample,
                    port=args.port,
                    gpu_id=args.gpu_id,
                    behavior_env=args.behavior_env,
                    behavior_dir=args.behavior_dir,
                    demo_data_path=args.demo_data_path,
                    rawdata_path=args.rawdata_path,
                    out_dir=args.out_dir,
                    default_max_steps=args.max_steps,
                    dry_run=args.dry_run,
                    write_video=args.write_video,
                    segment_predicate_dump_trace=args.segment_predicate_dump_trace,
                )
                row["worker_rank"] = args.worker_rank
                row["node_rank"] = args.node_rank
                row["gpu_id"] = args.gpu_id
                append_jsonl(results_path, row)
                total_runtime_ok += int(bool(row.get("runtime_ok")))
                total_success += int(bool(row.get("success")))
        finally:
            stop_process(server_proc)

    final_status = {
        **status,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_jobs": len(worker_jobs),
        "runtime_ok": total_runtime_ok,
        "success": total_success,
    }
    json_dump(args.out_dir / "worker_status" / f"worker_{args.worker_rank:03d}.done.json", final_status)
    return 0


def render_summary_md(summary: Dict[str, Any]) -> str:
    lines = ["# Multinode Skill Segment Sweep Summary", ""]
    lines.append(f"- out_dir: `{summary['out_dir']}`")
    lines.append(f"- planned_jobs: `{summary['planned_jobs']}`")
    lines.append(f"- completed_jobs: `{summary['completed_jobs']}`")
    lines.append(f"- runtime_pass: `{summary['runtime_pass']}`")
    lines.append(f"- policy_success: `{summary['policy_success']}`")
    lines.append(f"- missing_jobs: `{summary['missing_jobs']}`")
    lines.append("")
    lines.append("| Skill | Tasks | Demos | Segments | Runtime Pass Rate | Success Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["skill_summary"]:
        lines.append(
            f"| {row['skill']} | {row['task_count']} | {row['demo_count']} | {row['segment_count']} | "
            f"{row['runtime_pass_rate']:.4f} | {row['success_rate']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def merge_results(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    planned_jobs = manifest.get("jobs", [])
    planned_by_key = {row["job_key"]: row for row in planned_jobs}

    result_rows = load_jsonl_rows(sorted((args.out_dir / "worker_results").glob("worker_*.jsonl")))
    deduped: Dict[str, Dict[str, Any]] = {}
    for row in result_rows:
        deduped[row["job_key"]] = row
    rows = [deduped[key] for key in sorted(deduped)]
    missing_keys = sorted(set(planned_by_key) - set(deduped))

    skill_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skill_task_grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        skill_grouped[row["skill"]].append(row)
        skill_task_grouped[(row["skill"], row["task_name"])].append(row)

    skill_summary = []
    for skill, skill_rows in sorted(skill_grouped.items()):
        runtime_pass = sum(int(bool(row.get("runtime_ok"))) for row in skill_rows)
        success = sum(int(bool(row.get("success"))) for row in skill_rows)
        total = len(skill_rows)
        skill_summary.append(
            {
                "skill": skill,
                "task_count": len({row["task_name"] for row in skill_rows}),
                "demo_count": len({row["demo_id"] for row in skill_rows}),
                "segment_count": total,
                "runtime_pass_count": runtime_pass,
                "runtime_pass_rate": runtime_pass / total if total else 0.0,
                "success_count": success,
                "success_rate": success / total if total else 0.0,
                "timeout_count": sum(1 for row in skill_rows if row.get("result_type") == "timeout"),
                "predicate_satisfied_count": sum(
                    1 for row in skill_rows if row.get("result_type") == "predicate_satisfied"
                ),
            }
        )

    skill_task_summary = []
    for (skill, task_name), task_rows in sorted(skill_task_grouped.items()):
        runtime_pass = sum(int(bool(row.get("runtime_ok"))) for row in task_rows)
        success = sum(int(bool(row.get("success"))) for row in task_rows)
        total = len(task_rows)
        skill_task_summary.append(
            {
                "skill": skill,
                "task_name": task_name,
                "demo_count": len({row["demo_id"] for row in task_rows}),
                "segment_count": total,
                "runtime_pass_count": runtime_pass,
                "runtime_pass_rate": runtime_pass / total if total else 0.0,
                "success_count": success,
                "success_rate": success / total if total else 0.0,
            }
        )

    summary = {
        "out_dir": str(args.out_dir),
        "planned_jobs": len(planned_jobs),
        "completed_jobs": len(rows),
        "runtime_pass": sum(int(bool(row.get("runtime_ok"))) for row in rows),
        "policy_success": sum(int(bool(row.get("success"))) for row in rows),
        "missing_jobs": len(missing_keys),
        "missing_job_keys": missing_keys,
        "skill_summary": skill_summary,
        "skill_task_summary": skill_task_summary,
    }

    json_dump(args.out_dir / "multinode_skill_results.json", rows)
    write_csv(
        args.out_dir / "multinode_skill_results.csv",
        rows,
        fieldnames=[
            "job_key",
            "skill",
            "task_name",
            "demo_id",
            "instance_id",
            "skill_idx",
            "frame_duration",
            "dynamic_max_steps",
            "runtime_ok",
            "success",
            "result_type",
            "metric_family",
            "start_all_satisfied",
            "final_step",
            "returncode",
            "metrics_path",
            "segment_log",
            "worker_rank",
            "node_rank",
            "gpu_id",
        ],
    )
    json_dump(args.out_dir / "multinode_skill_summary.json", summary)
    write_csv(
        args.out_dir / "multinode_skill_summary.csv",
        skill_summary,
        fieldnames=[
            "skill",
            "task_count",
            "demo_count",
            "segment_count",
            "runtime_pass_count",
            "runtime_pass_rate",
            "success_count",
            "success_rate",
            "timeout_count",
            "predicate_satisfied_count",
        ],
    )
    write_csv(
        args.out_dir / "multinode_skill_task_summary.csv",
        skill_task_summary,
        fieldnames=[
            "skill",
            "task_name",
            "demo_count",
            "segment_count",
            "runtime_pass_count",
            "runtime_pass_rate",
            "success_count",
            "success_rate",
        ],
    )
    (args.out_dir / "multinode_skill_summary.md").write_text(render_summary_md(summary))
    print(json.dumps(summary, indent=2))
    return 0


def launch_node(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if args.node_rank == 0 and (args.rebuild_manifest or not manifest_path.exists()):
        registry = load_registry(args.behavior_dir / REGISTRY_REL_PATH)
        jobs = collect_skill_jobs(
            registry=registry,
            demo_data_path=args.demo_data_path,
            skills_filter=args.skills_filter,
            max_samples_per_skill=args.max_samples_per_skill,
            max_samples_per_skill_task=args.max_samples_per_skill_task,
            max_total_jobs=args.max_total_jobs,
        )
        write_manifest(args, jobs)
    else:
        wait_for_path(manifest_path, args.prepare_timeout)

    if args.mode == "prepare":
        return 0

    local_gpu_ids = args.local_gpu_ids
    children: List[Tuple[subprocess.Popen[str], int, Path]] = []
    try:
        for local_rank, gpu_id in enumerate(local_gpu_ids):
            worker_rank = args.node_rank * args.gpus_per_node + local_rank
            port = args.port_base + local_rank
            worker_log = args.out_dir / "launcher_logs" / f"node{args.node_rank:02d}_worker{worker_rank:03d}.log"
            worker_log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--out-dir",
                str(args.out_dir),
                "--node-rank",
                str(args.node_rank),
                "--num-nodes",
                str(args.num_nodes),
                "--gpus-per-node",
                str(args.gpus_per_node),
                "--port-base",
                str(args.port_base),
                "--max-steps",
                str(args.max_steps),
                "--server-ready-timeout",
                str(args.server_ready_timeout),
                "--prepare-timeout",
                str(args.prepare_timeout),
                "--openpi-env",
                args.openpi_env,
                "--behavior-env",
                args.behavior_env,
                "--config-name",
                args.config_name,
                "--ckpt-dir",
                str(args.ckpt_dir),
                "--behavior-dir",
                str(args.behavior_dir),
                "--demo-data-path",
                str(args.demo_data_path),
                "--rawdata-path",
                str(args.rawdata_path),
                "--worker-rank",
                str(worker_rank),
                "--gpu-id",
                str(gpu_id),
                "--port",
                str(port),
            ]
            if args.server_start_stagger_s > 0:
                cmd.extend(["--server-start-stagger-s", str(args.server_start_stagger_s)])
            if args.dry_run:
                cmd.append("--dry-run")
            if args.write_video:
                cmd.append("--write-video")
            if args.segment_predicate_dump_trace:
                cmd.append("--segment-predicate-dump-trace")
            if args.resume:
                cmd.append("--resume")
            with worker_log.open("w") as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            children.append((proc, worker_rank, worker_log))

        remaining = list(children)
        while remaining:
            next_remaining: List[Tuple[subprocess.Popen[str], int, Path]] = []
            for proc, worker_rank, worker_log in remaining:
                code = proc.poll()
                if code is None:
                    next_remaining.append((proc, worker_rank, worker_log))
                    continue
                if code != 0:
                    print(
                        f"[launcher] worker {worker_rank:03d} exited with code {code}. "
                        f"log: {worker_log}"
                    )
                    print(f"[launcher] ===== worker {worker_rank:03d} log tail begin =====")
                    print(tail_text(worker_log))
                    print(f"[launcher] ===== worker {worker_rank:03d} log tail end =====")
                    return code
            if not next_remaining:
                return 0
            time.sleep(2.0)
            remaining = next_remaining
        return 0
    finally:
        for proc, _, _ in children:
            if proc.poll() is None:
                proc.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multinode segment/skill sweep for BEHAVIOR-1K skill success evaluation."
    )
    parser.add_argument("--mode", choices=["launch", "prepare", "merge", "worker"], default="launch")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "segment_eval_runs" / f"skill_multinode_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", "1")))
    parser.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GPUS_PER_NODE", "8")))
    parser.add_argument("--local-gpu-ids", default=os.environ.get("LOCAL_GPU_IDS", ""))
    parser.add_argument("--port-base", type=int, default=16000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--server-ready-timeout", type=int, default=1800)
    parser.add_argument(
        "--server-start-stagger-s",
        type=int,
        default=0,
        help="optional stagger (seconds) multiplied by worker_rank before starting the task server; helps avoid JIT storms",
    )
    parser.add_argument("--prepare-timeout", type=int, default=3600)
    parser.add_argument("--openpi-env", default="openpi-comet-nas")
    parser.add_argument("--behavior-env", default="behavior")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--behavior-dir", type=Path, default=BEHAVIOR_DIR_DEFAULT)
    parser.add_argument("--demo-data-path", type=Path, default=DEMO_DATA_PATH_DEFAULT)
    parser.add_argument("--rawdata-path", type=Path, default=RAWDATA_PATH_DEFAULT)
    parser.add_argument("--skills", default="", help="comma-separated subset of skill names")
    parser.add_argument("--max-samples-per-skill", type=int, default=0, help="0 means use all matched segments")
    parser.add_argument(
        "--max-samples-per-skill-task",
        type=int,
        default=0,
        help="optional cap per (skill, task) pair; 0 means unlimited",
    )
    parser.add_argument("--max-total-jobs", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--dry-run", action="store_true", help="pass dry_run=true to eval_segment.py")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--segment-predicate-dump-trace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument(
        "--allow-tmp-out-dir",
        action="store_true",
        help="allow writing outputs under /tmp; not recommended for formal evaluation",
    )
    parser.add_argument("--worker-rank", type=int, default=-1)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--port", type=int, default=-1)
    args = parser.parse_args()

    args.out_dir = args.out_dir.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.behavior_dir = args.behavior_dir.resolve()
    args.demo_data_path = args.demo_data_path.resolve()
    args.rawdata_path = args.rawdata_path.resolve()
    args.skills_filter = parse_csv_strings(args.skills)
    if args.local_gpu_ids.strip():
        args.local_gpu_ids = parse_csv_ints(args.local_gpu_ids)
    else:
        args.local_gpu_ids = list(range(args.gpus_per_node))
    if len(args.local_gpu_ids) != args.gpus_per_node:
        raise RuntimeError("LOCAL_GPU_IDS count must match --gpus-per-node")
    if str(args.out_dir).startswith("/tmp") and not args.allow_tmp_out_dir:
        raise RuntimeError(
            "refusing to write formal evaluation outputs under /tmp. "
            "Please use a persistent path under the repo, or pass --allow-tmp-out-dir for temporary smoke tests."
        )
    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "merge":
        return merge_results(args)
    if args.mode == "worker":
        if args.worker_rank < 0 or args.gpu_id < 0 or args.port < 0:
            raise RuntimeError("worker mode requires --worker-rank, --gpu-id and --port")
        return run_worker(args)
    return launch_node(args)


if __name__ == "__main__":
    sys.exit(main())
