#!/usr/bin/env python3
"""Multinode golden-rule episode sweep for BEHAVIOR-1K.

This is the golden-rule counterpart to ``run_skill_metric_multinode_sweep.py``.
Instead of evaluating individual skill segments, it evaluates full episodes
following the ground-truth skill plan.

Architecture per worker:
1. Start ``serve_golden_rule.py`` on a dedicated GPU/port
2. Run ``eval_golden_rule_batch.py`` (or ``eval_golden_rule.py`` per demo) to
   evaluate full episodes
3. Collect per-episode skill-level and end-to-end metrics

Usage (prepare + launch on node 0):
    python scripts/run_golden_rule_multinode_sweep.py \
        --out-dir ./golden_rule_eval_runs/exp001 \
        --num-nodes 1 \
        --gpus-per-node 8 \
        --task turning_on_radio \
        --ckpt-dir ./checkpoints/pi05-b1kpt50-cs32 \
        --config-name pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
        --demo-data-path /path/to/2025-challenge-demos \
        --num-demos 10

Usage (merge results after all workers finish):
    python scripts/run_golden_rule_multinode_sweep.py --mode merge \
        --out-dir ./golden_rule_eval_runs/exp001
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_DIR_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K")
DEMO_DATA_PATH_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos")
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "openpi_comet" / "pi05-b1kpt50-cs32"
DEFAULT_CONFIG = "pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k"
CONDA_SH = "/mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh"


def q(text: Any) -> str:
    return shlex.quote(str(text))


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", int(port)))
        except OSError:
            return False
        return True


def find_free_port(preferred: int, *, stride: int = 1, max_tries: int = 50) -> int:
    preferred = int(preferred)
    stride = max(1, int(stride))
    for attempt in range(max(1, int(max_tries))):
        candidate = preferred + attempt * stride
        if is_port_free(candidate):
            return candidate
    raise RuntimeError(f"no free port found near {preferred} (stride={stride}, max_tries={max_tries})")


def wait_for_port_free(port: int, *, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if is_port_free(port):
            return True
        time.sleep(0.5)
    return is_port_free(port)


def wait_for_server(port: int, timeout_s: int) -> None:
    start = time.time()
    last_log = start
    health_url = f"http://127.0.0.1:{port}/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() - start < timeout_s:
        try:
            req = urllib.request.Request(health_url, headers={"Connection": "close"})
            with opener.open(req, timeout=2.0) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if 200 <= int(status) < 300:
                    return
        except (urllib.error.URLError, ConnectionError, OSError, socket.timeout, ValueError):
            now = time.time()
            if now - last_log >= 30:
                print(f"[worker] waiting for server healthz {health_url}... elapsed={int(now - start)}s", flush=True)
                last_log = now
            time.sleep(1.0)
        else:
            now = time.time()
            if now - last_log >= 30:
                print(f"[worker] server healthz not ready yet at {health_url}... elapsed={int(now - start)}s", flush=True)
                last_log = now
            time.sleep(1.0)
    raise TimeoutError(f"server healthz not ready at {health_url} after {timeout_s}s")


def wait_for_server_proc(
    *,
    proc: subprocess.Popen[str],
    port: int,
    timeout_s: int,
    log_file: Optional[Path] = None,
) -> None:
    start = time.time()
    last_exc: Optional[BaseException] = None
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            tail = _tail_text(log_file) if log_file is not None else ""
            raise RuntimeError(
                f"server process exited before becoming healthy (code={proc.returncode}). "
                f"log: {log_file}\n{tail}"
            )
        try:
            wait_for_server(port, timeout_s=10)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    tail = _tail_text(log_file) if log_file is not None else ""
    raise TimeoutError(
        f"server not ready on port {port} after {timeout_s}s. last_error={last_exc}. log: {log_file}\n{tail}"
    )


def _tail_text(path: Optional[Path], max_lines: int = 80) -> str:
    if path is None or not path.exists():
        return "[launcher] log file does not exist yet"
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


def get_demo_ids(demo_data_path: Path, task_name: str, num_demos: int) -> List[str]:
    """Get sorted demo IDs for a task from annotation files."""
    from omnigibson.learning.eval_golden_rule import get_demo_ids_for_task
    return get_demo_ids_for_task(
        demo_data_path=str(demo_data_path),
        task_name=task_name,
        limit=num_demos if num_demos > 0 else None,
    )


def start_golden_rule_server(
    *,
    task_name: str,
    port: int,
    gpu_id: int,
    ckpt_dir: Path,
    config_name: str,
    policy_backend: str,
    demo_data_path: Path,
    demo_id: str,
    openpi_env: str,
    behavior_dir: Path,
    out_dir: Path,
) -> tuple[subprocess.Popen[str], Path]:
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
python scripts/serve_golden_rule.py \\
  --task_name={q(task_name)} \\
  --control_mode=receeding_horizon \\
  --max_len=32 \\
  --port={q(port)} \\
  --policy-backend={q(policy_backend)} \\
  --demo_data_path={q(demo_data_path)} \\
  --demo_id={q(demo_id)} \\
  --fine_grained_level=2 \\
  policy:checkpoint --policy.config={q(config_name)} --policy.dir={q(ckpt_dir)}
"""
    with log_file.open("w") as f:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return proc, log_file


def run_episode_eval(
    *,
    demo_id: str,
    task_name: str,
    port: int,
    gpu_id: int,
    behavior_env: str,
    behavior_dir: Path,
    demo_data_path: Path,
    out_dir: Path,
    max_steps: int,
    dry_run: bool,
    write_video: bool,
) -> Dict[str, Any]:
    """Run a single episode evaluation via eval_golden_rule.py."""
    episode_out = out_dir / "raw" / task_name / f"demo_{demo_id}"
    episode_out.mkdir(parents=True, exist_ok=True)
    episode_log = episode_out / "episode_eval.log"

    og_appdata_base = Path(os.environ.get("OMNIGIBSON_APPDATA_PATH_BASE", "/tmp/omnigibson-appdata"))
    og_user = os.environ.get("USER", "user")
    og_appdata = og_appdata_base / og_user / f"gpu{gpu_id}"
    og_appdata.mkdir(parents=True, exist_ok=True)

    cmd = f"""
set -euo pipefail
source {q(CONDA_SH)}
conda activate {q(behavior_env)}
cd {q(behavior_dir)}
export PYTHONUNBUFFERED=1
export PYTHONPATH={q(str(behavior_dir / 'joylo'))}:{q(str(behavior_dir / 'OmniGibson'))}:{q(str(behavior_dir / 'bddl3'))}${{PYTHONPATH:+:$PYTHONPATH}}
export NO_PROXY="localhost,127.0.0.1,::1${{NO_PROXY:+,$NO_PROXY}}"
export no_proxy="localhost,127.0.0.1,::1${{no_proxy:+,$no_proxy}}"
export CUDA_VISIBLE_DEVICES={q(gpu_id)}
unset OMNIGIBSON_GPU_ID
export OMNIGIBSON_DATA_PATH={q(str(behavior_dir / 'datasets'))}
export OMNIGIBSON_APPDATA_PATH={q(str(og_appdata))}
export MPLBACKEND="${{MPLBACKEND:-Agg}}"
export TORCHDYNAMO_DISABLE="${{TORCHDYNAMO_DISABLE:-1}}"
export TORCHINDUCTOR_DISABLE="${{TORCHINDUCTOR_DISABLE:-1}}"
export OMNIGIBSON_HEADLESS=true
export OMNIGIBSON_DISABLE_EXTENSION_REGISTRY=0
export OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK=1
python OmniGibson/omnigibson/learning/eval_golden_rule.py \\
  policy=websocket \\
  task.name={q(task_name)} \\
  demo_data_path={q(demo_data_path)} \\
  demo_id={q(demo_id)} \\
  log_path={q(episode_out)} \\
  headless=true \\
  write_video={q(str(write_video).lower())} \\
  max_steps={q(max_steps)} \\
  model.host=127.0.0.1 \\
  model.port={q(port)} \\
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \\
  partial_scene_load=true \\
  dry_run={q(str(dry_run).lower())}
"""
    with episode_log.open("w") as f:
        proc = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, text=True)

    # Load metrics
    metrics_glob = sorted((episode_out / "metrics").glob("*.json"))
    result: Dict[str, Any] = {
        "demo_id": demo_id,
        "task_name": task_name,
        "runtime_ok": proc.returncode == 0 and len(metrics_glob) > 0,
        "returncode": proc.returncode,
        "metrics_path": str(metrics_glob[0]) if metrics_glob else None,
        "episode_log": str(episode_log),
    }

    if metrics_glob:
        with metrics_glob[0].open() as f:
            metrics = json.load(f)
        result["n_skills"] = metrics.get("n_skills", 0)
        result["n_skill_successes"] = metrics.get("n_skill_successes", 0)
        result["skill_success_rate"] = result["n_skill_successes"] / max(1, result["n_skills"])
        result["endtoend_success"] = metrics.get("endtoend_success", False)
        result["total_steps"] = metrics.get("total_steps", 0)
        result["terminated"] = metrics.get("terminated", False)
        result["truncated"] = metrics.get("truncated", False)
        result["skill_results"] = metrics.get("skill_results", [])

    return result


def run_worker(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    with manifest_path.open() as f:
        manifest = json.load(f)

    worker_demos = manifest.get("worker_demos", {}).get(str(args.worker_rank), [])
    if not worker_demos:
        print(f"[worker {args.worker_rank:03d}] no demos assigned", flush=True)
        return 0

    results_path = args.out_dir / "worker_results" / f"worker_{args.worker_rank:03d}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_demos = set()
    if args.resume and results_path.exists():
        with results_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("runtime_ok"):
                    done_demos.add(row["demo_id"])

    pending = [d for d in worker_demos if d not in done_demos]
    print(f"[worker {args.worker_rank:03d}] {len(done_demos)} done, {len(pending)} pending", flush=True)

    ckpt_dir = resolve_checkpoint_dir(args.ckpt_dir)
    server_proc: Optional[subprocess.Popen[str]] = None
    server_log: Optional[Path] = None
    task_port = args.port

    try:
        # Find free port
        task_port = find_free_port(args.port, stride=args.gpus_per_node, max_tries=200)

        # Start golden rule server for this worker
        # For golden rule, we use the first demo's plan (server loads plan at startup)
        # The evaluator will connect and the server autonomously advances skills
        first_demo = pending[0] if pending else worker_demos[0]
        server_proc, server_log = start_golden_rule_server(
            task_name=args.task,
            port=task_port,
            gpu_id=args.gpu_id,
            ckpt_dir=ckpt_dir,
            config_name=args.config_name,
            policy_backend=args.policy_backend,
            demo_data_path=args.demo_data_path,
            demo_id=first_demo,
            openpi_env=args.openpi_env,
            behavior_dir=args.behavior_dir,
            out_dir=args.out_dir / "server_logs",
        )
        print(
            f"[worker {args.worker_rank:03d}] started golden-rule server for task={args.task} "
            f"gpu={args.gpu_id} port={task_port} demo={first_demo}",
            flush=True,
        )
        wait_for_server_proc(
            proc=server_proc,
            port=task_port,
            timeout_s=args.server_ready_timeout,
            log_file=server_log,
        )
        print(f"[worker {args.worker_rank:03d}] server ready on port {task_port}", flush=True)

        for demo_id in pending:
            print(f"[worker {args.worker_rank:03d}] evaluating demo={demo_id}", flush=True)
            row = run_episode_eval(
                demo_id=demo_id,
                task_name=args.task,
                port=task_port,
                gpu_id=args.gpu_id,
                behavior_env=args.behavior_env,
                behavior_dir=args.behavior_dir,
                demo_data_path=args.demo_data_path,
                out_dir=args.out_dir,
                max_steps=args.max_steps,
                dry_run=args.dry_run,
                write_video=args.write_video,
            )
            row["worker_rank"] = args.worker_rank
            row["node_rank"] = args.node_rank
            row["gpu_id"] = args.gpu_id
            row["port"] = task_port

            with results_path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

            status = "OK" if row.get("runtime_ok") else "FAIL"
            sr = row.get("skill_success_rate", 0.0)
            et = row.get("endtoend_success", False)
            print(
                f"[worker {args.worker_rank:03d}] demo={demo_id} status={status} "
                f"skills={row.get('n_skill_successes', 0)}/{row.get('n_skills', 0)} "
                f"sr={sr:.1%} et={et}",
                flush=True,
            )
    finally:
        stop_process(server_proc)
        wait_for_port_free(task_port)

    return 0


def build_manifest(args: argparse.Namespace) -> None:
    """Build manifest with demo assignments per worker."""
    demo_ids = get_demo_ids(args.demo_data_path, args.task, args.num_demos)
    if not demo_ids:
        raise RuntimeError(f"no demos found for task {args.task}")

    total_workers = args.num_nodes * args.gpus_per_node

    # Round-robin assign demos to workers
    worker_demos: Dict[str, List[str]] = {str(i): [] for i in range(total_workers)}
    for idx, demo_id in enumerate(demo_ids):
        worker_rank = idx % total_workers
        worker_demos[str(worker_rank)].append(demo_id)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(args.out_dir),
        "task": args.task,
        "ckpt_dir": str(args.ckpt_dir),
        "resolved_ckpt_dir": str(resolve_checkpoint_dir(args.ckpt_dir)),
        "config_name": args.config_name,
        "policy_backend": args.policy_backend,
        "num_nodes": args.num_nodes,
        "gpus_per_node": args.gpus_per_node,
        "total_workers": total_workers,
        "num_demos": len(demo_ids),
        "demo_ids": demo_ids,
        "worker_demos": worker_demos,
        "dry_run": args.dry_run,
        "write_video": args.write_video,
        "max_steps": args.max_steps,
    }
    json_dump(args.out_dir / "manifest.json", manifest)

    # Write worker plan
    worker_rows = []
    for worker_rank in range(total_workers):
        demos = worker_demos.get(str(worker_rank), [])
        worker_rows.append({
            "worker_rank": worker_rank,
            "demo_count": len(demos),
            "demos": demos,
        })
    json_dump(args.out_dir / "worker_plan.json", worker_rows)

    print(f"Manifest written to {args.out_dir / 'manifest.json'}")
    print(f"Total demos: {len(demo_ids)}, workers: {total_workers}")
    for wr in range(total_workers):
        print(f"  worker {wr:03d}: {len(worker_demos[str(wr)])} demos")


def merge_results(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    with manifest_path.open() as f:
        manifest = json.load(f)

    planned_demos = set(manifest.get("demo_ids", []))
    result_rows: List[Dict[str, Any]] = []

    results_path = args.out_dir / "worker_results"
    if results_path.exists():
        for result_file in sorted(results_path.glob("worker_*.jsonl")):
            with result_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    result_rows.append(json.loads(line))

    # Deduplicate by demo_id
    deduped = {row["demo_id"]: row for row in result_rows}
    rows = [deduped[d] for d in sorted(deduped) if d in planned_demos]
    missing = sorted(planned_demos - set(deduped))

    # Aggregate stats
    total_skills = sum(r.get("n_skills", 0) for r in rows)
    total_skill_successes = sum(r.get("n_skill_successes", 0) for r in rows)
    total_et_success = sum(1 for r in rows if r.get("endtoend_success"))

    # Per-skill breakdown
    skill_counter: Counter = Counter()
    skill_success_counter: Counter = Counter()
    for r in rows:
        for sr in r.get("skill_results", []):
            desc = sr[0] if isinstance(sr, (list, tuple)) else sr.get("description", "unknown")
            skill_counter[desc] += 1
            success = sr[1] if isinstance(sr, (list, tuple)) else sr.get("success", False)
            if success:
                skill_success_counter[desc] += 1

    skill_breakdown = {
        desc: {
            "attempts": skill_counter[desc],
            "successes": skill_success_counter[desc],
            "success_rate": float(skill_success_counter[desc] / skill_counter[desc]),
        }
        for desc in sorted(skill_counter.keys())
    }

    summary = {
        "task": manifest.get("task"),
        "out_dir": str(args.out_dir),
        "planned_demos": len(planned_demos),
        "completed_demos": len(rows),
        "missing_demos": missing,
        "total_skills": total_skills,
        "total_skill_successes": total_skill_successes,
        "skill_success_rate": float(total_skill_successes / max(1, total_skills)),
        "endtoend_success_count": total_et_success,
        "endtoend_success_rate": float(total_et_success / max(1, len(rows))),
        "skill_breakdown": skill_breakdown,
        "per_demo_results": rows,
    }

    json_dump(args.out_dir / "golden_rule_summary.json", summary)

    # Print summary
    print("")
    print("=" * 60)
    print("GOLDEN RULE MULTINODE SWEEP SUMMARY")
    print("=" * 60)
    print(f"Task: {manifest.get('task')}")
    print(f"Demos: {len(rows)}/{len(planned_demos)}")
    print(f"Skills: {total_skill_successes}/{total_skills} ({summary['skill_success_rate']:.1%})")
    print(f"End-to-end: {total_et_success}/{len(rows)} ({summary['endtoend_success_rate']:.1%})")
    print(f"Missing: {len(missing)} demos")
    print("=" * 60)

    return 0


def launch_node(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if args.node_rank == 0 and (args.rebuild_manifest or not manifest_path.exists()):
        build_manifest(args)
    else:
        # Wait for manifest
        start = time.time()
        while time.time() - start < args.prepare_timeout:
            if manifest_path.exists():
                break
            time.sleep(2.0)
        if not manifest_path.exists():
            raise TimeoutError(f"manifest not ready after {args.prepare_timeout}s")

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
                "--mode", "worker",
                "--out-dir", str(args.out_dir),
                "--node-rank", str(args.node_rank),
                "--num-nodes", str(args.num_nodes),
                "--gpus-per-node", str(args.gpus_per_node),
                "--port-base", str(args.port_base),
                "--max-steps", str(args.max_steps),
                "--server-ready-timeout", str(args.server_ready_timeout),
                "--prepare-timeout", str(args.prepare_timeout),
                "--openpi-env", args.openpi_env,
                "--behavior-env", args.behavior_env,
                "--config-name", args.config_name,
                "--policy-backend", args.policy_backend,
                "--ckpt-dir", str(args.ckpt_dir),
                "--behavior-dir", str(args.behavior_dir),
                "--demo-data-path", str(args.demo_data_path),
                "--task", args.task,
                "--worker-rank", str(worker_rank),
                "--gpu-id", str(gpu_id),
                "--port", str(port),
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            if args.write_video:
                cmd.append("--write-video")
            if args.resume:
                cmd.append("--resume")
            if args.num_demos > 0:
                cmd.extend(["--num-demos", str(args.num_demos)])
            with worker_log.open("w") as f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
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
                    print(_tail_text(worker_log))
                    print(f"[launcher] ===== worker {worker_rank:03d} log tail end =====")
                    return code
            if not next_remaining:
                return 0
            time.sleep(2.0)
            remaining = next_remaining
        return 0
    finally:
        for proc, _, _ in children:
            stop_process(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multinode golden-rule episode sweep for BEHAVIOR-1K."
    )
    parser.add_argument("--mode", choices=["launch", "prepare", "merge", "worker"], default="launch")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "golden_rule_eval_runs" / f"multinode_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", "1")))
    parser.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GPUS_PER_NODE", "8")))
    parser.add_argument("--local-gpu-ids", default=os.environ.get("LOCAL_GPU_IDS", ""))
    parser.add_argument("--port-base", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--server-ready-timeout", type=int, default=1800)
    parser.add_argument("--prepare-timeout", type=int, default=3600)
    parser.add_argument("--openpi-env", default="openpi-comet-nas")
    parser.add_argument("--behavior-env", default="behavior")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--policy-backend", choices=["auto", "torch", "jax"], default="auto")
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--behavior-dir", type=Path, default=BEHAVIOR_DIR_DEFAULT)
    parser.add_argument("--demo-data-path", type=Path, default=DEMO_DATA_PATH_DEFAULT)
    parser.add_argument("--task", default="turning_on_radio", help="task name to evaluate")
    parser.add_argument("--num-demos", type=int, default=0, help="0 means all demos")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--worker-rank", type=int, default=-1)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--port", type=int, default=-1)
    args = parser.parse_args()

    args.out_dir = args.out_dir.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.behavior_dir = args.behavior_dir.resolve()
    args.demo_data_path = args.demo_data_path.resolve()

    if args.local_gpu_ids.strip():
        args.local_gpu_ids = [int(x.strip()) for x in args.local_gpu_ids.split(",") if x.strip()]
    else:
        args.local_gpu_ids = list(range(args.gpus_per_node))
    if len(args.local_gpu_ids) != args.gpus_per_node:
        raise RuntimeError("LOCAL_GPU_IDS count must match --gpus-per-node")

    if int(args.port_base) <= 0:
        rng = random.SystemRandom()
        args.port_base = rng.randint(20000, 60000)

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
