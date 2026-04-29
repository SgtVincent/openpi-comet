#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_DIR_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K")
DEMO_DATA_PATH_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos")
RAWDATA_PATH_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata")
DEFAULT_CKPT = Path("/mnt/bn/behavior-data-hl/chenjunting/checkpoints_lf/pi05_skill_pt12_pretrain_4x8_20260325_065102/30997")
DEFAULT_CONFIG = "pi05_b1k_skill-pt12_pretrain_lr1e-4_2ep"


def q(text: Any) -> str:
    return shlex.quote(str(text))


def load_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("skill_metric_registry", registry_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return dict(mod.SKILL_METRIC_REGISTRY)


def flatten(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip().lower()
        return [s] if s else []
    if isinstance(x, (list, tuple)):
        out: List[str] = []
        for y in x:
            out.extend(flatten(y))
        return out
    s = str(x).strip().lower()
    return [s] if s else []


def get_dynamic_max_steps(frame_duration: Any, fallback: int = 100) -> int:
    try:
        if isinstance(frame_duration, list) and len(frame_duration) == 2:
            start, end = int(frame_duration[0]), int(frame_duration[1])
        elif isinstance(frame_duration, str):
            import ast
            res = ast.literal_eval(frame_duration)
            start, end = int(res[0]), int(res[1])
        else:
            return fallback
        
        duration = end - start
        if duration <= 0:
            return fallback
            
        # Set max_steps to 2x the demonstration length
        return duration * 2
    except Exception:
        return fallback

def pick_representative_skills(registry: Dict[str, Dict[str, Any]], demo_data_path: Path) -> List[Dict[str, Any]]:
    targets = set(registry)
    found: Dict[str, Dict[str, Any]] = {}
    ann_root = demo_data_path / "annotations"
    for task_dir in sorted(ann_root.glob("task-*")):
        for ann_path in sorted(task_dir.glob("episode_*.json")):
            with ann_path.open() as f:
                ann = json.load(f)
            task_name = str(ann.get("task_name", "")).strip().replace(" ", "_")
            for seg in ann.get("skill_annotation", []) or []:
                desc = " ".join(flatten(seg.get("skill_description")))
                if desc not in targets or desc in found:
                    continue
                demo_id = ann_path.stem.replace("episode_", "")
                found[desc] = {
                    "skill": desc,
                    "task_dir": task_dir.name,
                    "task_name": task_name,
                    "ann_path": str(ann_path),
                    "demo_id": demo_id,
                    "skill_idx": int(seg.get("skill_idx", 0)),
                    "frame_duration": seg.get("frame_duration"),
                    "object_id": seg.get("object_id"),
                    "manipulating_object_id": seg.get("manipulating_object_id"),
                    "spatial_prefix": seg.get("spatial_prefix"),
                }
                if len(found) == len(targets):
                    break
            if len(found) == len(targets):
                break
        if len(found) == len(targets):
            break
    missing = sorted(targets - set(found))
    if missing:
        raise RuntimeError(f"failed to find representative annotations for skills: {missing}")
    return [found[s] for s in sorted(found)]


def wait_for_server(port: int, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(1.0)
    raise TimeoutError(f"server not ready on port {port} after {timeout_s}s")


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
    log_file = out_dir / f"server_{task_name}_gpu{gpu_id}_p{port}.log"
    cmd = f"""
set -euo pipefail
source /mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh
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
        proc = subprocess.Popen(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)
    return proc


def stop_process(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, 15)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass


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
) -> Dict[str, Any]:
    task_name = sample["task_name"]
    demo_id = sample["demo_id"]
    skill_idx = sample["skill_idx"]
    
    # Calculate dynamic max_steps: 2x the demonstration length
    dynamic_max_steps = get_dynamic_max_steps(sample.get("frame_duration"), fallback=default_max_steps)
    print(f"    [Dynamic max_steps] {task_name} | {sample['skill']} -> duration: {sample.get('frame_duration')} => max_steps: {dynamic_max_steps}")
    
    skill_out = out_dir / "raw" / task_name / f"demo_{demo_id}" / f"skill_{skill_idx:03d}"
    skill_out.mkdir(parents=True, exist_ok=True)
    log_path = skill_out
    segment_log = skill_out / "segment_eval.log"
    cmd = f"""
set -euo pipefail
source /mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate {q(behavior_env)}
cd {q(behavior_dir)}
export PYTHONUNBUFFERED=1
export PYTHONPATH={q(str(behavior_dir / 'joylo'))}:{q(str(behavior_dir / 'OmniGibson'))}:{q(str(behavior_dir / 'bddl3'))}${{PYTHONPATH:+:$PYTHONPATH}}
export NO_PROXY="localhost,127.0.0.1,::1${{NO_PROXY:+,$NO_PROXY}}"
export no_proxy="localhost,127.0.0.1,::1${{no_proxy:+,$no_proxy}}"
export OMNIGIBSON_GPU_ID={q(gpu_id)}
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
  dry_run=false \
  log_path={q(log_path)} \
  demo_id={q(demo_id)} \
  headless=true \
  write_video=false \
  segment_max_steps={q(dynamic_max_steps)} \
  model.host=127.0.0.1 \
  model.port={q(port)} \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
  partial_scene_load=true \
  segment_predicate_window_mode=anytime \
  segment_predicate_dump_trace=false
"""
    with segment_log.open("w") as f:
        proc = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, text=True)

    metrics_glob = list((skill_out / "metrics").glob("*.json"))
    result: Dict[str, Any] = {
        "skill": sample["skill"],
        "task_name": task_name,
        "demo_id": demo_id,
        "skill_idx": skill_idx,
        "runtime_ok": proc.returncode == 0 and len(metrics_glob) > 0,
        "returncode": proc.returncode,
        "segment_log": str(segment_log),
        "metrics_path": str(metrics_glob[0]) if metrics_glob else None,
    }
    if metrics_glob:
        with metrics_glob[0].open() as f:
            metrics = json.load(f)
        result.update(
            {
                "success": metrics.get("success"),
                "result_type": metrics.get("result_type"),
                "metric_family": metrics.get("predicate_debug", {}).get("metric_family"),
                "start_all_satisfied": metrics.get("predicate_debug", {}).get("start_all_satisfied"),
                "final_step": metrics.get("rollout", {}).get("final_step"),
            }
        )
    return result


def write_reports(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    json_path = out_dir / "runtime_sweep_results.json"
    csv_path = out_dir / "runtime_sweep_results.csv"
    md_path = out_dir / "runtime_sweep_matrix.md"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "skill",
        "task_name",
        "demo_id",
        "skill_idx",
        "metric_family",
        "runtime_ok",
        "success",
        "result_type",
        "start_all_satisfied",
        "final_step",
        "returncode",
        "metrics_path",
        "segment_log",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    total = len(rows)
    runtime_pass = sum(bool(r.get("runtime_ok")) for r in rows)
    policy_success = sum(bool(r.get("success")) for r in rows)
    with md_path.open("w") as f:
        f.write("# 34-Skill Runtime Sweep Matrix\n\n")
        f.write(f"- total skills: `{total}`\n")
        f.write(f"- runtime pass: `{runtime_pass}`\n")
        f.write(f"- policy success: `{policy_success}`\n\n")
        f.write("| Skill | Task | Family | Runtime | Policy | Result | Final Step |\n")
        f.write("|---|---|---|---|---|---|---:|\n")
        for row in rows:
            runtime = "PASS" if row.get("runtime_ok") else "FAIL"
            policy = "PASS" if row.get("success") else "FAIL"
            f.write(
                f"| {row['skill']} | {row['task_name']} | {row.get('metric_family','')} | "
                f"{runtime} | {policy} | {row.get('result_type','')} | {row.get('final_step','')} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=3)
    parser.add_argument("--port-base", type=int, default=14020)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--server-ready-timeout", type=int, default=900)
    parser.add_argument("--openpi-env", default="openpi-comet-nas")
    parser.add_argument("--behavior-env", default="behavior")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--behavior-dir", type=Path, default=BEHAVIOR_DIR_DEFAULT)
    parser.add_argument("--demo-data-path", type=Path, default=DEMO_DATA_PATH_DEFAULT)
    parser.add_argument("--rawdata-path", type=Path, default=RAWDATA_PATH_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "segment_eval_runs" / f"skill_runtime_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--skills", default="", help="comma-separated subset of skill names to run")
    parser.add_argument("--merge-from", default="", help="comma-separated list of runtime_sweep_results.json files or directories to merge")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_from.strip():
        rows: List[Dict[str, Any]] = []
        seen = set()
        for item in [x.strip() for x in args.merge_from.split(",") if x.strip()]:
            p = Path(item)
            if p.is_dir():
                p = p / "runtime_sweep_results.json"
            with p.open() as f:
                part = json.load(f)
            for row in part:
                key = row["skill"]
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        rows = sorted(rows, key=lambda x: x["skill"])
        write_reports(args.out_dir, rows)
        print(json.dumps({"out_dir": str(args.out_dir), "skills": len(rows), "mode": "merge"}, indent=2))
        return 0

    registry = load_registry(args.behavior_dir / "OmniGibson/omnigibson/learning/utils/segment_skill_metric_registry.py")
    samples = pick_representative_skills(registry, args.demo_data_path)
    if args.skills.strip():
        wanted = {s.strip().lower() for s in args.skills.split(",") if s.strip()}
        samples = [s for s in samples if s["skill"] in wanted]
        if not samples:
            raise RuntimeError(f"no representative samples found for requested skills: {sorted(wanted)}")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["task_name"]].append(sample)

    results: List[Dict[str, Any]] = []
    for task_offset, task_name in enumerate(sorted(grouped)):
        port = args.port_base + task_offset
        server_proc: Optional[subprocess.Popen[str]] = None
        try:
            server_proc = start_server(
                task_name=task_name,
                port=port,
                gpu_id=args.gpu_id,
                ckpt_dir=args.ckpt_dir,
                config_name=args.config_name,
                openpi_env=args.openpi_env,
                behavior_dir=args.behavior_dir,
                out_dir=args.out_dir,
            )
            wait_for_server(port, args.server_ready_timeout)
            for sample in grouped[task_name]:
                print(f"\n--- Evaluating skill: {sample['skill']} ---")
                row = run_segment_eval(
                    sample=sample,
                    port=port,
                    gpu_id=args.gpu_id,
                    behavior_env=args.behavior_env,
                    behavior_dir=args.behavior_dir,
                    demo_data_path=args.demo_data_path,
                    rawdata_path=args.rawdata_path,
                    out_dir=args.out_dir,
                    default_max_steps=args.max_steps,
                )
                results.append(row)
        finally:
            stop_process(server_proc)

    results = sorted(results, key=lambda x: x["skill"])
    write_reports(args.out_dir, results)
    print(json.dumps({"out_dir": str(args.out_dir), "skills": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
