#!/usr/bin/env python3
"""Monitor segment_eval_runs progress and print completed skill list periodically."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor BEHAVIOR skill-eval worker progress and print completed skills.",
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to a segment_eval_runs/<run_dir>. Defaults to the latest run under segment_eval_runs.",
    )
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=300,
        help="Refresh interval in seconds. Default: 300.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used when auto-discovering the latest run.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable status snapshot after each refresh.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the machine-readable JSON snapshot instead of the human text view.",
    )
    parser.add_argument(
        "--emit-missing-jobs",
        action="store_true",
        help="Include missing job keys and missing skill counts in the JSON snapshot.",
    )
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        root = args.repo_root.resolve() / "segment_eval_runs"
        candidates = [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "manifest.json").exists() and (path / "worker_plan.csv").exists()
        ]
        if not candidates:
            raise FileNotFoundError(f"No segment_eval_runs found under {root}")
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_result_jsonl_paths(results_dir: Path) -> list[Path]:
    return sorted(list(results_dir.glob("worker_*.jsonl")) + list(results_dir.glob("persistent_worker_*.jsonl")))


def _worker_rank_token(path: Path) -> str:
    stem = path.stem
    for prefix in ("persistent_worker_", "worker_"):
        if stem.startswith(prefix):
            tail = stem[len(prefix) :]
            return tail.split(".", 1)[0]
    return stem


def load_all_result_rows(run_dir: Path) -> list[dict[str, Any]]:
    results_dir = run_dir / "worker_results"
    deduped: dict[str, dict[str, Any]] = {}
    if not results_dir.exists():
        return []
    for path in iter_result_jsonl_paths(results_dir):
        for row in load_jsonl_rows(path):
            deduped[row["job_key"]] = row
    return [deduped[key] for key in sorted(deduped)]


def load_planned_skill_counts(run_dir: Path) -> Counter[str]:
    planned_path = run_dir / "planned_skill_coverage.csv"
    counts: Counter[str] = Counter()
    if not planned_path.exists():
        return counts
    with planned_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["skill"]] += int(row.get("segment_count", 0) or 0)
    return counts


def load_worker_status_counts(run_dir: Path) -> tuple[int, int]:
    status_dir = run_dir / "worker_status"
    if not status_dir.exists():
        return 0, 0
    started_names = {_worker_rank_token(path) for path in status_dir.glob("worker_*.started.json")}
    started_names.update(_worker_rank_token(path) for path in status_dir.glob("persistent_worker_*.jsonl"))
    done_names = {_worker_rank_token(path) for path in status_dir.glob("worker_*.done.json")}
    done_names.update(_worker_rank_token(path) for path in status_dir.glob("persistent_worker_*.done.json"))
    started = len(started_names)
    done = len(done_names)
    return started, done


def load_planned_jobs(run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = load_json(manifest_path)
    return list(manifest.get("jobs", []) or [])


def build_status_snapshot(run_dir: Path, planned_jobs: list[dict[str, Any]], planned_counts: Counter[str]) -> dict[str, Any]:
    rows = load_all_result_rows(run_dir)
    started_workers, done_workers = load_worker_status_counts(run_dir)

    completed_counts: Counter[str] = Counter(row.get("skill", "") for row in rows)
    completed_job_keys = {row["job_key"] for row in rows if "job_key" in row}
    planned_by_key = {row["job_key"]: row for row in planned_jobs if "job_key" in row}
    missing_job_keys = sorted(set(planned_by_key) - completed_job_keys)
    missing_skill_counts: Counter[str] = Counter(str(planned_by_key[key].get("skill", "")) for key in missing_job_keys)
    result_type_counts: Counter[str] = Counter(str(row.get("result_type")) for row in rows)

    runtime_ok = sum(int(bool(row.get("runtime_ok"))) for row in rows)
    success = sum(int(bool(row.get("success"))) for row in rows)
    completed_jobs = len(rows)
    planned_job_count = len(planned_jobs)
    completed_skill_total = len({skill for skill in completed_counts if skill})

    return {
        "run_dir": str(run_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers_started": started_workers,
        "workers_done": done_workers,
        "planned_jobs": planned_job_count,
        "completed_jobs": completed_jobs,
        "missing_jobs": len(missing_job_keys),
        "runtime_ok": runtime_ok,
        "policy_success_raw": success,
        "planned_skills": len(planned_counts),
        "completed_skills": completed_skill_total,
        "completed_skill_counts": dict(sorted(completed_counts.items())),
        "planned_skill_counts": dict(sorted(planned_counts.items())),
        "missing_skill_counts": dict(sorted(missing_skill_counts.items())),
        "result_type_counts": dict(sorted(result_type_counts.items())),
        "completed_job_keys": sorted(completed_job_keys),
        "missing_job_keys": missing_job_keys,
    }


def write_json_snapshot(path: Path, snapshot: dict[str, Any], *, emit_missing_jobs: bool) -> None:
    payload = dict(snapshot)
    if not emit_missing_jobs:
        payload.pop("missing_job_keys", None)
        payload.pop("completed_job_keys", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def format_skill_list(completed_counts: Counter[str], planned_counts: Counter[str]) -> str:
    if not completed_counts:
        return "  - <none yet>"
    lines = []
    for skill in sorted(completed_counts):
        completed = completed_counts[skill]
        planned = planned_counts.get(skill)
        suffix = f"{completed}/{planned}" if planned else str(completed)
        lines.append(f"  - {skill}: {suffix}")
    return "\n".join(lines)


def format_recent_completions(new_rows: list[dict[str, Any]]) -> str:
    if not new_rows:
        return "  - <none>"
    lines = []
    for row in sorted(new_rows, key=lambda item: (item.get("worker_rank", -1), item["job_key"])):
        outcome = "success" if row.get("success") else ("runtime_ok" if row.get("runtime_ok") else "failed")
        lines.append(
            f"  - worker_{int(row.get('worker_rank', -1)):03d} | "
            f"{row['skill']} | {row['task_name']} | demo={row['demo_id']} | {outcome}"
        )
    return "\n".join(lines)


def print_snapshot(
    *,
    run_dir: Path,
    planned_jobs: list[dict[str, Any]],
    planned_counts: Counter[str],
    previous_job_keys: set[str],
) -> set[str]:
    rows = load_all_result_rows(run_dir)
    started_workers, done_workers = load_worker_status_counts(run_dir)

    completed_counts: Counter[str] = Counter(row["skill"] for row in rows)
    completed_job_keys = {row["job_key"] for row in rows}
    new_job_keys = completed_job_keys - previous_job_keys
    new_rows = [row for row in rows if row["job_key"] in new_job_keys]

    runtime_ok = sum(int(bool(row.get("runtime_ok"))) for row in rows)
    success = sum(int(bool(row.get("success"))) for row in rows)
    completed_jobs = len(rows)
    planned_skill_total = len(planned_counts)
    completed_skill_total = len(completed_counts)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 72, flush=True)
    print(f"[{now}] Skill Eval Progress", flush=True)
    print(f"run_dir:           {run_dir}", flush=True)
    print(f"workers:           started={started_workers} done={done_workers}", flush=True)
    print(f"jobs:              completed={completed_jobs}/{len(planned_jobs)}", flush=True)
    print(f"runtime_ok:        {runtime_ok}", flush=True)
    print(f"policy_success:    {success}", flush=True)
    print(f"completed skills:  {completed_skill_total}/{planned_skill_total}", flush=True)
    print("completed skill list:", flush=True)
    print(format_skill_list(completed_counts, planned_counts), flush=True)
    print("new completions since last check:", flush=True)
    print(format_recent_completions(new_rows), flush=True)
    return completed_job_keys


def main() -> int:
    args = parse_args()
    if args.interval_sec <= 0:
        raise ValueError("--interval-sec must be positive")

    run_dir = resolve_run_dir(args)
    planned_jobs = load_planned_jobs(run_dir)
    planned_counts = load_planned_skill_counts(run_dir)
    previous_job_keys: set[str] = set()

    while True:
        snapshot = build_status_snapshot(run_dir, planned_jobs, planned_counts)
        if args.json_out:
            write_json_snapshot(args.json_out.expanduser().resolve(), snapshot, emit_missing_jobs=args.emit_missing_jobs)
        if args.print_json:
            payload = dict(snapshot)
            if not args.emit_missing_jobs:
                payload.pop("missing_job_keys", None)
                payload.pop("completed_job_keys", None)
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            previous_job_keys = set(snapshot.get("completed_job_keys", []))
        else:
            previous_job_keys = print_snapshot(
                run_dir=run_dir,
                planned_jobs=planned_jobs,
                planned_counts=planned_counts,
                previous_job_keys=previous_job_keys,
            )
        if args.once:
            return 0
        print(f"(refresh in {args.interval_sec}s; Ctrl+C to stop)", flush=True)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)
