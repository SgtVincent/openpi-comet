#!/usr/bin/env python3
"""
Watch a skill sweep output directory; once runtime_sweep_results.csv appears,
compare it with a previous CSV and print a concise diff summary.

Default paths are set to the shard2-after-fix workflow used in this repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


FIELDS_TO_DIFF = [
    "runtime_ok",
    "success",
    "result_type",
    "metric_family",
    "final_step",
]


def _load_csv(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            skill = (r.get("skill") or "").strip()
            if not skill:
                continue
            rows[skill] = dict(r)
    return rows


def _summary(rows: Dict[str, Dict[str, str]]) -> Dict[str, object]:
    runtime_pass = sum(r.get("runtime_ok") == "True" for r in rows.values())
    policy_pass = sum(r.get("success") == "True" for r in rows.values())
    by_result = Counter(r.get("result_type") or "" for r in rows.values())
    return {
        "skills": len(rows),
        "runtime_pass": runtime_pass,
        "policy_pass": policy_pass,
        "by_result": dict(by_result),
    }


def _diff(old_rows: Dict[str, Dict[str, str]], new_rows: Dict[str, Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[str], List[str]]:
    changes: List[Dict[str, object]] = []
    removed: List[str] = []
    added: List[str] = []

    old_skills = set(old_rows)
    new_skills = set(new_rows)
    for s in sorted(old_skills - new_skills):
        removed.append(s)
    for s in sorted(new_skills - old_skills):
        added.append(s)

    for skill in sorted(old_skills & new_skills):
        o = old_rows[skill]
        n = new_rows[skill]
        d = {}
        for k in FIELDS_TO_DIFF:
            if (o.get(k) or "") != (n.get(k) or ""):
                d[k] = {"old": o.get(k), "new": n.get(k)}
        if d:
            changes.append(
                {
                    "skill": skill,
                    "diff": d,
                    "old_metrics_path": o.get("metrics_path"),
                    "new_metrics_path": n.get("metrics_path"),
                    "old_segment_log": o.get("segment_log"),
                    "new_segment_log": n.get("segment_log"),
                }
            )
    return changes, added, removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=Path("/tmp/skill_runtime_shard2_after_fix"),
        help="Directory that will contain runtime_sweep_results.csv",
    )
    parser.add_argument(
        "--old-csv",
        type=Path,
        default=Path("/tmp/skill_runtime_sweep_rerun_shard2/runtime_sweep_results.csv"),
        help="Baseline CSV to diff against",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="0 means no timeout")
    args = parser.parse_args()

    watch_csv = args.watch_dir / "runtime_sweep_results.csv"
    start = time.time()
    while True:
        if watch_csv.exists():
            break
        if args.timeout_seconds and (time.time() - start) > args.timeout_seconds:
            print(f"[watch] timeout: {watch_csv} not found after {args.timeout_seconds}s")
            return 2
        time.sleep(args.poll_seconds)

    print(f"[watch] detected: {watch_csv}")
    new_rows = _load_csv(watch_csv)
    new_sum = _summary(new_rows)

    out = {
        "watch_dir": str(args.watch_dir),
        "new_csv": str(watch_csv),
        "old_csv": str(args.old_csv),
        "new_summary": new_sum,
    }

    if args.old_csv.exists():
        old_rows = _load_csv(args.old_csv)
        old_sum = _summary(old_rows)
        changes, added, removed = _diff(old_rows, new_rows)
        out.update(
            {
                "old_summary": old_sum,
                "changed_skills": changes,
                "added_skills": added,
                "removed_skills": removed,
            }
        )
        print("[diff] summary")
        print(json.dumps({"old": old_sum, "new": new_sum}, indent=2))
        print(f"[diff] changed_skills: {len(changes)}")
        for c in changes:
            print(f"  - {c['skill']}: {c['diff']}")
        if added:
            print(f"[diff] added_skills: {added}")
        if removed:
            print(f"[diff] removed_skills: {removed}")
    else:
        print(f"[diff] baseline csv not found: {args.old_csv} (skipping diff)")
        print("[new] summary")
        print(json.dumps({"new": new_sum}, indent=2))

    diff_path = args.watch_dir / "diff_vs_previous.json"
    diff_path.write_text(json.dumps(out, indent=2))
    print(f"[diff] wrote: {diff_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

