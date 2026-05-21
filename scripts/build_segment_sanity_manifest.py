#!/usr/bin/env python3

"""Build a lightweight sanity manifest for per-segment skill metric validation.

This script scans one or more `segment_eval_runs/<run_dir>` directories and outputs:
- `sanity/sanity_manifest.jsonl`
- `sanity/sanity_manifest.csv`

Each row references:
- metrics json path
- start/end restore rgb (if present)
- final rollout rgb (if present)
- video path (if present)

The manifest is designed for quick human/LLM multimodal review.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    return bool(x)


def iter_metric_paths(run_dir: Path) -> Iterable[Path]:
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        return
    for p in raw_dir.rglob("*.json"):
        if p.parent.name != "metrics":
            continue
        if not p.name.startswith("segment_eval_"):
            continue
        yield p


def first_video_path(skill_dir: Path) -> Optional[str]:
    video_dir = skill_dir / "videos"
    if not video_dir.exists():
        return None
    videos = sorted(video_dir.glob("*.mp4"))
    return str(videos[0]) if videos else None


def resolve_review_path(metrics: Dict[str, Any], skill_dir: Path, key: str, fallback_name: str) -> Optional[str]:
    review_artifacts = metrics.get("review_artifacts") or {}
    val = review_artifacts.get(key)
    if val:
        p = Path(str(val))
        if p.exists():
            return str(p)
    # fallback: conventional location under per-segment output dir
    candidate = skill_dir / "review" / fallback_name
    if candidate.exists():
        return str(candidate)
    return None


def build_row(metrics_path: Path) -> Dict[str, Any]:
    metrics = read_json(metrics_path)
    skill_dir = metrics_path.parent.parent
    run_dir = metrics_path.parents[5]
    task_name = str(metrics.get("task_name") or "")
    demo_id = str(metrics.get("demo_id") or "")
    segment_idx = metrics.get("segment_idx")
    try:
        segment_idx = int(segment_idx) if segment_idx is not None else None
    except Exception:
        segment_idx = None
    skill = str(metrics.get("segment_desc") or "").strip().lower()

    start_obs = resolve_review_path(metrics, skill_dir, "start_restore_rgb", "start_restore.png")
    end_obs = resolve_review_path(metrics, skill_dir, "end_restore_rgb", "end_restore.png")
    final_obs = resolve_review_path(metrics, skill_dir, "final_rollout_rgb", "final_rollout.png")

    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "task_name": task_name,
        "demo_id": demo_id,
        "segment_idx": segment_idx,
        "skill": skill,
        "metric_success": safe_bool(metrics.get("success")),
        "result_type": str(metrics.get("result_type") or ""),
        "metric_family": (metrics.get("predicate_debug") or {}).get("metric_family"),
        "metrics_path": str(metrics_path),
        "skill_dir": str(skill_dir),
        "start_obs_rgb": start_obs,
        "end_obs_rgb": end_obs,
        "final_obs_rgb": final_obs,
        "video_path": first_video_path(skill_dir),
    }
    return row


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def summarize_missing(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_bucket: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        bucket = f"{row.get('skill')}|{row.get('result_type')}"
        for k in ("start_obs_rgb", "end_obs_rgb", "final_obs_rgb", "video_path"):
            if not row.get(k):
                by_bucket[bucket][k] += 1
        by_bucket[bucket]["total"] += 1

    summary_rows = []
    for bucket, cnt in sorted(by_bucket.items(), key=lambda item: (-item[1]["final_obs_rgb"], -item[1]["total"], item[0])):
        skill, result_type = bucket.split("|", 1)
        summary_rows.append(
            {
                "skill": skill,
                "result_type": result_type,
                "total": cnt["total"],
                "missing_start_obs": cnt["start_obs_rgb"],
                "missing_end_obs": cnt["end_obs_rgb"],
                "missing_final_obs": cnt["final_obs_rgb"],
                "missing_video": cnt["video_path"],
            }
        )
    return {"buckets": summary_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sanity manifest (start/end/final obs + metrics) for segment review.")
    parser.add_argument("--run-dir", action="append", required=True, help="segment_eval_runs/<run_dir> (repeatable)")
    parser.add_argument("--out-subdir", default="sanity", help="Output subdir name under each run_dir (default: sanity)")
    args = parser.parse_args()

    run_dirs = [Path(p).resolve() for p in args.run_dir]
    for run_dir in run_dirs:
        metric_paths = sorted(iter_metric_paths(run_dir))
        rows = [build_row(p) for p in metric_paths]
        rows.sort(key=lambda r: (str(r.get("skill")), str(r.get("task_name")), str(r.get("demo_id")), int(r.get("segment_idx") or 0)))

        out_dir = run_dir / args.out_subdir
        jsonl_path = out_dir / "sanity_manifest.jsonl"
        csv_path = out_dir / "sanity_manifest.csv"
        summary_path = out_dir / "missing_summary.json"

        write_jsonl(jsonl_path, rows)
        write_csv(csv_path, rows)
        summary = summarize_missing(rows)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps({"run_dir": str(run_dir), "segments": len(rows), "out_dir": str(out_dir)}, indent=2, ensure_ascii=False))
        # Print top missing buckets for quick signal
        top = (summary.get("buckets") or [])[:10]
        if top:
            print("Top missing buckets (first 10):")
            for item in top:
                print(
                    f"- skill={item['skill']}, result_type={item['result_type']}, total={item['total']}, "
                    f"missing_final={item['missing_final_obs']}, missing_start={item['missing_start_obs']}, missing_video={item['missing_video']}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

