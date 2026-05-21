#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import av
except ImportError:  # pragma: no cover
    av = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip().lower() for chunk in text.split(",") if chunk.strip()]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def iter_metric_paths(run_dirs: Sequence[Path]) -> Iterable[Path]:
    for run_dir in run_dirs:
        yield from sorted(run_dir.glob("raw/*/demo_*/skill_*/metrics/*.json"))


def safe_stem(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def extract_last_video_frame(video_path: Path, out_path: Path) -> Optional[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if av is not None:
        try:
            container = av.open(str(video_path))
            last_rgb = None
            for frame in container.decode(video=0):
                last_rgb = frame.to_ndarray(format="rgb24")
            container.close()
            if last_rgb is not None:
                if cv2 is not None:
                    cv2.imwrite(str(out_path), cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR))
                    return out_path
        except Exception:
            pass

    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    last_bgr = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_bgr = frame
    cap.release()
    if last_bgr is None:
        return None
    cv2.imwrite(str(out_path), last_bgr)
    return out_path


def trace_summary(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for item in trace:
        summary.append(
            {
                "predicate": item.get("predicate"),
                "metric_type": item.get("metric_type"),
                "desired": item.get("desired"),
                "value": item.get("value"),
                "satisfied": item.get("satisfied"),
                "diagnostics": item.get("diagnostics"),
            }
        )
    return summary


def load_review_source_paths(metrics_path: Path, metrics: Dict[str, Any]) -> Dict[str, Optional[str]]:
    skill_dir = metrics_path.parent.parent
    review_artifacts = metrics.get("review_artifacts", {}) or {}
    video_candidates = sorted((skill_dir / "videos").glob("*.mp4"))
    return {
        "metrics_path": str(metrics_path),
        "skill_dir": str(skill_dir),
        "video_path": str(video_candidates[0]) if video_candidates else None,
        "start_restore_rgb": review_artifacts.get("start_restore_rgb"),
        "end_restore_rgb": review_artifacts.get("end_restore_rgb"),
        "final_rollout_rgb": review_artifacts.get("final_rollout_rgb"),
    }


def build_review_payload(metrics: Dict[str, Any], source_paths: Dict[str, Optional[str]]) -> Dict[str, Any]:
    predicate_trace = metrics.get("predicate_trace") or []
    last_trace = predicate_trace[-1] if predicate_trace else []
    return {
        "task_name": metrics.get("task_name"),
        "demo_id": metrics.get("demo_id"),
        "segment_idx": metrics.get("segment_idx"),
        "segment_desc": metrics.get("segment_desc"),
        "frame_duration": metrics.get("frame_duration"),
        "success": metrics.get("success"),
        "result_type": metrics.get("result_type"),
        "metric_family": (metrics.get("predicate_debug") or {}).get("metric_family"),
        "predicate_spec": metrics.get("predicate_spec") or [],
        "template_trace_start": trace_summary((metrics.get("predicate_debug") or {}).get("template_trace_start") or []),
        "template_trace_end": trace_summary((metrics.get("predicate_debug") or {}).get("template_trace_end") or []),
        "rollout_final_trace": trace_summary(last_trace),
        "rollout": metrics.get("rollout") or {},
        "source_paths": source_paths,
        "human_review": {
            "human_judgement": "",
            "human_reason": "",
            "issue_bucket": "",
        },
    }


def stratified_select(rows: List[Dict[str, Any]], sample_limit: int, holdout_limit: int, seed: int) -> List[Dict[str, Any]]:
    if sample_limit <= 0 and holdout_limit <= 0:
        for row in rows:
            row["review_split"] = "discovery"
        return rows

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = f"{row.get('result_type')}|{int(bool(row.get('success')))}"
        buckets[bucket].append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda item: (item["task_name"], item["demo_id"], item["segment_idx"]))

    rng = random.Random(seed)
    ordered_buckets = sorted(buckets)
    selected: List[Dict[str, Any]] = []
    while len(selected) < max(sample_limit, 0):
        progressed = False
        for bucket in ordered_buckets:
            if not buckets[bucket]:
                continue
            pick_idx = 0 if len(buckets[bucket]) == 1 else rng.randrange(len(buckets[bucket]))
            selected.append(buckets[bucket].pop(pick_idx))
            progressed = True
            if len(selected) >= sample_limit:
                break
        if not progressed:
            break
    for row in selected:
        row["review_split"] = "discovery"

    holdout: List[Dict[str, Any]] = []
    while len(holdout) < max(holdout_limit, 0):
        progressed = False
        for bucket in ordered_buckets:
            if not buckets[bucket]:
                continue
            holdout.append(buckets[bucket].pop(0))
            progressed = True
            if len(holdout) >= holdout_limit:
                break
        if not progressed:
            break
    for row in holdout:
        row["review_split"] = "holdout"
    return selected + holdout


def copy_if_exists(src: Optional[str], dst: Path) -> Optional[str]:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def collect_rows(run_dirs: Sequence[Path], skill_filter: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in iter_metric_paths(run_dirs):
        metrics = read_json(metrics_path)
        skill = str(metrics.get("segment_desc", "")).strip().lower()
        if skill_filter and skill not in skill_filter:
            continue
        source_paths = load_review_source_paths(metrics_path, metrics)
        row = {
            "run_dir": str(metrics_path.parents[5]),
            "task_name": metrics.get("task_name"),
            "demo_id": metrics.get("demo_id"),
            "segment_idx": int(metrics.get("segment_idx", 0)),
            "skill": skill,
            "success": metrics.get("success"),
            "result_type": metrics.get("result_type"),
            "metric_family": (metrics.get("predicate_debug") or {}).get("metric_family"),
            "metrics_path": str(metrics_path),
            "source_paths": source_paths,
            "metrics": metrics,
        }
        rows.append(row)
    rows.sort(key=lambda item: (item["skill"], item["task_name"], item["demo_id"], item["segment_idx"]))
    return rows


def write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "review_split",
        "skill",
        "task_name",
        "demo_id",
        "segment_idx",
        "success",
        "result_type",
        "metric_family",
        "run_dir",
        "metrics_path",
        "video_path",
        "start_restore_rgb",
        "end_restore_rgb",
        "final_rollout_rgb",
        "final_rgb",
        "review_payload_path",
        "source_paths_path",
        "human_judgement",
        "human_reason",
        "issue_bucket",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a visual review set for skill metric validation.")
    parser.add_argument("--run-dir", action="append", required=True, help="segment_eval_runs/<run_dir> to scan; repeatable")
    parser.add_argument("--skills", default="", help="comma-separated skill subset")
    parser.add_argument("--samples-per-skill", type=int, default=8)
    parser.add_argument("--holdout-per-skill", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    run_dirs = [Path(path).resolve() for path in args.run_dir]
    skill_filter = parse_csv_strings(args.skills)
    rows = collect_rows(run_dirs, skill_filter)
    if not rows:
        raise RuntimeError("No matching metrics found.")

    review_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["skill"]].append(row)

    for skill, skill_rows in sorted(grouped.items()):
        selected = stratified_select(
            skill_rows,
            sample_limit=args.samples_per_skill,
            holdout_limit=args.holdout_per_skill,
            seed=args.seed,
        )
        review_rows.extend(selected)

    for row in review_rows:
        run_dir = Path(row["run_dir"])
        artifact_dir = (
            run_dir
            / "review"
            / "segments"
            / safe_stem(row["task_name"])
            / f"demo_{row['demo_id']}"
            / f"skill_{int(row['segment_idx']):03d}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        source_paths = dict(row["source_paths"])
        final_rgb = None

        final_rollout_src = source_paths.get("final_rollout_rgb")
        if final_rollout_src:
            copied = copy_if_exists(final_rollout_src, artifact_dir / "final_rollout.png")
            source_paths["final_rollout_rgb_copied"] = copied
            if copied:
                final_rgb = copied

        for key in ("start_restore_rgb", "end_restore_rgb"):
            copied = copy_if_exists(source_paths.get(key), artifact_dir / f"{key}.png")
            if copied:
                source_paths[f"{key}_copied"] = copied

        if final_rgb is None and source_paths.get("video_path"):
            extracted = extract_last_video_frame(Path(source_paths["video_path"]), artifact_dir / "final_rgb.png")
            final_rgb = str(extracted) if extracted is not None else None

        payload = build_review_payload(row["metrics"], source_paths)
        payload_path = artifact_dir / "review_payload.json"
        with payload_path.open("w") as f:
            json.dump(payload, f, indent=2)

        source_paths_path = artifact_dir / "source_paths.json"
        with source_paths_path.open("w") as f:
            json.dump(source_paths, f, indent=2)

        row.update(
            {
                "video_path": source_paths.get("video_path"),
                "start_restore_rgb": source_paths.get("start_restore_rgb") or source_paths.get("start_restore_rgb_copied"),
                "end_restore_rgb": source_paths.get("end_restore_rgb") or source_paths.get("end_restore_rgb_copied"),
                "final_rollout_rgb": source_paths.get("final_rollout_rgb") or source_paths.get("final_rollout_rgb_copied"),
                "final_rgb": final_rgb or "",
                "review_payload_path": str(payload_path),
                "source_paths_path": str(source_paths_path),
                "human_judgement": "",
                "human_reason": "",
                "issue_bucket": "",
            }
        )
        del row["metrics"]
        del row["source_paths"]

    manifest_rows = sorted(review_rows, key=lambda item: (item["skill"], item["review_split"], item["task_name"], item["demo_id"], item["segment_idx"]))
    for run_dir in run_dirs:
        per_run_rows = [row for row in manifest_rows if Path(row["run_dir"]) == run_dir]
        if not per_run_rows:
            continue
        review_dir = run_dir / "review"
        write_manifest(review_dir / "review_manifest.csv", per_run_rows)
        with (review_dir / "review_manifest.json").open("w") as f:
            json.dump(per_run_rows, f, indent=2)

    print(
        json.dumps(
            {
                "run_dirs": [str(path) for path in run_dirs],
                "review_rows": len(manifest_rows),
                "skills": len({row["skill"] for row in manifest_rows}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
