#!/usr/bin/env python3
"""Audit missing-object diagnostics in BEHAVIOR skill-eval metric traces.

This is a read-only post-processing tool for completed or partial `segment_eval_runs`.
It scans `raw/**/metrics/*.json`, extracts `diagnostics.missing_object` from
template and rollout predicate traces, joins the merged result CSV when present,
and highlights cases where object resolution problems can pollute success or
`pre_satisfied_start` accounting.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "run_dir",
    "job_key",
    "csv_line",
    "task_name",
    "demo_id",
    "segment_idx",
    "skill",
    "result_type",
    "success",
    "start_all_satisfied",
    "rollout_attempted",
    "termination_reason",
    "metric_family",
    "issue_bucket",
    "trace_stage",
    "occurrence_count",
    "first_rollout_step",
    "missing_object",
    "predicate",
    "desired",
    "value",
    "satisfied",
    "metrics_path",
    "video_path",
    "start_restore_rgb",
    "end_restore_rgb",
    "final_rollout_rgb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="segment_eval run directory; repeatable")
    parser.add_argument("--skills", default="", help="Optional comma-separated skill filter, e.g. 'sweep off,chop'")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path for flagged rows")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path for summary + rows")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional Markdown report output path")
    parser.add_argument(
        "--review-manifest-out",
        type=Path,
        default=None,
        help="Optional lightweight review manifest CSV, one row per affected segment",
    )
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Exit with code 2 if missing_object contaminates success or pre_satisfied_start",
    )
    return parser.parse_args()


def parse_skill_filter(text: str) -> set[str]:
    return {chunk.strip().lower() for chunk in text.split(",") if chunk.strip()}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_metric_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("raw/*/demo_*/skill_*/metrics/*.json"))


def make_job_key(metrics: dict[str, Any]) -> str:
    skill = str(metrics.get("segment_desc", ""))
    task = str(metrics.get("task_name", ""))
    demo = str(metrics.get("demo_id", ""))
    segment_idx_raw = metrics.get("segment_idx", "")
    try:
        segment_idx = f"{int(segment_idx_raw):03d}"
    except (TypeError, ValueError):
        segment_idx = str(segment_idx_raw)
    return f"{skill}|{task}|{demo}|{segment_idx}"


def normalize_segment_idx(segment_idx_raw: Any) -> str:
    try:
        return f"{int(segment_idx_raw):03d}"
    except (TypeError, ValueError):
        return str(segment_idx_raw)


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def load_result_csv(run_dir: Path) -> dict[str, dict[str, Any]]:
    csv_path = run_dir / "multinode_skill_results.csv"
    if not csv_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_number, row in enumerate(reader, start=2):
            key = row.get("job_key") or "|".join(
                [
                    row.get("skill", ""),
                    row.get("task_name", ""),
                    row.get("demo_id", ""),
                    normalize_segment_idx(row.get("segment_idx", "")),
                ]
            )
            row["csv_line"] = str(line_number)
            rows[key] = row
    return rows


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def first_existing_video(skill_dir: Path) -> str:
    videos = sorted((skill_dir / "videos").glob("*.mp4"))
    return str(videos[0]) if videos else ""


def get_metric_family(metrics: dict[str, Any]) -> str:
    debug = metrics.get("predicate_debug") or {}
    if debug.get("metric_family"):
        return str(debug.get("metric_family"))
    specs = metrics.get("predicate_spec") or []
    for spec in specs:
        params = spec.get("params") or {}
        if params.get("metric_family"):
            return str(params.get("metric_family"))
    return ""


def collect_trace_items(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Return trace items with normalized stage metadata.

    The rollout trace can be very long, so downstream aggregation deduplicates by
    stage/predicate/missing_object while retaining occurrence counts and first step.
    """

    debug = metrics.get("predicate_debug") or {}
    items: list[dict[str, Any]] = []
    for stage, trace in (
        ("template_start", debug.get("template_trace_start") or []),
        ("template_end", debug.get("template_trace_end") or []),
    ):
        items.extend(
            {"trace_stage": stage, "rollout_step": None, "item": item} for item in trace if isinstance(item, dict)
        )

    predicate_trace = metrics.get("predicate_trace") or []
    for step, trace in enumerate(predicate_trace):
        if not isinstance(trace, list):
            continue
        items.extend(
            {"trace_stage": "rollout", "rollout_step": step, "item": item}
            for item in trace
            if isinstance(item, dict)
        )
    return items


def extract_missing_object(item: dict[str, Any]) -> str:
    diagnostics = item.get("diagnostics") or {}
    value = diagnostics.get("missing_object")
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(chunk) for chunk in value)
    return str(value)


def classify_issue(metrics: dict[str, Any], *, has_missing: bool) -> str:
    if not has_missing:
        return ""
    result_type = str(metrics.get("result_type") or "")
    success = bool(metrics.get("success"))
    debug = metrics.get("predicate_debug") or {}
    rollout = metrics.get("rollout") or {}
    start_all_satisfied = bool(debug.get("start_all_satisfied") or rollout.get("start_all_satisfied"))
    if success or result_type == "predicate_satisfied":
        return "invalid_success_missing_object"
    if result_type == "pre_satisfied_start" or start_all_satisfied:
        return "invalid_pre_satisfied_start_missing_object"
    if result_type == "timeout":
        return "invalid_timeout_missing_object"
    if result_type == "env_terminated":
        return "invalid_env_terminated_missing_object"
    return "invalid_other_missing_object"


def build_flag_rows(run_dir: Path, metrics_path: Path, metrics: dict[str, Any], csv_row: dict[str, Any] | None) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for trace_item in collect_trace_items(metrics):
        item = trace_item["item"]
        missing_object = extract_missing_object(item)
        if not missing_object:
            continue
        key = (str(trace_item["trace_stage"]), str(item.get("predicate") or ""), missing_object)
        group = grouped.setdefault(
            key,
            {
                "trace_stage": trace_item["trace_stage"],
                "predicate": item.get("predicate") or "",
                "missing_object": missing_object,
                "desired": item.get("desired"),
                "value": item.get("value"),
                "satisfied": item.get("satisfied"),
                "occurrence_count": 0,
                "first_rollout_step": None,
            },
        )
        group["occurrence_count"] += 1
        step = trace_item.get("rollout_step")
        if step is not None and group["first_rollout_step"] is None:
            group["first_rollout_step"] = step

    if not grouped:
        return []

    debug = metrics.get("predicate_debug") or {}
    rollout = metrics.get("rollout") or {}
    review_artifacts = metrics.get("review_artifacts") or {}
    skill = str(metrics.get("segment_desc") or "")
    task = str(metrics.get("task_name") or "")
    demo = str(metrics.get("demo_id") or "")
    segment_idx = "" if metrics.get("segment_idx") is None else str(metrics.get("segment_idx"))
    job_key = make_job_key(metrics)
    skill_dir = metrics_path.parent.parent
    issue_bucket = classify_issue(metrics, has_missing=True)

    return [
        {
            "run_dir": str(run_dir),
            "job_key": csv_row.get("job_key", job_key) if csv_row else job_key,
            "csv_line": csv_row.get("csv_line", "") if csv_row else "",
            "task_name": task,
            "demo_id": demo,
            "segment_idx": segment_idx,
            "skill": skill,
            "result_type": str(metrics.get("result_type") or ""),
            "success": bool_text(metrics.get("success")),
            "start_all_satisfied": bool_text(first_non_none(debug.get("start_all_satisfied"), rollout.get("start_all_satisfied"))),
            "rollout_attempted": bool_text(rollout.get("rollout_attempted")),
            "termination_reason": str(rollout.get("termination_reason") or ""),
            "metric_family": get_metric_family(metrics),
            "issue_bucket": issue_bucket,
            "trace_stage": str(group["trace_stage"]),
            "occurrence_count": str(group["occurrence_count"]),
            "first_rollout_step": "" if group["first_rollout_step"] is None else str(group["first_rollout_step"]),
            "missing_object": str(group["missing_object"]),
            "predicate": str(group["predicate"]),
            "desired": bool_text(group.get("desired")),
            "value": bool_text(group.get("value")),
            "satisfied": bool_text(group.get("satisfied")),
            "metrics_path": str(metrics_path),
            "video_path": first_existing_video(skill_dir),
            "start_restore_rgb": str(review_artifacts.get("start_restore_rgb") or ""),
            "end_restore_rgb": str(review_artifacts.get("end_restore_rgb") or ""),
            "final_rollout_rgb": str(review_artifacts.get("final_rollout_rgb") or ""),
        }
        for group in sorted(grouped.values(), key=lambda item: (item["trace_stage"], item["predicate"], item["missing_object"]))
    ]


def collect_rows(run_dirs: list[Path], skill_filter: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        csv_rows = load_result_csv(run_dir)
        for metrics_path in iter_metric_paths(run_dir):
            metrics = load_json(metrics_path)
            skill = str(metrics.get("segment_desc") or "").strip().lower()
            if skill_filter and skill not in skill_filter:
                continue
            csv_row = csv_rows.get(make_job_key(metrics))
            rows.extend(build_flag_rows(run_dir, metrics_path, metrics, csv_row))
    rows.sort(key=lambda row: (row["run_dir"], row["skill"], row["task_name"], row["demo_id"], int(row["segment_idx"] or 0), row["trace_stage"]))
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    segment_keys = {(row["run_dir"], row["job_key"]) for row in rows}
    bucket_counts = Counter(row["issue_bucket"] for row in rows)
    segment_bucket_counts = Counter()
    for segment_rows in group_by_segment(rows).values():
        segment_bucket_counts[segment_rows[0]["issue_bucket"]] += 1
    return {
        "flagged_trace_rows": len(rows),
        "flagged_segments": len(segment_keys),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "segment_bucket_counts": dict(sorted(segment_bucket_counts.items())),
        "by_skill": dict(sorted(Counter(row["skill"] for row in rows).items())),
        "by_result_type": dict(sorted(Counter(row["result_type"] for row in rows).items())),
        "by_missing_object": dict(sorted(Counter(row["missing_object"] for row in rows).items())),
    }


def group_by_segment(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_dir"], row["job_key"])].append(row)
    return grouped


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] = CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_review_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    manifest_fields = [
        "run_dir",
        "job_key",
        "task_name",
        "demo_id",
        "segment_idx",
        "skill",
        "result_type",
        "success",
        "metric_family",
        "issue_bucket",
        "missing_objects",
        "metrics_path",
        "video_path",
        "start_restore_rgb",
        "end_restore_rgb",
        "final_rollout_rgb",
        "human_judgement",
        "human_reason",
        "issue_bucket_human",
    ]
    grouped_rows: list[dict[str, str]] = []
    for segment_rows in group_by_segment(rows).values():
        first = segment_rows[0]
        grouped_rows.append(
            {
                **{field: first.get(field, "") for field in manifest_fields},
                "missing_objects": ",".join(sorted({row["missing_object"] for row in segment_rows})),
                "human_judgement": "",
                "human_reason": "",
                "issue_bucket_human": "",
            }
        )
    grouped_rows.sort(key=lambda row: (row["skill"], row["task_name"], row["demo_id"], int(row["segment_idx"] or 0)))
    write_csv(path, grouped_rows, fieldnames=manifest_fields)


def render_markdown(run_dirs: list[Path], summary: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines: list[str] = []
    lines.append("# Missing Object Audit")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.extend(f"- `{run_dir}`" for run_dir in run_dirs)
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- flagged_trace_rows: `{summary['flagged_trace_rows']}`")
    lines.append(f"- flagged_segments: `{summary['flagged_segments']}`")
    lines.append("")
    lines.append("### Segment bucket counts")
    lines.append("")
    lines.append("| issue_bucket | segments |")
    lines.append("| --- | ---: |")
    for bucket, count in summary["segment_bucket_counts"].items():
        lines.append(f"| {bucket} | {count} |")
    lines.append("")
    lines.append("### Skill counts")
    lines.append("")
    lines.append("| skill | trace_rows |")
    lines.append("| --- | ---: |")
    for skill, count in summary["by_skill"].items():
        lines.append(f"| {skill} | {count} |")
    lines.append("")
    lines.append("## Flagged segments")
    lines.append("")
    lines.append("| run | skill | task | demo | segment | result_type | issue_bucket | missing_objects | metrics_path |")
    lines.append("| --- | --- | --- | --- | ---: | --- | --- | --- | --- |")
    for segment_rows in group_by_segment(rows).values():
        first = segment_rows[0]
        run_name = Path(first["run_dir"]).name
        missing = ",".join(sorted({row["missing_object"] for row in segment_rows}))
        lines.append(
            f"| {run_name} | {first['skill']} | {first['task_name']} | {first['demo_id']} | "
            f"{first['segment_idx']} | {first['result_type']} | {first['issue_bucket']} | {missing} | "
            f"`{first['metrics_path']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dir]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    rows = collect_rows(run_dirs, parse_skill_filter(args.skills))
    summary = summarize(rows)

    if args.csv_out:
        write_csv(args.csv_out.expanduser().resolve(), rows)
    if args.review_manifest_out:
        write_review_manifest(args.review_manifest_out.expanduser().resolve(), rows)
    if args.json_out:
        json_out = args.json_out.expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))

    markdown = render_markdown(run_dirs, summary, rows)
    if args.md_out:
        md_out = args.md_out.expanduser().resolve()
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(markdown)
    else:
        print(markdown)

    invalid_buckets = {"invalid_success_missing_object", "invalid_pre_satisfied_start_missing_object"}
    if args.fail_on_invalid and any(row["issue_bucket"] in invalid_buckets for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
