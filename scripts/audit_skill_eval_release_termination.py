#!/usr/bin/env python3
"""Audit early env_terminated samples for release/attach-style skill evals.

The tool is intentionally read-only. It scans run metric JSON files, finds early
`env_terminated` rows (default: `skill=release`, `final_step<=1`), and emits a
compact evidence table with predicate traces, review assets, and segment logs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "run_dir",
    "job_key",
    "task_name",
    "demo_id",
    "segment_idx",
    "skill",
    "result_type",
    "success",
    "metric_family",
    "final_step",
    "max_steps",
    "termination_reason",
    "rollout_attempted",
    "env_done_success",
    "env_terminated_seen",
    "first_env_terminated_step",
    "first_env_done_success_step",
    "env_terminal_debug_done_info_success",
    "mode_bucket",
    "goal_status",
    "predicate_done",
    "predicate_success",
    "predicate",
    "desired",
    "value",
    "satisfied",
    "diagnostics_json",
    "release_predicate_summary_json",
    "metrics_path",
    "segment_log",
    "video_path",
    "video_frame_count_estimate",
    "video_duration_s_estimate",
    "video_too_short_for_review",
    "start_restore_rgb",
    "end_restore_rgb",
    "final_rollout_rgb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="segment_eval run directory; repeatable")
    parser.add_argument("--skills", default="release", help="Comma-separated skills to audit. Default: release")
    parser.add_argument(
        "--result-type",
        default="env_terminated,env_task_success_before_segment_success",
        help="Comma-separated result types to select.",
    )
    parser.add_argument("--max-final-step", type=int, default=1, help="Select rows with rollout.final_step <= this value")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional Markdown report output path")
    return parser.parse_args()


def parse_filter(text: str) -> set[str]:
    return {chunk.strip().lower() for chunk in text.split(",") if chunk.strip()}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def make_job_key(metrics: dict[str, Any]) -> str:
    try:
        segment_idx = f"{int(metrics.get('segment_idx')):03d}"
    except (TypeError, ValueError):
        segment_idx = str(metrics.get("segment_idx", ""))
    return "|".join(
        [
            str(metrics.get("segment_desc", "")),
            str(metrics.get("task_name", "")),
            str(metrics.get("demo_id", "")),
            segment_idx,
        ]
    )


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def first_existing_video(skill_dir: Path) -> str:
    videos = sorted((skill_dir / "videos").glob("*.mp4"))
    return str(videos[0]) if videos else ""


def select_final_trace_items(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    predicate_trace = metrics.get("predicate_trace") or []
    if predicate_trace and isinstance(predicate_trace[-1], list) and predicate_trace[-1]:
        return [item for item in predicate_trace[-1] if isinstance(item, dict)]
    debug = metrics.get("predicate_debug") or {}
    for key in ("template_trace_end", "template_trace_start"):
        trace = debug.get(key) or []
        if trace:
            return [item for item in trace if isinstance(item, dict)]
    return []


def primary_trace_item(trace_items: list[dict[str, Any]]) -> dict[str, Any]:
    return trace_items[0] if trace_items else {}


def release_predicate_summary(trace_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "predicate": item.get("predicate"),
            "metric_type": item.get("metric_type"),
            "desired": item.get("desired"),
            "value": item.get("value"),
            "satisfied": item.get("satisfied"),
            "diagnostics": item.get("diagnostics") or {},
        }
        for item in trace_items
    ]


def get_combine_mode(metrics: dict[str, Any]) -> str:
    rollout = metrics.get("rollout") or {}
    debug = metrics.get("predicate_debug") or {}
    return str(rollout.get("combine_mode") or debug.get("combine_mode") or "all_of")


def aggregate_goal_status(trace_items: list[dict[str, Any]], combine_mode: str) -> str:
    if not trace_items:
        return ""
    satisfied_flags = [bool(item.get("satisfied")) for item in trace_items]
    if combine_mode == "any_of":
        return "satisfied" if any(satisfied_flags) else "unsatisfied"
    return "satisfied" if all(satisfied_flags) else "unsatisfied"


def get_metric_family(metrics: dict[str, Any]) -> str:
    debug = metrics.get("predicate_debug") or {}
    if debug.get("metric_family"):
        return str(debug.get("metric_family"))
    for spec in metrics.get("predicate_spec") or []:
        params = spec.get("params") or {}
        if params.get("metric_family"):
            return str(params.get("metric_family"))
    return ""


def classify_mode(metrics: dict[str, Any], goal_status: str) -> str:
    rollout = metrics.get("rollout") or {}
    final_step = int(rollout.get("final_step") or 0)
    metric_family = get_metric_family(metrics)
    env_terminal_debug = rollout.get("env_terminal_debug") or {}
    done_info = env_terminal_debug.get("done_info") or {}
    env_task_success = bool(rollout.get("env_done_success") is True or done_info.get("success") is True)
    task_name = str(metrics.get("task_name") or "")
    is_attach_task = task_name.startswith("attach_")
    if (
        final_step <= 1
        and metric_family == "grasp_release"
        and goal_status == "unsatisfied"
        and env_task_success
        and is_attach_task
    ):
        return "attach_task_env_success_release_predicate_unsatisfied"
    if final_step <= 1 and metric_family == "grasp_release" and goal_status == "unsatisfied":
        return "step1_env_terminated_release_predicate_unsatisfied"
    if final_step <= 1:
        return "step1_env_terminated"
    return "env_terminated_metric_unsatisfied"


def collect_rows(run_dirs: list[Path], skills: set[str], result_types: set[str], max_final_step: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        for metrics_path in sorted(run_dir.glob("raw/*/demo_*/skill_*/metrics/*.json")):
            metrics = load_json(metrics_path)
            skill = str(metrics.get("segment_desc") or "").strip().lower()
            if skills and skill not in skills:
                continue
            if result_types and str(metrics.get("result_type") or "") not in result_types:
                continue
            rollout = metrics.get("rollout") or {}
            final_step = int(rollout.get("final_step") or 0)
            if final_step > max_final_step:
                continue

            trace_items = select_final_trace_items(metrics)
            trace_item = primary_trace_item(trace_items)
            combine_mode = get_combine_mode(metrics)
            goal_status = aggregate_goal_status(trace_items, combine_mode)
            skill_dir = metrics_path.parent.parent
            review_artifacts = metrics.get("review_artifacts") or {}
            env_terminal_debug = rollout.get("env_terminal_debug") or {}
            done_info = env_terminal_debug.get("done_info") or {}
            summary_json = json.dumps(release_predicate_summary(trace_items), ensure_ascii=False, sort_keys=True)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "job_key": make_job_key(metrics),
                    "task_name": str(metrics.get("task_name") or ""),
                    "demo_id": str(metrics.get("demo_id") or ""),
                    "segment_idx": str(metrics.get("segment_idx") or ""),
                    "skill": skill,
                    "result_type": str(metrics.get("result_type") or ""),
                    "success": bool_text(metrics.get("success")),
                    "metric_family": get_metric_family(metrics),
                    "final_step": str(final_step),
                    "max_steps": str(rollout.get("max_steps") or ""),
                    "termination_reason": str(rollout.get("termination_reason") or ""),
                    "rollout_attempted": bool_text(rollout.get("rollout_attempted")),
                    "env_done_success": bool_text(rollout.get("env_done_success")),
                    "env_terminated_seen": bool_text(rollout.get("env_terminated_seen")),
                    "first_env_terminated_step": str(rollout.get("first_env_terminated_step") or ""),
                    "first_env_done_success_step": str(rollout.get("first_env_done_success_step") or ""),
                    "env_terminal_debug_done_info_success": bool_text(done_info.get("success")),
                    "mode_bucket": classify_mode(metrics, goal_status),
                    "goal_status": goal_status,
                    "predicate_done": bool_text(trace_item.get("value")),
                    "predicate_success": bool_text(trace_item.get("satisfied")),
                    "predicate": str(trace_item.get("predicate") or ""),
                    "desired": bool_text(trace_item.get("desired")),
                    "value": bool_text(trace_item.get("value")),
                    "satisfied": bool_text(trace_item.get("satisfied")),
                    "diagnostics_json": json.dumps(trace_item.get("diagnostics") or {}, ensure_ascii=False, sort_keys=True),
                    "release_predicate_summary_json": summary_json,
                    "metrics_path": str(metrics_path),
                    "segment_log": str(skill_dir / "segment_eval.log"),
                    "video_path": str(review_artifacts.get("video_path") or first_existing_video(skill_dir)),
                    "video_frame_count_estimate": str(review_artifacts.get("video_frame_count_estimate") or ""),
                    "video_duration_s_estimate": str(review_artifacts.get("video_duration_s_estimate") or ""),
                    "video_too_short_for_review": bool_text(review_artifacts.get("video_too_short_for_review")),
                    "start_restore_rgb": str(review_artifacts.get("start_restore_rgb") or ""),
                    "end_restore_rgb": str(review_artifacts.get("end_restore_rgb") or ""),
                    "final_rollout_rgb": str(review_artifacts.get("final_rollout_rgb") or ""),
                }
            )
    rows.sort(key=lambda row: (row["run_dir"], row["skill"], row["task_name"], row["demo_id"], int(row["segment_idx"] or 0)))
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "flagged_segments": len(rows),
        "by_run": dict(sorted(Counter(Path(row["run_dir"]).name for row in rows).items())),
        "by_skill": dict(sorted(Counter(row["skill"] for row in rows).items())),
        "by_mode_bucket": dict(sorted(Counter(row["mode_bucket"] for row in rows).items())),
        "by_metric_family": dict(sorted(Counter(row["metric_family"] for row in rows).items())),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(run_dirs: list[Path], summary: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines: list[str] = ["# Release Termination Audit", "", "## Inputs", ""]
    lines.extend(f"- `{run_dir}`" for run_dir in run_dirs)
    lines.extend(["", "## Summary", "", f"- flagged_segments: `{summary['flagged_segments']}`", ""])
    lines.extend(["### Mode buckets", "", "| mode_bucket | count |", "| --- | ---: |"])
    for bucket, count in summary["by_mode_bucket"].items():
        lines.append(f"| {bucket} | {count} |")
    lines.extend(["", "## Flagged segments", ""])
    lines.append(
        "| run | skill | task | demo | segment | final_step | env_done_success | env_terminated_seen | "
        "first_env_success_step | done_info_success | video_too_short | predicate_done | predicate_success | "
        "goal_status | mode_bucket | predicate | metrics_path |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.extend(
        f"| {Path(row['run_dir']).name} | {row['skill']} | {row['task_name']} | {row['demo_id']} | "
        f"{row['segment_idx']} | {row['final_step']} | {row['env_done_success']} | "
        f"{row['env_terminated_seen']} | {row['first_env_done_success_step']} | "
        f"{row['env_terminal_debug_done_info_success']} | {row['video_too_short_for_review']} | {row['predicate_done']} | "
        f"{row['predicate_success']} | {row['goal_status']} | {row['mode_bucket']} | {row['predicate']} | "
        f"`{row['metrics_path']}` |"
        for row in rows
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dir]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    rows = collect_rows(run_dirs, parse_filter(args.skills), parse_filter(args.result_type), args.max_final_step)
    summary = summarize(rows)

    if args.csv_out:
        write_csv(args.csv_out.expanduser().resolve(), rows)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
