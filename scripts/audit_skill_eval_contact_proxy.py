#!/usr/bin/env python3
"""Audit contact/effect proxy skill metrics and A/B segment-level flips.

This read-only tool scans skill-eval metrics JSON files, computes predicate trace
streak evidence for contact proxy metrics, and optionally joins two runs by
`skill|task|demo|segment` to show success/failure flips.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

SEGMENT_FIELDS = [
    "run_label",
    "run_dir",
    "job_key",
    "task_name",
    "demo_id",
    "segment_idx",
    "skill",
    "metric_family",
    "success_rule",
    "predicate_count",
    "primary_predicate",
    "primary_predicate_name",
    "primary_predicate_args_json",
    "success",
    "result_type",
    "start_all_satisfied",
    "require_unsatisfied_at_start",
    "final_step",
    "max_steps",
    "predicate_window_mode",
    "combine_mode",
    "min_consecutive",
    "rollout_attempted",
    "termination_reason",
    "env_termination_reason",
    "env_done_success",
    "rollout_terminated",
    "rollout_truncated",
    "trace_len",
    "satisfied_step_count",
    "satisfied_fraction",
    "first_satisfied_step",
    "first_window_satisfied_step",
    "max_streak",
    "final_streak",
    "last_step_satisfied",
    "window_reached",
    "template_start_all_satisfied",
    "template_end_all_satisfied",
    "metrics_path",
    "segment_log",
    "video_path",
    "start_restore_rgb",
    "end_restore_rgb",
    "final_rollout_rgb",
]

FLIP_FIELDS = [
    "job_key",
    "task_name",
    "demo_id",
    "segment_idx",
    "skill",
    "metric_family",
    "a_label",
    "b_label",
    "a_present",
    "b_present",
    "success_flip",
    "a_success",
    "b_success",
    "a_result_type",
    "b_result_type",
    "result_type_changed",
    "a_first_satisfied_step",
    "b_first_satisfied_step",
    "delta_first_satisfied_b_minus_a",
    "a_first_window_satisfied_step",
    "b_first_window_satisfied_step",
    "delta_first_window_b_minus_a",
    "a_max_streak",
    "b_max_streak",
    "delta_max_streak_b_minus_a",
    "a_final_streak",
    "b_final_streak",
    "delta_final_streak_b_minus_a",
    "a_trace_len",
    "b_trace_len",
    "a_final_step",
    "b_final_step",
    "a_metrics_path",
    "b_metrics_path",
    "a_video_path",
    "b_video_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", default=[], help="Run dir for single/multi-run scan; repeatable")
    parser.add_argument("--a", default=None, help="A run dir for A/B mode")
    parser.add_argument("--b", default=None, help="B run dir for A/B mode")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument(
        "--metric-family",
        action="append",
        default=None,
        help=(
            "Metric family filter. Repeat or pass comma-separated values; "
            "defaults to contact_effect_proxy."
        ),
    )
    parser.add_argument("--skills", default="", help="Optional comma-separated skill filter")
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--flip-csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=80)
    return parser.parse_args()


def parse_filter(text: str) -> set[str]:
    return {chunk.strip().lower() for chunk in text.split(",") if chunk.strip()}


def parse_metric_families(values: list[str] | None) -> set[str]:
    if not values:
        return {"contact_effect_proxy"}
    families: set[str] = set()
    for value in values:
        families.update(parse_filter(value))
    return families


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_metric_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("raw/*/demo_*/skill_*/metrics/*.json"))


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def get_metric_family(metrics: dict[str, Any]) -> str:
    debug = metrics.get("predicate_debug") or {}
    if debug.get("metric_family"):
        return str(debug.get("metric_family"))
    for spec in metrics.get("predicate_spec") or []:
        params = spec.get("params") or {}
        if params.get("metric_family"):
            return str(params.get("metric_family"))
    return ""


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


def first_existing_video(skill_dir: Path) -> str:
    videos = sorted((skill_dir / "videos").glob("*.mp4"))
    return str(videos[0]) if videos else ""


def aggregate_trace_step(step_items: Any, combine_mode: str) -> bool:
    if not isinstance(step_items, list) or not step_items:
        return False
    values = [bool(item.get("satisfied")) for item in step_items if isinstance(item, dict)]
    if not values:
        return False
    if combine_mode == "any_of":
        return any(values)
    return all(values)


def compute_trace_stats(metrics: dict[str, Any], min_consecutive: int) -> dict[str, Any]:
    rollout = metrics.get("rollout") or {}
    debug = metrics.get("predicate_debug") or {}
    combine_mode = str(rollout.get("combine_mode") or debug.get("combine_mode") or "all_of")
    window_mode = str(rollout.get("predicate_window_mode") or "consecutive")
    trace = metrics.get("predicate_trace") or []
    first_satisfied_step = None
    first_window_satisfied_step = None
    current_streak = 0
    max_streak = 0
    satisfied_step_count = 0
    last_step_satisfied = None
    for step_idx, step_items in enumerate(trace):
        ok = aggregate_trace_step(step_items, combine_mode)
        last_step_satisfied = ok
        if ok:
            satisfied_step_count += 1
            if first_satisfied_step is None:
                first_satisfied_step = step_idx
            current_streak += 1
            max_streak = max(max_streak, current_streak)
            if first_window_satisfied_step is None and current_streak >= max(min_consecutive, 1):
                first_window_satisfied_step = step_idx
        else:
            current_streak = 0
    return {
        "trace_len": len(trace),
        "predicate_window_mode": window_mode,
        "combine_mode": combine_mode,
        "min_consecutive": min_consecutive,
        "satisfied_step_count": satisfied_step_count,
        "satisfied_fraction": satisfied_step_count / len(trace) if trace else 0.0,
        "first_satisfied_step": first_satisfied_step,
        "first_window_satisfied_step": first_window_satisfied_step,
        "max_streak": max_streak,
        "final_streak": current_streak,
        "last_step_satisfied": last_step_satisfied,
        "window_reached": first_window_satisfied_step is not None,
    }


def template_all_satisfied(trace: Any, combine_mode: str) -> str:
    if not isinstance(trace, list) or not trace:
        return ""
    return bool_text(aggregate_trace_step(trace, combine_mode))


def extract_primary_predicate(metrics: dict[str, Any]) -> dict[str, str]:
    spec = (metrics.get("predicate_spec") or [{}])[0]
    debug = metrics.get("predicate_debug") or {}
    candidates = debug.get("template_trace_start") or debug.get("template_trace_end") or []
    if not candidates:
        predicate_trace = metrics.get("predicate_trace") or []
        candidates = predicate_trace[0] if predicate_trace else []
    item = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    diagnostics = item.get("diagnostics") or {}
    return {
        "primary_predicate": str(item.get("predicate") or spec.get("name") or ""),
        "primary_predicate_name": str(diagnostics.get("predicate_name") or spec.get("name") or ""),
        "primary_predicate_args_json": json.dumps(
            diagnostics.get("predicate_args") or spec.get("args") or [], ensure_ascii=False
        ),
    }


def build_segment_row(run_dir: Path, run_label: str, metrics_path: Path, metrics: dict[str, Any], min_consecutive: int) -> dict[str, str]:
    debug = metrics.get("predicate_debug") or {}
    rollout = metrics.get("rollout") or {}
    review = metrics.get("review_artifacts") or {}
    skill_dir = metrics_path.parent.parent
    trace_stats = compute_trace_stats(metrics, min_consecutive)
    primary = extract_primary_predicate(metrics)
    row = {
        "run_label": run_label,
        "run_dir": str(run_dir),
        "job_key": make_job_key(metrics),
        "task_name": str(metrics.get("task_name") or ""),
        "demo_id": str(metrics.get("demo_id") or ""),
        "segment_idx": "" if metrics.get("segment_idx") is None else str(metrics.get("segment_idx")),
        "skill": str(metrics.get("segment_desc") or "").strip().lower(),
        "metric_family": get_metric_family(metrics),
        "success_rule": str(debug.get("success_rule") or ""),
        "predicate_count": str(len(metrics.get("predicate_spec") or [])),
        **primary,
        "success": bool_text(metrics.get("success")),
        "result_type": str(metrics.get("result_type") or ""),
        "start_all_satisfied": bool_text(first_non_none(debug.get("start_all_satisfied"), rollout.get("start_all_satisfied"))),
        "require_unsatisfied_at_start": bool_text(
            first_non_none(debug.get("require_unsatisfied_at_start"), rollout.get("require_unsatisfied_at_start"))
        ),
        "final_step": "" if rollout.get("final_step") is None else str(rollout.get("final_step")),
        "max_steps": "" if rollout.get("max_steps") is None else str(rollout.get("max_steps")),
        "rollout_attempted": bool_text(rollout.get("rollout_attempted")),
        "termination_reason": str(rollout.get("termination_reason") or ""),
        "env_termination_reason": str(rollout.get("env_termination_reason") or ""),
        "env_done_success": bool_text(rollout.get("env_done_success")),
        "rollout_terminated": bool_text(rollout.get("terminated")),
        "rollout_truncated": bool_text(rollout.get("truncated")),
        "template_start_all_satisfied": template_all_satisfied(
            debug.get("template_trace_start") or [], str(trace_stats["combine_mode"])
        ),
        "template_end_all_satisfied": template_all_satisfied(
            debug.get("template_trace_end") or [], str(trace_stats["combine_mode"])
        ),
        "metrics_path": str(metrics_path),
        "segment_log": str(skill_dir / "segment_eval.log"),
        "video_path": first_existing_video(skill_dir),
        "start_restore_rgb": str(review.get("start_restore_rgb") or ""),
        "end_restore_rgb": str(review.get("end_restore_rgb") or ""),
        "final_rollout_rgb": str(review.get("final_rollout_rgb") or ""),
    }
    for key, value in trace_stats.items():
        if isinstance(value, float):
            row[key] = f"{value:.6f}"
        else:
            row[key] = bool_text(value) if isinstance(value, bool) or value is None else str(value)
    return row


def collect_segment_rows(
    run_dir: Path,
    run_label: str,
    metric_families: set[str],
    skill_filter: set[str],
    min_consecutive: int,
) -> list[dict[str, str]]:
    rows = []
    for metrics_path in iter_metric_paths(run_dir):
        metrics = load_json(metrics_path)
        if get_metric_family(metrics).lower() not in metric_families:
            continue
        skill = str(metrics.get("segment_desc") or "").strip().lower()
        if skill_filter and skill not in skill_filter:
            continue
        rows.append(build_segment_row(run_dir, run_label, metrics_path, metrics, min_consecutive))
    rows.sort(key=lambda row: (row["skill"], row["task_name"], row["demo_id"], int(row["segment_idx"] or 0)))
    return rows


def to_int_or_none(value: str) -> int | None:
    if value in ("", "None", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def delta_int(b_value: str, a_value: str) -> str:
    b_int = to_int_or_none(b_value)
    a_int = to_int_or_none(a_value)
    if b_int is None or a_int is None:
        return ""
    return f"{b_int - a_int:+d}"


def classify_success_flip(a_row: dict[str, str] | None, b_row: dict[str, str] | None) -> str:
    if a_row is None:
        return "missing_a"
    if b_row is None:
        return "missing_b"
    a_success = a_row.get("success") == "true"
    b_success = b_row.get("success") == "true"
    if not a_success and b_success:
        return "fail_to_success"
    if a_success and not b_success:
        return "success_to_fail"
    return "no_success_flip"


def build_ab_flip_rows(a_rows: list[dict[str, str]], b_rows: list[dict[str, str]], label_a: str, label_b: str) -> list[dict[str, str]]:
    a_by_key = {row["job_key"]: row for row in a_rows}
    b_by_key = {row["job_key"]: row for row in b_rows}
    rows = []
    for key in sorted(set(a_by_key) | set(b_by_key)):
        a_row = a_by_key.get(key)
        b_row = b_by_key.get(key)
        base = a_row or b_row or {}
        rows.append(
            {
                "job_key": key,
                "task_name": base.get("task_name", ""),
                "demo_id": base.get("demo_id", ""),
                "segment_idx": base.get("segment_idx", ""),
                "skill": base.get("skill", ""),
                "metric_family": base.get("metric_family", ""),
                "a_label": label_a,
                "b_label": label_b,
                "a_present": bool_text(a_row is not None),
                "b_present": bool_text(b_row is not None),
                "success_flip": classify_success_flip(a_row, b_row),
                "a_success": a_row.get("success", "") if a_row else "",
                "b_success": b_row.get("success", "") if b_row else "",
                "a_result_type": a_row.get("result_type", "") if a_row else "",
                "b_result_type": b_row.get("result_type", "") if b_row else "",
                "result_type_changed": bool_text(bool(a_row and b_row and a_row.get("result_type") != b_row.get("result_type"))),
                "a_first_satisfied_step": a_row.get("first_satisfied_step", "") if a_row else "",
                "b_first_satisfied_step": b_row.get("first_satisfied_step", "") if b_row else "",
                "delta_first_satisfied_b_minus_a": delta_int(
                    b_row.get("first_satisfied_step", "") if b_row else "",
                    a_row.get("first_satisfied_step", "") if a_row else "",
                ),
                "a_first_window_satisfied_step": a_row.get("first_window_satisfied_step", "") if a_row else "",
                "b_first_window_satisfied_step": b_row.get("first_window_satisfied_step", "") if b_row else "",
                "delta_first_window_b_minus_a": delta_int(
                    b_row.get("first_window_satisfied_step", "") if b_row else "",
                    a_row.get("first_window_satisfied_step", "") if a_row else "",
                ),
                "a_max_streak": a_row.get("max_streak", "") if a_row else "",
                "b_max_streak": b_row.get("max_streak", "") if b_row else "",
                "delta_max_streak_b_minus_a": delta_int(
                    b_row.get("max_streak", "") if b_row else "", a_row.get("max_streak", "") if a_row else ""
                ),
                "a_final_streak": a_row.get("final_streak", "") if a_row else "",
                "b_final_streak": b_row.get("final_streak", "") if b_row else "",
                "delta_final_streak_b_minus_a": delta_int(
                    b_row.get("final_streak", "") if b_row else "", a_row.get("final_streak", "") if a_row else ""
                ),
                "a_trace_len": a_row.get("trace_len", "") if a_row else "",
                "b_trace_len": b_row.get("trace_len", "") if b_row else "",
                "a_final_step": a_row.get("final_step", "") if a_row else "",
                "b_final_step": b_row.get("final_step", "") if b_row else "",
                "a_metrics_path": a_row.get("metrics_path", "") if a_row else "",
                "b_metrics_path": b_row.get("metrics_path", "") if b_row else "",
                "a_video_path": a_row.get("video_path", "") if a_row else "",
                "b_video_path": b_row.get("video_path", "") if b_row else "",
            }
        )

    def sort_key(row: dict[str, str]) -> tuple[int, int, str, str, str, int]:
        priority = {"success_to_fail": 0, "fail_to_success": 1, "missing_a": 2, "missing_b": 3}.get(
            row["success_flip"], 4
        )
        abs_delta = abs(to_int_or_none(row["delta_max_streak_b_minus_a"]) or 0)
        return (priority, -abs_delta, row["skill"], row["task_name"], row["demo_id"], int(row["segment_idx"] or 0))

    rows.sort(key=sort_key)
    return rows


def summarize_segments(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "segments": len(rows),
        "by_run": dict(sorted(Counter(row["run_label"] for row in rows).items())),
        "by_skill": dict(sorted(Counter(row["skill"] for row in rows).items())),
        "by_result_type": dict(sorted(Counter(row["result_type"] for row in rows).items())),
        "success_count": sum(1 for row in rows if row["success"] == "true"),
        "window_reached_count": sum(1 for row in rows if row["window_reached"] == "true"),
        "avg_max_streak": sum(int(row["max_streak"] or 0) for row in rows) / len(rows) if rows else 0.0,
    }


def summarize_flips(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "paired_or_union_segments": len(rows),
        "success_flip_counts": dict(sorted(Counter(row["success_flip"] for row in rows).items())),
        "result_type_changed_count": sum(1 for row in rows if row["result_type_changed"] == "true"),
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_markdown(segment_summary: dict[str, Any], flip_summary: dict[str, Any] | None, flip_rows: list[dict[str, str]], top_k: int) -> str:
    lines = ["# Contact Proxy Audit", "", "## Segment summary", ""]
    lines.append(f"- segments: `{segment_summary['segments']}`")
    lines.append(f"- success_count: `{segment_summary['success_count']}`")
    lines.append(f"- window_reached_count: `{segment_summary['window_reached_count']}`")
    lines.append(f"- avg_max_streak: `{segment_summary['avg_max_streak']:.2f}`")
    lines.append("")
    lines.append("### by_skill")
    lines.append("")
    lines.append("| skill | segments |")
    lines.append("| --- | ---: |")
    for skill, count in segment_summary["by_skill"].items():
        lines.append(f"| {skill} | {count} |")
    if flip_summary is not None:
        lines.extend(["", "## A/B flips", ""])
        lines.append(f"- paired_or_union_segments: `{flip_summary['paired_or_union_segments']}`")
        lines.append(f"- success_flip_counts: `{flip_summary['success_flip_counts']}`")
        lines.append(f"- result_type_changed_count: `{flip_summary['result_type_changed_count']}`")
        lines.append("")
        lines.append(
            "| flip | skill | task | demo | segment | A result | B result | A max | B max | Δmax | A metrics | B metrics |"
        )
        lines.append("| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |")
        lines.extend(
            (
                f"| {row['success_flip']} | {row['skill']} | {row['task_name']} | {row['demo_id']} | "
                f"{row['segment_idx']} | {row['a_result_type']}/{row['a_success']} | "
                f"{row['b_result_type']}/{row['b_success']} | {row['a_max_streak']} | "
                f"{row['b_max_streak']} | {row['delta_max_streak_b_minus_a']} | "
                f"`{row['a_metrics_path']}` | `{row['b_metrics_path']}` |"
            )
            for row in flip_rows[: max(top_k, 0)]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    skill_filter = parse_filter(args.skills)
    metric_families = parse_metric_families(args.metric_family)
    segment_rows: list[dict[str, str]] = []
    flip_rows: list[dict[str, str]] = []
    flip_summary = None

    if args.a and args.b:
        a_dir = Path(args.a).expanduser().resolve()
        b_dir = Path(args.b).expanduser().resolve()
        a_rows = collect_segment_rows(a_dir, args.label_a, metric_families, skill_filter, args.min_consecutive)
        b_rows = collect_segment_rows(b_dir, args.label_b, metric_families, skill_filter, args.min_consecutive)
        segment_rows = a_rows + b_rows
        flip_rows = build_ab_flip_rows(a_rows, b_rows, args.label_a, args.label_b)
        flip_summary = summarize_flips(flip_rows)
    else:
        for run_text in args.run_dir:
            run_dir = Path(run_text).expanduser().resolve()
            segment_rows.extend(
                collect_segment_rows(run_dir, run_dir.name, metric_families, skill_filter, args.min_consecutive)
            )

    segment_summary = summarize_segments(segment_rows)
    if args.csv_out:
        write_csv(args.csv_out.expanduser().resolve(), segment_rows, SEGMENT_FIELDS)
    if args.flip_csv_out:
        write_csv(args.flip_csv_out.expanduser().resolve(), flip_rows, FLIP_FIELDS)
    if args.json_out:
        json_out = args.json_out.expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "tool": "skill-eval-audit-contact-proxy",
                    "metric_families": sorted(metric_families),
                    "segment_summary": segment_summary,
                    "flip_summary": flip_summary,
                    "segment_rows": segment_rows,
                    "flip_rows": flip_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    markdown = render_markdown(segment_summary, flip_summary, flip_rows, args.top_k)
    if args.md_out:
        md_out = args.md_out.expanduser().resolve()
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(markdown)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
