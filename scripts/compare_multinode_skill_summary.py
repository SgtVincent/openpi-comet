#!/usr/bin/env python3
"""Compare two schema-v2 multinode skill summaries.

Design goals:
- stdlib-only and read-only over existing run outputs
- works with either a run directory or a direct `multinode_skill_summary.json` path
- gracefully reports pending progress when a summary file is not ready yet
- writes Markdown and/or CSV compare artifacts without touching eval logic
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RESULT_ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS = "env_task_success_before_segment_success"
ATTEMPTABLE_RESULT_TYPES = {
    "predicate_satisfied",
    "short_proxy_success",
    "likely_proxy_false_positive",
    "timeout",
    "truncated",
    "env_terminated",
    "short_video_problem",
    RESULT_ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS,
}
METRIC_INVALID_PREFIX = "metric_invalid"
RESULT_METRIC_INVALID_MISSING_OBJECT = "metric_invalid_missing_object"
RESULT_PRE_SATISFIED_START = "pre_satisfied_start"
RESULT_SHORT_PROXY_SUCCESS = "short_proxy_success"
RESULT_LIKELY_PROXY_FALSE_POSITIVE = "likely_proxy_false_positive"
RESULT_SHORT_VIDEO_PROBLEM = "short_video_problem"
SHORT_VIDEO_PROBLEM_STEP_THRESHOLD = 150
TRANSFER_POSE_PROXY_FAMILY = "transfer_pose_proxy"


OVERALL_ROWS: list[tuple[str, str, bool]] = [
    ("planned", "planned_jobs", False),
    ("completed", "completed_jobs", False),
    ("runtime_pass", "runtime_pass", False),
    ("pre_satisfied_start", "pre_satisfied_start", False),
    ("metric_invalid", "metric_invalid", False),
    ("metric_invalid_missing_object", "metric_invalid_missing_object", False),
    ("attemptable", "attemptable_segments", False),
    ("policy_success_attemptable", "policy_success_attemptable", False),
    ("rate", "policy_success_attemptable_rate", True),
    ("policy_success_clean_attemptable", "policy_success_clean_attemptable", False),
    ("clean_rate", "policy_success_clean_attemptable_rate", True),
    ("short_video_problem", "short_video_problem", False),
    (
        "early_metric_activation_review_needed",
        "early_metric_activation_review_needed",
        False,
    ),
    ("short_proxy_success", "short_proxy_success", False),
    ("likely_proxy_false_positive", "likely_proxy_false_positive", False),
    (
        "transfer_pose_proxy_success_unconfirmed",
        "transfer_pose_proxy_success_unconfirmed",
        False,
    ),
    (
        "meaningful_policy_caused_transition",
        "meaningful_policy_caused_transition",
        False,
    ),
    (
        "env_task_success_before_segment_success",
        "env_task_success_before_segment_success",
        False,
    ),
]

SKILL_METRICS: list[tuple[str, str, bool]] = [
    ("segment_count", "segment_count", False),
    ("runtime_pass", "runtime_pass_count", False),
    ("pre_satisfied_start", "pre_satisfied_start_count", False),
    ("metric_invalid", "metric_invalid_count", False),
    ("metric_invalid_missing_object", "metric_invalid_missing_object_count", False),
    ("attemptable", "attemptable_segment_count", False),
    ("policy_success_attemptable", "policy_success_attemptable_count", False),
    ("rate", "policy_success_attemptable_rate", True),
    ("policy_success_clean_attemptable", "policy_success_clean_attemptable_count", False),
    ("clean_rate", "policy_success_clean_attemptable_rate", True),
    ("short_video_problem", "short_video_problem_count", False),
    (
        "early_metric_activation_review_needed",
        "early_metric_activation_review_needed_count",
        False,
    ),
    ("short_proxy_success", "short_proxy_success_count", False),
    ("likely_proxy_false_positive", "likely_proxy_false_positive_count", False),
    (
        "transfer_pose_proxy_success_unconfirmed",
        "transfer_pose_proxy_success_unconfirmed_count",
        False,
    ),
    (
        "meaningful_policy_caused_transition",
        "meaningful_policy_caused_transition_count",
        False,
    ),
    (
        "env_task_success_before_segment_success",
        "env_task_success_before_segment_success_count",
        False,
    ),
]


def _safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def is_short_video_problem_row(row: dict[str, Any]) -> bool:
    if row.get("result_type") != "predicate_satisfied" or not bool(row.get("success")):
        return False
    final_step = row.get("final_step")
    try:
        return final_step is not None and int(final_step) < SHORT_VIDEO_PROBLEM_STEP_THRESHOLD
    except (TypeError, ValueError):
        return False


def is_transfer_pose_proxy_success_unconfirmed(row: dict[str, Any]) -> bool:
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == TRANSFER_POSE_PROXY_FAMILY
    )


def is_early_metric_activation_review_needed(row: dict[str, Any]) -> bool:
    if row.get("result_type") != "predicate_satisfied" or not bool(row.get("success")):
        return False
    if bool(row.get("start_all_satisfied")):
        return False

    early_hits = _safe_int(row.get("early_predicate_satisfied_steps"))
    if early_hits is not None and early_hits > 0:
        return True

    first_hit = _safe_int(row.get("first_predicate_satisfied_step"))
    min_steps = _safe_int(row.get("min_success_steps"))
    return first_hit is not None and min_steps is not None and min_steps > 0 and first_hit < min_steps


def has_meaningful_policy_caused_transition(row: dict[str, Any]) -> bool:
    if row.get("result_type") != "predicate_satisfied" or not bool(row.get("success")):
        return False
    if bool(row.get("start_all_satisfied")):
        return False
    if bool(row.get("early_metric_activation_review_needed")) or is_early_metric_activation_review_needed(row):
        return False

    first_hit = _safe_int(row.get("first_predicate_satisfied_step"))
    min_steps = _safe_int(row.get("min_success_steps"))
    if first_hit is None or min_steps is None or min_steps <= 0:
        return False
    return first_hit >= min_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Run dir or multinode_skill_summary.json for side A")
    parser.add_argument("--b", required=True, help="Run dir or multinode_skill_summary.json for side B")
    parser.add_argument("--label-a", default="A", help="Display label for side A")
    parser.add_argument("--label-b", default="B", help="Display label for side B")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional Markdown output path")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional structured JSON output path")
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many per-skill rows to include in Markdown (CSV always contains all rows)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def classify_result_row(row: dict[str, Any]) -> dict[str, bool]:
    runtime_pass = bool(row.get("runtime_ok"))
    result_type = row.get("result_type")
    pre_satisfied_start = result_type == RESULT_PRE_SATISFIED_START
    short_proxy_success = result_type in {RESULT_SHORT_PROXY_SUCCESS, RESULT_LIKELY_PROXY_FALSE_POSITIVE} or bool(
        row.get("short_proxy_success")
    )
    likely_proxy_false_positive = result_type == RESULT_LIKELY_PROXY_FALSE_POSITIVE or bool(
        row.get("likely_proxy_false_positive")
    )
    transfer_pose_proxy_success_unconfirmed = bool(
        row.get("transfer_pose_proxy_success_unconfirmed")
    ) or is_transfer_pose_proxy_success_unconfirmed(row)
    short_video_problem = (
        result_type == RESULT_SHORT_VIDEO_PROBLEM
        or bool(row.get("short_video_problem"))
        or bool(row.get("metrics_short_video_problem"))
        or is_short_video_problem_row(row)
    )
    early_metric_activation_review_needed = bool(row.get("early_metric_activation_review_needed")) or is_early_metric_activation_review_needed(
        row
    )
    metric_invalid_missing_object = result_type == RESULT_METRIC_INVALID_MISSING_OBJECT
    metric_invalid = str(result_type or "").startswith(METRIC_INVALID_PREFIX)
    rollout_attempted = row.get("rollout_attempted")
    if rollout_attempted is None:
        policy_attempted = result_type in ATTEMPTABLE_RESULT_TYPES
    else:
        policy_attempted = bool(rollout_attempted) and result_type in ATTEMPTABLE_RESULT_TYPES
    attemptable = runtime_pass and policy_attempted and not pre_satisfied_start and not metric_invalid
    policy_success_attemptable = attemptable and bool(row.get("success")) and result_type == "predicate_satisfied"
    policy_success_clean_attemptable = (
        policy_success_attemptable
        and not short_proxy_success
        and not short_video_problem
        and not transfer_pose_proxy_success_unconfirmed
        and not early_metric_activation_review_needed
    )
    meaningful_policy_caused_transition = (
        attemptable
        and (bool(row.get("meaningful_policy_caused_transition")) or has_meaningful_policy_caused_transition(row))
    )
    timeout = attemptable and result_type == "timeout"
    truncated = attemptable and result_type == "truncated"
    env_terminated_metric_unsatisfied = attemptable and result_type == "env_terminated" and not bool(row.get("success"))
    env_task_success_before_segment_success = (
        attemptable and result_type == RESULT_ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS and not bool(row.get("success"))
    )
    metric_unsatisfied_attemptable = attemptable and not policy_success_attemptable
    other_metric_unsatisfied = (
        metric_unsatisfied_attemptable
        and not timeout
        and not truncated
        and not env_terminated_metric_unsatisfied
        and not env_task_success_before_segment_success
        and not likely_proxy_false_positive
    )
    return {
        "runtime_pass": runtime_pass,
        "runtime_fail": not runtime_pass,
        "pre_satisfied_start": runtime_pass and pre_satisfied_start,
        "metric_invalid": runtime_pass and metric_invalid,
        "metric_invalid_missing_object": runtime_pass and metric_invalid_missing_object,
        "attemptable": attemptable,
        "policy_success_attemptable": policy_success_attemptable,
        "policy_success_clean_attemptable": policy_success_clean_attemptable,
        "short_video_problem": runtime_pass and short_video_problem,
        "early_metric_activation_review_needed": runtime_pass and early_metric_activation_review_needed,
        "short_proxy_success": runtime_pass and short_proxy_success,
        "likely_proxy_false_positive": runtime_pass and likely_proxy_false_positive,
        "transfer_pose_proxy_success_unconfirmed": runtime_pass and transfer_pose_proxy_success_unconfirmed,
        "meaningful_policy_caused_transition": runtime_pass and meaningful_policy_caused_transition,
        "timeout": timeout,
        "truncated": truncated,
        "env_terminated_metric_unsatisfied": env_terminated_metric_unsatisfied,
        "env_task_success_before_segment_success": env_task_success_before_segment_success,
        "metric_unsatisfied_attemptable": metric_unsatisfied_attemptable,
        "other_metric_unsatisfied": other_metric_unsatisfied,
    }


def summarize_result_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    classes = [classify_result_row(row) for row in rows]
    runtime_pass = sum(int(c["runtime_pass"]) for c in classes)
    runtime_fail = sum(int(c["runtime_fail"]) for c in classes)
    pre_satisfied_start = sum(int(c["pre_satisfied_start"]) for c in classes)
    metric_invalid = sum(int(c["metric_invalid"]) for c in classes)
    metric_invalid_missing_object = sum(int(c["metric_invalid_missing_object"]) for c in classes)
    attemptable = sum(int(c["attemptable"]) for c in classes)
    policy_success_attemptable = sum(int(c["policy_success_attemptable"]) for c in classes)
    policy_success_clean_attemptable = sum(int(c["policy_success_clean_attemptable"]) for c in classes)
    short_video_problem = sum(int(c["short_video_problem"]) for c in classes)
    early_metric_activation_review_needed = sum(int(c["early_metric_activation_review_needed"]) for c in classes)
    short_proxy_success = sum(int(c["short_proxy_success"]) for c in classes)
    likely_proxy_false_positive = sum(int(c["likely_proxy_false_positive"]) for c in classes)
    transfer_pose_proxy_success_unconfirmed = sum(
        int(c["transfer_pose_proxy_success_unconfirmed"]) for c in classes
    )
    meaningful_policy_caused_transition = sum(int(c["meaningful_policy_caused_transition"]) for c in classes)
    metric_unsatisfied_attemptable = sum(int(c["metric_unsatisfied_attemptable"]) for c in classes)
    timeout = sum(int(c["timeout"]) for c in classes)
    truncated = sum(int(c["truncated"]) for c in classes)
    env_unsat = sum(int(c["env_terminated_metric_unsatisfied"]) for c in classes)
    env_task_success_before_segment_success = sum(
        int(c["env_task_success_before_segment_success"]) for c in classes
    )
    other_unsat = sum(int(c["other_metric_unsatisfied"]) for c in classes)
    success_raw = sum(int(bool(row.get("success"))) for row in rows)
    env_done_success_count = sum(int(row.get("env_done_success") is True) for row in rows)
    rollout_terminated_count = sum(int(row.get("rollout_terminated") is True) for row in rows)
    rollout_truncated_flag_count = sum(int(row.get("rollout_truncated") is True) for row in rows)
    termination_reason_counts = dict(
        sorted(Counter(str(row.get("termination_reason")) for row in rows if row.get("termination_reason") is not None).items())
    )
    env_termination_reason_counts = dict(
        sorted(
            Counter(str(row.get("env_termination_reason")) for row in rows if row.get("env_termination_reason") is not None).items()
        )
    )
    return {
        "segment_count": total,
        "runtime_pass_count": runtime_pass,
        "runtime_fail_count": runtime_fail,
        "runtime_pass_rate": runtime_pass / total if total else 0.0,
        "pre_satisfied_start_count": pre_satisfied_start,
        "metric_invalid_count": metric_invalid,
        "metric_invalid_missing_object_count": metric_invalid_missing_object,
        "attemptable_segment_count": attemptable,
        "policy_success_attemptable_count": policy_success_attemptable,
        "policy_success_attemptable_rate": policy_success_attemptable / attemptable if attemptable else 0.0,
        "policy_success_clean_attemptable_count": policy_success_clean_attemptable,
        "policy_success_clean_attemptable_rate": policy_success_clean_attemptable / attemptable if attemptable else 0.0,
        "short_video_problem_count": short_video_problem,
        "early_metric_activation_review_needed_count": early_metric_activation_review_needed,
        "short_proxy_success_count": short_proxy_success,
        "likely_proxy_false_positive_count": likely_proxy_false_positive,
        "transfer_pose_proxy_success_unconfirmed_count": transfer_pose_proxy_success_unconfirmed,
        "meaningful_policy_caused_transition_count": meaningful_policy_caused_transition,
        "metric_unsatisfied_attemptable_count": metric_unsatisfied_attemptable,
        "timeout_count": timeout,
        "truncated_count": truncated,
        "env_terminated_metric_unsatisfied_count": env_unsat,
        "env_task_success_before_segment_success_count": env_task_success_before_segment_success,
        "other_metric_unsatisfied_count": other_unsat,
        "success_count_raw": success_raw,
        "success_rate_raw_completed": success_raw / total if total else 0.0,
        "predicate_satisfied_count": sum(1 for row in rows if row.get("result_type") == "predicate_satisfied"),
        "env_done_success_count": env_done_success_count,
        "rollout_terminated_count": rollout_terminated_count,
        "rollout_truncated_flag_count": rollout_truncated_flag_count,
        "termination_reason_counts": termination_reason_counts,
        "termination_reason_counts_json": json.dumps(termination_reason_counts, ensure_ascii=False, sort_keys=True),
        "env_termination_reason_counts": env_termination_reason_counts,
        "env_termination_reason_counts_json": json.dumps(env_termination_reason_counts, ensure_ascii=False, sort_keys=True),
        "success_count": success_raw,
        "success_rate": success_raw / total if total else 0.0,
    }


def resolve_input(path_text: str) -> tuple[Path, Path]:
    path = Path(path_text).expanduser().resolve()
    if path.is_dir():
        return path, path / "multinode_skill_summary.json"
    return path.parent, path


def build_partial_summary(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    planned_jobs = manifest.get("jobs", [])
    planned_by_key = {row["job_key"]: row for row in planned_jobs if "job_key" in row}

    result_rows = load_jsonl_rows(sorted((run_dir / "worker_results").glob("worker_*.jsonl")))
    deduped: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        deduped[row["job_key"]] = row
    rows = [deduped[key] for key in sorted(deduped)]
    missing_keys = sorted(set(planned_by_key) - set(deduped))

    skill_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skill_task_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        skill_grouped[row["skill"]].append(row)
        skill_task_grouped[(row["skill"], row["task_name"])].append(row)

    skill_summary = []
    for skill, skill_rows in sorted(skill_grouped.items()):
        skill_summary.append(
            {
                "skill": skill,
                "task_count": len({row["task_name"] for row in skill_rows}),
                "demo_count": len({row["demo_id"] for row in skill_rows}),
                **summarize_result_rows(skill_rows),
            }
        )

    skill_task_summary = []
    for (skill, task_name), task_rows in sorted(skill_task_grouped.items()):
        skill_task_summary.append(
            {
                "skill": skill,
                "task_name": task_name,
                "demo_count": len({row["demo_id"] for row in task_rows}),
                **summarize_result_rows(task_rows),
            }
        )

    top_summary = summarize_result_rows(rows)
    result_type_counts = dict(sorted(Counter(str(row.get("result_type")) for row in rows).items()))
    return {
        "schema_version": 2,
        "out_dir": str(run_dir),
        "planned_jobs": len(planned_jobs),
        "completed_jobs": len(rows),
        "missing_jobs": len(missing_keys),
        "raw_result_rows": len(result_rows),
        "deduped_result_rows": len(rows),
        "runtime_pass": top_summary["runtime_pass_count"],
        "runtime_fail": top_summary["runtime_fail_count"],
        "runtime_pass_rate_completed": top_summary["runtime_pass_rate"],
        "runtime_pass_rate_planned": top_summary["runtime_pass_count"] / len(planned_jobs) if planned_jobs else 0.0,
        "pre_satisfied_start": top_summary["pre_satisfied_start_count"],
        "metric_invalid": top_summary["metric_invalid_count"],
        "metric_invalid_missing_object": top_summary["metric_invalid_missing_object_count"],
        "attemptable_segments": top_summary["attemptable_segment_count"],
        "policy_success_attemptable": top_summary["policy_success_attemptable_count"],
        "policy_success_attemptable_rate": top_summary["policy_success_attemptable_rate"],
        "policy_success_clean_attemptable": top_summary["policy_success_clean_attemptable_count"],
        "policy_success_clean_attemptable_rate": top_summary["policy_success_clean_attemptable_rate"],
        "short_video_problem": top_summary["short_video_problem_count"],
        "early_metric_activation_review_needed": top_summary["early_metric_activation_review_needed_count"],
        "short_proxy_success": top_summary["short_proxy_success_count"],
        "likely_proxy_false_positive": top_summary["likely_proxy_false_positive_count"],
        "transfer_pose_proxy_success_unconfirmed": top_summary[
            "transfer_pose_proxy_success_unconfirmed_count"
        ],
        "meaningful_policy_caused_transition": top_summary["meaningful_policy_caused_transition_count"],
        "metric_unsatisfied_attemptable": top_summary["metric_unsatisfied_attemptable_count"],
        "timeout": top_summary["timeout_count"],
        "truncated": top_summary["truncated_count"],
        "env_terminated_metric_unsatisfied": top_summary["env_terminated_metric_unsatisfied_count"],
        "env_task_success_before_segment_success": top_summary["env_task_success_before_segment_success_count"],
        "other_metric_unsatisfied": top_summary["other_metric_unsatisfied_count"],
        "env_done_success_count": top_summary["env_done_success_count"],
        "rollout_terminated_count": top_summary["rollout_terminated_count"],
        "rollout_truncated_flag_count": top_summary["rollout_truncated_flag_count"],
        "termination_reason_counts": top_summary["termination_reason_counts"],
        "env_termination_reason_counts": top_summary["env_termination_reason_counts"],
        "policy_success_raw": top_summary["success_count_raw"],
        "policy_success": top_summary["success_count_raw"],
        "result_type_counts": result_type_counts,
        "missing_job_keys": missing_keys,
        "skill_summary": skill_summary,
        "skill_task_summary": skill_task_summary,
    }


def load_side(path_text: str, label: str) -> dict[str, Any]:
    run_dir, summary_path = resolve_input(path_text)
    state: dict[str, Any] = {
        "label": label,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "status": "missing",
        "summary": None,
        "worker_started": 0,
        "worker_done": 0,
        "note": "",
    }

    status_dir = run_dir / "worker_status"
    if status_dir.exists():
        state["worker_started"] = len(list(status_dir.glob("worker_*.started.json")))
        state["worker_done"] = len(list(status_dir.glob("worker_*.done.json")))

    if summary_path.exists():
        summary = load_json(summary_path)
        if summary.get("schema_version") != 2:
            raise SystemExit(f"Expected schema_version=2 in {summary_path}, got {summary.get('schema_version')}")
        state["status"] = "complete" if summary.get("missing_jobs", 0) == 0 else "partial_summary"
        state["summary"] = summary
        return state

    if (run_dir / "manifest.json").exists():
        state["status"] = "pending"
        state["summary"] = build_partial_summary(run_dir)
        state["note"] = f"summary not ready: {summary_path}"
        return state

    state["note"] = f"run metadata not found under {run_dir}"
    return state


def fmt_value(value: Any, *, is_rate: bool = False) -> str:
    if value is None:
        return ""
    if is_rate:
        return f"{float(value):.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fmt_delta(a_value: Any, b_value: Any, *, is_rate: bool = False) -> str:
    if a_value is None or b_value is None:
        return ""
    delta = float(b_value) - float(a_value)
    if is_rate:
        return f"{delta:+.4f} ({delta * 100:+.2f} pp)"
    if delta.is_integer():
        return f"{int(delta):+d}"
    return f"{delta:+.4f}"


def collect_csv_rows(a_side: dict[str, Any], b_side: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    a_summary = a_side.get("summary") or {}
    b_summary = b_side.get("summary") or {}

    for display_name, key, is_rate in OVERALL_ROWS:
        a_value = a_summary.get(key, 0)
        b_value = b_summary.get(key, 0)
        rows.append(
            {
                "scope": "overall",
                "key": "__overall__",
                "metric": display_name,
                "a_label": a_side["label"],
                "a_value": fmt_value(a_value, is_rate=is_rate),
                "b_label": b_side["label"],
                "b_value": fmt_value(b_value, is_rate=is_rate),
                "delta_b_minus_a": fmt_delta(a_value, b_value, is_rate=is_rate),
                "a_status": a_side["status"],
                "b_status": b_side["status"],
            }
        )

    a_rt = a_summary.get("result_type_counts", {})
    b_rt = b_summary.get("result_type_counts", {})
    for result_type in sorted(set(a_rt) | set(b_rt)):
        a_value = int(a_rt.get(result_type, 0))
        b_value = int(b_rt.get(result_type, 0))
        rows.append(
            {
                "scope": "result_type_counts",
                "key": result_type,
                "metric": "count",
                "a_label": a_side["label"],
                "a_value": str(a_value),
                "b_label": b_side["label"],
                "b_value": str(b_value),
                "delta_b_minus_a": fmt_delta(a_value, b_value),
                "a_status": a_side["status"],
                "b_status": b_side["status"],
            }
        )

    a_skills = {row["skill"]: row for row in a_summary.get("skill_summary", [])}
    b_skills = {row["skill"]: row for row in b_summary.get("skill_summary", [])}
    common_skills = sorted(set(a_skills) & set(b_skills))
    for skill in common_skills:
        a_row = a_skills[skill]
        b_row = b_skills[skill]
        for display_name, key, is_rate in SKILL_METRICS:
            a_value = a_row.get(key, 0)
            b_value = b_row.get(key, 0)
            rows.append(
                {
                    "scope": "per_skill",
                    "key": skill,
                    "metric": display_name,
                    "a_label": a_side["label"],
                    "a_value": fmt_value(a_value, is_rate=is_rate),
                    "b_label": b_side["label"],
                    "b_value": fmt_value(b_value, is_rate=is_rate),
                    "delta_b_minus_a": fmt_delta(a_value, b_value, is_rate=is_rate),
                    "a_status": a_side["status"],
                    "b_status": b_side["status"],
                }
            )
    return rows


def build_json_payload(a_side: dict[str, Any], b_side: dict[str, Any], csv_rows: list[dict[str, Any]]) -> dict[str, Any]:
    a_summary = a_side.get("summary") or {}
    b_summary = b_side.get("summary") or {}
    compare_status = "ready" if a_side["status"] == "complete" and b_side["status"] == "complete" else "pending"
    a_skills = {row["skill"]: row for row in a_summary.get("skill_summary", [])}
    b_skills = {row["skill"]: row for row in b_summary.get("skill_summary", [])}
    skill_deltas = []
    for skill in sorted(set(a_skills) & set(b_skills)):
        a_row = a_skills[skill]
        b_row = b_skills[skill]
        a_rate = float(a_row.get("policy_success_attemptable_rate", 0.0))
        b_rate = float(b_row.get("policy_success_attemptable_rate", 0.0))
        skill_deltas.append(
            {
                "skill": skill,
                "a_attemptable": a_row.get("attemptable_segment_count", 0),
                "b_attemptable": b_row.get("attemptable_segment_count", 0),
                "a_policy_success_attemptable": a_row.get("policy_success_attemptable_count", 0),
                "b_policy_success_attemptable": b_row.get("policy_success_attemptable_count", 0),
                "a_rate": a_rate,
                "b_rate": b_rate,
                "delta_rate_b_minus_a": b_rate - a_rate,
            }
        )
    skill_deltas.sort(key=lambda row: abs(row["delta_rate_b_minus_a"]), reverse=True)

    return {
        "compare_status": compare_status,
        "a": {"label": a_side["label"], "status": a_side["status"], "run_dir": a_side["run_dir"], "summary_path": a_side["summary_path"], "summary": a_summary},
        "b": {"label": b_side["label"], "status": b_side["status"], "run_dir": b_side["run_dir"], "summary_path": b_side["summary_path"], "summary": b_summary},
        "overall_delta": {
            key: (b_summary.get(key, 0) - a_summary.get(key, 0))
            for _, key, _ in OVERALL_ROWS
            if isinstance(a_summary.get(key), (int, float)) and isinstance(b_summary.get(key), (int, float))
        },
        "result_type_counts_delta": {
            result_type: int((b_summary.get("result_type_counts", {}) or {}).get(result_type, 0))
            - int((a_summary.get("result_type_counts", {}) or {}).get(result_type, 0))
            for result_type in sorted(set((a_summary.get("result_type_counts", {}) or {})) | set((b_summary.get("result_type_counts", {}) or {})))
        },
        "skill_deltas": skill_deltas,
        "csv_rows": csv_rows,
    }


def render_markdown(a_side: dict[str, Any], b_side: dict[str, Any], *, top_k: int) -> str:
    a_summary = a_side.get("summary") or {}
    b_summary = b_side.get("summary") or {}
    compare_status = "ready" if a_side["status"] == "complete" and b_side["status"] == "complete" else "pending"

    lines: list[str] = []
    lines.append(f"# A/B Compare: {a_side['label']} vs {b_side['label']}")
    lines.append("")
    lines.append(f"- compare_status: `{compare_status}`")
    lines.append(f"- {a_side['label']}: `{a_side['status']}` | run_dir=`{a_side['run_dir']}` | summary=`{a_side['summary_path']}`")
    if a_side.get("note"):
        lines.append(f"- {a_side['label']}_note: `{a_side['note']}`")
    lines.append(f"- {b_side['label']}: `{b_side['status']}` | run_dir=`{b_side['run_dir']}` | summary=`{b_side['summary_path']}`")
    if b_side.get("note"):
        lines.append(f"- {b_side['label']}_note: `{b_side['note']}`")
    lines.append("")
    lines.append("## Progress")
    lines.append("")
    lines.append(f"- {a_side['label']}: completed `{a_summary.get('completed_jobs', 0)}/{a_summary.get('planned_jobs', 0)}` | workers started/done `{a_side['worker_started']}/{a_side['worker_done']}`")
    lines.append(f"- {b_side['label']}: completed `{b_summary.get('completed_jobs', 0)}/{b_summary.get('planned_jobs', 0)}` | workers started/done `{b_side['worker_started']}/{b_side['worker_done']}`")
    lines.append("")

    lines.append("## Overall schema v2 metrics")
    lines.append("")
    lines.append(f"| metric | {a_side['label']} | {b_side['label']} | delta (B-A) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for display_name, key, is_rate in OVERALL_ROWS:
        a_value = a_summary.get(key, 0)
        b_value = b_summary.get(key, 0)
        lines.append(
            f"| {display_name} | {fmt_value(a_value, is_rate=is_rate)} | {fmt_value(b_value, is_rate=is_rate)} | {fmt_delta(a_value, b_value, is_rate=is_rate)} |"
        )
    lines.append("")

    lines.append("## result_type_counts")
    lines.append("")
    lines.append(f"| result_type | {a_side['label']} | {b_side['label']} | delta (B-A) |")
    lines.append("| --- | ---: | ---: | ---: |")
    a_rt = a_summary.get("result_type_counts", {})
    b_rt = b_summary.get("result_type_counts", {})
    for result_type in sorted(set(a_rt) | set(b_rt)):
        a_value = int(a_rt.get(result_type, 0))
        b_value = int(b_rt.get(result_type, 0))
        lines.append(f"| {result_type} | {a_value} | {b_value} | {fmt_delta(a_value, b_value)} |")
    lines.append("")

    a_skills = {row["skill"]: row for row in a_summary.get("skill_summary", [])}
    b_skills = {row["skill"]: row for row in b_summary.get("skill_summary", [])}
    common_skills = sorted(set(a_skills) & set(b_skills))
    only_a = sorted(set(a_skills) - set(b_skills))
    only_b = sorted(set(b_skills) - set(a_skills))

    if compare_status != "ready":
        lines.append(
            "> Pending input detected: per-skill delta below only uses common skills currently present in both inputs. Skills missing from the pending side are intentionally excluded instead of being treated as zero."
        )
        lines.append("")

    lines.append("## Per-skill delta")
    lines.append("")
    lines.append(f"- common_skills: `{len(common_skills)}`")
    lines.append(f"- only_in_{a_side['label']}: `{len(only_a)}`")
    lines.append(f"- only_in_{b_side['label']}: `{len(only_b)}`")
    if only_a:
        lines.append(f"- only_in_{a_side['label']}_examples: `{', '.join(only_a[:10])}`")
    if only_b:
        lines.append(f"- only_in_{b_side['label']}_examples: `{', '.join(only_b[:10])}`")
    lines.append("")
    lines.append(
        f"| skill | {a_side['label']} attemptable | {b_side['label']} attemptable | delta | {a_side['label']} success_attemptable | {b_side['label']} success_attemptable | delta | {a_side['label']} rate | {b_side['label']} rate | delta (B-A) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    ranked_skills = sorted(
        common_skills,
        key=lambda skill: abs(float(b_skills[skill].get("policy_success_attemptable_rate", 0.0)) - float(a_skills[skill].get("policy_success_attemptable_rate", 0.0))),
        reverse=True,
    )
    if top_k > 0:
        ranked_skills = ranked_skills[:top_k]
    for skill in ranked_skills:
        a_row = a_skills[skill]
        b_row = b_skills[skill]
        a_attemptable = a_row.get("attemptable_segment_count", 0)
        b_attemptable = b_row.get("attemptable_segment_count", 0)
        a_success = a_row.get("policy_success_attemptable_count", 0)
        b_success = b_row.get("policy_success_attemptable_count", 0)
        a_rate = a_row.get("policy_success_attemptable_rate", 0.0)
        b_rate = b_row.get("policy_success_attemptable_rate", 0.0)
        lines.append(
            f"| {skill} | {a_attemptable} | {b_attemptable} | {fmt_delta(a_attemptable, b_attemptable)} | {a_success} | {b_success} | {fmt_delta(a_success, b_success)} | {fmt_value(a_rate, is_rate=True)} | {fmt_value(b_rate, is_rate=True)} | {fmt_delta(a_rate, b_rate, is_rate=True)} |"
        )
    if common_skills and len(common_skills) > len(ranked_skills):
        lines.append("")
        lines.append(f"(showing top {len(ranked_skills)} / {len(common_skills)} common skills by absolute rate delta)")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "key",
        "metric",
        "a_label",
        "a_value",
        "b_label",
        "b_value",
        "delta_b_minus_a",
        "a_status",
        "b_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    a_side = load_side(args.a, args.label_a)
    b_side = load_side(args.b, args.label_b)

    markdown = render_markdown(a_side, b_side, top_k=max(0, args.top_k))
    csv_rows = collect_csv_rows(a_side, b_side)

    if args.md_out:
        args.md_out = args.md_out.expanduser().resolve()
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown)
    if args.csv_out:
        args.csv_out = args.csv_out.expanduser().resolve()
        write_csv(args.csv_out, csv_rows)
    if args.json_out:
        args.json_out = args.json_out.expanduser().resolve()
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(build_json_payload(a_side, b_side, csv_rows), ensure_ascii=False, indent=2))

    if not args.md_out:
        print(markdown)
    else:
        payload = {
            "md_out": str(args.md_out),
            "csv_out": str(args.csv_out.resolve()) if args.csv_out else "",
            "json_out": str(args.json_out.resolve()) if args.json_out else "",
            "a_status": a_side["status"],
            "b_status": b_side["status"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
