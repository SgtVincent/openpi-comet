#!/usr/bin/env python3
"""Gradio viewer for validated fixed-semantics exact segment failures.

The core index, outcome, annotation, path, and progress helpers use only the
Python standard library. Gradio, OpenCV, and NumPy are imported lazily only
when the interactive application is built. Protected result, annotation, and
video inputs are always read-only; an index is written only when
``--index-cache`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_RUN_ROOT = Path(
    "/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/segment_eval_runs/"
    "boundary_failure_matrix_20260717_170500/full_restore_only_20260717_190000"
)
DEFAULT_RESULTS_ROOT = DEFAULT_RUN_ROOT / "results"
DEFAULT_DATASET_ROOT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos")
EXPECTED_TOTAL_ROWS = 235_491
EXPECTED_FIXED_FAILURES = 46_578
EXACT_STATUS = "exact_sim_restored"
PLACE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "place in": ("inside",),
    "place on": ("ontop",),
    "insert": ("inside",),
    "place under": ("under",),
    "place on next to": ("ontop", "nextto"),
    "place in next to": ("inside", "nextto"),
}
CAMERAS = ("head", "left_wrist", "right_wrist")
INDEX_SCHEMA_VERSION = 1
DEFAULT_PAGE_SIZE = 100


@dataclass(frozen=True)
class FixedOutcome:
    category: str
    reason: str
    raw_category: str
    failure_reason: str

    @property
    def is_exact_failure(self) -> bool:
        return self.category == "fixed_failure"


@dataclass
class FailureIndex:
    entries: list[dict[str, Any]]
    summary: dict[str, Any]


def classify_raw(row: Mapping[str, Any]) -> str:
    """Classify persisted evidence exactly as the validated aggregate does."""
    status = row.get("evidence_status")
    exact = row.get("exact_evaluable")
    restored = row.get("exact_sim_restored")
    failure = row.get("raw_end_failure")
    success = row.get("raw_end_satisfied")
    if exact is True:
        if status != EXACT_STATUS or restored is not True:
            raise ValueError("raw exact row has inconsistent status/restore")
        if not isinstance(failure, bool) or not isinstance(success, bool) or failure == success:
            raise ValueError("raw exact outcomes are not complementary native booleans")
        return "raw_failure" if failure else "raw_success"
    if exact is False:
        if failure is not None or success is not None:
            raise ValueError("raw not-exact row has persisted exact outcome")
        if status == "exact_sim_restored_predicate_invalid" and restored is True:
            return "raw_persisted_predicate_invalid"
        if status == "not_evaluable_offline" and restored is False:
            return "raw_not_evaluable_offline"
        raise ValueError("raw not-exact row has inconsistent status")
    raise ValueError("exact_evaluable is not a native boolean")


def predicate_name(item: Mapping[str, Any]) -> str:
    diagnostics = item.get("diagnostics") if isinstance(item.get("diagnostics"), dict) else {}
    rendered = str(item.get("predicate") or "")
    return str(diagnostics.get("predicate_name") or item.get("name") or rendered.split("(", 1)[0]).strip().lower()


def nonempty_predicate_error(item: Mapping[str, Any]) -> tuple[str, Any] | None:
    diagnostics = item.get("diagnostics") if isinstance(item.get("diagnostics"), dict) else {}
    for key in ("error", "exception", "evaluation_error", "trace_error", "missing_object"):
        for container in (item, diagnostics):
            if key in container and container[key] not in (None, False, "", [], {}):
                return key, container[key]
    return None


def classify_fixed_place(row: Mapping[str, Any], skill: str) -> tuple[str, str]:
    """Mirror aggregate.py's six place-skill primary predicate conjunction."""
    trace = row.get("predicate_trace")
    if not isinstance(trace, list):
        return "fixed_semantic_invalid", "predicate_trace_not_list"
    if any(not isinstance(item, dict) for item in trace):
        return "fixed_semantic_invalid", "predicate_trace_contains_non_object_item"
    for item in trace:
        error = nonempty_predicate_error(item)
        if error is not None:
            return "fixed_semantic_invalid", f"predicate_trace_{error[0]}"
    required_items: list[Mapping[str, Any]] = []
    for required in PLACE_REQUIREMENTS[skill]:
        matches = [item for item in trace if predicate_name(item) == required]
        if not matches:
            return "fixed_semantic_invalid", f"missing_required_primary:{required}"
        if len(matches) != 1:
            return "fixed_semantic_invalid", f"duplicate_required_primary:{required}"
        item = matches[0]
        if item.get("desired") is not True:
            return "fixed_semantic_invalid", f"required_primary_not_desired_true:{required}"
        if not isinstance(item.get("satisfied"), bool):
            return "fixed_semantic_invalid", f"required_primary_satisfied_not_boolean:{required}"
        required_items.append(item)
    success = all(item["satisfied"] for item in required_items)
    return ("fixed_success" if success else "fixed_failure"), "primary_predicates_evaluated"


def fixed_failure_reason(row: Mapping[str, Any], skill: str, category: str) -> str:
    if category != "fixed_failure":
        return str(row.get("failure_reason") or category)
    if skill not in PLACE_REQUIREMENTS:
        return str(row.get("failure_reason") or "persisted_raw_end_failure")
    trace = row.get("predicate_trace") if isinstance(row.get("predicate_trace"), list) else []
    unsatisfied = [
        required
        for required in PLACE_REQUIREMENTS[skill]
        if any(predicate_name(item) == required and item.get("satisfied") is False for item in trace if isinstance(item, dict))
    ]
    return "required_primary_predicates_unsatisfied:" + ",".join(unsatisfied)


def derive_fixed_outcome(row: Mapping[str, Any]) -> FixedOutcome:
    """Derive the viewer outcome without treating invalid/not-exact rows as failures."""
    raw_category = classify_raw(row)
    skill = str(row.get("skill") or "")
    if raw_category == "raw_failure":
        category = "fixed_failure"
    elif raw_category == "raw_success":
        category = "fixed_success"
    elif raw_category == "raw_persisted_predicate_invalid":
        category = "fixed_persisted_predicate_invalid"
    else:
        category = "fixed_not_evaluable_offline"
    reason = "preserved_persisted_classification"
    if skill in PLACE_REQUIREMENTS and raw_category in {"raw_failure", "raw_success"}:
        category, reason = classify_fixed_place(row, skill)
    return FixedOutcome(
        category=category,
        reason=reason,
        raw_category=raw_category,
        failure_reason=fixed_failure_reason(row, skill, category),
    )


def result_jsonl_paths(results_root: Path) -> list[Path]:
    paths = sorted(results_root.glob("lane_*/shard_*/segments.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no lane_*/shard_*/segments.jsonl files under {results_root}")
    return paths


def source_signature(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    ]


def slim_failure_entry(row: Mapping[str, Any], source_path: Path, byte_offset: int, outcome: FixedOutcome) -> dict[str, Any]:
    return {
        "dedupe_key": str(row["dedupe_key"]),
        "source_path": str(source_path.resolve()),
        "byte_offset": byte_offset,
        "task_id": int(row["task_id"]),
        "task_dir": str(row["task_dir"]),
        "task_name": str(row["task_name"]),
        "demo_id": str(row["demo_id"]),
        "skill": str(row["skill"]),
        "skill_idx": int(row["skill_idx"]),
        "frame_start": int(row["frame_start"]),
        "segment_end_frame": int(row["segment_end_frame"]),
        "metric_family": str(row.get("metric_family") or ""),
        "failure_reason": outcome.failure_reason,
        "semantic_reason": outcome.reason,
        "raw_category": outcome.raw_category,
        "annotation_path": str(row.get("annotation_path") or ""),
        "parquet_path": str(row.get("parquet_path") or ""),
    }


def _load_index_cache(cache_path: Path, signature: list[dict[str, Any]]) -> FailureIndex | None:
    if not cache_path.is_file():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION or payload.get("source_signature") != signature:
        return None
    return FailureIndex(entries=list(payload["entries"]), summary=dict(payload["summary"]))


def _write_index_cache(cache_path: Path, signature: list[dict[str, Any]], index: FailureIndex) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "source_signature": signature,
        "summary": index.summary,
        "entries": index.entries,
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(cache_path)


def scan_failure_index(
    results_root: Path,
    *,
    expected_total: int | None = EXPECTED_TOTAL_ROWS,
    expected_failures: int | None = EXPECTED_FIXED_FAILURES,
    cache_path: Path | None = None,
) -> FailureIndex:
    """Scan protected JSONLs and retain only fixed-semantic exact failures."""
    paths = result_jsonl_paths(results_root)
    signature = source_signature(paths)
    if cache_path is not None:
        cached = _load_index_cache(cache_path, signature)
        if cached is not None:
            if expected_total is not None and cached.summary["total_rows"] != expected_total:
                raise RuntimeError("cached total row count does not match expected total")
            if expected_failures is not None and cached.summary["fixed_failure_count"] != expected_failures:
                raise RuntimeError("cached fixed failure count does not match expected count")
            return cached

    entries: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    raw_category_counts: Counter[str] = Counter()
    unique_keys: set[str] = set()
    rows_seen = 0
    for path in paths:
        with path.open("rb") as handle:
            while True:
                byte_offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    raise RuntimeError(f"blank JSONL row at {path}:{rows_seen + 1}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(f"non-object JSON row at {path}:{byte_offset}")
                rows_seen += 1
                key = row.get("dedupe_key")
                if not isinstance(key, str) or not key:
                    raise RuntimeError(f"missing dedupe key at {path}:{byte_offset}")
                if key in unique_keys:
                    raise RuntimeError(f"duplicate dedupe key: {key}")
                unique_keys.add(key)
                outcome = derive_fixed_outcome(row)
                category_counts[outcome.category] += 1
                raw_category_counts[outcome.raw_category] += 1
                if outcome.is_exact_failure:
                    entries.append(slim_failure_entry(row, path, byte_offset, outcome))

    entries.sort(
        key=lambda item: (
            item["task_id"],
            item["demo_id"],
            item["skill_idx"],
            item["segment_end_frame"],
            item["dedupe_key"],
        )
    )
    summary = {
        "results_root": str(results_root.resolve()),
        "source_file_count": len(paths),
        "total_rows": rows_seen,
        "unique_dedupe_keys": len(unique_keys),
        "fixed_failure_count": len(entries),
        "fixed_category_counts": dict(sorted(category_counts.items())),
        "raw_category_counts": dict(sorted(raw_category_counts.items())),
    }
    if expected_total is not None and rows_seen != expected_total:
        raise RuntimeError(f"source total mismatch: actual={rows_seen}, expected={expected_total}")
    if expected_failures is not None and len(entries) != expected_failures:
        raise RuntimeError(f"fixed failure mismatch: actual={len(entries)}, expected={expected_failures}")
    index = FailureIndex(entries=entries, summary=summary)
    if cache_path is not None:
        _write_index_cache(cache_path, signature, index)
    return index


def load_source_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(entry["source_path"]))
    with path.open("rb") as handle:
        handle.seek(int(entry["byte_offset"]))
        row = json.loads(handle.readline())
    if row.get("dedupe_key") != entry.get("dedupe_key"):
        raise RuntimeError(f"source row changed for {entry.get('dedupe_key')}")
    return row


def normalize_demo_id(value: Any) -> str:
    text = str(value)
    if text.startswith("episode_"):
        text = text.removeprefix("episode_")
    if not text.isdigit():
        raise ValueError(f"demo_id must be numeric, got {value!r}")
    return text.zfill(8)


def resolve_video_paths(dataset_root: Path, task_id: int, demo_id: Any) -> dict[str, Path]:
    episode = normalize_demo_id(demo_id)
    base = dataset_root / "videos" / f"task-{int(task_id):04d}"
    return {
        camera: base / f"observation.images.rgb.{camera}" / f"episode_{episode}.mp4" for camera in CAMERAS
    }


def resolve_annotation_path(dataset_root: Path, task_id: int, demo_id: Any, persisted_path: str | None = None) -> Path:
    if persisted_path:
        return Path(persisted_path)
    episode = normalize_demo_id(demo_id)
    return dataset_root / "annotations" / f"task-{int(task_id):04d}" / f"episode_{episode}.json"


def _list_text(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(str(item) for item in value)
    if value in (None, ""):
        return "unknown"
    return str(value)


def _objects_text(value: Any) -> str:
    if value in (None, [], ""):
        return "none"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_annotation_timeline(annotation: Mapping[str, Any], selected_skill_idx: int) -> list[dict[str, Any]]:
    """Build full half-open skill intervals with explicit unlabeled gaps."""
    skills: list[dict[str, Any]] = []
    for order, row in enumerate(annotation.get("skill_annotation", [])):
        if not isinstance(row, dict) or not isinstance(row.get("frame_duration"), list):
            continue
        duration = row["frame_duration"]
        if len(duration) != 2:
            continue
        start, end = int(duration[0]), int(duration[1])
        if end <= start:
            continue
        skills.append(
            {
                "kind": "skill",
                "start": start,
                "end": end,
                "skill_idx": int(row.get("skill_idx", order)),
                "label": _list_text(row.get("skill_description")),
                "skill_type": _list_text(row.get("skill_type")),
                "objects": _objects_text(row.get("object_id")),
                "manipulating_objects": _objects_text(row.get("manipulating_object_id")),
                "role": "other",
            }
        )
    skills.sort(key=lambda item: (item["start"], item["end"], item["skill_idx"]))
    selected_position = next((i for i, item in enumerate(skills) if item["skill_idx"] == selected_skill_idx), None)
    if selected_position is not None:
        for position, item in enumerate(skills):
            if position == selected_position:
                item["role"] = "current"
            elif position == selected_position - 1:
                item["role"] = "previous"
            elif position == selected_position + 1:
                item["role"] = "next"

    valid_duration = annotation.get("meta_data", {}).get("valid_duration")
    if isinstance(valid_duration, list) and len(valid_duration) == 2:
        lower, upper = int(valid_duration[0]), int(valid_duration[1])
    elif skills:
        lower, upper = skills[0]["start"], skills[-1]["end"]
    else:
        return []
    if upper <= lower:
        return skills

    timeline: list[dict[str, Any]] = []
    cursor = lower
    for skill in skills:
        if skill["start"] > cursor:
            timeline.append(
                {
                    "kind": "gap",
                    "start": cursor,
                    "end": skill["start"],
                    "skill_idx": None,
                    "label": "unlabeled gap",
                    "skill_type": "",
                    "objects": "",
                    "manipulating_objects": "",
                    "role": "gap",
                }
            )
        timeline.append(skill)
        cursor = max(cursor, skill["end"])
    if cursor < upper:
        timeline.append(
            {
                "kind": "gap",
                "start": cursor,
                "end": upper,
                "skill_idx": None,
                "label": "unlabeled gap",
                "skill_type": "",
                "objects": "",
                "manipulating_objects": "",
                "role": "gap",
            }
        )
    return timeline


def timeline_context(timeline: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    focus_skills = [item for item in timeline if item.get("role") in {"previous", "current", "next"}]
    if not focus_skills:
        return [dict(item) for item in timeline]
    start = min(int(item["start"]) for item in focus_skills)
    end = max(int(item["end"]) for item in focus_skills)
    return [dict(item) for item in timeline if int(item["end"]) > start and int(item["start"]) < end]


def frame_progress_metadata(
    frame_idx: int,
    timeline: Sequence[Mapping[str, Any]],
    selected_skill_idx: int,
) -> dict[str, Any]:
    selected = next(
        (item for item in timeline if item.get("kind") == "skill" and item.get("skill_idx") == selected_skill_idx),
        None,
    )
    active = next(
        (item for item in timeline if int(item["start"]) <= frame_idx < int(item["end"])),
        None,
    )
    if selected is None:
        selected_start = selected_end = selected_total = selected_done = 0
        selected_percent = 0.0
    else:
        selected_start, selected_end = int(selected["start"]), int(selected["end"])
        selected_total = max(1, selected_end - selected_start)
        selected_done = min(max(frame_idx - selected_start + 1, 0), selected_total)
        selected_percent = selected_done / selected_total * 100.0
    context = timeline_context(timeline)
    if context:
        context_start = min(int(item["start"]) for item in context)
        context_end = max(int(item["end"]) for item in context)
        context_total = max(1, context_end - context_start)
        context_done = min(max(frame_idx - context_start + 1, 0), context_total)
        context_percent = context_done / context_total * 100.0
    else:
        context_start = context_end = context_done = context_total = 0
        context_percent = 0.0
    return {
        "frame": int(frame_idx),
        "active_kind": active.get("kind") if active else "outside_annotation",
        "active_role": active.get("role") if active else "outside",
        "active_skill_idx": active.get("skill_idx") if active else None,
        "active_label": active.get("label") if active else "outside annotation",
        "selected_start": selected_start,
        "selected_end": selected_end,
        "selected_done": selected_done,
        "selected_total": selected_total,
        "selected_percent": selected_percent,
        "context_start": context_start,
        "context_end": context_end,
        "context_done": context_done,
        "context_total": context_total,
        "context_percent": context_percent,
    }


def predicate_metric_diagnostics(row: Mapping[str, Any], outcome: FixedOutcome) -> dict[str, Any]:
    skill = str(row.get("skill") or "")
    trace = row.get("predicate_trace") if isinstance(row.get("predicate_trace"), list) else []
    primary_names = PLACE_REQUIREMENTS.get(skill, ())
    primary = [item for item in trace if isinstance(item, dict) and predicate_name(item) in primary_names]
    release = [
        item
        for item in trace
        if isinstance(item, dict) and predicate_name(item) == "grasped" and item.get("desired") is False
    ]
    return {
        "fixed_outcome": outcome.category,
        "fixed_semantic_reason": outcome.reason,
        "fixed_failure_reason": outcome.failure_reason,
        "raw_category": outcome.raw_category,
        "place_fixed_semantics_applied": skill in PLACE_REQUIREMENTS,
        "required_primary_predicates": list(primary_names),
        "primary_predicate_trace": primary,
        "auxiliary_desired_false_grasped_release_trace": release,
        "predicate_spec": row.get("predicate_spec"),
        "predicate_trace": row.get("predicate_trace"),
        "metric_family": row.get("metric_family"),
        "metric_debug": row.get("metric_debug"),
        "success_rule": row.get("success_rule"),
        "combine_mode": row.get("combine_mode"),
        "persisted_failure_reason": row.get("failure_reason"),
        "raw_end_failure": row.get("raw_end_failure"),
        "raw_end_satisfied": row.get("raw_end_satisfied"),
        "evidence_status": row.get("evidence_status"),
        "exact_evaluable": row.get("exact_evaluable"),
    }


def format_entry_label(entry: Mapping[str, Any]) -> str:
    return (
        f"{entry['task_dir']} | {entry['task_name']} | demo {entry['demo_id']} | "
        f"skill[{entry['skill_idx']}] {entry['skill']} | "
        f"frames {entry['frame_start']}:{entry['segment_end_frame']}"
    )


def filter_entries(
    entries: Sequence[Mapping[str, Any]], task_dir: str = "All", skill: str = "All"
) -> list[Mapping[str, Any]]:
    return [
        entry
        for entry in entries
        if (task_dir == "All" or entry["task_dir"] == task_dir) and (skill == "All" or entry["skill"] == skill)
    ]


def page_entries(entries: Sequence[Mapping[str, Any]], page: int, page_size: int = DEFAULT_PAGE_SIZE) -> tuple[list[Any], int, int]:
    total_pages = max(1, math.ceil(len(entries) / page_size))
    page = min(max(int(page), 1), total_pages)
    start = (page - 1) * page_size
    choices = [(format_entry_label(entry), entry["dedupe_key"]) for entry in entries[start : start + page_size]]
    return choices, page, total_pages


def annotation_context_markdown(timeline: Sequence[Mapping[str, Any]]) -> str:
    lines = ["### Annotation context"]
    for role in ("previous", "current", "next"):
        item = next((segment for segment in timeline if segment.get("role") == role), None)
        if item is None:
            lines.append(f"- **{role}:** none")
        else:
            lines.append(
                f"- **{role}:** skill `{item['skill_idx']}` — {item['label']} "
                f"(`[{item['start']}, {item['end']})`), objects: `{item['objects']}`"
            )
    gaps = [segment for segment in timeline_context(timeline) if segment.get("kind") == "gap"]
    if gaps:
        lines.append("- **unlabeled gaps:** " + ", ".join(f"`[{gap['start']}, {gap['end']})`" for gap in gaps))
    else:
        lines.append("- **unlabeled gaps:** none in previous/current/next window")
    return "\n".join(lines)


def timeline_html(timeline: Sequence[Mapping[str, Any]], frame_idx: int) -> str:
    context = timeline_context(timeline)
    if not context:
        return "<div><strong>No valid annotation timeline.</strong></div>"
    start = min(int(item["start"]) for item in context)
    end = max(int(item["end"]) for item in context)
    span = max(1, end - start)
    palette = {"previous": "#6b7280", "current": "#dc2626", "next": "#2563eb", "other": "#9ca3af", "gap": "#f59e0b"}
    bars: list[str] = []
    for item in context:
        width = max(0.2, (int(item["end"]) - int(item["start"])) / span * 100.0)
        color = palette.get(str(item.get("role")), "#9ca3af")
        label = "GAP" if item.get("kind") == "gap" else f"{item.get('role')}: {item.get('skill_idx')} {item.get('label')}"
        title = f"{label} [{item['start']}, {item['end']})"
        bars.append(
            f'<div title="{escape(title)}" style="width:{width:.4f}%;background:{color};height:34px;'
            f'overflow:hidden;color:white;font-size:11px;padding:3px;box-sizing:border-box;border-right:1px solid white">'
            f"{escape(label)}</div>"
        )
    marker = min(max((frame_idx - start + 0.5) / span * 100.0, 0.0), 100.0)
    return (
        '<div style="position:relative;border:1px solid #9ca3af;background:#f3f4f6">'
        f'<div style="display:flex;width:100%">{"".join(bars)}</div>'
        f'<div style="position:absolute;left:{marker:.4f}%;top:-5px;bottom:-5px;width:3px;background:#111827"></div>'
        "</div>"
        f'<div style="display:flex;justify-content:space-between;font-family:monospace;font-size:12px">'
        f"<span>{start}</span><strong>frame {frame_idx}</strong><span>{end - 1}</span></div>"
        '<div style="font-size:12px">gray=previous, red=current failed skill, blue=next, orange=unlabeled gap</div>'
    )


def progress_markdown(metadata: Mapping[str, Any]) -> str:
    return (
        f"**Frame {metadata['frame']}** · active: `{metadata['active_role']}` / "
        f"`{metadata['active_label']}` · selected-skill progress: "
        f"**{metadata['selected_done']}/{metadata['selected_total']} ({metadata['selected_percent']:.1f}%)** · "
        f"context progress: {metadata['context_done']}/{metadata['context_total']} "
        f"({metadata['context_percent']:.1f}%)"
    )


def _load_ui_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    try:
        import gradio as gr
    except ImportError:
        gr = None
        missing.append("gradio==5.17.1")
    try:
        import cv2
    except ImportError:
        cv2 = None
        missing.append("opencv-python>=4.10.0.84")
    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy>=1.22.4,<2.0.0")
    if missing:
        quoted = " ".join(f"'{item}'" for item in missing)
        raise RuntimeError(
            "Interactive viewer dependencies are missing. Install them in an optional runtime environment with: "
            f"python -m pip install {quoted}. Index validation and unit tests do not require these packages."
        )
    return gr, cv2, np


class VideoFrameReader:
    def __init__(self, cv2: Any, np: Any) -> None:
        self.cv2 = cv2
        self.np = np

    def placeholder(self, camera: str, message: str) -> Any:
        image = self.np.zeros((360, 480, 3), dtype=self.np.uint8)
        image[:, :] = (45, 45, 55)
        self.cv2.putText(image, camera, (20, 55), self.cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for line_number, line in enumerate(message.splitlines()[:5], 1):
            self.cv2.putText(
                image,
                line[:55],
                (20, 80 + line_number * 35),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 100),
                1,
            )
        return image

    def read_aligned(self, paths: Mapping[str, Path], frame_idx: int) -> tuple[list[Any], str]:
        frames: list[Any] = []
        errors: list[str] = []
        for camera in CAMERAS:
            path = paths[camera]
            if not path.is_file():
                message = f"MISSING FILE\n{path}"
                frames.append(self.placeholder(camera, message))
                errors.append(f"{camera}: missing {path}")
                continue
            capture = self.cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise RuntimeError("VideoCapture could not open file")
                capture.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"could not decode frame {frame_idx}")
                frames.append(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            except Exception as exc:
                message = f"DECODE ERROR\nframe {frame_idx}\n{exc}"
                frames.append(self.placeholder(camera, message))
                errors.append(f"{camera}: {exc}")
            finally:
                capture.release()
        status = "All three cameras decoded at the same frame index." if not errors else "Visible camera errors: " + " | ".join(errors)
        return frames, status


def _row_summary_markdown(row: Mapping[str, Any], outcome: FixedOutcome) -> str:
    return "\n".join(
        [
            "### Selected fixed-semantic exact failure",
            f"- **task:** `{row.get('task_dir')}` / `{row.get('task_name')}` (id `{row.get('task_id')}`)",
            f"- **demo:** `{row.get('demo_id')}`",
            f"- **skill:** `{row.get('skill_idx')}` / `{row.get('skill')}`",
            f"- **evaluation bounds:** `[{row.get('frame_start')}, {row.get('segment_end_frame')})`",
            f"- **failure reason:** `{outcome.failure_reason}`",
            f"- **fixed semantic rule:** `{outcome.reason}`",
            f"- **metric family:** `{row.get('metric_family')}`",
            f"- **dedupe key:** `{row.get('dedupe_key')}`",
        ]
    )


def build_app(index: FailureIndex, dataset_root: Path, page_size: int = DEFAULT_PAGE_SIZE) -> Any:
    gr, cv2, np = _load_ui_dependencies()
    entries_by_key = {entry["dedupe_key"]: entry for entry in index.entries}
    reader = VideoFrameReader(cv2, np)
    tasks = ["All", *sorted({entry["task_dir"] for entry in index.entries})]
    skills = ["All", *sorted({entry["skill"] for entry in index.entries})]
    initial_choices, _, initial_pages = page_entries(index.entries, 1, page_size)
    initial_key = initial_choices[0][1] if initial_choices else None

    def refresh_filter(task_dir: str, skill: str, page: int) -> tuple[Any, Any, str]:
        filtered = filter_entries(index.entries, task_dir, skill)
        choices, current_page, total_pages = page_entries(filtered, page, page_size)
        value = choices[0][1] if choices else None
        summary = f"**{len(filtered):,} failures** · page {current_page}/{total_pages} · {page_size} rows/page"
        return (
            gr.update(minimum=1, maximum=total_pages, value=current_page),
            gr.update(choices=choices, value=value),
            summary,
        )

    def reset_filter(task_dir: str, skill: str) -> tuple[Any, Any, str]:
        return refresh_filter(task_dir, skill, 1)

    def present_row(row_key: str | None) -> tuple[Any, ...]:
        if not row_key or row_key not in entries_by_key:
            empty = reader.placeholder("no selection", "No failure selected")
            return (
                gr.update(minimum=0, maximum=0, value=0),
                empty,
                empty,
                empty,
                "No failure selected.",
                "",
                "",
                "",
                {},
                "Select a failure row.",
                {},
            )
        entry = entries_by_key[row_key]
        row = load_source_row(entry)
        outcome = derive_fixed_outcome(row)
        if not outcome.is_exact_failure:
            raise RuntimeError(f"indexed row no longer classifies as fixed failure: {row_key}")
        annotation_path = resolve_annotation_path(
            dataset_root, int(row["task_id"]), row["demo_id"], str(row.get("annotation_path") or "")
        )
        if annotation_path.is_file():
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            timeline = build_annotation_timeline(annotation, int(row["skill_idx"]))
        else:
            timeline = []
        if not any(item.get("role") == "current" for item in timeline):
            timeline = [
                {
                    "kind": "skill",
                    "start": int(row["frame_start"]),
                    "end": int(row["segment_end_frame"]),
                    "skill_idx": int(row["skill_idx"]),
                    "label": str(row["skill"]),
                    "skill_type": "",
                    "objects": "unknown",
                    "manipulating_objects": "unknown",
                    "role": "current",
                }
            ]
        context = timeline_context(timeline)
        lower = min(int(item["start"]) for item in context)
        upper_exclusive = max(int(item["end"]) for item in context)
        default_frame = min(max(int(row["segment_end_frame"]) - 1, lower), upper_exclusive - 1)
        paths = resolve_video_paths(dataset_root, int(row["task_id"]), row["demo_id"])
        frames, video_status = reader.read_aligned(paths, default_frame)
        progress = frame_progress_metadata(default_frame, timeline, int(row["skill_idx"]))
        selection_state = {
            "row_key": row_key,
            "task_id": int(row["task_id"]),
            "demo_id": str(row["demo_id"]),
            "skill_idx": int(row["skill_idx"]),
            "timeline": timeline,
            "video_paths": {camera: str(path) for camera, path in paths.items()},
        }
        annotation_status = "" if annotation_path.is_file() else f" Annotation missing: {annotation_path}."
        return (
            gr.update(minimum=lower, maximum=upper_exclusive - 1, value=default_frame, step=1),
            *frames,
            _row_summary_markdown(row, outcome),
            annotation_context_markdown(timeline),
            timeline_html(timeline, default_frame),
            progress_markdown(progress),
            predicate_metric_diagnostics(row, outcome),
            video_status + annotation_status,
            selection_state,
        )

    def present_frame(frame_idx: int, state: Mapping[str, Any]) -> tuple[Any, ...]:
        if not state:
            empty = reader.placeholder("no selection", "No failure selected")
            return empty, empty, empty, "", "", "Select a failure row."
        frame_idx = int(frame_idx)
        paths = {camera: Path(state["video_paths"][camera]) for camera in CAMERAS}
        frames, status = reader.read_aligned(paths, frame_idx)
        timeline = state["timeline"]
        progress = frame_progress_metadata(frame_idx, timeline, int(state["skill_idx"]))
        return *frames, timeline_html(timeline, frame_idx), progress_markdown(progress), status

    with gr.Blocks(title="Validated failed skill-segment viewer") as app:
        gr.Markdown(
            "# Validated fixed-semantic exact failure viewer\n"
            f"Read-only corpus: **{index.summary['total_rows']:,} rows**, "
            f"**{index.summary['fixed_failure_count']:,} fixed exact failures**. "
            "Six place skills use desired-true spatial predicates as primary success; desired-false grasped is auxiliary."
        )
        selection_state = gr.State({})
        with gr.Row():
            task_filter = gr.Dropdown(tasks, value="All", label="Task")
            skill_filter = gr.Dropdown(skills, value="All", label="Skill")
            page = gr.Slider(1, initial_pages, value=1, step=1, label="Result page")
        filter_summary = gr.Markdown(f"**{len(index.entries):,} failures** · page 1/{initial_pages}")
        selector = gr.Dropdown(initial_choices, value=initial_key, label="Failure row")
        frame_slider = gr.Slider(0, 0, value=0, step=1, label="Aligned frame index")
        progress = gr.Markdown()
        timeline_component = gr.HTML()
        with gr.Row():
            head = gr.Image(label="head RGB", type="numpy")
            left = gr.Image(label="left-wrist RGB", type="numpy")
            right = gr.Image(label="right-wrist RGB", type="numpy")
        status = gr.Markdown()
        with gr.Row():
            with gr.Column():
                row_summary = gr.Markdown()
                context_summary = gr.Markdown()
            diagnostics = gr.JSON(label="Predicate / metric diagnostics")

        filter_outputs = [page, selector, filter_summary]
        task_filter.change(reset_filter, [task_filter, skill_filter], filter_outputs)
        skill_filter.change(reset_filter, [task_filter, skill_filter], filter_outputs)
        page.change(refresh_filter, [task_filter, skill_filter, page], filter_outputs)
        row_outputs = [
            frame_slider,
            head,
            left,
            right,
            row_summary,
            context_summary,
            timeline_component,
            progress,
            diagnostics,
            status,
            selection_state,
        ]
        selector.change(present_row, selector, row_outputs)
        frame_slider.change(
            present_frame,
            [frame_slider, selection_state],
            [head, left, right, timeline_component, progress, status],
        )
        app.load(present_row, gr.State(initial_key), row_outputs)
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--index-cache",
        type=Path,
        help="Optional explicit output path for a validated index cache. No cache is written by default.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Scan/validate the index and exit before UI imports.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    index = scan_failure_index(args.results_root, cache_path=args.index_cache)
    print(json.dumps(index.summary, ensure_ascii=False, indent=2))
    if args.validate_only:
        return
    app = build_app(index, args.dataset_root, args.page_size)
    app.queue(default_concurrency_limit=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
