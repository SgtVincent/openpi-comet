#!/usr/bin/env python3
"""Manual Gradio review viewer for B1K annotation interval anomalies.

The dataset is read-only. This tool builds a deterministic manifest for reversed
skill intervals and malformed primitive ``frame_duration`` values, then stores
human decisions in a separate atomic JSON file. Diagnostic consumer
interpretations are not truth or a repair: source annotations are never changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC
from datetime import datetime
import fcntl
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

DEFAULT_DATASET_ROOT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/outputs/b1k_annotation_anomaly_review_viewer_20260728"
)
DEFAULT_AUDIT_ROOT = Path(
    "/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/outputs/behavior1k_native_success_audit_20260724T180422Z"
)
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "review_manifest.json"
DEFAULT_REVIEWS_PATH = DEFAULT_OUTPUT_DIR / "human_reviews.json"
CAMERAS = ("head", "left_wrist", "right_wrist")
MANIFEST_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1
EXPECTED_ANNOTATION_FILES = 10_000
EXPECTED_REVERSED_SKILLS = 2
EXPECTED_MALFORMED_PRIMITIVES = 595
EXPECTED_MALFORMED_EPISODES = 307
EXPECTED_SHAPE_COUNTS = {"nested_pair_then_scalar": 592, "scalar_then_nested_pair": 3}
EXPECTED_TASK_COUNTS = {"task-0004": 3, "task-0013": 1, "task-0020": 223, "task-0036": 368}
REVIEW_DECISIONS = (
    "unreviewed",
    "confirm_anomaly_no_proposal",
    "propose_correction",
    "needs_reannotation",
    "not_anomaly",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_int_pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_is_int(item) for item in value)


def canonical_segments(raw_duration: Any) -> list[list[int]] | None:
    """Return only structurally canonical integer segments, without normalization."""
    if _is_int_pair(raw_duration):
        return [[int(raw_duration[0]), int(raw_duration[1])]]
    if isinstance(raw_duration, list) and raw_duration and all(_is_int_pair(item) for item in raw_duration):
        return [[int(item[0]), int(item[1])] for item in raw_duration]
    return None


def malformed_primitive_shape(raw_duration: Any) -> str | None:
    if not isinstance(raw_duration, list) or len(raw_duration) != 2:
        return None
    if _is_int_pair(raw_duration[0]) and _is_int(raw_duration[1]):
        return "nested_pair_then_scalar"
    if _is_int(raw_duration[0]) and _is_int_pair(raw_duration[1]):
        return "scalar_then_nested_pair"
    return None


def flatten_ints(value: Any) -> list[int]:
    values: list[int] = []
    if _is_int(value):
        values.append(int(value))
    elif isinstance(value, list):
        for item in value:
            values.extend(flatten_ints(item))
    return values


def mirror_current_duration_consumer(raw_duration: Any) -> list[list[int]]:
    """Mirror the training dataset fallback parser, for diagnostics only."""
    canonical = canonical_segments(raw_duration)
    if canonical is not None:
        return canonical
    ints = flatten_ints(raw_duration)
    return [[min(ints), max(ints)]] if len(ints) >= 2 else []


def mirror_eval_segment_consumer(raw_duration: Any) -> list[int] | None:
    """Mirror eval_segment's recursive flatten + first/last normalization."""
    ints = flatten_ints(raw_duration)
    return [ints[0], ints[-1]] if len(ints) >= 2 else None


def mirror_sweep_duration_consumer(raw_duration: Any) -> dict[str, Any]:
    """Mirror sweep parsing and describe its caller's dynamic-step branch."""
    parsed: list[int] | None = None
    if isinstance(raw_duration, (list, tuple)) and len(raw_duration) == 2:
        try:
            parsed = [int(raw_duration[0]), int(raw_duration[1])]
        except (TypeError, ValueError):
            parsed = None
    duration = None if parsed is None else parsed[1] - parsed[0]
    return {
        "parsed_duration": parsed,
        "duration_frames": duration,
        "dynamic_step_branch": "fallback" if duration is None or duration <= 0 else "duration_times_two",
    }


def duration_diagnostics(raw_duration: Any) -> dict[str, Any]:
    shape = malformed_primitive_shape(raw_duration)
    segments = canonical_segments(raw_duration)
    flat = flatten_ints(raw_duration)
    nested_pairs: list[dict[str, Any]] = []
    scalar_endpoints: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if _is_int_pair(value):
            nested_pairs.append({"path": path, "raw_pair": deepcopy(value)})
        elif _is_int(value):
            scalar_endpoints.append({"path": path, "raw_scalar": int(value)})
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(raw_duration, "frame_duration")
    consumer_output = mirror_current_duration_consumer(raw_duration)
    return {
        "warning": (
            "DIAGNOSTIC ONLY — parser outputs are not ground truth and are not a repair. "
            "The source annotation remains unchanged."
        ),
        "raw_shape": shape or ("canonical" if segments is not None else "other_noncanonical"),
        "strict_canonical_parser": {
            "accepted": segments is not None,
            "segments_without_normalization": segments,
        },
        "current_repository_consumers": [
            {
                "source": "src/behavior/learning/datas/dataset_utils.py::_duration_to_segments",
                "diagnostic_output": consumer_output,
                "behavior": "canonical shapes are preserved; mixed shapes flatten recursively and use min/max",
            },
            {
                "source": "src/behavior/learning/datas/dataset.py::BehaviorLeRobotDataset._duration_to_segments",
                "diagnostic_output": consumer_output,
                "behavior": "canonical shapes are preserved; mixed shapes flatten recursively and use min/max",
            },
            {
                "source": "BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py::_normalize_frame_duration",
                "diagnostic_output": mirror_eval_segment_consumer(raw_duration),
                "behavior": "flatten recursively, then use the first and last integer; ordering is not validated",
            },
            {
                "source": "scripts/run_skill_metric_multinode_sweep.py::parse_frame_duration/get_dynamic_max_steps",
                "diagnostic_output": mirror_sweep_duration_consumer(raw_duration),
                "behavior": (
                    "directly int-cast the two outer values; nested values fail parsing, while non-positive "
                    "parsed durations select the caller-provided fallback step budget"
                ),
            },
        ],
        "flattened_integers_in_raw_order": flat,
        "nested_pair_candidates_from_raw": nested_pairs,
        "scalar_endpoint_candidates_from_raw": scalar_endpoints,
    }


def _compact_item(item: Any, position: int, kind: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"position": position, "kind": kind, "raw": deepcopy(item)}
    index_key = "skill_idx" if kind == "skill" else "primitive_idx"
    description_key = "skill_description" if kind == "skill" else "primitive_description"
    return {
        "position": int(position),
        "kind": kind,
        "item_idx": deepcopy(item.get(index_key, position)),
        "description": deepcopy(item.get(description_key)),
        "object_id": deepcopy(item.get("object_id")),
        "manipulating_object_id": deepcopy(item.get("manipulating_object_id")),
        "skill_idxes": deepcopy(item.get("skill_idxes")),
        "frame_duration": deepcopy(item.get("frame_duration")),
    }


def _neighbors(items: Sequence[Any], position: int, kind: str, radius: int) -> list[dict[str, Any]]:
    lower, upper = max(0, position - radius), min(len(items), position + radius + 1)
    return [_compact_item(items[index], index, kind) for index in range(lower, upper)]


def _related_skills(skills: Sequence[Any], skill_idxes: Any) -> list[dict[str, Any]]:
    wanted = {int(value) for value in skill_idxes or [] if _is_int(value)}
    positions = [
        position
        for position, item in enumerate(skills)
        if isinstance(item, dict) and _is_int(item.get("skill_idx")) and int(item["skill_idx"]) in wanted
    ]
    if not positions:
        return []
    expanded = set(positions)
    expanded.add(max(0, min(positions) - 1))
    expanded.add(min(len(skills) - 1, max(positions) + 1))
    return [_compact_item(skills[position], position, "skill") for position in sorted(expanded)]


def resolve_video_paths(dataset_root: Path, task_dir: str, episode: str) -> dict[str, Path]:
    base = dataset_root / "videos" / task_dir
    return {camera: base / f"observation.images.rgb.{camera}" / f"{episode}.mp4" for camera in CAMERAS}


def _media_record(dataset_root: Path, task_dir: str, episode: str) -> dict[str, dict[str, Any]]:
    return {
        camera: {"path": str(path), "exists": path.is_file()}
        for camera, path in resolve_video_paths(dataset_root, task_dir, episode).items()
    }


def _source_fields(
    task_dir: str,
    episode: str,
    path: Path,
    sha256: str,
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": int(task_dir.split("-")[-1]),
        "task_dir": task_dir,
        "task_name": annotation.get("task_name"),
        "episode": episode,
        "annotation_path": str(path),
        "annotation_sha256": sha256,
        "meta_data": deepcopy(annotation.get("meta_data")),
    }


def _make_reversed(
    *,
    task_dir: str,
    episode: str,
    path: Path,
    sha256: str,
    annotation: Mapping[str, Any],
    position: int,
    item: Mapping[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    raw_duration = deepcopy(item.get("frame_duration"))
    segments = canonical_segments(raw_duration) or []
    item_idx = item.get("skill_idx", position)
    return {
        **_source_fields(task_dir, episode, path, sha256, annotation),
        "kind": "reversed_skill",
        "key": f"reversed_skill|{task_dir}|{episode}|position={position}|skill_idx={item_idx}",
        "item_position": int(position),
        "item_idx": deepcopy(item_idx),
        "raw_duration": raw_duration,
        "raw_item": deepcopy(item),
        "reversed_segments": [segment for segment in segments if segment[0] > segment[1]],
        "skill_context": _neighbors(annotation.get("skill_annotation", []), position, "skill", 2),
        "primitive_context": [],
        "diagnostics": duration_diagnostics(raw_duration),
        "media": _media_record(dataset_root, task_dir, episode),
    }


def _make_malformed(
    *,
    task_dir: str,
    episode: str,
    path: Path,
    sha256: str,
    annotation: Mapping[str, Any],
    position: int,
    item: Mapping[str, Any],
    shape: str,
    dataset_root: Path,
) -> dict[str, Any]:
    raw_duration = deepcopy(item.get("frame_duration"))
    item_idx = item.get("primitive_idx", position)
    return {
        **_source_fields(task_dir, episode, path, sha256, annotation),
        "kind": "malformed_primitive",
        "key": f"malformed_primitive|{task_dir}|{episode}|position={position}|primitive_idx={item_idx}",
        "shape": shape,
        "item_position": int(position),
        "item_idx": deepcopy(item_idx),
        "raw_duration": raw_duration,
        "raw_item": deepcopy(item),
        "primitive_context": _neighbors(annotation.get("primitive_annotation", []), position, "primitive", 1),
        "skill_context": _related_skills(annotation.get("skill_annotation", []), item.get("skill_idxes")),
        "diagnostics": duration_diagnostics(raw_duration),
        "media": _media_record(dataset_root, task_dir, episode),
    }


def validate_manifest(manifest: Mapping[str, Any], *, enforce_expected: bool = True) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')}")
    reversed_entries = manifest.get("reversed_skills")
    primitive_entries = manifest.get("malformed_primitives")
    if not isinstance(reversed_entries, list) or not isinstance(primitive_entries, list):
        raise ValueError("manifest entry lists are missing")
    all_entries = [*reversed_entries, *primitive_entries]
    keys = [entry.get("key") for entry in all_entries]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate item keys")
    for entry in all_entries:
        raw_path = Path(entry["annotation_path"])
        if not raw_path.is_file():
            raise ValueError(f"source annotation missing: {raw_path}")
        if malformed_primitive_shape(entry["raw_duration"]) and entry["kind"] != "malformed_primitive":
            raise ValueError(f"entry kind mismatch: {entry['key']}")
    shape_counts = Counter(entry["shape"] for entry in primitive_entries)
    task_counts = Counter(entry["task_dir"] for entry in primitive_entries)
    primitive_episodes = {(entry["task_dir"], entry["episode"]) for entry in primitive_entries}
    summary = {
        "reversed_skill_items": len(reversed_entries),
        "malformed_primitive_items": len(primitive_entries),
        "malformed_primitive_episodes": len(primitive_episodes),
        "malformed_shape_counts": dict(sorted(shape_counts.items())),
        "malformed_task_counts": dict(sorted(task_counts.items())),
        "all_three_media_present_items": sum(
            all(bool(record.get("exists")) for record in entry["media"].values()) for entry in all_entries
        ),
    }
    if enforce_expected:
        expected = {
            "reversed_skill_items": EXPECTED_REVERSED_SKILLS,
            "malformed_primitive_items": EXPECTED_MALFORMED_PRIMITIVES,
            "malformed_primitive_episodes": EXPECTED_MALFORMED_EPISODES,
            "malformed_shape_counts": EXPECTED_SHAPE_COUNTS,
            "malformed_task_counts": EXPECTED_TASK_COUNTS,
        }
        for key, value in expected.items():
            if summary[key] != value:
                raise ValueError(f"manifest {key} mismatch: expected {value!r}, got {summary[key]!r}")
        known_reversed = {
            (entry["task_dir"], entry["episode"], entry["item_idx"], json.dumps(entry["raw_duration"]))
            for entry in reversed_entries
        }
        required = {
            ("task-0004", "episode_00042230", 15, "[[5949, 6456], [6738, 6459]]"),
            ("task-0049", "episode_00490320", 59, "[13973, 13931]"),
        }
        if known_reversed != required:
            raise ValueError(f"known reversed cases mismatch: {known_reversed!r}")
    return summary


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def build_manifest(dataset_root: Path, *, manifest_path: Path | None = None) -> dict[str, Any]:
    annotations_root = dataset_root / "annotations"
    task_dirs = sorted(annotations_root.glob("task-[0-9][0-9][0-9][0-9]"))
    if len(task_dirs) != 50:
        raise ValueError(f"expected 50 annotation task directories, got {len(task_dirs)}")
    reversed_entries: list[dict[str, Any]] = []
    primitive_entries: list[dict[str, Any]] = []
    source_files = 0
    source_digest = hashlib.sha256()
    for task_path in task_dirs:
        annotation_paths = sorted(task_path.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json"))
        if len(annotation_paths) != 200:
            raise ValueError(f"expected 200 annotations in {task_path}, got {len(annotation_paths)}")
        for path in annotation_paths:
            source_files += 1
            raw_bytes = path.read_bytes()
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            source_digest.update(f"{path.relative_to(dataset_root)}\0{sha256}\n".encode())
            annotation = json.loads(raw_bytes)
            task_dir, episode = task_path.name, path.stem
            skills = annotation.get("skill_annotation", [])
            primitives = annotation.get("primitive_annotation", [])
            for position, item in enumerate(skills):
                if not isinstance(item, dict):
                    continue
                segments = canonical_segments(item.get("frame_duration"))
                if segments and any(start > end for start, end in segments):
                    reversed_entries.append(
                        _make_reversed(
                            task_dir=task_dir,
                            episode=episode,
                            path=path,
                            sha256=sha256,
                            annotation=annotation,
                            position=position,
                            item=item,
                            dataset_root=dataset_root,
                        )
                    )
            for position, item in enumerate(primitives):
                if not isinstance(item, dict):
                    continue
                shape = malformed_primitive_shape(item.get("frame_duration"))
                if shape:
                    primitive_entries.append(
                        _make_malformed(
                            task_dir=task_dir,
                            episode=episode,
                            path=path,
                            sha256=sha256,
                            annotation=annotation,
                            position=position,
                            item=item,
                            shape=shape,
                            dataset_root=dataset_root,
                        )
                    )
    reversed_entries.sort(key=lambda entry: entry["key"])
    primitive_entries.sort(key=lambda entry: entry["key"])
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "accepted_audit_root": str(DEFAULT_AUDIT_ROOT),
        "source_annotation_file_count": source_files,
        "source_annotation_inventory_sha256": source_digest.hexdigest(),
        "reversed_skills": reversed_entries,
        "malformed_primitives": primitive_entries,
    }
    if source_files != EXPECTED_ANNOTATION_FILES:
        raise ValueError(f"expected {EXPECTED_ANNOTATION_FILES} annotation files, got {source_files}")
    manifest["summary"] = validate_manifest(manifest, enforce_expected=True)
    manifest["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest_path is not None:
        _atomic_write_json(manifest_path, manifest)
    return manifest


def load_manifest(path: Path, *, enforce_expected: bool = True) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, enforce_expected=enforce_expected)
    return manifest


@contextmanager
def _review_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_review_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": REVIEW_SCHEMA_VERSION, "reviews": {}}
    store = json.loads(path.read_text(encoding="utf-8"))
    if store.get("schema_version") != REVIEW_SCHEMA_VERSION or not isinstance(store.get("reviews"), dict):
        raise ValueError(f"invalid review store: {path}")
    return store


def parse_json_field(text: str, *, field_name: str) -> Any:
    if not str(text).strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc.msg} at character {exc.pos}") from exc


def upsert_review(
    path: Path,
    *,
    item_key: str,
    item_kind: str,
    source_annotation_path: str,
    source_annotation_sha256: str,
    decision: str,
    proposed_value: Any,
    notes: str,
) -> dict[str, Any]:
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"unknown review decision: {decision}")
    with _review_lock(path):
        store = load_review_store(path)
        now = datetime.now(UTC).isoformat()
        previous = store["reviews"].get(item_key, {})
        review = {
            "item_key": item_key,
            "item_kind": item_kind,
            "source_annotation_path": source_annotation_path,
            "source_annotation_sha256": source_annotation_sha256,
            "decision": decision,
            "proposed_value": deepcopy(proposed_value),
            "notes": str(notes),
            "created_at_utc": previous.get("created_at_utc", now),
            "updated_at_utc": now,
        }
        store["reviews"][item_key] = review
        store["updated_at_utc"] = now
        _atomic_write_json(path, store)
    return review


def review_for_entry(path: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    return load_review_store(path)["reviews"].get(entry["key"], {})


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
            "Interactive viewer dependencies are missing. Reuse the prior viewer environment or install: "
            f"python -m pip install {quoted}"
        )
    return gr, cv2, np


class VideoFrameReader:
    """Read aligned camera frames with visible placeholders on every failure."""

    def __init__(self, cv2: Any, np: Any) -> None:
        self.cv2 = cv2
        self.np = np

    def placeholder(self, camera: str, message: str) -> Any:
        image = self.np.zeros((360, 480, 3), dtype=self.np.uint8)
        image[:, :] = (45, 45, 55)
        self.cv2.putText(image, camera, (20, 48), self.cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        for line_number, line in enumerate(message.splitlines()[:6], 1):
            self.cv2.putText(
                image,
                line[:58],
                (20, 68 + line_number * 34),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
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
                frames.append(self.placeholder(camera, f"MISSING FILE\n{path}"))
                errors.append(f"{camera}: missing file")
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
            except Exception as exc:  # visible degradation is intentional
                frames.append(self.placeholder(camera, f"DECODE ERROR\nframe {frame_idx}\n{exc}"))
                errors.append(f"{camera}: {exc}")
            finally:
                capture.release()
        status = (
            f"All three cameras decoded at frame {frame_idx}."
            if not errors
            else f"Visible media fallback at frame {frame_idx}: " + " | ".join(errors)
        )
        return frames, status


def boundary_choices(entry: Mapping[str, Any]) -> list[tuple[str, int]]:
    choices: list[tuple[str, int]] = []

    def collect(value: Any, path: str) -> None:
        if _is_int(value):
            choices.append((f"RAW {path} = {value}", int(value)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect(item, f"{path}[{index}]")

    collect(entry.get("raw_duration"), "frame_duration")
    for context_name in ("primitive_context", "skill_context"):
        for item in entry.get(context_name, []):
            segments = canonical_segments(item.get("frame_duration"))
            if segments is None:
                continue
            for segment_position, (start, end) in enumerate(segments):
                prefix = f"{context_name}:{item.get('item_idx')} segment[{segment_position}]"
                choices.append((f"CONTEXT {prefix} start = {start}", start))
                choices.append((f"CONTEXT {prefix} end = {end}", end))
    valid = (entry.get("meta_data") or {}).get("valid_duration")
    if _is_int_pair(valid):
        choices.extend(((f"CONTEXT valid start = {valid[0]}", valid[0]), (f"CONTEXT valid end = {valid[1]}", valid[1])))
    seen: set[tuple[str, int]] = set()
    return [choice for choice in choices if not (choice in seen or seen.add(choice))]


def media_paths(entry: Mapping[str, Any]) -> dict[str, Path]:
    return {camera: Path(entry["media"][camera]["path"]) for camera in CAMERAS}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def entry_summary_markdown(entry: Mapping[str, Any]) -> str:
    label = "Reversed skill interval" if entry["kind"] == "reversed_skill" else "Malformed primitive duration"
    lines = [
        f"### {label}",
        f"- **task / episode:** `{entry['task_dir']}` / `{entry['episode']}` — {entry.get('task_name')}",
        f"- **item:** position `{entry['item_position']}`, idx `{entry['item_idx']}`",
        f"- **raw frame_duration:** `{json.dumps(entry['raw_duration'])}`",
        f"- **source:** `{entry['annotation_path']}`",
        f"- **source SHA-256:** `{entry['annotation_sha256']}`",
        f"- **review key:** `{entry['key']}`",
    ]
    if entry.get("shape"):
        lines.insert(3, f"- **malformed shape:** `{entry['shape']}`")
    return "\n".join(lines)


def context_markdown(entry: Mapping[str, Any]) -> str:
    lines = ["### Annotation context (raw intervals)"]
    for title, field in (("Primitive", "primitive_context"), ("Skill", "skill_context")):
        rows = entry.get(field, [])
        if not rows:
            lines.append(f"- **{title}:** none")
            continue
        for row in rows:
            selected = row.get("position") == entry["item_position"] and row.get("kind") in entry["kind"]
            marker = " **← selected**" if selected else ""
            lines.append(
                f"- **{title} {row.get('item_idx')}** {row.get('description')} · "
                f"raw `{json.dumps(row.get('frame_duration'))}` · objects `{json.dumps(row.get('object_id'))}`{marker}"
            )
    return "\n".join(lines)


def timeline_html(entry: Mapping[str, Any], frame_idx: int) -> str:
    """Render canonical context bars and raw selected endpoints without repairing either."""
    endpoints = [value for _label, value in boundary_choices(entry)]
    if not endpoints:
        return "<div><strong>No integer boundaries available.</strong></div>"
    lower, upper = min(endpoints), max(endpoints)
    if upper <= lower:
        upper = lower + 1
    span = upper - lower
    rows: list[str] = []
    palette = {"skill": "#2563eb", "primitive": "#6b7280"}
    for field, kind in (("skill_context", "skill"), ("primitive_context", "primitive")):
        for item in entry.get(field, []):
            segments = canonical_segments(item.get("frame_duration"))
            if segments is None:
                continue
            for start, end in segments:
                if end < start:
                    rows.append(
                        f'<div style="color:#b91c1c"><b>{kind} {escape(str(item.get("item_idx")))}</b>: '
                        f"raw reversed endpoints [{start}, {end}] — no interval bar drawn</div>"
                    )
                    continue
                left, width = (start - lower) / span * 100, max(0.25, (end - start) / span * 100)
                rows.append(
                    '<div style="position:relative;height:25px;border-bottom:1px solid #e5e7eb">'
                    f'<div title="raw [{start}, {end}]" style="position:absolute;left:{left:.4f}%;width:{width:.4f}%;'
                    f'height:20px;background:{palette[kind]};color:white;overflow:hidden;font-size:11px;padding:2px">'
                    f"{escape(kind)} {escape(str(item.get('item_idx')))}</div></div>"
                )
    raw_markers = []
    for label, value in boundary_choices(entry):
        if not label.startswith("RAW"):
            continue
        left = (value - lower) / span * 100
        raw_markers.append(
            f'<div title="{escape(label)}" style="position:absolute;left:{left:.4f}%;top:0;bottom:0;'
            'width:2px;background:#dc2626"></div>'
        )
    cursor = min(max((frame_idx - lower) / span * 100, 0), 100)
    return (
        '<div style="border:1px solid #9ca3af;background:#f9fafb;padding:6px">'
        '<div style="color:#b91c1c;font-weight:600">Red lines are raw selected endpoints; no repair is drawn.</div>'
        f'<div style="position:relative">{"".join(rows)}{"".join(raw_markers)}'
        f'<div style="position:absolute;left:{cursor:.4f}%;top:-4px;bottom:-4px;width:3px;background:#111827"></div></div>'
        f'<div style="display:flex;justify-content:space-between;font-family:monospace"><span>{lower}</span>'
        f"<b>frame {frame_idx}</b><span>{upper}</span></div></div>"
    )


def format_entry_label(entry: Mapping[str, Any]) -> str:
    shape = f" | {entry['shape']}" if entry.get("shape") else ""
    return (
        f"{entry['task_dir']} | {entry['episode']} | {entry['kind']}[{entry['item_idx']}]"
        f"{shape} | raw {json.dumps(entry['raw_duration'], separators=(',', ':'))}"
    )


def filter_primitive_entries(
    entries: Sequence[Mapping[str, Any]],
    reviews_path: Path,
    task_dir: str = "All",
    shape: str = "All",
    decision: str = "All",
) -> list[Mapping[str, Any]]:
    reviews = load_review_store(reviews_path)["reviews"]
    return [
        entry
        for entry in entries
        if (task_dir == "All" or entry["task_dir"] == task_dir)
        and (shape == "All" or entry["shape"] == shape)
        and (decision == "All" or reviews.get(entry["key"], {}).get("decision", "unreviewed") == decision)
    ]


def page_choices(entries: Sequence[Mapping[str, Any]], page: int, page_size: int) -> tuple[list[Any], int, int]:
    total_pages = max(1, math.ceil(len(entries) / page_size))
    current_page = min(max(int(page), 1), total_pages)
    start = (current_page - 1) * page_size
    choices = [(format_entry_label(entry), entry["key"]) for entry in entries[start : start + page_size]]
    return choices, current_page, total_pages


def selection_payload(entry: Mapping[str, Any], reviews_path: Path) -> dict[str, Any]:
    choices = boundary_choices(entry)
    default_frame = choices[0][1] if choices else 0
    review = review_for_entry(reviews_path, entry)
    return {
        "entry": deepcopy(entry),
        "boundary_choices": choices,
        "default_frame": default_frame,
        "review": review,
        "review_decision": review.get("decision", "unreviewed"),
        "proposed_json": _json_text(review.get("proposed_value")) if review.get("proposed_value") is not None else "",
        "notes": review.get("notes", ""),
    }


def build_app(
    manifest: Mapping[str, Any],
    dataset_root: Path,
    reviews_path: Path,
    *,
    page_size: int = 50,
) -> Any:
    gr, cv2, np = _load_ui_dependencies()
    reader = VideoFrameReader(cv2, np)
    reversed_entries = manifest["reversed_skills"]
    primitive_entries = manifest["malformed_primitives"]
    entries_by_key = {entry["key"]: entry for entry in [*reversed_entries, *primitive_entries]}

    def present_row(row_key: str | None) -> tuple[Any, ...]:
        if not row_key or row_key not in entries_by_key:
            empty = reader.placeholder("no selection", "No anomaly selected")
            return (
                gr.update(minimum=0, maximum=0, value=0),
                gr.update(choices=[], value=None),
                empty,
                empty,
                empty,
                "No anomaly selected.",
                "",
                {},
                {},
                "",
                "Select an anomaly.",
                {},
                "unreviewed",
                "",
                "",
                "No saved review loaded.",
            )
        entry = entries_by_key[row_key]
        payload = selection_payload(entry, reviews_path)
        choices = payload["boundary_choices"]
        default_frame = payload["default_frame"]
        values = [value for _label, value in choices] or [0]
        frames, media_status = reader.read_aligned(media_paths(entry), default_frame)
        saved = payload["review"]
        saved_status = (
            f"Loaded saved review from `{reviews_path}` · updated `{saved.get('updated_at_utc')}`."
            if saved
            else f"No saved review for this item. Saves go to `{reviews_path}`."
        )
        state = {"item_key": entry["key"]}
        return (
            gr.update(minimum=min(values), maximum=max(values), value=default_frame, step=1),
            gr.update(choices=choices, value=default_frame),
            *frames,
            entry_summary_markdown(entry),
            context_markdown(entry),
            entry["raw_item"],
            entry["diagnostics"],
            timeline_html(entry, default_frame),
            media_status,
            state,
            payload["review_decision"],
            payload["proposed_json"],
            payload["notes"],
            saved_status,
        )

    def present_frame(frame_idx: int, state: Mapping[str, Any]) -> tuple[Any, ...]:
        if not state or state.get("item_key") not in entries_by_key:
            empty = reader.placeholder("no selection", "No anomaly selected")
            return empty, empty, empty, "", "Select an anomaly."
        entry = entries_by_key[state["item_key"]]
        frame_idx = int(frame_idx)
        frames, status = reader.read_aligned(media_paths(entry), frame_idx)
        return *frames, timeline_html(entry, frame_idx), status

    def jump_boundary(boundary: int, state: Mapping[str, Any]) -> tuple[Any, ...]:
        if boundary is None:
            boundary = 0
        frames_and_context = present_frame(int(boundary), state)
        return gr.update(value=int(boundary)), *frames_and_context

    def save_current(
        state: Mapping[str, Any],
        decision: str,
        proposed_json: str,
        notes: str,
    ) -> str:
        if not state or state.get("item_key") not in entries_by_key:
            return "**Save failed:** select an anomaly first."
        entry = entries_by_key[state["item_key"]]
        try:
            current_sha256 = hashlib.sha256(Path(entry["annotation_path"]).read_bytes()).hexdigest()
            if current_sha256 != entry["annotation_sha256"]:
                raise ValueError(
                    "source annotation hash changed since manifest construction; rebuild manifest before review"
                )
            proposed = parse_json_field(proposed_json, field_name="proposed value")
            if decision == "propose_correction" and proposed is None:
                raise ValueError("decision 'propose_correction' requires proposed JSON")
            review = upsert_review(
                reviews_path,
                item_key=entry["key"],
                item_kind=entry["kind"],
                source_annotation_path=entry["annotation_path"],
                source_annotation_sha256=entry["annotation_sha256"],
                decision=decision,
                proposed_value=proposed,
                notes=notes,
            )
        except Exception as exc:
            return f"**Save failed:** `{exc}`"
        return (
            f"**Saved safely:** `{review['decision']}` at `{review['updated_at_utc']}` to `{reviews_path}`. "
            "Dataset annotation was not modified."
        )

    task_choices = ["All", *sorted({entry["task_dir"] for entry in primitive_entries})]
    shape_choices = ["All", *sorted({entry["shape"] for entry in primitive_entries})]
    initial_primitive_choices, _, initial_pages = page_choices(primitive_entries, 1, page_size)
    initial_primitive_key = initial_primitive_choices[0][1] if initial_primitive_choices else None
    reversed_choices = [(format_entry_label(entry), entry["key"]) for entry in reversed_entries]
    initial_reversed_key = reversed_choices[0][1] if reversed_choices else None

    def refresh_primitives(task_dir: str, shape: str, decision: str, page: int) -> tuple[Any, Any, str]:
        filtered = filter_primitive_entries(primitive_entries, reviews_path, task_dir, shape, decision)
        choices, current_page, total_pages = page_choices(filtered, page, page_size)
        value = choices[0][1] if choices else None
        reviewed = sum(
            review_for_entry(reviews_path, entry).get("decision", "unreviewed") != "unreviewed" for entry in filtered
        )
        summary = (
            f"**{len(filtered):,} malformed items** · **{reviewed:,} reviewed** · "
            f"page {current_page}/{total_pages} · {page_size} items/page"
        )
        return (
            gr.update(minimum=1, maximum=total_pages, value=current_page),
            gr.update(choices=choices, value=value),
            summary,
        )

    def reset_primitives(task_dir: str, shape: str, decision: str) -> tuple[Any, Any, str]:
        return refresh_primitives(task_dir, shape, decision, 1)

    css = """
    .raw-panel { border: 1px solid #fecaca; border-top: 5px solid #b91c1c; background: #fef2f2; padding: 10px; }
    .diagnostic-panel { border: 1px solid #fde68a; border-top: 5px solid #d97706; background: #fffbeb; padding: 10px; }
    .proposal-panel { border: 1px solid #bfdbfe; border-top: 5px solid #2563eb; background: #eff6ff; padding: 10px; }
    """
    with gr.Blocks(title="B1K annotation anomaly review viewer", css=css) as app:
        gr.Markdown(
            "# B1K annotation anomaly review viewer\n"
            f"Read-only source: **{manifest['source_annotation_file_count']:,} annotations** · "
            f"**{len(reversed_entries)} reversed skill items** · "
            f"**{len(primitive_entries)} malformed primitive items across "
            f"{manifest['summary']['malformed_primitive_episodes']} episodes**. "
            "Red = raw annotation, amber = diagnostic parser behavior, blue = human proposal. "
            "Nothing here repairs or writes the dataset."
        )

        def build_review_tab(
            *,
            selector_choices: list[Any],
            selector_value: str | None,
            selector_label: str,
            proposed_label: str,
        ) -> dict[str, Any]:
            state = gr.State({})
            selector = gr.Dropdown(selector_choices, value=selector_value, label=selector_label)
            boundary = gr.Dropdown([], label="Relevant raw/context boundary (quick jump)")
            frame = gr.Slider(0, 0, value=0, step=1, label="Aligned frame index")
            with gr.Row():
                head = gr.Image(label="head RGB", type="numpy")
                left = gr.Image(label="left-wrist RGB", type="numpy")
                right = gr.Image(label="right-wrist RGB", type="numpy")
            media_status = gr.Markdown()
            timeline = gr.HTML()
            with gr.Row():
                with gr.Column():
                    summary = gr.Markdown()
                    context = gr.Markdown()
                with gr.Column(elem_classes="raw-panel"):
                    gr.Markdown("### RAW SOURCE ANNOTATION (read-only)")
                    raw = gr.JSON(label="Exact raw item")
                with gr.Column(elem_classes="diagnostic-panel"):
                    gr.Markdown("### DIAGNOSTIC INTERPRETATIONS (not truth)")
                    diagnostics = gr.JSON(label="Current consumer diagnostics")
            with gr.Column(elem_classes="proposal-panel"):
                gr.Markdown("### HUMAN-PROPOSED REVIEW (separate artifact only)")
                with gr.Row():
                    decision = gr.Dropdown(REVIEW_DECISIONS, value="unreviewed", label="Review decision")
                    proposal = gr.Textbox(label=proposed_label, lines=4, placeholder="Valid JSON; never auto-applied")
                notes = gr.Textbox(label="Reviewer notes", lines=4)
                save = gr.Button("Save / update review artifact", variant="primary")
                save_status = gr.Markdown()
            row_outputs = [
                frame,
                boundary,
                head,
                left,
                right,
                summary,
                context,
                raw,
                diagnostics,
                timeline,
                media_status,
                state,
                decision,
                proposal,
                notes,
                save_status,
            ]
            selector.change(present_row, selector, row_outputs)
            frame.change(present_frame, [frame, state], [head, left, right, timeline, media_status])
            boundary.change(jump_boundary, [boundary, state], [frame, head, left, right, timeline, media_status])
            save.click(save_current, [state, decision, proposal, notes], save_status)
            return {"selector": selector, "row_outputs": row_outputs}

        with gr.Tab("A · Reversed Skill Review"):
            gr.Markdown(
                "Inspect both reversed cases with raw multi/single interval structure, neighboring skills, "
                "all endpoint quick-jumps, and aligned three-view frames."
            )
            reversed_ui = build_review_tab(
                selector_choices=reversed_choices,
                selector_value=initial_reversed_key,
                selector_label="Reversed skill item",
                proposed_label="Corrected-boundary proposal JSON",
            )

        with gr.Tab("B · Malformed Primitive Review"):
            gr.Markdown(
                "Review all 595 mixed-schema primitive items. Consumer min/max output is shown only as a "
                "diagnostic; use raw nested/scalar endpoints and visual evidence for human judgment."
            )
            with gr.Row():
                task_filter = gr.Dropdown(task_choices, value="All", label="Task")
                shape_filter = gr.Dropdown(shape_choices, value="All", label="Malformed shape")
                decision_filter = gr.Dropdown(["All", *REVIEW_DECISIONS], value="All", label="Saved decision")
                page = gr.Slider(1, initial_pages, value=1, step=1, label="Result page")
            filter_summary = gr.Markdown(
                f"**{len(primitive_entries):,} malformed items** · page 1/{initial_pages} · {page_size} items/page"
            )
            primitive_ui = build_review_tab(
                selector_choices=initial_primitive_choices,
                selector_value=initial_primitive_key,
                selector_label="Malformed primitive item",
                proposed_label="Proposed canonical schema JSON",
            )
            filter_outputs = [page, primitive_ui["selector"], filter_summary]
            task_filter.change(reset_primitives, [task_filter, shape_filter, decision_filter], filter_outputs)
            shape_filter.change(reset_primitives, [task_filter, shape_filter, decision_filter], filter_outputs)
            decision_filter.change(reset_primitives, [task_filter, shape_filter, decision_filter], filter_outputs)
            page.change(refresh_primitives, [task_filter, shape_filter, decision_filter, page], filter_outputs)

        app.load(present_row, gr.State(initial_reversed_key), reversed_ui["row_outputs"])
        app.load(present_row, gr.State(initial_primitive_key), primitive_ui["row_outputs"])
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--page-size", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest or (args.output_dir / "review_manifest.json")
    reviews_path = args.reviews or (args.output_dir / "human_reviews.json")
    if args.rebuild_manifest or not manifest_path.is_file():
        manifest = build_manifest(args.dataset_root, manifest_path=manifest_path)
    else:
        manifest = load_manifest(manifest_path, enforce_expected=True)
    print(
        json.dumps(
            {**manifest["summary"], "manifest_path": str(manifest_path), "reviews_path": str(reviews_path)}, indent=2
        )
    )
    if args.validate_only:
        return
    app = build_app(manifest, args.dataset_root, reviews_path, page_size=args.page_size)
    app.queue(default_concurrency_limit=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(args.dataset_root), str(args.output_dir)],
    )


if __name__ == "__main__":
    main()
