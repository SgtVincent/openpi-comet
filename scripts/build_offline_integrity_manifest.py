#!/usr/bin/env python3
"""Build an offline integrity manifest for BEHAVIOR segment samples.

This scanner is intentionally limited to offline data-integrity checks. It does
not use simulator replay, metric success, policy rollout success, or restore
pipeline status to decide whether a segment should be excluded from training.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_PARQUET_COLUMNS = ("timestamp", "observation.state", "action")
VIDEO_FIELDS = (
    "original_rgb_head",
    "original_rgb_left_wrist",
    "original_rgb_right_wrist",
    "original_depth_head",
)
PATH_FIELDS = ("episode_json", "episode_parquet", "rawdata_hdf5", *VIDEO_FIELDS)


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "segments", "items", "manifest"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unsupported manifest schema: {path}")


def parse_frame_range(record: dict[str, Any]) -> tuple[int, int] | None:
    start = record.get("frame_start")
    end = record.get("frame_end")
    try:
        if start is not None and end is not None:
            s, e = int(start), int(end)
            return (s, e) if s <= e else (e, s)
    except (TypeError, ValueError):
        pass

    duration = record.get("frame_duration")
    if isinstance(duration, str):
        try:
            duration = ast.literal_eval(duration)
        except (SyntaxError, ValueError):
            return None
    if isinstance(duration, list | tuple) and len(duration) == 2:
        try:
            s, e = int(duration[0]), int(duration[1])
            return (s, e) if s <= e else (e, s)
        except (TypeError, ValueError):
            return None
    return None


def episode_index(record: dict[str, Any]) -> int | None:
    for key in ("episode_index", "demo_id", "episode_id"):
        value = record.get(key)
        if value not in (None, ""):
            try:
                return int(str(value))
            except (TypeError, ValueError):
                continue
    return None


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def list_value(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, list):
        return value
    return None


def check_numeric_vectors(values: list[Any], *, name: str) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    lengths: Counter[int] = Counter()
    nonfinite = 0
    empty = 0
    for value in values:
        vector = list_value(value)
        if vector is None:
            issues.append(f"{name}_not_list")
            continue
        lengths[len(vector)] += 1
        if not vector:
            empty += 1
        nonfinite += sum(1 for item in vector if not finite_number(item))
    if empty:
        issues.append(f"{name}_empty_vector")
    if len(lengths) > 1:
        issues.append(f"{name}_inconsistent_vector_length")
    if nonfinite:
        issues.append(f"{name}_nonfinite_values")
    return sorted(set(issues)), {"length_counts": dict(sorted(lengths.items())), "nonfinite_values": nonfinite}


def check_paths(record: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    issues: list[str] = []
    paths: dict[str, str] = {}
    for field in PATH_FIELDS:
        value = record.get(field)
        if value in (None, ""):
            issues.append(f"missing_{field}_path")
            continue
        path = Path(str(value))
        paths[field] = str(path)
        if not path.exists():
            issues.append(f"missing_{field}")
        elif path.is_file() and path.stat().st_size == 0:
            issues.append(f"empty_{field}")
    return issues, paths


def check_annotation(path: Path, record: dict[str, Any], frame_range: tuple[int, int] | None) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    details: dict[str, Any] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"annotation_unreadable:{type(exc).__name__}"], details

    task_duration = data.get("meta_data", {}).get("task_duration")
    valid_duration = data.get("meta_data", {}).get("valid_duration")
    skill_annotation = data.get("skill_annotation")
    details.update(
        {
            "task_duration": task_duration,
            "valid_duration": valid_duration,
            "skill_annotation_count": len(skill_annotation) if isinstance(skill_annotation, list) else None,
        }
    )
    if not isinstance(skill_annotation, list):
        issues.append("annotation_missing_skill_annotation")
        skill_annotation = []
    if frame_range is None:
        issues.append("invalid_frame_range")
        return issues, details

    start, end = frame_range
    if start < 0 or end < 0:
        issues.append("negative_frame_range")
    annotation_lower_bound = 0
    annotation_upper_bound = None
    if isinstance(valid_duration, list) and len(valid_duration) == 2:
        try:
            annotation_lower_bound = int(valid_duration[0])
            annotation_upper_bound = int(valid_duration[1])
        except (TypeError, ValueError):
            annotation_upper_bound = None
    if annotation_upper_bound is None:
        try:
            annotation_upper_bound = int(task_duration)
        except (TypeError, ValueError):
            annotation_upper_bound = None
    details["annotation_frame_lower_bound"] = annotation_lower_bound
    details["annotation_frame_upper_bound"] = annotation_upper_bound
    if annotation_upper_bound is None:
        issues.append("annotation_missing_frame_upper_bound")
    else:
        if start < annotation_lower_bound:
            issues.append("frame_range_starts_before_annotation_valid_duration")
        if annotation_upper_bound < end:
            issues.append("frame_range_exceeds_annotation_frame_upper_bound")

    skill_idx = record.get("skill_idx")
    matched = None
    try:
        wanted_skill_idx = int(skill_idx)
    except (TypeError, ValueError):
        wanted_skill_idx = None
    if wanted_skill_idx is not None:
        for item in skill_annotation:
            if isinstance(item, dict) and item.get("skill_idx") == wanted_skill_idx:
                matched = item
                break
        if matched is None:
            issues.append("annotation_missing_skill_idx")
        else:
            annotated_range = parse_frame_range(matched)
            details["annotated_frame_duration"] = list(annotated_range) if annotated_range else None
            if annotated_range != frame_range:
                issues.append("annotation_frame_range_mismatch")
    return issues, details


def check_parquet(path: Path, frame_range: tuple[int, int] | None) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    details: dict[str, Any] = {}
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return ["pyarrow_missing"], details

    try:
        metadata = pq.read_metadata(path)
        schema_names = set(metadata.schema.to_arrow_schema().names)
    except Exception as exc:
        return [f"parquet_unreadable:{type(exc).__name__}"], details

    details["parquet_rows"] = metadata.num_rows
    missing_columns = [name for name in REQUIRED_PARQUET_COLUMNS if name not in schema_names]
    if missing_columns:
        issues.extend(f"parquet_missing_column:{name}" for name in missing_columns)
        return issues, details

    try:
        table = pq.read_table(path, columns=list(REQUIRED_PARQUET_COLUMNS))
    except Exception as exc:
        return [f"parquet_column_read_failed:{type(exc).__name__}"], details

    rows = table.num_rows
    details["parquet_rows"] = rows
    if frame_range is not None:
        start, end = frame_range
        if end > rows:
            issues.append("frame_range_exceeds_parquet_rows")
        segment_start = max(0, min(start, rows))
        segment_end = max(segment_start, min(end + 1, rows))
    else:
        segment_start, segment_end = 0, rows

    timestamps = table["timestamp"].to_pylist()
    nonfinite_ts = sum(1 for value in timestamps if not finite_number(value))
    if nonfinite_ts:
        issues.append("timestamp_nonfinite")
    if any(float(timestamps[i]) < float(timestamps[i - 1]) for i in range(1, len(timestamps)) if finite_number(timestamps[i]) and finite_number(timestamps[i - 1])):
        issues.append("timestamp_not_monotonic")
    details["timestamp_start"] = timestamps[0] if timestamps else None
    details["timestamp_end"] = timestamps[-1] if timestamps else None

    state_values = table["observation.state"].slice(segment_start, segment_end - segment_start).to_pylist()
    action_values = table["action"].slice(segment_start, segment_end - segment_start).to_pylist()
    state_issues, state_details = check_numeric_vectors(state_values, name="state")
    action_issues, action_details = check_numeric_vectors(action_values, name="action")
    issues.extend(state_issues)
    issues.extend(action_issues)
    details["state"] = state_details
    details["action"] = action_details
    return sorted(set(issues)), details


def check_hdf5(path: Path, frame_range: tuple[int, int] | None) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    details: dict[str, Any] = {}
    try:
        import h5py
    except ModuleNotFoundError:
        return ["h5py_missing"], details

    try:
        with h5py.File(path, "r") as handle:
            demo = handle.get("data/demo_0")
            if demo is None:
                return ["hdf5_missing_data_demo_0"], details
            pcd_dataset = demo.get("robot_r1::fused_pcd")
            if pcd_dataset is not None:
                details["hdf5_fused_pcd_shape"] = list(pcd_dataset.shape)
                if frame_range is not None and pcd_dataset.shape and frame_range[1] > int(pcd_dataset.shape[0]):
                    issues.append("frame_range_exceeds_hdf5_fused_pcd_rows")
            for key in ("action", "state"):
                dataset = demo.get(key)
                if dataset is None:
                    if pcd_dataset is not None:
                        details[f"hdf5_{key}_source"] = "absent_in_pcd_container"
                        continue
                    issues.append(f"hdf5_missing_{key}")
                    continue
                details[f"hdf5_{key}_shape"] = list(dataset.shape)
                if frame_range is not None and dataset.shape and frame_range[1] > int(dataset.shape[0]):
                    issues.append(f"frame_range_exceeds_hdf5_{key}_rows")
    except Exception as exc:
        return [f"hdf5_unreadable:{type(exc).__name__}"], details
    return sorted(set(issues)), details


def decode_video_samples(path: Path, sample_count: int) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    details: dict[str, Any] = {}
    try:
        import av
    except (ImportError, ModuleNotFoundError) as exc:
        details["pyav_unavailable"] = f"{type(exc).__name__}: {exc}"
        return decode_video_samples_cv2(path, sample_count, details)

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0] if container.streams.video else None
            if stream is None:
                return ["video_missing_stream"], details
            details.update(
                {
                    "codec": stream.codec_context.name,
                    "width": stream.width,
                    "height": stream.height,
                    "frames": int(stream.frames or 0),
                    "average_rate": str(stream.average_rate) if stream.average_rate else "",
                }
            )
            decoded = []
            max_decode = max(1, sample_count)
            for idx, frame in enumerate(container.decode(stream)):
                if idx >= max_decode:
                    break
                array = frame.to_ndarray(format="rgb24")
                decoded.append(
                    {
                        "mean": float(array.mean()),
                        "std": float(array.std()),
                    }
                )
            details["decoded_samples"] = len(decoded)
            if not decoded:
                issues.append("video_decode_no_frames")
            elif all(sample["std"] < 1.0 for sample in decoded):
                issues.append("video_likely_blank")
            elif len(decoded) >= 2 and all(abs(decoded[i]["mean"] - decoded[0]["mean"]) < 1e-6 and abs(decoded[i]["std"] - decoded[0]["std"]) < 1e-6 for i in range(1, len(decoded))):
                issues.append("video_initial_frames_frozen")
    except Exception as exc:
        return [f"video_decode_failed:{type(exc).__name__}"], details
    return sorted(set(issues)), details


def decode_video_samples_cv2(
    path: Path,
    sample_count: int,
    details: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    details = {} if details is None else details
    try:
        import cv2
    except (ImportError, ModuleNotFoundError) as exc:
        details["video_check_skipped"] = f"cv2_unavailable:{type(exc).__name__}: {exc}"
        return [], details

    if not hasattr(cv2, "VideoCapture"):
        details["video_check_skipped"] = "cv2_unavailable:missing_VideoCapture"
        return [], details
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return ["video_open_failed"], details
        details.update(
            {
                "codec": "cv2",
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "average_rate": str(capture.get(cv2.CAP_PROP_FPS) or ""),
            }
        )
        decoded = []
        for _ in range(max(1, sample_count)):
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append({"mean": float(frame.mean()), "std": float(frame.std())})
        details["decoded_samples"] = len(decoded)
        if not decoded:
            issues.append("video_decode_no_frames")
        elif all(sample["std"] < 1.0 for sample in decoded):
            issues.append("video_likely_blank")
    finally:
        capture.release()
    return sorted(set(issues)), details


def check_videos(paths: dict[str, str], sample_count: int) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    details: dict[str, Any] = {}
    for field in VIDEO_FIELDS:
        value = paths.get(field)
        if not value:
            continue
        field_issues, field_details = decode_video_samples(Path(value), sample_count)
        issues.extend(f"{field}:{issue}" for issue in field_issues)
        details[field] = field_details
    return sorted(set(issues)), details


def build_output_record(
    record: dict[str, Any],
    *,
    source: Path,
    video_sample_frames: int,
    skip_video_checks: bool = False,
) -> dict[str, Any]:
    frame_range = parse_frame_range(record)
    ep_index = episode_index(record)
    issues: list[str] = []
    details: dict[str, Any] = {}

    path_issues, paths = check_paths(record)
    issues.extend(path_issues)
    details["paths"] = paths

    if frame_range is None:
        issues.append("invalid_frame_range")
    if ep_index is None:
        issues.append("invalid_episode_index")

    episode_json = paths.get("episode_json")
    if episode_json and Path(episode_json).exists():
        annotation_issues, annotation_details = check_annotation(Path(episode_json), record, frame_range)
        issues.extend(annotation_issues)
        details["annotation"] = annotation_details

    episode_parquet = paths.get("episode_parquet")
    if episode_parquet and Path(episode_parquet).exists():
        parquet_issues, parquet_details = check_parquet(Path(episode_parquet), frame_range)
        issues.extend(parquet_issues)
        details["parquet"] = parquet_details

    rawdata_hdf5 = paths.get("rawdata_hdf5")
    if rawdata_hdf5 and Path(rawdata_hdf5).exists():
        hdf5_issues, hdf5_details = check_hdf5(Path(rawdata_hdf5), frame_range)
        issues.extend(hdf5_issues)
        details["hdf5"] = hdf5_details

    if skip_video_checks:
        video_issues, video_details = [], {"video_check_skipped": "disabled_by_cli"}
    else:
        video_issues, video_details = check_videos(paths, video_sample_frames)
    issues.extend(video_issues)
    details["videos"] = video_details

    hard_issues = sorted(set(issues))
    bucket = "offline_hard_exclude" if hard_issues else "clean_pass"
    frame_start, frame_end = frame_range if frame_range else (None, None)
    return {
        "sample_id": record.get("sample_id"),
        "task_id": record.get("task_id"),
        "task_name": record.get("task_name", ""),
        "demo_id": str(record.get("demo_id") or record.get("episode_id") or ""),
        "episode_index": ep_index,
        "skill_idx": record.get("skill_idx"),
        "skill_name": record.get("skill_desc") or record.get("skill_name") or "",
        "skill_type": record.get("skill_type") or "",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "result_type": bucket,
        "recommended_bucket": bucket,
        "recommendation": bucket,
        "confidence": "high" if hard_issues else "medium",
        "train_disposition": "exclude" if hard_issues else "keep",
        "offline_integrity_issues": hard_issues,
        "attributions": hard_issues,
        "recommended_actions": ["fix_or_replace_offline_data"] if hard_issues else [],
        "source_manifest": str(source),
        "details": details,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "task_id",
        "task_name",
        "demo_id",
        "episode_index",
        "skill_idx",
        "skill_name",
        "skill_type",
        "frame_start",
        "frame_end",
        "result_type",
        "recommended_bucket",
        "confidence",
        "train_disposition",
        "offline_integrity_issues",
        "recommended_actions",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def build_summary(source: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_manifest": str(source),
        "total_segments": len(rows),
        "bucket_counts": dict(sorted(Counter(row["recommended_bucket"] for row in rows).items())),
        "train_disposition_counts": dict(sorted(Counter(row["train_disposition"] for row in rows).items())),
        "issue_counts": dict(
            sorted(Counter(issue for row in rows for issue in row.get("offline_integrity_issues", [])).items())
        ),
        "task_counts": dict(sorted(Counter(str(row.get("task_name") or row.get("task_id") or "") for row in rows).items())),
        "note": "Offline integrity only; simulator replay, policy success, metric success, and restore fallback are intentionally excluded from hard-exclude decisions.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Segment source manifest JSON")
    parser.add_argument("--out-json", required=True, type=Path, help="Output manifest JSON")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional CSV mirror")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON")
    parser.add_argument("--video-sample-frames", type=int, default=5, help="Frames to decode from the start of each video")
    parser.add_argument("--skip-video-checks", action="store_true", help="Skip video decoding checks; path existence is still verified")
    args = parser.parse_args()

    source = args.input.resolve()
    records = load_records(source)
    rows = [
        build_output_record(
            record,
            source=source,
            video_sample_frames=max(1, args.video_sample_frames),
            skip_video_checks=args.skip_video_checks,
        )
        for record in records
    ]
    def sort_sample_id(row: dict[str, Any]) -> tuple[int, str]:
        sample_id = row.get("sample_id")
        if sample_id is None:
            return 10**9, ""
        try:
            return int(sample_id), str(sample_id)
        except (TypeError, ValueError):
            return 10**9, str(sample_id)

    rows.sort(key=sort_sample_id)
    write_json(args.out_json, rows)
    if args.out_csv is not None:
        write_csv(args.out_csv, rows)
    summary = build_summary(source, rows)
    if args.summary_json is not None:
        write_json(args.summary_json, summary)
    print(
        json.dumps(
            {
                "input": str(source),
                "output_json": str(args.out_json.resolve()),
                "total_segments": len(rows),
                "bucket_counts": summary["bucket_counts"],
                "issue_counts": summary["issue_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
