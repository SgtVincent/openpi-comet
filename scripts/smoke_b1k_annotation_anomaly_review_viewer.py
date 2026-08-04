#!/usr/bin/env python3
"""Focused real-data smoke test for the B1K annotation anomaly viewer."""

# The smoke intentionally exercises private dependency/atomic-write helpers.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path``
import socket
import sys
import tempfile
import urllib.request

MODULE_PATH = Path(__file__).with_name("b1k_annotation_anomaly_review_viewer.py")
SPEC = importlib.util.spec_from_file_location("b1k_annotation_anomaly_review_viewer", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
viewer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = viewer
SPEC.loader.exec_module(viewer)


def free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=viewer.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=viewer.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-ui", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_dir / "review_manifest.json"
    manifest = viewer.build_manifest(args.dataset_root, manifest_path=manifest_path)
    summary = viewer.validate_manifest(manifest, enforce_expected=True)

    assert summary["reversed_skill_items"] == 2
    assert summary["malformed_primitive_items"] == 595
    assert summary["malformed_primitive_episodes"] == 307
    assert summary["malformed_shape_counts"] == viewer.EXPECTED_SHAPE_COUNTS
    assert summary["malformed_task_counts"] == viewer.EXPECTED_TASK_COUNTS

    representatives = [*manifest["reversed_skills"]]
    representatives.extend(
        next(entry for entry in manifest["malformed_primitives"] if entry["shape"] == shape)
        for shape in viewer.EXPECTED_SHAPE_COUNTS
    )
    representative_by_key = {entry["key"]: entry for entry in representatives}

    def consumer_output(entry: dict, source_suffix: str):
        consumer = next(
            item
            for item in entry["diagnostics"]["current_repository_consumers"]
            if item["source"].endswith(source_suffix)
        )
        return consumer["diagnostic_output"]

    reversed_multi = representative_by_key[
        "reversed_skill|task-0004|episode_00042230|position=15|skill_idx=15"
    ]
    assert consumer_output(reversed_multi, "eval_segment.py::_normalize_frame_duration") == [5949, 6459]
    assert consumer_output(reversed_multi, "parse_frame_duration/get_dynamic_max_steps") == {
        "parsed_duration": None,
        "duration_frames": None,
        "dynamic_step_branch": "fallback",
    }
    reversed_single = representative_by_key[
        "reversed_skill|task-0049|episode_00490320|position=59|skill_idx=59"
    ]
    assert consumer_output(reversed_single, "eval_segment.py::_normalize_frame_duration") == [13973, 13931]
    assert consumer_output(reversed_single, "parse_frame_duration/get_dynamic_max_steps") == {
        "parsed_duration": [13973, 13931],
        "duration_frames": -42,
        "dynamic_step_branch": "fallback",
    }
    malformed_nested_scalar = representative_by_key[
        "malformed_primitive|task-0013|episode_00130230|position=2|primitive_idx=2"
    ]
    assert consumer_output(malformed_nested_scalar, "dataset_utils.py::_duration_to_segments") == [[2578, 8271]]
    assert consumer_output(malformed_nested_scalar, "eval_segment.py::_normalize_frame_duration") == [2578, 8271]
    assert consumer_output(malformed_nested_scalar, "parse_frame_duration/get_dynamic_max_steps")["dynamic_step_branch"] == "fallback"
    malformed_scalar_nested = representative_by_key[
        "malformed_primitive|task-0004|episode_00041510|position=1|primitive_idx=1"
    ]
    assert consumer_output(malformed_scalar_nested, "dataset_utils.py::_duration_to_segments") == [[2466, 8222]]
    assert consumer_output(malformed_scalar_nested, "eval_segment.py::_normalize_frame_duration") == [2466, 8222]
    assert consumer_output(malformed_scalar_nested, "parse_frame_duration/get_dynamic_max_steps")["dynamic_step_branch"] == "fallback"

    for entry in representatives:
        payload = viewer.selection_payload(entry, args.output_dir / ".smoke_no_reviews.json")
        assert payload["boundary_choices"]
        assert viewer.entry_summary_markdown(entry)
        assert viewer.context_markdown(entry)
        assert viewer.timeline_html(entry, payload["default_frame"])
        assert all(Path(record["path"]).is_file() for record in entry["media"].values())

    source_entry = representatives[0]
    source_path = Path(source_entry["annotation_path"])
    source_before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="b1k-review-smoke-") as temp_dir:
        review_path = Path(temp_dir) / "human_reviews.json"
        proposed = {"frame_duration": [[5949, 6456], [6738, 7000]], "status": "human_proposal_only"}
        saved = viewer.upsert_review(
            review_path,
            item_key=source_entry["key"],
            item_kind=source_entry["kind"],
            source_annotation_path=str(source_path),
            source_annotation_sha256=source_entry["annotation_sha256"],
            decision="propose_correction",
            proposed_value=proposed,
            notes="smoke test; temporary review artifact",
        )
        assert saved["proposed_value"] == proposed
        assert viewer.review_for_entry(review_path, source_entry)["notes"].startswith("smoke test")
        assert json.loads(review_path.read_text())["reviews"][source_entry["key"]]["decision"] == "propose_correction"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_before == source_entry["annotation_sha256"]

    ui_report: dict[str, object] = {"skipped": args.skip_ui}
    if not args.skip_ui:
        gr, cv2, np = viewer._load_ui_dependencies()
        del gr
        reader = viewer.VideoFrameReader(cv2, np)
        real_frames, real_status = reader.read_aligned(
            viewer.media_paths(representatives[0]), representatives[0]["raw_duration"][0][0]
        )
        assert len(real_frames) == 3
        assert "decoded" in real_status
        missing_paths = {camera: Path("/definitely/missing") / f"{camera}.mp4" for camera in viewer.CAMERAS}
        fallback_frames, fallback_status = reader.read_aligned(missing_paths, 0)
        assert len(fallback_frames) == 3
        assert "Visible media fallback" in fallback_status

        app = viewer.build_app(manifest, args.dataset_root, args.output_dir / "human_reviews.json", page_size=50)
        present_callback = next(block.fn for block in app.fns.values() if block.fn.__name__ == "present_row")
        callback_rendered_keys = []
        for entry in representatives:
            outputs = present_callback(entry["key"])
            assert len(outputs) == 16
            assert outputs[5].startswith("### ")
            assert outputs[7] == entry["raw_item"]
            assert outputs[8]["warning"].startswith("DIAGNOSTIC ONLY")
            assert len(outputs[2:5]) == 3
            callback_rendered_keys.append(entry["key"])
        port = free_loopback_port()
        result = app.queue(default_concurrency_limit=2).launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            show_error=True,
            prevent_thread_lock=True,
            quiet=True,
            allowed_paths=[str(args.dataset_root), str(args.output_dir)],
        )
        local_url = result[1] if isinstance(result, tuple) else f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(local_url, timeout=20) as response:
            body = response.read()
            assert response.status == 200
            assert b"gradio" in body.lower()
        app.close()
        ui_report = {
            "skipped": False,
            "dependency_versions": {
                "gradio": __import__("gradio").__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "real_media_status": real_status,
            "fallback_status": fallback_status,
            "temporary_launch_url": local_url,
            "health_status": 200,
            "callback_rendered_keys": callback_rendered_keys,
        }

    report = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "summary": summary,
        "representative_keys": [entry["key"] for entry in representatives],
        "source_annotation_unchanged": True,
        "atomic_review_save_reload": True,
        "ui": ui_report,
    }
    report_path = args.output_dir / "smoke_report.json"
    viewer._atomic_write_json(report_path, report)
    print(json.dumps({**report, "smoke_report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
