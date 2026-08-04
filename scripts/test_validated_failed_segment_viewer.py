# ruff: noqa: FBT002, FBT003, PT009

import importlib.util
import json
from pathlib import Path
import sys
import unittest

_MODULE_PATH = Path(__file__).with_name("validated_failed_segment_viewer.py")
_SPEC = importlib.util.spec_from_file_location("validated_failed_segment_viewer", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def exact_row(**overrides):
    row = {
        "dedupe_key": "task-0001|00010001|2|30",
        "task_id": 1,
        "task_dir": "task-0001",
        "task_name": "test_task",
        "demo_id": "00010001",
        "skill": "pick up from",
        "skill_idx": 2,
        "frame_start": 20,
        "segment_end_frame": 30,
        "metric_family": "grasp_relation",
        "annotation_path": "/readonly/annotation.json",
        "parquet_path": "/readonly/episode.parquet",
        "evidence_status": "exact_sim_restored",
        "exact_evaluable": True,
        "exact_sim_restored": True,
        "raw_end_failure": True,
        "raw_end_satisfied": False,
        "failure_reason": "required_end_predicate_unsatisfied",
        "predicate_spec": [],
        "predicate_trace": [],
    }
    row.update(overrides)
    return row


def predicate(name, satisfied, desired=True):
    return {
        "predicate": f"{name}(a,b)",
        "desired": desired,
        "satisfied": satisfied,
        "diagnostics": {"predicate_name": name},
    }


class FixedSemanticsTests(unittest.TestCase):
    def test_non_place_preserves_persisted_failure(self):
        outcome = mod.derive_fixed_outcome(exact_row())
        self.assertEqual(outcome.category, "fixed_failure")
        self.assertEqual(outcome.raw_category, "raw_failure")
        self.assertIn("required_end_predicate", outcome.failure_reason)

    def test_place_conjunction_uses_spatial_primaries_and_ignores_release_auxiliary(self):
        row = exact_row(
            skill="place on next to",
            predicate_trace=[
                predicate("ontop", True),
                predicate("nextto", True),
                predicate("grasped", False, desired=False),
            ],
        )
        outcome = mod.derive_fixed_outcome(row)
        self.assertEqual(outcome.category, "fixed_success")
        self.assertEqual(outcome.reason, "primary_predicates_evaluated")

    def test_place_failure_names_unsatisfied_primary(self):
        row = exact_row(
            skill="place on next to",
            raw_end_failure=False,
            raw_end_satisfied=True,
            predicate_trace=[predicate("ontop", True), predicate("nextto", False), predicate("grasped", True, False)],
        )
        outcome = mod.derive_fixed_outcome(row)
        self.assertEqual(outcome.raw_category, "raw_success")
        self.assertEqual(outcome.category, "fixed_failure")
        self.assertEqual(outcome.failure_reason, "required_primary_predicates_unsatisfied:nextto")

    def test_not_exact_invalid_row_is_never_a_failure(self):
        row = exact_row(
            evidence_status="exact_sim_restored_predicate_invalid",
            exact_evaluable=False,
            exact_sim_restored=True,
            raw_end_failure=None,
            raw_end_satisfied=None,
        )
        outcome = mod.derive_fixed_outcome(row)
        self.assertEqual(outcome.category, "fixed_persisted_predicate_invalid")
        self.assertFalse(outcome.is_exact_failure)

    def test_place_trace_error_becomes_semantic_invalid_not_failure(self):
        row = exact_row(skill="place in", predicate_trace=[{**predicate("inside", False), "error": "bad object"}])
        outcome = mod.derive_fixed_outcome(row)
        self.assertEqual(outcome.category, "fixed_semantic_invalid")
        self.assertFalse(outcome.is_exact_failure)


class AnnotationTimelineTests(unittest.TestCase):
    def setUp(self):
        self.annotation = {
            "meta_data": {"valid_duration": [8, 45]},
            "skill_annotation": [
                {
                    "skill_idx": 0,
                    "skill_description": ["move to"],
                    "skill_type": ["navigation"],
                    "object_id": [["cup"]],
                    "frame_duration": [10, 20],
                },
                {
                    "skill_idx": 1,
                    "skill_description": ["pick up from"],
                    "skill_type": ["uncoordinated"],
                    "object_id": [["cup", "table"]],
                    "frame_duration": [21, 30],
                },
                {
                    "skill_idx": 2,
                    "skill_description": ["place on"],
                    "skill_type": ["uncoordinated"],
                    "object_id": [["cup", "tray"]],
                    "frame_duration": [35, 40],
                },
            ],
        }

    def test_timeline_has_previous_current_next_and_explicit_gaps(self):
        timeline = mod.build_annotation_timeline(self.annotation, 1)
        roles = {item["skill_idx"]: item["role"] for item in timeline if item["kind"] == "skill"}
        self.assertEqual(roles, {0: "previous", 1: "current", 2: "next"})
        gaps = [(item["start"], item["end"]) for item in timeline if item["kind"] == "gap"]
        self.assertEqual(gaps, [(8, 10), (20, 21), (30, 35), (40, 45)])
        context_gaps = [
            (item["start"], item["end"]) for item in mod.timeline_context(timeline) if item["kind"] == "gap"
        ]
        self.assertEqual(context_gaps, [(20, 21), (30, 35)])

    def test_progress_is_half_open_and_identifies_gap(self):
        timeline = mod.build_annotation_timeline(self.annotation, 1)
        gap = mod.frame_progress_metadata(20, timeline, 1)
        self.assertEqual(gap["active_kind"], "gap")
        self.assertEqual(gap["selected_done"], 0)
        started = mod.frame_progress_metadata(21, timeline, 1)
        self.assertEqual(started["active_role"], "current")
        self.assertEqual(started["selected_done"], 1)
        self.assertEqual(started["selected_total"], 9)
        finished = mod.frame_progress_metadata(30, timeline, 1)
        self.assertEqual(finished["selected_percent"], 100.0)


class PathAndIndexTests(unittest.TestCase):
    def test_video_and_annotation_path_resolution(self):
        root = Path("/dataset")
        paths = mod.resolve_video_paths(root, 26, "260010")
        self.assertEqual(
            paths["head"],
            root / "videos/task-0026/observation.images.rgb.head/episode_00260010.mp4",
        )
        self.assertEqual(
            mod.resolve_annotation_path(root, 26, 260010),
            root / "annotations/task-0026/episode_00260010.json",
        )
        persisted = "/protected/episode.json"
        self.assertEqual(mod.resolve_annotation_path(root, 26, 260010, persisted), Path(persisted))

    def test_scanner_selects_only_fixed_failures_and_round_trips_source_offset(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            jsonl = root / "lane_1/shard_01/segments.jsonl"
            jsonl.parent.mkdir(parents=True)
            failure = exact_row()
            place_raw_failure_fixed_success = exact_row(
                dedupe_key="task-0001|00010001|3|40",
                skill="place on",
                skill_idx=3,
                segment_end_frame=40,
                predicate_trace=[predicate("ontop", True), predicate("grasped", False, False)],
            )
            invalid = exact_row(
                dedupe_key="task-0001|00010001|4|50",
                skill_idx=4,
                segment_end_frame=50,
                evidence_status="not_evaluable_offline",
                exact_evaluable=False,
                exact_sim_restored=False,
                raw_end_failure=None,
                raw_end_satisfied=None,
            )
            jsonl.write_text(
                "\n".join(json.dumps(row) for row in (failure, place_raw_failure_fixed_success, invalid)) + "\n",
                encoding="utf-8",
            )
            cache = Path(tmp) / "runtime/index.json"
            index = mod.scan_failure_index(root, expected_total=3, expected_failures=1, cache_path=cache)
            self.assertEqual(index.summary["total_rows"], 3)
            self.assertEqual(index.summary["fixed_failure_count"], 1)
            self.assertTrue(cache.is_file())
            loaded = mod.load_source_row(index.entries[0])
            self.assertEqual(loaded["dedupe_key"], failure["dedupe_key"])
            cached = mod.scan_failure_index(root, expected_total=3, expected_failures=1, cache_path=cache)
            self.assertEqual(cached.entries, index.entries)

    def test_scanner_writes_no_implicit_cache(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            jsonl = root / "lane_1/shard_01/segments.jsonl"
            jsonl.parent.mkdir(parents=True)
            jsonl.write_text(json.dumps(exact_row()) + "\n", encoding="utf-8")
            index = mod.scan_failure_index(root, expected_total=1, expected_failures=1)
            self.assertEqual(len(index.entries), 1)
            self.assertEqual(list(Path(tmp).rglob("*index*")), [])


if __name__ == "__main__":
    unittest.main()
