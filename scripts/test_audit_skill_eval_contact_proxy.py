import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("audit_skill_eval_contact_proxy.py")
_SPEC = importlib.util.spec_from_file_location("audit_skill_eval_contact_proxy", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_compute_trace_stats_handles_all_of_malformed_and_window() -> None:
    metrics = {
        "rollout": {"combine_mode": "all_of"},
        "predicate_trace": [
            [{"satisfied": True}, {"satisfied": True}],
            "malformed",
            [{"satisfied": True}, {"satisfied": False}],
            [{"satisfied": True}, {"satisfied": True}],
            [{"satisfied": True}, {"satisfied": True}],
            [{"satisfied": False}],
        ],
    }

    stats = mod.compute_trace_stats(metrics, min_consecutive=2)

    assert stats["trace_len"] == 6
    assert stats["satisfied_step_count"] == 3
    assert stats["satisfied_fraction"] == 0.5
    assert stats["first_satisfied_step"] == 0
    assert stats["first_window_satisfied_step"] == 4
    assert stats["max_streak"] == 2
    assert stats["final_streak"] == 0
    assert stats["last_step_satisfied"] is False
    assert stats["window_reached"] is True


def test_compute_trace_stats_any_of_and_min_consecutive_zero() -> None:
    metrics = {
        "predicate_debug": {"combine_mode": "any_of"},
        "predicate_trace": [
            [{"satisfied": False}, {"satisfied": True}],
            [{"satisfied": False}],
        ],
    }

    stats = mod.compute_trace_stats(metrics, min_consecutive=0)

    assert stats["combine_mode"] == "any_of"
    assert stats["first_satisfied_step"] == 0
    assert stats["first_window_satisfied_step"] == 0
    assert stats["max_streak"] == 1


def test_build_segment_row_preserves_false_and_zero_values(tmp_path: Path) -> None:
    metrics_path = tmp_path / "raw" / "task" / "demo_00000000" / "skill_000" / "metrics" / "m.json"
    metrics_path.parent.mkdir(parents=True)
    metrics = {
        "task_name": "task",
        "demo_id": "00000000",
        "segment_idx": 0,
        "segment_desc": "wipe hard",
        "success": False,
        "result_type": "timeout",
        "predicate_debug": {
            "metric_family": "contact_effect_proxy",
            "start_all_satisfied": False,
            "require_unsatisfied_at_start": False,
        },
        "rollout": {"final_step": 0, "max_steps": 0, "rollout_attempted": False},
        "predicate_trace": [],
    }

    row = mod.build_segment_row(tmp_path, "A", metrics_path, metrics, min_consecutive=3)

    assert row["segment_idx"] == "0"
    assert row["success"] == "false"
    assert row["start_all_satisfied"] == "false"
    assert row["require_unsatisfied_at_start"] == "false"
    assert row["final_step"] == "0"
    assert row["max_steps"] == "0"
    assert row["rollout_attempted"] == "false"


def test_parse_metric_families_defaults_and_accepts_repeated_csv_values() -> None:
    assert mod.parse_metric_families(None) == {"contact_effect_proxy"}
    assert mod.parse_metric_families(["contact_effect_proxy,transfer_pose_proxy", "relation_transfer_proxy"]) == {
        "contact_effect_proxy",
        "transfer_pose_proxy",
        "relation_transfer_proxy",
    }


def test_collect_segment_rows_accepts_multiple_metric_families(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    def write_metrics(task: str, demo: str, skill_idx: int, skill: str, family: str) -> None:
        metrics_path = run_dir / "raw" / task / f"demo_{demo}" / f"skill_{skill_idx:03d}" / "metrics" / "m.json"
        metrics_path.parent.mkdir(parents=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "task_name": task,
                    "demo_id": demo,
                    "segment_idx": skill_idx,
                    "segment_desc": skill,
                    "success": True,
                    "result_type": "predicate_satisfied",
                    "predicate_debug": {"metric_family": family},
                    "rollout": {"final_step": 10, "max_steps": 20, "rollout_attempted": True},
                    "predicate_trace": [],
                }
            )
        )

    write_metrics("task_a", "00000001", 1, "wipe hard", "contact_effect_proxy")
    write_metrics("task_b", "00000002", 2, "hand over", "transfer_pose_proxy")
    write_metrics("task_c", "00000003", 3, "close lid", "articulation_close")

    rows = mod.collect_segment_rows(
        run_dir,
        "run",
        {"contact_effect_proxy", "transfer_pose_proxy"},
        set(),
        min_consecutive=3,
    )

    assert [row["metric_family"] for row in rows] == ["transfer_pose_proxy", "contact_effect_proxy"]


def test_build_ab_flip_rows_classifies_and_computes_deltas() -> None:
    a_rows = [
        {
            "job_key": "skill|task|demo|001",
            "task_name": "task",
            "demo_id": "demo",
            "segment_idx": "1",
            "skill": "skill",
            "metric_family": "contact_effect_proxy",
            "success": "true",
            "result_type": "predicate_satisfied",
            "first_satisfied_step": "2",
            "first_window_satisfied_step": "4",
            "max_streak": "5",
            "final_streak": "3",
            "trace_len": "10",
            "final_step": "9",
            "metrics_path": "a.json",
            "video_path": "a.mp4",
        },
        {"job_key": "skill|task|demo|002", "segment_idx": "2", "skill": "skill", "task_name": "task", "demo_id": "demo", "success": "false", "result_type": "timeout", "max_streak": "1"},
    ]
    b_rows = [
        {
            "job_key": "skill|task|demo|001",
            "segment_idx": "1",
            "skill": "skill",
            "task_name": "task",
            "demo_id": "demo",
            "metric_family": "contact_effect_proxy",
            "success": "false",
            "result_type": "timeout",
            "first_satisfied_step": "3",
            "first_window_satisfied_step": "null",
            "max_streak": "2",
            "final_streak": "0",
        },
        {"job_key": "skill|task|demo|003", "segment_idx": "3", "skill": "skill", "task_name": "task", "demo_id": "demo", "success": "true", "result_type": "predicate_satisfied", "max_streak": "4"},
    ]

    rows = mod.build_ab_flip_rows(a_rows, b_rows, "jax", "torch")

    assert [row["success_flip"] for row in rows] == ["success_to_fail", "missing_a", "missing_b"]
    first = rows[0]
    assert first["result_type_changed"] == "true"
    assert first["delta_first_satisfied_b_minus_a"] == "+1"
    assert first["delta_first_window_b_minus_a"] == ""
    assert first["delta_max_streak_b_minus_a"] == "-3"
    assert first["delta_final_streak_b_minus_a"] == "-3"
