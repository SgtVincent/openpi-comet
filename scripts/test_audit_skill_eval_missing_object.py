import csv
import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("audit_skill_eval_missing_object.py")
_SPEC = importlib.util.spec_from_file_location("audit_skill_eval_missing_object", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_classify_issue_marks_missing_object_success_invalid() -> None:
    assert mod.classify_issue({"result_type": "predicate_satisfied", "success": True}, has_missing=True) == (
        "invalid_success_missing_object"
    )


def test_classify_issue_marks_missing_object_pre_satisfied_invalid() -> None:
    assert mod.classify_issue({"result_type": "pre_satisfied_start", "success": False}, has_missing=True) == (
        "invalid_pre_satisfied_start_missing_object"
    )


def test_classify_issue_marks_missing_object_timeout_invalid() -> None:
    assert mod.classify_issue({"result_type": "timeout", "success": False}, has_missing=True) == (
        "invalid_timeout_missing_object"
    )


def test_classify_issue_ignores_rows_without_missing_object() -> None:
    assert mod.classify_issue({"result_type": "timeout", "success": False}, has_missing=False) == ""


def test_classify_issue_covers_env_other_and_start_all_satisfied() -> None:
    assert mod.classify_issue({"result_type": "env_terminated", "success": False}, has_missing=True) == (
        "invalid_env_terminated_missing_object"
    )
    assert mod.classify_issue({"result_type": "unknown", "success": False}, has_missing=True) == (
        "invalid_other_missing_object"
    )
    assert mod.classify_issue(
        {"result_type": "timeout", "success": False, "predicate_debug": {"start_all_satisfied": True}},
        has_missing=True,
    ) == "invalid_pre_satisfied_start_missing_object"


def test_collect_trace_items_and_extract_missing_object() -> None:
    metrics = {
        "predicate_debug": {
            "template_trace_start": [{"predicate": "p_start", "diagnostics": {"missing_object": "obj_a"}}],
            "template_trace_end": [{"predicate": "p_end", "diagnostics": {"missing_object": ["obj_b", "obj_c"]}}],
        },
        "predicate_trace": [
            "malformed",
            [{"predicate": "p_rollout", "diagnostics": {"missing_object": None}}, {"predicate": "p_rollout_2", "diagnostics": {"missing_object": []}}],
        ],
    }

    items = mod.collect_trace_items(metrics)

    assert [item["trace_stage"] for item in items] == ["template_start", "template_end", "rollout", "rollout"]
    assert [item["rollout_step"] for item in items] == [None, None, 1, 1]
    assert mod.extract_missing_object(items[0]["item"]) == "obj_a"
    assert mod.extract_missing_object(items[1]["item"]) == "obj_b,obj_c"
    assert mod.extract_missing_object(items[2]["item"]) == ""
    assert mod.extract_missing_object(items[3]["item"]) == ""


def test_build_flag_rows_groups_duplicates_and_preserves_false_values(tmp_path: Path) -> None:
    metrics_path = tmp_path / "raw" / "task" / "demo_00000000" / "skill_000" / "metrics" / "m.json"
    metrics_path.parent.mkdir(parents=True)
    metrics = {
        "task_name": "task",
        "demo_id": "00000000",
        "segment_idx": 0,
        "segment_desc": "wipe hard",
        "result_type": "timeout",
        "success": False,
        "predicate_debug": {
            "metric_family": "contact_effect_proxy",
            "start_all_satisfied": False,
            "template_trace_start": [
                {
                    "predicate": "touching(agent,obj)",
                    "desired": False,
                    "value": False,
                    "satisfied": False,
                    "diagnostics": {"missing_object": "obj"},
                }
            ],
        },
        "rollout": {"rollout_attempted": False},
        "predicate_trace": [
            [
                {
                    "predicate": "touching(agent,obj)",
                    "desired": False,
                    "value": False,
                    "satisfied": False,
                    "diagnostics": {"missing_object": "obj"},
                }
            ],
            [
                {
                    "predicate": "touching(agent,obj)",
                    "desired": False,
                    "value": False,
                    "satisfied": False,
                    "diagnostics": {"missing_object": "obj"},
                }
            ],
        ],
    }

    rows = mod.build_flag_rows(tmp_path, metrics_path, metrics, csv_row={"job_key": "csv-key", "csv_line": "2"})

    assert len(rows) == 2
    by_stage = {row["trace_stage"]: row for row in rows}
    assert by_stage["template_start"]["occurrence_count"] == "1"
    assert by_stage["template_start"]["desired"] == "false"
    assert by_stage["template_start"]["value"] == "false"
    assert by_stage["template_start"]["satisfied"] == "false"
    assert by_stage["template_start"]["segment_idx"] == "0"
    assert by_stage["template_start"]["start_all_satisfied"] == "false"
    assert by_stage["rollout"]["occurrence_count"] == "2"
    assert by_stage["rollout"]["first_rollout_step"] == "0"
    assert all(row["job_key"] == "csv-key" and row["csv_line"] == "2" for row in rows)


def test_collect_rows_joins_csv_with_unpadded_segment_idx(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "raw" / "task" / "demo_00000010" / "skill_005" / "metrics"
    metrics_dir.mkdir(parents=True)
    metrics = {
        "task_name": "task",
        "demo_id": "00000010",
        "segment_idx": 5,
        "segment_desc": "wipe hard",
        "result_type": "predicate_satisfied",
        "success": True,
        "predicate_debug": {"template_trace_start": [{"predicate": "p", "diagnostics": {"missing_object": "obj"}}]},
    }
    (metrics_dir / "segment_eval_task_00000010_skill005.json").write_text(json.dumps(metrics))
    with (tmp_path / "multinode_skill_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["skill", "task_name", "demo_id", "segment_idx"])
        writer.writeheader()
        writer.writerow({"skill": "wipe hard", "task_name": "task", "demo_id": "00000010", "segment_idx": "5"})

    rows = mod.collect_rows([tmp_path], skill_filter=set())

    assert rows[0]["job_key"] == "wipe hard|task|00000010|005"
    assert rows[0]["csv_line"] == "2"
