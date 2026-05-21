import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("audit_skill_eval_release_termination.py")
_SPEC = importlib.util.spec_from_file_location("audit_skill_eval_release_termination", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _write_metrics(tmp_path: Path, metrics: dict) -> Path:
    metrics_dir = tmp_path / "raw" / "task" / "demo_00000010" / "skill_005" / "metrics"
    metrics_dir.mkdir(parents=True)
    metrics_path = metrics_dir / "segment_eval_task_00000010_skill005.json"
    metrics_path.write_text(json.dumps(metrics))
    return metrics_path


def _base_metrics() -> dict:
    return {
        "task_name": "attach_a_camera_to_a_tripod",
        "demo_id": "00000010",
        "segment_idx": 5,
        "segment_desc": "release",
        "result_type": "env_terminated",
        "success": False,
        "predicate_debug": {"metric_family": "grasp_release"},
        "rollout": {
            "final_step": 1,
            "max_steps": 10,
            "rollout_attempted": True,
            "termination_reason": "env_terminated",
            "env_done_success": True,
            "env_terminal_debug": {
                "done_info": {
                    "success": False,
                    "termination_conditions": {"predicate": {"done": True, "success": False}},
                }
            },
            "combine_mode": "all_of",
        },
        "predicate_trace": [
            [
                {
                    "predicate": "grasped(agent,obj_a)",
                    "metric_type": "predicate",
                    "desired": False,
                    "value": True,
                    "satisfied": False,
                    "diagnostics": {"arm_grasp_states": {"left": True}},
                },
                {
                    "predicate": "grasped(agent,obj_b)",
                    "metric_type": "predicate",
                    "desired": False,
                    "value": False,
                    "satisfied": True,
                    "diagnostics": {},
                },
            ]
        ],
    }


def test_collect_rows_flattens_env_and_release_mismatch_fields(tmp_path: Path) -> None:
    _write_metrics(tmp_path, _base_metrics())

    rows = mod.collect_rows([tmp_path], {"release"}, {"env_terminated"}, 1)

    assert len(rows) == 1
    row = rows[0]
    assert row["env_done_success"] == "true"
    assert row["env_terminated_seen"] == ""
    assert row["env_terminal_debug_done_info_success"] == "false"
    assert row["predicate_done"] == "true"
    assert row["predicate_success"] == "false"
    assert row["goal_status"] == "unsatisfied"
    assert row["mode_bucket"] == "attach_task_env_success_release_predicate_unsatisfied"
    assert len(json.loads(row["release_predicate_summary_json"])) == 2


def test_select_final_trace_items_falls_back_to_template_trace_end() -> None:
    metrics = _base_metrics()
    metrics["predicate_trace"] = []
    metrics["predicate_debug"]["template_trace_end"] = [
        {"predicate": "grasped(agent,obj)", "desired": False, "value": True, "satisfied": False}
    ]

    trace_items = mod.select_final_trace_items(metrics)

    assert trace_items[0]["predicate"] == "grasped(agent,obj)"
    assert mod.aggregate_goal_status(trace_items, "all_of") == "unsatisfied"


def test_goal_status_uses_all_predicates_for_all_of() -> None:
    trace_items = [
        {"predicate": "p1", "satisfied": True},
        {"predicate": "p2", "satisfied": False},
    ]

    assert mod.aggregate_goal_status(trace_items, "all_of") == "unsatisfied"
    assert mod.classify_mode(_base_metrics(), "unsatisfied") == "attach_task_env_success_release_predicate_unsatisfied"


def test_missing_env_terminal_done_success_is_empty(tmp_path: Path) -> None:
    metrics = _base_metrics()
    metrics["rollout"]["env_terminal_debug"] = {}
    _write_metrics(tmp_path, metrics)

    rows = mod.collect_rows([tmp_path], {"release"}, {"env_terminated"}, 1)

    assert rows[0]["env_terminal_debug_done_info_success"] == ""


def test_render_markdown_includes_mismatch_columns(tmp_path: Path) -> None:
    _write_metrics(tmp_path, _base_metrics())
    rows = mod.collect_rows([tmp_path], {"release"}, {"env_terminated"}, 1)
    markdown = mod.render_markdown([tmp_path], mod.summarize(rows), rows)

    assert "env_done_success" in markdown
    assert "env_terminated_seen" in markdown
    assert "video_too_short" in markdown
    assert "done_info_success" in markdown
    assert "predicate_done" in markdown
    assert "goal_status" in markdown
