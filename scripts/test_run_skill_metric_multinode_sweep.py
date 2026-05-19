import importlib.util
import json
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("run_skill_metric_multinode_sweep.py")
_SPEC = importlib.util.spec_from_file_location("run_skill_metric_multinode_sweep", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

_COMPARE_MODULE_PATH = Path(__file__).with_name("compare_multinode_skill_summary.py")
_COMPARE_SPEC = importlib.util.spec_from_file_location("compare_multinode_skill_summary", _COMPARE_MODULE_PATH)
assert _COMPARE_SPEC is not None and _COMPARE_SPEC.loader is not None
compare_mod = importlib.util.module_from_spec(_COMPARE_SPEC)
_COMPARE_SPEC.loader.exec_module(compare_mod)


def test_classify_result_row_excludes_metric_invalid_from_attemptable() -> None:
    row = {"runtime_ok": True, "success": False, "result_type": "metric_invalid_missing_object"}

    classes = mod.classify_result_row(row)

    assert classes["runtime_pass"] is True
    assert classes["metric_invalid"] is True
    assert classes["metric_invalid_missing_object"] is True
    assert classes["attemptable"] is False
    assert classes["metric_unsatisfied_attemptable"] is False


def test_summarize_result_rows_counts_env_telemetry() -> None:
    rows = [
        {
            "runtime_ok": True,
            "success": True,
            "result_type": "predicate_satisfied",
            "env_done_success": False,
            "rollout_terminated": False,
            "rollout_truncated": False,
            "termination_reason": "predicate_satisfied",
            "env_termination_reason": "running",
        },
        {
            "runtime_ok": True,
            "success": False,
            "result_type": "pre_satisfied_start",
            "termination_reason": "pre_satisfied_start",
        },
        {
            "runtime_ok": True,
            "success": False,
            "result_type": "metric_invalid_missing_object",
            "termination_reason": "metric_invalid_missing_object",
        },
        {
            "runtime_ok": True,
            "success": False,
            "result_type": "env_terminated",
            "env_done_success": True,
            "rollout_terminated": True,
            "rollout_truncated": False,
            "termination_reason": "env_terminated",
            "env_termination_reason": "terminated",
        },
    ]

    summary = mod.summarize_result_rows(rows)

    assert summary["segment_count"] == 4
    assert summary["pre_satisfied_start_count"] == 1
    assert summary["metric_invalid_missing_object_count"] == 1
    assert summary["attemptable_segment_count"] == 2
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["env_done_success_count"] == 1
    assert summary["rollout_terminated_count"] == 1
    assert summary["termination_reason_counts"]["metric_invalid_missing_object"] == 1
    assert summary["env_termination_reason_counts"]["terminated"] == 1


def test_classify_short_proxy_and_likely_false_positive_are_separate_from_clean_success() -> None:
    short_row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "short_proxy_success",
        "short_proxy_success": True,
        "rollout_attempted": True,
    }
    likely_row = {
        "runtime_ok": True,
        "success": False,
        "result_type": "likely_proxy_false_positive",
        "short_proxy_success": True,
        "likely_proxy_false_positive": True,
        "rollout_attempted": True,
    }

    short_classes = mod.classify_result_row(short_row)
    likely_classes = mod.classify_result_row(likely_row)
    summary = mod.summarize_result_rows([short_row, likely_row])

    assert short_classes["attemptable"] is True
    assert short_classes["policy_success_attemptable"] is False
    assert short_classes["short_proxy_success"] is True
    assert likely_classes["attemptable"] is True
    assert likely_classes["likely_proxy_false_positive"] is True
    assert likely_classes["other_metric_unsatisfied"] is False
    assert summary["short_proxy_success_count"] == 2
    assert summary["likely_proxy_false_positive_count"] == 1
    assert summary["policy_success_attemptable_count"] == 0


def test_short_predicate_success_is_excluded_from_clean_success() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "rollout_attempted": True,
        "final_step": 22,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["short_video_problem"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["short_video_problem_count"] == 1
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0


def test_metrics_short_video_problem_result_is_attemptable_not_success() -> None:
    row = {
        "runtime_ok": True,
        "success": False,
        "result_type": "short_video_problem",
        "metrics_short_video_problem": True,
        "rollout_attempted": True,
        "final_step": 150,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is False
    assert classes["short_video_problem"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["short_video_problem_count"] == 1
    assert summary["policy_success_attemptable_count"] == 0


def test_transfer_pose_proxy_success_requires_review_not_clean_success() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "transfer_pose_proxy",
        "rollout_attempted": True,
        "final_step": 188,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["transfer_pose_proxy_success_unconfirmed"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["transfer_pose_proxy_success_unconfirmed_count"] == 1


def test_contact_effect_proxy_success_requires_review_not_clean_success() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "contact_effect_proxy",
        "rollout_attempted": True,
        "final_step": 188,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["contact_effect_proxy_success_unconfirmed"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["contact_effect_proxy_success_unconfirmed_count"] == 1


def test_articulation_close_success_requires_review_not_clean_success() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "articulation_close",
        "rollout_attempted": True,
        "final_step": 188,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["articulation_close_success_unconfirmed"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["articulation_close_success_unconfirmed_count"] == 1


def test_geometry_base_facing_success_requires_review_not_clean_success() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "geometry_base_facing",
        "rollout_attempted": True,
        "final_step": 188,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["geometry_base_facing_success_unconfirmed"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["geometry_base_facing_success_unconfirmed_count"] == 1


def test_relation_transfer_proxy_success_requires_review_not_clean_success() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "relation_transfer_proxy",
        "rollout_attempted": True,
        "final_step": 188,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["relation_transfer_proxy_success_unconfirmed"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["relation_transfer_proxy_success_unconfirmed_count"] == 1


def test_orientation_proxy_success_requires_review_not_clean_success() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "orientation_proxy",
        "rollout_attempted": True,
        "final_step": 188,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["orientation_proxy_success_unconfirmed"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["orientation_proxy_success_unconfirmed_count"] == 1


def test_articulation_open_proxy_success_requires_review_not_clean_success() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "articulation_open_proxy",
        "rollout_attempted": True,
        "final_step": 188,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["articulation_open_proxy_success_unconfirmed"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["articulation_open_proxy_success_unconfirmed_count"] == 1


def test_articulation_open_door_drawer_success_requires_review_not_clean_success() -> None:
    open_door = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "task_name": "cleaning_up_plates_and_food",
        "skill": "open door",
        "metric_family": "articulation_open",
        "rollout_attempted": True,
        "final_step": 188,
    }
    open_drawer = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "task_name": "putting_away_Halloween_decorations",
        "skill": "open drawer",
        "metric_family": "articulation_open",
        "rollout_attempted": True,
        "final_step": 188,
    }

    door_classes = mod.classify_result_row(open_door)
    drawer_classes = mod.classify_result_row(open_drawer)
    summary = mod.summarize_result_rows([open_door, open_drawer])

    assert door_classes["policy_success_attemptable"] is True
    assert door_classes["articulation_open_success_unconfirmed"] is True
    assert door_classes["policy_success_clean_attemptable"] is False
    assert drawer_classes["policy_success_attemptable"] is True
    assert drawer_classes["articulation_open_success_unconfirmed"] is True
    assert drawer_classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 2
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["articulation_open_success_unconfirmed_count"] == 2


def test_can_meat_lid_success_requires_review_separately_from_open_door_drawer_guard() -> None:
    can_meat_open_lid = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "task_name": "can_meat",
        "skill": "open lid",
        "metric_family": "articulation_open",
        "rollout_attempted": True,
        "final_step": 188,
    }
    open_drawer = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "task_name": "putting_away_Halloween_decorations",
        "skill": "open drawer",
        "metric_family": "articulation_open",
        "rollout_attempted": True,
        "final_step": 188,
    }

    can_classes = mod.classify_result_row(can_meat_open_lid)
    drawer_classes = mod.classify_result_row(open_drawer)
    summary = mod.summarize_result_rows([can_meat_open_lid, open_drawer])

    assert can_classes["policy_success_attemptable"] is True
    assert can_classes["can_meat_lid_success_unconfirmed"] is True
    assert can_classes["articulation_open_success_unconfirmed"] is False
    assert can_classes["policy_success_clean_attemptable"] is False
    assert drawer_classes["policy_success_attemptable"] is True
    assert drawer_classes["can_meat_lid_success_unconfirmed"] is False
    assert drawer_classes["articulation_open_success_unconfirmed"] is True
    assert drawer_classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 2
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["can_meat_lid_success_unconfirmed_count"] == 1
    assert summary["articulation_open_success_unconfirmed_count"] == 1


def test_compare_summary_uses_same_new_proxy_guards() -> None:
    can_meat_row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "task_name": "can_meat",
        "skill": "open lid",
        "metric_family": "articulation_open",
        "rollout_attempted": True,
        "final_step": 188,
    }
    orientation_row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "orientation_proxy",
        "rollout_attempted": True,
        "final_step": 188,
    }
    articulation_open_proxy_row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "metric_family": "articulation_open_proxy",
        "rollout_attempted": True,
        "final_step": 188,
    }
    open_drawer_row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "task_name": "putting_away_Halloween_decorations",
        "skill": "open drawer",
        "metric_family": "articulation_open",
        "rollout_attempted": True,
        "final_step": 188,
    }

    can_classes = compare_mod.classify_result_row(can_meat_row)
    orientation_classes = compare_mod.classify_result_row(orientation_row)
    articulation_open_proxy_classes = compare_mod.classify_result_row(articulation_open_proxy_row)
    open_drawer_classes = compare_mod.classify_result_row(open_drawer_row)
    summary = compare_mod.summarize_result_rows(
        [can_meat_row, orientation_row, articulation_open_proxy_row, open_drawer_row]
    )

    assert can_classes["policy_success_attemptable"] is True
    assert can_classes["can_meat_lid_success_unconfirmed"] is True
    assert can_classes["policy_success_clean_attemptable"] is False
    assert orientation_classes["orientation_proxy_success_unconfirmed"] is True
    assert orientation_classes["policy_success_clean_attemptable"] is False
    assert articulation_open_proxy_classes["articulation_open_proxy_success_unconfirmed"] is True
    assert articulation_open_proxy_classes["policy_success_clean_attemptable"] is False
    assert open_drawer_classes["articulation_open_success_unconfirmed"] is True
    assert open_drawer_classes["policy_success_clean_attemptable"] is False
    assert summary["policy_success_attemptable_count"] == 4
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["can_meat_lid_success_unconfirmed_count"] == 1
    assert summary["orientation_proxy_success_unconfirmed_count"] == 1
    assert summary["articulation_open_proxy_success_unconfirmed_count"] == 1
    assert summary["articulation_open_success_unconfirmed_count"] == 1


def test_early_metric_activation_review_needed_is_not_clean_or_meaningful() -> None:
    row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "rollout_attempted": True,
        "start_all_satisfied": False,
        "min_success_steps": 150,
        "first_predicate_satisfied_step": 12,
        "early_predicate_satisfied_steps": 3,
        "final_step": 220,
    }

    classes = mod.classify_result_row(row)
    summary = mod.summarize_result_rows([row])

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is True
    assert classes["early_metric_activation_review_needed"] is True
    assert classes["policy_success_clean_attemptable"] is False
    assert classes["meaningful_policy_caused_transition"] is False
    assert summary["success_count_raw"] == 1
    assert summary["policy_success_attemptable_count"] == 1
    assert summary["policy_success_clean_attemptable_count"] == 0
    assert summary["early_metric_activation_review_needed_count"] == 1
    assert summary["meaningful_policy_caused_transition_count"] == 0


def test_meaningful_policy_caused_transition_counts_separately_from_attemptable() -> None:
    early_row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "rollout_attempted": True,
        "start_all_satisfied": False,
        "min_success_steps": 150,
        "first_predicate_satisfied_step": 20,
        "early_predicate_satisfied_steps": 1,
        "final_step": 250,
    }
    meaningful_row = {
        "runtime_ok": True,
        "success": True,
        "result_type": "predicate_satisfied",
        "rollout_attempted": True,
        "start_all_satisfied": False,
        "min_success_steps": 150,
        "first_predicate_satisfied_step": 151,
        "early_predicate_satisfied_steps": 0,
        "final_step": 250,
    }

    summary = mod.summarize_result_rows([early_row, meaningful_row])

    assert summary["attemptable_segment_count"] == 2
    assert summary["policy_success_attemptable_count"] == 2
    assert summary["success_count_raw"] == 2
    assert summary["policy_success_clean_attemptable_count"] == 1
    assert summary["early_metric_activation_review_needed_count"] == 1
    assert summary["meaningful_policy_caused_transition_count"] == 1


def test_load_metrics_row_flattens_env_terminal_debug(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    env_terminal_debug = {"termination_reason": "terminated", "done_info": {"success": True}}
    metrics_path.write_text(
        json.dumps(
            {
                "success": False,
                "result_type": "env_terminated",
                "predicate_debug": {"metric_family": "grasp_release", "start_all_satisfied": False},
                "rollout": {
                    "final_step": 1,
                    "rollout_attempted": True,
                    "termination_reason": "env_terminated",
                    "env_termination_reason": "terminated",
                    "env_done_success": True,
                    "terminated": True,
                    "truncated": False,
                    "env_terminal_debug": env_terminal_debug,
                    "max_steps": 10,
                },
            }
        )
    )

    row = mod.load_metrics_row(
        metrics_path,
        sample={
            "job_key": "release|task|00000001|005",
            "skill": "release",
            "task_name": "task",
            "demo_id": "00000001",
            "skill_idx": 5,
            "frame_duration": [0, 5],
        },
        runtime_ok=True,
        returncode=0,
        segment_log=tmp_path / "segment_eval.log",
    )

    assert row["termination_reason"] == "env_terminated"
    assert row["env_termination_reason"] == "terminated"
    assert row["env_done_success"] is True
    assert row["rollout_terminated"] is True
    assert row["rollout_truncated"] is False
    assert row["env_terminal_debug"] == env_terminal_debug
    assert json.loads(row["env_terminal_debug_json"]) == env_terminal_debug


def test_load_metrics_row_flattens_short_proxy_diagnostics(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "success": False,
                "result_type": "likely_proxy_false_positive",
                "predicate_debug": {
                    "metric_family": "transfer_pose_proxy",
                    "short_proxy_success": True,
                    "likely_proxy_false_positive": True,
                    "short_success_required_step": 64,
                },
                "rollout": {
                    "final_step": 6,
                    "rollout_attempted": True,
                    "termination_reason": "likely_proxy_false_positive",
                    "max_steps": 636,
                },
            }
        )
    )

    row = mod.load_metrics_row(
        metrics_path,
        sample={
            "job_key": "hand over|can_meat|00040020|004",
            "skill": "hand over",
            "task_name": "can_meat",
            "demo_id": "00040020",
            "skill_idx": 4,
            "frame_duration": [0, 318],
        },
        runtime_ok=True,
        returncode=0,
        segment_log=tmp_path / "segment_eval.log",
    )

    assert row["metric_family"] == "transfer_pose_proxy"
    assert row["short_proxy_success"] is True
    assert row["likely_proxy_false_positive"] is True
    assert row["short_success_required_step"] == 64


def test_load_metrics_row_flattens_activation_diagnostics(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "success": True,
                "result_type": "predicate_satisfied",
                "predicate_debug": {
                    "metric_family": "grasp_release",
                    "start_all_satisfied": False,
                    "min_success_steps": 150,
                    "first_predicate_satisfied_step": 12,
                    "early_predicate_satisfied_steps": 2,
                },
                "rollout": {
                    "final_step": 180,
                    "rollout_attempted": True,
                    "termination_reason": "predicate_satisfied",
                    "max_steps": 636,
                },
            }
        )
    )

    row = mod.load_metrics_row(
        metrics_path,
        sample={
            "job_key": "release|task|00000001|005",
            "skill": "release",
            "task_name": "task",
            "demo_id": "00000001",
            "skill_idx": 5,
            "frame_duration": [0, 5],
        },
        runtime_ok=True,
        returncode=0,
        segment_log=tmp_path / "segment_eval.log",
    )

    assert row["min_success_steps"] == 150
    assert row["first_predicate_satisfied_step"] == 12
    assert row["early_predicate_satisfied_steps"] == 2
    assert row["early_metric_activation_review_needed"] is True
    assert row["meaningful_policy_caused_transition"] is False


def test_env_task_success_before_segment_success_is_attemptable_not_success() -> None:
    row = {
        "runtime_ok": True,
        "success": False,
        "result_type": "env_task_success_before_segment_success",
        "rollout_attempted": True,
        "env_done_success": True,
        "rollout_terminated": True,
    }

    classes = mod.classify_result_row(row)

    assert classes["attemptable"] is True
    assert classes["policy_success_attemptable"] is False
    assert classes["env_task_success_before_segment_success"] is True
    assert classes["other_metric_unsatisfied"] is False

    summary = mod.summarize_result_rows([row])
    assert summary["attemptable_segment_count"] == 1
    assert summary["policy_success_attemptable_count"] == 0
    assert summary["env_task_success_before_segment_success_count"] == 1
    assert summary["env_done_success_count"] == 1


def test_load_metrics_row_flattens_continued_env_success_telemetry(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "success": False,
                "result_type": "env_task_success_before_segment_success",
                "rollout": {
                    "final_step": 10,
                    "rollout_attempted": True,
                    "termination_reason": "env_task_success_before_segment_success",
                    "env_done_success": True,
                    "terminated": True,
                    "last_terminated": False,
                    "env_terminated_seen": True,
                    "env_done_success_seen": True,
                    "first_env_terminated_step": 1,
                    "first_env_done_success_step": 1,
                    "env_task_success_before_segment_success": True,
                    "truncated": False,
                },
            }
        )
    )

    row = mod.load_metrics_row(
        metrics_path,
        sample={
            "job_key": "release|task|00000001|005",
            "skill": "release",
            "task_name": "task",
            "demo_id": "00000001",
            "skill_idx": 5,
            "frame_duration": [0, 5],
        },
        runtime_ok=True,
        returncode=0,
        segment_log=tmp_path / "segment_eval.log",
    )

    assert row["result_type"] == "env_task_success_before_segment_success"
    assert row["rollout_terminated"] is True
    assert row["rollout_last_terminated"] is False
    assert row["env_terminated_seen"] is True
    assert row["env_done_success_seen"] is True
    assert row["first_env_terminated_step"] == 1
    assert row["first_env_done_success_step"] == 1
    assert row["env_task_success_before_segment_success"] is True


def test_compare_partial_summary_matches_new_review_and_transition_schema(tmp_path) -> None:
    run_dir = tmp_path / "run"
    worker_results = run_dir / "worker_results"
    worker_results.mkdir(parents=True)

    manifest = {
        "jobs": [
            {
                "job_key": "skill_a|task_alpha|demo001|001",
                "skill": "skill_a",
                "task_name": "task_alpha",
                "demo_id": "demo001",
            },
            {
                "job_key": "skill_a|task_alpha|demo002|002",
                "skill": "skill_a",
                "task_name": "task_alpha",
                "demo_id": "demo002",
            },
            {
                "job_key": "skill_a|task_alpha|demo003|003",
                "skill": "skill_a",
                "task_name": "task_alpha",
                "demo_id": "demo003",
            },
            {
                "job_key": "skill_b|task_beta|demo004|004",
                "skill": "skill_b",
                "task_name": "task_beta",
                "demo_id": "demo004",
            },
            {
                "job_key": "skill_c|task_gamma|demo005|005",
                "skill": "skill_c",
                "task_name": "task_gamma",
                "demo_id": "demo005",
            },
            {
                "job_key": "open lid|can_meat|demo006|006",
                "skill": "open lid",
                "task_name": "can_meat",
                "demo_id": "demo006",
            },
        ]
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (worker_results / "worker_000.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "job_key": "skill_a|task_alpha|demo001|001",
                        "skill": "skill_a",
                        "task_name": "task_alpha",
                        "demo_id": "demo001",
                        "runtime_ok": True,
                        "success": True,
                        "result_type": "predicate_satisfied",
                        "rollout_attempted": True,
                        "start_all_satisfied": False,
                        "min_success_steps": 150,
                        "first_predicate_satisfied_step": 10,
                        "early_predicate_satisfied_steps": 2,
                        "termination_reason": "predicate_satisfied",
                    }
                ),
                json.dumps(
                    {
                        "job_key": "skill_a|task_alpha|demo002|002",
                        "skill": "skill_a",
                        "task_name": "task_alpha",
                        "demo_id": "demo002",
                        "runtime_ok": True,
                        "success": True,
                        "result_type": "predicate_satisfied",
                        "rollout_attempted": True,
                        "start_all_satisfied": False,
                        "min_success_steps": 150,
                        "first_predicate_satisfied_step": 170,
                        "early_predicate_satisfied_steps": 0,
                        "termination_reason": "predicate_satisfied",
                    }
                ),
                json.dumps(
                    {
                        "job_key": "skill_a|task_alpha|demo003|003",
                        "skill": "skill_a",
                        "task_name": "task_alpha",
                        "demo_id": "demo003",
                        "runtime_ok": True,
                        "success": True,
                        "result_type": "predicate_satisfied",
                        "metric_family": "relation_transfer_proxy",
                        "rollout_attempted": True,
                        "start_all_satisfied": False,
                        "min_success_steps": 150,
                        "first_predicate_satisfied_step": 170,
                        "early_predicate_satisfied_steps": 0,
                        "termination_reason": "predicate_satisfied",
                    }
                ),
                json.dumps(
                    {
                        "job_key": "skill_b|task_beta|demo004|004",
                        "skill": "skill_b",
                        "task_name": "task_beta",
                        "demo_id": "demo004",
                        "runtime_ok": True,
                        "success": True,
                        "result_type": "predicate_satisfied",
                        "metric_family": "orientation_proxy",
                        "rollout_attempted": True,
                        "start_all_satisfied": False,
                        "min_success_steps": 150,
                        "first_predicate_satisfied_step": 170,
                        "early_predicate_satisfied_steps": 0,
                        "termination_reason": "predicate_satisfied",
                    }
                ),
                json.dumps(
                    {
                        "job_key": "skill_c|task_gamma|demo005|005",
                        "skill": "skill_c",
                        "task_name": "task_gamma",
                        "demo_id": "demo005",
                        "runtime_ok": True,
                        "success": True,
                        "result_type": "predicate_satisfied",
                        "metric_family": "articulation_open_proxy",
                        "rollout_attempted": True,
                        "start_all_satisfied": False,
                        "min_success_steps": 150,
                        "first_predicate_satisfied_step": 170,
                        "early_predicate_satisfied_steps": 0,
                        "termination_reason": "predicate_satisfied",
                    }
                ),
                json.dumps(
                    {
                        "job_key": "open lid|can_meat|demo006|006",
                        "skill": "open lid",
                        "task_name": "can_meat",
                        "demo_id": "demo006",
                        "runtime_ok": True,
                        "success": True,
                        "result_type": "predicate_satisfied",
                        "metric_family": "articulation_open",
                        "rollout_attempted": True,
                        "start_all_satisfied": False,
                        "min_success_steps": 150,
                        "first_predicate_satisfied_step": 170,
                        "early_predicate_satisfied_steps": 0,
                        "termination_reason": "predicate_satisfied",
                    }
                ),
                json.dumps(
                    {
                        "job_key": "open drawer|task_delta|demo007|007",
                        "skill": "open drawer",
                        "task_name": "task_delta",
                        "demo_id": "demo007",
                        "runtime_ok": True,
                        "success": True,
                        "result_type": "predicate_satisfied",
                        "metric_family": "articulation_open",
                        "rollout_attempted": True,
                        "start_all_satisfied": False,
                        "min_success_steps": 150,
                        "first_predicate_satisfied_step": 170,
                        "early_predicate_satisfied_steps": 0,
                        "termination_reason": "predicate_satisfied",
                    }
                ),
            ]
        )
        + "\n"
    )

    summary = compare_mod.build_partial_summary(run_dir)

    assert summary["policy_success_attemptable"] == 7
    assert summary["policy_success_clean_attemptable"] == 1
    assert summary["early_metric_activation_review_needed"] == 1
    assert summary["relation_transfer_proxy_success_unconfirmed"] == 1
    assert summary["orientation_proxy_success_unconfirmed"] == 1
    assert summary["articulation_open_proxy_success_unconfirmed"] == 1
    assert summary["articulation_open_success_unconfirmed"] == 1
    assert summary["can_meat_lid_success_unconfirmed"] == 1
    assert summary["meaningful_policy_caused_transition"] == 6
    task_alpha = next(row for row in summary["skill_task_summary"] if row["task_name"] == "task_alpha")
    assert task_alpha["early_metric_activation_review_needed_count"] == 1
    assert task_alpha["relation_transfer_proxy_success_unconfirmed_count"] == 1
    assert task_alpha["meaningful_policy_caused_transition_count"] == 2
    assert any(
        row["task_name"] == "task_beta" and row["orientation_proxy_success_unconfirmed_count"] == 1
        for row in summary["skill_task_summary"]
    )
    assert any(
        row["task_name"] == "task_gamma" and row["articulation_open_proxy_success_unconfirmed_count"] == 1
        for row in summary["skill_task_summary"]
    )
    assert any(
        row["task_name"] == "task_delta" and row["articulation_open_success_unconfirmed_count"] == 1
        for row in summary["skill_task_summary"]
    )
    assert any(
        row["task_name"] == "can_meat" and row["can_meat_lid_success_unconfirmed_count"] == 1
        for row in summary["skill_task_summary"]
    )
