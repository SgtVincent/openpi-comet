import importlib.util
from pathlib import Path
from unittest import mock

_MODULE_PATH = Path(__file__).with_name("build_offline_integrity_manifest.py")
_SPEC = importlib.util.spec_from_file_location("build_offline_integrity_manifest", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_parse_frame_range_accepts_explicit_or_duration_and_sorts() -> None:
    assert mod.parse_frame_range({"frame_start": "8", "frame_end": "3"}) == (3, 8)
    assert mod.parse_frame_range({"frame_duration": "[10, 20]"}) == (10, 20)
    assert mod.parse_frame_range({"frame_duration": [20, 10]}) == (10, 20)
    assert mod.parse_frame_range({"frame_duration": "bad"}) is None


def test_episode_index_uses_full_behavior_episode_id() -> None:
    assert mod.episode_index({"episode_index": "270670"}) == 270670
    assert mod.episode_index({"demo_id": "00270670"}) == 270670
    assert mod.episode_index({"episode_id": "00480290"}) == 480290
    assert mod.episode_index({"demo_id": "bad"}) is None


def test_annotation_duration_check_treats_frame_end_as_exclusive(tmp_path: Path) -> None:
    annotation = tmp_path / "episode.json"
    annotation.write_text(
        """
{
  "meta_data": {"task_duration": 20, "valid_duration": [0, 20]},
  "skill_annotation": [
    {"skill_idx": 1, "frame_duration": [10, 20]}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    issues, details = mod.check_annotation(
        annotation,
        {"skill_idx": 1},
        (10, 20),
    )

    assert issues == []
    assert details["annotated_frame_duration"] == [10, 20]
    assert details["annotation_frame_upper_bound"] == 20


def test_annotation_duration_check_uses_global_valid_duration_for_shifted_episode(tmp_path: Path) -> None:
    annotation = tmp_path / "episode.json"
    annotation.write_text(
        """
{
  "meta_data": {"task_duration": 16122, "valid_duration": [90, 16212]},
  "skill_annotation": [
    {"skill_idx": 60, "frame_duration": [16085, 16212]}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    issues, details = mod.check_annotation(
        annotation,
        {"skill_idx": 60},
        (16085, 16212),
    )

    assert issues == []
    assert details["annotation_frame_lower_bound"] == 90
    assert details["annotation_frame_upper_bound"] == 16212


def test_summary_counts_offline_hard_exclude_only() -> None:
    rows = [
        {
            "recommended_bucket": "clean_pass",
            "train_disposition": "keep",
            "offline_integrity_issues": [],
            "task_name": "task_a",
        },
        {
            "recommended_bucket": "offline_hard_exclude",
            "train_disposition": "exclude",
            "offline_integrity_issues": ["missing_episode_parquet"],
            "task_name": "task_b",
        },
    ]

    summary = mod.build_summary(Path("/tmp/source.json"), rows)

    assert summary["bucket_counts"] == {"clean_pass": 1, "offline_hard_exclude": 1}
    assert summary["train_disposition_counts"] == {"exclude": 1, "keep": 1}
    assert summary["issue_counts"] == {"missing_episode_parquet": 1}
    assert "simulator replay" in summary["note"]


def test_hdf5_pcd_only_container_does_not_require_action_state(tmp_path: Path) -> None:
    fake_h5py = mock.MagicMock()
    fake_file = mock.MagicMock()
    fake_demo = mock.MagicMock()
    fake_pcd = mock.MagicMock()
    fake_pcd.shape = (30, 4096, 6)
    fake_demo.get.side_effect = lambda key: {
        "robot_r1::fused_pcd": fake_pcd,
        "action": None,
        "state": None,
    }.get(key)
    fake_file.get.return_value = fake_demo
    fake_file.__enter__.return_value = fake_file
    fake_file.__exit__.return_value = False
    fake_h5py.File.return_value = fake_file

    with mock.patch.dict("sys.modules", {"h5py": fake_h5py}):
        issues, details = mod.check_hdf5(tmp_path / "episode.hdf5", (5, 20))

    assert issues == []
    assert details["hdf5_fused_pcd_shape"] == [30, 4096, 6]
    assert details["hdf5_action_source"] == "absent_in_pcd_container"
    assert details["hdf5_state_source"] == "absent_in_pcd_container"


def test_build_output_record_can_skip_video_decode_checks(tmp_path: Path) -> None:
    record = {
        "sample_id": 0,
        "episode_index": 1,
        "frame_start": 0,
        "frame_end": 1,
        "episode_json": "",
        "episode_parquet": "",
        "rawdata_hdf5": "",
        "original_rgb_head": "",
        "original_rgb_left_wrist": "",
        "original_rgb_right_wrist": "",
        "original_depth_head": "",
    }

    output = mod.build_output_record(
        record,
        source=tmp_path / "source.json",
        video_sample_frames=1,
        skip_video_checks=True,
    )

    assert output["details"]["videos"] == {"video_check_skipped": "disabled_by_cli"}
