from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = _REPO_ROOT / "gates" / "moma_pretrain_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_moma_pretrain_gate", _GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _row(*, task_id=0, task="turn on radio", task_key="task-000", episode="episode_000000",
         episode_index=0, memory=0, frame=(0, 4), anchor=(0, 4), transition="normal",
         coverage="full_available_frames", path="task-000/episode_000000.json"):
    return {
        "anchor_range": None if anchor is None else list(anchor),
        "annotation_relative_path": path,
        "coverage_status": coverage,
        "data_version": "data-v1",
        "episode": episode,
        "episode_index": episode_index,
        "frame_range": list(frame),
        "memory_idx": memory,
        "planner_update_target": {
            "current_primitive": "pick up radio",
            "current_skill": "reach radio",
            "next_primitive": "turn on radio",
            "next_skill": "press radio button",
        },
        "prompt_version": "prompt-v1",
        "provenance": {"producer": "v3-smoke"},
        "sampling_group": f"{task_key}:{transition}",
        "schema_version": "b1k_fixed_compact_sampling_spans_v1",
        "task": task,
        "task_id": task_id,
        "task_key": task_key,
        "transition_type": transition,
        "weight_hint": 1.0,
    }


def _canonical(row):
    return (
        row["task_id"], row["episode_index"], row["memory_idx"],
        json.dumps(row["frame_range"], sort_keys=True, separators=(",", ":")),
        json.dumps(row["anchor_range"], sort_keys=True, separators=(",", ":")),
        str(row["sampling_group"]), row["transition_type"],
    )


def _write_dataset(root: Path, rows: list[dict], *, sort=True, manifest_overrides=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=_canonical) if sort else rows
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    spans = root / "sampling_spans.jsonl"
    spans.write_bytes(payload)
    stats_by_key: dict[tuple[int, str], dict[str, int]] = {}
    for row in rows:
        key = (row["task_id"], row["transition_type"])
        stat = stats_by_key.setdefault(key, {"intervals": 0, "frames": 0, "anchors": 0})
        stat["intervals"] += 1
        stat["frames"] += row["frame_range"][1] - row["frame_range"][0]
        if row["anchor_range"] is not None:
            stat["anchors"] += row["anchor_range"][1] - row["anchor_range"][0]
    stats = [
        {"task_id": task_id, "transition_type": transition, **counts}
        for (task_id, transition), counts in sorted(stats_by_key.items())
    ]
    manifest = {
        "sampling_spans": {
            "relative_path": spans.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "count": len(rows),
            "interval_convention": "half-open [start,end)",
            "schema_version": "b1k_fixed_compact_sampling_spans_v1",
        },
        "per_task_transition_stats": stats,
        "data_version": "data-v1",
        "prompt_version": "prompt-v1",
        "schema_version": "b1k_fixed_compact_memory_v2",
    }
    manifest.update(manifest_overrides or {})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _valid_rows():
    return [
        _row(memory=0, frame=(0, 4), anchor=(0, 4)),
        _row(memory=1, frame=(4, 8), anchor=(4, 8), transition="inter_primitive_bridge"),
        # Rare normal group is deliberately only one sample: it need not cover every rank.
        _row(task_id=1, task="open drawer", task_key="task-001", episode="episode_000001",
             episode_index=1, memory=0, frame=(0, 1), anchor=(0, 1),
             path="task-001/episode_000001.json"),
    ]


def _run(root: Path, **kwargs):
    return gate.validate_sampling_spans(
        data_root=root, world_size=2, num_workers=1, min_ranks=2, min_samples=1, **kwargs
    )


def _failed(result, key):
    return key in result["hard_failures"]


def test_v3_smoke_manifest_and_exact_canonical_row_integrate(tmp_path: Path):
    root = _write_dataset(tmp_path, _valid_rows())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    first_row = json.loads((root / "sampling_spans.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(manifest["sampling_spans"]) == {
        "relative_path", "sha256", "count", "interval_convention", "schema_version",
    }
    assert manifest["schema_version"] == "b1k_fixed_compact_memory_v2"
    assert manifest["sampling_spans"]["schema_version"] == "b1k_fixed_compact_sampling_spans_v1"
    assert first_row["schema_version"] == manifest["sampling_spans"]["schema_version"]
    assert first_row["schema_version"] != manifest["schema_version"]
    assert set(manifest["per_task_transition_stats"][0]) == {
        "task_id", "transition_type", "intervals", "frames", "anchors",
    }
    assert set(first_row) == gate.SAMPLING_SPAN_FIELDS
    assert isinstance(first_row["task_id"], int)
    assert isinstance(first_row["planner_update_target"], dict)
    result = _run(root)
    assert result["verdict"] == "PASS", result
    assert result["rank_totals"] == [5, 4]
    assert result["rank_coverage"]["0::inter_primitive_bridge"] == [2, 2]
    assert result["rank_coverage"]["1::normal"] in ([1, 0], [0, 1])


def test_legacy_flat_manifest_shape_is_rejected(tmp_path: Path):
    rows = _valid_rows()
    root = _write_dataset(tmp_path, rows)
    spans = root / "sampling_spans.jsonl"
    legacy = {
        "sampling_spans_path": spans.name,
        "sha256": hashlib.sha256(spans.read_bytes()).hexdigest(),
        "records": len(rows),
        "sampling_stats_by_task_transition": {},
        "data_version": "data-v1",
        "prompt_version": "prompt-v1",
        "schema_version": "sampling-v1",
    }
    (root / "manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
    result = _run(root)
    assert result["verdict"] == "FAIL"
    assert _failed(result, "SAMPLING_MANIFEST_CONTRACT")
    assert _failed(result, "SAMPLING_SPANS_EXISTS")


def test_nested_sampling_schema_version_is_required(tmp_path: Path):
    root = _write_dataset(tmp_path, _valid_rows(), manifest_overrides={
        "sampling_spans": {
            "relative_path": "sampling_spans.jsonl",
            "sha256": "0" * 64,
            "count": len(_valid_rows()),
            "interval_convention": "half-open [start,end)",
        },
    })
    result = _run(root)
    assert _failed(result, "SAMPLING_MANIFEST_CONTRACT")


def test_sha_record_and_version_are_fail_closed(tmp_path: Path):
    root = _write_dataset(tmp_path, _valid_rows(), manifest_overrides={
        "sampling_spans": {
            "relative_path": "sampling_spans.jsonl",
            "sha256": "0" * 64,
            "count": 999,
            "interval_convention": "half-open [start,end)",
            "schema_version": "b1k_fixed_compact_sampling_spans_v1",
        },
        "prompt_version": "other",
    })
    result = _run(root)
    assert _failed(result, "SAMPLING_SPANS_SHA256")
    assert _failed(result, "SAMPLING_SPANS_RECORDS")
    assert _failed(result, "SAMPLING_ROW_SCHEMA_VERSION")


@pytest.mark.parametrize(
    ("mutate", "failure"),
    [
        (lambda rows: rows[0]["planner_update_target"].__setitem__("next_skill", ""),
         "SAMPLING_ROW_SCHEMA_VERSION"),
        (lambda rows: rows[0].__setitem__("anchor_range", [0, 5]), "SAMPLING_HALF_OPEN_RANGES"),
        (lambda rows: rows[0].__setitem__("coverage_status", "dead"), "SAMPLING_DEAD_NOT_TRAINABLE"),
        (lambda rows: rows[1].__setitem__("annotation_relative_path", "task-001/episode_000001.json"),
         "SAMPLING_ANNOTATION_PATH_KEY"),
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "SAMPLING_STABLE_SORT_UNIQUE"),
    ],
)
def test_structural_negative_controls(tmp_path: Path, mutate, failure: str):
    rows = _valid_rows()
    mutate(rows)
    result = _run(_write_dataset(tmp_path, rows))
    assert _failed(result, failure), result


def test_unsorted_rows_are_rejected(tmp_path: Path):
    rows = _valid_rows()
    rows[0], rows[-1] = rows[-1], rows[0]
    result = _run(_write_dataset(tmp_path, rows, sort=False))
    assert _failed(result, "SAMPLING_STABLE_SORT_UNIQUE")


def test_bridge_or_update_cannot_land_on_one_rank(tmp_path: Path):
    rows = [_row(anchor=(0, 1), frame=(0, 1), transition="primitive_entry_skill_bridge")]
    result = _run(_write_dataset(tmp_path, rows))
    assert not _failed(result, "SAMPLING_ROW_SCHEMA_VERSION")
    assert _failed(result, "SAMPLING_SHARD_EVERY_RANK_NONEMPTY")
    assert _failed(result, "SAMPLING_TASK_TRANSITION_RANK_COVERAGE")
    failure = next(c for c in result["checks"] if c["key"] == "SAMPLING_TASK_TRANSITION_RANK_COVERAGE")
    assert failure["threshold"]["bridge_update_min_ranks"] == 2


def test_manifest_task_transition_stats_must_match(tmp_path: Path):
    root = _write_dataset(tmp_path, _valid_rows(), manifest_overrides={
        "per_task_transition_stats": [
            {"task_id": 0, "transition_type": "normal", "intervals": 999, "frames": 4, "anchors": 4},
        ],
    })
    assert _failed(_run(root), "SAMPLING_TASK_TRANSITION_STATS")


def test_configurable_min_samples_is_enforced(tmp_path: Path):
    result = gate.validate_sampling_spans(
        data_root=_write_dataset(tmp_path, _valid_rows()),
        world_size=2,
        num_workers=1,
        min_ranks=2,
        min_samples=2,
    )
    assert _failed(result, "SAMPLING_TASK_TRANSITION_RANK_COVERAGE")
    assert any(x["group"] == "1::normal" for x in next(
        c for c in result["checks"] if c["key"] == "SAMPLING_TASK_TRANSITION_RANK_COVERAGE"
    )["measured"]["failures"])


def test_mode_uses_existing_formal_diagnostic_contract():
    diagnostic = gate.parse_args(["--self-test", "--mode", "diagnostic"])
    assert diagnostic.mode == "diagnostic"
    with pytest.raises(SystemExit):
        gate.parse_args(["--self-test", "--mode", "smoke"])


def test_formal_missing_sampling_product_hard_fails_and_writes_report(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    rc = gate.main([
        "--data-root", str(tmp_path), "--mode", "formal", "--world-size", "2",
        "--num-workers", "1", "--subtask-max-len", "128", "--prompt-max-len", "512",
        "--frames-meta-root", str(tmp_path / "frames"), "--anchor-stride", "1",
        "--anchor-offset", "0", "--chunk-stats-json", str(tmp_path / "chunks.json"),
        "--config-scope-expect", str(tmp_path / "scope.json"),
        "--distribution-baselines", str(tmp_path / "baselines.json"),
        "--expect-tokenizer-md5", "deadbeef", "--expect-vocab-size", "1",
        "--out", str(report),
    ])
    assert rc == 1
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["verdict"] == "FAIL"
    assert "SAMPLING_MANIFEST_CONTRACT" in saved["hard_failures"]
    assert "SAMPLING_SPANS_EXISTS" in saved["hard_failures"]
    assert all(check["hard"] for check in saved["checks"])


def test_diagnostic_missing_sampling_product_is_not_measured_warning(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "diagnostic-report.json"
    rc = gate.main([
        "--data-root", str(tmp_path), "--mode", "diagnostic", "--world-size", "2",
        "--num-workers", "1", "--subtask-max-len", "128", "--prompt-max-len", "512",
        "--frames-meta-root", "", "--out", str(report),
    ])
    # 后续因没有 episode 文件而 CANNOT-ASSESS；sampling 本身不得 hard-fail。
    assert rc == 3
    saved = json.loads(report.read_text(encoding="utf-8"))
    sampling = saved["sampling"]
    assert sampling["verdict"] == "NOT-MEASURED"
    assert sampling["hard_failures"] == []
    assert "SAMPLING_SPANS_EXISTS" in sampling["warnings"]
    sampling_checks = [check for check in saved["checks"] if check["key"].startswith("SAMPLING_")]
    assert sampling_checks
    assert all(not check["hard"] for check in sampling_checks)
    failed_checks = [check for check in sampling_checks if not check["passed"]]
    assert failed_checks
    assert any("NOT-MEASURED" in check["detail"] for check in failed_checks)


def test_episode_schema_is_bound_to_manifest_not_legacy_constant():
    v2 = "b1k_fixed_compact_memory_v2"
    assert gate.episode_schema_matches({"memory_schema_version": v2}, v2)
    assert not gate.episode_schema_matches({"memory_schema_version": gate.EXPECTED_SCHEMA_VERSION}, v2)
    assert not gate.episode_schema_matches({"memory_schema_version": v2}, None)


def test_current_tokenize_memory_api_counts_untruncated_tokens_without_legacy_method():
    class Codec:
        def encode(self, text, *, validate=True):
            assert text == "Memory: x"
            assert validate is True
            return [10, 11, 12]

    class Tokenizer:
        # Deliberately has no memory_token_length: this is the current smoke API contract.
        def memory_codec(self):
            return Codec()

        def tokenize_memory(self, text, *, validate=True):
            assert text == "Memory: x"
            assert validate is True
            return (
                gate.np.asarray([1, 10, 11, 12, 2, 0, 0, 0], dtype=gate.np.int32),
                gate.np.asarray([1, 1, 1, 1, 1, 0, 0, 0], dtype=gate.np.bool_),
                gate.np.asarray([1, 1, 1, 1, 1, 0, 0, 0], dtype=gate.np.int32),
                gate.np.asarray([0, 1, 1, 1, 1, 0, 0, 0], dtype=gate.np.bool_),
            )

    result = gate.tokenize_memory_untruncated(Tokenizer(), "Memory: x", subtask_max_len=8)
    assert result["length"] == 5  # three body tokens plus BOS/EOS, not padded length 8
    assert int(result["encoded"][1].sum()) == 5


def test_scan_episode_uses_manifest_schema_and_checks_production_masks(tmp_path: Path, monkeypatch):
    class SP:
        def encode(self, _text, add_bos=False):
            return [1, 10] if add_bos else [10]

        def bos_id(self):
            return 1

        def eos_id(self):
            return 2

    class Codec:
        def encode(self, _text, *, validate=True):
            assert validate is True
            return [10, 11, 12]

    class Tokenizer:
        def __init__(self, *, break_production_eos=False):
            self._tokenizer = SP()
            self.break_production_eos = break_production_eos

        def tokenize_subtask(self, _text):
            return (
                gate.np.asarray([1, 10, 2, 0], dtype=gate.np.int32),
                gate.np.asarray([1, 1, 1, 0], dtype=gate.np.bool_),
                gate.np.asarray([1, 1, 1, 0], dtype=gate.np.int32),
                gate.np.asarray([0, 1, 1, 0], dtype=gate.np.bool_),
            )

        def memory_codec(self):
            return Codec()

        def tokenize_memory(self, _text, *, validate=True):
            assert validate is True
            loss = [0, 1, 1, 1, 1, 0, 0, 0]
            if self.break_production_eos:
                loss[4] = 0
            return (
                gate.np.asarray([1, 10, 11, 12, 2, 0, 0, 0], dtype=gate.np.int32),
                gate.np.asarray([1, 1, 1, 1, 1, 0, 0, 0], dtype=gate.np.bool_),
                gate.np.asarray([1, 1, 1, 1, 1, 0, 0, 0], dtype=gate.np.int32),
                gate.np.asarray(loss, dtype=gate.np.bool_),
            )

    row = {
        "memory_idx": 0,
        "frame_duration": [0, 4],
        "previous_fixed_compact_memory": gate.INITIAL_PREVIOUS_MEMORY,
        "fixed_compact_memory": "ready",
        "completed_primitive_stack": [],
        "transition_type": "normal",
        "current_primitive": "pick up radio",
        "current_skill": "reach radio",
        "next_skill": "press button",
        "next_primitive": "turn on radio",
    }
    row["current_memory"] = {
        "primitive": row["current_primitive"], "skill": row["current_skill"],
        "next_skill": row["next_skill"], "next_primitive": row["next_primitive"],
        "transition_type": row["transition_type"],
    }
    row["model_target_text"] = gate.build_planner_target_text(row) + gate.ACTION_QUERY_SUFFIX
    episode = {
        "task_name": "turn on radio",
        "memory_schema_version": "b1k_fixed_compact_memory_v2",
        "memory_interval_convention": gate.EXPECTED_INTERVAL_CONVENTION,
        "meta_data": {"valid_duration": [0, 4]},
        "skill_annotation": [], "primitive_annotation": [], "memory_annotation": [row],
    }
    path = tmp_path / "episode_000000.json"
    path.write_text(json.dumps(episode), encoding="utf-8")
    opts = {
        "subtask_max_len": 8, "prompt_max_len": 512, "deep_paths": set(),
        "frames_per_episode": 0, "seed": 0, "episode_lengths": None,
        "chunk_size": 250, "anchor_stride": 1, "anchor_offset": 0,
        "task_text_source": "annotation", "episode_tasks": {}, "action_dim": 1,
        "expected_schema_version": "b1k_fixed_compact_memory_v2",
    }
    monkeypatch.setattr(gate, "_STATE_STR", "0")
    monkeypatch.setattr(gate, "_OPTS", opts)
    monkeypatch.setattr(gate, "_TOK", Tokenizer())
    good = gate.scan_episode(str(path))
    assert good["violations"].get("schema_version", 0) == 0
    assert good["violations"].get("memory_codec_rejects_text", 0) == 0
    assert good["prod_lengths"] == [5]

    episode["memory_schema_version"] = gate.EXPECTED_SCHEMA_VERSION
    path.write_text(json.dumps(episode), encoding="utf-8")
    mixed = gate.scan_episode(str(path))
    assert mixed["violations"]["schema_version"] == 1

    episode["memory_schema_version"] = "b1k_fixed_compact_memory_v2"
    path.write_text(json.dumps(episode), encoding="utf-8")
    monkeypatch.setattr(gate, "_TOK", Tokenizer(break_production_eos=True))
    broken_mask = gate.scan_episode(str(path))
    assert broken_mask["violations"]["eos_not_supervised"] == 1
