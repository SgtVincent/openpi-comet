import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

_MODULE_PATH = Path(__file__).with_name("preprocess_b1k_structured_memory.py")
_SPEC = importlib.util.spec_from_file_location("preprocess_b1k_structured_memory", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def _skill(index: int, description: str, start: int, end: int, objects=None, manipulating=None) -> dict:
    objects = objects or [["radio_89"]]
    return {
        "skill_idx": index,
        "skill_description": [description],
        "object_id": objects,
        "manipulating_object_id": manipulating if manipulating is not None else [],
        "spatial_prefix": [],
        "frame_duration": [start, end],
    }


def _primitive(
    index: int, description: str, start: int, end: int, skill_indices: list[int], objects=None, manipulating=None
) -> dict:
    objects = objects or [["radio_89", "coffee_table_koagbh_0"]]
    return {
        "primitive_idx": index,
        "primitive_description": [description],
        "object_id": objects,
        "manipulating_object_id": manipulating if manipulating is not None else ["radio_89"],
        "spatial_prefix": [],
        "frame_duration": [start, end],
        "skill_idxes": skill_indices,
    }


def _write_episode(tmp_path: Path, skills: list[dict], primitives: list[dict], duration: int = 8) -> Path:
    path = tmp_path / "annotations" / "task-0000" / "episode_00000010.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "task_name": "synthetic task",
                "meta_data": {"task_duration": duration},
                "skill_annotation": skills,
                "primitive_annotation": primitives,
            }
        ),
        encoding="utf-8",
    )
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Do the synthetic task."}) + "\n", encoding="utf-8"
    )
    return path


def _metadata(tmp_path: Path, *, include_radio=True, include_template=True, final=None):
    metadata_dir = tmp_path / "language"
    metadata_dir.mkdir(parents=True)
    mapping = {"coffee_table_koagbh_0": "coffee table", "apple_1": "apple", "bowl_2": "bowl"}
    if include_radio:
        mapping["radio_89"] = "radio"
    templates = {
        "primitive": {"pick up from": {"fields": ["obj"], "template": "pick up the {obj} from the {src}"}}
        if include_template
        else {},
        "skill": {
            "move to": {"fields": ["target"], "template": "move to the {target}"},
            "hold": {"fields": ["obj"], "template": "hold the {obj}"},
        },
    }
    paths = []
    for name, value in (("mapping.json", mapping), ("templates.json", templates), ("final.json", final or {})):
        path = metadata_dir / name
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    return mod.load_language_metadata(
        object_mapping_path=paths[0], phrase_templates_path=paths[1], final_supervision_path=paths[2]
    )


def _load(path: Path, metadata, *, strict=False, object_fallback="heuristic", phrase_fallback="label"):
    stats = mod.Counter()
    episode = mod.load_episode(
        path,
        instructions={0: "Do the synthetic task."},
        stats=stats,
        strict=strict,
        language_metadata=metadata,
        object_fallback=object_fallback,
        phrase_fallback=phrase_fallback,
    )
    return episode, stats


def test_object_mapping_skill_template_and_final_supervision_priority(tmp_path: Path) -> None:
    path = _write_episode(tmp_path, [_skill(0, "move to", 0, 4)], [_primitive(0, "pick up from", 0, 4, [0])], 4)
    final = {
        "synthetic_task": {"task_description": "x", "primitive_captions": ["lift the radio from the coffee table"]}
    }
    episode, stats = _load(path, _metadata(tmp_path, final=final))
    assert episode.primitives[0].natural_language_phrase == "lift the radio from the coffee table"
    assert episode.primitives[0].phrase_source == "final_supervision"
    assert episode.skills[0].natural_language_phrase == "move to the radio"
    assert stats["final_supervision_hits"] == 1


def test_multiple_objects_preserve_order_and_grammar(tmp_path: Path) -> None:
    primitive = _primitive(
        0,
        "pick up from",
        0,
        4,
        [0],
        objects=[["apple_1", "radio_89", "coffee_table_koagbh_0"]],
        manipulating=["apple_1", "radio_89"],
    )
    path = _write_episode(tmp_path, [_skill(0, "move to", 0, 4)], [primitive], 4)
    episode, _ = _load(path, _metadata(tmp_path))
    assert episode.primitives[0].natural_language_phrase == "pick up the apple and the radio from the coffee table"


def test_repeated_occurrence_is_readable_and_canonical_has_no_audit_leaks(tmp_path: Path) -> None:
    path = _write_episode(
        tmp_path,
        [_skill(0, "move to", 0, 2), _skill(1, "move to", 2, 4)],
        [_primitive(3, "pick up from", 0, 2, [0]), _primitive(9, "pick up from", 2, 4, [1])],
        4,
    )
    episode, _ = _load(path, _metadata(tmp_path))
    record = mod.build_record(episode, 2, text_modes=["state_target"], sample_kind="boundary")
    canonical = record["canonical_structured_text"]
    assert record["natural_language_current_primitive"]["occurrence"] == "2nd occurrence"
    assert record["raw_current_primitive"]["annotation_index"] == 9
    for forbidden in (
        "radio_89",
        "coffee_table_koagbh_0",
        "annotation_index",
        "segments",
        "task_id",
        "frame_index",
        "phrase_source",
    ):
        assert forbidden not in canonical


def test_unresolved_object_fallback_and_strict_failure(tmp_path: Path) -> None:
    path = _write_episode(tmp_path, [_skill(0, "move to", 0, 4)], [_primitive(0, "pick up from", 0, 4, [0])], 4)
    episode, stats = _load(path, _metadata(tmp_path, include_radio=False), object_fallback="placeholder")
    assert "unknown object" in episode.primitives[0].natural_language_phrase
    assert stats["unresolved_object_handle_occurrences"] > 0
    with pytest.raises(mod.ValidationError, match="unresolved object handle"):
        _load(path, _metadata(tmp_path / "strict", include_radio=False), strict=True)


def test_unresolved_template_fallback_and_strict_failure(tmp_path: Path) -> None:
    path = _write_episode(tmp_path, [_skill(0, "move to", 0, 4)], [_primitive(0, "mystery action", 0, 4, [0])], 4)
    episode, stats = _load(path, _metadata(tmp_path, include_template=False), phrase_fallback="placeholder")
    assert episode.primitives[0].natural_language_phrase == "perform an unknown primitive action"
    assert stats["phrase_fallbacks"] == 1
    with pytest.raises(mod.ValidationError, match="phrase template"):
        _load(path, _metadata(tmp_path / "strict", include_template=False), strict=True)


def test_t_minus_one_boundary_semantics_and_parseable_modes(tmp_path: Path) -> None:
    path = _write_episode(
        tmp_path,
        [_skill(0, "move to", 0, 2), _skill(1, "move to", 2, 4)],
        [_primitive(0, "pick up from", 0, 2, [0]), _primitive(1, "pick up from", 2, 4, [1])],
        4,
    )
    episode, _ = _load(path, _metadata(tmp_path))
    first = mod.build_record(episode, 0, text_modes=["state_target"], sample_kind="boundary")
    boundary = mod.build_record(episode, 2, text_modes=["state_target", "committed_memory"], sample_kind="boundary")
    assert first["natural_language_memory_before"] == {
        "completed_primitive_occurrences": [],
        "active_primitive_occurrence": None,
    }
    assert boundary["natural_language_memory_before"]["active_primitive_occurrence"]["occurrence"] == "1st occurrence"
    assert boundary["natural_language_updated_memory"]["active_primitive_occurrence"]["occurrence"] == "2nd occurrence"
    committed = ET.fromstring(boundary["structured_text"]["committed_memory"]["prefix"])
    committed_value = json.loads(committed.text)
    assert committed_value["active_primitive_occurrence"]["occurrence"] == "2nd occurrence"
    assert "phrase_source" not in committed.text


def test_skill_and_primitive_boundaries_and_end_sentinels(tmp_path: Path) -> None:
    path = _write_episode(
        tmp_path,
        [_skill(0, "move to", 0, 1), _skill(1, "hold", 1, 2), _skill(2, "move to", 2, 4)],
        [_primitive(0, "pick up from", 0, 2, [0, 1]), _primitive(1, "pick up from", 2, 4, [2])],
        4,
    )
    episode, _ = _load(path, _metadata(tmp_path))
    skill_boundary = mod.build_record(episode, 1, text_modes=["state_target"], sample_kind="boundary")
    primitive_boundary = mod.build_record(episode, 2, text_modes=["state_target"], sample_kind="boundary")
    end = mod.build_record(episode, 4, text_modes=["state_target"], sample_kind="boundary")
    assert skill_boundary["natural_language_current_skill"]["phrase"] == "hold the radio"
    assert skill_boundary["natural_language_next_skill"] == {"sentinel": mod.SENTINEL_END_PRIMITIVE}
    assert primitive_boundary["natural_language_current_primitive"]["occurrence"] == "2nd occurrence"
    assert end["natural_language_updated_memory"]["active_primitive_occurrence"] is None
    assert end["natural_language_current_primitive"] == {"sentinel": mod.SENTINEL_GAP}


def test_gap_overlap_and_sampling_regressions(tmp_path: Path) -> None:
    path = _write_episode(
        tmp_path,
        [_skill(0, "move to", 0, 2), _skill(1, "move to", 3, 5)],
        [_primitive(0, "pick up from", 0, 2, [0]), _primitive(1, "pick up from", 3, 5, [1])],
        5,
    )
    episode, stats = _load(path, _metadata(tmp_path))
    assert stats["skill_gaps"] == stats["primitive_gaps"] == 1
    assert mod.build_record(episode, 2, text_modes=["state_target"], sample_kind="boundary")[
        "natural_language_current_primitive"
    ] == {"sentinel": mod.SENTINEL_GAP}
    assert mod.sample_frames(episode, "boundary") == [0, 2, 3, 5]
    with pytest.raises(mod.ValidationError):
        _load(path, _metadata(tmp_path / "strict"), strict=True)
