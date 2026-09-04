#!/usr/bin/env python3
"""Stream natural-language structured memory labels from BEHAVIOR-1K annotations.

Durations are half-open [start, end).  For frame t, memory_before is the
committed state at t-1 and updated_memory is the state at t.  Boundary
sampling emits every annotation start/end, including terminal completion.
Raw annotation-derived labels remain in raw_* audit fields; structured_text
is built exclusively from natural_language_* fields.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import re
import sys
from typing import Any
from xml.sax.saxutils import escape

DEFAULT_METADATA_DIR = Path("/mnt/bn/behavior-data-hl/chenjunting/repo/il_lib/tutorials")
SENTINEL_GAP = "GAP"
SENTINEL_END_PRIMITIVE = "END_OF_PRIMITIVE"
SENTINEL_END_EPISODE = "END_OF_EPISODE"
SENTINEL_NO_ACTIVE_PRIMITIVE = "NO_ACTIVE_PRIMITIVE"


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    start: int
    end: int


@dataclass
class Occurrence:
    kind: str
    source_index: int
    annotation_index: int
    raw_description: str
    raw_annotation: dict[str, Any]
    object_binding: dict[str, list[str]]
    object_groups: list[list[str]]
    spatial_prefix: Any
    segments: list[Segment]
    occurrence_index: int = -1
    description_occurrence_index: int = -1
    occurrence_id: str = ""
    natural_language_phrase: str = ""
    phrase_source: str = ""
    skill_indices: list[int] = field(default_factory=list)
    skills: list[Occurrence] = field(default_factory=list)

    @property
    def start(self) -> int:
        return min(segment.start for segment in self.segments)

    @property
    def end(self) -> int:
        return max(segment.end for segment in self.segments)

    def active_at(self, frame: int) -> bool:
        return any(segment.start <= frame < segment.end for segment in self.segments)


@dataclass
class Episode:
    path: Path
    task_id: int
    episode_index: int
    task_name: str | None
    task_instruction: str | None
    duration: int
    primitives: list[Occurrence]
    skills: list[Occurrence]


@dataclass
class LanguageMetadata:
    object_names: dict[str, str]
    templates: dict[str, dict[str, dict[str, Any]]]
    final_supervision: dict[str, dict[str, Any]]
    raw_handle_slug_tokens: set[str]


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be a JSON object: {path}")
    return value


def load_language_metadata(
    *, object_mapping_path: Path, phrase_templates_path: Path, final_supervision_path: Path
) -> LanguageMetadata:
    object_names = _load_json_object(object_mapping_path, "object mapping")
    templates = _load_json_object(phrase_templates_path, "phrase templates")
    final_supervision = _load_json_object(final_supervision_path, "final supervision")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in object_names.items()):
        raise ValidationError("object mapping must be string-to-string")
    for kind in ("primitive", "skill"):
        if not isinstance(templates.get(kind), dict):
            raise ValidationError(f"phrase templates missing {kind!r} object")
    slug_tokens: set[str] = set()
    for handle, mapped_name in object_names.items():
        normalized = handle.strip().replace("-", "_")
        parts = [part for part in normalized.split("_") if part]
        if parts and parts[-1].isdigit():
            parts.pop()
        if len(parts) >= 2 and re.fullmatch(r"[a-z]{5,10}", parts[-1]):
            mapped_tokens = set(re.findall(r"[a-z]+", mapped_name.casefold()))
            if parts[-1] not in mapped_tokens:
                slug_tokens.add(parts[-1])
    return LanguageMetadata(object_names, templates, final_supervision, slug_tokens)


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_strings(item))
        return output
    return []


def _flatten_ints(value: Any) -> list[int]:
    if type(value) is int:
        return [value]
    if isinstance(value, list):
        output: list[int] = []
        for item in value:
            output.extend(_flatten_ints(item))
        return output
    return []


def _description(annotation: dict[str, Any], kind: str) -> str:
    values = [item.strip() for item in _flatten_strings(annotation.get(f"{kind}_description")) if item.strip()]
    return " + ".join(values) if values else f"UNKNOWN_{kind.upper()}"


def _object_groups(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        values = _flatten_strings(value)
        return [values] if values else []
    if value and all(not isinstance(item, list | tuple) for item in value):
        values = _flatten_strings(value)
        return [values] if values else []
    groups = [_flatten_strings(item) for item in value]
    return [group for group in groups if group]


def _binding(annotation: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "object_ids": _flatten_strings(annotation.get("object_id")),
        "manipulating_object_ids": _flatten_strings(annotation.get("manipulating_object_id")),
    }


def parse_duration(value: Any, *, context: str, stats: Counter[str], strict: bool) -> list[Segment]:
    raw_pairs: list[tuple[int, int]] = []
    canonical = False
    if isinstance(value, list) and len(value) == 2 and all(type(item) is int for item in value):
        raw_pairs = [(value[0], value[1])]
        canonical = True
    elif (
        isinstance(value, list)
        and value
        and all(
            isinstance(item, list) and len(item) == 2 and all(type(point) is int for point in item) for item in value
        )
    ):
        raw_pairs = [(item[0], item[1]) for item in value]
        canonical = True
    else:
        integers = _flatten_ints(value)
        if len(integers) >= 2:
            raw_pairs = [(min(integers), max(integers))]
            stats["coerced_durations"] += 1
        stats["schema_errors"] += 1
        if strict:
            raise ValidationError(f"{context}: unsupported frame_duration {value!r}")
    segments: list[Segment] = []
    for start, end in raw_pairs:
        if start < 0 or end <= start:
            stats["schema_errors"] += 1
            if strict:
                raise ValidationError(f"{context}: invalid half-open interval [{start}, {end})")
            continue
        segments.append(Segment(start, end))
    if canonical and not segments:
        stats["empty_durations"] += 1
    return sorted(segments, key=lambda item: (item.start, item.end))


def _timeline_anomalies(occurrences: list[Occurrence], prefix: str, stats: Counter[str]) -> None:
    segments = sorted(
        ((segment.start, segment.end, item.source_index) for item in occurrences for segment in item.segments),
        key=lambda item: (item[0], item[1], item[2]),
    )
    if not segments:
        return
    frontier = segments[0][1]
    for start, end, _ in segments[1:]:
        if start > frontier:
            stats[f"{prefix}_gaps"] += 1
            stats[f"{prefix}_gap_frames"] += start - frontier
        elif start < frontier:
            stats[f"{prefix}_overlaps"] += 1
            stats[f"{prefix}_overlap_frames"] += min(end, frontier) - start
        frontier = max(frontier, end)


def _make_occurrences(
    annotations: Any, *, kind: str, path: Path, stats: Counter[str], strict: bool
) -> list[Occurrence]:
    if not isinstance(annotations, list):
        stats["schema_errors"] += 1
        if strict:
            raise ValidationError(f"{path}: {kind}_annotation must be a list")
        return []
    output: list[Occurrence] = []
    for source_index, annotation in enumerate(annotations):
        context = f"{path}:{kind}_annotation[{source_index}]"
        if not isinstance(annotation, dict):
            stats["schema_errors"] += 1
            if strict:
                raise ValidationError(f"{context}: expected object")
            continue
        segments = parse_duration(annotation.get("frame_duration"), context=context, stats=stats, strict=strict)
        if not segments:
            continue
        binding = _binding(annotation)
        if not binding["object_ids"]:
            stats[f"missing_{kind}_object_binding"] += 1
        index_value = annotation.get(f"{kind}_idx", source_index)
        if type(index_value) is not int:
            stats["schema_errors"] += 1
            if strict:
                raise ValidationError(f"{context}: {kind}_idx must be an integer")
            index_value = source_index
        skill_indices = annotation.get("skill_idxes", []) if kind == "primitive" else []
        if not isinstance(skill_indices, list) or not all(type(item) is int for item in skill_indices):
            stats["schema_errors"] += 1
            if strict:
                raise ValidationError(f"{context}: skill_idxes must be integer list")
            skill_indices = _flatten_ints(skill_indices)
        output.append(
            Occurrence(
                kind=kind,
                source_index=source_index,
                annotation_index=index_value,
                raw_description=_description(annotation, kind),
                raw_annotation=annotation,
                object_binding=binding,
                object_groups=_object_groups(annotation.get("object_id")),
                spatial_prefix=annotation.get("spatial_prefix", []),
                segments=segments,
                skill_indices=skill_indices,
            )
        )
    output.sort(key=lambda item: (item.start, item.source_index))
    repetitions: Counter[str] = Counter()
    for occurrence_index, occurrence in enumerate(output):
        repetitions[occurrence.raw_description] += 1
        occurrence.occurrence_index = occurrence_index
        occurrence.description_occurrence_index = repetitions[occurrence.raw_description]
        occurrence.occurrence_id = f"{kind}:{occurrence_index:04d}"
    return output


def _normalize_task_key(value: str | None) -> str:
    return re.sub(r"[\s_]+", "_", (value or "").strip()).casefold()


def _humanize_handle(handle: str) -> str:
    parts = [part for part in handle.strip().replace("-", "_").split("_") if part]
    while parts and parts[-1].isdigit():
        parts.pop()
    if parts and re.fullmatch(r"[a-z]{5,10}", parts[-1]):
        parts.pop()
    if parts and parts[0] == "floors":
        parts[0] = "floor"
    return " ".join(parts).strip().lower()


def _natural_object(
    handle: str, *, metadata: LanguageMetadata, stats: Counter[str], strict: bool, fallback: str
) -> str:
    if handle in metadata.object_names:
        return metadata.object_names[handle]
    stripped = handle.strip()
    if stripped in metadata.object_names:
        stats["object_mapping_normalized_fallbacks"] += 1
        return metadata.object_names[stripped]
    stats["unresolved_object_handle_occurrences"] += 1
    stats[f"unresolved_object_handle::{handle}"] += 1
    if strict:
        raise ValidationError(f"unresolved object handle: {handle!r}")
    stats["object_fallbacks"] += 1
    if fallback == "placeholder":
        return "unknown object"
    phrase = _humanize_handle(handle)
    return phrase or "unknown object"


def _join_objects(names: list[str]) -> str:
    names = list(dict.fromkeys(name for name in names if name))
    if not names:
        return "object"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and the {names[1]}"
    return ", ".join(names[:-1]) + f", and the {names[-1]}"


def _template_phrase(
    occurrence: Occurrence,
    *,
    metadata: LanguageMetadata,
    stats: Counter[str],
    strict: bool,
    object_fallback: str,
    phrase_fallback: str,
) -> str:
    template_spec = metadata.templates.get(occurrence.kind, {}).get(occurrence.raw_description)
    if not isinstance(template_spec, dict) or not isinstance(template_spec.get("template"), str):
        stats["unresolved_phrase_template_occurrences"] += 1
        stats[f"unresolved_phrase_template::{occurrence.kind}::{occurrence.raw_description}"] += 1
        if strict:
            raise ValidationError(f"unresolved {occurrence.kind} phrase template: {occurrence.raw_description!r}")
        stats["phrase_fallbacks"] += 1
        if phrase_fallback == "placeholder":
            return f"perform an unknown {occurrence.kind} action"
        return occurrence.raw_description.replace("_", " ").strip().lower()
    mapped_groups = [
        [
            _natural_object(handle, metadata=metadata, stats=stats, strict=strict, fallback=object_fallback)
            for handle in group
        ]
        for group in occurrence.object_groups
    ]
    flat = [name for group in mapped_groups for name in group]
    manipulating = [
        _natural_object(handle, metadata=metadata, stats=stats, strict=strict, fallback=object_fallback)
        for handle in occurrence.object_binding["manipulating_object_ids"]
    ]
    target = flat[-1] if flat else "object"
    mains = manipulating or (flat[:-1] if len(flat) > 1 else flat)
    values = {
        "obj": _join_objects(mains),
        "target": target,
        "src": target,
        "dst": target,
    }
    fields = re.findall(r"{([^}]+)}", template_spec["template"])
    missing = [name for name in fields if name not in values]
    if missing:
        stats["unresolved_phrase_template_occurrences"] += 1
        if strict:
            raise ValidationError(f"unsupported template fields {missing!r}")
        stats["phrase_fallbacks"] += 1
        return occurrence.raw_description.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", template_spec["template"].format(**values)).strip()


def _final_caption(occurrence: Occurrence, task_name: str | None, metadata: LanguageMetadata) -> str | None:
    if occurrence.kind != "primitive":
        return None
    normalized = _normalize_task_key(task_name)
    matches = [value for key, value in metadata.final_supervision.items() if _normalize_task_key(key) == normalized]
    if len(matches) != 1:
        return None
    captions = matches[0].get("primitive_captions")
    if not isinstance(captions, list) or not (0 <= occurrence.annotation_index < len(captions)):
        return None
    caption = captions[occurrence.annotation_index]
    return caption.strip() if isinstance(caption, str) and caption.strip() else None


def _phrase_pollution(phrase: str, metadata: LanguageMetadata) -> list[str]:
    words = set(re.findall(r"[a-z]+", phrase.casefold()))
    return sorted(words & metadata.raw_handle_slug_tokens)


def apply_natural_language(
    episode: Episode,
    *,
    metadata: LanguageMetadata,
    stats: Counter[str],
    strict: bool,
    object_fallback: str,
    phrase_fallback: str,
) -> None:
    for occurrence in [*episode.primitives, *episode.skills]:
        generated = _template_phrase(
            occurrence,
            metadata=metadata,
            stats=stats,
            strict=strict,
            object_fallback=object_fallback,
            phrase_fallback=phrase_fallback,
        )
        final = _final_caption(occurrence, episode.task_name, metadata)
        if final is not None:
            pollution = _phrase_pollution(final, metadata)
            if pollution:
                stats["final_supervision_pollution_rejections"] += 1
                occurrence.natural_language_phrase = generated
                occurrence.phrase_source = "phrase_template_after_final_pollution_rejection"
            else:
                stats["final_supervision_hits"] += 1
                occurrence.natural_language_phrase = final
                occurrence.phrase_source = "final_supervision"
        else:
            if occurrence.kind == "primitive":
                stats["final_supervision_misses"] += 1
            occurrence.natural_language_phrase = generated
            occurrence.phrase_source = "phrase_template"


def _task_id_from_path(path: Path) -> int:
    match = re.fullmatch(r"task-(\d+)", path.parent.name)
    if match is None:
        raise ValidationError(f"{path}: parent directory must be task-NNNN")
    return int(match.group(1))


def _episode_index_from_path(path: Path) -> int:
    match = re.fullmatch(r"episode_(\d+)", path.stem)
    if match is None:
        raise ValidationError(f"{path}: filename must be episode_NNNNNNNN.json")
    return int(match.group(1))


def load_task_instructions(annotation_root: Path) -> dict[int, str]:
    meta_path = annotation_root.parent / "meta" / "tasks.jsonl"
    if not meta_path.is_file():
        return {}
    output: dict[int, str] = {}
    with meta_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if type(item.get("task_index")) is int and isinstance(item.get("task"), str):
                    output[item["task_index"]] = item["task"]
    return output


def load_episode(
    path: Path,
    *,
    instructions: dict[int, str],
    stats: Counter[str],
    strict: bool,
    language_metadata: LanguageMetadata | None = None,
    object_fallback: str = "heuristic",
    phrase_fallback: str = "label",
) -> Episode:
    try:
        annotation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        stats["schema_errors"] += 1
        raise ValidationError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(annotation, dict):
        stats["schema_errors"] += 1
        raise ValidationError(f"{path}: root must be an object")
    task_id = _task_id_from_path(path)
    episode_index = _episode_index_from_path(path)
    skills = _make_occurrences(annotation.get("skill_annotation"), kind="skill", path=path, stats=stats, strict=strict)
    primitives = _make_occurrences(
        annotation.get("primitive_annotation"), kind="primitive", path=path, stats=stats, strict=strict
    )
    skills_by_index = {skill.annotation_index: skill for skill in skills}
    for primitive in primitives:
        primitive.skills = [skills_by_index[index] for index in primitive.skill_indices if index in skills_by_index]
        missing_links = sum(index not in skills_by_index for index in primitive.skill_indices)
        stats["missing_skill_links"] += missing_links
        if missing_links and strict:
            raise ValidationError(f"{path}: primitive {primitive.annotation_index} has missing skill_idxes")
        primitive.skills.sort(key=lambda item: (item.start, item.source_index))
    linked = {skill.annotation_index for primitive in primitives for skill in primitive.skills}
    stats["unlinked_skills"] += sum(skill.annotation_index not in linked for skill in skills)
    anomaly_keys = ("skill_gaps", "skill_overlaps", "primitive_gaps", "primitive_overlaps")
    anomalies_before = sum(stats[key] for key in anomaly_keys)
    _timeline_anomalies(skills, "skill", stats)
    _timeline_anomalies(primitives, "primitive", stats)
    if strict and sum(stats[key] for key in anomaly_keys) > anomalies_before:
        raise ValidationError(f"{path}: gap or overlap found in strict mode")
    metadata = annotation.get("meta_data")
    duration_value = metadata.get("task_duration") if isinstance(metadata, dict) else None
    max_end = max((item.end for item in [*skills, *primitives]), default=0)
    if type(duration_value) is int and duration_value >= max_end:
        duration = duration_value
    else:
        duration = max_end
        stats["invalid_task_duration"] += 1
        if strict:
            raise ValidationError(f"{path}: invalid task_duration {duration_value!r}; max end is {max_end}")
    episode = Episode(
        path=path,
        task_id=task_id,
        episode_index=episode_index,
        task_name=annotation.get("task_name") if isinstance(annotation.get("task_name"), str) else None,
        task_instruction=instructions.get(task_id),
        duration=duration,
        primitives=primitives,
        skills=skills,
    )
    if language_metadata is not None:
        apply_natural_language(
            episode,
            metadata=language_metadata,
            stats=stats,
            strict=strict,
            object_fallback=object_fallback,
            phrase_fallback=phrase_fallback,
        )
    return episode


def _choose_active(occurrences: Iterable[Occurrence], frame: int) -> Occurrence | None:
    candidates = [item for item in occurrences if item.active_at(frame)]
    return max(candidates, key=lambda item: (item.start, item.source_index), default=None)


def _ordinal(number: int) -> str:
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix} occurrence"


def _raw_label(occurrence: Occurrence | None, sentinel: str = SENTINEL_GAP) -> dict[str, Any]:
    if occurrence is None:
        return {"sentinel": sentinel}
    return {
        "occurrence_id": occurrence.occurrence_id,
        "occurrence_index": occurrence.occurrence_index,
        "description_occurrence_index": occurrence.description_occurrence_index,
        "annotation_index": occurrence.annotation_index,
        "raw_description": occurrence.raw_description,
        "raw_object_binding": occurrence.object_binding,
        "raw_annotation": occurrence.raw_annotation,
        "segments": [[segment.start, segment.end] for segment in occurrence.segments],
    }


def _natural_label(occurrence: Occurrence | None, sentinel: str = SENTINEL_GAP) -> dict[str, Any]:
    if occurrence is None:
        return {"sentinel": sentinel}
    return {
        "phrase": occurrence.natural_language_phrase,
        "occurrence": _ordinal(occurrence.description_occurrence_index),
        "phrase_source": occurrence.phrase_source,
    }


def _memory_at(episode: Episode, frame: int, labeler: Any) -> dict[str, Any]:
    if frame < 0:
        return {"completed_primitive_occurrences": [], "active_primitive_occurrence": None}
    active = _choose_active(episode.primitives, frame)
    return {
        "completed_primitive_occurrences": [labeler(item) for item in episode.primitives if item.end <= frame],
        "active_primitive_occurrence": labeler(active) if active else None,
    }


def _next_occurrences(
    episode: Episode, primitive: Occurrence | None, skill: Occurrence | None
) -> tuple[Occurrence | str, Occurrence | str]:
    if primitive is None:
        return SENTINEL_NO_ACTIVE_PRIMITIVE, SENTINEL_GAP
    if skill is None:
        next_skill: Occurrence | str = SENTINEL_END_PRIMITIVE
    else:
        position = next((index for index, item in enumerate(primitive.skills) if item is skill), -1)
        next_skill = (
            primitive.skills[position + 1] if 0 <= position + 1 < len(primitive.skills) else SENTINEL_END_PRIMITIVE
        )
    next_position = primitive.occurrence_index + 1
    next_primitive: Occurrence | str = (
        episode.primitives[next_position] if next_position < len(episode.primitives) else SENTINEL_END_EPISODE
    )
    return next_skill, next_primitive


def _label_next(value: Occurrence | str, labeler: Any) -> dict[str, Any]:
    return labeler(value) if isinstance(value, Occurrence) else {"sentinel": value}


def _model_value(value: Any) -> Any:
    """Remove provenance-only keys before serialization into model text."""
    if isinstance(value, dict):
        return {key: _model_value(item) for key, item in value.items() if key != "phrase_source"}
    if isinstance(value, list):
        return [_model_value(item) for item in value]
    return value


def _tag(name: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"<{name}>{escape(payload)}</{name}>"


def structured_text(
    *,
    mode: str,
    memory_before: dict[str, Any],
    updated_memory: dict[str, Any],
    current_primitive: dict[str, Any],
    current_skill: dict[str, Any],
    next_skill: dict[str, Any],
    next_primitive: dict[str, Any],
) -> dict[str, str]:
    labels = (
        _tag("current_primitive", current_primitive)
        + _tag("current_skill", current_skill)
        + _tag("next_skill", next_skill)
        + _tag("next_primitive", next_primitive)
    )
    if mode == "state_target":
        prefix = _tag("memory_before", memory_before)
        target = _tag("updated_memory", updated_memory) + labels
    elif mode == "committed_memory":
        prefix = _tag("committed_memory", updated_memory)
        target = labels
    else:
        raise ValueError(f"unknown text mode: {mode}")
    return {"prefix": prefix, "target": target, "canonical": f"<prefix>{prefix}</prefix><target>{target}</target>"}


def build_record(episode: Episode, frame: int, *, text_modes: list[str], sample_kind: str) -> dict[str, Any]:
    primitive = _choose_active(episode.primitives, frame)
    skill = _choose_active(primitive.skills, frame) if primitive is not None else None
    if skill is None:
        skill = _choose_active(episode.skills, frame)
    next_skill_occurrence, next_primitive_occurrence = _next_occurrences(episode, primitive, skill)
    raw_memory_before = _memory_at(episode, frame - 1, _raw_label)
    raw_updated_memory = _memory_at(episode, frame, _raw_label)
    natural_memory_before = _memory_at(episode, frame - 1, _natural_label)
    natural_updated_memory = _memory_at(episode, frame, _natural_label)
    raw_fields = {
        "raw_memory_before": raw_memory_before,
        "raw_updated_memory": raw_updated_memory,
        "raw_current_primitive": _raw_label(primitive),
        "raw_current_skill": _raw_label(skill),
        "raw_next_skill": _label_next(next_skill_occurrence, _raw_label),
        "raw_next_primitive": _label_next(next_primitive_occurrence, _raw_label),
    }
    natural_fields = {
        "natural_language_memory_before": natural_memory_before,
        "natural_language_updated_memory": natural_updated_memory,
        "natural_language_current_primitive": _natural_label(primitive),
        "natural_language_current_skill": _natural_label(skill),
        "natural_language_next_skill": _label_next(next_skill_occurrence, _natural_label),
        "natural_language_next_primitive": _label_next(next_primitive_occurrence, _natural_label),
    }
    texts = {
        mode: structured_text(
            mode=mode,
            memory_before=_model_value(natural_memory_before),
            updated_memory=_model_value(natural_updated_memory),
            current_primitive=_model_value(natural_fields["natural_language_current_primitive"]),
            current_skill=_model_value(natural_fields["natural_language_current_skill"]),
            next_skill=_model_value(natural_fields["natural_language_next_skill"]),
            next_primitive=_model_value(natural_fields["natural_language_next_primitive"]),
        )
        for mode in text_modes
    }
    return {
        "task_id": episode.task_id,
        "episode_index": episode.episode_index,
        "task_name": episode.task_name,
        "task_instruction": episode.task_instruction,
        "frame_index": frame,
        "sample_kind": sample_kind,
        **raw_fields,
        **natural_fields,
        "structured_text": texts,
        "canonical_structured_text": texts[text_modes[0]]["canonical"]
        if len(text_modes) == 1
        else {mode: text["canonical"] for mode, text in texts.items()},
    }


def sample_frames(episode: Episode, sampling: str) -> list[int] | range:
    if sampling == "frame":
        return range(episode.duration)
    starts = {segment.start for item in [*episode.skills, *episode.primitives] for segment in item.segments}
    if sampling == "segment":
        return sorted(starts)
    ends = {segment.end for item in [*episode.skills, *episode.primitives] for segment in item.segments}
    return sorted(starts | ends)


def _parse_filter(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    output: set[int] = set()
    for raw_token in value.split(","):
        token = re.sub(r"^(?:task-|episode_)", "", raw_token.strip())
        if not token:
            continue
        if "-" in token:
            left, right = (int(item) for item in token.split("-", 1))
            output.update(range(min(left, right), max(left, right) + 1))
        else:
            output.add(int(token))
    return output


def discover_files(annotation_root: Path, tasks: set[int] | None, episodes: set[int] | None) -> Iterable[Path]:
    for task_dir in sorted(annotation_root.glob("task-*")):
        if not task_dir.is_dir():
            continue
        match = re.fullmatch(r"task-(\d+)", task_dir.name)
        if match is None or (tasks is not None and int(match.group(1)) not in tasks):
            continue
        for path in sorted(task_dir.glob("episode_*.json")):
            try:
                episode_index = _episode_index_from_path(path)
            except ValidationError:
                continue
            if episodes is None or episode_index in episodes:
                yield path


def _stats_payload(stats: Counter[str], task_ids: set[int], episodes: int, records: int) -> dict[str, Any]:
    keys = (
        "schema_errors",
        "coerced_durations",
        "skill_gaps",
        "skill_gap_frames",
        "skill_overlaps",
        "skill_overlap_frames",
        "primitive_gaps",
        "primitive_gap_frames",
        "primitive_overlaps",
        "primitive_overlap_frames",
        "missing_skill_object_binding",
        "missing_primitive_object_binding",
        "missing_skill_links",
        "unlinked_skills",
        "invalid_task_duration",
        "failed_episodes",
        "unresolved_object_handle_occurrences",
        "unresolved_phrase_template_occurrences",
        "object_fallbacks",
        "object_mapping_normalized_fallbacks",
        "phrase_fallbacks",
        "final_supervision_hits",
        "final_supervision_misses",
        "final_supervision_pollution_rejections",
        "canonical_polluted_records",
        "canonical_pollution_matches",
    )
    payload: dict[str, Any] = {"tasks": len(task_ids), "episodes": episodes, "records": records}
    payload.update({key: stats[key] for key in keys})
    payload["unresolved_object_handles"] = {
        key.removeprefix("unresolved_object_handle::"): value
        for key, value in sorted(stats.items())
        if key.startswith("unresolved_object_handle::")
    }
    payload["unresolved_phrase_templates"] = {
        key.removeprefix("unresolved_phrase_template::"): value
        for key, value in sorted(stats.items())
        if key.startswith("unresolved_phrase_template::")
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream BEHAVIOR-1K structured natural-language memory labels.")
    parser.add_argument("annotation_root", type=Path, help="annotations/ directory, or dataset root")
    parser.add_argument("--tasks", help="task ids/ranges, e.g. 0,3-5")
    parser.add_argument("--episodes", help="global episode ids/ranges, e.g. 10,20")
    parser.add_argument("--output", type=Path, help="output JSONL (required unless --preview or --dry-run)")
    parser.add_argument("--sampling", choices=("segment", "boundary", "frame"), default="segment")
    parser.add_argument("--text-mode", choices=("both", "state_target", "committed_memory"), default="both")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--object-mapping", type=Path)
    parser.add_argument("--phrase-templates", type=Path)
    parser.add_argument("--final-supervision", type=Path)
    parser.add_argument("--object-fallback", choices=("heuristic", "placeholder"), default="heuristic")
    parser.add_argument("--phrase-fallback", choices=("label", "placeholder"), default="label")
    parser.add_argument("--preview", type=int, default=0, metavar="N")
    parser.add_argument("--dry-run", action="store_true", help="validate all annotations and language metadata")
    parser.add_argument("--strict", action="store_true", help="reject schema/timeline/link/language resolution errors")
    parser.add_argument("--limit-episodes", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    annotation_root = (
        args.annotation_root / "annotations"
        if (args.annotation_root / "annotations").is_dir()
        else args.annotation_root
    )
    if not annotation_root.is_dir():
        raise SystemExit(f"annotation root does not exist: {annotation_root}")
    if not args.dry_run and args.preview <= 0 and args.output is None:
        raise SystemExit("--output is required unless --preview N or --dry-run is used")
    metadata = load_language_metadata(
        object_mapping_path=args.object_mapping or args.metadata_dir / "b1k_object_id_name_mapping.json",
        phrase_templates_path=args.phrase_templates or args.metadata_dir / "b1k_subtask_phrase_templates.json",
        final_supervision_path=args.final_supervision or args.metadata_dir / "b1k_subtask_supervision.json",
    )
    task_filter = _parse_filter(args.tasks)
    episode_filter = _parse_filter(args.episodes)
    instructions = load_task_instructions(annotation_root)
    text_modes = ["state_target", "committed_memory"] if args.text_mode == "both" else [args.text_mode]
    stats: Counter[str] = Counter()
    task_ids: set[int] = set()
    episode_count = record_count = 0
    output_stream = None
    try:
        if args.output is not None and not args.dry_run and args.preview <= 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_stream = args.output.open("w", encoding="utf-8")
        for path in discover_files(annotation_root, task_filter, episode_filter):
            if args.limit_episodes is not None and episode_count >= args.limit_episodes:
                break
            try:
                episode = load_episode(
                    path,
                    instructions=instructions,
                    stats=stats,
                    strict=args.strict,
                    language_metadata=metadata,
                    object_fallback=args.object_fallback,
                    phrase_fallback=args.phrase_fallback,
                )
            except ValidationError as error:
                stats["failed_episodes"] += 1
                if args.strict:
                    raise
                print(f"warning: {error}", file=sys.stderr)
                continue
            episode_count += 1
            task_ids.add(episode.task_id)
            if args.dry_run:
                continue
            for frame in sample_frames(episode, args.sampling):
                record = build_record(episode, frame, text_modes=text_modes, sample_kind=args.sampling)
                canonical = json.dumps(record["canonical_structured_text"], ensure_ascii=False)
                matches = re.findall(r"\b[a-z][a-z0-9_]*_(?:[a-z]{5,10}_)?\d+\b", canonical.casefold())
                if matches:
                    stats["canonical_polluted_records"] += 1
                    stats["canonical_pollution_matches"] += len(matches)
                line = json.dumps(record, ensure_ascii=False, sort_keys=True)
                if args.preview > 0:
                    print(line)
                elif output_stream is not None:
                    output_stream.write(line + "\n")
                record_count += 1
                if args.preview > 0 and record_count >= args.preview:
                    break
            if args.preview > 0 and record_count >= args.preview:
                break
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if output_stream is not None:
            output_stream.close()
    print(
        json.dumps(
            {"summary": _stats_payload(stats, task_ids, episode_count, record_count)},
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
