"""Tests for splitting generated Memory text back into fields.

Design section 6.2's six Planner metrics all need the generated text split into
fields first, so a wrong split does not fail -- it produces metrics computed on
mangled fields. The specific hazard is real: 274 of 10,489 sampled rows (2.61%)
have a colon inside the Memory value, so a colon-based split cuts values in half.
"""

from __future__ import annotations

import json
import os
import random

import pytest

from openpi.models import memory_text as mt

FIELDS = dict(
    memory="No steps completed; currently picking up the radio, next step is to press it.",
    primitive="pick up the radio from the coffee table",
    skill="move to the radio",
    next_skill="pick up the radio from the coffee table",
    next_primitive="press the radio",
)


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------

def test_parse_is_the_inverse_of_build():
    assert mt.parse_memory_text(mt.build_memory_text(**FIELDS)) == FIELDS


def test_all_five_field_names_are_recovered_in_order():
    out = mt.parse_memory_text(mt.build_memory_text(**FIELDS))
    assert list(out) == list(mt.FIELD_NAMES)


def test_trailing_action_query_marker_is_tolerated_not_treated_as_a_field():
    text = mt.build_memory_text(**FIELDS) + "\n" + mt.ACTION_QUERY_MARKER
    assert mt.parse_memory_text(text) == FIELDS


# --------------------------------------------------------------------------
# the colon hazard -- this is why the split is on newline
# --------------------------------------------------------------------------

def test_internal_colon_in_the_memory_value_is_preserved():
    """Shape taken from the real corpus: `..., next: grab chocolate chip cookie.`"""
    f = dict(FIELDS, memory="Completed pick up the cookie; active place it, next: grab chocolate chip cookie.")
    out = mt.parse_memory_text(mt.build_memory_text(**f))
    assert out["memory"] == f["memory"]
    assert "next: grab chocolate chip cookie." in out["memory"]
    assert out["primitive"] == f["primitive"], "a colon in Memory must not shift later fields"


def test_a_colon_split_would_have_been_wrong_on_that_input():
    """Positive control for the hazard itself.

    If this ever stops holding, the colon-based split would be harmless and the
    newline requirement would be over-engineering -- so the test states the
    premise rather than assuming it.
    """
    f = dict(FIELDS, memory="Completed pick up the cookie; active place it, next: grab chocolate chip cookie.")
    text = mt.build_memory_text(**f)
    assert text.count(":") > len(mt.MEMORY_FIELDS), (
        "the corpus shape must contain more colons than labels, else the hazard is absent"
    )


def test_colons_in_every_field_still_parse():
    f = {k: f"{v} extra: tail" for k, v in FIELDS.items()}
    assert mt.parse_memory_text(mt.build_memory_text(**f)) == f


# --------------------------------------------------------------------------
# longest-first and case sensitivity
# --------------------------------------------------------------------------

def test_next_skill_is_not_consumed_as_skill():
    out = mt.parse_memory_text(mt.build_memory_text(**FIELDS))
    assert out["skill"] == FIELDS["skill"]
    assert out["next_skill"] == FIELDS["next_skill"]
    assert out["next_primitive"] == FIELDS["next_primitive"]


def test_matching_is_case_sensitive():
    """Bare lower-case `skill:` does not occur in the corpus (0 of 10,489), so a
    case-insensitive match would only add ways to be wrong."""
    text = mt.build_memory_text(**FIELDS).replace("Skill:", "skill:", 1)
    with pytest.raises(mt.MemoryTextError, match="should start with"):
        mt.parse_memory_text(text)


# --------------------------------------------------------------------------
# strict mode refuses partial results
# --------------------------------------------------------------------------

@pytest.mark.parametrize("drop", [0, 1, 2, 3, 4])
def test_a_missing_field_raises_rather_than_returning_a_partial_parse(drop):
    lines = mt.build_memory_text(**FIELDS).split("\n")
    del lines[drop]
    with pytest.raises(mt.MemoryTextError, match="expected 5 newline-separated fields"):
        mt.parse_memory_text("\n".join(lines))


def test_an_empty_field_value_raises():
    text = mt.build_memory_text(**FIELDS).replace(FIELDS["next_skill"], "", 1)
    with pytest.raises(mt.MemoryTextError):
        mt.parse_memory_text(text)


def test_flattened_text_raises_in_strict_mode():
    """tokenize_subtask flattens newlines; the production path (tokenize_memory)
    does not. If a flattened generation reaches the parser, strict mode must say
    so rather than guess."""
    flat = mt.build_memory_text(**FIELDS).replace("\n", " ")
    with pytest.raises(mt.MemoryTextError):
        mt.parse_memory_text(flat)


def test_none_raises():
    with pytest.raises(mt.MemoryTextError, match="requires text"):
        mt.parse_memory_text(None)


# --------------------------------------------------------------------------
# the label fallback is opt-in and only for the diagnostic
# --------------------------------------------------------------------------

def test_label_fallback_recovers_flattened_text_when_not_strict():
    flat = mt.build_memory_text(**FIELDS).replace("\n", " ")
    out = mt.parse_memory_text(flat, strict=False)
    assert out["skill"] == FIELDS["skill"]
    assert out["next_primitive"] == FIELDS["next_primitive"]


def test_format_completion_reports_instead_of_raising():
    ok = mt.format_completion(mt.build_memory_text(**FIELDS))
    assert ok["strict_parse_ok"] is True
    assert ok["newline_field_count"] == 5 == ok["expected_field_count"]
    assert ok["labels_present"] == 5
    assert ok["labels_in_expected_order"] is True
    assert ok["line_prefixes_matching"] == 5

    lines = mt.build_memory_text(**FIELDS).split("\n")
    bad = mt.format_completion("\n".join(lines[:3]))          # truncated generation
    assert bad["strict_parse_ok"] is False, "a truncated generation must be reported as such"
    assert bad["newline_field_count"] == 3
    # and it must not raise -- that is the whole point of the diagnostic
    assert mt.format_completion("")["strict_parse_ok"] is False
    assert mt.format_completion(None)["newline_field_count"] == 0


# --------------------------------------------------------------------------
# real corpus
# --------------------------------------------------------------------------

DATA_ROOT = ("/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos"
             "/derived/fixed_compact_memory_annotations")


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT), reason="dataset not mounted")
def test_real_model_target_text_parses_and_colon_rows_are_exercised():
    """Cross-offset sample: one episode per task directory, since these files are
    task/episode ordered and a head sample would draw from a single task."""
    tasks = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith("task-"))
    rng = random.Random(20260907)
    parsed = 0
    colon_rows = 0
    for t in tasks:
        eps = sorted(f for f in os.listdir(os.path.join(DATA_ROOT, t)) if f.endswith(".json"))
        with open(os.path.join(DATA_ROOT, t, rng.choice(eps))) as fh:
            episode = json.load(fh)
        for row in episode["memory_annotation"]:
            out = mt.parse_memory_text(row["model_target_text"])
            assert out["memory"] == row["fixed_compact_memory"]
            assert out["primitive"] == row["current_primitive"]
            assert out["next_skill"] == row["next_skill"]
            assert out["next_primitive"] == row["next_primitive"]
            if ":" in out["memory"]:
                colon_rows += 1
            parsed += 1
    assert parsed > 500, f"sample too small to be meaningful: {parsed} rows"
    # the hazard must actually be present in the sample, else the colon tests above
    # are guarding something this corpus never produces
    assert colon_rows > 0, "no Memory value with an internal colon in the sample"
