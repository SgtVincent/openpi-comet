"""Tests for memory field text and the label schemes (MoMA-VLA P1, item 1).

Each test states what would make it fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.models import memory_text as mem
from openpi.models.memory_text import (
    CURRENT_MEMORY_CUE,
    FIELD_NAMES,
    MEMORY_FIELDS,
    LabelScheme,
    MemoryTextCodec,
    MemoryTextError,
    build_memory_text,
    validate_field_structure,
)

# A representative sample using the notation the user's actual data uses: "(1)".
SAMPLE = {
    "memory": "Completed: pick up the radio from the coffee table (1). Active: press the radio (1).",
    "primitive": "press the radio (1)",
    "skill": "press the radio",
    "next_skill": "END_OF_PRIMITIVE",
    "next_primitive": "place the radio on the coffee table (1)",
}


@pytest.fixture(scope="module")
def sp_tokenizer():
    """The real PaliGemma SentencePiece tokenizer.

    Skips (rather than silently passing) when the asset cannot be fetched, so a
    missing tokenizer can never look like a green test.
    """
    sentencepiece = pytest.importorskip("sentencepiece")
    try:
        from openpi.shared import download

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        return sentencepiece.SentencePieceProcessor(model_proto=path.open("rb").read())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PaliGemma tokenizer unavailable: {exc}")


# ---------------------------------------------------------------------------
# Field set and order
# ---------------------------------------------------------------------------


def test_five_fields_in_the_designs_autoregressive_order():
    """FAILS IF: a field is dropped or reordered.

    The order is the generation order of design section 2.2 and is part of the
    checkpoint contract -- reordering silently changes what a trained model's
    tokens mean. `next_primitive` is the fifth field added in doc revision 8.
    """
    assert FIELD_NAMES == ("memory", "primitive", "skill", "next_skill", "next_primitive")


def test_labels_match_the_design_text():
    """FAILS IF: a label string drifts from what the design (and the training
    data) writes. These strings are the supervision target."""
    assert [f.label for f in MEMORY_FIELDS] == [
        "Memory:",
        "Primitive:",
        "Skill:",
        "Next skill:",
        "Next primitive:",
    ]


def test_build_memory_text_emits_labels_in_order():
    """FAILS IF: the emitted order stops matching the declared field order."""
    text = build_memory_text(**SAMPLE)
    positions = [text.index(f.label) for f in MEMORY_FIELDS]
    assert positions == sorted(positions)
    validate_field_structure(text)


def test_next_labels_are_not_shadowed_by_shorter_ones():
    """FAILS IF: 'Next skill:' is matched as 'Skill:' (or 'Next primitive:' as
    'Primitive:'). Longest-first matching is what prevents it; a naive
    alternation would mis-split every sample."""
    text = build_memory_text(**SAMPLE)
    found = mem._LABEL_RE.findall(text)  # noqa: SLF001
    assert found == [f.label for f in MEMORY_FIELDS]


def test_build_memory_text_requires_exactly_the_five_fields():
    """FAILS IF: a missing or misspelled field is silently accepted, which would
    emit a label with no body and train the model on a blank field."""
    partial = dict(SAMPLE)
    partial.pop("next_primitive")
    with pytest.raises(MemoryTextError, match="missing="):
        build_memory_text(**partial)
    with pytest.raises(MemoryTextError, match="unexpected="):
        build_memory_text(**SAMPLE, bogus="x")


def test_build_memory_text_collapses_inner_newlines():
    """FAILS IF: a newline inside a field survives and can be mistaken for a
    field separator -- newline is the ONLY separator between fields."""
    text = build_memory_text(**{**SAMPLE, "memory": "a\nb"})
    assert "Memory: a b" in text
    assert text.count("\n") == 4  # exactly the four separators between five fields


def test_build_memory_text_rejects_label_text_in_a_body():
    """FAILS IF: annotation content containing a label can forge a field
    boundary -- a text-injection route into the supervision target."""
    with pytest.raises(MemoryTextError, match="contains a field label"):
        build_memory_text(**{**SAMPLE, "skill": "press\nNext skill: evil"})


def test_build_memory_text_rejects_none_field():
    """FAILS IF: a None field is silently rendered as the string 'None'."""
    with pytest.raises(MemoryTextError, match="is None"):
        build_memory_text(**{**SAMPLE, "skill": None})


def test_validate_field_structure_rejects_missing_and_misordered():
    """FAILS IF: the diagnostic accepts truncated or out-of-order label sequences.

    Truncation matters specifically because it drops the LAST fields, which reads
    as 'the model is bad at next-skill' rather than as a format bug.
    """
    with pytest.raises(MemoryTextError):
        validate_field_structure("Memory: a\nPrimitive: b\nSkill: c\nNext skill: d")  # no Next primitive
    with pytest.raises(MemoryTextError):
        validate_field_structure("Primitive: b\nMemory: a\nSkill: c\nNext skill: d\nNext primitive: e")


# ---------------------------------------------------------------------------
# Both label schemes must work -- the choice is still open upstream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", list(LabelScheme))
def test_round_trip_preserves_field_boundaries_exactly(sp_tokenizer, scheme):
    """REQUIRED TEST, for BOTH schemes.

    FAILS IF: encode->decode does not reproduce the text byte-for-byte, i.e. a
    label is split, merged, dropped or re-spelled anywhere in the round trip.

    Parametrised over both schemes on purpose: the plain-text vs reserved-slot
    decision is still open, so neither may be allowed to rot.
    """
    codec = MemoryTextCodec(tokenizer=sp_tokenizer, scheme=scheme)
    text = build_memory_text(**SAMPLE)
    decoded = codec.decode(codec.encode(text))
    assert decoded == text, f"[{scheme}] round trip changed the text:\n in : {text!r}\n out: {decoded!r}"


def test_default_scheme_is_the_designs_plain_text():
    """FAILS IF: the default stops being what the design specifies.

    The doc is explicit that P1 uses the existing SentencePiece vocabulary with
    ordinary text labels, so plain text must be what a caller gets by default.
    """
    assert mem.DEFAULT_LABEL_SCHEME is LabelScheme.PLAIN_TEXT
    assert MemoryTextCodec(tokenizer=_FakeSp()).scheme is LabelScheme.PLAIN_TEXT


def test_plain_text_scheme_adds_no_new_vocabulary(sp_tokenizer):
    """FAILS IF: the plain-text scheme somehow needs an id outside the existing
    vocabulary -- the whole point is no resize and no checkpoint break."""
    codec = MemoryTextCodec(tokenizer=sp_tokenizer, scheme=LabelScheme.PLAIN_TEXT)
    ids = codec.encode(build_memory_text(**SAMPLE))
    assert ids, "encoded to nothing"
    assert max(ids) < sp_tokenizer.vocab_size()


def test_reserved_slot_scheme_is_atomic_and_cheaper(sp_tokenizer):
    """FAILS IF: reserved slots stop being one id each, or stop being cheaper
    than plain text.

    This is the measured trade-off behind the open decision. If it ever stops
    holding, the decision should be revisited rather than silently kept.
    """
    slot = MemoryTextCodec(tokenizer=sp_tokenizer, scheme=LabelScheme.RESERVED_SLOT)
    plain = MemoryTextCodec(tokenizer=sp_tokenizer, scheme=LabelScheme.PLAIN_TEXT)
    per_label = slot.label_token_ids
    assert all(len(v) == 1 for v in per_label.values()), f"not atomic: {per_label}"
    assert slot.label_token_cost() == len(MEMORY_FIELDS)
    assert slot.label_token_cost() < plain.label_token_cost()


def test_plain_text_labels_drift_with_context_but_slots_do_not(sp_tokenizer):
    """FAILS IF: the drift that motivates the reserved-slot option does not exist.

    Then the reserved-slot complexity is unjustified and this should be
    reconsidered -- so the test is written to fail in that direction too.
    """
    first_id_at_start = sp_tokenizer.encode("Memory:")[0]
    first_id_after_nl = sp_tokenizer.encode("x\nMemory:")[1]
    assert first_id_at_start != first_id_after_nl, "plain-text label did not drift; slot scheme is unmotivated"

    slot_codec = MemoryTextCodec(tokenizer=sp_tokenizer, scheme=LabelScheme.RESERVED_SLOT)
    slot_id = slot_codec.label_token_ids["memory"][0]
    for prefix in ("", "x", " ", ";\n", "radio"):
        wire = slot_codec.to_wire(prefix + "Memory:")
        assert sp_tokenizer.encode(wire)[-1] == slot_id, f"slot id changed after prefix {prefix!r}"


def test_reserved_slot_codec_rejects_a_tokenizer_without_atomic_slots():
    """FAILS IF: the reserved-slot codec accepts a tokenizer lacking the slots and
    would therefore silently encode labels as several ordinary pieces."""
    with pytest.raises(MemoryTextError, match="atomic reserved slots"):
        MemoryTextCodec(tokenizer=_FakeSp(), scheme=LabelScheme.RESERVED_SLOT)


def test_slot_assignment_is_unique():
    """FAILS IF: two fields share a reserved slot, making them indistinguishable
    at the token level."""
    slots = [f.slot for f in MEMORY_FIELDS]
    assert len(slots) == len(set(slots))


def test_reserved_slots_do_not_collide_with_fast_action_tokens(sp_tokenizer):
    """FAILS IF: a slot id lands in the FAST action-token band, which would make
    a memory label alias an action token."""
    fast_hi = sp_tokenizer.vocab_size() - 1 - 128
    fast_lo = fast_hi - 2048
    for field in MEMORY_FIELDS:
        slot_id = sp_tokenizer.piece_to_id(field.slot)
        assert not (fast_lo <= slot_id <= fast_hi), f"{field.name}->{field.slot} id {slot_id} collides"


def test_memory_labels_are_the_only_reserved_slot_protocol():
    """The old paired-tag protocol must not coexist with the five Memory labels.

    Both protocols assigned different meanings to ``<unused0>`` through
    ``<unused4>``. Keeping both would make one checkpoint token ambiguous even
    though every id stays inside the unchanged vocabulary.
    """
    import importlib.util

    assert importlib.util.find_spec("openpi.models.hierarchy_tokens") is None
    assert [field.slot for field in MEMORY_FIELDS] == [
        "<unused0>",
        "<unused1>",
        "<unused2>",
        "<unused3>",
        "<unused4>",
    ]


def test_no_vocabulary_growth(sp_tokenizer):
    """FAILS IF: someone grows the vocabulary.

    A resize changes embed_tokens.weight and its tied lm_head.weight, which
    raises on EVERY checkpoint load path -- strict=False does not help, because
    safetensors always calls load_state_dict(strict=False) and torch raises the
    size-mismatch RuntimeError outside the `if strict:` block. It would also turn
    gemma_pytorch.py:150's image_token_index=257152 sentinel into a real row.
    """
    assert sp_tokenizer.vocab_size() == 257152


def test_section_cues_are_defined_in_one_place():
    """FAILS IF: the 'Current memory:' / 'Previous memory:' cues get spelled by
    callers instead of taken from here, which is how two spellings diverge."""
    # Lower-case per the design document; the upper-case form shared two token
    # ids with the target's own "Memory:" label.
    assert CURRENT_MEMORY_CUE == "Current memory:"
    assert mem.PREVIOUS_MEMORY_CUE == "Previous memory:"


def test_decode_accepts_numpy_ids(sp_tokenizer):
    """FAILS IF: decode breaks on the numpy arrays the data pipeline produces."""
    codec = MemoryTextCodec(tokenizer=sp_tokenizer)
    text = build_memory_text(**SAMPLE)
    ids = np.asarray(codec.encode(text), dtype=np.int32)
    assert codec.decode(ids) == text


class _FakeSp:
    """Tokenizer stub with no reserved slots, for the negative cases above."""

    def unk_id(self):
        return 3

    def piece_to_id(self, piece):  # noqa: ARG002
        return 3

    def encode(self, text):  # noqa: ARG002
        return [1, 2, 3]

    def decode(self, ids):  # noqa: ARG002
        return ""
