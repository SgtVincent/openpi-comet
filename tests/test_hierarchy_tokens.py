"""Tests for hierarchy special tokens (Hierarchical-MoMA-VLA P1, item 1).

Each test states what would make it fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.models import hierarchy_tokens as hier
from openpi.models.hierarchy_tokens import HIERARCHY_TAG_TO_SLOT
from openpi.models.hierarchy_tokens import HierarchyTagCodec
from openpi.models.hierarchy_tokens import HierarchyTagError
from openpi.models.hierarchy_tokens import build_hierarchy_text
from openpi.models.hierarchy_tokens import validate_tag_structure


@pytest.fixture(scope="module")
def sp_tokenizer():
    """The real PaliGemma SentencePiece tokenizer.

    Skips (rather than silently passing) when the tokenizer cannot be fetched,
    so a missing asset can never look like a green test.
    """
    sentencepiece = pytest.importorskip("sentencepiece")
    try:
        from openpi.shared import download

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        return sentencepiece.SentencePieceProcessor(model_proto=path.open("rb").read())
    except Exception as exc:
        pytest.skip(f"PaliGemma tokenizer unavailable: {exc}")


# ---------------------------------------------------------------------------
# Canonical text construction
# ---------------------------------------------------------------------------


def test_build_hierarchy_text_emits_all_four_pairs_in_order():
    """FAILS IF: a field is dropped, or the autoregressive order
    memory -> primitive -> skill -> next changes."""
    text = build_hierarchy_text(
        memory="finished: pick up the radio, 1st occurrence; active: press the radio, 1st occurrence",
        primitive="press the radio",
        skill="press the radio",
        next_skill="place the radio on the coffee table",
    )
    assert text.index("<MEM>") < text.index("<PRIM>") < text.index("<SKILL>") < text.index("<NEXT>")
    validate_tag_structure(text)


def test_build_hierarchy_text_collapses_inner_newlines():
    """FAILS IF: a newline inside a field survives and can be mistaken for a
    field separator (newline is the only separator between fields)."""
    text = build_hierarchy_text(memory="a\nb", primitive="c", skill="d", next_skill="e")
    assert "<MEM>a b</MEM>" in text
    assert text.count("\n") == 3  # exactly the three separators


def test_build_hierarchy_text_rejects_tag_characters_in_body():
    """FAILS IF: annotation content containing a tag can forge a field
    boundary (a text-injection route into the supervision target)."""
    with pytest.raises(HierarchyTagError, match="contains a hierarchy tag"):
        build_hierarchy_text(memory="x</MEM><PRIM>evil", primitive="c", skill="d", next_skill="e")


def test_build_hierarchy_text_rejects_none_field():
    """FAILS IF: a None field is silently rendered as the string 'None'."""
    with pytest.raises(HierarchyTagError, match="is None"):
        build_hierarchy_text(memory=None, primitive="c", skill="d", next_skill="e")


def test_validate_tag_structure_rejects_missing_and_misordered_tags():
    """FAILS IF: the diagnostic accepts truncated or out-of-order tag sequences."""
    with pytest.raises(HierarchyTagError):
        validate_tag_structure("<MEM>a</MEM><PRIM>b</PRIM><SKILL>c</SKILL>")  # missing <NEXT>
    with pytest.raises(HierarchyTagError):
        validate_tag_structure("<PRIM>b</PRIM><MEM>a</MEM><SKILL>c</SKILL><NEXT>d</NEXT>")  # swapped


# ---------------------------------------------------------------------------
# The reserved-slot decision, verified against the real tokenizer
# ---------------------------------------------------------------------------


def test_reserved_slots_exist_and_are_atomic(sp_tokenizer):
    """FAILS IF: a chosen <unusedN> slot is absent, or splits into several
    pieces. Either would silently reintroduce multi-token, driftable tags."""
    for tag, slot in HIERARCHY_TAG_TO_SLOT.items():
        slot_id = sp_tokenizer.piece_to_id(slot)
        assert slot_id > 0, f"{slot} (for {tag}) is not in the vocabulary"
        assert list(sp_tokenizer.encode(slot)) == [slot_id], f"{slot} does not encode to a single token"


def test_slot_assignment_is_unique():
    """FAILS IF: two tags are mapped to the same reserved slot, which would make
    them indistinguishable at the token level."""
    slots = list(HIERARCHY_TAG_TO_SLOT.values())
    assert len(slots) == len(set(slots))


def test_no_vocabulary_growth(sp_tokenizer):
    """FAILS IF: someone switches to added tokens and grows the vocabulary.

    Growing it resizes embed_tokens.weight and its tied lm_head.weight, which
    raises on EVERY checkpoint load path (safetensors always calls
    load_state_dict(strict=False), and torch raises the size-mismatch
    RuntimeError outside the `if strict:` block). It would also collide with
    gemma_pytorch.py:150's image_token_index=257152 sentinel.
    """
    assert sp_tokenizer.vocab_size() == 257152
    for slot in HIERARCHY_TAG_TO_SLOT.values():
        assert sp_tokenizer.piece_to_id(slot) < 257152


def test_reserved_slots_do_not_collide_with_fast_action_tokens(sp_tokenizer):
    """FAILS IF: a hierarchy slot id lands inside the FAST action-token band
    (vocab_size-1-128-2048 .. vocab_size-1-128), which would make a hierarchy
    tag alias an action token."""
    fast_skip = 128
    fast_hi = sp_tokenizer.vocab_size() - 1 - fast_skip
    fast_lo = fast_hi - 2048
    for tag, slot in HIERARCHY_TAG_TO_SLOT.items():
        slot_id = sp_tokenizer.piece_to_id(slot)
        assert not (fast_lo <= slot_id <= fast_hi), f"{tag}->{slot} id {slot_id} collides with FAST band"


# ---------------------------------------------------------------------------
# Round trip -- the explicitly required test
# ---------------------------------------------------------------------------


def test_special_token_round_trip_preserves_tag_boundaries_exactly(sp_tokenizer):
    """REQUIRED TEST. FAILS IF: encode->decode does not reproduce the tag
    boundaries byte-for-byte, i.e. if a tag is split, merged, dropped or
    re-spelled anywhere in the round trip."""
    codec = HierarchyTagCodec(tokenizer=sp_tokenizer)
    text = build_hierarchy_text(
        memory="finished: pick up the radio from the coffee table, 1st occurrence; "
        "active: press the radio, 1st occurrence",
        primitive="press the radio",
        skill="press the radio",
        next_skill="place the radio on the coffee table",
    )
    ids = codec.encode(text)
    decoded = codec.decode(ids)
    assert decoded == text, f"round trip changed the text:\n  in : {text!r}\n  out: {decoded!r}"

    # Each tag must occupy exactly one id, and appear exactly once.
    slot_ids = codec.slot_ids
    for tag, tag_id in slot_ids.items():
        assert ids.count(tag_id) == 1, f"tag {tag} appears {ids.count(tag_id)} times in the id stream"


def test_round_trip_is_stable_across_surrounding_context(sp_tokenizer):
    """FAILS IF: the tag's token id depends on what precedes it.

    This is the concrete drift that ordinary-text tags suffer: as raw text,
    '<MEM>' encodes as ['<','MEM','>'] but as ['_<','MEM','>'] after a space --
    a different leading id. A reserved slot must be invariant.
    """
    codec = HierarchyTagCodec(tokenizer=sp_tokenizer)
    mem_id = codec.slot_ids["<MEM>"]
    for prefix in ("", "x", " ", ";\n", "radio", "\n"):
        ids = sp_tokenizer.encode(codec.to_wire(prefix + "<MEM>"))
        assert ids[-1] == mem_id, f"tag id changed after prefix {prefix!r}: {ids}"


def test_raw_text_tags_would_drift_and_cost_more(sp_tokenizer):
    """Positive control for the decision itself.

    FAILS IF: the drift/cost problem we rejected option C over does not actually
    exist -- in which case the reserved-slot complexity is unjustified and this
    decision should be revisited rather than silently kept.
    """
    # Cost: 8 tags as raw text vs 8 single ids.
    raw_cost = sum(len(sp_tokenizer.encode(tag)) for tag in HIERARCHY_TAG_TO_SLOT)
    assert raw_cost > 8, "raw-text tags are not more expensive; option C's cost argument is void"
    # Drift: leading whitespace changes the first id of the raw-text tag.
    assert sp_tokenizer.encode("<MEM>")[0] != sp_tokenizer.encode(" <MEM>")[1 - 1], (
        "raw-text tag did not drift with context; option C's drift argument is void"
    )


def test_codec_rejects_a_tokenizer_without_atomic_slots():
    """FAILS IF: the codec accepts a tokenizer that lacks the reserved slots and
    would therefore encode tags as several ordinary pieces."""

    class _NoSlots:
        def unk_id(self):
            return 3

        def piece_to_id(self, piece):
            return 3  # everything is <unk>

        def encode(self, text):
            return [1, 2, 3]

    with pytest.raises(HierarchyTagError, match="atomic reserved slots"):
        HierarchyTagCodec(tokenizer=_NoSlots())


def test_to_wire_from_wire_are_inverse(sp_tokenizer):
    """FAILS IF: the readable<->wire mapping loses or aliases a tag."""
    codec = HierarchyTagCodec(tokenizer=sp_tokenizer)
    text = build_hierarchy_text(memory="m", primitive="p", skill="s", next_skill="n")
    assert codec.from_wire(codec.to_wire(text)) == text
    # And the wire form must contain no readable tags left over.
    wire = codec.to_wire(text)
    for tag in HIERARCHY_TAG_TO_SLOT:
        assert tag not in wire


def test_decode_accepts_numpy_ids(sp_tokenizer):
    """FAILS IF: decode breaks on the numpy arrays the data pipeline produces."""
    codec = HierarchyTagCodec(tokenizer=sp_tokenizer)
    text = build_hierarchy_text(memory="m", primitive="p", skill="s", next_skill="n")
    ids = np.asarray(codec.encode(text), dtype=np.int32)
    assert codec.decode(ids) == text


def test_module_exposes_stable_field_order():
    """FAILS IF: the field order constant drifts away from the design's
    factorization (task,obs,prev) -> memory -> primitive -> skill -> next."""
    assert [name for name, _, _ in hier.HIERARCHY_FIELD_ORDER] == [
        "memory",
        "primitive",
        "skill",
        "next_skill",
    ]
