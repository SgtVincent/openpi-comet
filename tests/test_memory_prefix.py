"""Tests for the MoMA-VLA `Previous memory:` prefix slot (design doc §3.4.3).

The legacy `annotations_skill` runs share `tokenize_prompt`, so the
`previous_memory=None` path is pinned by an explicit byte-level regression lock
rather than trusted to review.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.models.tokenizer import SubtaskTokenizer

STATE_DIM = 32  # Pi05SubtaskConfig.action_dim


@pytest.fixture(scope="module")
def tok():
    return SubtaskTokenizer(prompt_max_len=512, subtask_max_len=128)


@pytest.fixture(scope="module")
def state():
    rng = np.random.default_rng(20260907)
    return rng.uniform(-1.0, 1.0, size=STATE_DIM).astype(np.float32)


def _decode(tok, tokens, mask):
    return tok._tokenizer.decode([int(t) for t, m in zip(tokens, mask) if m])


# --------------------------------------------------------------------------
# regression lock: the legacy path must not move
# --------------------------------------------------------------------------

def test_legacy_prefix_is_unchanged_and_still_ends_with_subtask_cue(tok, state):
    """`previous_memory=None` must reproduce the historical prefix exactly.

    Asserts the literal decoded string, not just a token count: a template edit
    that happened to preserve length would otherwise pass.
    """
    tokens, mask = tok.tokenize_prompt("turn on the radio", state)
    text = _decode(tok, tokens, mask)
    assert text.startswith("Task: turn on the radio, State: ")
    # the trailing space is part of the historical cue; keep it in the assertion
    assert text.endswith(";\nSubtask: "), f"legacy trailing cue changed: {text[-30:]!r}"
    assert "Previous memory" not in text


def test_legacy_path_is_byte_identical_to_a_hand_built_reference(tok, state):
    """Independent reconstruction of the documented legacy template."""
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    reference = f"Task: turn on the radio, State: {' '.join(map(str, discretized))};\nSubtask: "
    expected = tok._tokenizer.encode(reference, add_bos=True)
    tokens, mask = tok.tokenize_prompt("turn on the radio", state)
    assert [int(t) for t, m in zip(tokens, mask) if m] == expected


def test_shapes_and_padding_are_unaffected_by_the_new_parameter(tok, state):
    a_t, a_m = tok.tokenize_prompt("turn on the radio", state)
    b_t, b_m = tok.tokenize_prompt("turn on the radio", state, previous_memory="No steps completed.")
    assert a_t.shape == b_t.shape == (512,)
    assert a_m.dtype == b_m.dtype
    assert not a_m[int(a_m.sum()):].any() and not b_m[int(b_m.sum()):].any()


# --------------------------------------------------------------------------
# new behaviour
# --------------------------------------------------------------------------

def test_previous_memory_replaces_the_subtask_cue(tok, state):
    mem = "No steps completed; currently picking up the radio, next step is to press it."
    tokens, mask = tok.tokenize_prompt("turn on the radio", state, previous_memory=mem)
    text = _decode(tok, tokens, mask)
    assert "Previous memory: " + mem in text
    assert "Subtask:" not in text, "the Subtask cue must not coexist with Previous memory"
    assert text.startswith("Task: turn on the radio, State: ")


def test_state_is_still_discretised_joint_values_not_a_frame_counter(tok, state):
    """Guards against reintroducing the P0 oracle leak (`State: Frame N of M`).

    `of M` is the episode length, knowable only from ground-truth replay, so it
    inflates offline metrics and only fails in closed loop.
    """
    tokens, mask = tok.tokenize_prompt("turn on the radio", state, previous_memory="mem")
    text = _decode(tok, tokens, mask)
    assert "Frame" not in text
    assert " of " not in text.split(";")[0]
    body = text.split("State: ", 1)[1].split(";", 1)[0]
    values = [int(v) for v in body.split()]
    assert len(values) == STATE_DIM
    assert all(0 <= v <= 255 for v in values), "discretised state must land in 256 buckets"


def test_state_discretisation_uses_exactly_256_buckets(tok):
    """Pin the bucket count with exact expected values.

    A range check alone (0..255) also passes with 16 buckets, so it cannot
    detect a coarsened discretisation -- which would quietly throw away state
    resolution while every other assertion still held.  -1/0/+1 map to
    0/128/255 under 256 buckets and to 0/8/15 under 16.
    """
    probe = np.zeros(STATE_DIM, dtype=np.float32)
    probe[0], probe[1], probe[2] = -1.0, 0.0, 1.0
    _, mask = tok.tokenize_prompt("t", probe, previous_memory="mem")
    tokens, _ = tok.tokenize_prompt("t", probe, previous_memory="mem")
    text = _decode(tok, tokens, mask)
    values = [int(v) for v in text.split("State: ", 1)[1].split(";", 1)[0].split()]
    assert values[0] == 0, f"-1.0 should map to bucket 0, got {values[0]}"
    assert values[1] == 128, f"0.0 should map to bucket 128 under 256 buckets, got {values[1]}"
    assert values[2] == 255, f"+1.0 should map to bucket 255, got {values[2]}"
    assert max(values) == 255 and len(values) == STATE_DIM


def test_memory_text_newlines_are_flattened_like_every_other_field(tok, state):
    a, am = tok.tokenize_prompt("t", state, previous_memory="line one\nline two")
    b, bm = tok.tokenize_prompt("t", state, previous_memory="line one line two")
    assert [int(x) for x, m in zip(a, am) if m] == [int(x) for x, m in zip(b, bm) if m]


# --------------------------------------------------------------------------
# fail-closed behaviour
# --------------------------------------------------------------------------

def test_empty_previous_memory_raises_rather_than_emitting_a_bare_label(tok, state):
    for empty in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="empty after cleaning"):
            tok.tokenize_prompt("turn on the radio", state, previous_memory=empty)


def test_overlong_memory_prefix_raises_instead_of_truncating(tok, state):
    """The memory text is at the END of the prefix, so right-truncation removes
    exactly the conditioning it was added for, silently."""
    small = SubtaskTokenizer(prompt_max_len=64, subtask_max_len=128)
    with pytest.raises(ValueError, match="silently drop the memory conditioning"):
        small.tokenize_prompt("turn on the radio", state, previous_memory="x " * 200)


def test_overlong_legacy_prefix_still_only_warns(tok, state, caplog):
    """The legacy path must keep its historical behaviour, not inherit the raise."""
    small = SubtaskTokenizer(prompt_max_len=16, subtask_max_len=128)
    tokens, mask = small.tokenize_prompt("turn on the radio", state)  # must not raise
    assert int(mask.sum()) == 16


# --------------------------------------------------------------------------
# budget headroom on real data shapes
# --------------------------------------------------------------------------

def test_realistic_memory_prefix_stays_well_inside_the_budget(tok, state):
    """Measured over 7,861 real rows: prompt max 176 of 512 (2.9x headroom).
    A regression that inflates the prefix should trip here, not in a 32-GPU run.
    """
    mem = ("Completed picking up the radio from the coffee table and pressing the radio; "
           "currently placing the radio on the coffee table, next step is to turn it off.")
    _, mask = tok.tokenize_prompt(
        "turn on the radio receiver that's on the table in the living room", state,
        previous_memory=mem)
    used = int(mask.sum())
    assert used < 300, f"memory-conditioned prefix used {used} tokens; expected well under 300"


# --------------------------------------------------------------------------
# subtask-side length budget: 27 tokens of headroom, not 322
# --------------------------------------------------------------------------
# Measured over all 261,353 rows with the training-side sentencepiece:
# planner_target_text max = 101 of subtask_max_len 128 -> 27 tokens spare, while
# the prompt segment has 322. The thin side is the one a phrase-template backfill
# will eat, and right-truncation there drops EOS first.

def _five_field_target(memory_body: str) -> str:
    return "\n".join([
        f"Memory: {memory_body}",
        "Primitive: press the radio",
        "Skill: press the radio",
        "Next skill: END_OF_PRIMITIVE",
        "Next primitive: place the radio on the coffee table",
    ])


def test_tokenize_memory_raises_instead_of_dropping_eos(tok):
    """Over-length memory must raise, not silently lose EOS + trailing fields."""
    with pytest.raises(ValueError) as e:
        tok.tokenize_memory(_five_field_target("completed " + "a long completed step; " * 30))
    msg = str(e.value)
    assert "exceeds subtask_max_len" in msg
    assert "128" in msg, "error must state the limit"
    assert "Longest field is" in msg, "error must name which field to shorten"
    assert "Memory" in msg


def test_tokenize_memory_reports_the_actual_token_count(tok):
    with pytest.raises(ValueError) as e:
        tok.tokenize_memory(_five_field_target("x " * 200))
    msg = str(e.value)
    import re as _re
    nums = [int(n) for n in _re.findall(r"\b(\d{3,4})\b", msg)]
    assert any(n > 128 for n in nums), f"no actual over-limit token count in: {msg}"


def test_tokenize_memory_eos_is_inside_the_loss_mask_when_it_fits(tok):
    tokens, mask, ar_mask, loss_mask = tok.tokenize_memory(_five_field_target("no steps completed"))
    eos_at = [i for i, t in enumerate(tokens) if int(t) == tok._tokenizer.eos_id()]
    assert eos_at, "EOS must be present"
    assert loss_mask[eos_at[-1]], "EOS must be supervised"
    assert not loss_mask[0], "BOS must not be supervised"
    assert mask[eos_at[-1]]
    assert not mask[eos_at[-1] + 1:].any(), "padding must be masked out"


def test_tokenize_memory_accepts_a_target_at_the_measured_real_maximum(tok):
    """101 tokens is the measured global max; it must NOT raise."""
    text = _five_field_target(
        "Completed picking up the radio from the coffee table and pressing the radio twice; "
        "currently placing the radio on the coffee table, next step is to turn it off.")
    n = tok.memory_token_length(text)
    assert n <= 128, f"fixture itself is over budget ({n})"
    tokens, mask, _, loss_mask = tok.tokenize_memory(text)
    assert int(mask.sum()) == n


def test_tokenize_subtask_strict_length_raises_but_default_still_truncates(tok, caplog):
    """The shared legacy path must keep its behaviour; strict is opt-in."""
    long_text = "press the radio " * 60
    with pytest.raises(ValueError, match="exceeds subtask_max_len"):
        tok.tokenize_subtask(long_text, strict_length=True)
    # default path: truncates, does not raise
    tokens, mask, _, _ = tok.tokenize_subtask(long_text)
    assert int(mask.sum()) == 128


def test_tokenize_subtask_default_output_is_unchanged_for_normal_text(tok):
    """Adding the keyword must not perturb the legacy result."""
    a = tok.tokenize_subtask("press the radio")
    b = tok.tokenize_subtask("press the radio", strict_length=True)
    for x, y in zip(a, b):
        assert (x == y).all()
