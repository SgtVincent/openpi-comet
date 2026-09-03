"""Tests for the fail-closed hierarchy path through the data transforms
(Hierarchical-MoMA-VLA P1, item 2).

The defect these guard against is a three-stage silent drop that exists today:

  1. ``transforms.PromptFromLeRobotItem`` pops ``subtask_text`` when
     ``include_subtask_text=False``;
  2. ``transforms.TokenizeSubtaskInputs`` then fabricates an all-zero,
     mask-False subtask;
  3. ``SubtaskActionExpert.encode_prefix`` short-circuits on
     ``not torch.any(subtask_mask)`` and drops the segment from the prefix
     entirely.

Nothing raises, nothing warns, and the run silently becomes unconditioned while
still reporting a loss. Hierarchy tokens travel the same path.

Each test states what would make it fail.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi import transforms as _transforms
from openpi.models import hierarchy_tokens as hier


@pytest.fixture(scope="module")
def subtask_tokenizer():
    try:
        from openpi.models import tokenizer as _tokenizer

        return _tokenizer.SubtaskTokenizer(prompt_max_len=128, subtask_max_len=96)
    except Exception as exc:
        pytest.skip(f"SubtaskTokenizer unavailable: {exc}")


HIER_TEXT = hier.build_hierarchy_text(
    memory="finished: pick up the radio, 1st occurrence; active: press the radio, 1st occurrence",
    primitive="press the radio",
    skill="press the radio",
    next_skill="place the radio on the coffee table",
)


# ---------------------------------------------------------------------------
# Stage 1: PromptFromLeRobotItem
# ---------------------------------------------------------------------------


def test_include_subtask_text_false_still_drops_subtask_text():
    """Characterisation of existing behaviour, kept deliberately.

    FAILS IF: the legacy default changes. We are adding a fail-closed path for
    hierarchy, not silently altering what existing subtask configs do.
    """
    tf = _transforms.PromptFromLeRobotItem()
    out = tf({"task": "t", "subtask_text": "press the radio"})
    assert "subtask_text" not in out


def test_hierarchy_text_is_dropped_by_default_but_kept_when_included():
    """FAILS IF: hierarchy_text leaks through by default (surprising existing
    configs), or cannot be kept when explicitly requested."""
    dropped = _transforms.PromptFromLeRobotItem()({"task": "t", "hierarchy_text": HIER_TEXT})
    assert "hierarchy_text" not in dropped

    kept = _transforms.PromptFromLeRobotItem(include_hierarchy_text=True)(
        {"task": "t", "hierarchy_text": HIER_TEXT}
    )
    assert kept["hierarchy_text"] == HIER_TEXT


def test_require_hierarchy_text_rejects_the_contradictory_config():
    """REQUIRED TEST (part 1). FAILS IF: a config that requires hierarchy while
    also dropping it is accepted -- i.e. the exact silent-drop shape survives
    behind a flag that claims to prevent it."""
    tf = _transforms.PromptFromLeRobotItem(require_hierarchy_text=True, include_hierarchy_text=False)
    with pytest.raises(ValueError, match="would drop the field it is required to keep"):
        tf({"task": "t", "hierarchy_text": HIER_TEXT})


def test_require_hierarchy_text_raises_when_the_field_is_absent():
    """REQUIRED TEST (part 2). FAILS IF: a missing hierarchy_text passes through
    quietly, which downstream becomes an unconditioned model with no error."""
    tf = _transforms.PromptFromLeRobotItem(require_hierarchy_text=True, include_hierarchy_text=True)
    with pytest.raises(ValueError, match="hierarchy_text is required but missing"):
        tf({"task": "t"})


# ---------------------------------------------------------------------------
# Stage 2: TokenizeSubtaskInputs
# ---------------------------------------------------------------------------


def test_tokenize_uses_hierarchy_text_when_present(subtask_tokenizer):
    """FAILS IF: hierarchy_text is ignored, or routed anywhere other than the
    subtask_* slots (which would make it a third conditioning channel)."""
    tf = _transforms.TokenizeSubtaskInputs(tokenizer=subtask_tokenizer)
    out = tf({"prompt": "turn on the radio", "state": np.zeros(8), "hierarchy_text": HIER_TEXT})
    assert out["subtask_mask"].any(), "hierarchy produced an empty (dropped) segment"
    mem_id = subtask_tokenizer.hierarchy_codec.slot_ids["<MEM>"]
    assert mem_id in out["subtask_tokens"].tolist(), "hierarchy tags absent from the token stream"


def test_require_hierarchy_raises_instead_of_fabricating_zeros(subtask_tokenizer):
    """REQUIRED TEST (part 3). FAILS IF: the all-zero / mask-False fabrication
    happens when hierarchy is required.

    Asserting the raise rather than asserting a zero tensor is the point: a zero
    tensor is precisely the silent failure, so a test that accepted it would
    lock in the bug.
    """
    tf = _transforms.TokenizeSubtaskInputs(tokenizer=subtask_tokenizer, require_hierarchy=True)
    with pytest.raises(ValueError, match="no 'hierarchy_text'"):
        tf({"prompt": "turn on the radio", "state": np.zeros(8)})


def test_legacy_zero_fabrication_still_happens_when_not_required(subtask_tokenizer):
    """Characterisation. FAILS IF: the fail-closed switch changed the default and
    broke existing subtask configs. It documents the silent path rather than
    endorsing it."""
    tf = _transforms.TokenizeSubtaskInputs(tokenizer=subtask_tokenizer)
    out = tf({"prompt": "turn on the radio", "state": np.zeros(8)})
    assert not out["subtask_mask"].any()


def test_subtask_text_still_works_unchanged(subtask_tokenizer):
    """FAILS IF: adding the hierarchy branch changed how plain subtask text is
    tokenized (which would invalidate every existing subtask checkpoint)."""
    tf = _transforms.TokenizeSubtaskInputs(tokenizer=subtask_tokenizer)
    out = tf({"prompt": "turn on the radio", "state": np.zeros(8), "subtask_text": "press the radio"})
    direct = subtask_tokenizer.tokenize_subtask("press the radio")
    np.testing.assert_array_equal(out["subtask_tokens"], direct[0])
    np.testing.assert_array_equal(out["subtask_mask"], direct[1])


# ---------------------------------------------------------------------------
# Stage 3: the short-circuit that makes the drop invisible
# ---------------------------------------------------------------------------


def test_encode_prefix_really_does_drop_an_all_false_mask(subtask_tokenizer):
    """Proves stage 3 is real, so the fail-closed work above is justified.

    FAILS IF: encode_prefix stops short-circuiting on an all-False mask -- in
    which case the silent-drop story would be wrong and this reasoning should be
    revisited rather than left in place.
    """
    # Reuse the stub model from the sibling test file (same convention as
    # tests/test_pi05_ki_joint_trainer.py:30-36).
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from test_hierarchy_cache import _obs
    from test_hierarchy_cache import _StubModel

    from openpi.models_pytorch.action_experts.subtask_expert import SubtaskActionExpert

    expert = SubtaskActionExpert()
    model = _StubModel()
    images, img_masks, lang_tokens, lang_masks = _obs(7)

    empty_tokens = torch.zeros(1, 16, dtype=torch.long)
    empty_mask = torch.zeros(1, 16, dtype=torch.bool)
    ctx_dropped = expert.encode_prefix(
        model=model,
        images=images,
        img_masks=img_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        subtask_tokens=empty_tokens,
        subtask_mask=empty_mask,
    )
    ctx_none = expert.encode_prefix(
        model=model, images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
    )
    # Identical prefix length => the 16-token segment vanished without a trace.
    assert ctx_dropped["prefix_pad_masks"].shape == ctx_none["prefix_pad_masks"].shape

    live_mask = torch.ones(1, 16, dtype=torch.bool)
    ctx_live = expert.encode_prefix(
        model=model,
        images=images,
        img_masks=img_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        subtask_tokens=empty_tokens,
        subtask_mask=live_mask,
    )
    assert ctx_live["prefix_pad_masks"].shape[1] == ctx_none["prefix_pad_masks"].shape[1] + 16


# ---------------------------------------------------------------------------
# Tokenizer-level hierarchy encoding
# ---------------------------------------------------------------------------


def test_tokenize_hierarchy_round_trips_through_decode(subtask_tokenizer):
    """FAILS IF: padding/BOS/EOS handling corrupts the hierarchy text on the way
    back out, which would silently break rollout logging and eval."""
    tokens, mask, ar_mask, loss_mask = subtask_tokenizer.tokenize_hierarchy(HIER_TEXT)
    assert subtask_tokenizer.decode_hierarchy(tokens) == HIER_TEXT
    assert ar_mask[: int(mask.sum())].all(), "hierarchy segment must be causal"
    assert not loss_mask[0], "BOS must not be a CE target"
    assert loss_mask[1 : int(mask.sum())].all()


def test_tokenize_hierarchy_rejects_malformed_text(subtask_tokenizer):
    """FAILS IF: malformed hierarchy text is tokenized anyway, producing a target
    the model is then trained to imitate."""
    with pytest.raises(hier.HierarchyTagError):
        subtask_tokenizer.tokenize_hierarchy("<MEM>only memory</MEM>")


def test_hierarchy_is_cheaper_than_raw_text_tags(subtask_tokenizer):
    """FAILS IF: the reserved-slot encoding stops saving tokens, i.e. the
    'compress the canonical text' half of item 1 regressed."""
    special_len = int(subtask_tokenizer.tokenize_hierarchy(HIER_TEXT)[1].sum())
    raw_len = len(subtask_tokenizer._tokenizer.encode(HIER_TEXT)) + 2  # +BOS/EOS
    assert special_len < raw_len, f"special={special_len} not cheaper than raw={raw_len}"
