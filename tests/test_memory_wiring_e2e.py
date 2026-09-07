"""End-to-end wiring test for the MoMA-VLA memory path.

Each of the three links below was individually "implemented" and the chain was
still broken: memory_text was dropped before the tokenizer, the fail-closed
switch was off so a missing field became an all-zero segment, and
previous_memory_text rode along as an unconsumed key while the prefix stayed
byte-identical to an unconditioned one. Nothing raised, and the loss looked fine.

Unit tests could not see that, because every part passed on its own. So these
tests run the real transform chain in the real production configuration and
assert on its output.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpi.models.tokenizer import SubtaskTokenizer
from openpi.training.memory_annotation import MEMORY_SUBTASK_SOURCE
from openpi.transforms import PromptFromLeRobotItem, TokenizeSubtaskInputs

STATE_DIM = 32
MEMORY_BODY = "No steps completed; currently picking up the radio, next step is to press it."
PREV_MEMORY = "No task steps have been completed; prepare to begin the task."
TARGET = "\n".join([
    "Memory: " + MEMORY_BODY,
    "Primitive: pick up the radio from the coffee table",
    "Skill: move to the radio",
    "Next skill: pick up the radio from the coffee table",
    "Next primitive: press the radio",
])


@pytest.fixture(scope="module")
def tok():
    return SubtaskTokenizer(prompt_max_len=512, subtask_max_len=128)


@pytest.fixture(scope="module")
def state():
    return np.random.default_rng(20260907).uniform(-1, 1, STATE_DIM).astype(np.float32)


def _dataset_item(state, *, memory: bool):
    """What BehaviorLeRobotDataset.__getitem__ actually puts on the item."""
    item = {"task": "turn on the radio", "state": state}
    if memory:
        item["memory_text"] = TARGET
        item["previous_memory_text"] = PREV_MEMORY
    else:
        item["subtask_text"] = "press the radio"
    return item


def _chain(tok, item, *, source):
    """The production chain: PromptFromLeRobotItem -> TokenizeSubtaskInputs.

    Flags are derived from `source` exactly as data_loader/data_config derive
    them, so that a regression in that derivation shows up here.
    """
    is_memory = source == MEMORY_SUBTASK_SOURCE
    out = PromptFromLeRobotItem(
        include_subtask_text=True,
        include_memory_text=is_memory,
        require_memory_text=is_memory,
    )(item)
    return TokenizeSubtaskInputs(tokenizer=tok, require_memory=is_memory)(out)


def _prefix_text(tok, out):
    ids = [int(t) for t, m in zip(out["tokenized_prompt"], out["tokenized_prompt_mask"]) if m]
    return tok._tokenizer.decode(ids)


# --------------------------------------------------------------------------
# the four assertions
# --------------------------------------------------------------------------

def test_memory_run_produces_a_supervised_subtask_segment(tok, state):
    out = _chain(tok, _dataset_item(state, memory=True), source=MEMORY_SUBTASK_SOURCE)
    assert out["subtask_mask"].any(), "conditioning segment is empty -> encode_prefix drops it"
    supervised = int(out["subtask_loss_mask"].sum())
    assert supervised > 0, "no supervised tokens -> planner CE contributes nothing"
    assert not out["subtask_loss_mask"][0], "BOS must not be supervised"
    assert int(out["subtask_mask"].sum()) > supervised >= 1


def test_memory_run_prefix_actually_carries_previous_memory(tok, state):
    out = _chain(tok, _dataset_item(state, memory=True), source=MEMORY_SUBTASK_SOURCE)
    text = _prefix_text(tok, out)
    assert "Previous memory:" in text, f"prefix has no Previous memory label: {text[-80:]!r}"
    assert PREV_MEMORY.rstrip(".") in text, "prefix has the label but not the memory body"
    assert "Subtask:" not in text, "legacy cue must not coexist with Previous memory"


def test_memory_run_prefix_differs_from_the_unconditioned_one(tok, state):
    """The original defect was that these two were byte-identical."""
    with_mem = _chain(tok, _dataset_item(state, memory=True), source=MEMORY_SUBTASK_SOURCE)
    without = _chain(tok, _dataset_item(state, memory=False), source="annotations_skill")
    a = [int(t) for t, m in zip(with_mem["tokenized_prompt"], with_mem["tokenized_prompt_mask"]) if m]
    b = [int(t) for t, m in zip(without["tokenized_prompt"], without["tokenized_prompt_mask"]) if m]
    assert a != b, "memory-conditioned prefix is identical to the unconditioned one"


def test_legacy_skill_path_is_byte_identical_to_the_pre_change_behaviour(tok, state):
    """Negative control: the annotations_skill runs in flight must not move.

    Compared against a direct call to the legacy tokenizer entry points, which
    is what the chain did before memory existed.
    """
    out = _chain(tok, _dataset_item(state, memory=False), source="annotations_skill")
    exp_tokens, exp_mask = tok.tokenize_prompt("turn on the radio", state)
    assert (out["tokenized_prompt"] == exp_tokens).all()
    assert (out["tokenized_prompt_mask"] == exp_mask).all()
    st, sm, sar, sl = tok.tokenize_subtask("press the radio")
    assert (out["subtask_tokens"] == st).all()
    assert (out["subtask_mask"] == sm).all()
    assert (out["subtask_ar_mask"] == sar).all()
    assert (out["subtask_loss_mask"] == sl).all()
    text = _prefix_text(tok, out)
    assert text.endswith(";\nSubtask: "), "legacy prefix cue changed"
    assert "Previous memory" not in text


def test_zero_fabrication_branch_is_unreachable_on_a_memory_run(tok, state):
    """A memory run missing memory_text must raise, not fabricate an empty segment."""
    broken = {"task": "turn on the radio", "state": state,
              "previous_memory_text": PREV_MEMORY}          # memory_text absent
    with pytest.raises(ValueError, match="memory_text is required"):
        _chain(tok, broken, source=MEMORY_SUBTASK_SOURCE)


def test_missing_previous_memory_on_a_memory_run_raises(tok, state):
    broken = {"task": "turn on the radio", "state": state,
              "memory_text": TARGET}                        # previous_memory_text absent
    with pytest.raises(ValueError, match="previous_memory_text"):
        _chain(tok, broken, source=MEMORY_SUBTASK_SOURCE)


# --------------------------------------------------------------------------
# the flags themselves: the defect was that these defaulted to off
# --------------------------------------------------------------------------

def test_flags_default_to_off_so_the_derivation_must_be_explicit(tok, state):
    """Records why the production call sites must pass these explicitly.

    With the defaults, memory_text is dropped and the segment is fabricated
    all-zero -- exactly the state the chain was shipped in.
    """
    dropped = PromptFromLeRobotItem(include_subtask_text=True)(
        _dataset_item(state, memory=True))
    assert "memory_text" not in dropped, "default still keeps memory_text (behaviour changed)"
    assert "previous_memory_text" not in dropped
    out = TokenizeSubtaskInputs(tokenizer=tok)(dropped)
    assert not out["subtask_mask"].any(), (
        "this is the silent all-zero segment; asserted so its shape stays documented"
    )


def test_production_call_sites_derive_the_flags_from_the_data_config():
    """Source-level: pins that the two loader sites and the tokenize site were
    actually changed, so a revert is caught here rather than in a 32-GPU run."""
    import pathlib
    import re

    def code_only(text: str) -> str:
        """Strip comments and docstrings.

        The first version of this test searched the raw file and matched its own
        explanatory comment, which quotes the defective call verbatim -- the check
        reported the bug it was written to rule out. Any assertion that a source
        pattern is ABSENT has to look at code only, because prose about the
        pattern is indistinguishable from the pattern.
        """
        text = re.sub(r'""".*?"""', "", text, flags=re.S)
        return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))

    root = pathlib.Path(__file__).resolve().parents[1]
    loader = code_only((root / "src/openpi/training/data_loader.py").read_text())
    # Both construction sites now route through one helper, so the flags are set
    # in a single place that tests can actually CALL. The semantics are covered
    # behaviourally in tests/test_memory_wiring_callable.py; what remains here is
    # only "both sites go through the helper", which is a structural claim.
    # Count CALL sites, not the substring: the `def` line matches it too. Three
    # earlier source assertions in this change set misfired the same way -- one
    # matched its own explanatory comment, one matched `model_action_horizon`, and
    # this one matched the function definition. Substring counting over source is
    # the recurring mistake, so match the call form specifically.
    call_sites = loader.count("[prompt_transform_for(")
    assert call_sites == 2, (
        f"expected 2 call sites building the prompt transform via the helper, found "
        f"{call_sites}; both dataset construction paths must route through it"
    )
    assert loader.count("def prompt_transform_for(") == 1, "the helper must have one definition"
    assert "include_memory_text=active" in loader and "require_memory_text=active" in loader
    cfg = code_only((root / "src/openpi/training/data_config.py").read_text())
    assert "require_memory=self.subtask_source == _MEMORY_SUBTASK_SOURCE" in cfg
    # ...and the field it reads must exist, which a string match alone cannot tell.
    # This exact combination shipped broken: the string was present, the field was
    # not, and the call raised AttributeError the first time anything invoked it.
    from openpi.training.data_config import ModelTransformFactory

    assert any(
        f.name == "subtask_source" for f in dataclasses.fields(ModelTransformFactory)
    ), "data_config reads self.subtask_source but ModelTransformFactory has no such field"
    tf_raw = (root / "src/openpi/transforms.py").read_text()
    tf = code_only(tf_raw)
    assert "previous_memory=previous_memory_text" in tf
    assert "tokenize_prompt(prompt, state)" not in tf, (
        "the unconditional two-argument call is back; the prefix would carry no memory"
    )
    # Positive control: the stripper must not be so aggressive that it deletes the
    # code it is meant to inspect, which would make the assertion above vacuous.
    assert "def __call__" in tf and "tokenize_prompt(" in tf
    # and it must actually have removed the comment that fooled the first version
    assert "tokenize_prompt(prompt, state)" in tf_raw, (
        "if the explanatory comment is gone this control no longer proves anything"
    )
