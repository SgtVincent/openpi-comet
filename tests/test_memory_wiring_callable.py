"""Behavioural tests for the two wiring points that previously had only source
assertions.

The gap these close was not theoretical. The `data_config.py` wiring read
`self.subtask_source` on a class that has no such field. The string was present,
so the source-level assertion in `test_memory_wiring_e2e.py` passed, and the code
raised `AttributeError` the first time it was actually invoked -- which nothing
did, because the surrounding path needs the full lerobot import chain (currently
broken by a numpy/numba conflict). A source assertion cannot notice that the code
it matched would raise when run.

Both entry points here are callable without constructing a dataset, so these run
on a machine where `BehaviorLeRobotDataset` cannot be imported at all.
"""

from __future__ import annotations

import dataclasses

import pytest

from openpi.models.pi05_subtask_config import Pi05SubtaskConfig
from openpi.training.data_config import DataConfig, ModelTransformFactory
from openpi.training.data_loader import _memory_source_active, prompt_transform_for
from openpi.training.memory_annotation import MEMORY_SUBTASK_SOURCE

NON_MEMORY_SOURCES = ("orchestrator", "annotations_skill", "annotations_primitive")


def _tokenize_transform(group):
    found = [t for t in group.inputs if type(t).__name__ == "TokenizeSubtaskInputs"]
    assert len(found) == 1, f"expected exactly one TokenizeSubtaskInputs, got {len(found)}"
    return found[0]


# --------------------------------------------------------------------------
# break 2: data_config -> TokenizeSubtaskInputs(require_memory=...)
# --------------------------------------------------------------------------

def test_model_transform_factory_is_callable_at_all():
    """The regression this file exists for: this call used to raise AttributeError."""
    group = ModelTransformFactory()(Pi05SubtaskConfig())
    assert group.inputs, "factory returned no input transforms"


def test_require_memory_is_true_only_for_the_memory_source():
    on = _tokenize_transform(
        ModelTransformFactory(subtask_source=MEMORY_SUBTASK_SOURCE)(Pi05SubtaskConfig())
    )
    assert on.require_memory is True, (
        "a memory run must fail closed; with this False a missing memory_text becomes "
        "an all-zero segment that encode_prefix drops, training an unconditioned model"
    )
    for src in NON_MEMORY_SOURCES:
        off = _tokenize_transform(ModelTransformFactory(subtask_source=src)(Pi05SubtaskConfig()))
        assert off.require_memory is False, f"{src} must keep its historical behaviour"


def test_the_default_is_the_non_memory_value():
    """Callers that do not pass it must be unaffected."""
    assert ModelTransformFactory().subtask_source == "orchestrator"
    assert _tokenize_transform(ModelTransformFactory()(Pi05SubtaskConfig())).require_memory is False


def test_the_flag_is_actually_derived_and_not_constant():
    """Positive control. If require_memory were hardcoded either way, one of the two
    assertions above would still pass on its own."""
    values = {
        src: _tokenize_transform(
            ModelTransformFactory(subtask_source=src)(Pi05SubtaskConfig())
        ).require_memory
        for src in (*NON_MEMORY_SOURCES, MEMORY_SUBTASK_SOURCE)
    }
    assert set(values.values()) == {True, False}, f"flag never varies: {values}"


# --------------------------------------------------------------------------
# break 1: data_loader -> PromptFromLeRobotItem(include/require_memory_text=...)
# --------------------------------------------------------------------------

def test_prompt_transform_keeps_memory_text_only_for_the_memory_source():
    model = Pi05SubtaskConfig()
    on = prompt_transform_for(DataConfig(subtask_source=MEMORY_SUBTASK_SOURCE), model)
    assert on.include_memory_text is True, (
        "with this False, transforms.py pops memory_text before the tokenizer sees it"
    )
    assert on.require_memory_text is True
    for src in NON_MEMORY_SOURCES:
        off = prompt_transform_for(DataConfig(subtask_source=src), model)
        assert off.include_memory_text is False
        assert off.require_memory_text is False


def test_both_flags_move_together():
    """require_memory_text=True with include_memory_text=False is a contradiction the
    transform itself rejects, so the wiring must never produce it."""
    for src in (*NON_MEMORY_SOURCES, MEMORY_SUBTASK_SOURCE):
        t = prompt_transform_for(DataConfig(subtask_source=src), Pi05SubtaskConfig())
        assert t.include_memory_text == t.require_memory_text


def test_memory_source_predicate_is_exact():
    assert _memory_source_active(DataConfig(subtask_source=MEMORY_SUBTASK_SOURCE)) is True
    for src in NON_MEMORY_SOURCES:
        assert _memory_source_active(DataConfig(subtask_source=src)) is False
    # a near-miss must not activate it
    assert _memory_source_active(DataConfig(subtask_source="annotations_memory_v2")) is False

    class NoField:
        pass

    assert _memory_source_active(NoField()) is False, "must not raise on a foreign config"


def test_subtask_text_flag_still_follows_the_model_type():
    """Negative control on the other half: the memory flags must not disturb it."""
    t = prompt_transform_for(DataConfig(subtask_source=MEMORY_SUBTASK_SOURCE), Pi05SubtaskConfig())
    assert t.include_subtask_text is True, "PI05_SUBTASK still expects subtask text carried"


# --------------------------------------------------------------------------
# the two halves must agree with each other
# --------------------------------------------------------------------------

@pytest.mark.parametrize("src", [*NON_MEMORY_SOURCES, MEMORY_SUBTASK_SOURCE])
def test_loader_and_config_halves_agree(src):
    """They are wired in different files from different objects. Disagreement means
    memory_text survives the first transform and is then required by the second, or
    the reverse -- either way a run that neither raises nor conditions."""
    model = Pi05SubtaskConfig()
    loader_says = prompt_transform_for(DataConfig(subtask_source=src), model).include_memory_text
    config_says = _tokenize_transform(
        ModelTransformFactory(subtask_source=src)(model)
    ).require_memory
    assert loader_says == config_says, (
        f"{src}: data_loader says include_memory_text={loader_says} but data_config "
        f"says require_memory={config_says}"
    )


def test_replace_on_the_data_config_propagates():
    """Configs are built with dataclasses.replace in the config files, so the field
    must survive that rather than being captured at class definition."""
    base = DataConfig()
    assert _memory_source_active(base) is False
    assert _memory_source_active(dataclasses.replace(base, subtask_source=MEMORY_SUBTASK_SOURCE))


# --------------------------------------------------------------------------
# the forwarding itself, through the real factory classes
# --------------------------------------------------------------------------
# The tests above construct ModelTransformFactory directly, so they do NOT cover
# the three sites that forward subtask_source into it. Verified by mutation:
# deleting the forwarding at one site left all of them green. These go through
# `create()`, which turns out to be callable without any assets on disk.

import pathlib  # noqa: E402

from openpi.training.data_config import (  # noqa: E402
    LeRobotB1KDataConfig,
    LeRobotB1KRGBDDataConfig,
    LeRobotB1KRGBSegmentationDataConfig,
)

FACTORY_CLASSES = (
    LeRobotB1KDataConfig,
    LeRobotB1KRGBDDataConfig,
    LeRobotB1KRGBSegmentationDataConfig,
)


def _require_memory_via(factory_cls, src):
    factory = factory_cls(
        repo_id="behavior-1k/2025-challenge-demos",
        base_config=DataConfig(subtask_source=src),
    )
    cfg = factory.create(pathlib.Path("/tmp/openpi_test_assets_absent"), Pi05SubtaskConfig())
    return _tokenize_transform(cfg.model_transforms).require_memory


@pytest.mark.parametrize("factory_cls", FACTORY_CLASSES, ids=lambda c: c.__name__)
def test_each_factory_forwards_subtask_source(factory_cls):
    """Kills the mutation that deletes the forwarding at one site.

    Without forwarding, ModelTransformFactory falls back to its "orchestrator"
    default and require_memory is False even on a memory run -- the run then
    trains unconditioned while reporting a loss.
    """
    assert _require_memory_via(factory_cls, MEMORY_SUBTASK_SOURCE) is True, (
        f"{factory_cls.__name__} does not forward subtask_source; the memory run's "
        "fail-closed switch stays off"
    )


@pytest.mark.parametrize("factory_cls", FACTORY_CLASSES, ids=lambda c: c.__name__)
def test_each_factory_leaves_skill_runs_alone(factory_cls):
    assert _require_memory_via(factory_cls, "annotations_skill") is False


@pytest.mark.parametrize("factory_cls", FACTORY_CLASSES, ids=lambda c: c.__name__)
def test_forwarding_is_derived_not_constant(factory_cls):
    """Positive control: a hardcoded True or False would satisfy one of the two
    tests above on its own."""
    seen = {
        src: _require_memory_via(factory_cls, src)
        for src in ("annotations_skill", MEMORY_SUBTASK_SOURCE)
    }
    assert set(seen.values()) == {True, False}, f"{factory_cls.__name__}: {seen}"


def test_a_factory_without_a_base_config_does_not_raise():
    """base_config is optional on the factory, so the getattr fallback must hold."""
    factory = LeRobotB1KDataConfig(repo_id="behavior-1k/2025-challenge-demos")
    cfg = factory.create(pathlib.Path("/tmp/openpi_test_assets_absent"), Pi05SubtaskConfig())
    assert _tokenize_transform(cfg.model_transforms).require_memory is False
