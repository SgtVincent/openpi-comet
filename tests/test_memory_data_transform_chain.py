"""Tests for the data-transform layer of the memory chain (breaks 4 and 5).

These cover the two drops that sat UPSTREAM of every earlier guard: the
``RepackTransform`` allowlist, and ``B1kInputs`` rebuilding the item dict.

Both were invisible to source-level checks.  The allowlist omission looked like
three independent dicts that each happened not to mention memory, and
``B1kInputs`` looked fine because the key it *does* forward (``subtask_text``)
was right there.  Only running the transforms and watching a sentinel disappear
showed it, which is why every test here calls the real transform on real-shaped
data rather than asserting about source text.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import openpi.models.model as _model
import openpi.training.data_config as dc
from openpi.policies.b1k_policy import B1kInputs
from openpi.training.memory_annotation import MEMORY_SUBTASK_SOURCE

MEMORY_KEYS = ("memory_text", "previous_memory_text")


def _item() -> dict:
    """An item shaped like what the dataset emits, with traceable sentinels."""
    return {
        "observation/egocentric_camera": np.zeros((3, 224, 224), np.uint8),
        "observation/wrist_image_left": np.zeros((3, 224, 224), np.uint8),
        "observation/wrist_image_right": np.zeros((3, 224, 224), np.uint8),
        # 256 wide because PROPRIOCEPTION_INDICES slices reach index 255.
        "observation/state": np.zeros(256, np.float32),
        "actions": np.zeros((32, 23), np.float32),
        "prompt": "do the thing",
        "subtask_text": "SUBTASK-SENTINEL",
        "memory_text": "MEMORY-SENTINEL",
        "previous_memory_text": "PREV-SENTINEL",
    }


class TestRepackAllowlist:
    """Break 4: the allowlist decides what survives; omissions are silent."""

    def test_memory_source_adds_the_memory_keys(self):
        pats: dict = {}
        dc._add_conditioning_text_keys(pats, _model.ModelType.PI05_SUBTASK, MEMORY_SUBTASK_SOURCE)
        for k in MEMORY_KEYS:
            assert k in pats, f"{k} missing from the repack allowlist for the memory source"

    def test_non_memory_source_does_not_add_them(self):
        # Without this the first test passes on an unconditional insertion.
        pats: dict = {}
        dc._add_conditioning_text_keys(pats, _model.ModelType.PI05_SUBTASK, "annotations_skill")
        for k in MEMORY_KEYS:
            assert k not in pats, f"{k} leaked into a non-memory run"

    @pytest.mark.parametrize("source", [MEMORY_SUBTASK_SOURCE, "annotations_skill", "orchestrator"])
    def test_subtask_text_is_unaffected(self, source):
        """Positive control: the pre-existing key must keep working in every arm."""
        pats: dict = {}
        dc._add_conditioning_text_keys(pats, _model.ModelType.PI05_SUBTASK, source)
        assert pats.get("subtask_text") == "subtask_text"

    def test_non_subtask_model_gets_neither(self):
        pats: dict = {}
        dc._add_conditioning_text_keys(pats, _model.ModelType.PI0, MEMORY_SUBTASK_SOURCE)
        assert "subtask_text" not in pats

    def test_all_three_factories_read_subtask_source_off_base_config(self):
        """The field lives on ``base_config``, not on the factory.

        Reading it off the factory returns the default forever and adds nothing,
        with no error -- the same wrong-object mistake that made
        ``ModelTransformFactory`` raise AttributeError. A behavioural check is
        used because ``getattr(self, ..., default)`` cannot fail loudly.
        """
        for factory_cls in (
            dc.LeRobotB1KDataConfig,
            dc.LeRobotB1KRGBDDataConfig,
            dc.LeRobotB1KRGBSegmentationDataConfig,
        ):
            fields = {f.name for f in dataclasses.fields(factory_cls)}
            assert "subtask_source" not in fields, (
                f"{factory_cls.__name__} now has its own subtask_source; the repack call sites "
                "read base_config and would ignore it"
            )
            assert "base_config" in fields


class TestB1kInputsForwarding:
    """Break 5: B1kInputs REBUILDS the dict, so unnamed keys vanish."""

    def _run(self, model_type, item=None):
        return B1kInputs(action_dim=32, model_type=model_type)(item or _item())

    def test_memory_keys_survive_for_the_subtask_model(self):
        out = self._run(_model.ModelType.PI05_SUBTASK)
        assert out.get("memory_text") == "MEMORY-SENTINEL"
        assert out.get("previous_memory_text") == "PREV-SENTINEL"

    def test_subtask_text_still_survives(self):
        """Positive control: proves the transform runs and forwards at all."""
        out = self._run(_model.ModelType.PI05_SUBTASK)
        assert out.get("subtask_text") == "SUBTASK-SENTINEL"

    def test_non_subtask_model_gets_neither(self):
        out = self._run(_model.ModelType.PI0)
        assert "memory_text" not in out
        assert "subtask_text" not in out

    def test_absent_memory_keys_are_not_fabricated(self):
        item = _item()
        for k in MEMORY_KEYS:
            del item[k]
        out = self._run(_model.ModelType.PI05_SUBTASK, item)
        for k in MEMORY_KEYS:
            assert k not in out, f"{k} was invented from nothing"

    def test_state_reaching_the_tokenizer_is_23_not_32(self):
        """The prompt budget depends on this number.

        32 is the model's padded width; ``PadStatesAndActions`` applies it AFTER
        tokenization, so the 9 padding slots never reach the ``State:`` text.
        A budget computed at 32 overestimates the prompt segment.
        """
        out = self._run(_model.ModelType.PI05_SUBTASK)
        assert np.asarray(out["state"]).shape[-1] == 23


class TestMemoryTrainConfig:
    def test_registered_configs_pin_the_decided_budgets(self):
        import openpi.training.moma_memory_config as mm

        assert mm.MOMA_MEMORY_CONFIGS, "no memory configs registered"
        for cfg in mm.MOMA_MEMORY_CONFIGS:
            assert cfg.model.subtask_max_len == 192
            assert cfg.model.max_token_len == 320
            data = cfg.data[0] if isinstance(cfg.data, (list, tuple)) else cfg.data
            assert data.base_config.subtask_source == MEMORY_SUBTASK_SOURCE

    def test_a_registered_config_actually_turns_the_memory_path_on(self):
        """End-to-end on the config: the switch must reach the repack allowlist.

        This is the check whose absence made the memory path unreachable: every
        component was implemented and no config selected it.
        """
        import openpi.training.moma_memory_config as mm

        cfg = mm.MOMA_MEMORY_CONFIGS[0]
        data = cfg.data[0] if isinstance(cfg.data, (list, tuple)) else cfg.data
        pats: dict = {}
        dc._add_conditioning_text_keys(
            pats, cfg.model.model_type, data.base_config.subtask_source
        )
        for k in MEMORY_KEYS:
            assert k in pats

    @pytest.mark.parametrize(
        ("kwargs", "exc"),
        [
            ({"planner_stride": True}, TypeError),
            ({"planner_stride": -1}, ValueError),
            ({"planner_stride": 0}, ValueError),
            ({"planner_stride": 5, "planner_stride_weights": ((1, 0.5), (1, 0.5))}, ValueError),
            ({"planner_stride": 5, "planner_stride_weights": ((1, 0.0),)}, ValueError),
            ({"planner_stride": 5, "planner_stride_weights": ((1, 0.4), (5, 0.6))}, NotImplementedError),
        ],
    )
    def test_stride_guards_reject(self, kwargs, exc):
        import openpi.training.moma_memory_config as mm

        with pytest.raises(exc):
            mm.make_memory_train_config(name="probe", **kwargs)

    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_valid_strides_are_accepted(self, k):
        """Positive control: the guards are specific, not a blanket refusal."""
        import openpi.training.moma_memory_config as mm

        assert mm.make_memory_train_config(name="probe", planner_stride=k).model.planner_stride == k


class TestFactoryCreateEndToEnd:
    """The gap the direct-helper tests leave open.

    Calling ``_add_conditioning_text_keys`` directly proves the helper works. It
    does NOT prove the call sites pass it the right thing -- and the first version
    of this fix passed ``getattr(self, "subtask_source", "orchestrator")``, where
    ``self`` is the factory and the field lives on ``self.base_config``. That
    returns the default forever, adds no memory keys, and raises nothing.

    So this drives the real ``create()`` and inspects the repack allowlist it
    actually built.
    """

    def _patterns(self, subtask_source: str) -> dict:
        import pathlib as _pl

        import openpi.models.pi05_subtask_config as _p5

        factory = dc.LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=dc.DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="/nonexistent-root-for-this-test",
                subtask_source=subtask_source,
            ),
        )
        created = factory.create(_pl.Path("/nonexistent-assets"), _p5.Pi05SubtaskConfig())
        repack = created.repack_transforms.inputs[0]
        return dict(repack.structure)

    def test_created_config_carries_memory_keys_for_the_memory_source(self):
        pats = self._patterns(MEMORY_SUBTASK_SOURCE)
        for k in MEMORY_KEYS:
            assert k in pats, (
                f"{k} absent from the allowlist the factory actually built; the call site is "
                "probably reading subtask_source off the wrong object"
            )

    def test_created_config_omits_them_for_a_non_memory_source(self):
        pats = self._patterns("annotations_skill")
        for k in MEMORY_KEYS:
            assert k not in pats
        assert "subtask_text" in pats, "positive control: the non-memory arm is still wired"
