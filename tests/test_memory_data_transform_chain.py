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
MEMORY_TELEMETRY_KEYS = (
    "memory_selected_stride",
    "memory_anchor_kind",
    "memory_chunk_lag",
    "memory_frame_lag",
)


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
        "memory_selected_stride": 5,
        "memory_anchor_kind": "periodic",
        "memory_chunk_lag": 3,
        "memory_frame_lag": 96,
    }


class TestRepackAllowlist:
    """Break 4: the allowlist decides what survives; omissions are silent."""

    def test_memory_source_adds_the_memory_keys(self):
        pats: dict = {}
        dc._add_conditioning_text_keys(pats, _model.ModelType.PI05_SUBTASK, MEMORY_SUBTASK_SOURCE)
        for k in (*MEMORY_KEYS, *MEMORY_TELEMETRY_KEYS):
            assert k in pats, f"{k} missing from the repack allowlist for the memory source"

    def test_non_memory_source_does_not_add_them(self):
        # Without this the first test passes on an unconditional insertion.
        pats: dict = {}
        dc._add_conditioning_text_keys(pats, _model.ModelType.PI05_SUBTASK, "annotations_skill")
        for k in (*MEMORY_KEYS, *MEMORY_TELEMETRY_KEYS):
            assert k not in pats, f"{k} leaked into a non-memory run"

    @pytest.mark.parametrize("source", ["annotations_skill", "orchestrator"])
    def test_subtask_text_still_requested_for_non_memory_sources(self, source):
        """Positive control: the pre-existing key keeps working where it applies."""
        pats: dict = {}
        dc._add_conditioning_text_keys(pats, _model.ModelType.PI05_SUBTASK, source)
        assert pats.get("subtask_text") == "subtask_text"

    def test_memory_source_does_NOT_request_subtask_text(self):
        """The two channels are mutually exclusive, and this is not cosmetic.

        An earlier version of this test asserted subtask_text was present in every
        arm including memory. That assumption was wrong, and asserting it locked
        the wrong belief in: the dataset does not attach subtask_text on memory
        items, and RepackTransform is strict, so requesting it raises
        KeyError('subtask_text') on the first batch of every memory run.
        """
        pats: dict = {}
        dc._add_conditioning_text_keys(pats, _model.ModelType.PI05_SUBTASK, MEMORY_SUBTASK_SOURCE)
        assert "subtask_text" not in pats
        assert "memory_text" in pats

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
        assert out["memory_selected_stride"] == 5
        assert out["memory_anchor_kind"] == 1
        assert out["memory_chunk_lag"] == 3
        assert out["memory_frame_lag"] == 96

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

        assert mm.memory_configs(), "no memory configs registered"
        for cfg in mm.memory_configs():
            assert cfg.model.subtask_max_len == 192
            assert cfg.model.max_token_len == 320
            # `pytorch_model_name` is the factory dispatch key, not ModelType's
            # value. "pi05_subtask" falls through to PI0Pytorch; "subtask" is the
            # registered branch used by every other Pi05SubtaskConfig.
            assert cfg.pytorch_model_name == "subtask"
            data = cfg.data[0] if isinstance(cfg.data, (list, tuple)) else cfg.data
            assert data.base_config.subtask_source == MEMORY_SUBTASK_SOURCE

    def test_a_registered_config_actually_turns_the_memory_path_on(self):
        """End-to-end on the config: the switch must reach the repack allowlist.

        This is the check whose absence made the memory path unreachable: every
        component was implemented and no config selected it.
        """
        import openpi.training.moma_memory_config as mm

        cfg = mm.memory_configs()[0]
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
            ({"planner_stride": 5, "planner_stride_weights": ((1, 0.4), (5, 0.6))}, ValueError),
        ],
    )
    def test_stride_guards_reject(self, kwargs, exc):
        import openpi.training.moma_memory_config as mm

        with pytest.raises(exc):
            mm.make_memory_train_config(name="probe", **kwargs)

    def test_mix_c_config_records_one_authoritative_spec(self):
        import openpi.training.moma_memory_config as mm

        cfg = mm.make_memory_train_config(
            name="probe-mix-c",
            planner_stride_weights=mm.MIX_C_WEIGHTS,
            planner_stride_seed=mm.MIX_C_SEED,
        )
        factory = cfg.data[0] if isinstance(cfg.data, (list, tuple)) else cfg.data
        base = factory.base_config
        assert base.memory_planner_stride_weights == mm.MIX_C_WEIGHTS
        assert base.memory_planner_stride_seed == mm.MIX_C_SEED
        assert base.memory_frames_per_chunk == cfg.model.action_horizon == 32

    def test_registered_k1_short_is_a_single_variable_derivation(self):
        import openpi.training.moma_memory_config as mm
        from openpi.training.train_config import get_config

        k1 = get_config("pi05_moma_memory_b1k-k1-short")
        mix = get_config("pi05_moma_memory_b1k-mix-c-short")
        assert k1.name == "pi05_moma_memory_b1k-k1-short"
        assert k1.data[0].base_config.memory_planner_stride_weights == ((1, 1.0),)
        assert k1.data[0].base_config.memory_planner_stride_seed == mm.MIX_C_SEED
        assert k1.data[0].base_config.memory_frames_per_chunk == 32
        assert mix.data[0].base_config.memory_planner_stride_weights == mm.MIX_C_WEIGHTS
        assert k1.num_train_steps == mix.num_train_steps == 200
        assert k1.lr_schedule == mix.lr_schedule
        assert k1.lr_schedule.warmup_steps == 20
        assert k1.lr_schedule.decay_steps == 200
        schedule = k1.lr_schedule.create()
        assert float(schedule(0)) < float(schedule(20))
        assert float(schedule(199)) < float(schedule(20))
        assert k1.model == mix.model
        assert k1.optimizer == mix.optimizer
        assert k1.pytorch_weight_path == mix.pytorch_weight_path
        assert k1.data[0].base_config.behavior_dataset_root == mix.data[0].base_config.behavior_dataset_root

    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_valid_strides_are_accepted(self, k):
        """Positive control: the guards are specific, not a blanket refusal."""
        import openpi.training.moma_memory_config as mm

        assert mm.make_memory_train_config(name="probe", planner_stride=k).model.planner_stride == k


def test_real_mix_c_sample_and_memory_provenance_share_one_decision():
    import dataclasses
    import pathlib

    import pytest

    pytest.importorskip("behavior.learning.datas.dataset")
    from openpi.training import data_loader
    from openpi.training.train_config import get_config

    cfg = get_config("pi05_moma_memory_b1k-mix-c")
    factory = cfg.data[0] if isinstance(cfg.data, (list, tuple)) else cfg.data
    factory = dataclasses.replace(
        factory,
        base_config=dataclasses.replace(factory.base_config, episodes_index=[0]),
    )
    data_config = factory.create(pathlib.Path(cfg.assets_base_dir), cfg.model)
    dataset = data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)._dataset
    episode = int(dataset.episodes[0])
    fields = (
        "memory_selected_stride",
        "memory_anchor_kind",
        "memory_anchor_frame",
        "memory_anchor_interval_idx",
        "memory_target_interval_idx",
        "memory_chunk_lag",
        "memory_frame_lag",
    )
    expected = {
        0: (0, "initial", 0, 0),
        256: (5, "periodic", 160, 3),
        384: (10, "periodic", 320, 2),
    }
    for frame, (stride, kind, anchor_frame, chunk_lag) in expected.items():
        sample = dataset.sample_for_frame(episode, frame, decode_observations=False)
        provenance = dataset.memory_provenance(episode, frame)
        assert {field: sample[field] for field in fields} == {
            field: provenance[field] for field in fields
        }
        assert sample["memory_selected_stride"] == stride
        assert sample["memory_anchor_kind"] == kind
        assert sample["memory_anchor_frame"] == anchor_frame
        assert sample["memory_chunk_lag"] == chunk_lag

        # Drive the exact post-Dataset production transform sequence without
        # decoding video pixels. The three image placeholders already have the
        # real shape/dtype expected by B1kInputs; all text/provenance comes from
        # the real Dataset row above.
        item = _item()
        item["action"] = item.pop("actions")
        item["task"] = item["prompt"]
        item["observation.images.rgb.head"] = item.pop("observation/egocentric_camera")
        item["observation.images.rgb.left_wrist"] = item.pop("observation/wrist_image_left")
        item["observation.images.rgb.right_wrist"] = item.pop("observation/wrist_image_right")
        item["observation.state"] = item.pop("observation/state")
        for key in ("memory_text", "previous_memory_text", *MEMORY_TELEMETRY_KEYS):
            item[key] = sample[key]
        repack = data_config.repack_transforms.inputs[0](item)
        transformed = data_config.data_transforms.inputs[0](repack)
        tokenized = data_config.model_transforms.inputs[-2](transformed)
        from openpi.training.data_loader import _collate_fn
        import torch

        batch = _collate_fn([tokenized])
        batch = {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value for key, value in batch.items()}
        observation = _model.Observation.from_dict(batch)
        assert int(observation.memory_selected_stride) == stride
        assert int(observation.memory_chunk_lag) == chunk_lag

        import importlib.util
        script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_accelerate.py"
        spec = importlib.util.spec_from_file_location("mixc_e2e_train_accelerate", script)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        metrics, model_observation = trainer._memory_telemetry_from_observation(observation)
        assert metrics[f"memory_k{stride}_count"] == 1
        expected_model_observation = observation.without_memory_telemetry()
        assert torch.equal(model_observation.state, expected_model_observation.state)
        assert model_observation.images.keys() == expected_model_observation.images.keys()
        for image_key in model_observation.images:
            assert np.array_equal(
                np.asarray(model_observation.images[image_key]),
                np.asarray(expected_model_observation.images[image_key]),
            )
        assert model_observation.memory_selected_stride is None
        assert model_observation.memory_anchor_kind is None
        assert model_observation.memory_chunk_lag is None
        assert model_observation.memory_frame_lag is None


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
