# ruff: noqa: SLF001 - focused tests intentionally exercise private lean-mode helpers.
"""Focused CPU tests for the lean formal B1K stride-12 mode."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import behavior.learning.datas.dataset as behavior_dataset_module
from behavior.learning.datas.dataset import BehaviorLeRobotDataset
from behavior.learning.datas.dataset import _aligned_streaming_chunk_start
from behavior.learning.datas.dataset import _read_streaming_anchor_env


_STREAMING_ENV_KEYS = (
    "OPENPI_B1K_ANCHOR_STRIDE",
    "OPENPI_B1K_ANCHOR_OFFSET",
    "OPENPI_B1K_DROP_INCOMPLETE_HORIZON",
)


class _EnvCaptureDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.contract = _read_streaming_anchor_env()

    def __len__(self):
        return 8

    def __getitem__(self, _index):
        stride, offset, drop = self.contract
        return torch.tensor([stride, offset, int(drop)])


class _RecordingLoader:
    def __init__(self, key: str, count: int):
        self.key = key
        self.values = iter(range(count))
        self.consumed = []
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        self.consumed.append(value)
        return torch.tensor([value])

    def close(self):
        self.closed = True


def _streaming_stub(*, chunk=(1000, 1250, 250), cursor=1006, stride=12, offset=4, frames=300):
    dataset = object.__new__(BehaviorLeRobotDataset)
    dataset._active_chunks = [chunk]
    dataset.current_streaming_chunk_idx = 0
    dataset.current_streaming_frame_idx = cursor
    dataset._streaming_anchor_stride = stride
    dataset._streaming_anchor_offset = offset
    dataset._streaming_drop_incomplete_horizon = True
    dataset._should_obs_loaders_reload = False
    dataset.meta = SimpleNamespace(video_keys=("rgb.head", "depth.head"))
    dataset.obs_loaders = {
        key: _RecordingLoader(key, frames)
        for key in dataset.meta.video_keys
    }
    return dataset


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"OPENPI_B1K_ANCHOR_STRIDE": "bad"}, "must be an integer"),
        ({"OPENPI_B1K_ANCHOR_STRIDE": "0"}, "must be >= 1"),
        (
            {"OPENPI_B1K_ANCHOR_STRIDE": "12", "OPENPI_B1K_ANCHOR_OFFSET": "12"},
            "0 <= offset < stride",
        ),
        ({"OPENPI_B1K_DROP_INCOMPLETE_HORIZON": "true"}, "must be 0 or 1"),
    ],
)
def test_streaming_env_validation(monkeypatch, env, message):
    for key in _STREAMING_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=message):
        _read_streaming_anchor_env()


def test_streaming_env_absent_preserves_legacy_defaults(monkeypatch):
    for key in _STREAMING_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert _read_streaming_anchor_env() == (1, 0, False)


def test_episode_local_alignment_ignores_global_row_modulo():
    # global_start % 12 == 4, while episode_local_start % 12 == 10. Offset 4
    # must therefore select global_start + 6 rather than global_start.
    assert 1000 % 12 == 4
    assert _aligned_streaming_chunk_start((1000, 1250, 250), stride=12, offset=4) == 1006
    assert _aligned_streaming_chunk_start((1250, 1500, 500), stride=12, offset=4) == 1258
    assert _aligned_streaming_chunk_start((10, 11, 1), stride=12, offset=0) is None
    assert _aligned_streaming_chunk_start((10, 12, 11), stride=12, offset=0) == 11


def test_stride_advance_discards_eleven_frames_in_every_modality():
    dataset = _streaming_stub()
    for key in dataset.meta.video_keys:
        assert dataset._next_streaming_observation(key, context="test").item() == 0
    dataset._advance_streaming_anchor(observation_consumed=True, context="test")

    assert dataset.current_streaming_frame_idx == 1018
    for loader in dataset.obs_loaders.values():
        assert loader.consumed == list(range(12))
    assert dataset._should_obs_loaders_reload is False


def test_chunk_crossing_never_over_discards_and_marks_reload():
    dataset = _streaming_stub(chunk=(1000, 1010, 250), cursor=1006)
    for key in dataset.meta.video_keys:
        dataset._next_streaming_observation(key, context="test")
    dataset._advance_streaming_anchor(observation_consumed=True, context="test")

    assert dataset.current_streaming_frame_idx == 1010
    assert dataset._should_obs_loaders_reload is True
    for loader in dataset.obs_loaders.values():
        assert loader.consumed == [0]


def test_chunk_reload_closes_old_loaders_and_uses_episode_local_cursor(monkeypatch, tmp_path):
    dataset = object.__new__(BehaviorLeRobotDataset)
    dataset.root = tmp_path
    dataset._active_chunks = [(1000, 1250, 250)]
    dataset.current_streaming_chunk_idx = 0
    dataset.current_streaming_frame_idx = 1006
    dataset.current_streaming_episode_idx = None
    dataset.train_rgb_type = "regular"
    dataset.omnigibson_mapping = {}
    dataset.meta = SimpleNamespace(
        video_keys=("observation.images.rgb.head", "observation.images.depth.head")
    )
    old_loaders = {
        key: _RecordingLoader(key, 1)
        for key in dataset.meta.video_keys
    }
    dataset.obs_loaders = old_loaders
    dataset._should_obs_loaders_reload = True
    calls = []

    def loader_factory(**kwargs):
        calls.append(kwargs)
        return _RecordingLoader(kwargs["camera_id"], 20)

    monkeypatch.setattr(
        behavior_dataset_module,
        "OBS_LOADER_MAP",
        {"rgb": loader_factory, "depth": loader_factory},
    )
    dataset._reload_streaming_observation_loaders(
        {"task_index": torch.tensor(3)},
        ep_idx=17,
    )

    assert all(loader.closed for loader in old_loaders.values())
    assert len(dataset.obs_loaders) == 2
    assert [call["start_idx"] for call in calls] == [256, 256]
    assert all(call["batch_size"] == 1 and call["stride"] == 1 for call in calls)
    assert dataset.current_streaming_episode_idx == 17
    assert dataset._should_obs_loaders_reload is False


def test_rejected_anchor_consumes_current_plus_stride_minus_one():
    dataset = _streaming_stub()
    dataset._advance_streaming_anchor(observation_consumed=False, context="rejection")
    assert dataset.current_streaming_frame_idx == 1018
    for loader in dataset.obs_loaders.values():
        assert loader.consumed == list(range(12))


def test_streaming_stop_iteration_fails_closed_with_context():
    dataset = _streaming_stub(frames=1)
    for key in dataset.meta.video_keys:
        dataset._next_streaming_observation(key, context="return")
    with pytest.raises(RuntimeError, match=r"modality=rgb\.head.*cursor=1006.*stride=12"):
        dataset._advance_streaming_anchor(observation_consumed=True, context="discard")


def test_tail_drop_excludes_final_31_starts_but_legacy_padding_remains():
    dataset = object.__new__(BehaviorLeRobotDataset)
    dataset.episode_data_index_pos = {7: 0}
    dataset.episode_data_index = {
        "from": torch.tensor([0]),
        "to": torch.tensor([100]),
    }
    dataset.delta_indices = {"action": list(range(32))}

    eligible = []
    for anchor in range(100):
        indices, padding = dataset._get_query_indices(anchor, 7)
        if not dataset._action_horizon_is_padded(padding):
            eligible.append(anchor)
            assert indices["action"] == list(range(anchor, anchor + 32))
    assert eligible == list(range(69))

    # Legacy behavior still exposes clamped indices plus padding; only the
    # construction-time drop flag decides whether streaming __getitem__ skips it.
    indices, padding = dataset._get_query_indices(69, 7)
    assert indices["action"][-1] == 99
    assert padding["action_is_pad"].sum().item() == 1


@pytest.mark.parametrize("num_workers", [0, 1, 4])
def test_each_pass_construction_captures_fresh_offset(monkeypatch, num_workers):
    monkeypatch.setenv("OPENPI_B1K_ANCHOR_STRIDE", "12")
    monkeypatch.setenv("OPENPI_B1K_DROP_INCOMPLETE_HORIZON", "1")
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")
    monkeypatch.setenv("MKL_SERVICE_FORCE_INTEL", "1")
    observed = []
    for offset in (0, 4, 8):
        monkeypatch.setenv("OPENPI_B1K_ANCHOR_OFFSET", str(offset))
        loader = torch.utils.data.DataLoader(
            _EnvCaptureDataset(),
            batch_size=2,
            num_workers=num_workers,
            persistent_workers=False,
        )
        iterator = iter(loader)
        observed.append(tuple(next(iterator)[0].tolist()))
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        del iterator, loader
    assert observed == [(12, 0, 1), (12, 4, 1), (12, 8, 1)]


def _load_train_accelerate():
    try:
        import accelerate  # noqa: F401
    except ModuleNotFoundError:
        accelerate_module = ModuleType("accelerate")
        accelerate_utils = ModuleType("accelerate.utils")

        class Accelerator:
            pass

        class DistributedDataParallelKwargs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class DistributedType:
            DEEPSPEED = "DEEPSPEED"

        accelerate_module.Accelerator = Accelerator
        accelerate_utils.DistributedDataParallelKwargs = DistributedDataParallelKwargs
        accelerate_utils.DistributedType = DistributedType
        sys.modules["accelerate"] = accelerate_module
        sys.modules["accelerate.utils"] = accelerate_utils

    module_name = "_openpi_train_accelerate_lean_b1k_test"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    trainer_path = _REPO_ROOT / "scripts/train_accelerate.py"
    spec = importlib.util.spec_from_file_location(module_name, trainer_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load_train_accelerate()


def _formal_config_stub(
    *,
    name="pi05_ki_joint_query_b1k-full_task-ki_on_bf16",
    resume=False,
    prepare_only=False,
):
    is_v100 = name.endswith("_v100_fp32")

    def data_config(episodes):
        return SimpleNamespace(
            base_config=SimpleNamespace(
                tasks=None,
                episodes_index=episodes,
                skill_bridge=SimpleNamespace(enabled=False),
            )
        )

    return SimpleNamespace(
        name=name,
        pytorch_model_name=("pi05_ki_joint_fast" if "joint_fast" in name else "pi05_ki_joint_query"),
        data=[data_config(list(range(180)))],
        val_data=[data_config(list(range(180, 200)))],
        batch_size_per_gpu=1 if is_v100 else 8,
        gradient_accumulation_steps=8 if is_v100 else 1,
        pytorch_training_precision="float32" if is_v100 else "bfloat16",
        accelerate_mixed_precision="no" if is_v100 else "bf16",
        num_train_steps=104_912,
        num_train_epochs=None,
        save_interval=10_000,
        checkpoint_policy="step",
        rolling_checkpoint_interval=10_000,
        val_log_interval=1_000,
        val_num_batches=20,
        streaming_anchor_stride=12,
        overwrite=False,
        wandb_enabled=True,
        project_name="pi05_ki",
        resume=resume,
        prepare_hf_cache_only=prepare_only,
        lr_schedule=SimpleNamespace(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=104_912,
            decay_lr=0.0,
        ),
    )


class _FakeDeepSpeedOptimizer:
    def __init__(self, engine):
        self.engine = engine
        self.zero_grad_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1
        self.engine.gradients = {"backbone": 0, "expert": 0}


class _FakeDeepSpeedEngine:
    def __init__(self):
        self.boundaries = []
        self.backward_phases = []
        self.gradients = {"backbone": 0, "expert": 0}
        self.step_gradients = []
        self.step_calls = 0
        self.optimizer = _FakeDeepSpeedOptimizer(self)
        self._boundary = None

    def set_gradient_accumulation_boundary(self, boundary):
        self._boundary = bool(boundary)
        self.boundaries.append(self._boundary)

    def backward(self, phase):
        self.backward_phases.append(phase)
        self.gradients[phase] += 1

    def step(self):
        assert self._boundary is True
        self.step_calls += 1
        self.step_gradients.append(dict(self.gradients))
        self.optimizer.zero_grad()


class _FakeOptimizer:
    def __init__(self):
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self):
        self.step_calls += 1

    def zero_grad(self, *, set_to_none):
        assert set_to_none is True
        self.zero_grad_calls += 1


def _deepspeed_controller(trainer):
    engine = _FakeDeepSpeedEngine()
    accelerator = SimpleNamespace(
        distributed_type=trainer.DistributedType.DEEPSPEED,
        sync_gradients=False,
        deepspeed_engine_wrapped=SimpleNamespace(engine=engine),
    )
    return trainer._TwoPhaseUpdateController(accelerator), accelerator, engine


@pytest.mark.parametrize("grad_accum", [1, 8])
def test_two_phase_deepspeed_updates_once_after_both_phases(trainer, grad_accum):
    controller, accelerator, engine = _deepspeed_controller(trainer)
    optimizer_wrapper = SimpleNamespace()

    for microbatch in range(1, grad_accum + 1):
        accelerator.sync_gradients = microbatch == grad_accum
        controller.backward("backbone", final_phase=False)
        assert engine.boundaries[-1] is False
        controller.backward("expert", final_phase=True)
        assert engine.boundaries[-1] is accelerator.sync_gradients
        controller.step_and_zero_grad(optimizer_wrapper)

        if microbatch < grad_accum:
            assert engine.step_calls == 0
            assert engine.optimizer.zero_grad_calls == 0
            assert engine.gradients == {"backbone": microbatch, "expert": microbatch}

    assert engine.backward_phases == ["backbone", "expert"] * grad_accum
    assert engine.boundaries == [False, False] * (grad_accum - 1) + [False, True]
    assert engine.step_calls == 1
    assert engine.step_gradients == [{"backbone": grad_accum, "expert": grad_accum}]
    assert engine.optimizer.zero_grad_calls == 1
    assert engine.gradients == {"backbone": 0, "expert": 0}


def test_two_phase_non_deepspeed_retains_standard_optimizer_semantics(trainer):
    backward_calls = []
    accelerator = SimpleNamespace(
        distributed_type="NO",
        sync_gradients=False,
        backward=backward_calls.append,
    )
    controller = trainer._TwoPhaseUpdateController(accelerator)
    optimizer = _FakeOptimizer()

    for microbatch in range(1, 9):
        accelerator.sync_gradients = microbatch == 8
        controller.backward("backbone", final_phase=False)
        controller.backward("expert", final_phase=True)
        controller.step_and_zero_grad(optimizer)
        if microbatch < 8:
            assert optimizer.step_calls == 0
            assert optimizer.zero_grad_calls == 0

    assert backward_calls == ["backbone", "expert"] * 8
    assert optimizer.step_calls == 1
    assert optimizer.zero_grad_calls == 1


def test_all_formal_ab_names_share_pass_wandb_and_resume_guards(trainer, monkeypatch):
    names = tuple(trainer._FORMAL_B1K_CONFIG_CONTRACTS)
    assert names == (
        "pi05_ki_joint_fast_b1k-full_task-ki_on_bf16",
        "pi05_ki_joint_query_b1k-full_task-ki_on_bf16",
        "pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32",
        "pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32",
    )
    for name in names:
        config = _formal_config_stub(name=name)
        accelerator = SimpleNamespace(
            num_processes=32,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            distributed_type=trainer.DistributedType.DEEPSPEED if name.endswith("_v100_fp32") else "NO",
            state=SimpleNamespace(
                deepspeed_plugin=SimpleNamespace(
                    deepspeed_config={
                        "zero_optimization": {
                            "stage": 2,
                            "offload_optimizer": {"device": "cpu"},
                        }
                    }
                )
            ),
        )
        for key in (
            *_STREAMING_ENV_KEYS,
            "FRAME_ANCHOR_STRIDE",
            "FRAME_ANCHOR_OFFSETS",
            "OPENPI_PERSISTENT_WORKERS",
        ):
            monkeypatch.delenv(key, raising=False)
        trainer._validate_formal_b1k_contract(config, accelerator=accelerator)
        assert trainer._is_formal_b1k_mode(config)

        config.resume = True
        with pytest.raises(ValueError, match="resume is unsupported"):
            trainer._validate_formal_b1k_contract(config)


@pytest.mark.parametrize(
    ("distributed_type", "zero_stage", "offload_device", "message"),
    [
        ("NO", 2, "cpu", "requires DeepSpeed"),
        ("DEEPSPEED", 3, "cpu", "ZeRO stage 2"),
        ("DEEPSPEED", 2, "none", "CPU optimizer offload"),
    ],
)
def test_formal_v100_rejects_deepspeed_contract_drift(
    trainer, monkeypatch, distributed_type, zero_stage, offload_device, message
):
    config = _formal_config_stub(name="pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32")
    for key in (
        *_STREAMING_ENV_KEYS,
        "FRAME_ANCHOR_STRIDE",
        "FRAME_ANCHOR_OFFSETS",
        "OPENPI_PERSISTENT_WORKERS",
    ):
        monkeypatch.delenv(key, raising=False)
    accelerator = SimpleNamespace(
        num_processes=32,
        gradient_accumulation_steps=8,
        distributed_type=(trainer.DistributedType.DEEPSPEED if distributed_type == "DEEPSPEED" else distributed_type),
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(
                deepspeed_config={
                    "zero_optimization": {
                        "stage": zero_stage,
                        "offload_optimizer": {"device": offload_device},
                    }
                }
            )
        ),
    )
    with pytest.raises(ValueError, match=message):
        trainer._validate_formal_b1k_contract(config, accelerator=accelerator)


def test_formal_validation_sets_exact_dataset_contract(trainer, monkeypatch):
    for key in (
        *_STREAMING_ENV_KEYS,
        "FRAME_ANCHOR_STRIDE",
        "FRAME_ANCHOR_OFFSETS",
        "OPENPI_PERSISTENT_WORKERS",
    ):
        monkeypatch.delenv(key, raising=False)
    accelerator = SimpleNamespace(num_processes=32, gradient_accumulation_steps=1)

    trainer._validate_formal_b1k_contract(_formal_config_stub(), accelerator=accelerator)

    assert os.environ["FRAME_ANCHOR_STRIDE"] == "12"
    assert os.environ["FRAME_ANCHOR_OFFSETS"] == "0,4,8"
    assert os.environ["OPENPI_B1K_ANCHOR_STRIDE"] == "12"
    assert os.environ["OPENPI_B1K_ANCHOR_OFFSET"] == "0"
    assert os.environ["OPENPI_B1K_DROP_INCOMPLETE_HORIZON"] == "1"
    assert os.environ["OPENPI_PERSISTENT_WORKERS"] == "0"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resume", True, "resume is unsupported"),
        ("batch_size_per_gpu", 4, "batch_size_per_gpu=8"),
        ("num_train_steps", 1, "num_train_steps=104912"),
        ("wandb_enabled", False, "wandb_enabled=True"),
    ],
)
def test_formal_validation_rejects_contract_drift(trainer, monkeypatch, field, value, message):
    config = _formal_config_stub()
    setattr(config, field, value)
    for key in (*_STREAMING_ENV_KEYS, "FRAME_ANCHOR_STRIDE", "FRAME_ANCHOR_OFFSETS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENPI_PERSISTENT_WORKERS", "0")
    with pytest.raises(ValueError, match=message):
        trainer._validate_formal_b1k_contract(config)


def test_validation_dataset_env_is_cleared_then_restored(trainer, monkeypatch):
    expected = {
        "OPENPI_B1K_ANCHOR_STRIDE": "12",
        "OPENPI_B1K_ANCHOR_OFFSET": "8",
        "OPENPI_B1K_DROP_INCOMPLETE_HORIZON": "1",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)

    with trainer._baseline_b1k_dataset_env():
        assert all(key not in os.environ for key in expected)
        assert _read_streaming_anchor_env() == (1, 0, False)

    assert {key: os.environ[key] for key in expected} == expected


def test_formal_pass_boundaries_are_continuous_and_require_two_rebuilds(trainer):
    transitions = []
    counts = [0, 0, 0]
    previous_pass = None
    for global_step in range(104_912):
        pass_index, offset, pass_steps, pass_start, pass_end = trainer._formal_b1k_pass_for_step(global_step)
        counts[pass_index] += 1
        if pass_index != previous_pass:
            transitions.append((pass_index, offset, pass_steps, pass_start, pass_end))
            previous_pass = pass_index

    assert transitions == [
        (0, 0, 34_982, 0, 34_982),
        (1, 4, 34_971, 34_982, 69_953),
        (2, 8, 34_959, 69_953, 104_912),
    ]
    assert counts == [34_982, 34_971, 34_959]
    assert len(transitions) - 1 == 2
    with pytest.raises(ValueError, match="out of range"):
        trainer._formal_b1k_pass_for_step(104_912)


def _gloo_pass_counter_worker(rank, world_size, init_file, pass_specs, queue):
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    local_counts = []
    for _offset, steps in pass_specs:
        counter = 0
        for _ in range(steps):
            counter += 1
        value = torch.tensor(counter, dtype=torch.int64)
        torch.distributed.all_reduce(value)
        assert value.item() == counter * world_size
        local_counts.append(counter)
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_counts)
    if rank == 0:
        queue.put(gathered)
    torch.distributed.destroy_process_group()


def test_two_rank_gloo_pass_counters_finish_without_collective_hang(trainer, tmp_path):
    context = torch.multiprocessing.get_context("fork")
    queue = context.SimpleQueue()
    init_file = tmp_path / "gloo-init"
    torch.multiprocessing.start_processes(
        _gloo_pass_counter_worker,
        args=(2, str(init_file), trainer._FORMAL_B1K_PASS_SPECS, queue),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    assert queue.get() == [[34_982, 34_971, 34_959], [34_982, 34_971, 34_959]]


def test_required_wandb_world1_success_and_failure(trainer, monkeypatch):
    accelerator = SimpleNamespace(is_main_process=True, num_processes=1)
    calls = []
    monkeypatch.setattr(trainer, "_init_wandb_run", lambda config, resuming: calls.append((config, resuming)))
    config = _formal_config_stub()
    trainer._init_formal_b1k_wandb(config, accelerator=accelerator)
    assert calls == [(config, False)]

    def fail_init(_config, *, resuming):
        raise RuntimeError("auth denied")

    monkeypatch.setattr(trainer, "_init_wandb_run", fail_init)
    with pytest.raises(RuntimeError, match="auth denied"):
        trainer._init_formal_b1k_wandb(config, accelerator=accelerator)


def test_required_wandb_non_main_receives_rank0_error(trainer, monkeypatch):
    accelerator = SimpleNamespace(is_main_process=False, num_processes=2)
    monkeypatch.setattr(trainer.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.distributed, "is_initialized", lambda: True)

    def broadcast(payload, src):
        assert src == 0
        payload[0] = "ImportError: byted-wandb unavailable"

    monkeypatch.setattr(trainer.torch.distributed, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="byted-wandb unavailable"):
        trainer._init_formal_b1k_wandb(_formal_config_stub(), accelerator=accelerator)


def test_prepare_only_bypasses_required_wandb(trainer, monkeypatch):
    monkeypatch.setattr(
        trainer,
        "_init_wandb_run",
        lambda *_args, **_kwargs: pytest.fail("prepare-only must not initialize W&B"),
    )
    trainer._init_formal_b1k_wandb(
        _formal_config_stub(prepare_only=True),
        accelerator=SimpleNamespace(is_main_process=True, num_processes=1),
    )


def test_formal_scheduler_edges(trainer):
    kwargs = {
        "warmup_steps": 1_000,
        "peak_lr": 1e-5,
        "decay_steps": 104_912,
        "end_lr": 0.0,
    }
    assert trainer._cosine_lr_value(0, **kwargs) == pytest.approx(1e-5 / 1_001)
    assert trainer._cosine_lr_value(1_000, **kwargs) == pytest.approx(1e-5)
    assert 0.0 < trainer._cosine_lr_value(104_911, **kwargs) < 1e-5
    assert trainer._cosine_lr_value(104_912, **kwargs) == pytest.approx(0.0, abs=1e-15)


def test_formal_runtime_wandb_logging_limitation_is_explicit():
    trainer_source = (_REPO_ROOT / "scripts/train_accelerate.py").read_text()
    assert "runtime log calls intentionally remain best-effort" in trainer_source
    assert "exact unique coverage is not claimed" in trainer_source
    assert "loader, _ = build_datasets(config)" in trainer_source
    assert "loader = accelerator.prepare(loader)" in trainer_source
    assert "_close_training_iterator(train_iterator)" in trainer_source


def test_formal_config_exact_budget_and_online_wandb():
    from openpi.training.train_config import get_config

    config = get_config("pi05_ki_joint_query_b1k-full_task-ki_on_bf16")
    assert config.batch_size_per_gpu == 8
    assert config.gradient_accumulation_steps == 1
    assert config.num_train_steps == 104_912
    assert config.num_train_epochs is None
    assert config.wandb_enabled is True
    assert config.project_name == "pi05_ki"
    assert config.lr_schedule.warmup_steps == 1_000
    assert config.lr_schedule.peak_lr == 1e-5
    assert config.lr_schedule.decay_steps == 104_912
    assert config.lr_schedule.decay_lr == 0.0


def test_launcher_contains_exact_tail_budgets_and_wandb_guards():
    launcher = (_REPO_ROOT / "scripts/run_pi05_ki_joint_query_full_b1k_bf16_multinode_hl.sh").read_text()
    for value in (
        "source8955603 steps34982 consumed8955392 drop211",
        "source8952584 steps34971 consumed8952576 drop8",
        "source8949525 steps34959 consumed8949504 drop21",
        "theoretical eligible 26857712",
        "global-batch drop 240",
        "24.9383588896%",
        "approximate coverage only",
    ):
        assert value in launcher
    assert "WANDB_DISABLED is set" in launcher
    assert "requires WANDB_MODE=online" in launcher
    assert "prepare-only skips W&B init" in launcher
    assert "RESUME=1 is unsupported" in launcher
    assert "--num-train-epochs" not in launcher


# The formal trainer intentionally leaves runtime metric logging best-effort;
# only import/auth/init is consensus-required across ranks.
