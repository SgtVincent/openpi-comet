# ruff: noqa: SLF001 - these tests intentionally exercise private checkpoint helpers.
"""Focused tests for train_accelerate epoch-oriented checkpointing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure imports resolve to this worktree instead of an editable installation
# from another checkout.
sys.path.insert(0, str(_REPO_ROOT / "src"))

_TRAINER_PATH = _REPO_ROOT / "scripts/train_accelerate.py"


def _load_train_accelerate():
    """Load the trainer without requiring Accelerate in the static-test env."""
    try:
        import accelerate  # noqa: F401
    except ModuleNotFoundError:
        accelerate_module = ModuleType("accelerate")
        accelerate_utils = ModuleType("accelerate.utils")

        class Accelerator:  # pragma: no cover - import-only placeholder
            pass

        class DistributedDataParallelKwargs:  # pragma: no cover - import-only placeholder
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class DistributedType:  # pragma: no cover - import-only placeholder
            DEEPSPEED = "DEEPSPEED"

        accelerate_module.Accelerator = Accelerator
        accelerate_utils.DistributedDataParallelKwargs = DistributedDataParallelKwargs
        accelerate_utils.DistributedType = DistributedType
        sys.modules["accelerate"] = accelerate_module
        sys.modules["accelerate.utils"] = accelerate_utils

    module_name = "_openpi_train_accelerate_checkpoint_policy_test"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, _TRAINER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load_train_accelerate()


def _epoch_config(*, total_steps: int = 500, rolling_interval: int = 40):
    return SimpleNamespace(
        checkpoint_policy="epoch_with_rolling",
        rolling_checkpoint_interval=rolling_interval,
        num_train_steps=total_steps,
        save_interval=0,
    )


def test_five_epoch_schedule_has_only_epoch_durable_boundaries(trainer):
    config = _epoch_config(total_steps=500, rolling_interval=40)
    schedule = {
        step: trainer._checkpoint_save_kind(
            config,
            global_step=step,
            steps_per_epoch=100,
        )
        for step in range(1, 501)
    }

    assert [step for step, kind in schedule.items() if kind == "epoch"] == [100, 200, 300, 400, 500]
    assert [step for step, kind in schedule.items() if kind == "rolling"] == [
        40,
        80,
        120,
        160,
        240,
        280,
        320,
        360,
        440,
        480,
    ]
    assert "step" not in schedule.values()


def test_epoch_policy_final_mid_epoch_checkpoint_is_rolling(trainer):
    config = _epoch_config(total_steps=250, rolling_interval=1000)

    assert trainer._checkpoint_save_kind(config, global_step=100, steps_per_epoch=100) == "epoch"
    assert trainer._checkpoint_save_kind(config, global_step=200, steps_per_epoch=100) == "epoch"
    assert trainer._checkpoint_save_kind(config, global_step=250, steps_per_epoch=100) == "rolling"


def test_legacy_step_policy_trigger_is_unchanged(trainer):
    config = SimpleNamespace(
        checkpoint_policy="step",
        save_interval=40,
        num_train_steps=95,
        rolling_checkpoint_interval=1,
    )
    saves = [
        step
        for step in range(1, 96)
        if trainer._checkpoint_save_kind(config, global_step=step, steps_per_epoch=10) is not None
    ]

    assert saves == [40, 80, 95]
    assert all(trainer._checkpoint_save_kind(config, global_step=step, steps_per_epoch=10) == "step" for step in saves)


def test_rolling_publication_replaces_old_target_and_preserves_durable(trainer, tmp_path):
    durable_epoch_1 = tmp_path / "100"
    durable_epoch_1.mkdir()
    rolling_1 = tmp_path / ".rolling_step_000000000140"
    rolling_1.mkdir()
    trainer._publish_rolling_checkpoint(tmp_path, rolling_1)

    latest = trainer._latest_step_dir(tmp_path)
    assert latest == (140, tmp_path / "rolling_latest")
    assert (tmp_path / "rolling_latest").is_symlink()

    rolling_2 = tmp_path / ".rolling_step_000000000180"
    rolling_2.mkdir()
    (tmp_path / "tmp_rolling_000000000179").mkdir()
    trainer._publish_rolling_checkpoint(tmp_path, rolling_2)

    assert not rolling_1.exists()
    assert not (tmp_path / "tmp_rolling_000000000179").exists()
    assert durable_epoch_1.is_dir()
    assert trainer._rolling_checkpoint_target(tmp_path) == rolling_2.resolve()
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".rolling_step_")] == [rolling_2.name]

    durable_epoch_2 = tmp_path / "200"
    durable_epoch_2.mkdir()
    trainer._publish_rolling_checkpoint(tmp_path, durable_epoch_2)

    assert not rolling_2.exists()
    assert durable_epoch_1.is_dir()
    assert durable_epoch_2.is_dir()
    assert trainer._rolling_checkpoint_target(tmp_path) == durable_epoch_2.resolve()
    # A tie prefers the durable numeric directory rather than its rolling symlink.
    assert trainer._latest_step_dir(tmp_path) == (200, durable_epoch_2)


def test_five_epoch_filesystem_retains_five_durable_and_at_most_one_rolling(trainer, tmp_path):
    for epoch in range(1, 6):
        epoch_start = (epoch - 1) * 100
        for step in (epoch_start + 40, epoch_start + 80):
            rolling = tmp_path / f".rolling_step_{step:012d}"
            rolling.mkdir()
            trainer._publish_rolling_checkpoint(tmp_path, rolling)
            rolling_dirs = [path for path in tmp_path.iterdir() if path.name.startswith(".rolling_step_")]
            assert rolling_dirs == [rolling]

        durable = tmp_path / str(epoch * 100)
        durable.mkdir()
        trainer._publish_rolling_checkpoint(tmp_path, durable)
        assert not any(path.name.startswith(".rolling_step_") for path in tmp_path.iterdir())

    assert sorted(int(path.name) for path in tmp_path.iterdir() if path.name.isdigit()) == [
        100,
        200,
        300,
        400,
        500,
    ]
    assert trainer._latest_step_dir(tmp_path) == (500, tmp_path / "500")
    assert trainer._rolling_checkpoint_target(tmp_path) == (tmp_path / "500").resolve()


def test_resume_discovery_prefers_newer_rolling_then_newer_durable(trainer, tmp_path):
    (tmp_path / "100").mkdir()
    rolling = tmp_path / ".rolling_step_000000000140"
    rolling.mkdir()
    trainer._publish_rolling_checkpoint(tmp_path, rolling)
    assert trainer._latest_step_dir(tmp_path) == (140, tmp_path / "rolling_latest")

    (tmp_path / "200").mkdir()
    assert trainer._latest_step_dir(tmp_path) == (200, tmp_path / "200")


def _write_basic_valid_checkpoint(path, step):
    path.mkdir()
    (path / "accelerate_state").mkdir()
    trainer_metadata = {"global_step": step}
    import torch
    torch.save(trainer_metadata, path / "metadata.pt")
    (path / "manifest.json").write_text(json.dumps({"run_metadata": {"global_step": step}}))


def test_epoch_resume_prefers_published_rolling_over_stale_higher_numeric(trainer, tmp_path):
    _write_basic_valid_checkpoint(tmp_path / "10000", 10000)
    rolling = tmp_path / ".rolling_step_000000005000"
    _write_basic_valid_checkpoint(rolling, 5000)
    trainer._publish_rolling_checkpoint(tmp_path, rolling)

    assert trainer._latest_step_dir(
        tmp_path, checkpoint_policy="epoch_with_rolling"
    ) == (5000, tmp_path / "rolling_latest")


def test_epoch_resume_uses_numeric_fallback_when_no_rolling_link(trainer, tmp_path):
    durable = tmp_path / "10000"
    _write_basic_valid_checkpoint(durable, 10000)

    assert trainer._latest_step_dir(
        tmp_path, checkpoint_policy="epoch_with_rolling"
    ) == (10000, durable)


def test_epoch_resume_fails_closed_on_malformed_or_outside_rolling_link(trainer, tmp_path):
    checkpoint_root, _ = _checkpoint_root_with_durable(tmp_path)
    malformed = checkpoint_root / "not-a-checkpoint"
    malformed.mkdir()
    link = checkpoint_root / "rolling_latest"
    link.symlink_to(malformed.name, target_is_directory=True)

    with pytest.raises(ValueError, match="invalid rolling_latest"):
        trainer._latest_step_dir(checkpoint_root, checkpoint_policy="epoch_with_rolling")

    link.unlink()
    outside = tmp_path / ".rolling_step_000000000999"
    outside.mkdir()
    link.symlink_to(outside.resolve(), target_is_directory=True)
    with pytest.raises(ValueError, match="invalid rolling_latest"):
        trainer._latest_step_dir(checkpoint_root, checkpoint_policy="epoch_with_rolling")


def _checkpoint_root_with_durable(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    durable = checkpoint_root / "100"
    durable.mkdir()
    return checkpoint_root, durable


def test_resume_discovery_selects_valid_in_root_relative_rolling_link(trainer, tmp_path):
    checkpoint_root, _ = _checkpoint_root_with_durable(tmp_path)
    rolling = checkpoint_root / ".rolling_step_000000000140"
    rolling.mkdir()
    link = checkpoint_root / "rolling_latest"
    link.symlink_to(rolling.name, target_is_directory=True)

    assert trainer._rolling_checkpoint_target(checkpoint_root) == rolling.resolve()
    assert trainer._latest_step_dir(checkpoint_root) == (140, link)


def test_resume_discovery_rejects_outside_root_absolute_rolling_link(trainer, tmp_path):
    checkpoint_root, durable = _checkpoint_root_with_durable(tmp_path)
    outside = tmp_path / ".rolling_step_000000000999"
    outside.mkdir()
    (checkpoint_root / "rolling_latest").symlink_to(outside.resolve(), target_is_directory=True)

    assert trainer._rolling_checkpoint_target(checkpoint_root) is None
    assert trainer._latest_step_dir(checkpoint_root) == (100, durable)


def test_resume_discovery_rejects_parent_escape_rolling_link(trainer, tmp_path):
    checkpoint_root, durable = _checkpoint_root_with_durable(tmp_path)
    outside = tmp_path / ".rolling_step_000000000999"
    outside.mkdir()
    (checkpoint_root / "rolling_latest").symlink_to(
        f"../{outside.name}",
        target_is_directory=True,
    )

    assert trainer._rolling_checkpoint_target(checkpoint_root) is None
    assert trainer._latest_step_dir(checkpoint_root) == (100, durable)


def test_resume_discovery_rejects_broken_rolling_link_and_falls_back_to_durable(trainer, tmp_path):
    checkpoint_root, durable = _checkpoint_root_with_durable(tmp_path)
    (checkpoint_root / "rolling_latest").symlink_to(
        ".rolling_step_000000000999",
        target_is_directory=True,
    )

    assert trainer._rolling_checkpoint_target(checkpoint_root) is None
    assert trainer._latest_step_dir(checkpoint_root) == (100, durable)


def test_resume_discovery_rejects_malformed_in_root_rolling_target(trainer, tmp_path):
    checkpoint_root, durable = _checkpoint_root_with_durable(tmp_path)
    malformed = checkpoint_root / "not-a-checkpoint"
    malformed.mkdir()
    (malformed / "manifest.json").write_text('{"run_metadata": {"global_step": 999}}')
    (checkpoint_root / "rolling_latest").symlink_to(malformed.name, target_is_directory=True)

    assert trainer._rolling_checkpoint_target(checkpoint_root) is None
    assert trainer._latest_step_dir(checkpoint_root) == (100, durable)


def test_non_main_rank_participates_in_collective_state_save(trainer, tmp_path, monkeypatch):
    config = _epoch_config(total_steps=500, rolling_interval=40)
    config.checkpoint_dir = tmp_path
    expected_tmp = tmp_path / "tmp_rolling_000000000040"
    expected_tmp.mkdir()

    class NonMainAccelerator:
        is_main_process = False

        def __init__(self):
            self.wait_calls = 0
            self.saved_state_paths = []

        def wait_for_everyone(self):
            self.wait_calls += 1

        def save_state(self, path):
            self.saved_state_paths.append(Path(path))

    accelerator = NonMainAccelerator()
    monkeypatch.setenv("OPENPI_SAVE_ACCELERATE_STATE", "1")
    trainer.save_checkpoint(
        accelerator=accelerator,
        model=None,
        optimizer=None,
        global_step=40,
        config=config,
        data_config=None,
        steps_per_epoch=100,
    )

    assert accelerator.saved_state_paths == [expected_tmp / "accelerate_state"]
    assert accelerator.wait_calls == 3
    assert not (tmp_path / ".rolling_step_000000000040").exists()
    assert not (tmp_path / "rolling_latest").exists()


def test_formal_state_save_failure_preserves_published_checkpoint(trainer, tmp_path, monkeypatch):
    from dataclasses import dataclass

    old = tmp_path / ".rolling_step_000000000020"
    old.mkdir()
    trainer._publish_rolling_checkpoint(tmp_path, old)

    @dataclass
    class Config:
        checkpoint_dir: Path
        name: str = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16"
        checkpoint_policy: str = "epoch_with_rolling"
        rolling_checkpoint_interval: int = 40
        num_train_steps: int = 100
        save_interval: int = 0
        pytorch_training_precision: str = "bfloat16"
        wandb_enabled: bool = False

    class Accelerator:
        is_main_process = True
        distributed_type = "DEEPSPEED"
        num_processes = 32

        def wait_for_everyone(self):
            return None

        def save_state(self, _path):
            raise RuntimeError("state save failed")

    monkeypatch.setenv("OPENPI_SAVE_ACCELERATE_STATE", "1")
    with pytest.raises(RuntimeError, match="state save failed"):
        trainer.save_checkpoint(
            accelerator=Accelerator(),
            model=object(),
            optimizer=None,
            global_step=40,
            config=Config(checkpoint_dir=tmp_path),
            data_config=SimpleNamespace(norm_stats=None, asset_id=None),
            steps_per_epoch=100,
        )

    assert trainer._rolling_checkpoint_target(tmp_path) == old.resolve()
    assert not (tmp_path / ".rolling_step_000000000040").exists()


def test_formal_manifest_failure_preserves_published_checkpoint(trainer, tmp_path, monkeypatch):
    from dataclasses import dataclass

    old = tmp_path / ".rolling_step_000000000020"
    old.mkdir()
    trainer._publish_rolling_checkpoint(tmp_path, old)

    @dataclass
    class Config:
        checkpoint_dir: Path
        name: str = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16"
        checkpoint_policy: str = "epoch_with_rolling"
        rolling_checkpoint_interval: int = 40
        num_train_steps: int = 100
        save_interval: int = 0
        pytorch_training_precision: str = "bfloat16"
        wandb_enabled: bool = False

    class Accelerator:
        is_main_process = True
        distributed_type = "DEEPSPEED"
        num_processes = 32

        def wait_for_everyone(self):
            return None

        def save_state(self, path):
            state = Path(path)
            state.mkdir(parents=True)
            (state / "pytorch_model").mkdir()
            (state / "pytorch_model/mp_rank_00_model_states.pt").touch()
            for rank in range(self.num_processes):
                (state / f"rank{rank:02d}_optim_states.pt").touch()
                (state / f"random_states_{rank}.pkl").touch()

        def unwrap_model(self, model):
            return model

        def get_state_dict(self, _model):
            return {}

    monkeypatch.setattr(
        trainer.safetensors.torch,
        "save_file",
        lambda _state, path: Path(path).write_bytes(b"model"),
    )
    monkeypatch.setattr(
        trainer,
        "_build_checkpoint_manifest",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("manifest failed")),
    )
    monkeypatch.setenv("OPENPI_SAVE_ACCELERATE_STATE", "1")
    with pytest.raises(RuntimeError, match="manifest failed"):
        trainer.save_checkpoint(
            accelerator=Accelerator(),
            model=object(),
            optimizer=None,
            global_step=40,
            config=Config(checkpoint_dir=tmp_path),
            data_config=SimpleNamespace(norm_stats=None, asset_id=None),
            steps_per_epoch=100,
        )

    assert trainer._rolling_checkpoint_target(tmp_path) == old.resolve()
    assert not (tmp_path / ".rolling_step_000000000040").exists()


def test_main_rank_writes_rolling_then_durable_epoch_metadata(trainer, tmp_path, monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class Config:
        checkpoint_dir: Path
        checkpoint_policy: str = "epoch_with_rolling"
        rolling_checkpoint_interval: int = 40
        num_train_steps: int = 100
        save_interval: int = 0
        pytorch_training_precision: str = "bfloat16"
        wandb_enabled: bool = False

    class MainAccelerator:
        is_main_process = True
        distributed_type = "DEEPSPEED"
        num_processes = 8

        def __init__(self):
            self.wait_calls = 0

        def wait_for_everyone(self):
            self.wait_calls += 1

        def save_state(self, path):
            Path(path).mkdir(parents=True)

        def unwrap_model(self, model):
            return model

        def get_state_dict(self, model):
            return {}

    accelerator = MainAccelerator()
    config = Config(checkpoint_dir=tmp_path)
    data_config = SimpleNamespace(norm_stats=None, asset_id=None)
    monkeypatch.setattr(
        trainer.safetensors.torch,
        "save_file",
        lambda _state, path: Path(path).write_bytes(b"model"),
    )
    monkeypatch.setattr(
        trainer,
        "_build_checkpoint_manifest",
        lambda **kwargs: {
            "run_metadata": {
                "global_step": kwargs["global_step"],
                "checkpoint_kind": kwargs["checkpoint_kind"],
                "epoch": kwargs["checkpoint_epoch"],
            }
        },
    )

    trainer.save_checkpoint(
        accelerator=accelerator,
        model=object(),
        optimizer=None,
        global_step=40,
        config=config,
        data_config=data_config,
        steps_per_epoch=100,
    )
    rolling_target = tmp_path / ".rolling_step_000000000040"
    rolling_metadata = trainer.torch.load(
        rolling_target / "metadata.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert rolling_metadata["checkpoint_kind"] == "rolling"
    assert rolling_metadata["epoch"] == 1
    assert trainer._latest_step_dir(tmp_path) == (40, tmp_path / "rolling_latest")

    trainer.save_checkpoint(
        accelerator=accelerator,
        model=object(),
        optimizer=None,
        global_step=100,
        config=config,
        data_config=data_config,
        steps_per_epoch=100,
    )
    durable_metadata = trainer.torch.load(
        tmp_path / "100/metadata.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert durable_metadata["checkpoint_kind"] == "epoch"
    assert durable_metadata["epoch"] == 1
    assert not rolling_target.exists()
    assert trainer._rolling_checkpoint_target(tmp_path) == (tmp_path / "100").resolve()
    assert trainer._latest_step_dir(tmp_path) == (100, tmp_path / "100")
    assert accelerator.wait_calls == 6


def test_epoch_policy_rejects_invalid_rolling_interval(trainer):
    config = _epoch_config(total_steps=500, rolling_interval=0)
    with pytest.raises(ValueError, match="rolling-checkpoint-interval"):
        trainer._checkpoint_save_kind(config, global_step=1, steps_per_epoch=100)


def test_build_datasets_retries_only_canonical_retryable_cache_error(trainer, monkeypatch):
    from behavior.learning.datas.hf_cache_sync import DistributedCacheError

    calls = 0
    loader = SimpleNamespace(data_config=lambda: "data-config")

    def create_data_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DistributedCacheError(
                "canonical retry",
                retryable=True,
                request_id="a" * 24,
                generation_id="b" * 24,
                origin_rank=0,
                error_type="FileNotFoundError",
                error_text="transient filelock ENOENT",
            )
        return loader

    monkeypatch.setattr(trainer, "_data_loader", SimpleNamespace(create_data_loader=create_data_loader))
    monkeypatch.setenv("OPENPI_BUILD_DATASET_RETRIES", "2")
    monkeypatch.setenv("OPENPI_BUILD_DATASET_RETRY_SLEEP_S", "0")
    assert trainer.build_datasets(SimpleNamespace()) == (loader, "data-config")
    assert calls == 2


def test_build_datasets_aborts_canonical_nonretryable_cache_error(trainer, monkeypatch):
    from behavior.learning.datas.hf_cache_sync import DistributedCacheError

    error = DistributedCacheError(
        "canonical abort",
        retryable=False,
        request_id="a" * 24,
        generation_id="c" * 24,
        origin_rank=0,
        error_type="ValueError",
        error_text="bad schema",
    )
    calls = 0

    def create_data_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(trainer, "_data_loader", SimpleNamespace(create_data_loader=create_data_loader))
    monkeypatch.setenv("OPENPI_BUILD_DATASET_RETRIES", "3")
    monkeypatch.setenv("OPENPI_BUILD_DATASET_RETRY_SLEEP_S", "0")
    with pytest.raises(DistributedCacheError, match="canonical abort"):
        trainer.build_datasets(SimpleNamespace())
    assert calls == 1
