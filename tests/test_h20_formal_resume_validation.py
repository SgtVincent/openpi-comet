# ruff: noqa: SLF001 - focused contract tests intentionally exercise private helpers.
"""Focused CPU tests for H20 engineering resume and formal validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
_TRAINER_PATH = _REPO_ROOT / "scripts/train_accelerate.py"
_H20_NAME = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16"


def _load_train_accelerate():
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

    module_name = "_openpi_h20_formal_resume_validation_test"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, _TRAINER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load_train_accelerate()


class _SyntheticFormalRawDataset:
    def __init__(self, *, tasks: int = 50, episodes_per_task: int = 20, episode_len: int = 32):
        self.episodes = [
            task * 10_000 + episode
            for task in range(tasks)
            for episode in range(180, 180 + episodes_per_task)
        ]
        starts = [index * episode_len for index in range(len(self.episodes))]
        self.episode_data_index = {
            "from": torch.tensor(starts),
            "to": torch.tensor([start + episode_len for start in starts]),
        }

    def __len__(self):
        return int(self.episode_data_index["to"][-1]) if self.episodes else 0


def _formal_val_config():
    return SimpleNamespace(
        name=_H20_NAME,
        val_subset_seed=12345,
        val_episodes_per_task=20,
        val_anchors_per_episode=4,
        val_batch_size=8,
        val_num_batches=16,
    )


@pytest.mark.parametrize(
    ("values", "target", "expected"),
    [
        ([0, 1, 2], 3, [0, 1, 2]),
        ([0, 1, 2], 5, [0, 1, 2, 0, 1]),
        ([0, 1], 7, [0, 1, 0, 1, 0, 1, 0]),
    ],
)
def test_cyclic_padding_handles_padding_larger_than_source(trainer, values, target, expected):
    assert trainer._cyclic_pad_to_length(values, target) == expected


def test_formal_validation_builds_4000_distinct_interior_anchors(trainer):
    raw = _SyntheticFormalRawDataset()
    indices, task_ids, coverage = trainer._build_stratified_val_indices(raw, _formal_val_config())

    assert len(indices) == len(set(indices)) == 4_000
    assert len(task_ids) == 4_000
    assert coverage["n_raw_episodes"] == 1_000
    assert coverage["n_episodes"] == 1_000
    assert coverage["n_tasks"] == 50
    assert coverage["n_unique_anchors"] == 4_000
    assert {task_ids.count(task) for task in range(50)} == {80}
    for index in indices:
        assert index % 32 not in {0, 31}

    replay = trainer._build_stratified_val_indices(raw, _formal_val_config())
    assert replay == (indices, task_ids, coverage)


@pytest.mark.parametrize(
    "raw",
    [
        _SyntheticFormalRawDataset(tasks=49),
        _SyntheticFormalRawDataset(episodes_per_task=19),
        _SyntheticFormalRawDataset(episode_len=5),
    ],
)
def test_formal_validation_fails_closed_on_incomplete_population(trainer, raw):
    with pytest.raises(ValueError, match="Formal H20 validation"):
        trainer._build_stratified_val_indices(raw, _formal_val_config())


def test_task_homogeneous_plan_executes_4096_and_marks_96_padding(trainer):
    raw = _SyntheticFormalRawDataset()
    indices, task_ids, _ = trainer._build_stratified_val_indices(raw, _formal_val_config())
    plan = trainer._build_task_homogeneous_batch_plan(
        indices,
        task_ids,
        batch_size=8,
        world_size=32,
    )

    assert len(plan.batches) == 512
    assert plan.batches_per_rank == 16
    assert sum(plan.unique_counts) == 4_000
    assert sum(8 - count for count in plan.unique_counts) == 96
    assert plan.unique_counts.count(0) == 12
    assert all(len(batch) == 8 for batch in plan.batches)
    assert all(len({task_ids[position] for position in batch}) == 1 for batch in plan.batches)


def test_validation_metric_totals_exclude_padding_and_mixed_task_attribution(trainer):
    batch_metrics = [
        {"flow_mse": 1.0, "total_loss": 2.0},
        {"flow_mse": 3.0, "total_loss": 4.0},
        {"flow_mse": 999.0, "total_loss": 999.0},
    ]
    batch_task_ids = [[0] * 8, [1] * 8, [0] * 8]
    unique_counts = [8, 8, 0]

    metric_totals, task_totals = trainer._validation_metric_totals(
        batch_metrics,
        batch_task_ids=batch_task_ids,
        batch_unique_counts=unique_counts,
        per_task_metric="flow_mse",
    )

    assert metric_totals == {"flow_mse": [32.0, 16], "total_loss": [48.0, 16]}
    assert task_totals == {0: [8.0, 8], 1: [24.0, 8]}

    _, mixed_task_totals = trainer._validation_metric_totals(
        [{"flow_mse": 5.0}],
        batch_task_ids=[[0, 1]],
        batch_unique_counts=[2],
        per_task_metric="flow_mse",
    )
    assert mixed_task_totals == {}


def test_formal_metric_totals_reject_missing_or_nonfinite_core_metrics(trainer):
    batch_metrics = [{"total_loss": 1.0, "flow_mse": float("nan")}]
    totals, tasks = trainer._validation_metric_totals(
        batch_metrics,
        batch_task_ids=[[0] * 8],
        batch_unique_counts=[8],
        per_task_metric="flow_mse",
    )
    assert totals == {"total_loss": [8.0, 8]}
    assert tasks == {}


def test_formal_task_totals_expose_short_task_count(trainer):
    metrics = [{"total_loss": 1.0, "flow_mse": 2.0}]
    totals, tasks = trainer._validation_metric_totals(
        metrics,
        batch_task_ids=[[7] * 8],
        batch_unique_counts=[8],
        per_task_metric="flow_mse",
    )
    assert totals["flow_mse"] == [16.0, 8]
    assert tasks[7] == [16.0, 8]
    assert tasks[7][1] != 80


def test_formal_validation_manifest_fields_are_truthful(trainer):
    coverage = {
        "n_raw_episodes": 1_000,
        "n_episodes": 1_000,
        "n_tasks": 50,
        "n_unique_anchors": 4_000,
        "n_samples": 4_000,
        "n_padded": 4_096,
        "n_duplicated": 96,
        "n_batches_per_rank": 16,
        "val_global_batch": 256,
    }
    fields = trainer._validation_manifest_fields(
        SimpleNamespace(coverage=coverage), world_size=32
    )
    assert fields == {
        "raw_episodes": 1_000,
        "unique_anchors": 4_000,
        "duplicate_anchors": 96,
        "executed_anchors": 4_096,
        "batches_per_rank": 16,
        "global_batch": 256,
    }


def test_sample_progress_counts_only_committed_updates_and_resumes_with_changed_batch(trainer):
    progress = trainer._SampleProgress.fresh(samples_per_update=1_024, epoch_num_samples=26_857_472)
    assert progress.metrics() == {
        "dataset/accum_num_samples": 0,
        "dataset/epoch_fraction": 0.0,
        "dataset/samples_per_update": 1_024,
        "dataset/epoch_num_samples": 26_857_472,
    }

    progress.record_update(committed=True)
    assert progress.accum_num_samples == 1_024  # first legacy logged step=0
    progress.record_update(committed=False)  # overflow/non-finite skips do not count
    assert progress.accum_num_samples == 1_024
    progress.record_update(committed=True)  # two-phase backward still calls this once
    assert progress.accum_num_samples == 2_048

    restored = trainer._restore_sample_progress(
        {
            "global_step": 2,
            "sample_progress": progress.checkpoint_payload(),
            "config": {"batch_size_per_gpu": 32, "gradient_accumulation_steps": 1},
            "accelerate": {"num_processes": 32},
        },
        current_samples_per_update=2_048,
        default_epoch_num_samples=26_857_472,
    )
    assert restored.accum_num_samples == 2_048
    assert restored.samples_per_update == 2_048
    assert restored.epoch_num_samples == 26_857_472
    restored.record_update(committed=True)
    assert restored.accum_num_samples == 4_096


def test_legacy_expected_global_batch_is_not_multiplied_by_ga(trainer):
    restored = trainer._restore_sample_progress(
        {
            "global_step": 5,
            "config": {"expected_global_batch": 256, "gradient_accumulation_steps": 2},
            "accelerate": {"num_processes": 32},
        },
        current_samples_per_update=1_024,
        default_epoch_num_samples=26_857_472,
    )
    assert restored.accum_num_samples == 5 * 256


def test_legacy_sample_progress_uses_saved_not_current_global_batch(trainer):
    restored = trainer._restore_sample_progress(
        {
            "global_step": 5,
            "config": {"batch_size_per_gpu": 32, "gradient_accumulation_steps": 1},
            "accelerate": {"num_processes": 32},
        },
        current_samples_per_update=2_048,
        default_epoch_num_samples=26_857_472,
    )
    assert restored.accum_num_samples == 5 * 1_024
    assert restored.samples_per_update == 2_048


def _write_resume_checkpoint(root: Path, *, step: int, metadata_step: int | None = None):
    target = root / f".rolling_step_{step:012d}"
    state = target / "accelerate_state"
    state.mkdir(parents=True)
    (state / "pytorch_model").mkdir()
    (state / "pytorch_model/mp_rank_00_model_states.pt").touch()
    for rank in range(32):
        (state / f"rank{rank:02d}_optim_states.pt").touch()
        (state / f"random_states_{rank}.pkl").touch()
    metadata = {
        "global_step": step if metadata_step is None else metadata_step,
        "config": {"batch_size_per_gpu": 32, "gradient_accumulation_steps": 1},
        "accelerate": {"num_processes": 32},
    }
    torch.save(metadata, target / "metadata.pt")
    (target / "manifest.json").write_text(json.dumps({"run_metadata": {"global_step": step}}))
    (root / "rolling_latest").symlink_to(target.name, target_is_directory=True)
    return target


def test_resume_loads_selected_rolling_state_and_logs_engineering_semantics(trainer, tmp_path, caplog):
    target = _write_resume_checkpoint(tmp_path, step=5_000)

    class Accelerator:
        loaded = None

        def load_state(self, path):
            self.loaded = Path(path)

    accelerator = Accelerator()
    with caplog.at_level("INFO"):
        step, metadata, selected = trainer._load_resume_state(
            accelerator,
            checkpoint_dir=tmp_path,
            checkpoint_policy="epoch_with_rolling",
            formal=True,
        )

    assert step == metadata["global_step"] == 5_000
    assert selected == tmp_path / "rolling_latest"
    assert accelerator.loaded.resolve() == (target / "accelerate_state").resolve()
    assert "optimizer resume; data order restart" in caplog.text
    assert trainer._cosine_lr_value(
        step,
        warmup_steps=250,
        peak_lr=2e-5,
        decay_steps=26_293,
        end_lr=0.0,
    ) == pytest.approx(trainer._cosine_lr_value(
        5_000,
        warmup_steps=250,
        peak_lr=2e-5,
        decay_steps=26_293,
        end_lr=0.0,
    ))


def test_resume_rejects_name_manifest_metadata_step_mismatch(trainer, tmp_path):
    _write_resume_checkpoint(tmp_path, step=5_000, metadata_step=4_999)

    class Accelerator:
        def load_state(self, path):  # pragma: no cover - must fail before load
            raise AssertionError(path)

    with pytest.raises(ValueError, match="checkpoint step mismatch|invalid rolling_latest"):
        trainer._load_resume_state(
            Accelerator(),
            checkpoint_dir=tmp_path,
            checkpoint_policy="epoch_with_rolling",
            formal=True,
        )


def test_legacy_validation_totals_without_metadata_preserve_batch_mean(trainer):
    totals, tasks = trainer._validation_metric_totals(
        [{"loss": 2.0}, {"loss": 4.0}],
        batch_task_ids=[[], []],
        batch_unique_counts=[1, 1],
        per_task_metric="loss",
    )
    assert totals == {"loss": [6.0, 2]}
    assert tasks == {}


def test_formal_wandb_resume_uses_existing_id_with_must_semantics(trainer, monkeypatch):
    calls = []
    monkeypatch.setattr(
        trainer,
        "_init_wandb_run",
        lambda config, *, resuming: calls.append((config, resuming)) or object(),
    )
    monkeypatch.setattr(trainer.torch.distributed, "is_available", lambda: False)
    monkeypatch.setattr(trainer.torch.distributed, "is_initialized", lambda: False)
    config = SimpleNamespace(
        name=_H20_NAME,
        prepare_hf_cache_only=False,
        resume=True,
    )
    accelerator = SimpleNamespace(is_main_process=True, num_processes=1)

    trainer._init_formal_b1k_wandb(config, accelerator=accelerator, resuming=True)
    assert calls == [(config, True)]
