"""Behavioral contracts for TrainConfig-driven launcher profiles."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from accelerate.utils import DistributedType
import pytest

from openpi.training.launcher_profile import materialize_effective_recipe
from openpi.training.launcher_profile import resolve_effective_recipe
from openpi.training.launcher_profile import resolve_launcher_profile
from openpi.training.train_config import cli
from openpi.training.train_config import get_config


V100_FAST = "pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32"
V100_QUERY = "pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32"
V100_VAL10 = "pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32_validation10"
H20_A = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16"
H20_B = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16"
BF16_FAST = "pi05_ki_joint_fast_b1k-full_task-ki_on_bf16"
BF16_QUERY = "pi05_ki_joint_query_b1k-full_task-ki_on_bf16"
FORMAL_PROFILES = (
    BF16_FAST,
    BF16_QUERY,
    V100_FAST,
    V100_VAL10,
    V100_QUERY,
    H20_A,
    H20_B,
)


def _load_trainer():
    path = Path(__file__).resolve().parents[1] / "scripts/train_accelerate.py"
    spec = importlib.util.spec_from_file_location("_launcher_profile_trainer", path)
    assert spec and spec.loader
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def _accelerator_for(cfg):
    state = SimpleNamespace(deepspeed_plugin=None)
    distributed_type = None
    if cfg.pytorch_training_precision == "float32":
        distributed_type = DistributedType.DEEPSPEED
        state.deepspeed_plugin = SimpleNamespace(
            deepspeed_config={
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu"},
                }
            }
        )
    return SimpleNamespace(
        num_processes=32,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        distributed_type=distributed_type,
        state=state,
    )


@pytest.mark.parametrize(("name", "batch", "ga", "precision", "mixed", "expected_gb"), [
    (V100_FAST, 1, 8, "float32", "no", 256),
    (V100_QUERY, 1, 8, "float32", "no", 256),
    (V100_VAL10, 1, 8, "float32", "no", 256),
    (H20_A, 32, 1, "bfloat16", "bf16", 1024),
    (H20_B, 32, 1, "bfloat16", "bf16", 1024),
])
def test_profile_matrix_resolves_registered_recipe(name, batch, ga, precision, mixed, expected_gb):
    profile = resolve_launcher_profile(name)
    assert profile.name == name
    assert profile.batch_size_per_gpu == batch
    assert profile.gradient_accumulation_steps == ga
    assert profile.pytorch_training_precision == precision
    assert profile.accelerate_mixed_precision == mixed
    assert profile.expected_global_batch == expected_gb


def test_v100_formal_pair_matches_outside_objective_and_output_identity():
    a = get_config(V100_FAST)
    b = get_config(V100_QUERY)
    exclusions = {"name", "exp_name", "model", "pytorch_model_name", "assets_base_dir", "checkpoint_base_dir", "log_base_dir"}
    for field in dataclasses.fields(a):
        if field.name not in exclusions:
            assert getattr(a, field.name) == getattr(b, field.name), field.name
    assert a.val_log_interval == b.val_log_interval == 1_000
    assert a.num_workers == b.num_workers == 2


def test_v100_validation10_differs_from_formal_fast_only_by_identity_outputs_and_cadence():
    formal = get_config(V100_FAST)
    validation = get_config(V100_VAL10)
    exclusions = {"name", "exp_name", "assets_base_dir", "checkpoint_base_dir", "log_base_dir", "val_log_interval"}
    for field in dataclasses.fields(formal):
        if field.name not in exclusions:
            assert getattr(formal, field.name) == getattr(validation, field.name), field.name
    assert formal.val_log_interval == 1_000
    assert validation.val_log_interval == 10
    assert validation.val_interval_samples is None
    assert validation.num_workers == 2


def test_unknown_profile_bites_instead_of_accepting_get_config_fallback():
    with pytest.raises(ValueError, match="refusing silent fallback"):
        resolve_launcher_profile("typo-does-not-exist")


def test_expected_model_mismatch_fails_closed():
    with pytest.raises(ValueError, match="resolved model"):
        resolve_launcher_profile(V100_VAL10, expected_model="pi05_ki_joint_query")


def test_expected_global_batch_equality_bites_wrong_world_size():
    with pytest.raises(ValueError, match="requires global batch 256"):
        resolve_launcher_profile(V100_VAL10, world_size=16)
    with pytest.raises(ValueError, match="requires global batch 1024"):
        resolve_launcher_profile(H20_A, world_size=16)


def test_trainer_rejects_direct_registered_profile_recipe_override():
    trainer = _load_trainer()
    mutated = dataclasses.replace(get_config(V100_VAL10), batch_size_per_gpu=2)
    with pytest.raises(ValueError, match="effective batch_size_per_gpu=1"):
        trainer._validate_formal_b1k_contract(mutated)
    mutated = dataclasses.replace(get_config(V100_VAL10), val_log_interval=11)
    with pytest.raises(ValueError, match="effective val_log_interval=10"):
        trainer._validate_formal_b1k_contract(mutated)


def test_h20_profiles_preserve_intended_epoch_recipe_batch_and_effective_cadence():
    for name in (H20_A, H20_B):
        profile = resolve_launcher_profile(name, expected_model="pi05_ki_joint_fast")
        assert profile.batch_size_per_gpu == 32
        assert profile.gradient_accumulation_steps == 1
        assert profile.expected_global_batch == 1024
        assert profile.effective_val_log_interval == 250
        assert profile.effective_save_interval == 2_500
        assert profile.num_train_epochs == 1
        assert profile.num_train_steps == 0
        assert profile.decay_steps == 0
        assert profile.streaming_anchor_stride == 4
        assert profile.action_token_max_len == 256
        assert profile.val_interval_samples is not None
        assert profile.save_interval_samples is not None


@pytest.mark.parametrize("name", FORMAL_PROFILES)
def test_every_formal_registered_profile_has_explicit_matching_global_batch(name):
    cfg = get_config(name)
    assert cfg.name == name
    assert cfg.expected_global_batch is not None
    assert (
        cfg.batch_size_per_gpu * 32 * cfg.gradient_accumulation_steps
        == cfg.expected_global_batch
    )
    effective = resolve_effective_recipe(cfg, world_size=32)
    assert effective.global_batch == cfg.expected_global_batch
    materialized = materialize_effective_recipe(cfg, world_size=32)
    _load_trainer()._validate_formal_b1k_contract(
        materialized, accelerator=_accelerator_for(materialized)
    )


@pytest.mark.parametrize("name", (H20_A, H20_B))
def test_h20_exact_tyro_argv_parses_raw_then_validates_shared_effective_recipe(
    name, monkeypatch
):
    trainer = _load_trainer()
    raw = get_config(name)
    argv = [
        "train_accelerate.py",
        name,
        "--pytorch-weight-path",
        str(raw.pytorch_weight_path),
        "--exp-name",
        "tyro-contract",
        "--pytorch-training-precision",
        raw.pytorch_training_precision,
        "--num-train-steps",
        str(raw.num_train_steps),
        "--num-train-epochs",
        str(raw.num_train_epochs),
        "--batch-size-per-gpu",
        str(raw.batch_size_per_gpu),
        "--gradient-accumulation-steps",
        str(raw.gradient_accumulation_steps),
        "--num-workers",
        str(raw.num_workers),
        "--save-interval",
        str(raw.save_interval),
        "--val-log-interval",
        str(raw.val_log_interval),
        "--val-num-batches",
        str(raw.val_num_batches),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    parsed = cli()
    assert parsed.save_interval == 10_000
    assert parsed.val_log_interval == 1_000
    trainer._validate_runtime_config(parsed)

    effective = materialize_effective_recipe(parsed, world_size=32)
    assert effective.save_interval == 2_500
    assert effective.val_log_interval == 250
    trainer._validate_formal_b1k_contract(
        effective, accelerator=_accelerator_for(effective)
    )


@pytest.mark.parametrize("name", (H20_A, H20_B))
def test_h20_effective_recipe_mutation_is_rejected(name):
    trainer = _load_trainer()
    effective = materialize_effective_recipe(get_config(name), world_size=32)
    mutated = dataclasses.replace(effective, val_log_interval=251)
    with pytest.raises(ValueError, match="effective val_log_interval=250"):
        trainer._validate_formal_b1k_contract(
            mutated, accelerator=_accelerator_for(mutated)
        )
