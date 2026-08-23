"""Contracts for the bounded 1x8 A100 optimizer-offload experiment."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

_PRODUCTION_CONFIG = "pi05_ki_joint_query_b1k-full_task-ki_on_a100_bf16"
_SHORT_CONFIG = "pi05_ki_joint_query_b1k-full_task-ki_on_a100_bf16_offload_short"
_LAUNCHER = _REPO_ROOT / "scripts/run_pi05_ki_B_a100_bf16_optimizer_offload_short_1x8.sh"
_TRAINER = _REPO_ROOT / "scripts/train_accelerate.py"
_BASELINE_DS = _REPO_ROOT / "configs/deepspeed_zero2_a100_bf16.json"
_ON_DS = _REPO_ROOT / "configs/deepspeed_zero2_a100_bf16_offload_on_short.json"
_OFF_DS = _REPO_ROOT / "configs/deepspeed_zero2_a100_bf16_offload_off_short.json"
_BASELINE_ACCEL = _REPO_ROOT / "configs/accelerate_ds_zero2_a100_bf16.yaml"
_ON_ACCEL = _REPO_ROOT / "configs/accelerate_ds_zero2_a100_bf16_offload_on_short.yaml"
_OFF_ACCEL = _REPO_ROOT / "configs/accelerate_ds_zero2_a100_bf16_offload_off_short.yaml"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _expected_deepspeed_arm(mode: str) -> dict:
    expected = copy.deepcopy(_load_json(_BASELINE_DS))
    if mode == "off":
        del expected["zero_optimization"]["offload_optimizer"]
    elif mode != "on":
        raise ValueError(f"unexpected mode: {mode}")
    return expected


def _assert_deepspeed_arm(mode: str, actual: dict) -> None:
    assert actual == _expected_deepspeed_arm(mode)
    zero = actual["zero_optimization"]
    if mode == "on":
        assert zero["offload_optimizer"] == {"device": "cpu", "pin_memory": True}
    else:
        assert "offload_optimizer" not in zero


def test_deepspeed_arms_are_exact_single_variable_transforms():
    baseline = _load_json(_BASELINE_DS)
    on = _load_json(_ON_DS)
    off = _load_json(_OFF_DS)

    _assert_deepspeed_arm("on", on)
    _assert_deepspeed_arm("off", off)
    assert on == baseline

    expected_off = copy.deepcopy(on)
    removed = expected_off["zero_optimization"].pop("offload_optimizer")
    assert removed == {"device": "cpu", "pin_memory": True}
    assert off == expected_off


def test_deepspeed_contract_negative_oracle_rejects_extra_drift():
    malformed = _expected_deepspeed_arm("off")
    malformed["zero_optimization"]["reduce_bucket_size"] = 1
    with pytest.raises(AssertionError):
        _assert_deepspeed_arm("off", malformed)

    malformed = _expected_deepspeed_arm("off")
    malformed["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    with pytest.raises(AssertionError):
        _assert_deepspeed_arm("off", malformed)


def test_accelerate_arms_only_retarget_the_deepspeed_json():
    baseline = _BASELINE_ACCEL.read_text()
    for selected, ds_name in (
        (_ON_ACCEL, _ON_DS.name),
        (_OFF_ACCEL, _OFF_DS.name),
    ):
        expected = baseline.replace(
            "configs/deepspeed_zero2_a100_bf16.json",
            f"configs/{ds_name}",
        )
        assert selected.read_text() == expected
        assert "num_machines: 1" in expected
        assert "num_processes: 8" in expected
        assert not re.search(r"^mixed_precision\s*:", expected, re.MULTILINE)


def test_short_config_is_production_b_with_exact_bounded_deltas():
    from openpi.training.train_config import get_config

    production = get_config(_PRODUCTION_CONFIG)
    short = get_config(_SHORT_CONFIG)
    assert production.name == _PRODUCTION_CONFIG
    assert short.name == _SHORT_CONFIG

    changed = {
        field.name
        for field in dataclasses.fields(production)
        if getattr(short, field.name) != getattr(production, field.name)
    }
    assert changed == {"name", "exp_name", "num_train_steps", "log_interval", "val_data"}

    assert short.pytorch_model_name == production.pytorch_model_name == "pi05_ki_joint_query"
    assert short.model == production.model
    assert short.data == production.data
    assert short.seed == production.seed == 42
    assert short.batch_size_per_gpu == production.batch_size_per_gpu == 4
    assert short.gradient_accumulation_steps == production.gradient_accumulation_steps == 2
    assert short.pytorch_training_precision == production.pytorch_training_precision == "bfloat16"
    assert short.accelerate_mixed_precision == production.accelerate_mixed_precision == "bf16"
    assert short.num_train_steps == 100
    assert short.log_interval == 1
    assert short.val_data == []
    assert production.val_data


def test_short_runtime_contract_is_b4_ga2_world8_global64():
    from openpi.training.train_config import get_config

    cfg = get_config(_SHORT_CONFIG)
    assert cfg.batch_size_per_gpu * 8 * cfg.gradient_accumulation_steps == 64
    assert cfg.num_train_steps == 100
    assert cfg.num_train_epochs == 4
    assert cfg.streaming_anchor_stride == 4
    assert cfg.epoch_anchor_offsets == [0, 1, 2, 3]
    assert cfg.save_interval == 10_000
    assert cfg.val_log_interval == 1_000
    assert cfg.log_interval == 1
    assert cfg.val_data == []


def test_launcher_locks_identical_training_contract_for_both_modes():
    source = _LAUNCHER.read_text()

    assert f'EXPECTED_CONFIG="{_SHORT_CONFIG}"' in source
    assert f'PRODUCTION_CONFIG="{_PRODUCTION_CONFIG}"' in source
    assert 'EXPECTED_MODEL="pi05_ki_joint_query"' in source
    assert 'RUN_LABEL="A100-B4GA2-offload-on-1x8-100step"' in source
    assert 'RUN_LABEL="A100-B4GA2-offload-off-1x8-100step"' in source
    assert "OPTIMIZER_OFFLOAD_MODE must be exactly 'on' or 'off'" in source

    for required in (
        'BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"',
        'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"',
        'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100}"',
        'SEED="${SEED:-42}"',
        'LOG_INTERVAL="${LOG_INTERVAL:-1}"',
        "BATCH_SIZE_PER_GPU == 4",
        "GRADIENT_ACCUMULATION_STEPS == 2",
        "MAX_TRAIN_STEPS == 100",
        "SEED == 42",
        "LOG_INTERVAL == 1",
        "GLOBAL_BATCH_SIZE == 64",
        "B4xW8xGA2",
        "--num_processes 8",
        "--num_machines 1",
        "--batch-size-per-gpu 4",
        "--gradient-accumulation-steps 2",
        '--num-train-steps "${MAX_TRAIN_STEPS}"',
        "--seed 42",
        "--log-interval 1",
        "--no-resume",
        "--no-overwrite",
    ):
        assert required in source

    assert source.count("--gradient-accumulation-steps 2") == 1
    assert source.count("--batch-size-per-gpu 4") == 1
    assert source.count("--num-train-steps") == 1
    assert source.count("--log-interval 1") == 1
    assert "KEEPALIVE_DISABLE=0 KEEPALIVE_ON_SUCCESS=0 STRICT_GPU_COUNT=0" in source
    assert '[[ "${KEEPALIVE_ON_SUCCESS:-}" == "0" ]]' in source
    assert "bounded run requires KEEPALIVE_ON_SUCCESS=0 so successful jobs release GPUs" in source
    assert "KEEPALIVE_ON_SUCCESS=1" not in source


def test_outer_wrapper_releases_success_and_holds_failure():
    wrapper = (_REPO_ROOT / "scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh").read_text()
    success_branch = (
        'if [[ "${TRAIN_RC}" -eq 0 && "${KEEPALIVE_ON_SUCCESS}" != "1" ]]; then'
    )
    failure_branch = "Training FAILED (rc=${TRAIN_RC}) — holding the allocation for manual inspection."
    heartbeat_marker = "KEEPALIVE_HEARTBEAT_LOOP_STARTED"

    assert success_branch in wrapper
    assert 'write_status "success_no_keepalive"' in wrapper
    assert "exit 0" in wrapper[wrapper.index(success_branch):wrapper.index(success_branch) + 700]
    assert failure_branch in wrapper
    assert wrapper.index(failure_branch) < wrapper.index(heartbeat_marker)


def test_both_arm_paths_use_one_config_and_disable_pollution():
    source = _LAUNCHER.read_text()
    trainer_source = _TRAINER.read_text()

    assert source.count('EXPECTED_CONFIG="') == 1
    assert 'export CONFIG_NAME="${EXPECTED_CONFIG}"' in source
    assert "cfg.val_data == []" in source
    assert "validation=disabled(empty-val-data)" in source
    assert 'OPENPI_DISABLE_CHECKPOINT="${OPENPI_DISABLE_CHECKPOINT:-1}"' in source
    assert '[[ "${OPENPI_DISABLE_CHECKPOINT}" == "1" ]]' in source
    assert "export OPENPI_DISABLE_CHECKPOINT" in source
    assert "OPENPI_DISABLE_VALIDATION" not in source
    assert "OPENPI_DISABLE_VALIDATION" not in trainer_source
    assert "MAX_TRAIN_STEPS < SAVE_INTERVAL" in source
    assert "MAX_TRAIN_STEPS < VAL_LOG_INTERVAL" in source
    assert 'mkdir -p "${CONSOLE_LOG_DIR}" "${ASSETS_BASE_DIR}" "${LOG_BASE_DIR}"' in source
    assert 'mkdir -p "${CONSOLE_LOG_DIR}" "${ASSETS_BASE_DIR}" "${CHECKPOINT_BASE_DIR}"' not in source
    # The unchanged trainer creates the checkpoint directory for W&B's fresh
    # run ID, but OPENPI_DISABLE_CHECKPOINT prevents any model checkpoint save.
    assert "OPENPI_DISABLE_CHECKPOINT" in trainer_source


def test_checkpoint_disable_existing_oracle_returns_before_policy(monkeypatch):
    spec = importlib.util.spec_from_file_location("_offload_short_train", _TRAINER)
    assert spec is not None
    assert spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    monkeypatch.setenv("OPENPI_DISABLE_CHECKPOINT", "1")
    assert trainer.save_checkpoint(
        accelerator=None,
        model=None,
        optimizer=None,
        global_step=100,
        config=None,
        data_config=None,
    ) is None


def test_launcher_uses_fresh_disjoint_identity_and_rejects_invalid_mode():
    source = _LAUNCHER.read_text()
    assert "unset WANDB_RESUME WANDB_RUN_ID" in source
    assert '[[ ! -e "${PERSISTENT_OUTPUT_ROOT}" ]]' in source
    assert 'PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_BASE}/${RUN_LABEL}/${JOB_ID_SAFE}"' in source
    assert 'EXP_NAME="${RUN_LABEL}_${JOB_ID_SAFE}"' in source
    assert 'WANDB_PROJECT="pi05_ki_a100_offload_short"' in source

    env = os.environ.copy()
    env.pop("OPTIMIZER_OFFLOAD_MODE", None)
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    assert result.returncode == 2
    assert "OPTIMIZER_OFFLOAD_MODE must be exactly 'on' or 'off'" in result.stderr


def test_launcher_self_reports_resolved_measurement_contract():
    source = _LAUNCHER.read_text()
    for report in (
        "micro_bs=4",
        "grad_accum=2",
        "world_size=8",
        "global_batch=64",
        "offload_optimizer_device=",
        "max_steps=100",
        "log_interval=1",
        "validation=disabled(empty-val-data)",
        "timing_analysis=drop_step_0_warmup",
    ):
        assert report in source


def test_changes_are_experiment_only_and_shared_trainer_is_unchanged():
    source = _LAUNCHER.read_text()
    assert "This experiment branch must" in source
    assert "never be used as a V100 or formal-production relaunch base" in source

    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "diff", "--quiet", "7019917", "--", "scripts/train_accelerate.py"],
        check=False,
    )
    assert result.returncode == 0
