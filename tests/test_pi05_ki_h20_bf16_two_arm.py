# ruff: noqa: SLF001 - contract tests intentionally inspect private state.
"""CPU contracts for the formal 4x8 NVIDIA_H20 BF16 pi0.5-KI Variant A two-arm run.

The experiment holds everything fixed except the warm-start package, so these
tests exist mainly to protect the *controlled* part of "controlled experiment":

  * both arms share one FAST action-token capacity, so capacity can never become
    a third confound alongside weights + normalization
  * each arm's weights are paired with the ``assets`` that ship beside them
  * neither arm references saiwenresearch, which the H20 pool cannot mount
  * the arms differ in exactly the fields we intend, and nothing else
  * B8 x W32 x GA1 = 256 and the formal contract table agrees with the configs
  * the launcher gates on H20 (not A100) and runs the per-device BF16 preflight
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re

import pytest

from openpi.training.train_config import get_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = _REPO_ROOT / "scripts" / "run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh"
_DS_CONFIG = _REPO_ROOT / "configs" / "deepspeed_zero2_h20_bf16.json"
_ACCEL_CONFIG = _REPO_ROOT / "configs" / "accelerate_ds_zero2_h20_bf16.yaml"

_ARM_A = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16"
_ARM_B = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16"
_ARM_A_SMOKE = f"{_ARM_A}_smoke"
_ARM_B_SMOKE = f"{_ARM_B}_smoke"
_ALL = (_ARM_A, _ARM_B, _ARM_A_SMOKE, _ARM_B_SMOKE)

_EXPECTED_CAP = 256


@pytest.mark.parametrize("name", _ALL)
def test_variant_a_shape(name: str) -> None:
    cfg = get_config(name)
    assert cfg.name == name
    assert cfg.pytorch_model_name == "pi05_ki_joint_fast"
    assert cfg.model.dtype == "bfloat16"
    assert cfg.pytorch_training_precision == "bfloat16"
    assert cfg.accelerate_mixed_precision == "bf16"
    assert cfg.model.knowledge_insulation is True
    assert cfg.model.truncate_expert_kv is True
    assert cfg.gradient_checkpointing is True
    assert cfg.wandb_enabled is True
    # A fixed-step budget: the formal B1K contract rejects an epoch budget, and
    # the smokes must not silently acquire one either.
    assert cfg.num_train_epochs is None


@pytest.mark.parametrize("name", _ALL)
def test_global_batch_is_256(name: str) -> None:
    cfg = get_config(name)
    assert cfg.batch_size_per_gpu == 8
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.batch_size_per_gpu * 32 * cfg.gradient_accumulation_steps == 256


def test_cap_is_identical_across_all_four_configs() -> None:
    """The cap must never differ between arms.

    Both arms share one value so FAST capacity cannot act as a third confound.
    This is safe precisely because padded positions carry mask/ar_mask/loss_mask
    False/0/False and the action objective divides by ``shift_loss_mask.sum()``,
    so raising the cap does not rescale CE or accuracy.
    """
    caps = {name: get_config(name).model.action_token_max_len for name in _ALL}
    assert set(caps.values()) == {_EXPECTED_CAP}, caps


@pytest.mark.parametrize("name", _ALL)
def test_no_lq_paths_anywhere(name: str) -> None:
    """H20 mounts behavior-data-hl / navigation-hl / robot-mllm-data-hl only."""
    cfg = get_config(name)
    probes = [str(cfg.pytorch_weight_path)]
    for group in (cfg.data, cfg.val_data):
        for data_cfg in group:
            probes.append(str(data_cfg.assets.assets_dir))
            probes.append(str(data_cfg.base_config.behavior_dataset_root))
    for probe in probes:
        assert "saiwenresearch" not in probe, probe


@pytest.mark.parametrize(("name", "expected_leaf"), [
    (_ARM_A, "pi05-b1kpt50-cs32"),
    (_ARM_A_SMOKE, "pi05-b1kpt50-cs32"),
    (_ARM_B, "pi05_base_pytorch"),
    (_ARM_B_SMOKE, "pi05_base_pytorch"),
])
def test_assets_come_from_the_same_package_as_the_weights(name: str, expected_leaf: str) -> None:
    """Each arm must use the norm_stats that ships with its own weights.

    This is the pairing the whole comparison rests on: the flow expert and action
    head were fit in that normalization space. Crossing them would both corrupt
    the warm start and void the measured token-length bound.
    """
    cfg = get_config(name)
    weights = str(cfg.pytorch_weight_path)
    assert weights.endswith(expected_leaf), weights
    for group in (cfg.data, cfg.val_data):
        for data_cfg in group:
            assert str(data_cfg.assets.assets_dir) == f"{weights}/assets"
            assert data_cfg.assets.asset_id == "behavior-1k/2025-challenge-demos"


def test_the_two_formal_arms_differ_only_in_the_warm_start_package() -> None:
    """Any drift beyond the intended fields turns the A/B into an uncontrolled test."""
    a = dataclasses.asdict(get_config(_ARM_A))
    b = dataclasses.asdict(get_config(_ARM_B))
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    # name/exp_name plus the identity-bearing path fields, and the output roots
    # derived from the name. Notably NOT: model, batch, schedule, stride, budget.
    allowed = {"name", "exp_name", "pytorch_weight_path", "data", "val_data",
               "assets_base_dir", "checkpoint_base_dir", "log_base_dir"}
    assert differing <= allowed, f"unexpected divergence between arms: {differing - allowed}"
    # And prove the model config itself is byte-identical, objective included.
    assert a["model"] == b["model"]


def test_formal_arms_carry_the_cont2_schedule() -> None:
    for name in (_ARM_A, _ARM_B):
        cfg = get_config(name)
        assert cfg.num_train_steps == 104_912
        assert cfg.streaming_anchor_stride == 12
        assert cfg.save_interval == 10_000
        assert cfg.val_log_interval == 1_000
        assert cfg.val_num_batches == 20
        assert cfg.project_name == "pi05_ki"
        sched = cfg.lr_schedule
        assert (
            int(sched.warmup_steps),
            float(sched.peak_lr),
            int(sched.decay_steps),
            float(sched.decay_lr),
        ) == (1_000, 1e-5, 104_912, 0.0)


def test_smoke_arms_are_bounded_but_keep_the_formal_per_gpu_batch() -> None:
    """The smoke exists to measure real memory, so B8 must not be reduced."""
    for name in (_ARM_A_SMOKE, _ARM_B_SMOKE):
        cfg = get_config(name)
        assert 0 < cfg.num_train_steps <= 16
        assert cfg.streaming_anchor_stride == 1
        assert cfg.batch_size_per_gpu == 8
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.model.action_token_max_len == _EXPECTED_CAP


def test_formal_contract_table_registers_both_arms_consistently() -> None:
    """The trainer's formal table must agree with the registered configs."""
    source = (_REPO_ROOT / "scripts" / "train_accelerate.py").read_text()
    for name in (_ARM_A, _ARM_B):
        assert f'"{name}": {{' in source, f"{name} missing from formal contract table"
        cfg = get_config(name)
        assert cfg.batch_size_per_gpu == 8
        assert cfg.gradient_accumulation_steps == 1
    # The smokes must NOT be in the formal table: that table pins 104,912 steps.
    formal_block = source.split("_FORMAL_B1K_CONFIG_CONTRACTS = {", 1)[1].split("\n}", 1)[0]
    for name in (_ARM_A_SMOKE, _ARM_B_SMOKE):
        assert name not in formal_block, f"{name} must not be a formal contract"


def test_h20_no_offload_policy_covers_all_four_configs() -> None:
    source = (_REPO_ROOT / "scripts" / "train_accelerate.py").read_text()
    block = source.split("_H20_BF16_NO_OPTIMIZER_OFFLOAD_CONFIGS = {", 1)[1].split("}", 1)[0]
    for name in _ALL:
        assert name in block, f"{name} not covered by the H20 no-offload policy"


def test_deepspeed_config_is_zero2_bf16_without_offload() -> None:
    ds = json.loads(_DS_CONFIG.read_text())
    zero = ds["zero_optimization"]
    assert zero["stage"] == 2
    assert "offload_optimizer" not in zero
    assert "offload_param" not in zero
    assert ds["bf16"]["enabled"] is True
    assert ds["fp16"]["enabled"] is False
    assert ds["gradient_accumulation_steps"] == "auto"


def test_accelerate_config_defers_precision_to_deepspeed() -> None:
    text = _ACCEL_CONFIG.read_text()
    assert "deepspeed_config_file: configs/deepspeed_zero2_h20_bf16.json" in text
    assert re.search(r"^mixed_precision\s*:", text, re.MULTILINE) is None


def test_launcher_gates_on_h20_and_not_a100() -> None:
    text = _LAUNCHER.read_text()
    assert "this BF16 launcher is H20-only" in text
    # Inspect the executable GPU-model gate itself rather than the whole file, so
    # a comment that merely *mentions* A100 cannot fail (or pass) this test.
    gate_lines = [
        line
        for line in text.splitlines()
        if "GPU_MODEL^^" in line and not line.lstrip().startswith("#")
    ]
    assert len(gate_lines) == 1, f"expected exactly one GPU-model gate, got {gate_lines}"
    gate = gate_lines[0]
    assert "*H20*" in gate, gate
    assert "A100" not in gate, f"H20 launcher must not gate on A100: {gate}"


def test_launcher_runs_the_per_device_bf16_preflight() -> None:
    """A GPU-0-only probe would let a node with one bad GPU reach c10d bootstrap."""
    text = _LAUNCHER.read_text()
    assert "cuda_preflight_all_devices.py" in text
    assert "--require-bf16" in text
    assert "--min-driver-major 525" in text
    assert (_REPO_ROOT / "scripts" / "cuda_preflight_all_devices.py").is_file()


def test_launcher_enforces_the_key_invariants() -> None:
    text = _LAUNCHER.read_text()
    # provenance + clean tree + import path
    assert "OPENPI_EXPECTED_CODE_COMMIT" in text
    assert "^[0-9a-f]{40}$" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "openpi import does not resolve inside the pinned tree" in text
    # batch contract
    assert "global batch must be 256 (B8×W32×GA1)" in text
    # occupier handoff
    assert "__GPU_OCCUPY__torch_mm_512" in text
    # warm-start mapping is proven before GPUs are spent
    assert "verify_warm_start_keymap.py" in text
    # the working conda env, and an explicit warning about the broken one
    assert "behavior-data-hl/chenjunting/miniconda3" in text
    assert "GemmaForCausalLM" in text


def test_launcher_pins_each_arm_to_its_own_norm_stats_digest() -> None:
    """The digest assert is the cheap hard proof of the weight/normalization pair."""
    text = _LAUNCHER.read_text()
    assert "d66ed16830a98f90dde8a315058b4a0df59f5e05734c1686d8b3f66787d0a929" in text
    assert "4dde119e69123ed865072c71a714095ae746c6d294fefba910a842757a7083ce" in text
    assert "norm_stats digest mismatch" in text
    # and the weight-size assert that distinguishes fp32 Arm A from bf16 Arm B
    assert "14467165872" in text
    assert "7233650408" in text


def test_cap_provenance_comment_does_not_reuse_the_4dde119e_exhaustive_ids() -> None:
    """Reusing that provenance under a different normalization would be a false claim."""
    text = (_REPO_ROOT / "src" / "openpi" / "training" / "pi05_ki_joint_query_config.py").read_text()
    block = text.split("_H20_FAST_ACTION_TOKEN_MAX_LEN", 1)[0][-4000:]
    assert "SAMPLED, NOT EXHAUSTIVE" in block
    assert "d66ed168" in block
    # The exhaustive ids may be *named* as belonging to 4dde119e, but the block
    # must say so rather than presenting them as this cap's provenance.
    if "0bb9280746" in block:
        assert "belongs to" in block or "NOT reused" in block
