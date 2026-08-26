# ruff: noqa: SLF001 - contract tests intentionally inspect private checkpoint state.
"""CPU contracts for formal 4x8 V100 FP32 π0.5-KI A/B training."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = _REPO_ROOT / "scripts/run_pi05_ki_formal_fp32_4x8_v100.sh"
_A_FORMAL_CONFIG = "pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32"
_A_CONFIG = "pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32_validation10"
_B_CONFIG = "pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32"


def test_formal_v100_configs_are_exactly_matched_outside_objective():
    from openpi.models.pi05_ki_joint_fast_config import Pi05KIJointFastConfig
    from openpi.models.pi05_ki_joint_query_config import Pi05KIJointQueryConfig
    from openpi.training.train_config import get_config

    variant_a = get_config(_A_FORMAL_CONFIG)
    variant_b = get_config(_B_CONFIG)
    assert variant_a.name == _A_FORMAL_CONFIG
    assert variant_b.name == _B_CONFIG
    assert type(variant_a.model) is Pi05KIJointFastConfig
    assert type(variant_b.model) is Pi05KIJointQueryConfig
    assert variant_a.pytorch_model_name == "pi05_ki_joint_fast"
    assert variant_b.pytorch_model_name == "pi05_ki_joint_query"

    for config in (variant_a, variant_b):
        assert config.pytorch_training_precision == "float32"
        assert config.accelerate_mixed_precision == "no"
        assert config.model.dtype == "float32"
        assert config.batch_size_per_gpu == 1
        assert config.gradient_accumulation_steps == 8
        assert config.num_workers == 2
        assert config.expected_global_batch == 256
        assert config.num_train_steps == 104_912
        assert config.num_train_epochs is None
        # Stride 4, not the historical 12, and this PRESERVES coverage rather than
        # loosening it. The trainer no longer rotates anchor offsets at pass
        # boundaries, and {0,4,8} mod 12 unions to exactly {0} mod 4, so one
        # stride-4 sweep selects the identical anchor set the three stride-12
        # passes selected; 26,857,712 // 256 == 104,912 leaves the step budget
        # byte-identical. Keeping stride 12 with one fixed offset would have
        # covered only 1/12 of frames and wrapped ~3x over that same subset.
        assert config.streaming_anchor_stride == 4
        assert config.epoch_anchor_offsets is None
        assert config.save_interval == 10_000
        assert config.checkpoint_policy == "step"
        assert config.rolling_checkpoint_interval == 10_000
        assert config.val_log_interval == 1_000
        assert config.val_num_batches == 20
        assert config.lr_schedule.warmup_steps == 1_000
        assert config.lr_schedule.peak_lr == 1e-5
        assert config.lr_schedule.decay_steps == 104_912
        assert config.lr_schedule.decay_lr == 0.0
        assert config.wandb_enabled is True
        assert config.project_name == "pi05_ki"
        assert config.model.knowledge_insulation is True
        assert config.model.truncate_expert_kv is True
        assert config.data[0].base_config.skill_bridge.enabled is False
        assert config.val_data[0].base_config.skill_bridge.enabled is False
        assert config.data[0].base_config.tasks is None
        assert config.data[0].base_config.episodes_index == list(range(180))
        assert config.val_data[0].base_config.episodes_index == list(range(180, 200))
        assert config.pytorch_weight_path.startswith("/mnt/bn/saiwenresearch/")
        assert config.data[0].base_config.behavior_dataset_root.startswith("/mnt/bn/saiwenresearch/")

    # Every TrainConfig field is identical except arm identity, model selection,
    # and intentionally disjoint output directories.
    exclusions = {
        "name",
        "exp_name",
        "model",
        "pytorch_model_name",
        "assets_base_dir",
        "checkpoint_base_dir",
        "log_base_dir",
    }
    for field in dataclasses.fields(variant_a):
        if field.name not in exclusions:
            assert getattr(variant_a, field.name) == getattr(variant_b, field.name), field.name

    assert variant_a.data == variant_b.data
    assert variant_a.val_data == variant_b.val_data
    assert variant_a.assets_base_dir != variant_b.assets_base_dir
    assert variant_a.checkpoint_base_dir != variant_b.checkpoint_base_dir
    assert variant_a.log_base_dir != variant_b.log_base_dir

    # Exhaustive formal data measurement found a 199-token maximum. The smallest
    # 16-aligned cap leaves nine tokens (4.52%) of headroom. Variant B is unchanged.
    assert variant_a.model.action_token_max_len == 208
    assert not hasattr(variant_b.model, "action_token_max_len")

    a_fields = {field.name for field in dataclasses.fields(variant_a.model)}
    b_fields = {field.name for field in dataclasses.fields(variant_b.model)}
    assert a_fields - b_fields == {"action_token_max_len", "beta_action", "pi05_ki_joint_fast"}
    assert b_fields - a_fields == set()
    for field_name in a_fields & b_fields:
        assert getattr(variant_a.model, field_name) == getattr(variant_b.model, field_name), field_name


def test_query_arm_declares_exactly_three_query_head_tensors():
    """The runtime-only arm delta is embeddings plus Linear weight/bias."""

    source = (_REPO_ROOT / "src/openpi/models_pytorch/pi05_ki_joint_query.py").read_text()
    assert re.search(r"self\.query_embeddings\s*=\s*nn\.Parameter\(", source)
    assert re.search(
        r"self\.query_action_head\s*=\s*nn\.Linear\(self\._vlm_hidden_dim, action_dim, bias=True\)",
        source,
    )
    assert 'name == "query_embeddings"' in source
    assert 'name.startswith("query_action_head.")' in source


def test_recursive_gradient_checkpointing_reaches_decoder_layers_and_recomputes():
    """Formal GC must checkpoint real Gemma layers and preserve KV-cache handling."""
    from openpi.models import gemma as _gemma
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch

    tiny = _gemma.Config(
        width=128,
        depth=2,
        mlp_dim=256,
        num_heads=8,
        num_kv_heads=1,
        head_dim=16,
    )
    joint = PaliGemmaWithExpertModel(tiny, tiny, precision="float32")
    wrapper = SimpleNamespace(
        paligemma_with_expert=joint,
        gradient_checkpointing_enabled=False,
    )

    PI0Pytorch.gradient_checkpointing_enable(wrapper)
    backbone = joint.paligemma.language_model
    expert = joint.gemma_expert.model
    assert wrapper.gradient_checkpointing_enabled is True
    assert backbone.is_gradient_checkpointing
    assert expert.is_gradient_checkpointing
    assert all(layer.gradient_checkpointing for layer in backbone.layers)
    assert all(layer.gradient_checkpointing for layer in expert.layers)
    assert all(hasattr(layer, "_gradient_checkpointing_func") for layer in backbone.layers)

    # The phase-2 prefix pass needs a cache. Its context must recursively disable
    # only backbone decoder checkpointing, then restore the exact policy.
    query_model = PI05KIJointQueryPytorch.__new__(PI05KIJointQueryPytorch)
    torch.nn.Module.__init__(query_model)
    query_model.paligemma_with_expert = joint
    old_use_cache = backbone.config.use_cache
    backbone_gc_modules = [
        module for module in backbone.modules() if hasattr(module, "gradient_checkpointing")
    ]
    backbone_flags = [module.gradient_checkpointing for module in backbone_gc_modules]
    backbone_checkpoint_funcs = [
        module._gradient_checkpointing_func for module in backbone_gc_modules
    ]
    expert_flags = [
        module.gradient_checkpointing
        for module in expert.modules()
        if hasattr(module, "gradient_checkpointing")
    ]
    cache_inputs = torch.randn(1, 4, tiny.width)
    cache_positions = torch.arange(4).unsqueeze(0)
    _, active_gc_cache = joint.forward(
        position_ids=cache_positions,
        inputs_embeds=[cache_inputs, None],
        use_cache=True,
    )
    assert active_gc_cache is None, "HF disables cache while decoder checkpointing is active"

    def _probe_context_then_raise():
        with query_model._no_gc_on_backbone():
            assert not backbone.is_gradient_checkpointing
            assert not any(layer.gradient_checkpointing for layer in backbone.layers)
            assert expert.is_gradient_checkpointing
            assert backbone.config.use_cache is True
            _, prefix_cache = joint.forward(
                position_ids=cache_positions,
                inputs_embeds=[cache_inputs, None],
                use_cache=True,
            )
            assert prefix_cache is not None
            assert prefix_cache.get_seq_length() == 4
            raise RuntimeError("cache probe")

    with pytest.raises(RuntimeError, match="cache probe"):
        _probe_context_then_raise()
    assert [module.gradient_checkpointing for module in backbone_gc_modules] == backbone_flags
    assert [module._gradient_checkpointing_func for module in backbone_gc_modules] == backbone_checkpoint_funcs
    assert [
        module.gradient_checkpointing
        for module in expert.modules()
        if hasattr(module, "gradient_checkpointing")
    ] == expert_flags
    assert backbone.is_gradient_checkpointing
    assert all(layer.gradient_checkpointing for layer in backbone.layers)
    assert expert.is_gradient_checkpointing
    assert backbone.config.use_cache is old_use_cache

    # Count a real projection call in each decoder. Activation checkpointing
    # runs it once in forward and once again during backward recomputation.
    joint.train()
    call_counts = [0 for _ in backbone.layers]
    hooks = []
    for index, layer in enumerate(backbone.layers):
        def _count_call(_module, _inputs, _output, *, index=index):
            call_counts[index] += 1

        hooks.append(layer.self_attn.q_proj.register_forward_hook(_count_call))
    try:
        inputs = torch.randn(1, 4, tiny.width, requires_grad=True)
        positions = torch.arange(4).unsqueeze(0)
        (output, _), cache = joint.forward(
            position_ids=positions,
            inputs_embeds=[inputs, None],
            use_cache=False,
        )
        assert cache is None
        assert call_counts == [1, 1]
        output.sum().backward()
        assert call_counts == [2, 2]
        assert inputs.grad is not None
    finally:
        for hook in hooks:
            hook.remove()


def test_checkpointed_stochastic_probe_preserves_rng_and_downstream_draws():
    """Backward recomputation must not perturb the following expert RNG stream."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    class _FakeHFModel:
        def __init__(self):
            self.checkpoint_kwargs = None

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.checkpoint_kwargs = gradient_checkpointing_kwargs

    paligemma = _FakeHFModel()
    expert = _FakeHFModel()
    wrapper = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(paligemma=paligemma, gemma_expert=expert),
        gradient_checkpointing_enabled=False,
    )
    PI0Pytorch.gradient_checkpointing_enable(wrapper)
    assert paligemma.checkpoint_kwargs == expert.checkpoint_kwargs
    checkpoint_kwargs = paligemma.checkpoint_kwargs
    assert checkpoint_kwargs == {"use_reentrant": False, "preserve_rng_state": True}

    class _StochasticDecoderProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8, bias=False)
            self.forward_calls = 0
            self.draws = []

        def forward(self, inputs):
            self.forward_calls += 1
            output = self.proj(inputs)
            output = torch.nn.functional.dropout(output, p=0.5, training=True)
            self.draws.append(output.detach().clone())
            return output.tanh()

    def _run(*, checkpointed):
        torch.manual_seed(1234)
        probe = _StochasticDecoderProbe()
        inputs = torch.linspace(-1.0, 1.0, 16).reshape(2, 8).requires_grad_()
        torch.manual_seed(5678)
        if checkpointed:
            checkpoint_owner = SimpleNamespace(
                gradient_checkpointing_enabled=True,
                training=True,
            )
            output = PI0Pytorch._apply_checkpoint(checkpoint_owner, probe, inputs)
        else:
            output = probe(inputs)
        output.square().sum().backward()
        rng_state = torch.random.get_rng_state().clone()
        downstream_draws = torch.rand(16)
        return SimpleNamespace(
            output=output.detach(),
            input_grad=inputs.grad.detach(),
            weight_grad=probe.proj.weight.grad.detach(),
            forward_calls=probe.forward_calls,
            recompute_draws=probe.draws,
            rng_state=rng_state,
            downstream_draws=downstream_draws,
        )

    baseline = _run(checkpointed=False)
    recomputed = _run(checkpointed=True)

    assert baseline.forward_calls == 1
    assert recomputed.forward_calls == 2
    assert torch.equal(recomputed.recompute_draws[0], recomputed.recompute_draws[1])
    assert torch.equal(recomputed.output, baseline.output)
    assert torch.equal(recomputed.input_grad, baseline.input_grad)
    assert torch.equal(recomputed.weight_grad, baseline.weight_grad)
    assert torch.equal(recomputed.rng_state, baseline.rng_state)
    assert torch.equal(recomputed.downstream_draws, baseline.downstream_draws)


def test_model_loaded_banner_is_arm_correct():
    source = (_REPO_ROOT / "scripts/train_accelerate.py").read_text()
    assert 'variant_label = "FAST action-token CE variant"' in source
    assert 'variant_label = "query-MSE variant"' in source
    assert "joint query query-MSE variant model loaded" not in source


def test_formal_launcher_resolves_recipe_from_registered_profile():
    source = _LAUNCHER.read_text()
    assert "openpi.training.launcher_profile" in source
    assert "CFG_BATCH_SIZE_PER_GPU" in source
    assert "CFG_GRADIENT_ACCUMULATION_STEPS" in source
    assert "CFG_NUM_TRAIN_STEPS" in source
    assert "CFG_VAL_LOG_INTERVAL" in source
    assert "CFG_SAVE_INTERVAL" in source
    assert "CFG_EXPECTED_GLOBAL_BATCH" in source
    assert '"${GLOBAL_BATCH_SIZE}" -eq "${CFG_EXPECTED_GLOBAL_BATCH}"' in source
    assert "OPENPI_EXPECTED_CODE_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "formal Merlin entrypoint must be the outer keepalive wrapper" in source
    assert "OPENPI_REUSE_PREFIX_KV remains HOLD" in source
    assert 'FORMAL_CUDA_ALLOC_CONF="expandable_segments:True"' in source
    assert "validation=1000" not in source
    assert "--batch-size-per-gpu 1" not in source
    assert "--gradient-accumulation-steps 8" not in source
    assert "--num-train-steps 104912" not in source


def test_formal_deepspeed_file_is_fp32_zero2_cpu_offload():
    config = json.loads((_REPO_ROOT / "configs/deepspeed_zero2_v100_fp32.json").read_text())
    assert config["zero_optimization"]["stage"] == 2
    assert config["zero_optimization"]["offload_optimizer"] == {
        "device": "cpu",
        "pin_memory": True,
    }
    assert config["fp16"]["enabled"] is False
    assert config["bf16"]["enabled"] is False
    assert config["gradient_accumulation_steps"] == "auto"


@pytest.fixture
def formal_preflight_env(tmp_path):
    base = tmp_path / "base"
    assets = base / "assets/behavior-1k/2025-challenge-demos"
    dataset = tmp_path / "dataset"
    cache = tmp_path / "openpi-cache"
    fast = tmp_path / "fast"
    fake_modules = tmp_path / "fake-modules"
    fake_bin = tmp_path / "fake-bin"
    output = tmp_path / "must-not-exist"
    markers = tmp_path / "markers"
    expected_commit = "a" * 40

    assets.mkdir(parents=True)
    dataset.mkdir()
    (cache / "big_vision").mkdir(parents=True)
    fast.mkdir()
    (fake_modules / "transformers").mkdir(parents=True)
    fake_bin.mkdir()
    markers.mkdir()
    (base / "model.safetensors").write_bytes(b"fixture")
    (assets / "norm_stats.json").write_text("{}")
    (cache / "big_vision/paligemma_tokenizer.model").write_bytes(b"fixture")
    (fake_modules / "transformers/__init__.py").write_text(
        """from pathlib import Path
class AutoProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert kwargs == {"trust_remote_code": True, "local_files_only": True}
        assert Path(path).is_dir()
        return cls()
"""
    )
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"""#!/usr/bin/env bash
case "$*" in
  *"rev-parse HEAD"*) printf '%s\\n' {expected_commit!r} ;;
  *"status --porcelain --untracked-files=all"*) exit 0 ;;
  *) echo "unexpected fake git command: $*" >&2; exit 9 ;;
esac
"""
    )
    fake_git.chmod(0o755)

    env = os.environ.copy()
    for variable in (
        "CONFIG_NAME",
        "OPENPI_KI_ARM",
        "OPENPI_REUSE_PREFIX_KV",
        "PYTORCH_TRAINING_PRECISION",
        "ACCELERATE_MIXED_PRECISION",
        "WANDB_MODE",
        "WANDB_DISABLED",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        env.pop(variable, None)
    env.update(
        {
            "OPENPI_LAUNCH_PREFLIGHT_ONLY": "1",
            "OPENPI_EXPECTED_CODE_COMMIT": expected_commit,
            "OPENPI_PREFLIGHT_PYTHON": sys.executable,
            "ARNOLD_WORKER_NUM": "4",
            "ARNOLD_WORKER_GPU": "8",
            "ARNOLD_ID": "0",
            "ARNOLD_WORKER_0_HOST": "127.0.0.1",
            "ARNOLD_WORKER_0_PORT": "29514",
            "ARNOLD_WORKER_GPU_TYPE": "Tesla_V100_SXM2_32GB",
            "BASE_PI05_CKPT": str(base),
            "B1K_DATASET_ROOT": str(dataset),
            "B1K_ASSETS_DIR": str(assets),
            "NORM_STATS_PATH": str(assets / "norm_stats.json"),
            "REPO_OPENPI_CACHE": str(cache),
            "PALIGEMMA_TOKENIZER": str(cache / "big_vision/paligemma_tokenizer.model"),
            "OPENPI_FAST_TOKENIZER_PATH": str(fast),
            "PERSISTENT_OUTPUT_BASE": str(output),
            "PYTHONPATH": os.pathsep.join([str(fake_modules), str(_REPO_ROOT), str(_REPO_ROOT / "src")]),
            "PATH": os.pathsep.join([str(fake_bin), env.get("PATH", "")]),
        }
    )
    return env, output


def _run_formal_preflight(env, arm):
    return subprocess.run(
        ["bash", str(_LAUNCHER), arm],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    ("arm", "expected_config", "fast_expected"),
    [
        ("A", _A_FORMAL_CONFIG, True),
        ("B", _B_CONFIG, False),
    ],
)
def test_formal_launcher_cpu_preflight_is_strict_and_side_effect_free(
    formal_preflight_env, arm, expected_config, fast_expected
):
    env, output = formal_preflight_env
    if not fast_expected:
        env.pop("OPENPI_FAST_TOKENIZER_PATH")
    result = _run_formal_preflight(env, arm)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "FORMAL_FP32_ZERO2_PREFLIGHT_OK" in combined
    assert f"profile={expected_config}" in combined
    assert "PREFLIGHT_OK" in combined
    assert ("FAST_OFFLINE_PROCESSOR_PREFLIGHT_OK" in combined) is fast_expected
    assert not output.exists()


def test_validation10_launcher_profile_preflight_is_side_effect_free(formal_preflight_env):
    env, output = formal_preflight_env
    env["CONFIG_NAME"] = _A_CONFIG
    result = _run_formal_preflight(env, "A")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert f"profile={_A_CONFIG}" in combined
    assert "save=10000 val=10x20" in combined
    assert not output.exists()


def test_formal_launcher_rejects_incompatible_allocator_contract(formal_preflight_env):
    env, output = formal_preflight_env
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    result = _run_formal_preflight(env, "B")
    assert result.returncode != 0
    assert "formal V100 requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in result.stderr
    assert not output.exists()


def test_formal_launcher_normal_run_rejects_missing_outer_wrapper_contract(formal_preflight_env):
    env, output = formal_preflight_env
    env["OPENPI_LAUNCH_PREFLIGHT_ONLY"] = "0"
    result = _run_formal_preflight(env, "B")
    assert result.returncode != 0
    assert "formal Merlin entrypoint must be the outer keepalive wrapper" in result.stderr
    assert not output.exists()


def test_formal_launcher_rejects_debug_config_without_fallback(formal_preflight_env):
    env, output = formal_preflight_env
    env["CONFIG_NAME"] = "pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32_debug"
    result = _run_formal_preflight(env, "B")
    assert result.returncode != 0
    assert "refusing unknown/mismatched profile" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("BATCH_SIZE_PER_GPU", "8"),
        ("GRADIENT_ACCUMULATION_STEPS", "1"),
        ("NUM_TRAIN_STEPS", "5"),
        ("SAVE_INTERVAL", "5"),
        ("VAL_LOG_INTERVAL", "5"),
        ("VAL_NUM_BATCHES", "1"),
    ],
)
def test_recipe_environment_cannot_override_registered_profile(formal_preflight_env, variable, value):
    env, output = formal_preflight_env
    env[variable] = value
    result = _run_formal_preflight(env, "B")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "B1xW32xGA8=256" in result.stdout
    assert "save=10000 val=1000x20" in result.stdout
    assert not output.exists()
