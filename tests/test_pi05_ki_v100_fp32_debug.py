"""CPU-only contracts for the separate-allocation 4x8 V100 debug launch."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = _REPO_ROOT / "scripts/run_pi05_ki_variant_fp32_4x8_v100.sh"
_FP32_ACCELERATE = _REPO_ROOT / "configs/accelerate_ds_zero2_v100_fp32.yaml"
_FP32_DEEPSPEED = _REPO_ROOT / "configs/deepspeed_zero2_v100_fp32.json"
_A_CONFIG = "pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32_debug"
_B_CONFIG = "pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32_debug"

sys.path.insert(0, str(_REPO_ROOT / "src"))


def _skill_bridge_enabled(config) -> bool:
    return config.data[0].base_config.skill_bridge.enabled


def test_fp32_debug_configs_are_registered_matched_and_isolated():
    from openpi.models.pi05_ki_joint_fast_config import Pi05KIJointFastConfig
    from openpi.models.pi05_ki_joint_query_config import Pi05KIJointQueryConfig
    from openpi.training.train_config import get_config

    variant_a = get_config(_A_CONFIG)
    variant_b = get_config(_B_CONFIG)

    assert variant_a.name == _A_CONFIG
    assert variant_b.name == _B_CONFIG
    assert variant_a.pytorch_model_name == "pi05_ki_joint_fast"
    assert variant_b.pytorch_model_name == "pi05_ki_joint_query"
    assert type(variant_a.model) is Pi05KIJointFastConfig
    assert type(variant_b.model) is Pi05KIJointQueryConfig

    for config in (variant_a, variant_b):
        assert config.pytorch_training_precision == "float32"
        assert config.accelerate_mixed_precision == "no"
        assert config.model.dtype == "float32"
        assert config.model.knowledge_insulation is True
        assert config.model.truncate_expert_kv is True
        assert config.batch_size_per_gpu == 1
        assert config.num_train_steps == 5
        assert config.num_train_epochs is None
        assert config.streaming_anchor_stride == 1
        assert _skill_bridge_enabled(config) is False
        assert config.val_data[0].base_config.skill_bridge.enabled is False

    # All TrainConfig fields must match except identity/output/model selection.
    train_exclusions = {
        "name",
        "exp_name",
        "pytorch_model_name",
        "model",
        "assets_base_dir",
        "checkpoint_base_dir",
        "log_base_dir",
    }
    for field in dataclasses.fields(variant_a):
        if field.name not in train_exclusions:
            assert getattr(variant_a, field.name) == getattr(variant_b, field.name), field.name

    # Model fields match except action-token/query-specific objective fields.
    objective_fields = {
        "action_token_max_len",
        "beta_action",
        "beta_query",
        "num_query_tokens",
        "query_emb_dim",
        "pi05_ki_joint_fast",
    }
    common_model_fields = {field.name for field in dataclasses.fields(variant_a.model)} & {
        field.name for field in dataclasses.fields(variant_b.model)
    }
    for field_name in common_model_fields - objective_fields:
        assert getattr(variant_a.model, field_name) == getattr(variant_b.model, field_name), field_name

    assert variant_a.checkpoint_base_dir != variant_b.checkpoint_base_dir
    assert variant_a.log_base_dir != variant_b.log_base_dir
    assert variant_a.assets_base_dir != variant_b.assets_base_dir


def test_fp32_accelerate_omits_duplicate_precision_and_deepspeed_enforces_fp32():
    accelerate_text = _FP32_ACCELERATE.read_text()
    assert not any(line.startswith("mixed_precision:") for line in accelerate_text.splitlines())
    assert "deepspeed_config_file: configs/deepspeed_zero2_v100_fp32.json" in accelerate_text

    deepspeed = json.loads(_FP32_DEEPSPEED.read_text())
    assert deepspeed["fp16"]["enabled"] is False
    assert deepspeed["bf16"]["enabled"] is False
    assert deepspeed["torch_autocast"] == {"enabled": False, "dtype": "float32"}


def _parse_deepspeed_plugin_from_accelerate_config(config_path: Path):
    """Exercise the Accelerate launch-to-plugin path without starting workers."""
    from accelerate.commands import launch as accelerate_launch
    from accelerate.utils import DeepSpeedPlugin
    from accelerate.utils.launch import prepare_deepspeed_cmd_env

    args = accelerate_launch.launch_command_parser().parse_args(
        [
            "--config_file",
            str(config_path),
            "--num_processes",
            "1",
            "--num_machines",
            "1",
            "--machine_rank",
            "0",
            "--main_process_ip",
            "127.0.0.1",
            "--main_process_port",
            "0",
            "--same_network",
            str(_REPO_ROOT / "scripts/train_accelerate.py"),
        ]
    )
    result = {}

    def parse_plugin_without_launching(launch_args):
        _command, launch_env = prepare_deepspeed_cmd_env(launch_args)
        with mock.patch.dict(os.environ, launch_env, clear=True):
            plugin = DeepSpeedPlugin(hf_ds_config=launch_env["ACCELERATE_DEEPSPEED_CONFIG_FILE"])
        result.update(plugin=plugin, launch_env=launch_env)

    with mock.patch.object(accelerate_launch, "deepspeed_launcher", parse_plugin_without_launching):
        accelerate_launch.launch_command(args)
    return result["plugin"], result["launch_env"]


def test_installed_accelerate_plugin_rejects_old_duplicate_and_accepts_fixed(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(_REPO_ROOT)
    old_config = tmp_path / "old-accelerate-v100-fp32.yaml"
    old_config.write_text(
        _FP32_ACCELERATE.read_text().replace(
            "main_training_function: main\n",
            'main_training_function: main\nmixed_precision: "no"\n',
        )
    )

    with pytest.raises(ValueError, match=r"When using `deepspeed_config_file`"):
        _parse_deepspeed_plugin_from_accelerate_config(old_config)

    plugin, launch_env = _parse_deepspeed_plugin_from_accelerate_config(_FP32_ACCELERATE)
    config_fields = launch_env["ACCELERATE_CONFIG_DS_FIELDS"].split(",")
    assert "mixed_precision" not in config_fields
    assert launch_env["ACCELERATE_MIXED_PRECISION"] == "no"
    assert plugin.deepspeed_config["fp16"]["enabled"] is False
    assert plugin.deepspeed_config["bf16"]["enabled"] is False
    assert plugin.deepspeed_config["torch_autocast"] == {"enabled": False, "dtype": "float32"}


def test_fast_tokenizer_uses_explicit_offline_processor_cache(tmp_path, monkeypatch):
    import openpi.models.tokenizer as tokenizer_module

    paligemma_model = tmp_path / "paligemma.model"
    paligemma_model.write_bytes(b"fixture")
    fast_cache = tmp_path / "fast"
    fast_cache.mkdir()
    calls = []

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return object()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=FakeAutoProcessor))
    monkeypatch.setattr(tokenizer_module.download, "maybe_download", lambda *_args, **_kwargs: paligemma_model)
    monkeypatch.setattr(
        tokenizer_module.sentencepiece,
        "SentencePieceProcessor",
        lambda **_kwargs: object(),
    )
    monkeypatch.setenv("OPENPI_FAST_TOKENIZER_PATH", str(fast_cache))
    monkeypatch.setenv("OPENPI_OFFLINE", "1")

    tokenizer_module.FASTTokenizer()

    assert calls == [
        (
            str(fast_cache),
            {"trust_remote_code": True, "local_files_only": True},
        )
    ]


@pytest.fixture
def launcher_fixture(tmp_path: Path):
    base = tmp_path / "base"
    assets = base / "assets/behavior-1k/2025-challenge-demos"
    dataset = tmp_path / "dataset"
    cache = tmp_path / "openpi-cache"
    fast = tmp_path / "fast-processor"
    fake_modules = tmp_path / "fake-modules"
    fake_bin = tmp_path / "fake-bin"
    output = tmp_path / "must-not-be-created-by-preflight"
    markers = tmp_path / "markers"

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
    (fast / "processor_config.json").write_text("{}")

    # The launcher still exercises the exact offline AutoProcessor call, while
    # this fixture avoids requiring the real remote-code processor in CI.
    (fake_modules / "transformers/__init__.py").write_text(
        """import json
import os
from pathlib import Path

class AutoProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        if kwargs.get(\"local_files_only\") is not True:
            raise RuntimeError(\"local_files_only was not true\")
        if not Path(path).is_dir():
            raise RuntimeError(\"processor path is not a directory\")
        marker = os.environ.get(\"FAKE_FAST_LOAD_MARKER\")
        if marker:
            Path(marker).write_text(json.dumps({\"path\": path, **kwargs}, default=str))
        return cls()
"""
    )

    wrapper_marker = markers / "wrapper"
    fake_wrapper = tmp_path / "must-not-run-wrapper.sh"
    fake_wrapper.write_text(
        f"""#!/usr/bin/env bash
{{
  echo "ARM=${{OPENPI_KI_ARM:-}}"
  echo "CONFIG_NAME=${{CONFIG_NAME:-}}"
  echo "LAUNCHER=${{LAUNCHER:-}}"
  echo "KEEPALIVE_ON_SUCCESS=${{KEEPALIVE_ON_SUCCESS:-}}"
  echo "STRICT_GPU_COUNT=${{STRICT_GPU_COUNT:-}}"
  echo "PERSISTENT_OUTPUT_ROOT=${{PERSISTENT_OUTPUT_ROOT:-}}"
}} > {wrapper_marker!s}
exit 0
"""
    )
    fake_wrapper.chmod(0o755)

    gpu_marker = markers / "nvidia-smi"
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(f"#!/usr/bin/env bash\necho invoked > {gpu_marker!s}\nexit 99\n")
    fake_nvidia_smi.chmod(0o755)
    fake_ps = fake_bin / "ps"
    fake_ps.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_ps.chmod(0o755)

    env = os.environ.copy()
    for variable in (
        "CONFIG_NAME",
        "OPENPI_KI_ARM",
        "OPENPI_FAST_TOKENIZER_PATH",
        "OPENPI_REUSE_PREFIX_KV",
        "PYTORCH_TRAINING_PRECISION",
        "ACCELERATE_MIXED_PRECISION",
        "NUM_NODES",
        "GPUS_PER_NODE",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "ARNOLD_WORKER_GPU_TYPE",
        "ARNOLD_GPU_TYPE",
        "GPU_MODEL",
        "WANDB_MODE",
        "WANDB_DISABLED",
    ):
        env.pop(variable, None)
    env.update(
        {
            "OPENPI_LAUNCH_PREFLIGHT_ONLY": "1",
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
            "ACCEL_CONFIG": str(_FP32_ACCELERATE),
            "DEEPSPEED_CONFIG": str(_FP32_DEEPSPEED),
            "KEEPALIVE_WRAPPER": str(fake_wrapper),
            "PERSISTENT_OUTPUT_BASE": str(output),
            "FAKE_FAST_LOAD_MARKER": str(markers / "fast-load"),
            "PYTHONPATH": os.pathsep.join([str(fake_modules), str(_REPO_ROOT / "src"), env.get("PYTHONPATH", "")]),
            "PATH": os.pathsep.join([str(fake_bin), env.get("PATH", "")]),
        }
    )
    return {
        "env": env,
        "fast": fast,
        "output": output,
        "wrapper_marker": wrapper_marker,
        "gpu_marker": gpu_marker,
        "fast_marker": markers / "fast-load",
    }


def _run_launcher(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_LAUNCHER), *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


@pytest.mark.parametrize(
    ("selection", "expected_config", "expects_fast"),
    [("A", _A_CONFIG, True), ("B", _B_CONFIG, False)],
)
def test_preflight_selects_arm_and_touches_no_gpu_or_occupier(
    launcher_fixture, selection, expected_config, expects_fast
):
    fixture = launcher_fixture
    result = _run_launcher(fixture["env"], selection)
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert f"CONFIG_PREFLIGHT_OK name={expected_config}" in output
    assert "PREFLIGHT_OK" in output
    assert fixture["fast_marker"].exists() is expects_fast
    assert not fixture["wrapper_marker"].exists()
    assert not fixture["gpu_marker"].exists()
    assert not fixture["output"].exists()


def test_arm_can_be_selected_by_environment(launcher_fixture):
    env = launcher_fixture["env"].copy()
    env["OPENPI_KI_ARM"] = "query-mse"
    result = _run_launcher(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"CONFIG_PREFLIGHT_OK name={_B_CONFIG}" in result.stdout


def test_normal_path_hands_off_to_keepalive_with_arm_scoped_state(launcher_fixture):
    fixture = launcher_fixture
    env = fixture["env"].copy()
    env["OPENPI_LAUNCH_PREFLIGHT_ONLY"] = "0"

    result = _run_launcher(env, "B")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    handoff = fixture["wrapper_marker"].read_text()
    assert "ARM=B" in handoff
    assert f"CONFIG_NAME={_B_CONFIG}" in handoff
    assert f"LAUNCHER={_LAUNCHER}" in handoff
    assert "KEEPALIVE_ON_SUCCESS=1" in handoff
    assert "STRICT_GPU_COUNT=0" in handoff
    assert "variantB_query_mse" in handoff
    assert fixture["output"].exists()
    assert not fixture["gpu_marker"].exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ARNOLD_WORKER_NUM": "8"}, "expected exactly 4 nodes"),
        ({"ARNOLD_WORKER_GPU": "4"}, "expected exactly 8 GPUs per node"),
        ({"ARNOLD_ID": "4"}, "must be less than NUM_NODES"),
    ],
)
def test_preflight_rejects_non_4x8_topology(launcher_fixture, overrides, message):
    env = launcher_fixture["env"].copy()
    env.update(overrides)
    result = _run_launcher(env, "B")
    assert result.returncode != 0
    assert message in result.stderr
    assert not launcher_fixture["wrapper_marker"].exists()
    assert not launcher_fixture["gpu_marker"].exists()


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("PYTORCH_TRAINING_PRECISION", "bfloat16", "requires PYTORCH_TRAINING_PRECISION=float32"),
        ("PYTORCH_TRAINING_PRECISION", "float16", "requires PYTORCH_TRAINING_PRECISION=float32"),
        ("ACCELERATE_MIXED_PRECISION", "bf16", "requires ACCELERATE_MIXED_PRECISION=no"),
        ("ACCELERATE_MIXED_PRECISION", "fp16", "requires ACCELERATE_MIXED_PRECISION=no"),
        ("OPENPI_REUSE_PREFIX_KV", "1", "must remain disabled"),
    ],
)
def test_preflight_rejects_unsafe_precision_or_kv_reuse(launcher_fixture, variable, value, message):
    env = launcher_fixture["env"].copy()
    env[variable] = value
    result = _run_launcher(env, "B")
    assert result.returncode != 0
    assert message in result.stderr


def test_preflight_rejects_config_name_fallback(launcher_fixture):
    env = launcher_fixture["env"].copy()
    env["CONFIG_NAME"] = "pi05_ki_joint_fast_typo"
    result = _run_launcher(env, "A")
    assert result.returncode != 0
    assert "silently fall back" in result.stderr
    assert not launcher_fixture["wrapper_marker"].exists()


def test_preflight_rejects_non_v100_model_when_platform_reports_it(launcher_fixture):
    env = launcher_fixture["env"].copy()
    env["ARNOLD_WORKER_GPU_TYPE"] = "NVIDIA_A100_40GB"
    result = _run_launcher(env, "B")
    assert result.returncode != 0
    assert "V100-only" in result.stderr


def test_preflight_rejects_accelerate_file_with_duplicate_mixed_precision(launcher_fixture, tmp_path):
    bad_accelerate = tmp_path / "bad-accelerate.yaml"
    bad_accelerate.write_text(
        _FP32_ACCELERATE.read_text().replace(
            "main_training_function: main\n",
            'main_training_function: main\nmixed_precision: "no"\n',
        )
    )
    env = launcher_fixture["env"].copy()
    env["ACCEL_CONFIG"] = str(bad_accelerate)
    result = _run_launcher(env, "B")
    assert result.returncode != 0
    assert "must not define top-level mixed_precision" in result.stderr


def test_variant_a_requires_actionable_offline_fast_cache(launcher_fixture):
    env = launcher_fixture["env"].copy()
    env.pop("OPENPI_FAST_TOKENIZER_PATH")
    result = _run_launcher(env, "A")
    assert result.returncode != 0
    assert "Variant A requires OPENPI_FAST_TOKENIZER_PATH" in result.stderr
    assert "pre-cached physical-intelligence/fast" in result.stderr
    assert not launcher_fixture["wrapper_marker"].exists()
    assert not launcher_fixture["gpu_marker"].exists()


def test_variant_b_does_not_require_fast_cache(launcher_fixture):
    env = launcher_fixture["env"].copy()
    env.pop("OPENPI_FAST_TOKENIZER_PATH")
    env["FAKE_FAST_LOAD_MARKER"] = str(launcher_fixture["fast_marker"])
    result = _run_launcher(env, "B")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAST processor cache is not required" in result.stdout
    assert not launcher_fixture["fast_marker"].exists()


def test_arm_selection_is_mandatory_and_conflicts_are_rejected(launcher_fixture):
    missing = _run_launcher(launcher_fixture["env"])
    assert missing.returncode != 0
    assert "arm selection is required" in missing.stderr

    env = launcher_fixture["env"].copy()
    env["OPENPI_KI_ARM"] = "B"
    conflict = _run_launcher(env, "A")
    assert conflict.returncode != 0
    assert "conflicts with OPENPI_KI_ARM" in conflict.stderr
