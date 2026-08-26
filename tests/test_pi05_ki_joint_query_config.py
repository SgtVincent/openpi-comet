"""Tests for π0.5-KI joint query training configs.

Verifies that the experimental π0.5-KI joint query query-MSE variant configs are properly registered
and have the expected model and KI settings.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure config imports resolve to this worktree even when another checkout is
# installed editable in the active environment.
sys.path.insert(0, str(_REPO_ROOT / "src"))

_FULL_TASK_BF16_CONFIG = "pi05_ki_joint_query_b1k-full_task-ki_on_bf16"
_FULL_TASK_BF16_LAUNCHER = _REPO_ROOT / "scripts/run_pi05_ki_joint_query_full_b1k_bf16_multinode_hl.sh"
_LQ_FP32_LAUNCHER = _REPO_ROOT / "scripts/run_pi05_ki_joint_query_single_task_radio_fp32_multinode_v100.sh"
_B1K_HEADLESS_EVAL_LAUNCHER = _REPO_ROOT / "scripts/run_b1k_eval_parallel_single_task_headless.sh"


def _manual_preflight_env(**overrides):
    env = os.environ.copy()
    for variable in ("ARNOLD_JOB_ID", "ARNOLD_TASK_ID", "OPENPI_HF_CACHE_RUN_ID", "LOCAL_CACHE_ROOT"):
        env.pop(variable, None)
    env.update(
        {
            "OPENPI_LAUNCH_PREFLIGHT_ONLY": "1",
            "CONFIG_NAME": _FULL_TASK_BF16_CONFIG,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}",
            "PREPARE_HF_CACHE_ONLY": "0",
            "FORCE_LOAD_CACHE": "0",
            "WANDB_DISABLED": "0",
            "WANDB_MODE": "online",
            **overrides,
        }
    )
    return env


def _run_full_task_preflight(env):
    return subprocess.run(
        ["bash", str(_FULL_TASK_BF16_LAUNCHER)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _tmp_alias_for(local_root):
    alias_digest = hashlib.sha256(os.fsencode(local_root)).hexdigest()[:24]
    return Path("/tmp") / f"openpi-tmp-{os.getuid()}-{alias_digest}"


def _tree_state(root):
    state = {}
    for path in [root, *sorted(root.rglob("*"))]:
        stat_result = path.lstat()
        relative_path = "." if path == root else path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        state[relative_path] = (stat_result.st_mode, stat_result.st_size, stat_result.st_mtime_ns, digest)
    return state


def test_ki_on_config_resolves():
    """KI=ON smoke config should be registered and resolve via get_config."""
    from openpi.training.train_config import get_config

    config = get_config("pi05_ki_joint_query_b1k-ki_on_smoke")
    assert config is not None
    assert config.name == "pi05_ki_joint_query_b1k-ki_on_smoke"


def test_ki_off_config_resolves():
    """KI=OFF smoke config should be registered and resolve via get_config."""
    from openpi.training.train_config import get_config

    config = get_config("pi05_ki_joint_query_b1k-ki_off_smoke")
    assert config is not None
    assert config.name == "pi05_ki_joint_query_b1k-ki_off_smoke"


def test_both_configs_use_pi05_ki_joint_query_model():
    """Both KI ON and OFF configs should use pi05_ki_joint_query pytorch model."""
    from openpi.training.train_config import get_config

    ki_on = get_config("pi05_ki_joint_query_b1k-ki_on_smoke")
    ki_off = get_config("pi05_ki_joint_query_b1k-ki_off_smoke")

    assert ki_on.pytorch_model_name == "pi05_ki_joint_query"
    assert ki_off.pytorch_model_name == "pi05_ki_joint_query"


def test_ki_on_has_knowledge_insulation_true():
    """KI=ON config should have knowledge_insulation=True in model config."""
    from openpi.training.train_config import get_config

    config = get_config("pi05_ki_joint_query_b1k-ki_on_smoke")
    assert config.model.knowledge_insulation is True
    assert config.model.truncate_expert_kv is True


def test_ki_off_has_knowledge_insulation_false():
    """KI=OFF config should have knowledge_insulation=False in model config."""
    from openpi.training.train_config import get_config

    config = get_config("pi05_ki_joint_query_b1k-ki_off_smoke")
    assert config.model.knowledge_insulation is False
    assert config.model.truncate_expert_kv is True


def test_both_configs_have_pi05_ki_joint_query_flag():
    """Both configs should have pi05_ki_joint_query=True to identify as π0.5-KI joint query configs."""
    from openpi.training.train_config import get_config

    ki_on = get_config("pi05_ki_joint_query_b1k-ki_on_smoke")
    ki_off = get_config("pi05_ki_joint_query_b1k-ki_off_smoke")

    assert hasattr(ki_on.model, "pi05_ki_joint_query")
    assert hasattr(ki_off.model, "pi05_ki_joint_query")
    assert ki_on.model.pi05_ki_joint_query is True
    assert ki_off.model.pi05_ki_joint_query is True


def test_smoke_configs_have_small_batch_and_few_steps():
    """Smoke configs should have tiny batch and few steps for fast validation."""
    from openpi.training.train_config import get_config

    ki_on = get_config("pi05_ki_joint_query_b1k-ki_on_smoke")

    assert ki_on.batch_size_per_gpu == 1
    assert ki_on.num_train_steps == 5
    assert ki_on.gradient_accumulation_steps == 1
    assert ki_on.ema_decay is None
    assert ki_on.wandb_enabled is False
    # Base smoke configs use fp16 (intended production)
    assert ki_on.pytorch_training_precision == "float16"
    assert ki_on.accelerate_mixed_precision == "fp16"


def test_smoke_configs_use_subtask_annotations():
    """Smoke configs should use annotations_skill subtask source."""
    from openpi.training.train_config import get_config

    ki_on = get_config("pi05_ki_joint_query_b1k-ki_on_smoke")

    # Data is a list of DataConfigFactory; check base_config subtask_source
    data_factory = ki_on.data[0]
    assert hasattr(data_factory, "base_config")
    assert data_factory.base_config.subtask_source == "annotations_skill"


# --- fp32 V100 smoke variants ---


def test_fp32_ki_on_config_resolves():
    """FP32 KI=ON smoke config should resolve."""
    from openpi.training.train_config import get_config

    config = get_config("pi05_ki_joint_query_b1k-ki_on_smoke_fp32")
    assert config.name == "pi05_ki_joint_query_b1k-ki_on_smoke_fp32"
    assert config.pytorch_training_precision == "float32"
    assert config.accelerate_mixed_precision == "no"


def test_fp32_ki_off_config_resolves():
    """FP32 KI=OFF smoke config should resolve."""
    from openpi.training.train_config import get_config

    config = get_config("pi05_ki_joint_query_b1k-ki_off_smoke_fp32")
    assert config.name == "pi05_ki_joint_query_b1k-ki_off_smoke_fp32"
    assert config.model.knowledge_insulation is False
    assert config.pytorch_training_precision == "float32"


def test_fp32_variants_have_same_model_settings_as_fp16():
    """FP32 variants should have identical model settings to FP16 variants."""
    from openpi.training.train_config import get_config

    ki_on_fp16 = get_config("pi05_ki_joint_query_b1k-ki_on_smoke")
    ki_on_fp32 = get_config("pi05_ki_joint_query_b1k-ki_on_smoke_fp32")

    # Model settings should be identical
    assert ki_on_fp16.model.knowledge_insulation == ki_on_fp32.model.knowledge_insulation
    assert ki_on_fp16.model.truncate_expert_kv == ki_on_fp32.model.truncate_expert_kv
    assert ki_on_fp16.model.alpha == ki_on_fp32.model.alpha
    assert ki_on_fp16.model.num_query_tokens == ki_on_fp32.model.num_query_tokens
    assert ki_on_fp16.model.action_horizon == ki_on_fp32.model.action_horizon

    # Batch/steps should be identical
    assert ki_on_fp16.batch_size_per_gpu == ki_on_fp32.batch_size_per_gpu
    assert ki_on_fp16.num_train_steps == ki_on_fp32.num_train_steps

    # Only precision should differ
    assert ki_on_fp16.pytorch_training_precision != ki_on_fp32.pytorch_training_precision


# --- Output directory isolation (no collisions between configs) ---

_ALL_CONFIG_NAMES = [
    "pi05_ki_joint_query_b1k-ki_on_smoke",
    "pi05_ki_joint_query_b1k-ki_off_smoke",
    "pi05_ki_joint_query_b1k-ki_on_smoke_fp32",
    "pi05_ki_joint_query_b1k-ki_off_smoke_fp32",
    "pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp32",
    "pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp16",
    "pi05_ki_joint_query_b1k-single_task-radio-ki_on_bf16",
    _FULL_TASK_BF16_CONFIG,
]


def test_all_configs_have_unique_exp_name():
    """Each config must have a unique exp_name to prevent output collisions."""
    from openpi.training.train_config import get_config

    exp_names = [get_config(name).exp_name for name in _ALL_CONFIG_NAMES]
    assert len(exp_names) == len(set(exp_names)), (
        f"Duplicate exp_names found: {exp_names}"
    )


def test_all_configs_have_unique_checkpoint_dirs():
    """Each config must have a unique checkpoint_base_dir."""
    from openpi.training.train_config import get_config

    ckpt_dirs = [get_config(name).checkpoint_base_dir for name in _ALL_CONFIG_NAMES]
    assert len(ckpt_dirs) == len(set(ckpt_dirs)), (
        f"Duplicate checkpoint_base_dirs found: {ckpt_dirs}"
    )


def test_all_configs_have_unique_log_dirs():
    """Each config must have a unique log_base_dir."""
    from openpi.training.train_config import get_config

    log_dirs = [get_config(name).log_base_dir for name in _ALL_CONFIG_NAMES]
    assert len(log_dirs) == len(set(log_dirs)), (
        f"Duplicate log_base_dirs found: {log_dirs}"
    )


def test_all_configs_have_unique_asset_dirs():
    """Each config must have a unique assets_base_dir."""
    from openpi.training.train_config import get_config

    asset_dirs = [get_config(name).assets_base_dir for name in _ALL_CONFIG_NAMES]
    assert len(asset_dirs) == len(set(asset_dirs)), (
        f"Duplicate assets_base_dirs found: {asset_dirs}"
    )


def test_exp_name_matches_config_name():
    """exp_name should equal the config name for easy traceability."""
    from openpi.training.train_config import get_config

    for name in _ALL_CONFIG_NAMES:
        config = get_config(name)
        assert config.exp_name == name, (
            f"exp_name mismatch: config '{name}' has exp_name='{config.exp_name}'"
        )


# --- Multi-task data config regression tests ---

_EXPECTED_B1K_MULTITASK_TASKS = (
    "turning_on_radio",
    "picking_up_trash",
    "putting_away_Halloween_decorations",
    "cleaning_up_plates_and_food",
    "can_meat",
)


def test_b1k_multitask_tasks_exact_names():
    """Regression: _B1K_MULTITASK_TASKS must have exactly 5 canonical task names.

    Guards against stale task lists (e.g. putting_dishes_in_dishwasher vs can_meat)
    and ensures the 5-task subset matches the canonical B1K task indices 0-4.
    """
    # Import via get_config side-loading to avoid circular imports
    from openpi.training.train_config import get_config

    # Access the module's constant through the loaded module
    import openpi.training.pi05_ki_joint_query_config as pi05_ki_joint_query_cfg
    tasks = pi05_ki_joint_query_cfg._B1K_MULTITASK_TASKS

    assert len(tasks) == 5, (
        f"Expected 5 tasks, got {len(tasks)}: {tasks}"
    )
    assert tuple(tasks) == _EXPECTED_B1K_MULTITASK_TASKS, (
        f"Task name mismatch.\n  Expected: {_EXPECTED_B1K_MULTITASK_TASKS}\n  Got:      {tuple(tasks)}"
    )
    # All task names must be non-empty strings
    for t in tasks:
        assert isinstance(t, str) and len(t) > 0, f"Invalid task name: {t!r}"


def test_multitask_data_config_per_task_episodes_index():
    """Regression: episodes_index must be applied per-task, giving num_tasks × episodes_per_task total.

    With tasks=[t1..t5] and episodes_index=[0..19], the LeRobotB1KDataConfig
    base_config should have 5 tasks and 20 episodes per task = 100 total episodes.
    """
    import openpi.training.pi05_ki_joint_query_config as pi05_ki_joint_query_cfg

    data_cfg = pi05_ki_joint_query_cfg._make_b1k_multitask_data_config(num_tasks=5, episodes_per_task=20)

    # tasks and episodes_index live on base_config (DataConfig), not on the factory
    base = data_cfg.base_config

    # Check tasks list
    assert base.tasks is not None, "tasks should not be None (explicit list required)"
    assert len(base.tasks) == 5, f"Expected 5 tasks, got {len(base.tasks)}"
    assert tuple(base.tasks) == _EXPECTED_B1K_MULTITASK_TASKS

    # Check episodes_index is per-task (0..19, not global indices)
    assert base.episodes_index is not None
    assert len(base.episodes_index) == 20, (
        f"Expected 20 episodes per task, got {len(base.episodes_index)}"
    )
    assert base.episodes_index == list(range(20)), (
        f"episodes_index should be range(20) for per-task indexing, got {base.episodes_index[:5]}...{base.episodes_index[-5:]}"
    )

    # Total episodes = num_tasks × episodes_per_task = 100
    total_expected = 5 * 20
    total_from_config = len(base.tasks) * len(base.episodes_index)
    assert total_from_config == total_expected, (
        f"Expected {total_expected} total episodes, got {total_from_config}"
    )


def test_multitask_num_tasks_bounds_check():
    """num_tasks > available tasks should raise ValueError."""
    import openpi.training.pi05_ki_joint_query_config as pi05_ki_joint_query_cfg
    import pytest

    with pytest.raises(ValueError, match="exceeds available tasks"):
        pi05_ki_joint_query_cfg._make_b1k_multitask_data_config(num_tasks=999, episodes_per_task=1)


def test_long_baseline_configs_use_multitask_data():
    """Long baseline configs (KI=ON 500-step and KI=OFF 200-step) should use 5-task × 20-ep data."""
    from openpi.training.train_config import get_config

    for config_name in [
        "pi05_ki_joint_query_b1k-multitask-ki_on_500step_fp32",
        "pi05_ki_joint_query_b1k-multitask-ki_off_200step_fp32",
    ]:
        config = get_config(config_name)
        assert config is not None, f"Config {config_name} not found"

        data_factory = config.data[0]
        # tasks/episodes_index live on base_config (DataConfig)
        base = data_factory.base_config
        assert hasattr(base, "tasks"), f"{config_name}: base_config has no 'tasks' field"
        assert hasattr(base, "episodes_index"), f"{config_name}: base_config has no 'episodes_index' field"

        assert base.tasks is not None, f"{config_name}: tasks should not be None"
        assert len(base.tasks) == 5, (
            f"{config_name}: expected 5 tasks, got {len(base.tasks)}"
        )
        assert len(base.episodes_index) == 20, (
            f"{config_name}: expected 20 eps/task, got {len(base.episodes_index)}"
        )
        assert base.subtask_source == "annotations_skill", (
            f"{config_name}: expected subtask_source=annotations_skill, got {base.subtask_source}"
        )


# ======================================================================
#  Single-task overfit configs (validation split)
# ======================================================================

_SINGLE_TASK_CONFIGS = [
    "pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp32",
    "pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp16",
    "pi05_ki_joint_query_b1k-single_task-radio-ki_on_bf16",
]


def test_single_task_configs_resolve():
    """Single-task overfit configs should be registered and resolve via get_config."""
    from openpi.training.train_config import get_config

    for name in _SINGLE_TASK_CONFIGS:
        config = get_config(name)
        assert config is not None, f"Config {name} not found"
        assert config.name == name


def test_single_task_configs_use_single_turning_on_radio():
    """Single-task configs should use exactly one task: turning_on_radio."""
    from openpi.training.train_config import get_config

    for name in _SINGLE_TASK_CONFIGS:
        config = get_config(name)
        data_factory = config.data[0]
        base = data_factory.base_config
        assert base.tasks is not None, f"{name}: tasks should not be None"
        assert len(base.tasks) == 1, (
            f"{name}: expected 1 task, got {len(base.tasks)}: {base.tasks}"
        )
        assert base.tasks[0] == "turning_on_radio", (
            f"{name}: expected task 'turning_on_radio', got '{base.tasks[0]}'"
        )


def test_single_task_train_episodes_180():
    """Single-task train data should have 180 episodes (0..179)."""
    from openpi.training.train_config import get_config

    for name in _SINGLE_TASK_CONFIGS:
        config = get_config(name)
        data_factory = config.data[0]
        base = data_factory.base_config
        assert len(base.episodes_index) == 180, (
            f"{name}: expected 180 train episodes, got {len(base.episodes_index)}"
        )
        assert base.episodes_index[0] == 0, (
            f"{name}: first train episode should be 0, got {base.episodes_index[0]}"
        )
        assert base.episodes_index[-1] == 179, (
            f"{name}: last train episode should be 179, got {base.episodes_index[-1]}"
        )


def test_single_task_val_data_config_exists():
    """Single-task configs should have val_data set (not None)."""
    from openpi.training.train_config import get_config

    for name in _SINGLE_TASK_CONFIGS:
        config = get_config(name)
        assert config.val_data, (
            f"{name}: val_data should not be empty"
        )
        assert len(config.val_data) == 1, (
            f"{name}: expected 1 val data config, got {len(config.val_data)}"
        )


def test_single_task_val_episodes_20_disjoint_from_train():
    """Single-task val data should have 20 episodes (180..199), disjoint from train."""
    from openpi.training.train_config import get_config

    for name in _SINGLE_TASK_CONFIGS:
        config = get_config(name)
        val_factory = config.val_data[0]
        val_base = val_factory.base_config

        # Same task
        assert val_base.tasks is not None
        assert len(val_base.tasks) == 1
        assert val_base.tasks[0] == "turning_on_radio"

        # 20 val episodes
        assert len(val_base.episodes_index) == 20, (
            f"{name}: expected 20 val episodes, got {len(val_base.episodes_index)}"
        )
        assert val_base.episodes_index[0] == 180, (
            f"{name}: first val episode should be 180, got {val_base.episodes_index[0]}"
        )
        assert val_base.episodes_index[-1] == 199, (
            f"{name}: last val episode should be 199, got {val_base.episodes_index[-1]}"
        )

        # Disjoint from train
        train_eps = set(config.data[0].base_config.episodes_index)
        val_eps = set(val_base.episodes_index)
        overlap = train_eps & val_eps
        assert len(overlap) == 0, (
            f"{name}: train and val episode indices overlap: {sorted(overlap)}"
        )


def test_single_task_bf16_hl_contract():
    """BF16 HL config should preserve the formal KI joint-query experiment contract."""
    from openpi.models.pi05_ki_joint_query_config import Pi05KIJointQueryConfig
    from openpi.training.train_config import get_config

    config = get_config("pi05_ki_joint_query_b1k-single_task-radio-ki_on_bf16")

    assert config.pytorch_model_name == "pi05_ki_joint_query"
    assert isinstance(config.model, Pi05KIJointQueryConfig)
    assert config.pytorch_training_precision == "bfloat16"
    assert config.accelerate_mixed_precision == "bf16"
    assert config.model.dtype == "bfloat16"
    assert config.model.knowledge_insulation is True
    assert config.model.action_horizon == 32
    assert config.model.num_query_tokens == 32
    assert config.model.truncate_expert_kv is True
    assert config.model.beta_text == 1.0
    assert config.model.beta_query == 1.0
    assert config.model.flow_loss_weight == 10.0
    assert config.pytorch_weight_path == "checkpoints/pi05_base_pytorch"
    assert config.data[0].base_config.tasks == ["turning_on_radio"]
    assert config.data[0].base_config.episodes_index == list(range(180))
    assert config.val_data[0].base_config.episodes_index == list(range(180, 200))
    assert config.data[0].base_config.subtask_source == "annotations_skill"
    assert config.data[0].base_config.behavior_dataset_root == (
        "/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/"
    )
    assert config.num_train_steps == 2000
    assert config.num_train_epochs == 1
    assert config.save_interval == 200
    assert config.checkpoint_policy == "step"
    assert config.rolling_checkpoint_interval == 1000
    assert config.val_log_interval == 100


def test_git_validated_lq_fp32_launcher_contract_remains_unchanged():
    """The 869ea7a LQ workflow remains the 4x8 V100 single-task reference."""
    script = _LQ_FP32_LAUNCHER.read_text()

    assert _LQ_FP32_LAUNCHER.is_file()
    assert _LQ_FP32_LAUNCHER.stat().st_mode & 0o111
    assert "Multi-Node: 4 nodes x 8 V100 = 32 GPUs" in script
    assert "Task: turning_on_radio (180 train / 20 val)" in script
    assert 'CONFIG_NAME="${CONFIG_NAME:-pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp32}"' in script
    assert 'TASK_NAME="${TASK_NAME:-turning_on_radio}"' in script
    assert 'NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"' in script
    assert 'PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-float32}"' in script
    assert 'ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_ds_zero2_v100_fp32.yaml}"' in script
    assert '--num_processes "${TOTAL_GPUS}"' in script
    assert '--num_machines "${NUM_NODES}"' in script
    assert '--same_network' in script


def test_headless_eval_launcher_scopes_viewer_dimensions_to_eval_custom():
    """Canonical eval.py must not receive eval_custom.py-only Hydra overrides."""
    script = _B1K_HEADLESS_EVAL_LAUNCHER.read_text()
    launch_eval = script.split("launch_eval() {", maxsplit=1)[1].split("\n}\n\nrun_single_checkpoint_mode()", maxsplit=1)[0]
    custom_marker = 'if [[ "$EVAL_ENTRYPOINT" == "eval_custom.py" ]]; then'
    eval_py_path, custom_path = launch_eval.split(custom_marker, maxsplit=1)
    custom_overrides = custom_path.split("\n  fi", maxsplit=1)[0]

    common_overrides = (
        'feature_args="$feature_args render_viewer_camera=$RENDER_VIEWER_CAMERA '
        'gui_viewport_only=$GUI_VIEWPORT_ONLY"'
    )
    assert common_overrides in eval_py_path
    assert "viewer_width=" not in eval_py_path
    assert "viewer_height=" not in eval_py_path
    assert 'feature_args="$feature_args viewer_width=$VIEWER_WIDTH viewer_height=$VIEWER_HEIGHT"' in custom_overrides
    assert launch_eval.count("viewer_width=") == 1
    assert launch_eval.count("viewer_height=") == 1


def test_full_task_bf16_hl_contract():
    """Full-task BF16 config should exactly encode the formal HL experiment."""
    from openpi.models.pi05_ki_joint_query_config import Pi05KIJointQueryConfig
    from openpi.training.train_config import get_config

    config = get_config(_FULL_TASK_BF16_CONFIG)
    train = config.data[0]
    val = config.val_data[0]
    single_task = get_config("pi05_ki_joint_query_b1k-single_task-radio-ki_on_bf16")

    assert config.name == _FULL_TASK_BF16_CONFIG
    assert config.pytorch_model_name == "pi05_ki_joint_query"
    assert isinstance(config.model, Pi05KIJointQueryConfig)
    assert config.pytorch_training_precision == "bfloat16"
    assert config.accelerate_mixed_precision == "bf16"
    assert config.model.dtype == "bfloat16"
    assert config.model.knowledge_insulation is True
    assert config.model.action_horizon == 32
    assert config.model.num_query_tokens == 32
    assert config.model.truncate_expert_kv is True
    assert config.model.beta_text == 1.0
    assert config.model.beta_query == 1.0
    assert config.model.flow_loss_weight == 10.0
    assert config.pytorch_weight_path == "checkpoints/pi05_base_pytorch"

    # No task filter means every challenge task. Episode indices are per task.
    assert train.base_config.tasks is None
    assert val.base_config.tasks is None
    assert train.base_config.episodes_index == list(range(180))
    assert val.base_config.episodes_index == list(range(180, 200))
    assert set(train.base_config.episodes_index).isdisjoint(val.base_config.episodes_index)
    assert train.base_config.behavior_dataset_root == (
        "/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/"
    )
    assert val.base_config.behavior_dataset_root == train.base_config.behavior_dataset_root
    assert train.base_config.subtask_source == "annotations_skill"
    assert val.base_config.subtask_source == "annotations_skill"
    assert train.base_config.prompt_from_task is True
    assert train.base_config.subtask_joiner == " then "
    assert train.base_config.subtask_template_path == single_task.data[0].base_config.subtask_template_path
    assert (
        train.base_config.subtask_object_name_mapping_path
        == single_task.data[0].base_config.subtask_object_name_mapping_path
    )
    assert train.assets.assets_dir == "checkpoints/pi05_base_pytorch/assets"
    assert train.assets.asset_id == "behavior-1k/2025-challenge-demos"

    # Lean formal mode has a fixed three-pass optimizer-step/LR budget.
    assert config.num_train_steps == 104_912
    assert config.num_train_epochs is None
    assert config.lr_schedule.warmup_steps == 1_000
    assert config.lr_schedule.peak_lr == 1e-5
    assert config.lr_schedule.decay_steps == 104_912
    assert config.lr_schedule.decay_lr == 0.0
    assert config.save_interval == 10_000
    assert config.checkpoint_policy == "step"
    assert config.val_log_interval == 1_000
    assert config.val_num_batches == 20
    assert config.batch_size_per_gpu == 8
    assert config.gradient_accumulation_steps == 1
    assert config.ema_decay is None
    assert config.wandb_enabled is True
    assert config.project_name == "pi05_ki"


def test_full_task_bf16_launcher_contract():
    """HL launcher should retain full-task, cache, topology, and budget defaults."""
    from openpi.training.train_config import get_config

    launcher = _FULL_TASK_BF16_LAUNCHER
    script = launcher.read_text()

    assert launcher.is_file()
    assert launcher.stat().st_mode & 0o111
    assert f'CONFIG_NAME="${{CONFIG_NAME:-{_FULL_TASK_BF16_CONFIG}}}"' in script
    assert 'NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-104912}"' in script
    assert 'NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-' not in script
    assert '--num-train-steps "${NUM_TRAIN_STEPS}"' in script
    assert '--num-train-epochs' not in script
    assert 'CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-step}"' in script
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"' in script
    assert '--checkpoint-policy "${CHECKPOINT_POLICY}"' in script
    assert '--save-interval "${SAVE_INTERVAL}"' in script
    assert "ROLLING_CHECKPOINT_INTERVAL" not in script
    assert "KEEP_PERIOD" not in script
    assert 'VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-1000}"' in script
    assert 'VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"' in script
    assert 'BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"' in script
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"' in script
    assert 'WARMUP_STEPS="${WARMUP_STEPS:-1000}"' in script
    assert 'PEAK_LR="${PEAK_LR:-1e-5}"' in script
    # Stride 4 / single offset 0, not the historical stride 12 / offsets 0,4,8.
    # This PRESERVES the anchor set rather than loosening it: the trainer no longer
    # rotates offsets at pass boundaries, and {0,4,8} mod 12 unions to exactly {0}
    # mod 4, so one stride-4 sweep selects the same anchors the three stride-12
    # passes selected, with 26,857,712 // 256 == 104,912 keeping the step budget
    # identical. FRAME_ANCHOR_* are now informational; OPENPI_B1K_ANCHOR_STRIDE is
    # the value the dataset actually reads and it must equal the config's stride,
    # or the trainer's formal contract validator fails closed on the mismatch.
    assert 'FRAME_ANCHOR_STRIDE="${FRAME_ANCHOR_STRIDE:-4}"' in script
    assert 'FRAME_ANCHOR_OFFSETS="${FRAME_ANCHOR_OFFSETS:-0}"' in script
    assert 'DROP_INCOMPLETE_ACTION_HORIZON="${DROP_INCOMPLETE_ACTION_HORIZON:-1}"' in script
    assert 'export OPENPI_B1K_ANCHOR_STRIDE="4"' in script
    assert get_config(_FULL_TASK_BF16_CONFIG).streaming_anchor_stride == 4
    assert 'export OPENPI_B1K_ANCHOR_OFFSET="0"' in script
    assert 'export OPENPI_B1K_DROP_INCOMPLETE_HORIZON="1"' in script
    assert 'export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-0}"' in script
    assert "24.9383588896%" in script
    assert "approximate coverage only" in script
    assert "formal resume unsupported" in script
    assert 'PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-bfloat16}"' in script
    assert 'PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"' in script
    assert 'FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-0}"' in script
    assert 'OPENPI_DATA_MANIFEST_PROBE_BATCHES="${OPENPI_DATA_MANIFEST_PROBE_BATCHES:-0}"' in script
    assert '--prepare-hf-cache-only' in script
    assert '--force-load-cache' in script
    assert "mutually exclusive" in script

    assert 'CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/navigation-hl/mlx/users/chenjunting/miniconda3}"' in script
    assert (
        'B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-/mnt/bn/navigation-hl/mlx/users/'
        'chenjunting/data/2025-challenge-demos/}"'
    ) in script
    assert 'B1K_ASSETS_DIR="${BASE_PI05_CKPT}/assets/behavior-1k/2025-challenge-demos"' in script
    assert 'NORM_STATS_PATH="${B1K_ASSETS_DIR}/norm_stats.json"' in script
    assert 'LOCAL_CACHE_ROOT="${_EXPLICIT_LOCAL_CACHE_ROOT:-/tmp/openpi-comet/' in script
    assert 'PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/${CONFIG_NAME}}"' in script
    assert 'export OPENPI_DATA_HOME="${LOCAL_CACHE_ROOT}/openpi"' in script
    assert 'export HF_DATASETS_CACHE="${HF_HOME}/datasets"' in script
    cache_preflight = script[
        script.index("_LOCAL_CACHE_DIRS=("):script.index("_cache_write_probe=")
    ]
    assert '"${HF_DATASETS_CACHE}"' not in cache_preflight
    for forbidden in (
        '-d "${HF_DATASETS_CACHE}"',
        '-r "${HF_DATASETS_CACHE}"',
        '-w "${HF_DATASETS_CACHE}"',
        '-x "${HF_DATASETS_CACHE}"',
        'mkdir -p "${HF_DATASETS_CACHE}"',
    ):
        assert forbidden not in script
    assert "generation-scoped c10d failure coordination" in script
    assert 'export TRITON_CACHE_DIR="${LOCAL_CACHE_ROOT}/triton/autotune"' in script
    assert 'export MPLCONFIGDIR="${LOCAL_CACHE_ROOT}/matplotlib"' in script
    assert 'LOCAL_TMP_BACKING="${LOCAL_CACHE_ROOT}/tmp"' in script
    assert 'TMP_ALIAS="/tmp/openpi-tmp-' in script
    assert 'export TMPDIR="${TMP_ALIAS}"' in script
    assert '[[ -e "${TMP_ALIAS}" && ! -L "${TMP_ALIAS}" ]]' in script
    assert "with multiprocess.Manager() as manager:" in script
    assert "MANAGER_SOCKET_REPRESENTATIVE_BYTES" in script
    assert "export OPENPI_HF_DATASETS_CACHE_PER_RANK=1" in script
    assert 'export OPENPI_HF_CACHE_RUN_ID="${CACHE_RUN_ID}"' in script
    assert 'CACHE_RUN_ID="${_MANAGED_CACHE_RUN_ID}"' in script
    assert "manual_local" not in script
    assert "manual LOCAL_CACHE_ROOT reuse requires FORCE_LOAD_CACHE=1" in script
    assert 'export OPENPI_HF_LOCAL_SYNC_TIMEOUT_S="${OPENPI_HF_LOCAL_SYNC_TIMEOUT_S:-7200}"' in script
    assert 'export OPENPI_HF_LOCAL_SYNC_POLL_S="${OPENPI_HF_LOCAL_SYNC_POLL_S:-2}"' in script
    assert "HF_LOCAL_SYNC_PREFLIGHT_OK" in script
    assert "no NCCL collective waits on Arrow I/O" in script
    assert 'export TORCH_NCCL_ASYNC_ERROR_HANDLING=' in script
    assert 'export NCCL_ASYNC_ERROR_HANDLING=' not in script
    assert 'df -P "${TRITON_CACHE_DIR}"' in script
    assert 'OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-' not in script
    assert 'HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-' not in script
    assert 'TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-' not in script
    assert 'TOKENIZER_SOURCE="${REPO_OPENPI_CACHE}/${TOKENIZER_REL}"' in script
    assert 'TOKENIZER_LOCAL="${OPENPI_DATA_HOME}/${TOKENIZER_REL}"' in script

    for arnold_variable in (
        "ARNOLD_WORKER_NUM",
        "ARNOLD_WORKER_GPU",
        "ARNOLD_ID",
        "ARNOLD_WORKER_0_HOST",
        "ARNOLD_WORKER_0_PORT",
    ):
        assert arnold_variable in script
    assert 'required_packages = ("accelerate", "deepspeed")' in script
    assert 'pin missing from environment.yml' in script
    assert 'package metadata missing' in script
    assert 'installed_version != expected_version' in script
    assert 'python -m accelerate.commands.launch' in script


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"RESUME": "1"}, "RESUME=1 is unsupported"),
        ({"WANDB_DISABLED": "1"}, "requires online Byted-W&B"),
        ({"WANDB_MODE": "offline"}, "requires WANDB_MODE=online"),
    ],
)
def test_full_task_launcher_rejects_unsupported_formal_runtime_modes(overrides, expected):
    result = _run_full_task_preflight(_manual_preflight_env(**overrides))
    assert result.returncode != 0
    assert expected in result.stderr


def test_full_task_prepare_only_bypasses_wandb_guard(tmp_path):
    run_id = f"prepare-only-{tmp_path.name}"
    user = f"pytest-{os.getpid()}-{tmp_path.name}"
    local_root = Path("/tmp/openpi-comet") / user / _FULL_TASK_BF16_CONFIG / run_id
    tmp_alias = _tmp_alias_for(local_root)
    try:
        result = _run_full_task_preflight(
            _manual_preflight_env(
                PREPARE_HF_CACHE_ONLY="1",
                WANDB_DISABLED="1",
                WANDB_MODE="offline",
                OPENPI_HF_CACHE_RUN_ID=run_id,
                USER=user,
            )
        )
        assert result.returncode == 0, result.stderr
        assert "LOCAL_CACHE_PREFLIGHT_OK" in result.stdout
    finally:
        if tmp_alias.is_symlink() and os.readlink(tmp_alias) == str(local_root / "tmp"):
            tmp_alias.unlink()
        shutil.rmtree(Path("/tmp/openpi-comet") / user, ignore_errors=True)


def test_full_task_launcher_manual_run_without_identity_fails_closed():
    missing_result = _run_full_task_preflight(_manual_preflight_env())
    whitespace_result = _run_full_task_preflight(
        _manual_preflight_env(OPENPI_HF_CACHE_RUN_ID="   ")
    )

    assert missing_result.returncode != 0
    assert "manual runs require a unique non-empty OPENPI_HF_CACHE_RUN_ID" in missing_result.stderr
    assert "manual_local" not in missing_result.stdout + missing_result.stderr
    assert "LOCAL_CACHE_PREFLIGHT_OK" not in missing_result.stdout
    assert whitespace_result.returncode != 0
    assert "must contain non-whitespace characters" in whitespace_result.stderr


def test_full_task_launcher_manual_run_with_explicit_identity_uses_run_specific_tmp_root(tmp_path):
    run_id = f"manual-run-{tmp_path.name}"
    user = f"pytest-{os.getpid()}-{tmp_path.name}"
    user_root = Path("/tmp/openpi-comet") / user
    local_root = user_root / _FULL_TASK_BF16_CONFIG / run_id
    tmp_alias = _tmp_alias_for(local_root)
    assert not local_root.exists()
    assert not tmp_alias.exists()
    env = _manual_preflight_env(OPENPI_HF_CACHE_RUN_ID=run_id, USER=user)

    try:
        result = _run_full_task_preflight(env)
        assert result.returncode == 0, result.stderr
        assert f"LOCAL_CACHE_ROOT={local_root}" in result.stdout
        assert f"OPENPI_HF_CACHE_RUN_ID={run_id}" in result.stdout
        assert not (local_root / "huggingface/datasets").exists()
    finally:
        if tmp_alias.is_symlink() and os.readlink(tmp_alias) == str(local_root / "tmp"):
            tmp_alias.unlink()
        shutil.rmtree(user_root, ignore_errors=True)


def test_full_task_launcher_explicit_cross_job_force_load_keeps_dataset_tree_read_only(tmp_path):
    local_root = tmp_path / "cross-job-cache"
    datasets_cache = local_root / "huggingface" / "datasets"
    artifact = datasets_cache / ".openpi_hf_cache_sync" / "prepared" / "manifest.ready.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"protocol_version": 3}\n')
    before = _tree_state(datasets_cache)
    tmp_alias = _tmp_alias_for(local_root)
    expected_digest = hashlib.sha256(os.fsencode(local_root)).hexdigest()[:24]
    env = _manual_preflight_env(LOCAL_CACHE_ROOT=str(local_root), FORCE_LOAD_CACHE="1")

    try:
        result = _run_full_task_preflight(env)
        assert result.returncode == 0, result.stderr
        assert f"OPENPI_HF_CACHE_RUN_ID=manual-cache-root-{expected_digest}" in result.stdout
        assert _tree_state(datasets_cache) == before
    finally:
        if tmp_alias.is_symlink() and os.readlink(tmp_alias) == str(local_root / "tmp"):
            tmp_alias.unlink()


def test_full_task_launcher_force_load_defers_missing_cache_without_creating_dataset_root(tmp_path):
    local_root = tmp_path / "missing-cross-job-cache"
    datasets_cache = local_root / "huggingface" / "datasets"
    tmp_alias = _tmp_alias_for(local_root)
    env = _manual_preflight_env(LOCAL_CACHE_ROOT=str(local_root), FORCE_LOAD_CACHE="1")

    try:
        result = _run_full_task_preflight(env)
        assert result.returncode == 0, result.stderr
        assert f"HF_DATASETS_CACHE={datasets_cache}" in result.stdout
        assert not datasets_cache.exists()
        assert "requires an existing HF_DATASETS_CACHE" not in result.stderr
    finally:
        if tmp_alias.is_symlink() and os.readlink(tmp_alias) == str(local_root / "tmp"):
            tmp_alias.unlink()


def test_full_task_launcher_local_cache_preflight_overrides_inherited_component_paths(tmp_path):
    """Long job roots must use a short Manager-safe TMP alias without cache leaks."""
    local_root = (
        tmp_path
        / "openpi-comet"
        / "preflight-user-with-a-realistically-long-name"
        / _FULL_TASK_BF16_CONFIG
        / "arnold-job-20260728-164904-restarted"
    )
    local_tmp_backing = local_root / "tmp"
    previous_manager_socket = local_tmp_backing / "pymp-12345678" / "listener-12345678"
    assert len(os.fsencode(previous_manager_socket)) > 107

    alias_digest = hashlib.sha256(os.fsencode(local_root)).hexdigest()[:24]
    tmp_alias = Path("/tmp") / f"openpi-tmp-{os.getuid()}-{alias_digest}"
    stale_alias_target = tmp_path / "stale-tmp-target"
    assert not tmp_alias.exists()
    assert not tmp_alias.is_symlink()
    tmp_alias.symlink_to(stale_alias_target)

    inherited_nas_cache = "/mnt/bn/behavior-data-hl/chenjunting/.cache/huggingface"
    inherited_home_cache = "/home/tiger/.triton/autotune"
    inherited_tmp = "/mnt/bn/shared/tmp"
    env = os.environ.copy()
    env.update(
        {
            "OPENPI_LAUNCH_PREFLIGHT_ONLY": "1",
            "LOCAL_CACHE_ROOT": str(local_root),
            "CONFIG_NAME": _FULL_TASK_BF16_CONFIG,
            "ARNOLD_JOB_ID": "20260728-164904-restarted",
            "CONDA_ROOT": "/does/not/exist-preflight-must-not-activate-conda",
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}",
            "OPENPI_DATA_HOME": f"{inherited_nas_cache}/openpi",
            "HF_HOME": inherited_nas_cache,
            "HF_HUB_CACHE": f"{inherited_nas_cache}/hub-via-hf",
            "HUGGINGFACE_HUB_CACHE": f"{inherited_nas_cache}/hub-via-legacy",
            "HF_DATASETS_CACHE": f"{inherited_nas_cache}/datasets",
            "HF_MODULES_CACHE": f"{inherited_nas_cache}/modules",
            "HF_ASSETS_CACHE": f"{inherited_nas_cache}/assets",
            "HF_XET_CACHE": f"{inherited_nas_cache}/xet",
            "TRANSFORMERS_CACHE": f"{inherited_nas_cache}/transformers",
            "TRITON_CACHE_DIR": inherited_home_cache,
            "XDG_CACHE_HOME": "/home/tiger/.cache",
            "MPLCONFIGDIR": "/home/tiger/.config/matplotlib",
            "TORCH_HOME": "/home/tiger/.cache/torch",
            "TORCHINDUCTOR_CACHE_DIR": "/home/tiger/.cache/torch/inductor",
            "TORCH_EXTENSIONS_DIR": "/home/tiger/.cache/torch/extensions",
            "TMPDIR": inherited_tmp,
            "TMP": inherited_tmp,
            "TEMP": inherited_tmp,
            "OPENPI_HF_DATASETS_CACHE_PER_RANK": "0",
            "OPENPI_HF_LOCAL_SYNC_TIMEOUT_S": "8100",
            "OPENPI_HF_LOCAL_SYNC_POLL_S": "0.5",
        }
    )
    env.pop("PERSISTENT_OUTPUT_ROOT", None)

    def run_preflight():
        return subprocess.run(
            ["bash", str(_FULL_TASK_BF16_LAUNCHER)],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    try:
        result = run_preflight()
        assert result.returncode == 0, result.stderr
        assert "HF_LOCAL_SYNC_PREFLIGHT_OK" in result.stdout
        assert "MULTIPROCESS_MANAGER_PREFLIGHT_OK" in result.stdout
        assert "LOCAL_CACHE_PREFLIGHT_OK; distributed launch skipped" in result.stdout

        # A second pass exercises idempotent reuse of the already-correct alias.
        repeated_result = run_preflight()
        assert repeated_result.returncode == 0, repeated_result.stderr
        assert "MULTIPROCESS_MANAGER_PREFLIGHT_OK" in repeated_result.stdout

        resolved = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                resolved[key] = value

        expected_bulk_suffixes = {
            "OPENPI_DATA_HOME": "openpi",
            "HF_HOME": "huggingface",
            "HF_HUB_CACHE": "huggingface/hub",
            "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
            "HF_MODULES_CACHE": "huggingface/modules",
            "HF_ASSETS_CACHE": "huggingface/assets",
            "HF_XET_CACHE": "huggingface/xet",
            "TRANSFORMERS_CACHE": "huggingface/transformers",
            "TRITON_CACHE_DIR": "triton/autotune",
            "XDG_CACHE_HOME": "xdg",
            "MPLCONFIGDIR": "matplotlib",
            "TORCH_HOME": "torch",
            "TORCHINDUCTOR_CACHE_DIR": "torch/inductor",
            "TORCH_EXTENSIONS_DIR": "torch/extensions",
        }
        assert resolved["LOCAL_CACHE_ROOT"] == str(local_root)
        for variable, suffix in expected_bulk_suffixes.items():
            expected = local_root / suffix
            assert resolved[variable] == str(expected), variable
            assert expected.is_dir(), variable
            assert expected.is_relative_to(local_root), variable
        deferred_datasets_cache = local_root / "huggingface/datasets"
        assert resolved["HF_DATASETS_CACHE"] == str(deferred_datasets_cache)
        assert not deferred_datasets_cache.exists()

        assert resolved["LOCAL_TMP_BACKING"] == str(local_tmp_backing)
        assert resolved["TMP_ALIAS"] == str(tmp_alias)
        assert resolved["TMPDIR"] == str(tmp_alias)
        assert resolved["TMP"] == str(tmp_alias)
        assert resolved["TEMP"] == str(tmp_alias)
        assert resolved["TEMPFILE_GETTEMPDIR"] == str(tmp_alias)
        assert resolved["TMPDIR_REALPATH"] == str(local_tmp_backing.resolve())
        assert tmp_alias.is_symlink()
        assert os.readlink(tmp_alias) == str(local_tmp_backing)
        assert tmp_alias.resolve() == local_tmp_backing.resolve()
        assert not str(tmp_alias).startswith(str(local_root))

        assert int(resolved["MANAGER_SOCKET_REPRESENTATIVE_BYTES"]) <= 107
        assert int(resolved["MULTIPROCESS_MANAGER_SOCKET_BYTES"]) <= 107
        assert resolved["AF_UNIX_PATH_MAX_BYTES"] == "107"
        assert resolved["OPENPI_HF_DATASETS_CACHE_PER_RANK"] == "1"
        assert resolved["OPENPI_HF_CACHE_RUN_ID"] == "20260728-164904-restarted"
        assert resolved["OPENPI_HF_LOCAL_SYNC_TIMEOUT_S"] == "8100"
        assert resolved["OPENPI_HF_LOCAL_SYNC_POLL_S"] == "0.5"
        assert resolved["PERSISTENT_OUTPUT_ROOT"] == str(
            _REPO_ROOT / "outputs" / _FULL_TASK_BF16_CONFIG
        )
        combined_output = result.stdout + result.stderr
        assert inherited_nas_cache not in combined_output
        assert inherited_home_cache not in combined_output
        assert inherited_tmp not in combined_output
        assert "accelerate.commands.launch" not in combined_output
    finally:
        if tmp_alias.is_symlink() and os.readlink(tmp_alias) in {
            str(stale_alias_target),
            str(local_tmp_backing),
        }:
            tmp_alias.unlink()


def test_full_task_launcher_rejects_invalid_local_sync_settings(tmp_path):
    """Cache synchronization bounds must fail before distributed launch."""
    local_root = tmp_path / "invalid-local-sync-cache-root"
    alias_digest = hashlib.sha256(os.fsencode(local_root)).hexdigest()[:24]
    tmp_alias = Path("/tmp") / f"openpi-tmp-{os.getuid()}-{alias_digest}"
    env = os.environ.copy()
    env.update(
        {
            "OPENPI_LAUNCH_PREFLIGHT_ONLY": "1",
            "LOCAL_CACHE_ROOT": str(local_root),
            "OPENPI_HF_CACHE_RUN_ID": "manual-invalid-local-sync-test",
            "CONFIG_NAME": _FULL_TASK_BF16_CONFIG,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}",
            "OPENPI_HF_LOCAL_SYNC_TIMEOUT_S": "1",
            "OPENPI_HF_LOCAL_SYNC_POLL_S": "2",
        }
    )
    try:
        result = subprocess.run(
            ["bash", str(_FULL_TASK_BF16_LAUNCHER)],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode != 0
        assert "OPENPI_HF_LOCAL_SYNC_POLL_S must not exceed" in result.stderr
        assert "HF_LOCAL_SYNC_PREFLIGHT_OK" not in result.stdout
        assert "accelerate.commands.launch" not in result.stdout + result.stderr
    finally:
        if tmp_alias.is_symlink() and os.readlink(tmp_alias) == str(local_root / "tmp"):
            tmp_alias.unlink()


def test_full_task_launcher_refuses_non_symlink_tmp_alias(tmp_path):
    """Alias setup must never delete an arbitrary non-symlink /tmp path."""
    local_root = tmp_path / "non-symlink-alias-cache-root"
    alias_digest = hashlib.sha256(os.fsencode(local_root)).hexdigest()[:24]
    tmp_alias = Path("/tmp") / f"openpi-tmp-{os.getuid()}-{alias_digest}"
    assert not tmp_alias.exists()
    assert not tmp_alias.is_symlink()
    tmp_alias.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "OPENPI_LAUNCH_PREFLIGHT_ONLY": "1",
            "LOCAL_CACHE_ROOT": str(local_root),
            "OPENPI_HF_CACHE_RUN_ID": "manual-non-symlink-alias-test",
            "CONFIG_NAME": _FULL_TASK_BF16_CONFIG,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}",
        }
    )
    try:
        result = subprocess.run(
            ["bash", str(_FULL_TASK_BF16_LAUNCHER)],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode != 0
        assert f"refusing to replace non-symlink TMP alias path: {tmp_alias}" in result.stderr
        assert tmp_alias.is_dir()
        assert not tmp_alias.is_symlink()
        assert "MULTIPROCESS_MANAGER_PREFLIGHT_OK" not in result.stdout
        assert "accelerate.commands.launch" not in result.stdout + result.stderr
    finally:
        if tmp_alias.is_dir() and not tmp_alias.is_symlink():
            tmp_alias.rmdir()


def test_full_task_launcher_tyro_arguments_parse(monkeypatch):
    """Every training argument emitted by the launcher should parse through Tyro."""
    from openpi.training.train_config import cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_accelerate.py",
            _FULL_TASK_BF16_CONFIG,
            "--pytorch-weight-path",
            "/shared/checkpoints/pi05_base_pytorch",
            "--exp-name",
            "tyro_full_b1k_contract",
            "--pytorch-training-precision",
            "bfloat16",
            "--num-train-steps",
            "104912",
            "--checkpoint-policy",
            "step",
            "--save-interval",
            "10000",
            "--val-log-interval",
            "1000",
            "--val-num-batches",
            "20",
            "--batch-size-per-gpu",
            "8",
            "--num-workers",
            "4",
            "--gradient-accumulation-steps",
            "1",
            "--assets-base-dir",
            "/shared/outputs/assets",
            "--checkpoint-base-dir",
            "/shared/outputs/checkpoints",
            "--log-base-dir",
            "/shared/outputs/logs",
        ],
    )

    parsed = cli()
    assert parsed.name == _FULL_TASK_BF16_CONFIG
    assert parsed.pytorch_weight_path == "/shared/checkpoints/pi05_base_pytorch"
    assert parsed.exp_name == "tyro_full_b1k_contract"
    assert parsed.pytorch_training_precision == "bfloat16"
    assert parsed.num_train_steps == 104_912
    assert parsed.num_train_epochs is None
    assert parsed.save_interval == 10_000
    assert parsed.checkpoint_policy == "step"
    assert parsed.val_log_interval == 1_000
    assert parsed.val_num_batches == 20
    assert parsed.batch_size_per_gpu == 8
    assert parsed.num_workers == 4
    assert parsed.gradient_accumulation_steps == 1
    assert parsed.assets_base_dir == "/shared/outputs/assets"
    assert parsed.checkpoint_base_dir == "/shared/outputs/checkpoints"
    assert parsed.log_base_dir == "/shared/outputs/logs"
    assert parsed.wandb_enabled is True


def test_full_task_launcher_uses_bf16_zero2_without_optimizer_offload():
    """Accelerate/DeepSpeed files selected by the launcher must be BF16 ZeRO-2."""
    accelerate_config = (_REPO_ROOT / "configs/accelerate_ds_zero2.yaml").read_text()
    deepspeed_config = json.loads((_REPO_ROOT / "configs/deepspeed_zero2.json").read_text())
    zero = deepspeed_config["zero_optimization"]

    assert "distributed_type: DEEPSPEED" in accelerate_config
    assert "deepspeed_config_file: configs/deepspeed_zero2.json" in accelerate_config
    assert zero["stage"] == 2
    assert "offload_optimizer" not in zero
    assert "offload_param" not in zero
    assert deepspeed_config["bf16"]["enabled"] is True
    assert deepspeed_config["torch_autocast"]["enabled"] is True
    assert deepspeed_config["torch_autocast"]["dtype"] == "bfloat16"


def test_full_task_launcher_dependency_pins_match_environment():
    """Launcher dependency gate should source the exact committed training pins."""
    environment = (_REPO_ROOT / "environment.yml").read_text()

    assert "- accelerate==1.13.0" in environment
    assert "- deepspeed==0.18.8" in environment


def test_single_task_fp32_fp16_differ_only_in_precision():
    """FP32 and FP16 single-task configs should differ only in precision settings."""
    from openpi.training.train_config import get_config

    fp32 = get_config("pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp32")
    fp16 = get_config("pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp16")

    # Both should have KI=ON, same task, same train/val split
    assert fp32.model.knowledge_insulation == fp16.model.knowledge_insulation == True
    assert fp32.data[0].base_config.tasks == fp16.data[0].base_config.tasks
    assert fp32.data[0].base_config.episodes_index == fp16.data[0].base_config.episodes_index
    assert fp32.val_data[0].base_config.episodes_index == fp16.val_data[0].base_config.episodes_index
    assert fp32.num_train_epochs == fp16.num_train_epochs
    assert fp32.batch_size_per_gpu == fp16.batch_size_per_gpu
    assert fp32.val_log_interval == fp16.val_log_interval
    assert fp32.val_num_batches == fp16.val_num_batches
    assert fp32.save_interval == fp16.save_interval

    # Precision should differ
    assert fp32.pytorch_training_precision == "float32"
    assert fp16.pytorch_training_precision == "float16"
    assert fp32.accelerate_mixed_precision == "no"
    assert fp16.accelerate_mixed_precision == "fp16"


def test_single_task_configs_have_val_log_interval_and_val_num_batches():
    """Single-task configs should have non-default val_log_interval and val_num_batches."""
    from openpi.training.train_config import get_config

    for name in _SINGLE_TASK_CONFIGS:
        config = get_config(name)
        assert config.val_log_interval > 0, f"{name}: val_log_interval should be > 0"
        assert config.val_num_batches > 0, f"{name}: val_num_batches should be > 0"


def test_single_task_configs_have_num_train_epochs():
    """Single-task configs should use num_train_epochs=1 (epoch-based training)."""
    from openpi.training.train_config import get_config

    for name in _SINGLE_TASK_CONFIGS:
        config = get_config(name)
        assert config.num_train_epochs == 1, (
            f"{name}: expected num_train_epochs=1, got {config.num_train_epochs}"
        )
