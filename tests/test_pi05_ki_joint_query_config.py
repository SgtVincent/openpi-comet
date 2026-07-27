"""Tests for π0.5-KI joint query training configs.

Verifies that the experimental π0.5-KI joint query query-MSE variant configs are properly registered
and have the expected model and KI settings.
"""

import pytest


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
