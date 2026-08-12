"""π0.5-KI joint-query training configs.

TrainConfig entries for π0.5-KI joint-query MSE training, spanning legacy
Knowledge Insulation (KI) ON/OFF smoke tests and formal training runs.

This module contains both legacy experimental/smoke-test entries and formal
single-task/full-task-set training entries. The ``*_smoke*`` configs retain tiny
batches and step counts for quick validation; formal configs preserve their
training budgets and disjoint data splits.

Precision variants:
- ``*_smoke``: fp16 (intended production precision, unstable on V100)
- ``*_smoke_fp32``: float32 (V100 smoke test, numerically stable but slower)
- ``*_bf16``: bfloat16 (formal HL/Arnold training precision)
"""

from pathlib import Path

import openpi.models.pi05_ki_joint_query_config as pi05_ki_joint_query_config
import openpi.training.optimizer as _optimizer
from openpi.training.data_config import AssetsConfig, DataConfig, LeRobotB1KDataConfig
from openpi.training.skill_bridge_config import SkillBridgeConfig
from openpi.training.train_config import TrainConfig


_REPO_ROOT = Path(__file__).resolve().parents[3]
_B1K_DATA_ROOT = "/mnt/bn/saiwenresearch/mlx/users/chenjunting/data/2025-challenge-demos/"
_HL_B1K_DATA_ROOT = "/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/"
# Verified base checkpoint path (canonical repo; feat worktrees inherit this)
_CANONICAL_BASE_CKPT = (
    "/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch"
)
_B1K_SUBTASK_TEMPLATES = str(
    _REPO_ROOT / "src/behavior/learning/datas/b1k_subtask_phrase_templates.json"
)
_B1K_OBJECT_MAPPING = str(
    _REPO_ROOT / "src/behavior/learning/datas/b1k_object_id_name_mapping.json"
)
_PI05_BASE_CKPT = "checkpoints/pi05_base_pytorch"


def _make_b1k_subtask_data_config(episodes_index: list[int] | None = None) -> LeRobotB1KDataConfig:
    """Shared B1K data config with subtask annotations.

    Args:
        episodes_index: which episodes to use; None = first 2 (smoke-test scale)

    Returns:
        LeRobotB1KDataConfig with subtask_source="annotations_skill"
    """
    if episodes_index is None:
        episodes_index = list(range(2))  # smoke-test scale: 2 episodes only

    return LeRobotB1KDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        assets=AssetsConfig(
            assets_dir=f"{_PI05_BASE_CKPT}/assets",
            asset_id="behavior-1k/2025-challenge-demos",
        ),
        base_config=DataConfig(
            prompt_from_task=True,
            episodes_index=episodes_index,
            behavior_dataset_root=_B1K_DATA_ROOT,
            fine_grained_level=0,
            subtask_source="annotations_skill",
            subtask_template_path=_B1K_SUBTASK_TEMPLATES,
            subtask_object_name_mapping_path=_B1K_OBJECT_MAPPING,
            subtask_joiner=" then ",
        ),
    )


# Canonical 5-task subset for π0.5-KI joint query long baseline experiments.
# First 5 tasks from B1K challenge (indices 0-4 in TASK_NAMES_TO_INDICES).
# Selected as the standard multi-task baseline spanning diverse manipulation
# skills: articulation (radio), pick-and-place (trash), organizing (decorations),
# cleaning (plates), and food prep (canning).
_B1K_MULTITASK_TASKS: tuple[str, ...] = (
    "turning_on_radio",
    "picking_up_trash",
    "putting_away_Halloween_decorations",
    "cleaning_up_plates_and_food",
    "can_meat",
)


def _make_b1k_multitask_data_config(num_tasks: int = 5, episodes_per_task: int = 20) -> LeRobotB1KDataConfig:
    """Multi-task B1K data config with balanced episodes across tasks.

    Selects ``episodes_per_task`` episodes from each of the first ``num_tasks``
    tasks in ``_B1K_MULTITASK_TASKS`` for a balanced multi-task training subset.

    NOTE: ``episodes_index`` is applied **per selected task**, not globally.
    With tasks=[t1, t2, ..., tN] and episodes_index=[0..M-1], the total
    number of episodes is N × M, not M.

    Args:
        num_tasks: number of tasks to include (first N from _B1K_MULTITASK_TASKS)
        episodes_per_task: number of episodes per task

    Returns:
        LeRobotB1KDataConfig with subtask_source="annotations_skill"
    """
    if num_tasks > len(_B1K_MULTITASK_TASKS):
        raise ValueError(
            f"num_tasks={num_tasks} exceeds available tasks ({len(_B1K_MULTITASK_TASKS)}): "
            f"{_B1K_MULTITASK_TASKS}"
        )
    tasks = list(_B1K_MULTITASK_TASKS[:num_tasks])
    episodes_index = list(range(episodes_per_task))

    return LeRobotB1KDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        assets=AssetsConfig(
            assets_dir=f"{_PI05_BASE_CKPT}/assets",
            asset_id="behavior-1k/2025-challenge-demos",
        ),
        base_config=DataConfig(
            prompt_from_task=True,
            tasks=tasks,
            episodes_index=episodes_index,
            behavior_dataset_root=_B1K_DATA_ROOT,
            fine_grained_level=0,
            subtask_source="annotations_skill",
            subtask_template_path=_B1K_SUBTASK_TEMPLATES,
            subtask_object_name_mapping_path=_B1K_OBJECT_MAPPING,
            subtask_joiner=" then ",
        ),
    )


def _make_b1k_single_task_data_config(
    task_name: str,
    episodes_index: list[int],
    *,
    behavior_dataset_root: str = _B1K_DATA_ROOT,
    skill_bridge_enabled: bool = False,
    skill_bridge_min_pre: int = 1,
    skill_bridge_min_post: int = 1,
    base_assets_dir: str | None = None,
) -> LeRobotB1KDataConfig:
    """Single-task B1K data config with specific episode indices.

    Args:
        task_name: single task name (e.g. "turning_on_radio")
        episodes_index: list of episode indices for this task
        behavior_dataset_root: persistent B1K dataset location for this config
        skill_bridge_enabled: if True, enable skill bridge baseline
            (combined subtask_text for valid single-boundary crossings).
        skill_bridge_min_pre: minimum steps before boundary for valid bridge.
        skill_bridge_min_post: minimum steps after boundary for valid bridge.
        base_assets_dir: if provided, override default assets_dir (useful
            when base checkpoint lives outside the worktree).

    Returns:
        LeRobotB1KDataConfig with subtask_source="annotations_skill"
    """
    assets_dir = base_assets_dir or f"{_PI05_BASE_CKPT}/assets"
    return LeRobotB1KDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        assets=AssetsConfig(
            assets_dir=assets_dir,
            asset_id="behavior-1k/2025-challenge-demos",
        ),
        base_config=DataConfig(
            prompt_from_task=True,
            tasks=[task_name],
            episodes_index=episodes_index,
            behavior_dataset_root=behavior_dataset_root,
            fine_grained_level=0,
            subtask_source="annotations_skill",
            subtask_template_path=_B1K_SUBTASK_TEMPLATES,
            subtask_object_name_mapping_path=_B1K_OBJECT_MAPPING,
            subtask_joiner=" then ",
            skill_bridge=SkillBridgeConfig(
                enabled=skill_bridge_enabled,
                min_pre_boundary_steps=skill_bridge_min_pre,
                min_post_boundary_steps=skill_bridge_min_post,
            ),
        ),
    )


def _make_b1k_full_task_set_data_config(
    episodes_index: list[int],
    *,
    behavior_dataset_root: str = _B1K_DATA_ROOT,
    skill_bridge_enabled: bool = False,
    skill_bridge_min_pre: int = 1,
    skill_bridge_min_post: int = 1,
    base_assets_dir: str | None = None,
) -> LeRobotB1KDataConfig:
    """Full B1K challenge task-set data config with per-task episode indices.

    ``tasks`` is intentionally left unset so the dataset loader covers every B1K
    challenge task. ``episodes_index`` is applied independently to each task.

    Args:
        episodes_index: episode indices to use for every B1K challenge task
        behavior_dataset_root: persistent B1K dataset location for this config
        skill_bridge_enabled: if True, enable the skill bridge baseline that
            concatenates adjacent subtask spans across single skill boundaries.
        skill_bridge_min_pre: minimum steps before a boundary for a valid bridge.
        skill_bridge_min_post: minimum steps after a boundary for a valid bridge.
        base_assets_dir: if provided, override the default assets_dir (useful
            when the base checkpoint lives outside the worktree).

    Returns:
        LeRobotB1KDataConfig with no task filter and annotations_skill subtasks
    """
    assets_dir = base_assets_dir or f"{_PI05_BASE_CKPT}/assets"
    return LeRobotB1KDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        assets=AssetsConfig(
            assets_dir=assets_dir,
            asset_id="behavior-1k/2025-challenge-demos",
        ),
        base_config=DataConfig(
            prompt_from_task=True,
            episodes_index=episodes_index,
            behavior_dataset_root=behavior_dataset_root,
            fine_grained_level=0,
            subtask_source="annotations_skill",
            subtask_template_path=_B1K_SUBTASK_TEMPLATES,
            subtask_object_name_mapping_path=_B1K_OBJECT_MAPPING,
            subtask_joiner=" then ",
            skill_bridge=SkillBridgeConfig(
                enabled=skill_bridge_enabled,
                min_pre_boundary_steps=skill_bridge_min_pre,
                min_post_boundary_steps=skill_bridge_min_post,
            ),
        ),
    )


def _make_pi05_ki_joint_query_config(
    *,
    name: str,
    knowledge_insulation: bool,
    num_train_steps: int = 5,
    batch_size_per_gpu: int = 1,
    peak_lr: float = 1e-5,
    precision: str = "float16",
) -> TrainConfig:
    """Factory for π0.5-KI joint query query-MSE variant TrainConfig (smoke-test scale).

    Args:
        name: config name (used for get_config lookup)
        knowledge_insulation: whether to enable Knowledge Insulation
        num_train_steps: total training steps
        batch_size_per_gpu: per-GPU batch size
        peak_lr: peak learning rate
        precision: "float16" or "float32" (fp16 = intended production,
            fp32 = V100 smoke test stability workaround)

    Returns:
        TrainConfig for PI05KIJointQueryPytorch (query-MSE variant)
    """
    if precision == "float16":
        pytorch_precision = "float16"
        accel_mp = "fp16"
    elif precision == "float32":
        pytorch_precision = "float32"
        accel_mp = "no"
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # Each config gets its own output dir so KI ON/OFF and fp16/fp32
    # runs never overwrite each other's checkpoints or logs.
    output_root = f"./outputs/{name}"

    return TrainConfig(
        name=name,
        exp_name=name,  # unique per config
        project_name="pi05_ki",
        pytorch_model_name="pi05_ki_joint_query",
        model=pi05_ki_joint_query_config.Pi05KIJointQueryConfig(
            alpha=10.0,
            subtask_max_len=128,
            action_horizon=32,
            num_query_tokens=32,
            knowledge_insulation=knowledge_insulation,
            truncate_expert_kv=True,
            beta_text=1.0,
            beta_query=1.0,
            flow_loss_weight=10.0,
        ),
        data=_make_b1k_subtask_data_config(),
        pytorch_weight_path=_PI05_BASE_CKPT,
        num_train_steps=num_train_steps,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=peak_lr,
            decay_steps=num_train_steps,
        ),
        pytorch_training_precision=pytorch_precision,
        accelerate_mixed_precision=accel_mp,
        ema_decay=None,
        wandb_enabled=False,
        assets_base_dir=f"{output_root}/assets",
        checkpoint_base_dir=f"{output_root}/checkpoints",
        log_base_dir=f"{output_root}/logs",
        num_workers=2,
        batch_size_per_gpu=batch_size_per_gpu,
        gradient_accumulation_steps=1,
        save_interval=2,
        log_interval=1,
    )


def _make_pi05_ki_joint_query_long_baseline_config(
    *,
    name: str,
    knowledge_insulation: bool,
    num_train_steps: int,
    shared_warmup_steps: int = 50,
    shared_decay_steps: int = 500,
    shared_peak_lr: float = 1e-5,
    num_tasks: int = 5,
    episodes_per_task: int = 20,
    save_interval: int = 50,
    precision: str = "float32",
) -> TrainConfig:
    """Factory for π0.5-KI joint query query-MSE variant long-baseline TrainConfig.

    Both KI=ON and KI=OFF baselines share the same LR schedule
    (warmup + cosine decay) so that step-by-step LR values are
    identical for the first 200 steps.  KI=OFF simply stops earlier.

    Uses a balanced multi-task subset (num_tasks × episodes_per_task).

    Args:
        name: config name (used for get_config lookup)
        knowledge_insulation: whether to enable Knowledge Insulation
        num_train_steps: total training steps
        shared_warmup_steps: warmup steps (same for both baselines)
        shared_decay_steps: total decay steps (same for both baselines)
        shared_peak_lr: peak learning rate (same for both baselines)
        num_tasks: number of tasks for multi-task subset
        episodes_per_task: episodes per task
        save_interval: checkpoint save interval in steps
        precision: "float16" or "float32"

    Returns:
        TrainConfig for PI05KIJointQueryPytorch (query-MSE variant) long baseline
    """
    if precision == "float16":
        pytorch_precision = "float16"
        accel_mp = "fp16"
    elif precision == "float32":
        pytorch_precision = "float32"
        accel_mp = "no"
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    output_root = f"./outputs/{name}"

    return TrainConfig(
        name=name,
        exp_name=name,  # unique per config
        project_name="pi05_ki",
        pytorch_model_name="pi05_ki_joint_query",
        model=pi05_ki_joint_query_config.Pi05KIJointQueryConfig(
            alpha=10.0,
            subtask_max_len=128,
            action_horizon=32,
            num_query_tokens=32,
            knowledge_insulation=knowledge_insulation,
            truncate_expert_kv=True,
            beta_text=1.0,
            beta_query=1.0,
            flow_loss_weight=10.0,
        ),
        data=_make_b1k_multitask_data_config(
            num_tasks=num_tasks,
            episodes_per_task=episodes_per_task,
        ),
        pytorch_weight_path=_PI05_BASE_CKPT,
        num_train_steps=num_train_steps,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=shared_warmup_steps,
            peak_lr=shared_peak_lr,
            decay_steps=shared_decay_steps,
            decay_lr=0.0,
        ),
        pytorch_training_precision=pytorch_precision,
        accelerate_mixed_precision=accel_mp,
        ema_decay=None,
        wandb_enabled=False,
        assets_base_dir=f"{output_root}/assets",
        checkpoint_base_dir=f"{output_root}/checkpoints",
        log_base_dir=f"{output_root}/logs",
        num_workers=2,
        batch_size_per_gpu=1,
        gradient_accumulation_steps=1,
        save_interval=save_interval,
        log_interval=1,
    )


def _make_pi05_ki_joint_query_single_task_overfit_config(
    *,
    name: str,
    knowledge_insulation: bool = True,
    task_name: str = "turning_on_radio",
    train_episodes: int = 180,
    val_episodes_start: int = 180,
    val_episodes_end: int = 200,
    num_train_epochs: int = 1,
    peak_lr: float = 1e-5,
    warmup_steps: int = 100,
    save_interval: int = 200,
    val_log_interval: int = 50,
    val_num_batches: int = 20,
    precision: str = "float32",
    behavior_dataset_root: str = _B1K_DATA_ROOT,
    skill_bridge_enabled: bool = False,
    skill_bridge_min_pre: int = 1,
    skill_bridge_min_post: int = 1,
    base_checkpoint_path: str = _PI05_BASE_CKPT,
    base_assets_dir: str | None = None,
    output_root: str | None = None,
) -> TrainConfig:
    """Factory for single-task overfit experiment with validation split.

    Uses a single B1K task with disjoint train/val episode ranges:
    - Train: episodes [0, train_episodes)
    - Val: episodes [val_episodes_start, val_episodes_end)

    Both train and val share the same norm stats (from train data assets).

    Args:
        name: config name (used for get_config lookup)
        knowledge_insulation: whether to enable Knowledge Insulation
        task_name: B1K task name
        train_episodes: number of training episodes (0..train_episodes-1)
        val_episodes_start: start of val episode range (inclusive)
        val_episodes_end: end of val episode range (exclusive)
        num_train_epochs: number of training epochs (sets num_train_steps via steps_per_epoch)
        peak_lr: peak learning rate
        warmup_steps: warmup steps
        save_interval: checkpoint save interval in steps
        val_log_interval: validation interval in steps
        val_num_batches: number of val batches per validation run
        precision: "bfloat16", "float16", or "float32"
        behavior_dataset_root: persistent B1K dataset location for this config

    Returns:
        TrainConfig for PI05KIJointQueryPytorch (query-MSE variant) single-task overfit
    """
    if precision == "bfloat16":
        pytorch_precision = "bfloat16"
        accel_mp = "bf16"
    elif precision == "float16":
        pytorch_precision = "float16"
        accel_mp = "fp16"
    elif precision == "float32":
        pytorch_precision = "float32"
        accel_mp = "no"
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    train_episodes_index = list(range(train_episodes))
    val_episodes_index = list(range(val_episodes_start, val_episodes_end))

    if output_root is None:
        output_root = f"./outputs/{name}"

    # Decay steps = 1 epoch (estimated upper bound; exact steps_per_epoch
    # will be computed at runtime from the actual dataloader length).
    # Use a generous estimate that won't truncate early.
    estimated_steps_per_epoch = 2000  # conservative upper bound
    decay_steps = estimated_steps_per_epoch * num_train_epochs

    return TrainConfig(
        name=name,
        exp_name=name,  # unique per config
        project_name="pi05_ki",
        pytorch_model_name="pi05_ki_joint_query",
        model=pi05_ki_joint_query_config.Pi05KIJointQueryConfig(
            alpha=10.0,
            subtask_max_len=128,
            action_horizon=32,
            num_query_tokens=32,
            knowledge_insulation=knowledge_insulation,
            truncate_expert_kv=True,
            beta_text=1.0,
            beta_query=1.0,
            flow_loss_weight=10.0,
        ),
        data=_make_b1k_single_task_data_config(
            task_name,
            train_episodes_index,
            behavior_dataset_root=behavior_dataset_root,
            skill_bridge_enabled=skill_bridge_enabled,
            skill_bridge_min_pre=skill_bridge_min_pre,
            skill_bridge_min_post=skill_bridge_min_post,
            base_assets_dir=base_assets_dir,
        ),
        val_data=_make_b1k_single_task_data_config(
            task_name,
            val_episodes_index,
            behavior_dataset_root=behavior_dataset_root,
            skill_bridge_enabled=skill_bridge_enabled,
            skill_bridge_min_pre=skill_bridge_min_pre,
            skill_bridge_min_post=skill_bridge_min_post,
            base_assets_dir=base_assets_dir,
        ),
        pytorch_weight_path=base_checkpoint_path,
        num_train_steps=decay_steps,
        num_train_epochs=num_train_epochs,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            decay_steps=decay_steps,
            decay_lr=0.0,
        ),
        pytorch_training_precision=pytorch_precision,
        accelerate_mixed_precision=accel_mp,
        ema_decay=None,
        wandb_enabled=False,
        assets_base_dir=f"{output_root}/assets",
        checkpoint_base_dir=f"{output_root}/checkpoints",
        log_base_dir=f"{output_root}/logs",
        num_workers=2,
        batch_size_per_gpu=1,
        gradient_accumulation_steps=1,
        save_interval=save_interval,
        log_interval=10,
        val_log_interval=val_log_interval,
        val_num_batches=val_num_batches,
    )


def _make_pi05_ki_joint_query_full_task_set_bf16_config(
    *,
    name: str,
    train_episodes: int = 180,
    val_episodes_start: int = 180,
    val_episodes_end: int = 200,
    behavior_dataset_root: str = _HL_B1K_DATA_ROOT,
    skill_bridge_enabled: bool = False,
    skill_bridge_min_pre: int = 1,
    skill_bridge_min_post: int = 1,
    base_checkpoint_path: str = _PI05_BASE_CKPT,
    base_assets_dir: str | None = None,
    num_train_steps: int = 104_912,
    num_train_epochs: int | None = None,
    warmup_steps: int = 1_000,
    peak_lr: float = 1e-5,
    decay_steps: int = 104_912,
    decay_lr: float = 0.0,
    batch_size_per_gpu: int = 8,
    save_interval: int = 10_000,
    val_log_interval: int = 1_000,
    val_num_batches: int = 20,
    log_interval: int = 10,
    streaming_anchor_stride: int = 12,
) -> TrainConfig:
    """Formal lean BF16 config over the full B1K task set.

    Defaults reproduce the non-Skill-Bridge Run 2 control
    (``pi05_ki_joint_query_b1k-full_task-ki_on_bf16``): three streaming stride-12
    passes with offsets ``(0, 4, 8)``, a fixed 104,912 optimizer-step budget,
    warmup 1,000, peak LR 1e-5, cosine decay to 0, B8 (global batch 256 on
    32 GPUs), HL dataset root, base ``pi05`` checkpoint.

    Pass ``skill_bridge_enabled=True`` to obtain the Skill-Bridge variant that
    is otherwise matched to the control. The formal stride-12 / fixed-budget
    guard in ``train_accelerate.py`` keys on the control config name only, so
    the Skill-Bridge variant runs with the standard stride-1 data loader and
    the caller's ``num_train_epochs`` / ``num_train_steps`` budget.

    ``streaming_anchor_stride`` (default 12, matching the formal control) pins
    the B1K chunk-streaming anchor stride for the *training* loader. It is
    stored on ``TrainConfig``; the trainer applies it by setting
    ``OPENPI_B1K_ANCHOR_STRIDE`` scoped to the train-loader construction. The
    validation loader always runs with the baseline stride-1 / no-drop
    contract (see ``_baseline_b1k_dataset_env`` in ``train_accelerate.py``),
    so validation metrics are computed on the full-resolution data regardless
    of the training stride.
    """
    train_episodes_index = list(range(train_episodes))
    val_episodes_index = list(range(val_episodes_start, val_episodes_end))
    output_root = f"./outputs/{name}"

    return TrainConfig(
        name=name,
        exp_name=name,
        project_name="pi05_ki",
        pytorch_model_name="pi05_ki_joint_query",
        model=pi05_ki_joint_query_config.Pi05KIJointQueryConfig(
            alpha=10.0,
            subtask_max_len=128,
            action_horizon=32,
            num_query_tokens=32,
            knowledge_insulation=True,
            truncate_expert_kv=True,
            beta_text=1.0,
            beta_query=1.0,
            flow_loss_weight=10.0,
        ),
        data=_make_b1k_full_task_set_data_config(
            train_episodes_index,
            behavior_dataset_root=behavior_dataset_root,
            skill_bridge_enabled=skill_bridge_enabled,
            skill_bridge_min_pre=skill_bridge_min_pre,
            skill_bridge_min_post=skill_bridge_min_post,
            base_assets_dir=base_assets_dir,
        ),
        val_data=_make_b1k_full_task_set_data_config(
            val_episodes_index,
            behavior_dataset_root=behavior_dataset_root,
            skill_bridge_enabled=skill_bridge_enabled,
            skill_bridge_min_pre=skill_bridge_min_pre,
            skill_bridge_min_post=skill_bridge_min_post,
            base_assets_dir=base_assets_dir,
        ),
        pytorch_weight_path=base_checkpoint_path,
        num_train_steps=num_train_steps,
        num_train_epochs=num_train_epochs,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            decay_steps=decay_steps,
            decay_lr=decay_lr,
        ),
        pytorch_training_precision="bfloat16",
        accelerate_mixed_precision="bf16",
        ema_decay=None,
        wandb_enabled=True,
        assets_base_dir=f"{output_root}/assets",
        checkpoint_base_dir=f"{output_root}/checkpoints",
        log_base_dir=f"{output_root}/logs",
        num_workers=2,
        batch_size_per_gpu=batch_size_per_gpu,
        gradient_accumulation_steps=1,
        save_interval=save_interval,
        checkpoint_policy="step",
        rolling_checkpoint_interval=10_000,
        log_interval=log_interval,
        val_log_interval=val_log_interval,
        val_num_batches=val_num_batches,
        streaming_anchor_stride=streaming_anchor_stride,
    )


_PI05_KI_JOINT_QUERY_CONFIGS = [
    # --- fp16 base configs (intended production; may be unstable on V100) ---
    _make_pi05_ki_joint_query_config(
        name="pi05_ki_joint_query_b1k-ki_on_smoke",
        knowledge_insulation=True,
        precision="float16",
    ),
    _make_pi05_ki_joint_query_config(
        name="pi05_ki_joint_query_b1k-ki_off_smoke",
        knowledge_insulation=False,
        precision="float16",
    ),
    # --- fp32 V100 smoke variants (numerically stable, for validation only) ---
    _make_pi05_ki_joint_query_config(
        name="pi05_ki_joint_query_b1k-ki_on_smoke_fp32",
        knowledge_insulation=True,
        precision="float32",
    ),
    _make_pi05_ki_joint_query_config(
        name="pi05_ki_joint_query_b1k-ki_off_smoke_fp32",
        knowledge_insulation=False,
        precision="float32",
    ),
    # --- Long baselines: multi-task, shared LR schedule, fp32 ---
    # Both share identical LR schedule (warmup=50, decay=500, peak=1e-5)
    # KI=OFF stops at 200 steps on the same schedule curve
    _make_pi05_ki_joint_query_long_baseline_config(
        name="pi05_ki_joint_query_b1k-multitask-ki_on_500step_fp32",
        knowledge_insulation=True,
        num_train_steps=500,
        shared_warmup_steps=50,
        shared_decay_steps=500,
        shared_peak_lr=1e-5,
        num_tasks=5,
        episodes_per_task=20,
        save_interval=50,
        precision="float32",
    ),
    _make_pi05_ki_joint_query_long_baseline_config(
        name="pi05_ki_joint_query_b1k-multitask-ki_off_200step_fp32",
        knowledge_insulation=False,
        num_train_steps=200,
        shared_warmup_steps=50,
        shared_decay_steps=500,
        shared_peak_lr=1e-5,
        num_tasks=5,
        episodes_per_task=20,
        save_interval=50,
        precision="float32",
    ),
    # --- Single-task overfit: turning_on_radio, 180 train / 20 val, KI=ON ---
    # FP32 variant (numerically stable, reference baseline)
    _make_pi05_ki_joint_query_single_task_overfit_config(
        name="pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp32",
        knowledge_insulation=True,
        precision="float32",
    ),
    # FP16 variant (production precision, V100-accelerated)
    _make_pi05_ki_joint_query_single_task_overfit_config(
        name="pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp16",
        knowledge_insulation=True,
        precision="float16",
    ),
    # BF16 HL/Arnold variant. Keep the formal 2000-step cap and one-epoch
    # budget; train_accelerate.py uses whichever limit is reached first.
    _make_pi05_ki_joint_query_single_task_overfit_config(
        name="pi05_ki_joint_query_b1k-single_task-radio-ki_on_bf16",
        knowledge_insulation=True,
        precision="bfloat16",
        behavior_dataset_root=_HL_B1K_DATA_ROOT,
        save_interval=200,
        val_log_interval=100,
    ),
    # --- Skill bridge baseline: single-task radio, KI=ON, bridge enabled ---
    # Uses verified absolute local paths for base checkpoint/assets (canonical
    # repo) and outputs (feat worktree outputs dir).
    # Paired control (bridge disabled, same paths, 2000 steps, FP32)
    # For A/B comparison with the bridge variant on the same allocation.
    # num_train_epochs=1 × ~2000 steps/epoch ≈ 2000 total steps
    _make_pi05_ki_joint_query_single_task_overfit_config(
        name="pi05_ki_joint_query_b1k-single_task-radio-ki_on_control_fp32",
        knowledge_insulation=True,
        precision="float32",
        num_train_epochs=1,
        skill_bridge_enabled=False,
        base_checkpoint_path=_CANONICAL_BASE_CKPT,
        base_assets_dir=f"{_CANONICAL_BASE_CKPT}/assets",
        output_root=str(
            _REPO_ROOT / "outputs" / "pi05_ki_joint_query_b1k-single_task-radio-ki_on_control_fp32"
        ),
    ),
    # FP32 variant (V100 reference, numerically stable) — bridge enabled
    # num_train_epochs=1 × ~2000 steps/epoch ≈ 2000 total steps
    _make_pi05_ki_joint_query_single_task_overfit_config(
        name="pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_fp32",
        knowledge_insulation=True,
        precision="float32",
        num_train_epochs=1,
        skill_bridge_enabled=True,
        base_checkpoint_path=_CANONICAL_BASE_CKPT,
        base_assets_dir=f"{_CANONICAL_BASE_CKPT}/assets",
        output_root=str(
            _REPO_ROOT / "outputs" / "pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_fp32"
        ),
    ),
    # BF16 variant (A100/Arnold fast path)
    _make_pi05_ki_joint_query_single_task_overfit_config(
        name="pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_bf16",
        knowledge_insulation=True,
        precision="bfloat16",
        skill_bridge_enabled=True,
        base_checkpoint_path=_CANONICAL_BASE_CKPT,
        base_assets_dir=f"{_CANONICAL_BASE_CKPT}/assets",
        output_root=str(
            _REPO_ROOT / "outputs" / "pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_bf16"
        ),
    ),
    # Formal lean B8/W32 run: three stride-12 offsets provide approximate
    # quarter exposure; the exact fixed optimizer-step budget is 104,912.
    # This is the non-Skill-Bridge control (Tracking Run 2). Left untouched.
    _make_pi05_ki_joint_query_full_task_set_bf16_config(
        name="pi05_ki_joint_query_b1k-full_task-ki_on_bf16",
    ),
    # Full-B1K Skill Bridge variant (LQ, 8x8 A100, B4, 3 epochs, stride-1).
    # NOT a strict Run 2 A/B: warmstarts from the LQ base pi05 checkpoint
    # (Run 2's 360k-step warmstart lives on HL NAS and is unavailable on LQ),
    # uses 3 real stride-1 epochs (~1.26M optimizer steps) instead of Run 2's
    # fixed 104,912-step three stride-12 passes, and B4 on 64 GPUs for the same
    # global batch 256. The only intended algorithmic A/B difference vs the
    # control above is skill_bridge.enabled=True.
    _make_pi05_ki_joint_query_full_task_set_bf16_config(
        name="pi05_ki_joint_query_b1k-full_task-ki_on_skillbridge_bf16",
        behavior_dataset_root=_B1K_DATA_ROOT,
        skill_bridge_enabled=True,
        base_checkpoint_path=_CANONICAL_BASE_CKPT,
        base_assets_dir=f"{_CANONICAL_BASE_CKPT}/assets",
        num_train_steps=0,
        num_train_epochs=3,
        warmup_steps=10_000,
        peak_lr=1e-5,
        decay_steps=0,
        decay_lr=0.0,
        batch_size_per_gpu=4,
        save_interval=10_000,
        val_log_interval=2_000,
        streaming_anchor_stride=4,
    ),
]
