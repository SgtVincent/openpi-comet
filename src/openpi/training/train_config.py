"""See _CONFIGS for the list of available configs."""

from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any, Literal, TypeAlias

import flax.nnx as nnx
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
from openpi.models.vlm2_vla_config import VLM2VLAConfig
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

from openpi.training.data_config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    FakeDataConfig,
    LeRobotB1KDataConfig,
)

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING

    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    pytorch_weight_path: str | None = None

    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    pytorch_model_name: Literal["pi0", "vlm2"] = "pi0"

    vlm2_geometry_dim: int = 512
    vlm2_view_dim: int = 512
    vlm2_working_memory_size: int = 8
    vlm2_episodic_memory_capacity: int = 32
    vlm2_episodic_similarity_threshold: float = 0.7
    vlm2_episodic_fusion_alpha: float = 0.5
    vlm2_num_frames: int = 3

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)

    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)

    ema_decay: float | None = 0.99

    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    data: Sequence[DataConfigFactory] | DataConfigFactory = dataclasses.field(default_factory=lambda: [FakeDataConfig()])

    sample_weights: list[float] | None = None

    assets_base_dir: str = "./outputs/assets/train"

    checkpoint_base_dir: str = "./checkpoints"

    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000

    log_interval: int = 100
    save_interval: int = 5000
    keep_period: int | None = 5000

    overwrite: bool = False
    resume: bool = False

    wandb_enabled: bool = True

    rank0_only_output: bool = True

    policy_metadata: dict[str, Any] | None = None

    fsdp_devices: int = 1

    val_log_interval: int = 100
    val_batch_size: int | None = None
    val_num_batches: int = 10
    val_repo_id: str | None = None
    val_episodes_index: list[int] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")

        return (pathlib.Path(self.checkpoint_base_dir) / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def eps_index_fn(*indexs):
    eps_index = []
    for item in indexs:
        if isinstance(item, (list, tuple)):
            eps_index.extend(list(range(item[0], item[1])))
        else:
            eps_index.extend(list(range(item)))
    return eps_index


_CONFIGS = [
    TrainConfig(
        name="pi05_b1k-base",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=["turning_on_radio"],
                fine_grained_level=0,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/",
                tasks=["turning_on_radio"],
                fine_grained_level=0,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi05-b1kpt50-cs32/params"),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-pt12_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=[
                    "turning_on_radio",
                    "picking_up_trash",
                    "hiding_Easter_eggs",
                    "wash_a_baseball_cap",
                    "hanging_pictures",
                    "attach_a_camera_to_a_tripod",
                    "make_microwave_popcorn",
                    "bringing_water",
                    "tidying_bedroom",
                    "putting_shoes_on_rack",
                    "setting_the_fire",
                    "cook_hot_dogs",
                ],
                fine_grained_level=0,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                fine_grained_level=0,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=VLM2VLAConfig(
            action_horizon=32,
            freeze_vggt_backbone=True,
            freeze_image_encoder=True,
        ),
        pytorch_model_name="vlm2",
        vlm2_num_frames=3,
        vlm2_geometry_dim=512,
        vlm2_view_dim=512,
        vlm2_working_memory_size=8,
        vlm2_episodic_memory_capacity=32,
        vlm2_episodic_similarity_threshold=0.7,
        vlm2_episodic_fusion_alpha=0.5,
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
                asset_id="behavior-1k/2025-challenge-demos",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/",
                fine_grained_level=0,
            ),
        ),
        pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 8,
    ),
    TrainConfig(
        name="pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=["turning_on_radio"],
                fine_grained_level=0,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 16,
    ),
    TrainConfig(
        name="vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft",
        exp_name="openpi",
        project_name="B1K",
        model=VLM2VLAConfig(action_horizon=32),
        pytorch_model_name="vlm2",
        vlm2_num_frames=3,
        vlm2_geometry_dim=512,
        vlm2_view_dim=512,
        vlm2_working_memory_size=8,
        vlm2_episodic_memory_capacity=32,
        vlm2_episodic_similarity_threshold=0.7,
        vlm2_episodic_fusion_alpha=0.5,
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
                asset_id="behavior-1k/2025-challenge-demos",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/",
                tasks=["turning_on_radio"],
                fine_grained_level=0,
            ),
        ),
        pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
        weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 8,
    ),
    TrainConfig(
        name="pi05_b1k-turning_on_radio_lr2.5e-6_step20k_rft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos-rft",
                tasks=["turning_on_radio"],
                fine_grained_level=0,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("path_to_your_pretrained_checkpoint"),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05-b1k-demo0_6-comet0_4-step20k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        sample_weights=[0.6, 0.4],
        data=[
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                    fine_grained_level=0,
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="../DATASETS/behavior/comet-1.5k",
                    fine_grained_level=0,
                ),
            ),
        ],
        weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="vlm2_8gpu_test",
        exp_name="vlm2_test_run",
        project_name="B1K",
        model=VLM2VLAConfig(action_horizon=32),
        pytorch_model_name="vlm2",
        vlm2_num_frames=3,
        vlm2_geometry_dim=512,
        vlm2_view_dim=512,
        vlm2_working_memory_size=8,
        vlm2_episodic_memory_capacity=32,
        vlm2_episodic_similarity_threshold=0.7,
        vlm2_episodic_fusion_alpha=0.5,
        data=FakeDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
        num_train_steps=10,
        log_interval=1,
        save_interval=5,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=10,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 4,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        logging.warning("Config '%s' not found, using default config 'pi05_b1k-base'", config_name)
        return _CONFIGS_DICT["pi05_b1k-base"]

    return _CONFIGS_DICT[config_name]
