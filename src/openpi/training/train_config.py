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

from openpi.training.data_config import DataConfigFactory, FakeDataConfig

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

    data: Sequence[DataConfigFactory] = dataclasses.field(default_factory=lambda: [FakeDataConfig()])

    sample_weights: list[float] | None = None

    assets_base_dir: str = "./outputs/assets/train"

    checkpoint_base_dir: str = "./checkpoints"

    seed: int = 42
    batch_size: int = 32
    batch_size_per_gpu: int | None = None
    num_workers: int = 2
    num_train_steps: int = 30_000
    num_train_epochs: int | None = None

    log_interval: int = 100
    save_interval: int = 5000
    keep_period: int | None = 5000
    save_at_epoch_end_only: bool = False

    overwrite: bool = False
    resume: bool = False
    prepare_hf_cache_only: bool = False
    force_load_cache: bool = False

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
        # Compatibility: Allow passing a single DataConfigFactory by wrapping it in a list.
        if not isinstance(self.data, (list, tuple)):
            object.__setattr__(self, "data", [self.data])

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


from openpi.training.pretrain_config import _PRETRAIN_CONFIGS
from openpi.training.rft_config import _RFT_CONFIGS
from openpi.training.sft_config import _SFT_CONFIGS
from openpi.training.test_config import _TEST_CONFIGS

_CONFIGS = [*_PRETRAIN_CONFIGS, *_SFT_CONFIGS, *_RFT_CONFIGS, *_TEST_CONFIGS]


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
