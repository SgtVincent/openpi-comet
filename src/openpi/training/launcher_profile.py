"""Resolve a registered TrainConfig for shell launchers without duplicating recipe literals."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
from typing import Any

from openpi.training.train_config import TrainConfig
from openpi.training.train_config import get_config


@dataclasses.dataclass(frozen=True)
class EffectiveRecipe:
    """Runtime recipe fields derived from one registered TrainConfig."""

    global_batch: int
    val_log_interval: int
    save_interval: int
    rolling_checkpoint_interval: int


def resolve_effective_recipe(config: TrainConfig, *, world_size: int) -> EffectiveRecipe:
    """Purely derive runtime fields without mutating the registered config."""

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    global_batch = (
        int(config.batch_size_per_gpu)
        * int(world_size)
        * int(config.gradient_accumulation_steps)
    )
    expected_global_batch = config.expected_global_batch
    if expected_global_batch is not None and global_batch != int(expected_global_batch):
        raise ValueError(
            f"profile {config.name!r} requires global batch {expected_global_batch}, got {global_batch} "
            f"(B{config.batch_size_per_gpu} x W{world_size} x "
            f"GA{config.gradient_accumulation_steps})"
        )

    def _steps_from_samples(samples: int | None, fallback: int, field: str) -> int:
        if samples is None:
            return int(fallback)
        if int(samples) <= 0:
            raise ValueError(f"{field} must be positive when set; got {samples!r}")
        return max(1, int(samples) // global_batch)

    val_log_interval = _steps_from_samples(
        config.val_interval_samples, config.val_log_interval, "val_interval_samples"
    )
    save_interval = _steps_from_samples(
        config.save_interval_samples, config.save_interval, "save_interval_samples"
    )
    rolling_checkpoint_interval = (
        save_interval
        if config.save_interval_samples is not None
        else int(config.rolling_checkpoint_interval)
    )
    return EffectiveRecipe(
        global_batch=global_batch,
        val_log_interval=val_log_interval,
        save_interval=save_interval,
        rolling_checkpoint_interval=rolling_checkpoint_interval,
    )


def materialize_effective_recipe(config: TrainConfig, *, world_size: int) -> TrainConfig:
    """Return a copy with the pure effective recipe materialized exactly once."""

    effective = resolve_effective_recipe(config, world_size=world_size)
    return dataclasses.replace(
        config,
        val_log_interval=effective.val_log_interval,
        save_interval=effective.save_interval,
        rolling_checkpoint_interval=effective.rolling_checkpoint_interval,
    )


@dataclasses.dataclass(frozen=True)
class LauncherProfile:
    name: str
    model_name: str
    pytorch_training_precision: str
    accelerate_mixed_precision: str
    batch_size_per_gpu: int
    gradient_accumulation_steps: int
    expected_global_batch: int | None
    num_train_steps: int
    num_train_epochs: int | None
    num_workers: int
    save_interval: int
    val_log_interval: int
    val_num_batches: int
    streaming_anchor_stride: int
    warmup_steps: int
    peak_lr: float
    decay_steps: int
    decay_lr: float
    checkpoint_policy: str
    gradient_checkpointing: bool
    weight_path: str
    train_data_root: str
    train_assets_dir: str
    action_token_max_len: int | None
    val_interval_samples: int | None
    save_interval_samples: int | None
    effective_val_log_interval: int
    effective_save_interval: int
    wandb_enabled: bool
    project_name: str

    def shell_values(self) -> dict[str, str]:
        values: dict[str, Any] = dataclasses.asdict(self)
        return {
            f"CFG_{key.upper()}": (
                "" if value is None else "1" if value is True else "0" if value is False else str(value)
            )
            for key, value in values.items()
        }


def resolve_launcher_profile(
    name: str, *, expected_model: str | None = None, world_size: int = 32
) -> LauncherProfile:
    config = get_config(name)
    if config.name != name:
        raise ValueError(
            f"requested config {name!r} resolved as {config.name!r}; refusing silent fallback"
        )
    if expected_model is not None and config.pytorch_model_name != expected_model:
        raise ValueError(
            f"config {name!r} resolved model {config.pytorch_model_name!r}, expected {expected_model!r}"
        )
    if len(config.data) != 1:
        raise ValueError(f"launcher profile {name!r} requires exactly one training data config")

    data = config.data[0]
    fields = {
        "batch_size_per_gpu": int(config.batch_size_per_gpu),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "num_workers": int(config.num_workers),
        "save_interval": int(config.save_interval),
        "val_log_interval": int(config.val_log_interval),
        "val_num_batches": int(config.val_num_batches),
        "streaming_anchor_stride": int(config.streaming_anchor_stride),
        "warmup_steps": int(config.lr_schedule.warmup_steps),
    }
    invalid = [key for key, value in fields.items() if value <= 0]
    if invalid:
        raise ValueError(f"launcher profile {name!r} has non-positive fields: {invalid}")
    effective = resolve_effective_recipe(config, world_size=world_size)
    expected_global_batch = (
        None if config.expected_global_batch is None else int(config.expected_global_batch)
    )
    if config.num_train_epochs is None and int(config.num_train_steps) <= 0:
        raise ValueError(f"fixed-step profile {name!r} requires num_train_steps > 0")
    if config.num_train_epochs is not None and int(config.num_train_epochs) <= 0:
        raise ValueError(f"epoch profile {name!r} requires num_train_epochs > 0")

    return LauncherProfile(
        name=config.name,
        model_name=config.pytorch_model_name,
        pytorch_training_precision=config.pytorch_training_precision,
        accelerate_mixed_precision=config.accelerate_mixed_precision,
        batch_size_per_gpu=fields["batch_size_per_gpu"],
        gradient_accumulation_steps=fields["gradient_accumulation_steps"],
        expected_global_batch=expected_global_batch,
        num_train_steps=int(config.num_train_steps),
        num_train_epochs=None if config.num_train_epochs is None else int(config.num_train_epochs),
        num_workers=fields["num_workers"],
        save_interval=fields["save_interval"],
        val_log_interval=fields["val_log_interval"],
        val_num_batches=fields["val_num_batches"],
        streaming_anchor_stride=fields["streaming_anchor_stride"],
        warmup_steps=fields["warmup_steps"],
        peak_lr=float(config.lr_schedule.peak_lr),
        decay_steps=int(config.lr_schedule.decay_steps),
        decay_lr=float(config.lr_schedule.decay_lr),
        checkpoint_policy=str(config.checkpoint_policy),
        gradient_checkpointing=bool(config.gradient_checkpointing),
        weight_path=str(config.pytorch_weight_path),
        train_data_root=str(data.base_config.behavior_dataset_root),
        train_assets_dir=str(data.assets.assets_dir),
        action_token_max_len=(
            int(config.model.action_token_max_len)
            if hasattr(config.model, "action_token_max_len")
            else None
        ),
        val_interval_samples=(
            None if config.val_interval_samples is None else int(config.val_interval_samples)
        ),
        save_interval_samples=(
            None if config.save_interval_samples is None else int(config.save_interval_samples)
        ),
        effective_val_log_interval=effective.val_log_interval,
        effective_save_interval=effective.save_interval,
        wandb_enabled=bool(config.wandb_enabled),
        project_name=str(config.project_name),
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name")
    parser.add_argument("--expected-model")
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()
    profile = resolve_launcher_profile(
        args.config_name, expected_model=args.expected_model, world_size=args.world_size
    )
    if args.format == "json":
        print(json.dumps(dataclasses.asdict(profile), sort_keys=True))
    else:
        for key, value in profile.shell_values().items():
            print(f"{key}={shlex.quote(value)}")


if __name__ == "__main__":
    _main()
