#!/usr/bin/env python3

"""Smoke-test PI05_HYBRID data wiring and hierarchical inference.

This script validates two paths:
1. Training-side sample/tokenization wiring: B1K sample -> subtask_text -> subtask token fields.
2. Inference-side hierarchical policy path: prompt -> generated subtask -> action chunk.

It follows the same local/offline dataset conventions as scripts/run_pi05_sft_make_pizza_5ep.sh.
"""

import argparse
import dataclasses
import logging
import os
import pathlib
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

LOGGER = logging.getLogger("pi05_hybrid_test")


def _trace(trace_file: pathlib.Path | None, message: str) -> None:
    if trace_file is None:
        return
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with trace_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_episodes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if ":" in value:
        start_str, end_str = value.split(":", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError(f"Invalid episode range: {value}")
        return list(range(start, end))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _configure_runtime_env(config_name: str) -> None:
    os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / ".cache" / "openpi"))
    os.environ.setdefault("B1K_VIDEO_BACKEND", "video_reader")
    os.environ.setdefault("OPENPI_OFFLINE", "1")
    os.environ.setdefault("OPENPI_SKIP_TRANSFORMERS_REPLACE_CHECK", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OPENPI_PERSISTENT_WORKERS", "1")
    os.environ.setdefault("OPENPI_DATALOADER_TIMEOUT_S", "600")
    os.environ.setdefault("OPENPI_DATALOADER_PREFETCH_FACTOR", "4")
    os.environ.setdefault("OPENPI_DATALOADER_PIN_MEMORY", "1")
    os.environ.setdefault("OPENPI_DDP_FIND_UNUSED_PARAMETERS", "0")
    os.environ.setdefault("OPENPI_DDP_STATIC_GRAPH", "1")
    os.environ.setdefault("OPENPI_LOAD_DATASET_NUM_PROC_CAP", "8")
    os.environ.setdefault("HF_DATASETS_CACHE", f"/opt/tiger/hf_datasets_cache/{config_name}/")


def _resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_checkpoint_dir(train_config, checkpoint_arg: str | None) -> pathlib.Path:
    if checkpoint_arg:
        checkpoint_dir = pathlib.Path(checkpoint_arg)
    elif train_config.pytorch_weight_path:
        checkpoint_dir = pathlib.Path(train_config.pytorch_weight_path)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = REPO_ROOT / checkpoint_dir
    else:
        checkpoint_dir = REPO_ROOT / "checkpoints" / "openpi_comet" / "pi05-b1kpt50-cs32"

    checkpoint_dir = checkpoint_dir.resolve()
    if not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"No model.safetensors found under checkpoint dir: {checkpoint_dir}")
    return checkpoint_dir


def _trace_model_load(message: str) -> None:
    trace_path = os.environ.get("OPENPI_POLICY_TRACE_FILE")
    if not trace_path:
        return
    trace_file = pathlib.Path(trace_path)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with trace_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


@dataclasses.dataclass(frozen=True)
class _HybridValidationDataConfig:
    asset_id: str
    norm_stats: dict[str, Any] | None
    repack_transforms: Any
    data_transforms: Any
    model_transforms: Any
    use_quantile_norm: bool


@dataclasses.dataclass(frozen=True)
class _HybridValidationDataFactory:
    repo_id: str
    behavior_dataset_root: str
    assets_dir: str
    asset_id: str
    episodes_index: list[int]
    fine_grained_level: int = 0
    action_sequence_keys: tuple[str, ...] = ("action",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_config",
            dataclasses.make_dataclass(
                "_BaseConfig",
                [
                    ("behavior_dataset_root", str),
                    ("episodes_index", list[int]),
                    ("fine_grained_level", int),
                    ("tolerance_s", float),
                    ("tasks", list[str] | None),
                    ("modalities", list[str]),
                    ("return_seg_instance", bool),
                    ("train_rgb_type", str),
                    ("skill_list", list[str]),
                ],
                frozen=True,
            )(
                behavior_dataset_root=self.behavior_dataset_root,
                episodes_index=self.episodes_index,
                fine_grained_level=self.fine_grained_level,
                tolerance_s=1e-4,
                tasks=None,
                modalities=["rgb"],
                return_seg_instance=False,
                train_rgb_type="regular",
                skill_list=["all"],
            ),
        )

    def create(self, assets_dirs: pathlib.Path, model_config) -> _HybridValidationDataConfig:
        from openpi.policies import b1k_policy
        import openpi.shared.download as _download
        import openpi.shared.normalize as _normalize
        from openpi import transforms as _transforms
        from openpi.models import tokenizer as _tokenizer

        norm_stats = None
        data_assets_dir = pathlib.Path(self.assets_dir)
        try:
            norm_stats = _normalize.load(_download.maybe_download(str(data_assets_dir / self.asset_id)))
        except FileNotFoundError:
            norm_stats = None

        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "subtask_text": "subtask_text",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizeHybridInputs(
                    _tokenizer.HybridTokenizer(
                        prompt_max_len=model_config.max_token_len,
                        subtask_max_len=model_config.subtask_max_len,
                    )
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
            outputs=[],
        )
        return _HybridValidationDataConfig(
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=True,
        )


@dataclasses.dataclass(frozen=True)
class _HybridValidationModelConfig:
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 512
    subtask_max_len: int = 128
    alpha: float = 10.0
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"
    pi05: bool = True
    discrete_state_input: bool = True

    @property
    def model_type(self):
        from openpi.models import model as _model

        return _model.ModelType.PI05_HYBRID

    def load_pytorch(self, train_config, weights_path: str):
        import safetensors.torch

        _trace_model_load("model_load:module_import_start")
        from openpi.models_pytorch.pi05_hybrid import PI05HybridPytorch
        _trace_model_load("model_load:module_import_done")

        _trace_model_load("model_load:model_name:hybrid")
        _trace_model_load("model_load:construct_start")
        model = PI05HybridPytorch(
            self,
            alpha=self.alpha,
            action_expert_name="hybrid",
        )
        _trace_model_load("model_load:construct_done")
        _trace_model_load("model_load:safetensors_start")
        safetensors.torch.load_model(model, weights_path, strict=True)
        _trace_model_load("model_load:safetensors_done")
        return model


def _load_train_config(config_name: str):
    if config_name != "pi05_hybrid_b1k-pt50_cs32_bs64_lr2.5e-5_5ep":
        from openpi.training import config as _config

        return _config.get_config(config_name)

    @dataclasses.dataclass(frozen=True)
    class _LightweightTrainConfig:
        name: str
        model: Any
        data: Any
        pytorch_weight_path: str | None
        assets_base_dir: str
        policy_metadata: dict[str, Any] | None = None

        @property
        def assets_dirs(self) -> pathlib.Path:
            return (REPO_ROOT / self.assets_base_dir / self.name).resolve()

    return _LightweightTrainConfig(
        name=config_name,
        model=_HybridValidationModelConfig(),
        data=_HybridValidationDataFactory(
            repo_id="behavior-1k/2025-challenge-demos",
            behavior_dataset_root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/",
            assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
            asset_id="behavior-1k/2025-challenge-demos",
            episodes_index=list(range(200)),
        ),
        pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
        assets_base_dir="outputs/assets",
    )


def _make_dummy_sample(prompt: str, subtask_text: str, action_horizon: int) -> dict[str, Any]:
    proprio_dim = 64
    return {
        "observation.images.rgb.head": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation.images.rgb.left_wrist": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation.images.rgb.right_wrist": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation.state": (np.random.randn(proprio_dim) * 0.05).astype(np.float32),
        "action": (np.random.randn(action_horizon, 23) * 0.05).astype(np.float32),
        "prompt": prompt,
        "task": prompt,
        "subtask_text": subtask_text,
    }


def _sanitize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sanitized[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            sanitized[key] = value
        elif isinstance(value, np.generic):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value
    return sanitized


def _primary_data_factory(train_config):
    if not train_config.data:
        raise ValueError("Train config does not define any data factory.")
    if isinstance(train_config.data, (list, tuple)):
        return train_config.data[0]
    return train_config.data


def _load_dataset_sample(train_config, sample_index: int, dataset_root: str | None) -> dict[str, Any]:
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    data_factory = _primary_data_factory(train_config)
    base_config = data_factory.base_config
    if base_config is None:
        raise ValueError("Primary data factory does not provide a base config.")

    root = dataset_root or base_config.behavior_dataset_root
    if root is None:
        raise ValueError("No behavior_dataset_root configured for dataset-backed validation.")

    root_path = pathlib.Path(root)

    def _existing_episode_positions() -> list[int]:
        episodes = base_config.episodes_index
        if not episodes:
            return []

        task_ids: list[int]
        if base_config.tasks:
            meta = BehaviorLeRobotDataset(
                repo_id=data_factory.repo_id,
                root=root,
                tolerance_s=base_config.tolerance_s,
                tasks=base_config.tasks,
                modalities=base_config.modalities,
                local_only=True,
                check_files=False,
                check_timestamp_sync=False,
                delta_timestamps=None,
                episodes=[],
                chunk_streaming_using_keyframe=True,
                shuffle=False,
                fine_grained_level=base_config.fine_grained_level,
                return_seg_instance=base_config.return_seg_instance,
                train_rgb_type=base_config.train_rgb_type,
                skill_list=base_config.skill_list,
            )
            task_ids = sorted(meta.task_indices)
        else:
            task_dirs = sorted((root_path / "data").glob("task-*"))
            task_ids = [int(path.name.split("-")[1]) for path in task_dirs if path.is_dir()]

        available_positions = set()
        for task_id in task_ids:
            available_files = sorted((root_path / "data" / f"task-{task_id:04d}").glob("episode_*.parquet"))
            for file_path in available_files:
                episode_id = int(file_path.stem.split("_")[1])
                position = episode_id % 10_000
                if position in episodes:
                    available_positions.add(position)
        return sorted(available_positions)

    delta_timestamps = {
        key: [t / 30.0 for t in range(train_config.model.action_horizon)]
        for key in getattr(data_factory, "action_sequence_keys", ("action",))
    }

    dataset_kwargs = dict(
        repo_id=data_factory.repo_id,
        root=root,
        tolerance_s=base_config.tolerance_s,
        tasks=base_config.tasks,
        modalities=base_config.modalities,
        local_only=True,
        check_files=False,
        check_timestamp_sync=False,
        delta_timestamps=delta_timestamps,
        episodes=base_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=False,
        fine_grained_level=base_config.fine_grained_level,
        return_seg_instance=base_config.return_seg_instance,
        train_rgb_type=base_config.train_rgb_type,
        skill_list=base_config.skill_list,
    )

    try:
        dataset = BehaviorLeRobotDataset(**dataset_kwargs)
    except FileNotFoundError as exc:
        available_positions = _existing_episode_positions()
        if not available_positions:
            raise
        LOGGER.warning(
            "Dataset init hit missing local parquet (%s). Retrying with %d locally available episode positions.",
            exc,
            len(available_positions),
        )
        dataset = BehaviorLeRobotDataset(**{**dataset_kwargs, "episodes": available_positions})
    return _sanitize_sample(dataset[sample_index])


def _load_raw_sample(args, train_config) -> tuple[str, dict[str, Any]]:
    if args.source == "dummy":
        return "dummy", _make_dummy_sample(args.prompt, args.subtask_text, train_config.model.action_horizon)

    if args.source == "dataset":
        return "dataset", _load_dataset_sample(train_config, args.sample_index, args.dataset_root)

    try:
        return "dataset", _load_dataset_sample(train_config, args.sample_index, args.dataset_root)
    except Exception as exc:
        LOGGER.warning("Falling back to dummy sample because dataset load failed: %s", exc)
        return "dummy", _make_dummy_sample(args.prompt, args.subtask_text, train_config.model.action_horizon)


def _build_model_input_pipeline(train_config):
    from openpi import transforms as _transforms

    data_factory = _primary_data_factory(train_config)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    input_transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    return data_config, _transforms.compose(input_transforms)


def _validate_training_wiring(train_config, raw_sample: dict[str, Any]) -> dict[str, Any]:
    sample = dict(raw_sample)
    if "prompt" not in sample and "task" in sample:
        sample["prompt"] = sample["task"]

    LOGGER.info("Training raw sample keys: %s", sorted(sample.keys()))

    _, pipeline = _build_model_input_pipeline(train_config)
    transformed = pipeline(sample)

    required = [
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "subtask_tokens",
        "subtask_mask",
        "subtask_ar_mask",
        "subtask_loss_mask",
        "state",
    ]
    missing = [key for key in required if key not in transformed]
    if missing:
        raise AssertionError(f"Training pipeline missing keys: {missing}")

    if not np.any(transformed["subtask_mask"]):
        raise AssertionError("subtask_mask is empty; hybrid CE supervision is not wired in.")
    if not np.any(transformed["subtask_loss_mask"]):
        raise AssertionError("subtask_loss_mask is empty; hybrid CE loss would be disabled.")

    LOGGER.info(
        "Training wiring OK: prompt_tokens=%s subtask_tokens=%s state=%s",
        tuple(transformed["tokenized_prompt"].shape),
        tuple(transformed["subtask_tokens"].shape),
        tuple(transformed["state"].shape),
    )
    return transformed


def _to_batched_training_inputs(transformed: dict[str, Any], device: torch.device):
    from openpi.models.model import Observation

    def _batch_array(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.expand_dims(value, axis=0))
        if isinstance(value, np.generic):
            return torch.from_numpy(np.expand_dims(np.asarray(value), axis=0))
        if np.isscalar(value) and not isinstance(value, str | bytes):
            return torch.from_numpy(np.expand_dims(np.asarray(value), axis=0))
        return value

    batched = {}
    for key, value in transformed.items():
        if isinstance(value, dict):
            batched[key] = {
                inner_key: _batch_array(inner_value)
                for inner_key, inner_value in value.items()
            }
        else:
            batched[key] = _batch_array(value)

    actions = batched.pop("actions").to(device=device, dtype=torch.float32)
    observation = Observation.from_dict(batched)
    observation = Observation(
        images={key: value.to(device) for key, value in observation.images.items()},
        image_masks={key: value.to(device) for key, value in observation.image_masks.items()},
        state=observation.state.to(device),
        tokenized_prompt=observation.tokenized_prompt.to(device) if observation.tokenized_prompt is not None else None,
        tokenized_prompt_mask=observation.tokenized_prompt_mask.to(device)
        if observation.tokenized_prompt_mask is not None
        else None,
        token_ar_mask=observation.token_ar_mask.to(device) if observation.token_ar_mask is not None else None,
        token_loss_mask=observation.token_loss_mask.to(device) if observation.token_loss_mask is not None else None,
        subtask_tokens=observation.subtask_tokens.to(device) if observation.subtask_tokens is not None else None,
        subtask_mask=observation.subtask_mask.to(device) if observation.subtask_mask is not None else None,
        subtask_loss_mask=observation.subtask_loss_mask.to(device) if observation.subtask_loss_mask is not None else None,
        subtask_ar_mask=observation.subtask_ar_mask.to(device) if observation.subtask_ar_mask is not None else None,
        pcd_xyz=observation.pcd_xyz.to(device) if observation.pcd_xyz is not None else None,
    )
    return observation, actions


def _run_training_smoke(
    train_config,
    checkpoint_dir: pathlib.Path,
    transformed: dict[str, Any],
    *,
    device: str,
    trace_file: pathlib.Path | None = None,
) -> None:
    torch_device = torch.device(device)
    observation, actions = _to_batched_training_inputs(transformed, torch_device)

    weight_path = checkpoint_dir / "model.safetensors"
    _trace(trace_file, "train_smoke:model_load_start")
    model = train_config.model.load_pytorch(train_config, str(weight_path)).to(torch_device)
    _trace(trace_file, "train_smoke:model_load_done")
    model.train()

    optim = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=1e-6)
    optim.zero_grad(set_to_none=True)

    use_autocast = torch_device.type == "cuda" and getattr(train_config, "pytorch_training_precision", "bfloat16") == "bfloat16"
    _trace(trace_file, "train_smoke:forward_start")
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16, enabled=use_autocast):
        losses = model(observation, actions)
    _trace(trace_file, "train_smoke:forward_done")

    if not isinstance(losses, dict):
        raise AssertionError(f"Expected hybrid training smoke to return a dict, got {type(losses)}")
    for key in ("loss", "flow_loss", "ce_loss"):
        if key not in losses:
            raise AssertionError(f"Hybrid training smoke missing loss component: {key}")

    loss = losses["loss"].float()
    if not torch.isfinite(loss):
        raise AssertionError("Hybrid training smoke produced a non-finite total loss.")

    loss.backward()
    _trace(trace_file, "train_smoke:backward_done")
    optim.step()
    _trace(trace_file, "train_smoke:step_done")

    LOGGER.info(
        "training smoke OK: loss=%.4f flow_loss=%.4f ce_loss=%.4f",
        float(loss.detach().cpu()),
        float(losses["flow_loss"].detach().cpu()),
        float(losses["ce_loss"].detach().cpu()),
    )


def _prepare_policy_obs(raw_sample: dict[str, Any], *, prompt: str | None, force_hierarchical: bool) -> dict[str, Any]:
    obs = dict(raw_sample)

    key_aliases = {
        "observation.images.rgb.head": "observation/egocentric_camera",
        "observation.images.rgb.left_wrist": "observation/wrist_image_left",
        "observation.images.rgb.right_wrist": "observation/wrist_image_right",
        "observation.state": "observation/state",
    }
    for source_key, target_key in key_aliases.items():
        if source_key in obs and target_key not in obs:
            obs[target_key] = obs[source_key]

    if prompt is not None:
        obs["prompt"] = prompt
    elif "prompt" not in obs and "task" in obs:
        obs["prompt"] = obs["task"]

    if force_hierarchical:
        obs.pop("subtask_text", None)
    return obs


def _build_policy(
    train_config,
    checkpoint_dir: pathlib.Path,
    device: str,
    *,
    num_action_steps: int,
    max_gen_steps: int,
    temperature: float,
    trace_file: pathlib.Path | None = None,
):
    from openpi.policies import policy_config

    _trace(trace_file, "policy_stage:data_pipeline_start")
    data_config, _ = _build_model_input_pipeline(train_config)
    _trace(trace_file, "policy_stage:data_pipeline_ready")
    previous_trace_path = os.environ.get("OPENPI_POLICY_TRACE_FILE")
    try:
        if trace_file is not None:
            os.environ["OPENPI_POLICY_TRACE_FILE"] = str(trace_file)
        return policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            sample_kwargs={
                "num_steps": num_action_steps,
                "max_subtask_tokens": max_gen_steps,
                "temperature": temperature,
            },
            norm_stats=data_config.norm_stats,
            pytorch_device=device,
        )
    finally:
        if previous_trace_path is None:
            os.environ.pop("OPENPI_POLICY_TRACE_FILE", None)
        else:
            os.environ["OPENPI_POLICY_TRACE_FILE"] = previous_trace_path


def _run_policy_validation(
    policy,
    obs: dict[str, Any],
    *,
    repeat: int,
    expect_generated_subtask: bool,
    label: str,
) -> None:
    first_generated_subtask = None
    for run_idx in range(repeat):
        start_time = time.monotonic()
        result = policy.infer(obs)
        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        actions = result["actions"]
        if actions.ndim != 2:
            raise AssertionError(f"Expected actions to have shape [horizon, dim], got {actions.shape}")
        if actions.shape[1] != 23:
            raise AssertionError(f"Expected 23-dim actions after B1K output transform, got {actions.shape}")

        generated_subtask = result.get("generated_subtask")
        if expect_generated_subtask and "generated_subtask" not in result:
            raise AssertionError("Hierarchical inference did not expose generated_subtask in policy output.")
        if not expect_generated_subtask and "generated_subtask" in result:
            raise AssertionError("Ground-truth subtask path unexpectedly returned generated_subtask.")

        if run_idx == 0:
            first_generated_subtask = generated_subtask
        elif expect_generated_subtask and generated_subtask != first_generated_subtask:
            raise AssertionError(
                "Repeated inference with the same prompt produced a different generated_subtask; cache path may be broken."
            )

        LOGGER.info(
            "%s infer run %d OK in %.1f ms: generated_subtask=%r action_shape=%s action_range=[%.4f, %.4f]",
            label,
            run_idx + 1,
            elapsed_ms,
            generated_subtask,
            tuple(actions.shape),
            float(actions.min()),
            float(actions.max()),
        )


def _resolve_inference_modes(args: argparse.Namespace) -> list[str]:
    if args.inference_mode == "both":
        return ["auto", "ground-truth"]
    if args.inference_mode == "ground-truth":
        return ["ground-truth"]
    return ["auto"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PI05_HYBRID data wiring and hierarchical inference")
    parser.add_argument("--config", type=str, default="pi05_hybrid_b1k-pt50_cs32_bs64_lr2.5e-5_5ep")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--source", choices=("auto", "dataset", "dummy"), default="auto")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="clean the table and place items neatly")
    parser.add_argument("--subtask-text", type=str, default="pick up the object and move it to the target area")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-gen-steps", type=int, default=64)
    parser.add_argument("--num-action-steps", type=int, default=10)
    parser.add_argument("--repeat-infer", type=int, default=2)
    parser.add_argument("--inference-mode", choices=("auto", "ground-truth", "both"), default="both")
    parser.add_argument("--use-ground-truth-subtask", action="store_true")
    parser.add_argument("--run-train-smoke", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--trace-file", type=str, default=None)
    args = parser.parse_args()
    if args.use_ground_truth_subtask and args.inference_mode == "both":
        args.inference_mode = "ground-truth"
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    args = _parse_args()
    trace_file = pathlib.Path(args.trace_file).resolve() if args.trace_file else None
    try:
        _trace(trace_file, "args_parsed")
        _configure_runtime_env(args.config)
        _trace(trace_file, "runtime_env_configured")

        train_config = _load_train_config(args.config)
        _trace(trace_file, f"config_loaded:{train_config.name}")
        model_type_name = getattr(train_config.model.model_type, "name", str(train_config.model.model_type))
        if model_type_name != "PI05_HYBRID":
            raise ValueError(f"Config {args.config!r} is not a PI05_HYBRID config.")

        device = _resolve_device(args.device)
        checkpoint_dir = _resolve_checkpoint_dir(train_config, args.checkpoint)
        _trace(trace_file, f"checkpoint_resolved:{checkpoint_dir}")
        source_name, raw_sample = _load_raw_sample(args, train_config)
        _trace(trace_file, f"raw_sample_loaded:{source_name}")

        if "prompt" not in raw_sample and "task" in raw_sample:
            raw_sample["prompt"] = raw_sample["task"]

        LOGGER.info("Using %s sample", source_name)
        LOGGER.info("Checkpoint: %s", checkpoint_dir)
        LOGGER.info("Device: %s", device)
        LOGGER.info("Prompt: %r", raw_sample.get("prompt"))
        LOGGER.info("Ground-truth subtask: %r", raw_sample.get("subtask_text"))

        transformed_training_sample = _validate_training_wiring(train_config, raw_sample)
        _trace(trace_file, "training_wiring_validated")

        if args.run_train_smoke:
            _run_training_smoke(
                train_config,
                checkpoint_dir,
                transformed_training_sample,
                device=device,
                trace_file=trace_file,
            )
            _trace(trace_file, "train_smoke_ok")

        if args.skip_infer:
            LOGGER.info("PI05_HYBRID validation finished successfully.")
            _trace(trace_file, "validation_success")
            raise SystemExit(0)

        policy = _build_policy(
            train_config,
            checkpoint_dir,
            device,
            num_action_steps=args.num_action_steps,
            max_gen_steps=args.max_gen_steps,
            temperature=args.temperature,
            trace_file=trace_file,
        )
        _trace(trace_file, "policy_built")
        for inference_mode in _resolve_inference_modes(args):
            use_ground_truth_subtask = inference_mode == "ground-truth"
            policy_obs = _prepare_policy_obs(
                raw_sample,
                prompt=args.prompt if source_name == "dummy" else None,
                force_hierarchical=not use_ground_truth_subtask,
            )
            _run_policy_validation(
                policy,
                policy_obs,
                repeat=args.repeat_infer,
                expect_generated_subtask=not use_ground_truth_subtask,
                label=inference_mode,
            )
            _trace(trace_file, f"inference_ok:{inference_mode}")

        LOGGER.info("PI05_HYBRID validation finished successfully.")
        _trace(trace_file, "validation_success")
    except Exception as exc:
        _trace(trace_file, f"validation_error:{type(exc).__name__}:{exc}")
        raise