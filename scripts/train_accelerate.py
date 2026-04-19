"""
Accelerate training entrypoint for PI0/PI05/VLM2 (PyTorch).

This script is a sibling of `scripts/train_pytorch.py`:
- Keeps the same config/data/model pipeline
- Replaces manual DDP orchestration with HuggingFace Accelerate
- Optionally supports DeepSpeed ZeRO via `accelerate launch --config_file ...`

Usage
Single process (CPU/GPU):
  python scripts/train_accelerate.py <config_name> --exp_name <run_name>

Multi-GPU:
  accelerate launch --multi_gpu --num_processes=<n> scripts/train_accelerate.py <config_name> --exp_name <run_name>

DeepSpeed ZeRO:
  accelerate launch --config_file configs/accelerate_ds_zero2.yaml scripts/train_accelerate.py <config_name> --exp_name <run_name>
"""

from __future__ import annotations

import dataclasses
import datetime
import faulthandler
import gc
import logging
import os
import platform
import signal
import shutil
import sys
import time
from pathlib import Path

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

_FAULT_TIMEOUT_S = int(os.environ.get("OPENPI_FAULT_TIMEOUT_S", "0"))
_FAULT_REPEAT = os.environ.get("OPENPI_FAULT_REPEAT", "0") == "1"
if _FAULT_TIMEOUT_S > 0:
    faulthandler.dump_traceback_later(_FAULT_TIMEOUT_S, repeat=_FAULT_REPEAT)

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedType

import openpi.models.pi0_config
import openpi.models.pi05_subtask_config
import openpi.models.vlm2_vla_config
import openpi.models.model as _model
import openpi.models_pytorch.pi0_pytorch
import openpi.models_pytorch.pi05_subtask
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


_WANDB = None


def _get_wandb():
    global _WANDB
    if _WANDB is None:
        import wandb

        _WANDB = wandb
    return _WANDB


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter: logging.Formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)

    return formatter


def add_file_logging(log_file: str, formatter: logging.Formatter) -> None:
    logger = logging.getLogger()
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file):
            return
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def _wait_for_path(path: Path, *, what: str) -> None:
    timeout_s = float(os.environ.get("OPENPI_FS_SYNC_TIMEOUT_S", "600"))
    poll_s = float(os.environ.get("OPENPI_FS_SYNC_POLL_S", "1"))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for {what}: {path}")


def install_excepthook() -> None:
    default_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        try:
            logging.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        finally:
            default_hook(exc_type, exc, tb)

    sys.excepthook = _hook


def configure_hf_cache(config: _config.TrainConfig, *, accelerator: Accelerator) -> None:
    offline = os.environ.get("OPENPI_OFFLINE", "1") == "1"
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if os.environ.get("OPENPI_TORCH_COMPILE_SAMPLE_ACTIONS", "0") != "1":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    checkpoints_root = Path(config.checkpoint_base_dir).expanduser().resolve()
    hf_home = Path(os.environ.get("HF_HOME", str(checkpoints_root / "hf_home"))).expanduser()
    hub_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))).expanduser()
    transformers_cache = Path(os.environ.get("TRANSFORMERS_CACHE", str(hf_home / "transformers"))).expanduser()
    datasets_cache = Path(
        os.environ.get("HF_DATASETS_CACHE", str(checkpoints_root / "hf_datasets_cache"))
    ).expanduser()

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)

    # Keep the existing per-node caching behavior to reduce filelock races in multi-process runs.
    if accelerator.num_processes > 1:
        os.environ.setdefault("OPENPI_HF_DATASETS_CACHE_PER_RANK", "1")
        os.environ.setdefault("OPENPI_LOAD_DATASET_NUM_PROC_CAP", "32")
        os.environ.setdefault("OPENPI_HF_LOAD_DATASET_RETRIES", "5")
        os.environ.setdefault("OPENPI_HF_LOAD_DATASET_RETRY_SLEEP_S", "2")

    if accelerator.is_main_process:
        hf_home.mkdir(parents=True, exist_ok=True)
        hub_cache.mkdir(parents=True, exist_ok=True)
        transformers_cache.mkdir(parents=True, exist_ok=True)
        datasets_cache.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logging.info("HF_HOME=%s", os.environ.get("HF_HOME"))
        logging.info("HF_DATASETS_CACHE=%s", os.environ.get("HF_DATASETS_CACHE"))
        logging.info("HUGGINGFACE_HUB_CACHE=%s", os.environ.get("HUGGINGFACE_HUB_CACHE"))
        logging.info("TRANSFORMERS_CACHE=%s", os.environ.get("TRANSFORMERS_CACHE"))
        logging.info("HF_HUB_OFFLINE=%s", os.environ.get("HF_HUB_OFFLINE"))
        logging.info("HF_DATASETS_OFFLINE=%s", os.environ.get("HF_DATASETS_OFFLINE"))
        logging.info("TRANSFORMERS_OFFLINE=%s", os.environ.get("TRANSFORMERS_OFFLINE"))
        logging.info("TORCHDYNAMO_DISABLE=%s", os.environ.get("TORCHDYNAMO_DISABLE"))
        logging.info(
            "OPENPI_HF_DATASETS_CACHE_PER_RANK=%s", os.environ.get("OPENPI_HF_DATASETS_CACHE_PER_RANK")
        )
        logging.info("OPENPI_LOAD_DATASET_NUM_PROC_CAP=%s", os.environ.get("OPENPI_LOAD_DATASET_NUM_PROC_CAP"))


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        logging.info("wandb logging disabled")
        return

    wandb = _get_wandb()

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    settings = wandb.Settings(init_timeout=120)
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, settings=settings)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            settings=settings,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _latest_step_dir(checkpoint_dir: Path) -> tuple[int, Path] | None:
    steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not steps:
        return None
    step = max(steps)
    return step, checkpoint_dir / f"{step}"


def build_datasets(config: _config.TrainConfig):
    retries = max(1, int(os.environ.get("OPENPI_BUILD_DATASET_RETRIES", "3")))
    rank = int(os.environ.get("RANK", "0"))
    skip_norm_stats = os.environ.get("OPENPI_SKIP_NORM_STATS", "0") == "1"
    for attempt in range(1, retries + 1):
        try:
            data_loader = _data_loader.create_data_loader(
                config,
                framework="pytorch",
                shuffle=True,
                skip_norm_stats=skip_norm_stats,
            )
            return data_loader, data_loader.data_config()
        except FileNotFoundError as exc:
            transient_lock_race = exc.filename is None and int(os.environ.get("WORLD_SIZE", "1")) > 1
            if (not transient_lock_race) or attempt >= retries:
                raise
            delay_s = float(os.environ.get("OPENPI_BUILD_DATASET_RETRY_SLEEP_S", "2")) * attempt
            logging.warning(
                "Rank %s hit transient ENOENT during dataset init (attempt %s/%s). Retrying in %.1fs",
                rank,
                attempt,
                retries,
                delay_s,
            )
            time.sleep(delay_s)


def log_memory_usage(accelerator: Accelerator, step: int, phase: str = "unknown") -> None:
    if not (torch.cuda.is_available() and accelerator.device.type == "cuda"):
        return
    device = accelerator.device
    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_reserved_unallocated = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1e9
    device_free = 0.0
    device_total = 0.0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        device_free = free_bytes / 1e9
        device_total = total_bytes / 1e9
    except Exception:
        pass
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9
    logging.info(
        "Step %s (%s): GPU memory - allocated: %.2fGB, reserved: %.2fGB, reserved_unallocated: %.2fGB, device_free: %.2fGB, device_total: %.2fGB, peak_allocated: %.2fGB, peak_reserved: %.2fGB | rank=%s/%s",
        step,
        phase,
        memory_allocated,
        memory_reserved,
        memory_reserved_unallocated,
        device_free,
        device_total,
        max_memory_allocated,
        max_memory_reserved,
        accelerator.process_index,
        accelerator.num_processes,
    )


def _memory_phase_logging_enabled() -> bool:
    return os.environ.get("OPENPI_PHASE_MEMORY_LOG", "0") == "1"


def _memory_phase_steps() -> set[int]:
    raw = os.environ.get("OPENPI_PHASE_MEMORY_LOG_STEPS", "0")
    steps: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            steps.add(int(token))
        except ValueError:
            logging.warning("Ignoring invalid OPENPI_PHASE_MEMORY_LOG_STEPS token: %s", token)
    return steps


def _should_profile_memory_step(step: int) -> bool:
    if not _memory_phase_logging_enabled():
        return False
    return step in _memory_phase_steps()


def _reset_peak_memory_stats(accelerator: Accelerator) -> None:
    if not (torch.cuda.is_available() and accelerator.device.type == "cuda"):
        return
    torch.cuda.reset_peak_memory_stats(accelerator.device)


def _prepare_vlm2_inputs(
    observation,
    config: _config.TrainConfig,
    device: torch.device,
    *,
    include_subtask: bool = False,
):
    image_keys = _model.IMAGE_KEYS
    frames = [observation.images[k] for k in image_keys if k in observation.images]
    if not frames:
        raise ValueError("No images found in observation for VLM2 inputs.")

    video_frames = torch.stack(frames, dim=1)  # (b, f, c, h, w)
    target_frames = config.vlm2_num_frames
    if video_frames.shape[1] < target_frames:
        pad_count = target_frames - video_frames.shape[1]
        pad_frame = video_frames[:, -1:].repeat(1, pad_count, 1, 1, 1)
        video_frames = torch.cat([video_frames, pad_frame], dim=1)
    elif video_frames.shape[1] > target_frames:
        video_frames = video_frames[:, :target_frames]

    if getattr(observation, "pcd_xyz", None) is not None:
        point_map = observation.pcd_xyz.to(torch.float32)
        if point_map.dim() != 4:
            raise ValueError(f"Expected pcd_xyz shape (b, s, n, 3), got {point_map.shape}")
        point_maps = point_map[:, None].repeat(1, target_frames, 1, 1, 1)
    else:
        batch_size, _, _, height, width = video_frames.shape
        point_maps = torch.zeros(
            batch_size,
            target_frames,
            height,
            width,
            3,
            device=device,
            dtype=torch.float32,
        )

    language_tokens = observation.tokenized_prompt
    language_masks = observation.tokenized_prompt_mask
    if language_tokens is None or language_masks is None:
        raise ValueError("tokenized_prompt and tokenized_prompt_mask are required for VLM2 training.")

    if not include_subtask:
        return video_frames, point_maps, language_tokens, language_masks

    subtask_tokens = getattr(observation, "subtask_tokens", None)
    subtask_mask = getattr(observation, "subtask_mask", None)
    subtask_ar_mask = getattr(observation, "subtask_ar_mask", None)
    subtask_loss_mask = getattr(observation, "subtask_loss_mask", None)
    return (
        video_frames,
        point_maps,
        language_tokens,
        language_masks,
        subtask_tokens,
        subtask_mask,
        subtask_ar_mask,
        subtask_loss_mask,
    )


def _infer_accelerate_mixed_precision(config: _config.TrainConfig) -> str:
    mp = getattr(config, "accelerate_mixed_precision", None)
    if mp is not None:
        return str(mp)
    if config.pytorch_training_precision == "bfloat16":
        return "bf16"
    if config.pytorch_training_precision == "float16":
        return "fp16"
    return "no"


def _atomic_write_checkpoint_dir(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    global_step: int,
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> None:
    # `global_step` is 1-based here: pass the post-update optimizer step so checkpoint directories
    # line up with the visible training step count.
    should_save = (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps
    if not should_save:
        return

    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

    # Rank 0 owns directory cleanup/creation to avoid races on shared filesystems.
    if accelerator.is_main_process:
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        _wait_for_path(tmp_ckpt_dir, what="tmp_ckpt_dir")

    # Save accelerate/deepspeed state for resume.
    # IMPORTANT: This must run on *all* ranks (DeepSpeed save is collective). Running it on rank0 only can hang.
    save_acc_state = os.environ.get("OPENPI_SAVE_ACCELERATE_STATE", "1") != "0"
    if save_acc_state:
        acc_state_dir = tmp_ckpt_dir / "accelerate_state"
        t0 = time.time()
        if accelerator.is_main_process and _should_profile_memory_step(global_step):
            _reset_peak_memory_stats(accelerator)
        try:
            accelerator.save_state(str(acc_state_dir))
        except Exception as exc:
            logging.warning(
                "accelerator.save_state failed (resume may not work for sharded optimizers): %s", exc
            )
        else:
            if accelerator.is_main_process:
                logging.info("accelerator.save_state finished in %.1fs", time.time() - t0)
                if _should_profile_memory_step(global_step):
                    log_memory_usage(accelerator, global_step, "after_save_state")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Save model weights (portable artifact).
        model_to_save = accelerator.unwrap_model(model)
        model_path = tmp_ckpt_dir / "model.safetensors"
        try:
            # DeepSpeed ZeRO-3 may keep params partitioned; use Accelerator to materialize a full state_dict.
            state_dict = accelerator.get_state_dict(model)
            safetensors.torch.save_file(state_dict, str(model_path))
        except Exception:
            # Fallback for non-partitioned models (handles tied/shared tensors).
            safetensors.torch.save_model(model_to_save, model_path)

        # Save optimizer state (non-DeepSpeed / non-sharded). With DeepSpeed, prefer accelerator.save_state.
        if optimizer is not None:
            try:
                torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
            except Exception as exc:
                logging.warning("Failed to save optimizer.pt (will rely on accelerate_state if present): %s", exc)

        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
            "accelerate": {
                "distributed_type": str(accelerator.distributed_type),
                "num_processes": accelerator.num_processes,
            },
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # Save norm stats.
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        _atomic_write_checkpoint_dir(tmp_ckpt_dir, final_ckpt_dir)
        logging.info("Saved checkpoint at step %s -> %s", global_step, final_ckpt_dir)

        if config.wandb_enabled:
            wandb = _get_wandb()
            wandb.log({"checkpoint_step": global_step}, step=global_step)

    accelerator.wait_for_everyone()


def train_loop(config: _config.TrainConfig, *, formatter: logging.Formatter) -> None:
    accelerator = Accelerator(
        mixed_precision=_infer_accelerate_mixed_precision(config),
        gradient_accumulation_steps=int(getattr(config, "gradient_accumulation_steps", 1)),
    )

    is_main = accelerator.is_main_process
    local_rank = accelerator.local_process_index

    # Seed: keep per-rank determinism.
    seed = int(config.seed) + int(local_rank)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # DDP-safe overwrite/resume handling.
    resuming = False
    if config.resume:
        if config.checkpoint_dir.exists():
            latest = _latest_step_dir(config.checkpoint_dir)
            if latest is None:
                raise FileNotFoundError(f"No valid checkpoints found in {config.checkpoint_dir} for resume")
            resuming = True
            if is_main:
                logging.info("Resuming from %s at step %s", latest[1], latest[0])
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {config.checkpoint_dir} does not exist for resume")
    elif config.overwrite:
        if is_main and config.checkpoint_dir.exists():
            shutil.rmtree(config.checkpoint_dir)
            logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)
        accelerator.wait_for_everyone()

    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if not is_main:
        _wait_for_path(config.log_dir, what="log_dir")

    add_file_logging(str(config.log_dir / f"rank{accelerator.process_index}.log"), formatter)
    install_excepthook()

    configure_hf_cache(config, accelerator=accelerator)
    os.environ["OPENPI_FORCE_LOAD_CACHE"] = "1" if config.force_load_cache else "0"
    if is_main:
        logging.info("prepare_hf_cache_only=%s", config.prepare_hf_cache_only)
        logging.info("force_load_cache=%s", config.force_load_cache)

    if is_main and not config.prepare_hf_cache_only:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Batch size semantics: keep compatibility with train_pytorch.py.
    world_size = accelerator.num_processes
    if config.batch_size_per_gpu is not None:
        per_gpu = int(config.batch_size_per_gpu)
        if per_gpu <= 0:
            raise ValueError("--batch_size_per_gpu must be a positive integer when provided.")
        object.__setattr__(config, "batch_size", per_gpu * world_size)
        effective_batch_size = per_gpu
    else:
        effective_batch_size = config.batch_size // world_size

    if is_main:
        logging.info(
            "Using batch size per GPU: %s (total batch size across %s procs: %s) grad_accum=%s effective_total=%s",
            effective_batch_size,
            world_size,
            config.batch_size,
            accelerator.gradient_accumulation_steps,
            config.batch_size * accelerator.gradient_accumulation_steps,
        )

    # Accelerate cannot infer micro-batch size from the custom OpenPI dataloader wrapper.
    # Populate the DeepSpeed plugin config explicitly before `prepare()`.
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        ds_config["train_micro_batch_size_per_gpu"] = int(effective_batch_size)
        ds_config["gradient_accumulation_steps"] = int(accelerator.gradient_accumulation_steps)
        ds_config["train_batch_size"] = int(
            effective_batch_size * accelerator.gradient_accumulation_steps * world_size
        )
        precision = str(config.pytorch_training_precision)
        ds_config.setdefault("bf16", {})
        ds_config.setdefault("fp16", {})
        ds_config.setdefault("torch_autocast", {})
        ds_config["bf16"]["enabled"] = precision == "bfloat16"
        ds_config["fp16"]["enabled"] = precision == "float16"
        ds_config["torch_autocast"]["enabled"] = precision in ("bfloat16", "float16")
        ds_config["torch_autocast"]["dtype"] = "bfloat16" if precision == "bfloat16" else "float16"
        if is_main:
            logging.info(
                "Patched DeepSpeed config: train_micro_batch_size_per_gpu=%s gradient_accumulation_steps=%s train_batch_size=%s bf16=%s fp16=%s autocast_dtype=%s",
                ds_config["train_micro_batch_size_per_gpu"],
                ds_config["gradient_accumulation_steps"],
                ds_config["train_batch_size"],
                ds_config["bf16"]["enabled"],
                ds_config["fp16"]["enabled"],
                ds_config["torch_autocast"]["dtype"],
            )

    loader, data_config = build_datasets(config)
    if config.prepare_hf_cache_only:
        if is_main:
            logging.info("Offline HF cache preparation completed; exiting as requested.")
        return

    # Epoch accounting: len(loader) is per-rank micro-batch count.
    steps_per_epoch_micro = len(loader)
    steps_per_epoch = max(1, steps_per_epoch_micro // accelerator.gradient_accumulation_steps)
    if steps_per_epoch <= 0:
        raise RuntimeError(f"Computed steps_per_epoch={steps_per_epoch}, expected a positive value.")

    if config.num_train_epochs is not None:
        if config.num_train_epochs <= 0:
            raise ValueError("--num_train_epochs must be a positive integer when provided.")
        computed_steps = int(config.num_train_epochs) * steps_per_epoch
        provided_steps = int(config.num_train_steps)
        target_steps = computed_steps if provided_steps <= 0 else min(provided_steps, computed_steps)
        object.__setattr__(config, "num_train_steps", target_steps)
        if is_main:
            logging.info(
                "Computed num_train_steps=%s from num_train_epochs=%s and steps_per_epoch=%s (micro=%s, grad_accum=%s)",
                target_steps,
                config.num_train_epochs,
                steps_per_epoch,
                steps_per_epoch_micro,
                accelerator.gradient_accumulation_steps,
            )
        if config.save_at_epoch_end_only:
            object.__setattr__(config, "save_interval", target_steps)
            if is_main:
                logging.info("save_at_epoch_end_only enabled: save_interval=%s", target_steps)

    # Build model (same logic as train_pytorch.py).
    if isinstance(config.model, openpi.models.vlm2_vla_config.VLM2VLAConfig):
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    elif isinstance(config.model, openpi.models.pi05_subtask_config.Pi05SubtaskConfig):
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    elif not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    use_vlm2 = config.pytorch_model_name in ("vlm2", "vlm2_subtask")
    if config.pytorch_model_name in ("vlm2", "vlm2_subtask"):
        import openpi.models_pytorch.vlm2.vlm2_model as _vlm2_model

        vlm2_config = _vlm2_model.VLM2Config(
            visual_dim=2048,
            geometry_dim=config.vlm2_geometry_dim,
            view_dim=config.vlm2_view_dim,
            working_memory_size=config.vlm2_working_memory_size,
            episodic_memory_capacity=config.vlm2_episodic_memory_capacity,
            episodic_similarity_threshold=config.vlm2_episodic_similarity_threshold,
            episodic_fusion_alpha=config.vlm2_episodic_fusion_alpha,
            sem_geo_fusion_tanh_gate_enable=config.vlm2_sem_geo_fusion_tanh_gate_enable,
            sem_geo_fusion_tanh_gate_init_alpha=config.vlm2_sem_geo_fusion_tanh_gate_init_alpha,
            num_heads=8,
            hidden_dim=1024,
            dropout=0.0,
            pi05=True,
            action_dim=model_cfg.action_dim,
            action_horizon=model_cfg.action_horizon,
            dtype=config.pytorch_training_precision,
            paligemma_variant=model_cfg.paligemma_variant,
            action_expert_variant=model_cfg.action_expert_variant,
            num_frames=config.vlm2_num_frames,
            frame_height=224,
            frame_width=224,
            patch_size=16,
            vggt_pretrained=getattr(model_cfg, "vggt_pretrained", None),
            vggt_load_strict=getattr(model_cfg, "vggt_load_strict", False),
            vggt_enable_track=getattr(model_cfg, "vggt_enable_track", False),
            freeze_vggt_backbone=getattr(model_cfg, "freeze_vggt_backbone", False),
            freeze_image_encoder=getattr(model_cfg, "freeze_image_encoder", False),
        )
        if config.pytorch_model_name == "vlm2_subtask":
            alpha = getattr(model_cfg, "alpha", 10.0)
            model = _vlm2_model.VLM2SubtaskWithPi05(vlm2_config, alpha=alpha)
        else:
            model = _vlm2_model.VLM2WithPi05(vlm2_config)
    elif config.pytorch_model_name == "subtask":
        alpha = getattr(model_cfg, "alpha", 10.0)
        model = openpi.models_pytorch.pi05_subtask.PI05SubtaskPytorch(
            model_cfg,
            alpha=alpha,
            action_expert_name="subtask",
        )
    else:
        model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if is_main:
            logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        if is_main:
            logging.info("Gradient checkpointing is not supported for this model")

    # Memory/perf knobs (keep same behavior as train_pytorch.py).
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        if is_main:
            logging.info("Enabled memory optimizations for 8+ GPU training")

    # Weight loading for fine-tuning.
    if config.pytorch_weight_path is not None:
        if is_main:
            logging.info("Loading weights from: %s", config.pytorch_weight_path)
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        if not os.path.exists(model_path):
            if is_main:
                logging.warning("Model checkpoint not found at %s. Skipping weight loading.", model_path)
        else:
            load_strict = config.pytorch_model_name not in ("vlm2", "vlm2_subtask", "subtask")
            safetensors.torch.load_model(model, model_path, strict=load_strict)
            if is_main:
                logging.info("Loaded PyTorch weights from %s", config.pytorch_weight_path)

    # Optimizer + LR schedule (reuse logic from train_pytorch.py).
    warmup_steps = int(config.lr_schedule.warmup_steps)
    peak_lr = float(config.lr_schedule.peak_lr)
    decay_steps = int(config.lr_schedule.decay_steps)
    end_lr = float(config.lr_schedule.decay_lr)
    if decay_steps <= 0:
        decay_steps = int(config.num_train_steps)
        if is_main:
            logging.info("Auto-set decay_steps=%d to match num_train_steps", decay_steps)
    elif decay_steps < int(config.num_train_steps):
        if is_main:
            logging.warning(
                "decay_steps=%d < num_train_steps=%d — LR will reach 0 before training ends; overriding to num_train_steps",
                decay_steps,
                int(config.num_train_steps),
            )
        decay_steps = int(config.num_train_steps)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if len(optim_params) == 0:
        raise RuntimeError("No trainable parameters found (all parameters are frozen).")

    use_8bit_optim = os.environ.get("USE_8BIT_OPTIM", "0") == "1"
    if use_8bit_optim:
        try:
            import bitsandbytes as bnb

            optimizer: torch.optim.Optimizer = bnb.optim.AdamW8bit(
                optim_params,
                lr=peak_lr,
                betas=(config.optimizer.b1, config.optimizer.b2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
            )
            if is_main:
                logging.info("Using 8-bit AdamW optimizer from bitsandbytes")
        except ImportError:
            if is_main:
                logging.warning("bitsandbytes not found, falling back to standard AdamW")
            optimizer = torch.optim.AdamW(
                optim_params,
                lr=peak_lr,
                betas=(config.optimizer.b1, config.optimizer.b2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
            )
    else:
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # Prepare with Accelerator (DDP or DeepSpeed).
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # Resume (after prepare so that accelerator can restore distributed states).
    global_step = 0
    if resuming:
        latest = _latest_step_dir(config.checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {config.checkpoint_dir}")
        latest_step, latest_dir = latest
        acc_state_dir = latest_dir / "accelerate_state"
        if acc_state_dir.exists():
            accelerator.load_state(str(acc_state_dir))
        metadata_path = latest_dir / "metadata.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
            global_step = int(metadata.get("global_step", latest_step))
        else:
            global_step = latest_step
        if is_main:
            logging.info("Resumed training from step %s", global_step)

    # Pre-training barrier to avoid watchdog timeouts on large init skew.
    if is_main:
        logging.info(
            "Running on: %s | num_processes=%s | distributed_type=%s",
            platform.node(),
            accelerator.num_processes,
            accelerator.distributed_type,
        )
        logging.info(
            "Training config: batch_size=%s effective_batch_size_per_gpu=%s num_train_steps=%s",
            config.batch_size,
            effective_batch_size,
            config.num_train_steps,
        )
        logging.info(
            "LR schedule: warmup=%s peak_lr=%.2e decay_steps=%s end_lr=%.2e",
            warmup_steps,
            peak_lr,
            decay_steps,
            end_lr,
        )
        logging.info(
            "Optimizer: %s weight_decay=%s clip_norm=%s",
            type(config.optimizer).__name__,
            config.optimizer.weight_decay,
            config.optimizer.clip_gradient_norm,
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info("Training precision: %s (accelerate mp=%s)", model_cfg.dtype, accelerator.mixed_precision)
    accelerator.wait_for_everyone()

    if torch.cuda.is_available() and accelerator.device.type == "cuda" and is_main:
        log_memory_usage(accelerator, global_step, "after_model_prepare")

    model.train()

    start_time = time.time()
    infos: list[dict[str, float]] = []

    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    last_epoch_logged = None
    while global_step < int(config.num_train_steps):
        for observation, actions in loader:
            if global_step >= int(config.num_train_steps):
                break

            profile_memory = is_main and _should_profile_memory_step(global_step)

            # Move data to device.
            observation = jax.tree.map(
                lambda x: x.to(accelerator.device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
                observation,
            )
            actions = actions.to(device=accelerator.device, dtype=torch.float32, non_blocking=True)

            with accelerator.accumulate(model):
                # Update LR per optimizer step (only when syncing grads).
                if accelerator.sync_gradients:
                    lr = lr_schedule(global_step)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                extra_metrics: dict[str, float] = {}
                ds_plugin = accelerator.state.deepspeed_plugin if accelerator.distributed_type == DistributedType.DEEPSPEED else None
                ds_uses_torch_autocast = bool(
                    ds_plugin is not None
                    and ds_plugin.deepspeed_config.get("torch_autocast", {}).get("enabled", False)
                )
                use_autocast = (
                    config.pytorch_training_precision == "bfloat16"
                    and accelerator.device.type == "cuda"
                    and not ds_uses_torch_autocast
                )
                with torch.autocast(
                    device_type=accelerator.device.type,
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    if profile_memory:
                        _reset_peak_memory_stats(accelerator)
                    if use_vlm2:
                        if config.pytorch_model_name == "vlm2_subtask":
                            (
                                video_frames,
                                point_maps,
                                language_tokens,
                                language_masks,
                                subtask_tokens,
                                subtask_mask,
                                subtask_ar_mask,
                                subtask_loss_mask,
                            ) = _prepare_vlm2_inputs(observation, config, accelerator.device, include_subtask=True)
                            losses = model(
                                video_frames=video_frames,
                                point_maps=point_maps,
                                language_tokens=language_tokens,
                                language_masks=language_masks,
                                actions=actions,
                                subtask_tokens=subtask_tokens,
                                subtask_mask=subtask_mask,
                                subtask_ar_mask=subtask_ar_mask,
                                subtask_loss_mask=subtask_loss_mask,
                            )
                        else:
                            video_frames, point_maps, language_tokens, language_masks = _prepare_vlm2_inputs(
                                observation, config, accelerator.device
                            )
                            losses = model(
                                video_frames=video_frames,
                                point_maps=point_maps,
                                language_tokens=language_tokens,
                                language_masks=language_masks,
                                actions=actions,
                            )
                    else:
                        losses = model(observation, actions)

                    if isinstance(losses, dict):
                        extra_metrics = {
                            k: v.item() for k, v in losses.items() if k != "loss" and isinstance(v, torch.Tensor)
                        }
                        loss = losses["loss"]
                    elif isinstance(losses, (list, tuple)):
                        loss = torch.stack(list(losses)).mean()
                    elif not isinstance(losses, torch.Tensor):
                        loss = torch.tensor(losses, device=accelerator.device, dtype=torch.float32)
                    else:
                        loss = losses.mean()

                if profile_memory:
                    log_memory_usage(accelerator, global_step, "after_forward")

                loss = loss.float()
                if profile_memory:
                    _reset_peak_memory_stats(accelerator)
                accelerator.backward(loss)
                if profile_memory:
                    log_memory_usage(accelerator, global_step, "after_backward")

                if accelerator.sync_gradients:
                    if global_step < 5 and is_main and torch.cuda.is_available() and not profile_memory:
                        log_memory_usage(accelerator, global_step, "after_backward")

                    grad_norm = accelerator.clip_grad_norm_(optim_params, max_norm=config.optimizer.clip_gradient_norm)
                    if profile_memory:
                        _reset_peak_memory_stats(accelerator)
                    optimizer.step()
                    if profile_memory:
                        log_memory_usage(accelerator, global_step, "after_optimizer_step")
                    optimizer.zero_grad(set_to_none=True)

                    # stats/logging use optimizer-step granularity
                    if is_main:
                        infos.append(
                            {
                                "loss": loss.item(),
                                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                                **extra_metrics,
                            }
                        )

                    if is_main and (global_step % int(config.log_interval) == 0):
                        elapsed = time.time() - start_time
                        epoch_idx = global_step // steps_per_epoch
                        epoch = epoch_idx + 1
                        epoch_step = (global_step % steps_per_epoch) + 1
                        if last_epoch_logged != epoch:
                            if config.num_train_epochs is not None:
                                logging.info("epoch=%s/%s", epoch, config.num_train_epochs)
                            else:
                                logging.info("epoch=%s", epoch)
                            last_epoch_logged = epoch

                        avg_loss = sum(info["loss"] for info in infos) / len(infos)
                        avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                        avg_grad_norm = sum(info["grad_norm"] for info in infos) / len(infos)
                        logging.info(
                            "step=%s epoch=%s epoch_step=%s/%s loss=%.4f lr=%.2e grad_norm=%.2f time=%.1fs",
                            global_step,
                            epoch,
                            epoch_step,
                            steps_per_epoch,
                            avg_loss,
                            avg_lr,
                            avg_grad_norm,
                            elapsed,
                        )

                        if config.wandb_enabled and len(infos) > 0:
                            wandb = _get_wandb()
                            log_payload: dict[str, float] = {
                                "loss": avg_loss,
                                "learning_rate": avg_lr,
                                "grad_norm": avg_grad_norm,
                                "step": float(global_step),
                                "epoch": float(epoch),
                                "epoch_step": float(epoch_step),
                                "steps_per_epoch": float(steps_per_epoch),
                                "time_per_step": elapsed / max(1, int(config.log_interval)),
                            }
                            for metric_key in ("flow_loss", "ce_loss"):
                                vals = [info[metric_key] for info in infos if metric_key in info]
                                if vals:
                                    log_payload[f"subtask/{metric_key}"] = sum(vals) / len(vals)
                            wandb.log(log_payload, step=global_step)

                        start_time = time.time()
                        infos = []

                    current_step = global_step + 1

                    # checkpoint/save + progress bar update
                    save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        global_step=current_step,
                        config=config,
                        data_config=data_config,
                    )

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "loss": f"{loss.item():.4f}",
                                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                                "step": current_step,
                            }
                        )

                    global_step = current_step

    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb = _get_wandb()
        wandb.finish()


def main():
    formatter = init_logging()
    logging.info("Host: %s PID: %s", platform.node(), os.getpid())
    logging.info("Python: %s (%s)", sys.version.split()[0], sys.executable)
    logging.info("CWD: %s", os.getcwd())
    logging.info("OPENPI_DATA_HOME=%s", os.environ.get("OPENPI_DATA_HOME"))
    logging.info("B1K_VIDEO_BACKEND=%s", os.environ.get("B1K_VIDEO_BACKEND"))
    logging.info("JAX_PLATFORMS=%s", os.environ.get("JAX_PLATFORMS"))
    logging.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vggt_dir = os.path.join(repo_root, "src", "openpi", "third_party", "vggt")
    cut3r_dir = os.path.join(repo_root, "src", "openpi", "third_party", "cut3r")
    if not os.path.isdir(vggt_dir) or not os.path.isdir(cut3r_dir):
        raise FileNotFoundError(
            "Missing third_party dependencies. Expected directories:\n"
            f"  - {vggt_dir}\n"
            f"  - {cut3r_dir}\n"
            "Fix by running: git submodule update --init --recursive"
        )

    config = _config.cli()
    logging.info(
        "Run: exp_name=%s project=%s wandb=%s num_workers=%s batch_size=%s grad_accum=%s",
        getattr(config, "exp_name", None),
        getattr(config, "project_name", None),
        getattr(config, "wandb_enabled", None),
        getattr(config, "num_workers", None),
        getattr(config, "batch_size", None),
        getattr(config, "gradient_accumulation_steps", 1),
    )
    train_loop(config, formatter=formatter)


if __name__ == "__main__":
    main()
