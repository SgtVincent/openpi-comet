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

import atexit
from contextlib import contextmanager
import dataclasses
import datetime
import faulthandler
import functools
import gc
import hashlib
import importlib
import json
import importlib.metadata as importlib_metadata
import logging
import os
import random
import platform
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

_FAULT_TIMEOUT_S = int(os.environ.get("OPENPI_FAULT_TIMEOUT_S", "0"))
_FAULT_REPEAT = os.environ.get("OPENPI_FAULT_REPEAT", "0") == "1"
if _FAULT_TIMEOUT_S > 0:
    faulthandler.dump_traceback_later(_FAULT_TIMEOUT_S, repeat=_FAULT_REPEAT)


def _patch_byted_wandb_metadata() -> None:
    """Let libraries that probe `wandb` metadata work with `byted-wandb`.

    The internal Tracking SDK is installed as `byted-wandb` while exposing the
    import name `wandb`. Some libraries, including HuggingFace Accelerate paths,
    probe package metadata by distribution name (`wandb`). When only
    `byted-wandb` is installed this can fail even though `import wandb` works.
    The byted-wandb guide recommends remapping that metadata lookup early.
    """

    if os.environ.get("OPENPI_DISABLE_BYTED_WANDB_METADATA_PATCH", "0") in {"1", "true", "TRUE", "True"}:
        return

    old_metadata = importlib_metadata.metadata

    def metadata(name: str):
        if name == "wandb":
            try:
                return old_metadata("byted-wandb")
            except importlib_metadata.PackageNotFoundError:
                pass
        return old_metadata(name)

    importlib_metadata.metadata = metadata


_patch_byted_wandb_metadata()

import numpy as np
import safetensors.torch
import torch
import tqdm
import tree
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType


def _strip_wandb_vendor_from_sys_path() -> None:
    """Prevent wandb vendored deps from shadowing real packages.

    Some internal wandb distributions may prepend `.../wandb/vendor` to sys.path,
    which can cause unrelated imports (e.g., IPython -> pygments) to pick up
    vendored, incompatible modules.
    """

    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    sys.path[:] = [p for p in sys.path if p and "/wandb/vendor" not in _norm(p)]

    # If pygments was already imported from wandb's vendor tree, drop it so the
    # next import resolves to the real `pygments` package.
    for name, mod in list(sys.modules.items()):
        if not name.startswith("pygments"):
            continue
        mod_file = getattr(mod, "__file__", "")
        if mod_file and "/wandb/vendor/pygments" in _norm(mod_file):
            sys.modules.pop(name, None)


_strip_wandb_vendor_from_sys_path()

# Lazily imported OpenPI modules.
_model = None  # type: ignore[assignment]
_normalize = None  # type: ignore[assignment]
_config = None  # type: ignore[assignment]
_data_loader = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import openpi.models.model as _model
    import openpi.shared.normalize as _normalize
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader


_WANDB = None


class _TwoPhaseUpdateController:
    """Keep two KI graphs separate while committing one optimizer update.

    Accelerate 1.13's DeepSpeed wrapper calls ``engine.step()`` from every
    synced ``accelerator.backward()``. The KI loop has two backwards per outer
    microbatch, so this adapter drives the engine directly and defers the one
    real update until both graphs have been freed.

    The installed DeepSpeed runtime does not have verified support for multiple
    disjoint backwards with ZeRO-1/2 optimizer offload before a single engine
    step, so that exact combination is rejected before model execution. This is
    not a blanket rejection of gradient accumulation: standard single-backward
    training never instantiates this controller. ZeRO-3, no optimizer offload,
    and parameter-only offload retain the existing controller semantics. The
    non-DeepSpeed path retains normal Accelerate semantics.
    """

    def __init__(self, accelerator) -> None:
        self._accelerator = accelerator
        self._is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
        self._engine = (
            accelerator.deepspeed_engine_wrapped.engine if self._is_deepspeed else None
        )
        if self._is_deepspeed:
            stage, offload_device, deepspeed_version = self._validate_deepspeed_two_phase_capability()
            if getattr(accelerator, "is_main_process", True):
                logging.getLogger().info(
                    "DeepSpeed two-phase capability: version=%s zero_stage=%s "
                    "optimizer_offload_device=%s boundary_policy=existing",
                    deepspeed_version,
                    stage,
                    offload_device,
                )

    def _validate_deepspeed_two_phase_capability(self) -> tuple[int, str, str]:
        """Reject only unverified ZeRO-1/2 optimizer-offload multi-backward."""

        try:
            deepspeed_version = importlib_metadata.version("deepspeed")
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError("DeepSpeed runtime metadata is unavailable") from exc

        required = ("zero_optimization_stage", "zero_offload_optimizer")
        missing = [name for name in required if not callable(getattr(self._engine, name, None))]
        if missing:
            raise RuntimeError(f"DeepSpeed engine is missing two-phase policy APIs: {missing}")

        try:
            stage = int(self._engine.zero_optimization_stage())
        except (TypeError, ValueError) as exc:
            raise RuntimeError("DeepSpeed ZeRO stage is unavailable") from exc

        # ZeRO-3 is outside the measured defect predicate. Do not classify or
        # reject its optimizer/parameter offload policy here.
        if stage not in {1, 2}:
            return stage, "outside-target", deepspeed_version

        offload_config = self._engine.zero_offload_optimizer()
        if offload_config is None:
            device = "none"
        else:
            raw_device = getattr(offload_config, "device", None)
            if raw_device is None:
                raise RuntimeError("DeepSpeed optimizer offload device is unavailable")
            device = str(getattr(raw_device, "value", raw_device)).lower()
            device = device.removeprefix("offloaddeviceenum.")
        if device not in {"none", "cpu", "nvme"}:
            raise RuntimeError(f"Unknown DeepSpeed optimizer offload device: {device}")

        if device in {"cpu", "nvme"}:
            raise RuntimeError(
                "PI05-KI two-phase training requires multiple engine.backward calls per optimizer step, "
                f"but DeepSpeed {deepspeed_version} ZeRO-{stage} optimizer offload device={device} is "
                "unsupported by the installed runtime. Use reviewed no-offload; do not enable this mode "
                "until a runtime containing DeepSpeed PR #7981 is source-fingerprint and effect validated. "
                "Standard single-backward gradient accumulation is unaffected."
            )
        return stage, device, deepspeed_version

    @property
    def is_deepspeed(self) -> bool:
        return self._is_deepspeed

    def backward_first_phase(self, loss: torch.Tensor) -> None:
        """Backpropagate the first phase without declaring an update boundary."""

        if not self._is_deepspeed:
            self._accelerator.backward(loss)
            return

        # The measured no-optimizer-offload path must retain both disjoint
        # phases until the final boundary. Optimizer-offload runtimes that need
        # the inverse sequence are rejected by the constructor.
        self._engine.set_gradient_accumulation_boundary(False)
        self._engine.backward(loss)

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate the final phase under the outer accumulation boundary."""

        if not self._is_deepspeed:
            self._accelerator.backward(loss)
            return

        boundary = bool(self._accelerator.sync_gradients)
        self._engine.set_gradient_accumulation_boundary(boundary)
        self._engine.backward(loss)

    def clip_grad_norm_before_step(self, parameters, *, max_norm: float):
        """Clip before a non-DeepSpeed update; DS clips inside ``engine.step``."""

        if self._is_deepspeed:
            # Before the first engine step, DeepSpeed's cached global norm is
            # None. Reading it through Accelerate's clip_grad_norm_ would not
            # clip anything and would later crash when converted to float.
            return None
        return self._accelerator.clip_grad_norm_(parameters, max_norm=max_norm)

    def step_and_zero_grad(self, optimizer):
        """Commit once and return DeepSpeed's post-step cached grad norm."""

        if not self._accelerator.sync_gradients:
            return None
        if self._is_deepspeed:
            # DeepSpeed computes/clips the real norm inside `_take_model_step`,
            # caches it on the engine, then owns the single gradient clear.
            self._engine.step()
            return self._engine.get_global_grad_norm()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return None

    def clear_gradients(self, optimizer) -> None:
        """Discard a partial accumulation after a rejected KI microbatch."""

        if self._is_deepspeed:
            # ZeRO optimizers own partitioned gradient state; clear through the
            # engine optimizer rather than Accelerate's no-op wrapper. Match
            # DeepSpeed's own `_take_model_step` clearing path for ZeRO-2.
            self._engine.optimizer.zero_grad()
            return
        optimizer.zero_grad(set_to_none=True)


def _grad_norm_to_float(grad_norm) -> float:
    """Convert a reported norm for logging without ever calling ``float(None)``."""

    if grad_norm is None:
        return float("nan")
    if isinstance(grad_norm, torch.Tensor):
        return float(grad_norm.item())
    return float(grad_norm)


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


def _default_hf_datasets_cache(config: _config.TrainConfig, *, accelerator: Accelerator) -> Path:
    """Derive a rank-consistent Arrow cache root on node-local ``/tmp``."""
    from behavior.learning.datas.hf_cache_sync import resolve_cache_run_id

    distributed = accelerator.num_processes > 1
    run_id = resolve_cache_run_id(distributed=distributed)
    if run_id == "standalone":
        checkpoint_identity = Path(
            getattr(config, "checkpoint_dir", config.checkpoint_base_dir)
        ).expanduser().resolve()
        run_id = f"standalone:{checkpoint_identity}"
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:24]
    user = os.environ.get("USER", "unknown").strip() or "unknown"
    return Path("/tmp") / "openpi-comet" / user / "hf-datasets" / run_digest


def configure_hf_cache(config: _config.TrainConfig, *, accelerator: Accelerator) -> None:
    offline = os.environ.get("OPENPI_OFFLINE", "1") == "1"
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if os.environ.get("OPENPI_TORCH_COMPILE_SAMPLE_ACTIONS", "0") != "1":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    checkpoints_root = Path(os.path.abspath(Path(config.checkpoint_base_dir).expanduser()))
    hf_home = Path(os.environ.get("HF_HOME", str(checkpoints_root / "hf_home"))).expanduser()
    hub_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))).expanduser()
    transformers_cache = Path(os.environ.get("TRANSFORMERS_CACHE", str(hf_home / "transformers"))).expanduser()
    configured_datasets_cache = os.environ.get("HF_DATASETS_CACHE", "").strip()
    datasets_cache = (
        Path(configured_datasets_cache).expanduser()
        if configured_datasets_cache
        else _default_hf_datasets_cache(config, accelerator=accelerator)
    )

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

    # Do not inspect or create HF_DATASETS_CACHE here. Strict validation and
    # normal mkdir are deferred until load_hf_dataset has a generation-scoped
    # c10d failure protocol, so no rank can fail before its peers can observe it.
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


def _init_wandb_run(config: _config.TrainConfig, *, resuming: bool):
    wandb = _get_wandb()
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    settings = wandb.Settings(init_timeout=120)
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        run = wandb.init(id=run_id, resume="must", project=config.project_name, settings=settings)
    else:
        run = wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            settings=settings,
        )
        if run is None or not getattr(run, "id", None):
            raise RuntimeError("wandb.init returned no run id")
        (ckpt_dir / "wandb_id.txt").write_text(run.id)
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    return run


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        logging.info("wandb logging disabled")
        return None

    try:
        return _init_wandb_run(config, resuming=resuming)
    except Exception as exc:
        # Legacy configs retain optional W&B initialization.
        debug = os.environ.get("OPENPI_WANDB_DEBUG", "0") in {"1", "true", "TRUE", "True"}
        if debug:
            logging.warning("wandb init failed; continuing without wandb", exc_info=True)
        else:
            logging.warning("wandb init failed (%s); continuing without wandb", type(exc).__name__)
        try:
            object.__setattr__(config, "wandb_enabled", False)
        except Exception:
            pass
        return None


# Names that opt into the formal B1K protocol. Recipe values live only in the
# registered TrainConfig; this set selects invariant validation without copying
# batch, precision, schedule, or cadence literals into the trainer.
_FORMAL_B1K_CHECKPOINT_POLICIES = {
    "pi05_ki_joint_fast_b1k-full_task-ki_on_bf16": "step",
    "pi05_ki_joint_query_b1k-full_task-ki_on_bf16": "step",
    "pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32": "step",
    "pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32_validation10": "step",
    "pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32": "step",
    "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16": "epoch_with_rolling",
    "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16": "epoch_with_rolling",
}
_FORMAL_B1K_CONFIGS = set(_FORMAL_B1K_CHECKPOINT_POLICIES)
_H20_FORMAL_B1K_CONFIGS = {
    "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16",
    "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16",
}
# H20 shares the A100 memory policy: ZeRO-2 with the optimizer resident on GPU.
# H20 carries ~96 GB HBM3 versus A100-40GB, so if no-offload is safe on A100 it
# is safe here; the bounded smoke still has to produce the real measured peak.
_H20_BF16_NO_OPTIMIZER_OFFLOAD_CONFIGS = {
    "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16",
    "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16",
    "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16_smoke",
    "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16_smoke",
}
_A100_BF16_NO_OPTIMIZER_OFFLOAD_CONFIGS = {
    "pi05_ki_joint_fast_b1k-full_task-ki_on_a100_bf16",
    "pi05_ki_joint_query_b1k-full_task-ki_on_a100_bf16",
}
_FORMAL_B1K_DATASET_ENV_KEYS = (
    "OPENPI_B1K_ANCHOR_STRIDE",
    "OPENPI_B1K_ANCHOR_OFFSET",
    "OPENPI_B1K_DROP_INCOMPLETE_HORIZON",
)


def _is_formal_b1k_mode(config) -> bool:
    return getattr(config, "name", None) in _FORMAL_B1K_CONFIGS


def _validate_profile_recipe_unchanged(config, *, world_size: int | None = None) -> None:
    """Reject mutation by comparing one effective recipe on both sides."""
    from openpi.training.launcher_profile import materialize_effective_recipe
    from openpi.training.train_config import TrainConfig
    from openpi.training.train_config import get_config

    # Lightweight unit-test stubs exercise the protocol's arithmetic branches;
    # pristine-profile equality applies to real CLI-produced TrainConfig values.
    if not isinstance(config, TrainConfig):
        return
    pristine = get_config(config.name)
    if pristine.name != config.name:
        raise ValueError(
            f"formal profile {config.name!r} is not registered exactly; refusing fallback"
        )
    if world_size is not None:
        pristine = materialize_effective_recipe(pristine, world_size=world_size)
    fields = (
        "pytorch_model_name",
        "pytorch_training_precision",
        "accelerate_mixed_precision",
        "batch_size_per_gpu",
        "gradient_accumulation_steps",
        "expected_global_batch",
        "num_train_steps",
        "num_train_epochs",
        "num_workers",
        "save_interval",
        "val_log_interval",
        "val_batch_size",
        "val_num_batches",
        "val_episodes_per_task",
        "val_anchors_per_episode",
        "val_deterministic_subset",
        "val_deterministic_flow",
        "val_log_per_task",
        "val_interval_samples",
        "save_interval_samples",
        "streaming_anchor_stride",
        "epoch_anchor_offsets",
        "checkpoint_policy",
        "rolling_checkpoint_interval",
        "gradient_checkpointing",
        "project_name",
        "wandb_enabled",
    )
    for field in fields:
        actual = getattr(config, field)
        expected = getattr(pristine, field)
        if actual != expected:
            raise ValueError(
                f"Formal B1K profile {config.name!r} requires effective {field}={expected!r}; "
                f"got {actual!r}. Direct recipe overrides are not supported; define a "
                "registered profile."
            )
    if config.lr_schedule != pristine.lr_schedule:
        raise ValueError(
            f"Formal B1K profile {config.name!r} requires its registered lr_schedule; "
            "direct schedule overrides are not supported"
        )


@contextmanager
def _baseline_b1k_dataset_env():
    """Construct validation datasets with legacy stride-1/padding defaults."""

    saved = {key: os.environ.get(key) for key in _FORMAL_B1K_DATASET_ENV_KEYS}
    try:
        for key in _FORMAL_B1K_DATASET_ENV_KEYS:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _train_b1k_anchor_stride_env(stride: int):
    """Pin ``OPENPI_B1K_ANCHOR_STRIDE`` for the train-loader build only.

    The B1K streaming dataset reads its anchor stride from the environment at
    construction time. We set it from ``config.streaming_anchor_stride`` scoped
    to the train-dataloader build and restore the prior state afterwards, so
    the config field stays the source of truth and validation can independently
    force stride-1 via :func:`_baseline_b1k_dataset_env`.
    """

    env_key = "OPENPI_B1K_ANCHOR_STRIDE"
    saved = os.environ.get(env_key)
    try:
        os.environ[env_key] = str(int(stride))
        yield
    finally:
        if saved is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = saved


@contextmanager
def _train_b1k_anchor_env(stride: int, offset: int = 0):
    """Pin both ``OPENPI_B1K_ANCHOR_STRIDE`` and ``..._OFFSET`` for a loader build.

    Used for per-epoch rebuilds when ``config.epoch_anchor_offsets`` rotates
    the offset between epochs. The dataset captures these at construction, so
    rebuilding the dataset under a new offset is the only way to make a later
    epoch consume different anchor positions.
    """

    stride_key = "OPENPI_B1K_ANCHOR_STRIDE"
    offset_key = "OPENPI_B1K_ANCHOR_OFFSET"
    saved_stride = os.environ.get(stride_key)
    saved_offset = os.environ.get(offset_key)
    try:
        os.environ[stride_key] = str(int(stride))
        os.environ[offset_key] = str(int(offset))
        yield
    finally:
        for key, saved in ((stride_key, saved_stride), (offset_key, saved_offset)):
            if saved is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved


def _close_training_iterator(iterator) -> None:
    if iterator is None:
        return
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _cosine_lr_value(
    step: int,
    *,
    warmup_steps: int,
    peak_lr: float,
    decay_steps: int,
    end_lr: float,
) -> float:
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return end_lr + (peak_lr - end_lr) * cosine


def _validate_formal_b1k_contract(config, *, accelerator=None) -> None:
    if not _is_formal_b1k_mode(config):
        return

    world_size = None if accelerator is None else int(accelerator.num_processes)
    _validate_profile_recipe_unchanged(config, world_size=world_size)

    # Only cross-profile protocol invariants are literals here. Batch shape,
    # precision, schedule, stride, and cadences are owned by TrainConfig and are
    # checked for internal/runtime consistency below.
    expected_fields = {
        "checkpoint_policy": _FORMAL_B1K_CHECKPOINT_POLICIES[config.name],
        "overwrite": False,
        "wandb_enabled": True,
        "project_name": "pi05_ki",
    }
    for field, expected in expected_fields.items():
        actual = getattr(config, field, None)
        if actual != expected:
            raise ValueError(
                f"Formal B1K requires {field}={expected!r}; got {actual!r}. "
                "Runtime overrides are not supported."
            )
    if getattr(config, "name", None) in _H20_FORMAL_B1K_CONFIGS:
        h20_validation_fields = {
            "val_batch_size": 8,
            "val_num_batches": 16,
            "val_episodes_per_task": 20,
            "val_anchors_per_episode": 4,
            "val_deterministic_subset": True,
            "val_deterministic_flow": True,
            "val_log_per_task": True,
        }
        for field, expected in h20_validation_fields.items():
            actual = getattr(config, field, None)
            if actual != expected:
                raise ValueError(
                    f"Formal H20 validation requires {field}={expected}; got {actual!r}"
                )
    elif bool(getattr(config, "resume", False)):
        raise ValueError(
            "Formal B1K resume is supported only for the two H20 formal profiles"
        )
    if len(config.data) != 1 or len(config.val_data) != 1:
        raise ValueError("Formal B1K requires exactly one train and one validation data config")
    train_data = config.data[0]
    val_data = config.val_data[0]
    for label, data, expected_episodes in (
        ("train", train_data, list(range(180))),
        ("validation", val_data, list(range(180, 200))),
    ):
        base_config = data.base_config
        if base_config.tasks is not None:
            raise ValueError(f"Formal B1K {label} data must cover all tasks (tasks=None)")
        if base_config.episodes_index != expected_episodes:
            raise ValueError(
                f"Formal B1K {label} episodes must be {expected_episodes[0]}..{expected_episodes[-1]}"
            )
        if base_config.skill_bridge.enabled:
            raise ValueError(f"Formal B1K {label} data requires Skill Bridge disabled")

    # ---- Schedule CONSISTENCY (not remembered values) ----
    # The budget is derived at runtime from the dataset and the global batch, so
    # there is no correct literal to compare against here. What must hold is
    # that the pieces agree with each other and with the epoch-based budget.
    schedule = config.lr_schedule
    warmup_steps = int(schedule.warmup_steps)
    decay_steps = int(schedule.decay_steps)
    peak_lr = float(schedule.peak_lr)
    epochs = getattr(config, "num_train_epochs", None)
    provided_steps = int(config.num_train_steps)

    if epochs is None:
        # LEGACY fixed-budget form, still used by the on_bf16 / on_v100_fp32
        # families. Consistency only: the budget must be positive and the cosine
        # decay must span exactly that budget. No remembered literal.
        if provided_steps <= 0:
            raise ValueError(
                "Formal B1K with num_train_epochs=None requires num_train_steps > 0; "
                f"got {provided_steps}"
            )
        if decay_steps != provided_steps:
            raise ValueError(
                f"Formal B1K fixed-budget form requires decay_steps == num_train_steps; "
                f"got decay_steps={decay_steps}, num_train_steps={provided_steps}"
            )
        if warmup_steps >= provided_steps:
            raise ValueError(
                f"Formal B1K requires warmup_steps < num_train_steps; "
                f"got {warmup_steps} >= {provided_steps}"
            )
    else:
        # DERIVED form: budget comes from epochs x steps_per_epoch at runtime, so
        # the step fields must be sentinels. A literal here would cap or decouple
        # the derived budget and silently change coverage.
        if int(epochs) <= 0:
            raise ValueError(f"Formal B1K requires num_train_epochs >= 1; got {epochs!r}")
        if provided_steps != 0:
            raise ValueError(
                "Formal B1K epoch-derived form requires num_train_steps=0 (derive from "
                f"epochs x steps_per_epoch); got {provided_steps}. A non-zero value caps the "
                "derived budget and would silently truncate coverage."
            )
        if decay_steps != 0:
            raise ValueError(
                "Formal B1K epoch-derived form requires decay_steps=0 so it is auto-set to the "
                f"derived num_train_steps; got {decay_steps}. A literal decouples LR decay from "
                "the real budget."
            )
    if float(schedule.decay_lr) != 0.0:
        raise ValueError(f"Formal B1K requires decay_lr=0; got {schedule.decay_lr!r}")
    if warmup_steps <= 0:
        raise ValueError(f"Formal B1K requires warmup_steps > 0; got {warmup_steps}")
    if peak_lr <= 0.0:
        raise ValueError(f"Formal B1K requires peak_lr > 0; got {peak_lr}")

    stride = int(getattr(config, "streaming_anchor_stride", 1))
    if stride < 1:
        raise ValueError(f"Formal B1K requires streaming_anchor_stride >= 1; got {stride}")
    if getattr(config, "epoch_anchor_offsets", None) is not None:
        raise ValueError(
            "Formal B1K no longer uses per-epoch anchor offsets: the offset rotation was an "
            "unnecessary decomposition of a single stride-N sweep and required mid-training "
            "dataloader rebuilds. Set streaming_anchor_stride and leave offsets unset."
        )

    # The dataset captures stride/offset from the environment at construction
    # time, so the stride the trainer will actually use is derived from the
    # config rather than pinned to a literal. Offset 0 with no rotation.
    formal_dataset_defaults = {
        "OPENPI_B1K_ANCHOR_STRIDE": str(stride),
        "OPENPI_B1K_ANCHOR_OFFSET": "0",
        "OPENPI_B1K_DROP_INCOMPLETE_HORIZON": "1",
    }
    for key, expected in formal_dataset_defaults.items():
        os.environ.setdefault(key, expected)
        if os.environ[key] != expected:
            raise ValueError(f"Formal B1K requires initial {key}={expected}; got {os.environ[key]!r}")

    os.environ.setdefault("OPENPI_PERSISTENT_WORKERS", "0")
    if os.environ["OPENPI_PERSISTENT_WORKERS"] != "0":
        raise ValueError("Formal B1K requires OPENPI_PERSISTENT_WORKERS=0")

    if accelerator is not None:
        if int(accelerator.num_processes) != 32:
            raise ValueError(f"Formal B1K requires world size 32; got {accelerator.num_processes}")
        expected_grad_accum = int(config.gradient_accumulation_steps)
        if int(accelerator.gradient_accumulation_steps) != expected_grad_accum:
            raise ValueError(
                "Formal B1K Accelerator/config mismatch: "
                f"gradient_accumulation_steps={accelerator.gradient_accumulation_steps}, "
                f"config={expected_grad_accum}"
            )
        effective_global_batch = (
            int(config.batch_size_per_gpu)
            * int(accelerator.num_processes)
            * int(accelerator.gradient_accumulation_steps)
        )
        expected_global_batch = getattr(config, "expected_global_batch", None)
        if expected_global_batch is None or effective_global_batch != int(expected_global_batch):
            raise ValueError(
                f"Formal B1K profile requires global batch {expected_global_batch}; "
                f"got {effective_global_batch} (B{config.batch_size_per_gpu} x "
                f"W{accelerator.num_processes} x GA{accelerator.gradient_accumulation_steps})"
            )
        if config.pytorch_training_precision == "float32":
            if accelerator.distributed_type != DistributedType.DEEPSPEED:
                raise ValueError("Formal V100 FP32 requires DeepSpeed")
            ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
            zero_config = ds_config.get("zero_optimization", {})
            if zero_config.get("stage") != 2:
                raise ValueError("Formal V100 FP32 requires DeepSpeed ZeRO stage 2")
            offload_config = zero_config.get("offload_optimizer", {})
            if offload_config.get("device") != "cpu":
                raise ValueError("Formal V100 FP32 requires DeepSpeed CPU optimizer offload")


def _init_formal_b1k_wandb(config, *, accelerator, resuming: bool = False) -> None:
    """Require rank-0 Byted-W&B init and broadcast one matched result."""

    if not _is_formal_b1k_mode(config) or config.prepare_hf_cache_only:
        return

    error = None
    if accelerator.is_main_process:
        try:
            _init_wandb_run(config, resuming=resuming)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    payload = [error]
    distributed_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
    if int(accelerator.num_processes) > 1:
        if not distributed_ready:
            payload[0] = "distributed process group is not initialized for W&B consensus"
        else:
            torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"Formal B1K requires online Byted-W&B initialization on rank 0: {payload[0]}")


_CHECKPOINT_POLICY_STEP = "step"
_CHECKPOINT_POLICY_EPOCH_WITH_ROLLING = "epoch_with_rolling"
_ROLLING_CHECKPOINT_LINK_NAME = "rolling_latest"
_ROLLING_CHECKPOINT_DIR_PREFIX = ".rolling_step_"


def _checkpoint_save_kind(
    config,
    *,
    global_step: int,
    steps_per_epoch: int | None,
) -> str | None:
    """Return the checkpoint kind due at ``global_step``.

    The default ``step`` branch intentionally retains the historical trigger
    expression. The opt-in epoch policy makes epoch boundaries durable and uses
    a separate fixed-name rolling pointer for within-epoch recovery.
    """
    if global_step <= 0:
        return None

    policy = getattr(config, "checkpoint_policy", _CHECKPOINT_POLICY_STEP)
    if policy == _CHECKPOINT_POLICY_STEP:
        should_save = global_step % config.save_interval == 0 or global_step == config.num_train_steps
        return "step" if should_save else None

    if policy != _CHECKPOINT_POLICY_EPOCH_WITH_ROLLING:
        raise ValueError(f"Unsupported checkpoint_policy: {policy}")
    if steps_per_epoch is None or int(steps_per_epoch) <= 0:
        raise ValueError("epoch_with_rolling checkpoint policy requires positive steps_per_epoch")

    rolling_interval = int(getattr(config, "rolling_checkpoint_interval", 0))
    if rolling_interval <= 0:
        raise ValueError(
            "epoch_with_rolling checkpoint policy requires --rolling-checkpoint-interval > 0"
        )

    steps_per_epoch = int(steps_per_epoch)
    if global_step % steps_per_epoch == 0:
        return "epoch"
    if global_step % rolling_interval == 0 or global_step == int(config.num_train_steps):
        return "rolling"
    return None


def _remove_checkpoint_path(path: Path) -> None:
    """Remove a file, symlink, or directory without following symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _checkpoint_step_from_name(name: str) -> int | None:
    if name.isdigit():
        return int(name)
    if name.startswith(_ROLLING_CHECKPOINT_DIR_PREFIX):
        step_text = name.removeprefix(_ROLLING_CHECKPOINT_DIR_PREFIX)
        if step_text.isdigit():
            return int(step_text)
    return None


def _rolling_checkpoint_target(checkpoint_dir: Path) -> Path | None:
    """Return the valid in-root target of ``rolling_latest``, if present."""
    link_path = checkpoint_dir / _ROLLING_CHECKPOINT_LINK_NAME
    if not link_path.is_symlink():
        return None
    try:
        target = (checkpoint_dir / os.readlink(link_path)).resolve(strict=True)
        checkpoint_root = checkpoint_dir.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    if (
        target.parent != checkpoint_root
        or not target.is_dir()
        or _checkpoint_step_from_name(target.name) is None
    ):
        return None
    return target


def _cleanup_rolling_checkpoint_artifacts(
    checkpoint_dir: Path,
    *,
    keep_target: Path | None = None,
) -> None:
    """Remove stale rolling targets and interrupted rolling-save temporaries."""
    if not checkpoint_dir.exists():
        return
    keep_resolved = keep_target.resolve(strict=False) if keep_target is not None else None
    for candidate in checkpoint_dir.iterdir():
        name = candidate.name
        if name.startswith(("tmp_rolling_", f".{_ROLLING_CHECKPOINT_LINK_NAME}.tmp.")):
            _remove_checkpoint_path(candidate)
            continue
        if name.startswith(_ROLLING_CHECKPOINT_DIR_PREFIX):
            candidate_resolved = candidate.resolve(strict=False)
            if keep_resolved is None or candidate_resolved != keep_resolved:
                _remove_checkpoint_path(candidate)


def _publish_rolling_checkpoint(checkpoint_dir: Path, target_dir: Path) -> None:
    """Atomically point ``rolling_latest`` at target and remove older rolling data.

    The checkpoint data is fully written before this function runs. Publishing a
    temporary symlink with ``os.replace`` makes resume discovery switch from the
    old complete checkpoint to the new complete checkpoint atomically.
    """
    checkpoint_root = checkpoint_dir.resolve(strict=True)
    target_resolved = target_dir.resolve(strict=True)
    if target_resolved.parent != checkpoint_root:
        raise ValueError(f"Rolling checkpoint target must be inside {checkpoint_root}: {target_dir}")

    link_path = checkpoint_dir / _ROLLING_CHECKPOINT_LINK_NAME
    tmp_link = checkpoint_dir / f".{_ROLLING_CHECKPOINT_LINK_NAME}.tmp.{os.getpid()}"
    _remove_checkpoint_path(tmp_link)
    tmp_link.symlink_to(target_resolved.name, target_is_directory=True)

    # A real directory at this reserved path can only come from an interrupted
    # pre-policy experiment or manual intervention. Remove it once so subsequent
    # symlink publications use atomic os.replace exclusively.
    if os.path.lexists(link_path) and not link_path.is_symlink():
        _remove_checkpoint_path(link_path)
    os.replace(tmp_link, link_path)
    _cleanup_rolling_checkpoint_artifacts(checkpoint_dir, keep_target=target_resolved)


def _checkpoint_step_from_dir(checkpoint_dir: Path) -> int | None:
    """Read a checkpoint step from a durable or rolling checkpoint directory."""
    try:
        resolved = checkpoint_dir.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None

    named_step = _checkpoint_step_from_name(resolved.name)
    if named_step is not None:
        return named_step

    manifest_path = resolved / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        return int(manifest["run_metadata"]["global_step"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _basic_checkpoint_step(checkpoint_dir: Path) -> int | None:
    """Return a step only when metadata, manifest, and resume state agree."""
    step = _checkpoint_step_from_dir(checkpoint_dir)
    if step is None:
        return None
    try:
        metadata = torch.load(
            checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False
        )
        manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
        metadata_step = int(metadata["global_step"])
        manifest_step = int(manifest["run_metadata"]["global_step"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if step != metadata_step or step != manifest_step:
        return None
    if not (checkpoint_dir / "accelerate_state").is_dir():
        return None
    return step


def _latest_step_dir(
    checkpoint_dir: Path,
    *,
    checkpoint_policy: str | None = None,
) -> tuple[int, Path] | None:
    """Discover a durable or atomically published rolling checkpoint.

    ``epoch_with_rolling`` treats ``rolling_latest`` as the current run's
    publication pointer and therefore prefers it over numeric directories that
    may belong to an older run. If no pointer exists, numeric discovery remains
    the compatibility fallback. An existing but invalid pointer fails closed.
    """
    if not checkpoint_dir.exists():
        return None

    prefer_rolling = checkpoint_policy == _CHECKPOINT_POLICY_EPOCH_WITH_ROLLING
    durable_candidates = [
        d
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and not d.is_symlink() and d.name.isdigit()
    ]
    latest = None
    for candidate in durable_candidates:
        step = (
            _basic_checkpoint_step(candidate)
            if prefer_rolling
            else _checkpoint_step_from_dir(candidate)
        )
        if step is not None and (latest is None or step > latest[0]):
            latest = (step, candidate)

    rolling_link = checkpoint_dir / _ROLLING_CHECKPOINT_LINK_NAME
    rolling_target = _rolling_checkpoint_target(checkpoint_dir)
    rolling_step = (
        _basic_checkpoint_step(rolling_target)
        if prefer_rolling and rolling_target is not None
        else _checkpoint_step_from_dir(rolling_target) if rolling_target is not None else None
    )
    if prefer_rolling and os.path.lexists(rolling_link) and rolling_step is None:
        raise ValueError(f"invalid rolling_latest checkpoint pointer: {rolling_link}")
    if rolling_step is not None and (prefer_rolling or latest is None or rolling_step > latest[0]):
        latest = (rolling_step, rolling_link)
    return latest


def _validate_accelerate_state_artifacts(
    state_dir: Path,
    *,
    world_size: int,
    require_h20_deepspeed: bool,
) -> None:
    """Fail closed when a published formal state is missing rank artifacts."""
    if not state_dir.is_dir():
        raise FileNotFoundError(f"checkpoint is missing accelerate_state: {state_dir}")
    if not require_h20_deepspeed:
        return
    # Accelerate + DeepSpeed ZeRO-2 writes one replicated model-state file,
    # one optimizer shard per rank, and one RNG file per rank.
    model_states = list(state_dir.glob("**/*model_states.pt"))
    optimizer_states = list(state_dir.glob("**/*optim_states.pt"))
    rng_states = list(state_dir.glob("random_states_*.pkl"))
    expected = {
        "model state files": (len(model_states), 1),
        "optimizer state shards": (len(optimizer_states), int(world_size)),
        "RNG states": (len(rng_states), int(world_size)),
    }
    for label, (actual_count, expected_count) in expected.items():
        if actual_count != expected_count:
            raise ValueError(
                f"Formal H20 checkpoint requires {expected_count} {label}; "
                f"found {actual_count} in {state_dir}"
            )


def _validate_h20_resume_compatibility(
    *,
    manifest: dict,
    config,
    data_config,
    accelerator,
) -> None:
    """Check only identity-bearing H20 fields; logger/validation changes are allowed."""
    saved_config = manifest.get("config") or {}
    saved_hardware = manifest.get("hardware") or {}
    current_fingerprint = _build_data_fingerprint(config, data_config)
    saved_fingerprint = manifest.get("data_fingerprint") or {}
    checks = {
        "config name/arm": (saved_config.get("name"), config.name),
        "batch_size_per_gpu": (
            saved_config.get("batch_size_per_gpu"),
            int(config.batch_size_per_gpu),
        ),
        "gradient_accumulation_steps": (
            saved_config.get("gradient_accumulation_steps"),
            int(config.gradient_accumulation_steps),
        ),
        "world_size": (saved_hardware.get("num_gpus"), int(accelerator.num_processes)),
        "precision": (
            saved_hardware.get("precision"),
            str(config.pytorch_training_precision),
        ),
        "strategy": (
            saved_hardware.get("strategy"),
            str(accelerator.distributed_type),
        ),
        "data fingerprint": (
            saved_fingerprint.get("sha256"),
            current_fingerprint.get("sha256"),
        ),
    }
    mismatches = {
        label: {"saved": saved, "current": current}
        for label, (saved, current) in checks.items()
        if saved is None or saved != current
    }
    saved_commit = (manifest.get("git") or {}).get("commit")
    current_commit = _get_git_info().get("commit")
    allowed_commits = {
        "f89a7ce1e7f0148a6ef1113b5861d09c70d0c2b6",
        current_commit,
    }
    if saved_commit not in allowed_commits:
        mismatches["source commit"] = {
            "saved": saved_commit,
            "allowed": sorted(str(value) for value in allowed_commits),
        }
    if mismatches:
        raise ValueError(f"Formal H20 resume compatibility mismatch: {mismatches}")


def _inspect_resume_checkpoint(
    selected_step: int,
    selected_dir: Path,
    *,
    formal: bool,
    config=None,
    data_config=None,
    accelerator=None,
) -> dict:
    """Validate basic checkpoint completeness and return saved metadata."""
    resolved = selected_dir.resolve(strict=True)
    name_step = _checkpoint_step_from_name(resolved.name)
    if name_step is None:
        raise ValueError(f"resume checkpoint has an invalid step-bearing name: {resolved.name}")

    metadata_path = resolved / "metadata.pt"
    manifest_path = resolved / "manifest.json"
    if formal and not metadata_path.is_file():
        raise FileNotFoundError(f"Formal resume checkpoint is missing metadata.pt: {resolved}")
    if formal and not manifest_path.is_file():
        raise FileNotFoundError(f"Formal resume checkpoint is missing manifest.json: {resolved}")

    metadata = (
        torch.load(metadata_path, map_location="cpu", weights_only=False)
        if metadata_path.is_file()
        else {}
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    metadata_step = int(metadata.get("global_step", selected_step))
    manifest_step = int((manifest.get("run_metadata") or {}).get("global_step", selected_step))
    recorded_steps = {
        "selected": int(selected_step),
        "name": int(name_step),
        "metadata": metadata_step,
        "manifest": manifest_step,
    }
    if len(set(recorded_steps.values())) != 1:
        raise ValueError(f"checkpoint step mismatch: {recorded_steps}")
    if formal and config is not None:
        if data_config is None or accelerator is None:
            raise ValueError("Formal H20 resume compatibility requires runtime data and accelerator")
        _validate_h20_resume_compatibility(
            manifest=manifest,
            config=config,
            data_config=data_config,
            accelerator=accelerator,
        )
    logging.info(
        "Resume checkpoint selected name=%s manifest_step=%s metadata_step=%s",
        resolved.name,
        manifest_step,
        metadata_step,
    )
    return metadata


def _load_resume_state(
    accelerator,
    *,
    checkpoint_dir: Path,
    checkpoint_policy: str,
    formal: bool,
    selected: tuple[int, Path] | None = None,
    config=None,
    data_config=None,
) -> tuple[int, dict, Path]:
    """Load the generic Accelerate/DeepSpeed state from one validated checkpoint."""
    selection = selected or _latest_step_dir(
        checkpoint_dir, checkpoint_policy=checkpoint_policy
    )
    if selection is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    selected_step, selected_dir = selection
    metadata = _inspect_resume_checkpoint(
        selected_step,
        selected_dir,
        formal=formal,
        config=config,
        data_config=data_config,
        accelerator=accelerator,
    )
    acc_state_dir = selected_dir / "accelerate_state"
    _validate_accelerate_state_artifacts(
        acc_state_dir,
        world_size=int(getattr(accelerator, "num_processes", 1)),
        require_h20_deepspeed=formal and config is not None,
    )
    accelerator.load_state(str(acc_state_dir))
    logging.info(
        "Resumed optimizer/model/RNG state at step %s; optimizer resume; data order restart",
        selected_step,
    )
    return selected_step, metadata, selected_dir


class _SampleProgress:
    def __init__(self, accum_num_samples: int, samples_per_update: int, epoch_num_samples: int):
        self.accum_num_samples = int(accum_num_samples)
        self.samples_per_update = int(samples_per_update)
        self.epoch_num_samples = int(epoch_num_samples)

    @classmethod
    def fresh(cls, *, samples_per_update: int, epoch_num_samples: int) -> "_SampleProgress":
        if samples_per_update <= 0 or epoch_num_samples <= 0:
            raise ValueError("sample progress requires positive update and epoch sample counts")
        return cls(0, int(samples_per_update), int(epoch_num_samples))

    def record_update(self, *, committed: bool) -> None:
        if committed:
            self.accum_num_samples += self.samples_per_update

    def metrics(self) -> dict[str, float | int]:
        return {
            "dataset/accum_num_samples": int(self.accum_num_samples),
            "dataset/epoch_fraction": float(self.accum_num_samples / self.epoch_num_samples),
            "dataset/samples_per_update": int(self.samples_per_update),
            "dataset/epoch_num_samples": int(self.epoch_num_samples),
        }

    def checkpoint_payload(self) -> dict[str, int]:
        return {
            "accum_num_samples": int(self.accum_num_samples),
            "samples_per_update": int(self.samples_per_update),
            "epoch_num_samples": int(self.epoch_num_samples),
        }


def _saved_samples_per_update(metadata: dict) -> int | None:
    saved_config = metadata.get("config") or {}
    saved_runtime = metadata.get("accelerate") or {}
    per_gpu = saved_config.get("batch_size_per_gpu")
    world_size = saved_runtime.get("num_processes")
    grad_accum = saved_config.get("gradient_accumulation_steps", 1)
    if per_gpu is not None and world_size is not None:
        value = int(per_gpu) * int(world_size) * int(grad_accum)
        return value if value > 0 else None
    saved_global_batch = saved_config.get("expected_global_batch") or saved_config.get("batch_size")
    if saved_global_batch is not None:
        # expected_global_batch/batch_size already describe the committed
        # optimizer-update batch, including gradient accumulation.
        value = int(saved_global_batch)
        return value if value > 0 else None
    return None


def _restore_sample_progress(
    metadata: dict,
    *,
    current_samples_per_update: int,
    default_epoch_num_samples: int,
) -> _SampleProgress:
    """Restore cumulative samples, using saved recipe values for legacy checkpoints."""
    payload = metadata.get("sample_progress") or {}
    if payload:
        accum_num_samples = int(payload["accum_num_samples"])
        epoch_num_samples = int(payload["epoch_num_samples"])
    else:
        saved_samples_per_update = _saved_samples_per_update(metadata)
        if saved_samples_per_update is None:
            raise ValueError("Legacy checkpoint cannot derive its saved global batch")
        accum_num_samples = int(metadata.get("global_step", 0)) * saved_samples_per_update
        saved_config = metadata.get("config") or {}
        saved_steps = int(saved_config.get("num_train_steps") or 0)
        saved_epochs = int(saved_config.get("num_train_epochs") or 0)
        epoch_num_samples = (
            (saved_steps // saved_epochs) * saved_samples_per_update
            if saved_steps > 0 and saved_epochs > 0
            else int(default_epoch_num_samples)
        )
    if accum_num_samples < 0 or epoch_num_samples <= 0 or current_samples_per_update <= 0:
        raise ValueError("Invalid sample progress in checkpoint metadata")
    return _SampleProgress(
        accum_num_samples=accum_num_samples,
        samples_per_update=int(current_samples_per_update),
        epoch_num_samples=epoch_num_samples,
    )


def build_datasets(config: _config.TrainConfig):
    from behavior.learning.datas.hf_cache_sync import DistributedCacheError

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
        except DistributedCacheError as exc:
            if (not exc.retryable) or attempt >= retries:
                raise
            delay_s = float(os.environ.get("OPENPI_BUILD_DATASET_RETRY_SLEEP_S", "2")) * attempt
            logging.warning(
                "Rank %s observed canonical retryable cache failure on attempt %s/%s; "
                "all ranks acknowledged generation %s. Retrying in %.1fs: %s",
                rank,
                attempt,
                retries,
                exc.generation_id,
                delay_s,
                exc,
            )
            time.sleep(delay_s)
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


class _StratifiedValUnavailable(Exception):
    """Raised internally when a stratified val subset cannot be built.

    Used to unwind to the legacy streaming val loader instead of validating on
    an empty subset.
    """


class _FixedIndexSubset(torch.utils.data.Dataset):
    """Map-style view over an explicit index list, carrying task ids alongside.

    Requires the underlying dataset to honor ``__getitem__(idx)``, i.e. it must
    be built with ``chunk_streaming_using_keyframe=False``.
    """

    def __init__(self, base, indices, task_ids):
        self._base = base
        self._indices = list(indices)
        self.task_ids = list(task_ids)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._base[self._indices[i]]


def _build_stratified_val_indices(raw_dataset, config, *, logger_=None):
    """Build a FIXED, task-stratified index list over the validation pool.

    Picks ``config.val_episodes_per_task`` episodes from EVERY task and
    ``config.val_anchors_per_episode`` anchors inside each chosen episode,
    with the anchors spread across the episode's PHASES (not clustered at the
    start, which is what the streaming cursor effectively did).

    Returns (indices, task_ids, coverage_dict). Deterministic given
    ``config.val_subset_seed``.
    """
    inner = getattr(raw_dataset, "_dataset", raw_dataset)
    episodes = list(getattr(inner, "episodes", []) or [])
    edi = getattr(inner, "episode_data_index", None)

    bounds: dict[int, tuple[int, int]] = {}
    if edi is not None:
        try:
            frm, to = edi["from"], edi["to"]
            for pos, ep in enumerate(episodes):
                bounds[int(ep)] = (int(frm[pos]), int(to[pos]))
        except Exception:  # noqa: BLE001
            bounds = {}
    if not bounds:
        # Fallback: uniform partition of the flat index space.
        per = max(1, len(raw_dataset) // max(1, len(episodes)))
        for i, ep in enumerate(episodes):
            bounds[int(ep)] = (i * per, min(len(raw_dataset), (i + 1) * per))

    # B1K encodes task id in the episode index: episode_index // 10000
    by_task: dict[int, list[tuple[int, int, int]]] = {}
    for ep, (a, b) in bounds.items():
        by_task.setdefault(ep // 10000, []).append((ep, a, b))

    rng = random.Random(int(getattr(config, "val_subset_seed", 12345)))
    n_eps = int(getattr(config, "val_episodes_per_task", 10))
    n_anch = max(1, int(getattr(config, "val_anchors_per_episode", 1)))
    formal_h20 = getattr(config, "name", None) in _H20_FORMAL_B1K_CONFIGS
    if formal_h20:
        if len(by_task) != 50:
            raise ValueError(
                f"Formal H20 validation requires 50 tasks; found {len(by_task)}"
            )
        short = {task: len(candidates) for task, candidates in by_task.items() if len(candidates) != n_eps}
        if short:
            raise ValueError(
                f"Formal H20 validation requires exactly {n_eps} episodes per task; found {short}"
            )

    indices: list[int] = []
    task_ids: list[int] = []
    used_eps = 0
    for task in sorted(by_task):
        cand = sorted(by_task[task], key=lambda x: x[0])
        if len(cand) > n_eps:
            cand = sorted(rng.sample(cand, n_eps), key=lambda x: x[0])
        for ep, a, b in cand:
            episode_length = b - a
            if episode_length <= 0:
                if formal_h20:
                    raise ValueError(f"Formal H20 validation episode {ep} is empty")
                continue
            if formal_h20 and episode_length < n_anch + 2:
                raise ValueError(
                    f"Formal H20 validation episode {ep} is too short for {n_anch} distinct interior anchors"
                )
            used_eps += 1
            episode_anchors = [
                a + ((j + 1) * (episode_length - 1)) // (n_anch + 1)
                for j in range(n_anch)
            ]
            if formal_h20 and (
                len(set(episode_anchors)) != n_anch
                or min(episode_anchors) <= a
                or max(episode_anchors) >= b - 1
            ):
                raise ValueError(
                    f"Formal H20 validation episode {ep} did not produce distinct interior anchors: "
                    f"{episode_anchors}"
                )
            indices.extend(episode_anchors)
            task_ids.extend([int(task)] * len(episode_anchors))

    # Non-formal validation retains the historical deterministic interleave.
    # Formal H20 keeps task order so a task-homogeneous batch plan can turn
    # scalar batch metrics into correct task means without changing model APIs.
    if not formal_h20:
        order = list(range(len(indices)))
        random.Random(int(getattr(config, "val_subset_seed", 12345)) + 1).shuffle(order)
        indices = [indices[i] for i in order]
        task_ids = [task_ids[i] for i in order]

    coverage = {
        "n_samples": len(indices),
        "n_unique_anchors": len(set(indices)),
        "n_tasks": len(set(task_ids)),
        "n_raw_episodes": len(bounds),
        "n_episodes": used_eps,
        "episodes_per_task": n_eps,
        "anchors_per_episode": n_anch,
        "seed": int(getattr(config, "val_subset_seed", 12345)),
    }
    if formal_h20:
        expected = {
            "n_samples": 4_000,
            "n_unique_anchors": 4_000,
            "n_tasks": 50,
            "n_raw_episodes": 1_000,
            "n_episodes": 1_000,
        }
        actual = {key: coverage[key] for key in expected}
        if actual != expected:
            raise ValueError(
                f"Formal H20 validation population invariant failed: expected={expected} actual={actual}"
            )
    if logger_ is not None:
        logger_.info(
            "Deterministic val subset: %d samples | %d tasks | %d episodes "
            "(episodes_per_task=%d anchors_per_episode=%d seed=%d)",
            coverage["n_samples"], coverage["n_tasks"], coverage["n_episodes"],
            n_eps, n_anch, coverage["seed"],
        )
    return indices, task_ids, coverage


def _cyclic_pad_to_length(values: list, target_length: int) -> list:
    """Return ``values`` repeated cyclically to exactly ``target_length``."""
    if target_length < len(values):
        raise ValueError(f"target_length={target_length} is shorter than source={len(values)}")
    if target_length == len(values):
        return list(values)
    if not values:
        raise ValueError("cannot pad an empty sequence")
    needed = target_length - len(values)
    repeats, remainder = divmod(needed, len(values))
    return list(values) + list(values) * repeats + list(values[:remainder])


class _TaskHomogeneousBatchPlan:
    def __init__(
        self,
        *,
        batches: list[list[int]],
        task_ids: list[int],
        unique_counts: list[int],
        batches_per_rank: int,
    ):
        self.batches = batches
        self.task_ids = task_ids
        self.unique_counts = unique_counts
        self.batches_per_rank = int(batches_per_rank)


def _build_task_homogeneous_batch_plan(
    indices: list[int],
    task_ids: list[int],
    *,
    batch_size: int,
    world_size: int,
) -> _TaskHomogeneousBatchPlan:
    """Build task-pure batches and append full padding batches for DDP symmetry."""
    if len(indices) != len(task_ids) or not indices:
        raise ValueError("task-homogeneous validation requires aligned non-empty indices")
    if batch_size <= 0 or world_size <= 0:
        raise ValueError("batch_size and world_size must be positive")

    by_task: dict[int, list[int]] = {}
    for position, task_id in enumerate(task_ids):
        by_task.setdefault(int(task_id), []).append(position)

    batches: list[list[int]] = []
    batch_tasks: list[int] = []
    unique_counts: list[int] = []
    for task_id in sorted(by_task):
        task_indices = by_task[task_id]
        if len(task_indices) % batch_size != 0:
            raise ValueError(
                f"task {task_id} has {len(task_indices)} anchors, not divisible by batch_size={batch_size}"
            )
        for offset in range(0, len(task_indices), batch_size):
            batches.append(task_indices[offset : offset + batch_size])
            batch_tasks.append(task_id)
            unique_counts.append(batch_size)

    target_batches = ((len(batches) + world_size - 1) // world_size) * world_size
    original_batches = list(batches)
    original_tasks = list(batch_tasks)
    for offset in range(target_batches - len(batches)):
        source = offset % len(original_batches)
        batches.append(list(original_batches[source]))
        batch_tasks.append(original_tasks[source])
        unique_counts.append(0)
    return _TaskHomogeneousBatchPlan(
        batches=batches,
        task_ids=batch_tasks,
        unique_counts=unique_counts,
        batches_per_rank=target_batches // world_size,
    )


class _DistributedBatchPlanSampler(torch.utils.data.Sampler[list[int]]):
    """Select every rank's deterministic slice of a global batch plan."""

    def __init__(self, plan: _TaskHomogeneousBatchPlan, *, rank: int, world_size: int):
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank={rank} is outside world_size={world_size}")
        self._batches = plan.batches[rank::world_size]
        self.batch_task_ids = plan.task_ids[rank::world_size]
        self.batch_unique_counts = plan.unique_counts[rank::world_size]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _DeterministicValLoader:
    """Yields ``(Observation, actions)`` from a fixed-index val subset.

    Mirrors ``data_loader.DataLoaderImpl`` (which converts the numpy output of
    ``_collate_fn`` into tensors) but iterates a plain finite DataLoader exactly
    ONCE per ``__iter__``, so every validation scores the same fixed sample set.
    Also exposes ``batch_task_ids`` so validation can report per-task means.
    """

    def __init__(self, torch_loader, data_config, subset, coverage):
        self._loader = torch_loader
        self._data_config = data_config
        self._subset = subset
        self.coverage = coverage
        self.batch_task_ids: list[list[int]] = []
        self.batch_unique_counts: list[int] = []

    def data_config(self):
        return self._data_config

    def __len__(self):
        return len(self._loader)

    @staticmethod
    def _to_tensors(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: _DeterministicValLoader._to_tensors(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_DeterministicValLoader._to_tensors(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return torch.as_tensor(obj)
        return obj

    def __iter__(self):
        import openpi.models.model as _model_mod

        # Rank-local slice of the fixed index list, in loader order, so batch i
        # can be mapped back to the tasks and unique-anchor counts it contains.
        batch_sampler = getattr(self._loader, "batch_sampler", None)
        if isinstance(batch_sampler, _DistributedBatchPlanSampler):
            self.batch_task_ids = [
                [task] * len(batch)
                for task, batch in zip(
                    batch_sampler.batch_task_ids,
                    batch_sampler._batches,
                    strict=True,
                )
            ]
            self.batch_unique_counts = list(batch_sampler.batch_unique_counts)
        else:
            sampler = getattr(self._loader, "sampler", None)
            local = list(sampler) if sampler is not None else list(range(len(self._subset)))
            bs = self._loader.batch_size
            self.batch_task_ids = [
                [self._subset.task_ids[j] for j in local[b * bs : (b + 1) * bs]]
                for b in range(len(local) // bs)
            ]
            self.batch_unique_counts = [len(tasks) for tasks in self.batch_task_ids]
        for batch in self._loader:
            batch = self._to_tensors(batch)
            yield _model_mod.Observation.from_dict(batch), batch["actions"]


def build_val_datasets(config: _config.TrainConfig):
    """Build validation data loader(s) from config.val_data.

    Returns (val_loader, val_data_config) or (None, None) if val_data is empty.
    Uses shuffle=False and reuses the same norm stats as the training data.
    """
    from behavior.learning.datas.hf_cache_sync import DistributedCacheError

    if not config.val_data:
        return None, None

    # ---- FIXED, task-stratified, reproducible validation subset (default ON) ----
    if getattr(config, "val_deterministic_subset", True):
        original_data = config.data
        try:
            object.__setattr__(config, "data", config.val_data)
            val_factory = config.val_data[0]
            val_dc = val_factory.create(config.assets_dirs, config.model)
            # Map-style access is REQUIRED for an explicit index list to mean
            # anything: in streaming mode __getitem__ ignores idx.
            val_dc = dataclasses.replace(
                val_dc, chunk_streaming_using_keyframe=False, dataset_shuffle=False
            )
            raw = _data_loader.create_torch_dataset(
                val_dc, config.model.action_horizon, config.model
            )
            indices, task_ids, coverage = _build_stratified_val_indices(
                raw, config, logger_=logging.getLogger()
            )
            if not indices:
                # The dataset did not expose per-episode index bounds, so no
                # stratified list could be built (e.g. a non-BEHAVIOR-1K val
                # dataset). Fall back to the legacy loader rather than running
                # validation on an empty subset.
                logging.warning(
                    "val_deterministic_subset is enabled but no stratified index list "
                    "could be built from this val dataset (no per-episode bounds). "
                    "Falling back to the legacy streaming val loader; val metrics will "
                    "NOT be reproducible or task-stratified for this run."
                )
                raise _StratifiedValUnavailable
            ds = _data_loader.transform_dataset(raw, val_dc, skip_norm_stats=False)

            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            per_gpu = (
                int(config.val_batch_size)
                if getattr(config, "val_batch_size", None) is not None
                else max(1, int(config.batch_size) // max(1, world_size))
            )

            formal_h20 = getattr(config, "name", None) in _H20_FORMAL_B1K_CONFIGS
            batch_sampler = None
            sampler = None
            if formal_h20:
                plan = _build_task_homogeneous_batch_plan(
                    indices,
                    task_ids,
                    batch_size=per_gpu,
                    world_size=world_size,
                )
                batch_sampler = _DistributedBatchPlanSampler(
                    plan,
                    rank=int(os.environ.get("RANK", "0")),
                    world_size=world_size,
                )
                coverage.update(
                    {
                        "n_padded": len(plan.batches) * per_gpu,
                        "n_duplicated": len(plan.batches) * per_gpu - len(indices),
                        "n_batches_per_rank": plan.batches_per_rank,
                        "val_global_batch": per_gpu * world_size,
                    }
                )
                expected = {
                    "n_samples": 4_000,
                    "n_unique_anchors": 4_000,
                    "n_padded": 4_096,
                    "n_duplicated": 96,
                    "n_batches_per_rank": 16,
                    "val_global_batch": 256,
                }
                actual = {key: coverage.get(key) for key in expected}
                if actual != expected:
                    raise ValueError(
                        f"Formal H20 validation execution invariant failed: expected={expected} actual={actual}"
                    )
                logging.info(
                    "Formal H20 val: raw_episodes=1000 unique=4000 executed=4096 "
                    "duplicates=96 batches_per_rank=16 global_batch=256"
                )
            else:
                # General deterministic validation pads cyclically to a whole
                # global batch, including when the required pad exceeds the
                # source length. Formal H20 uses the task-pure plan above.
                stride = max(1, per_gpu * max(1, world_size))
                remainder = len(indices) % stride
                if remainder:
                    pad = stride - remainder
                    target = len(indices) + pad
                    indices = _cyclic_pad_to_length(indices, target)
                    task_ids = _cyclic_pad_to_length(task_ids, target)
                else:
                    pad = 0
                coverage["n_padded"] = len(indices)
                coverage["n_duplicated"] = pad
                if torch.distributed.is_initialized() and world_size > 1:
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        _FixedIndexSubset(ds, indices, task_ids),
                        num_replicas=world_size,
                        rank=int(os.environ.get("RANK", "0")),
                        shuffle=False,
                        drop_last=True,
                    )

            subset = _FixedIndexSubset(ds, indices, task_ids)
            loader_kwargs = {
                "dataset": subset,
                "num_workers": int(config.num_workers),
                "collate_fn": _data_loader._collate_fn,
                "pin_memory": torch.cuda.is_available(),
                **({"prefetch_factor": 2} if int(config.num_workers) > 0 else {}),
                **({"persistent_workers": True} if int(config.num_workers) > 0 else {}),
            }
            if batch_sampler is not None:
                loader_kwargs["batch_sampler"] = batch_sampler
            else:
                loader_kwargs.update(
                    {
                        "batch_size": per_gpu,
                        "shuffle": False,
                        "sampler": sampler,
                        "drop_last": True,
                    }
                )
            # Persistent workers keep the deterministic validation worker pool
            # alive across repeated passes; loader_kwargs applies this only when
            # num_workers > 0, as required by PyTorch.
            loader = torch.utils.data.DataLoader(**loader_kwargs)
            loader = _DeterministicValLoader(loader, val_dc, subset, coverage)
            return loader, val_dc
        except _StratifiedValUnavailable:
            pass  # fall through to the legacy streaming loader below
        finally:
            object.__setattr__(config, "data", original_data)

    # Temporarily swap config.data → config.val_data to reuse create_data_loader.
    # We use a mutable copy approach: swap, build, swap back.
    original_data = config.data
    try:
        object.__setattr__(config, "data", config.val_data)
        # val batch size: use val_batch_size if set, else same as train
        if getattr(config, "val_batch_size", None) is not None:
            original_bs = config.batch_size
            object.__setattr__(config, "batch_size", config.val_batch_size)
        else:
            original_bs = None

        skip_norm_stats = os.environ.get("OPENPI_SKIP_NORM_STATS", "0") == "1"
        retries = max(1, int(os.environ.get("OPENPI_BUILD_DATASET_RETRIES", "3")))
        rank = int(os.environ.get("RANK", "0"))

        for attempt in range(1, retries + 1):
            try:
                val_loader = _data_loader.create_data_loader(
                    config,
                    framework="pytorch",
                    shuffle=False,  # no shuffle for validation
                    num_batches=getattr(config, "val_num_batches", None),
                    skip_norm_stats=skip_norm_stats,
                )
                val_data_config = val_loader.data_config()
                break
            except DistributedCacheError as exc:
                if (not exc.retryable) or attempt >= retries:
                    raise
                delay_s = float(os.environ.get("OPENPI_BUILD_DATASET_RETRY_SLEEP_S", "2")) * attempt
                logging.warning(
                    "Rank %s observed canonical retryable val-cache failure on attempt %s/%s; "
                    "all ranks acknowledged generation %s. Retrying in %.1fs: %s",
                    rank,
                    attempt,
                    retries,
                    exc.generation_id,
                    delay_s,
                    exc,
                )
                time.sleep(delay_s)
            except FileNotFoundError as exc:
                transient_lock_race = exc.filename is None and int(os.environ.get("WORLD_SIZE", "1")) > 1
                if (not transient_lock_race) or attempt >= retries:
                    raise
                delay_s = float(os.environ.get("OPENPI_BUILD_DATASET_RETRY_SLEEP_S", "2")) * attempt
                logging.warning(
                    "Rank %s hit transient ENOENT during val dataset init (attempt %s/%s). Retrying in %.1fs",
                    rank, attempt, retries, delay_s,
                )
                time.sleep(delay_s)

        if original_bs is not None:
            object.__setattr__(config, "batch_size", original_bs)
    finally:
        object.__setattr__(config, "data", original_data)

    return val_loader, val_data_config


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


def _safe_set_nested(config: dict, key_path: str, value) -> None:
    keys = key_path.split(".")
    node = config
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value in {"1", "true", "TRUE", "True", "yes", "YES", "y", "Y"}


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def _env_bool_for_json(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value in {"1", "true", "TRUE", "True", "yes", "YES", "y", "Y"}


def _fp16_stability_profile_enabled() -> bool:
    return _env_flag("OPENPI_FP16_STABILITY_PROFILE", False)


def _patch_deepspeed_config(
    ds_config: dict,
    *,
    effective_batch_size: int,
    grad_accum_steps: int,
    world_size: int,
    precision: str,
    clip_grad_norm: float,
) -> None:
    ds_config["train_micro_batch_size_per_gpu"] = int(effective_batch_size)
    ds_config["gradient_accumulation_steps"] = int(grad_accum_steps)
    ds_config["train_batch_size"] = int(effective_batch_size * grad_accum_steps * world_size)
    fp16_stability_profile = _fp16_stability_profile_enabled() and precision == "float16"
    ds_grad_clip = _env_float("OPENPI_DS_GRADIENT_CLIPPING")
    default_grad_clip = 0.5 if fp16_stability_profile else clip_grad_norm
    ds_config["gradient_clipping"] = float(default_grad_clip if ds_grad_clip is None else ds_grad_clip)

    if precision == "bfloat16":
        _safe_set_nested(ds_config, "bf16.enabled", True)
        _safe_set_nested(ds_config, "fp16.enabled", False)
    elif precision == "float16":
        _safe_set_nested(ds_config, "bf16.enabled", False)
        _safe_set_nested(ds_config, "fp16.enabled", True)
        _safe_set_nested(ds_config, "fp16.auto_cast", False)
        fp16_profile_defaults = {
            "OPENPI_FP16_INITIAL_SCALE_POWER": 10,
            "OPENPI_FP16_LOSS_SCALE_WINDOW": 1000,
            "OPENPI_FP16_HYSTERESIS": 2,
            "OPENPI_FP16_MIN_LOSS_SCALE": 1,
        }
        for env_name, key_path in (
            ("OPENPI_FP16_INITIAL_SCALE_POWER", "fp16.initial_scale_power"),
            ("OPENPI_FP16_LOSS_SCALE_WINDOW", "fp16.loss_scale_window"),
            ("OPENPI_FP16_HYSTERESIS", "fp16.hysteresis"),
            ("OPENPI_FP16_MIN_LOSS_SCALE", "fp16.min_loss_scale"),
        ):
            override = _env_int(env_name)
            if override is None and fp16_stability_profile:
                override = fp16_profile_defaults[env_name]
            if override is not None:
                _safe_set_nested(ds_config, key_path, override)
    else:
        _safe_set_nested(ds_config, "bf16.enabled", False)
        _safe_set_nested(ds_config, "fp16.enabled", False)

    reduce_bucket_size = _env_int("OPENPI_DS_REDUCE_BUCKET_SIZE")
    if reduce_bucket_size is None and fp16_stability_profile:
        reduce_bucket_size = 50_000_000
    if reduce_bucket_size is not None:
        _safe_set_nested(ds_config, "zero_optimization.reduce_bucket_size", reduce_bucket_size)
    allgather_bucket_size = _env_int("OPENPI_DS_ALLGATHER_BUCKET_SIZE")
    if allgather_bucket_size is None and fp16_stability_profile:
        allgather_bucket_size = 50_000_000
    if allgather_bucket_size is not None:
        _safe_set_nested(ds_config, "zero_optimization.allgather_bucket_size", allgather_bucket_size)
    overlap_comm = _env_bool_for_json("OPENPI_DS_OVERLAP_COMM")
    if overlap_comm is not None:
        _safe_set_nested(ds_config, "zero_optimization.overlap_comm", overlap_comm)
    pin_memory = _env_bool_for_json("OPENPI_DS_OFFLOAD_PIN_MEMORY")
    if pin_memory is None and fp16_stability_profile:
        pin_memory = False
    if pin_memory is not None:
        _safe_set_nested(ds_config, "zero_optimization.offload_optimizer.pin_memory", pin_memory)

    # Do not combine DeepSpeed precision engines with torch_autocast.
    _safe_set_nested(ds_config, "torch_autocast.enabled", False)


def _validate_deepspeed_precision_config(accelerator: Accelerator, ds_config: dict, *, precision: str) -> None:
    ds_fp16 = bool(ds_config.get("fp16", {}).get("enabled", False))
    ds_bf16 = bool(ds_config.get("bf16", {}).get("enabled", False))
    ds_torch_autocast = bool(ds_config.get("torch_autocast", {}).get("enabled", False))

    if ds_torch_autocast:
        raise ValueError("DeepSpeed torch_autocast.enabled must be false when using the Accelerate trainer.")
    if precision == "float16" and not ds_fp16:
        raise ValueError("Requested float16 training but DeepSpeed fp16.enabled=false.")
    if precision == "bfloat16" and not ds_bf16:
        raise ValueError("Requested bfloat16 training but DeepSpeed bf16.enabled=false.")
    if precision == "float32" and (ds_fp16 or ds_bf16):
        raise ValueError("Requested float32 training but DeepSpeed fp16/bf16 is still enabled.")

    accel_mp = accelerator.mixed_precision
    if precision == "float16" and accel_mp not in ("fp16", "no"):
        raise ValueError(f"Requested float16 training but accelerator.mixed_precision={accel_mp}.")
    if precision == "bfloat16" and accel_mp not in ("bf16", "no"):
        raise ValueError(f"Requested bfloat16 training but accelerator.mixed_precision={accel_mp}.")
    if precision == "float32" and accel_mp != "no":
        raise ValueError(f"Requested float32 training but accelerator.mixed_precision={accel_mp}.")


def _fast_grad_norm_enabled() -> bool:
    """Whether the batched fp32 ZeRO gradient-norm replacement is installed.

    Defaults to disabled so unset environments keep DeepSpeed's stock float64
    behavior. Set ``OPENPI_DS_FAST_GRAD_NORM=1`` to enable.
    """
    return os.environ.get("OPENPI_DS_FAST_GRAD_NORM", "0").strip().lower() in {"1", "true", "yes"}


def _patch_deepspeed_grad_norm() -> None:
    """Compute ZeRO gradient norms in fp32 with batched multi-tensor kernels.

    DeepSpeed's ``get_grad_norm_direct`` casts **every** gradient tensor to
    float64 before taking its norm::

        torch.linalg.vector_norm(g.data.double().detach(), ord=norm_type)

    This model has ~812 parameter tensors, so each call issues roughly 1600
    kernels (one cast plus one norm per tensor), and the call runs once per
    parameter group per optimizer step. Profiling with ``py-spy --native``
    showed the main thread busy inside ``libcuda`` rather than waiting on the
    GPU, i.e. the cost is kernel-launch overhead rather than arithmetic.

    float64 is not needed for this quantity, and it is not what the reference
    pipelines in this repository use:

    * ``scripts/train.py`` clips through ``optax.clip_by_global_norm``, and JAX
      leaves ``jax_enable_x64`` disabled by default, so float64 is silently
      downgraded and the norm is computed in float32 at best.
    * ``scripts/train_pytorch.py`` calls ``torch.nn.utils.clip_grad_norm_``,
      which preserves the gradient dtype and therefore returns bfloat16 for
      bfloat16 gradients.

    Measured against a float64 reference over 812 tensors, float32 accumulation
    differs by about 5e-7 relative, while the sum of squared norms keeps roughly
    1e32 headroom before float32 would overflow. Accumulating in the native
    bfloat16 instead would give ~7e-5, so the dtype is pinned to float32.

    The replacement keeps cross-rank and model-parallel reduction, the parameter
    filtering rules, the ``inf``-norm branch and the non-finite masking
    behavior. It falls back to the original implementation on any error.
    """
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    except ImportError:
        return

    if getattr(DeepSpeedZeroOptimizer, "_openpi_fast_grad_norm_patched", False):
        return

    ds_module = sys.modules[DeepSpeedZeroOptimizer.__module__]
    _orig_get_grad_norm_direct = DeepSpeedZeroOptimizer.get_grad_norm_direct

    @functools.wraps(_orig_get_grad_norm_direct)
    def _patched_get_grad_norm_direct(self, gradients, params, norm_type=2):
        norm_type = float(norm_type)
        # The infinity norm is cheap and rarely used; keep DeepSpeed's version.
        if norm_type == ds_module.inf:
            return _orig_get_grad_norm_direct(self, gradients, params, norm_type)

        try:
            selected = []
            for grad, param in zip(gradients, params):
                if grad is None:
                    continue
                # Pipeline parallelism may replicate parameters; avoid multi-counting.
                if getattr(param, ds_module.PIPE_REPLICATED, False):
                    continue
                if ds_module.is_model_parallel_parameter(param) or self.model_parallel_rank == 0:
                    selected.append(grad.detach())

            total_norm = torch.zeros((), dtype=torch.float32, device=self.device)
            if selected:
                # _foreach_norm requires a uniform device and dtype per batch.
                batches: dict[tuple, list] = {}
                for tensor in selected:
                    batches.setdefault((tensor.device, tensor.dtype), []).append(tensor)
                for batch in batches.values():
                    norms = torch._foreach_norm(batch, norm_type, dtype=torch.float32)
                    total_norm = total_norm + torch.stack(norms).square().sum()
        except Exception as exc:
            _safe_log_warning(
                "fast ZeRO grad-norm failed (%s); falling back to the DeepSpeed implementation", exc
            )
            return _orig_get_grad_norm_direct(self, gradients, params, norm_type)

        # Sum of squared norms across data-parallel and model-parallel ranks.
        ds_module.dist.all_reduce(total_norm, op=ds_module.dist.ReduceOp.SUM, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=total_norm, op=ds_module.dist.ReduceOp.SUM)

        total_norm = total_norm.pow(1.0 / norm_type)
        ds_module.mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)
        return total_norm

    DeepSpeedZeroOptimizer.get_grad_norm_direct = _patched_get_grad_norm_direct
    DeepSpeedZeroOptimizer._openpi_fast_grad_norm_patched = True
    logging.info(
        "Patched DeepSpeed get_grad_norm_direct: batched fp32 norms (was per-tensor float64)"
    )


def _patch_deepspeed_autocast(accelerator: Accelerator) -> None:
    """Patch DeepSpeed engine to be transparent to external torch.autocast contexts.

    In DeepSpeed >= 0.17.2, ``autocast_if_enabled()`` wraps the engine forward.
    When ``torch_autocast.enabled=false`` in the DS config (which we force to
    avoid double mixed-precision with ``fp16.enabled=true``), the engine detects
    an outer ``torch.autocast`` and **explicitly disables** it via
    ``torch.autocast(enabled=False)``.  This strips autocast from the entire
    forward pass, causing float32-activation-vs-float16-weight mismatches.

    The patch makes ``torch_autocast_enabled`` / ``torch_autocast_dtype`` on the
    engine fall through to the active ``torch.autocast`` state so that the
    engine re-enables (rather than disables) autocast during forward.
    """
    if getattr(accelerator.state, "deepspeed_plugin", None) is None:
        return

    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except ImportError:
        return

    if getattr(DeepSpeedEngine, "_openpi_autocast_patched", False):
        return

    _orig_enabled = DeepSpeedEngine.torch_autocast_enabled
    _orig_dtype = DeepSpeedEngine.torch_autocast_dtype

    def _patched_enabled(self):
        return _orig_enabled(self) or torch.is_autocast_enabled()

    def _patched_dtype(self):
        if not _orig_enabled(self) and torch.is_autocast_enabled():
            return torch.get_autocast_dtype("cuda")
        return _orig_dtype(self)

    DeepSpeedEngine.torch_autocast_enabled = _patched_enabled
    DeepSpeedEngine.torch_autocast_dtype = _patched_dtype
    DeepSpeedEngine._openpi_autocast_patched = True
    logging.info(
        "Patched DeepSpeedEngine autocast: engine now falls through to external torch.autocast context."
    )


def _patch_deepspeed_loss_scaler() -> None:
    """Keep training when dynamic loss scale reaches the configured minimum.

    Some DeepSpeed versions type `fp16.min_loss_scale` as an integer config field,
    which prevents using fractional minima such as `1e-8`. In long V100 FP16 runs,
    occasional overflow events can still drive the scaler down to the minimum.
    Instead of hard-failing the whole job at that point, keep DeepSpeed's normal
    overflow behavior (skip step + hold/reduce scale) and disable the fatal exit.
    """
    try:
        from deepspeed.runtime.fp16.loss_scaler import DynamicLossScaler
    except ImportError:
        return

    if getattr(DynamicLossScaler, "_openpi_min_scale_patched", False):
        return

    _orig_init = DynamicLossScaler.__init__

    @functools.wraps(_orig_init)
    def _patched_init(
        self,
        init_scale,
        scale_window,
        min_scale,
        delayed_shift,
        consecutive_hysteresis,
        raise_error_at_min_scale=True,
        dtype=torch.half,
    ):
        return _orig_init(
            self,
            init_scale,
            scale_window,
            min_scale,
            delayed_shift,
            consecutive_hysteresis,
            raise_error_at_min_scale=False,
            dtype=dtype,
        )

    DynamicLossScaler.__init__ = _patched_init
    DynamicLossScaler._openpi_min_scale_patched = True
    logging.info(
        "Patched DeepSpeed DynamicLossScaler: reaching min_loss_scale will skip steps instead of exiting."
    )


def _get_deepspeed_loss_scale(accelerator: Accelerator) -> float | None:
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        return None
    # Accelerate wraps the engine; try multiple known attribute paths.
    engine_wrapper = getattr(accelerator, "deepspeed_engine_wrapped", None)
    engine = getattr(engine_wrapper, "engine", engine_wrapper)  # unwrap if needed
    if engine is None:
        return None

    # Candidate objects that may hold the loss scale value:
    # 1. engine.optimizer.loss_scaler (FP16_DeepSpeedZeroOptimizer)
    # 2. engine.optimizer  (sometimes exposes cur_scale directly)
    # 3. engine itself     (DeepSpeedEngine.loss_scale property)
    optimizer = getattr(engine, "optimizer", None)
    loss_scaler = getattr(optimizer, "loss_scaler", None) if optimizer is not None else None
    candidates = [obj for obj in (loss_scaler, optimizer, engine) if obj is not None]

    for obj in candidates:
        for attr in ("cur_scale", "loss_scale"):
            value = getattr(obj, attr, None)
            if value is None:
                continue
            try:
                return float(value.item() if hasattr(value, "item") else value)
            except (TypeError, ValueError):
                continue
    return None


def _debug_overflow_enabled(config: _config.TrainConfig) -> bool:
    return bool(getattr(config, "debug_overflow", False))


def _summarize_tensor_for_debug(name: str, value: torch.Tensor) -> str:
    tensor = value.detach()
    finite_mask = torch.isfinite(tensor)
    finite_count = int(finite_mask.sum().item())
    total_count = tensor.numel()
    if finite_count > 0:
        finite_tensor = tensor[finite_mask].float()
        min_value = float(finite_tensor.min().item())
        max_value = float(finite_tensor.max().item())
        mean_abs = float(finite_tensor.abs().mean().item())
    else:
        min_value = float("nan")
        max_value = float("nan")
        mean_abs = float("nan")
    return (
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={finite_count}/{total_count} min={min_value:.6g} "
        f"max={max_value:.6g} mean_abs={mean_abs:.6g}"
    )


def _tensor_debug_payload(value: torch.Tensor, *, max_values: int = 16) -> dict[str, object]:
    tensor = value.detach()
    finite_mask = torch.isfinite(tensor)
    finite_count = int(finite_mask.sum().item())
    total_count = tensor.numel()
    if finite_count > 0:
        finite_tensor = tensor[finite_mask].float()
        min_value = float(finite_tensor.min().item())
        max_value = float(finite_tensor.max().item())
        mean_abs = float(finite_tensor.abs().mean().item())
    else:
        min_value = float("nan")
        max_value = float("nan")
        mean_abs = float("nan")

    flat = tensor.flatten()
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite_count": finite_count,
        "total_count": total_count,
        "min": min_value,
        "max": max_value,
        "mean_abs": mean_abs,
        "sample": flat[:max_values].float().cpu(),
    }


def _observation_debug_payload(observation) -> dict[str, object]:
    payload: dict[str, object] = {}
    for attr_name in (
        "state",
        "images",
        "image_masks",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
    ):
        value = getattr(observation, attr_name, None)
        if isinstance(value, torch.Tensor):
            payload[attr_name] = _tensor_debug_payload(value)
        elif isinstance(value, dict):
            payload[attr_name] = {
                str(k): _tensor_debug_payload(v) for k, v in value.items() if isinstance(v, torch.Tensor)
            }
    return payload


def _log_nonfinite_batch_state(*, loss: torch.Tensor, actions: torch.Tensor, observation) -> None:
    logging.error(
        "Encountered non-finite loss before backward: value=%s dtype=%s",
        loss.detach().float().item(),
        loss.dtype,
    )
    logging.error(_summarize_tensor_for_debug("actions", actions))

    for attr_name in ("state", "images", "image_masks", "tokenized_prompt", "tokenized_prompt_mask"):
        value = getattr(observation, attr_name, None)
        if isinstance(value, torch.Tensor):
            logging.error(_summarize_tensor_for_debug(f"observation.{attr_name}", value))
        elif isinstance(value, dict):
            for key, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    logging.error(_summarize_tensor_for_debug(f"observation.{attr_name}.{key}", tensor))


def _save_nonfinite_debug_dump(
    *,
    output_dir: Path,
    global_step: int,
    accelerator: Accelerator,
    loss: torch.Tensor,
    actions: torch.Tensor,
    observation,
) -> None:
    try:
        debug_dir = output_dir / "nonfinite_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        rank = int(accelerator.process_index)
        local_rank = int(accelerator.local_process_index)
        dump_path = debug_dir / f"nonfinite_rank{rank}_local{local_rank}_step{global_step}.pt"
        torch.save(
            {
                "global_step": global_step,
                "rank": rank,
                "local_rank": local_rank,
                "num_processes": int(accelerator.num_processes),
                "loss": _tensor_debug_payload(loss),
                "actions": _tensor_debug_payload(actions),
                "observation": _observation_debug_payload(observation),
            },
            dump_path,
        )
        logging.error("Saved non-finite debug dump to %s", dump_path)
    except Exception:
        logging.exception("Failed to save non-finite debug dump")


def _safe_float(value: torch.Tensor | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().reshape(-1)[0].item())
    return float(value)


def _loss_tensor_debug_metrics(losses: object, actions: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    loss_tensor: torch.Tensor | None = None
    if isinstance(losses, torch.Tensor):
        loss_tensor = losses.detach().float()
    elif isinstance(losses, dict):
        for key in ("flow_loss_per_sample", "per_sample_loss", "loss_per_sample"):
            value = losses.get(key)
            if isinstance(value, torch.Tensor):
                metrics[f"{key}_max"] = _safe_float(value.detach().float().amax())
                metrics[f"{key}_mean"] = _safe_float(value.detach().float().mean())
                break

    if loss_tensor is not None:
        finite_mask = torch.isfinite(loss_tensor)
        metrics["loss_tensor_finite_ratio"] = float(finite_mask.float().mean().item()) if loss_tensor.numel() else 1.0
        if finite_mask.any():
            finite_loss = loss_tensor[finite_mask]
            metrics["loss_tensor_max"] = _safe_float(finite_loss.amax())
            metrics["loss_tensor_mean"] = _safe_float(finite_loss.mean())
            metrics["loss_tensor_std"] = _safe_float(finite_loss.std(unbiased=False))
            if loss_tensor.ndim >= 2:
                per_sample = loss_tensor.flatten(1).mean(dim=1)
                metrics["per_sample_loss_max"] = _safe_float(per_sample.amax())
                metrics["per_sample_loss_mean"] = _safe_float(per_sample.mean())
                metrics["per_sample_loss_std"] = _safe_float(per_sample.std(unbiased=False))
                metrics["per_sample_loss_argmax"] = _safe_float(torch.argmax(per_sample))

    actions_f = actions.detach().float()
    metrics["target_action_abs_max"] = _safe_float(actions_f.abs().amax())
    metrics["target_action_mean_abs"] = _safe_float(actions_f.abs().mean())
    return metrics


_PI05_KI_LOSS_WANDB_KEYS = (
    ("loss_backbone", "loss/backbone"),
    ("loss_ce", "loss/ce"),
    ("loss_action_ce", "loss/action_ce"),
    ("loss_query_mse", "loss/query_mse"),
    ("loss_expert", "loss/expert"),
    ("loss_flow_raw", "loss/flow_raw"),
    ("expert_loss_fraction", "loss/expert_fraction"),
)


def _add_pi05_ki_structured_backbone_metrics(extra_metrics: dict[str, float], backbone_loss: float) -> None:
    """Add arm-correct structured keys without aliasing CE as query MSE."""
    has_action_ce = "action_ce_loss" in extra_metrics
    has_query_mse = "query_mse_loss" in extra_metrics
    if has_action_ce and has_query_mse:
        raise ValueError("π0.5-KI backbone metrics cannot contain both action CE and query MSE")

    extra_metrics["loss_backbone"] = backbone_loss
    extra_metrics["loss_ce"] = extra_metrics.get("ce_loss", float("nan"))
    if has_action_ce:
        extra_metrics["loss_action_ce"] = extra_metrics["action_ce_loss"]
    elif has_query_mse:
        extra_metrics["loss_query_mse"] = extra_metrics["query_mse_loss"]


def _update_pi05_ki_wandb_loss_metrics(
    log_payload: dict[str, float], infos: list[dict[str, float]]
) -> None:
    """Map only metrics present for the active π0.5-KI objective to W&B."""
    for metric_key, wandb_key in _PI05_KI_LOSS_WANDB_KEYS:
        vals = [info[metric_key] for info in infos if metric_key in info and np.isfinite(info[metric_key])]
        if vals:
            log_payload[wandb_key] = sum(vals) / len(vals)


# =====================================================================
# Buffered metrics.jsonl writer: reduce I/O overhead on the hot path by
# batching metrics.jsonl writes and flushing only at boundaries.
# =====================================================================

_metrics_buffer: list[str] = []
_metrics_file_handle: object = None  # file or None
_metrics_atexit_registered: bool = False


def _metrics_buffer_init(file_handle) -> None:
    """Initialize the metrics buffer with the given file handle.

    Registers an atexit flush hook on first call.
    """
    global _metrics_file_handle, _metrics_atexit_registered
    _metrics_file_handle = file_handle
    if not _metrics_atexit_registered:
        atexit.register(_metrics_buffer_flush)
        _metrics_atexit_registered = True


def _metrics_buffer_append(record: dict) -> None:
    """Append a record dict to the metrics buffer (rank 0 only).

    The record is serialized to a JSON line and buffered.  Does NOT flush.
    """
    if _metrics_file_handle is None:
        return
    _metrics_buffer.append(json.dumps(record, default=str) + "\n")


def _safe_log_warning(message: str, *args, exc_info=None) -> None:
    """Best-effort warning that cannot turn diagnostics into control flow."""
    try:
        logging.warning(message, *args, exc_info=exc_info)
    except Exception:
        return


def _metrics_buffer_disable(message: str) -> None:
    """Detach all metrics state before best-effort close and warning."""
    global _metrics_atexit_registered, _metrics_buffer, _metrics_file_handle  # noqa: PLW0603
    failure_exc_info = sys.exc_info()
    failed_handle = _metrics_file_handle
    _metrics_file_handle = None
    _metrics_buffer = []
    _metrics_atexit_registered = False

    close_error = None
    if failed_handle is not None:
        try:
            failed_handle.close()
        except Exception as exc:
            close_error = exc

    _safe_log_warning(message, exc_info=failure_exc_info)
    if close_error is not None:
        _safe_log_warning("metrics.jsonl close after failure also failed: %s", close_error)


def _metrics_buffer_flush() -> None:
    """Flush buffered metrics, disabling the writer after any I/O failure."""
    global _metrics_buffer
    if _metrics_file_handle is None or not _metrics_buffer:
        return
    try:
        _metrics_file_handle.write("".join(_metrics_buffer))
        _metrics_file_handle.flush()
    except Exception:
        _metrics_buffer_disable("metrics.jsonl flush failed; disabling metrics output")
        return
    _metrics_buffer = []


def _metrics_buffer_write_boundary(record: dict) -> None:
    """Append and flush one boundary record without propagating metrics errors."""
    try:
        _metrics_buffer_append(record)
    except Exception:
        _metrics_buffer_disable("metrics.jsonl record serialization failed; disabling metrics output")
        return
    _metrics_buffer_flush()


def _metrics_buffer_close() -> None:
    """Flush, detach, and close metrics output without propagating diagnostics."""
    global _metrics_atexit_registered, _metrics_file_handle  # noqa: PLW0603
    _metrics_buffer_flush()
    file_handle = _metrics_file_handle
    _metrics_file_handle = None
    _metrics_atexit_registered = False
    if file_handle is None:
        return
    try:
        file_handle.close()
    except Exception as exc:
        _safe_log_warning("metrics.jsonl close failed: %s", exc)


def _loss_finite_check_enabled() -> bool:
    """Whether the per-step loss-level finite consensus check runs.

    Defaults to enabled so unset environments keep their current behavior. Set
    ``OPENPI_LOSS_FINITE_CHECK=0`` to skip it.

    Skipping is safe for bf16 / fp32 because a non-finite loss still produces
    non-finite gradients, and DeepSpeed runs its own gradient-level overflow
    check (``check_grad_overflow`` defaults to True) which zeroes the gradients
    and returns without stepping the optimizer. The trainer additionally aborts
    after ``OPENPI_MAX_CONSECUTIVE_SKIPPED_UPDATES`` skipped updates, so the
    runaway-divergence guard is preserved.

    The benefit is removing a per-step host/device synchronization: the loss is
    the terminal node of the forward graph, so reading it drains the CUDA queue
    between forward and backward and destroys CPU run-ahead.
    """
    return os.environ.get("OPENPI_LOSS_FINITE_CHECK", "1").strip().lower() not in {"0", "false", "no"}


def _gather_finite_consensus(
    accelerator: Accelerator, *scalars: torch.Tensor
) -> bool:
    """Cheap cross-rank finiteness consensus for one or more scalar tensors.

    Fast path used on every training step.  Keeps the scalar checks on-device
    until the single distributed verdict is read, avoiding the per-loss
    ``.item()`` calls that previously forced extra host/device synchronizations
    on every rank.

    Detailed scalar values / min / max / mean / std should be gathered only on
    slower diagnostic paths such as log-interval boundaries or when a non-finite
    value has already been detected.
    """
    import torch.distributed as dist

    local_bad = torch.zeros(1, device=accelerator.device, dtype=torch.float32)

    for s in scalars:
        val = s.detach().float().reshape(1)
        local_bad += (~torch.isfinite(val)).float()

    # Single all-reduce: sum of "bad scalar" flags across all ranks.
    # If the sum is > 0, at least one rank has at least one non-finite scalar.
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_bad, op=dist.ReduceOp.SUM)

    return bool(local_bad.item() == 0.0)


def _gather_scalar_stats(accelerator: Accelerator, scalar: torch.Tensor) -> dict[str, float]:
    """Gather cross-rank scalar statistics using torch.distributed all_reduce.

    Uses raw ``torch.distributed.all_reduce`` (SUM / MIN / MAX) instead of
    ``accelerator.gather`` for reliability under DeepSpeed ZeRO-2.  Computes
    mean / std from the reduced sum / sum-of-squares / count so that every
    rank has the same global values without depending on gather shape.
    """
    import torch.distributed as dist

    value = scalar.detach().float().reshape(1).to(accelerator.device)
    finite = torch.isfinite(value)

    local_finite_count = finite.sum().float()
    local_total = torch.tensor(float(value.numel()), device=accelerator.device, dtype=torch.float32)

    # In-place all-reduce for count and total
    dist.all_reduce(local_finite_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_total, op=dist.ReduceOp.SUM)

    finite_count = float(local_finite_count.item())
    total_count = float(local_total.item())

    stats = {
        "finite_count": finite_count,
        "total_count": total_count,
        "all_finite": float(finite_count == total_count),
        "bad_rank": -1.0,
        "min": float("nan"),
        "max": float("nan"),
        "mean": float("nan"),
        "std": float("nan"),
    }

    if finite_count > 0:
        # Use finite values only; replace non-finite with extreme sentinels for min/max
        finite_val = torch.where(
            finite,
            value,
            torch.tensor(float("inf"), device=accelerator.device, dtype=torch.float32),
        )
        # min: all_reduce MIN with inf sentinel for non-finite (inf is identity for min)
        min_tensor = finite_val.clone()
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)

        finite_val_max = torch.where(
            finite,
            value,
            torch.tensor(float("-inf"), device=accelerator.device, dtype=torch.float32),
        )
        # max: all_reduce MAX with -inf sentinel for non-finite
        max_tensor = finite_val_max.clone()
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)

        # For mean and std: sum and sum-of-squares of finite values only
        local_sum = torch.where(finite, value, torch.zeros_like(value)).sum().float()
        local_sq_sum = torch.where(finite, value ** 2, torch.zeros_like(value)).sum().float()
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)

        mean_val = float(local_sum.item()) / finite_count
        sq_mean = float(local_sq_sum.item()) / finite_count
        var_val = max(0.0, sq_mean - mean_val * mean_val)
        std_val = var_val ** 0.5

        stats["min"] = float(min_tensor.item())
        stats["max"] = float(max_tensor.item())
        stats["mean"] = mean_val
        stats["std"] = std_val

    if finite_count < total_count:
        # We don't have per-rank bad rank info with all_reduce-only,
        # but we can report that there was a non-finite rank.
        stats["bad_rank"] = -2.0  # sentinel: "some rank(s) non-finite, rank unknown"

    return stats


def _collect_grad_debug_stats(parameters, accelerator: Accelerator) -> dict[str, float]:
    device = accelerator.device
    global_norm_sq = torch.zeros(1, device=device, dtype=torch.float64)
    max_abs_grad = torch.zeros(1, device=device, dtype=torch.float32)
    nonfinite_count = torch.zeros(1, device=device, dtype=torch.float32)
    total_grad_elements = torch.zeros(1, device=device, dtype=torch.float32)
    grad_tensors = torch.zeros(1, device=device, dtype=torch.float32)

    for param in parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        grad_tensors += 1
        grad_data = grad.detach()
        total_grad_elements += float(grad_data.numel())
        finite_mask = torch.isfinite(grad_data)
        nonfinite_count += (~finite_mask).sum().to(torch.float32)
        if finite_mask.any():
            finite_grad = grad_data[finite_mask].float()
            global_norm_sq += finite_grad.pow(2).sum(dtype=torch.float64)
            max_abs_grad = torch.maximum(max_abs_grad, finite_grad.abs().max().to(torch.float32))

    global_norm_sq = cast(torch.Tensor, accelerator.reduce(global_norm_sq, reduction="sum"))
    max_abs_grad = cast(torch.Tensor, accelerator.reduce(max_abs_grad, reduction="max"))
    nonfinite_count = cast(torch.Tensor, accelerator.reduce(nonfinite_count, reduction="sum"))
    total_grad_elements = cast(torch.Tensor, accelerator.reduce(total_grad_elements, reduction="sum"))
    grad_tensors = cast(torch.Tensor, accelerator.reduce(grad_tensors, reduction="sum"))

    total_grad_elements_value = float(total_grad_elements.item())
    nonfinite_count_value = float(nonfinite_count.item())
    finite_ratio = 1.0
    if total_grad_elements_value > 0:
        finite_ratio = max(0.0, 1.0 - nonfinite_count_value / total_grad_elements_value)

    return {
        "global_norm": float(torch.sqrt(global_norm_sq).item()),
        "max_abs_grad": float(max_abs_grad.item()),
        "nonfinite_count": nonfinite_count_value,
        "total_grad_elements": total_grad_elements_value,
        "finite_ratio": finite_ratio,
        "grad_tensors": float(grad_tensors.item()),
    }


def _compute_param_group_grad_norm(
    parameters: list[torch.nn.Parameter],
    accelerator: Accelerator,
) -> tuple[float, bool]:
    """Compute the total L2 gradient norm for a list of parameters.

    Returns a tuple ``(norm, available)`` where ``available`` is False when
    no usable gradient data was found across all ranks (e.g. all param grads
    are None or empty shards under ZeRO-2 before ``deepspeed.engine.step()``
    has consolidated them).  When unavailable, ``norm`` is ``float('nan')``
    instead of a misleading 0.0.

    Under DeepSpeed ZeRO-2, gradients are partitioned per-rank so we use
    ``deepspeed.utils.safe_get_local_grad`` to obtain each param's local
    gradient shard, then all-reduce sum-of-squares across ranks to recover
    the global norm.
    """
    device = accelerator.device
    total_sq = torch.zeros(1, device=device, dtype=torch.float64)
    any_grad_seen = False

    # Use DeepSpeed's safe_get_local_grad when available; otherwise fall back
    # to param.grad directly (DDP / single-GPU / etc.).
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    if is_deepspeed:
        try:
            from deepspeed.utils import safe_get_local_grad
        except ImportError:
            safe_get_local_grad = None
    else:
        safe_get_local_grad = None

    for param in parameters:
        if is_deepspeed and safe_get_local_grad is not None:
            try:
                # safe_get_local_grad is ZeRO-3 only (asserts param has ds_id).
                # Under ZeRO-2, params don't have ds_id — fall back to param.grad.
                if not hasattr(param, "ds_id"):
                    grad = getattr(param, "grad", None)
                else:
                    grad = safe_get_local_grad(param)
            except (AssertionError, RuntimeError, AttributeError):
                # Any DeepSpeed-specific failure → fall back to direct grad access.
                grad = getattr(param, "grad", None)
        else:
            grad = getattr(param, "grad", None)
        if grad is None:
            continue
        grad_data = grad.detach()
        if grad_data.numel() == 0:
            continue
        any_grad_seen = True
        finite_mask = torch.isfinite(grad_data)
        if finite_mask.any():
            total_sq += grad_data[finite_mask].float().pow(2).sum(dtype=torch.float64)

    # Reduce across ranks (sum of squares) to get global norm when possible.
    # NOTE: Under ZeRO-2 each rank holds a different shard of each param's
    # grad tensor, so summing the per-rank norms squared yields the squared
    # global norm.  Under DDP every rank has the full grad, so we take the
    # mean (which equals any single rank's value).
    if is_deepspeed:
        total_sq = cast(torch.Tensor, accelerator.reduce(total_sq, reduction="sum"))
    else:
        total_sq = cast(torch.Tensor, accelerator.reduce(total_sq, reduction="mean"))

    norm_value = float(torch.sqrt(total_sq).item())
    # If no rank had any gradient data, mark as unavailable (NaN instead of 0.0).
    if not any_grad_seen and norm_value == 0.0:
        return float("nan"), False
    return norm_value, True


def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _get_git_info() -> dict[str, str]:
    """Return git commit hash and branch name, best-effort.

    Returns empty strings if git is unavailable or not in a repo.
    """
    info: dict[str, str] = {"commit": "", "branch": ""}
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except Exception:
        pass
    return info


def _build_data_fingerprint(config, data_config) -> dict:
    """Build a data fingerprint dict from TrainConfig + DataConfig.

    Includes a stable SHA256 of canonical serialized data selection params
    for exact reproducibility of the training data subset.
    """
    try:
        repo_id = getattr(data_config, "repo_id", None) or ""
        episodes_index = getattr(data_config, "episodes_index", None)
        tasks = getattr(data_config, "tasks", None)
        fine_grained_level = getattr(data_config, "fine_grained_level", 0)
        modalities = getattr(data_config, "modalities", None)
        subtask_source = getattr(data_config, "subtask_source", None)
        prompt_from_task = getattr(data_config, "prompt_from_task", False)

        fingerprint = {
            "repo_id": str(repo_id) if repo_id else "",
            "seed": getattr(config, "seed", 42),
            "batch_size": getattr(config, "batch_size", 0),
            "num_train_steps": getattr(config, "num_train_steps", 0),
            "fine_grained_level": fine_grained_level,
        }
        if episodes_index is not None:
            fingerprint["num_episodes"] = len(episodes_index)
            fingerprint["episodes_index_first_last"] = [
                int(episodes_index[0]),
                int(episodes_index[-1]),
            ] if len(episodes_index) > 0 else []
        if tasks is not None:
            fingerprint["tasks"] = list(tasks)
        if modalities is not None:
            fingerprint["modalities"] = list(modalities)
        if subtask_source is not None:
            fingerprint["subtask_source"] = subtask_source
        fingerprint["prompt_from_task"] = prompt_from_task

        # Build SHA256 of canonical sorted data selection
        canonical = {
            "repo_id": str(repo_id) if repo_id else "",
            "seed": getattr(config, "seed", 42),
            "fine_grained_level": fine_grained_level,
            "subtask_source": subtask_source if subtask_source is not None else "",
            "prompt_from_task": prompt_from_task,
            "modalities": sorted(modalities) if modalities is not None else [],
            "tasks": sorted(tasks) if tasks is not None else [],
            "episodes_index": [int(i) for i in episodes_index] if episodes_index is not None else [],
        }
        canonical_json = json.dumps(canonical, sort_keys=True)
        fingerprint["sha256"] = hashlib.sha256(canonical_json.encode()).hexdigest()
        fingerprint["sha256_canonical_preview"] = canonical_json[:200] + "..."

        return fingerprint
    except Exception:
        return {"error": "failed_to_build_fingerprint"}


def _compute_data_manifest(
    *,
    config,
    data_config,
    train_loader,
    val_loader=None,
    val_data_config=None,
    steps_per_epoch: int,
    world_size: int,
    grad_accum_steps: int,
    seed: int,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    num_probe_batches: int = 0,
) -> dict:
    """Compute a comprehensive data manifest describing training and validation data.

    If ``num_probe_batches <= 0``, probing is skipped entirely (metadata-only manifest);
    this avoids spawning chunk-streaming video readers and speeds up model init.
    Otherwise, samples ``num_probe_batches`` batches from each loader to inspect
    tensor shapes, image keys, padding stats, episode diversity, and subtask token presence.
    Results are honest about streaming dataset semantics (no false "1 epoch" claims).

    Args:
        config: TrainConfig instance.
        data_config: DataConfig for training data.
        train_loader: Training data loader (iterable of (observation, actions)).
        val_loader: Optional validation data loader.
        val_data_config: Optional DataConfig for validation data.
        steps_per_epoch: Computed steps per epoch (optimizer-step granularity).
        world_size: Number of distributed processes.
        grad_accum_steps: Gradient accumulation steps.
        seed: Base random seed used by the data loader.
        train_shuffle: Whether the training loader uses shuffle.
        val_shuffle: Whether the validation loader uses shuffle.
        num_probe_batches: Number of batches to sample for stats inspection.
            If <= 0, probing is skipped (metadata-only manifest with probe_skipped=True).

    Returns:
        dict with data_manifest fields (see docstring in code for schema).
    """
    manifest: dict = {}

    # -- Metadata --
    manifest["generated_at"] = time.time()
    manifest["generated_at_iso"] = datetime.datetime.now().isoformat()

    # -- Basic shape / structure from config and data_config --
    try:
        manifest["action_horizon"] = int(config.model.action_horizon)
    except Exception:
        manifest["action_horizon"] = None

    # Episode counts from data_config (when available)
    train_episodes_index = getattr(data_config, "episodes_index", None)
    manifest["n_train_episodes"] = len(train_episodes_index) if train_episodes_index is not None else None
    manifest["train_tasks"] = list(getattr(data_config, "tasks", []) or [])
    manifest["train_repo_id"] = getattr(data_config, "repo_id", None)
    manifest["train_modalities"] = list(getattr(data_config, "modalities", []) or [])
    manifest["train_fine_grained_level"] = getattr(data_config, "fine_grained_level", 0)
    manifest["train_subtask_source"] = getattr(data_config, "subtask_source", None)
    manifest["train_prompt_from_task"] = getattr(data_config, "prompt_from_task", False)

    # -- Validation data info --
    manifest["has_val_data"] = val_loader is not None
    if val_loader is not None and val_data_config is not None:
        val_episodes_index = getattr(val_data_config, "episodes_index", None)
        manifest["n_val_episodes"] = len(val_episodes_index) if val_episodes_index is not None else None
        manifest["val_tasks"] = list(getattr(val_data_config, "tasks", []) or [])
        manifest["val_repo_id"] = getattr(val_data_config, "repo_id", None)
    else:
        manifest["n_val_episodes"] = None
        manifest["val_tasks"] = []
        manifest["val_repo_id"] = None

    # -- Loader length / steps_per_epoch --
    try:
        train_len_micro = len(train_loader)
    except Exception:
        train_len_micro = None

    manifest["n_train_frames_per_rank_micro"] = train_len_micro  # DEPRECATED: use n_train_microbatches_per_rank
    manifest["n_train_microbatches_per_rank"] = train_len_micro  # micro-batches per rank per "epoch"
    if train_len_micro is not None and world_size > 0:
        manifest["n_train_frames_total_estimate"] = train_len_micro * world_size  # DEPRECATED: use n_train_microbatches_total_estimate
        manifest["n_train_microbatches_total_estimate"] = train_len_micro * world_size
    else:
        manifest["n_train_frames_total_estimate"] = None
        manifest["n_train_microbatches_total_estimate"] = None

    manifest["steps_per_epoch"] = int(steps_per_epoch)
    manifest["world_size"] = int(world_size)
    manifest["grad_accum_steps"] = int(grad_accum_steps)

    # val loader length
    if val_loader is not None:
        try:
            val_len_micro = len(val_loader)
            manifest["n_val_frames_per_rank_micro"] = val_len_micro  # DEPRECATED: use n_val_microbatches_per_rank
            manifest["n_val_microbatches_per_rank"] = val_len_micro
            manifest["n_val_frames_total_estimate"] = val_len_micro * world_size  # DEPRECATED: use n_val_microbatches_total_estimate
            manifest["n_val_microbatches_total_estimate"] = val_len_micro * world_size
        except Exception:
            manifest["n_val_frames_per_rank_micro"] = None
            manifest["n_val_microbatches_per_rank"] = None
            manifest["n_val_frames_total_estimate"] = None
            manifest["n_val_microbatches_total_estimate"] = None
    else:
        manifest["n_val_frames_per_rank_micro"] = None
        manifest["n_val_microbatches_per_rank"] = None
        manifest["n_val_frames_total_estimate"] = None
        manifest["n_val_microbatches_total_estimate"] = None

    validation_fields = _validation_manifest_fields(val_loader, world_size=world_size)
    if validation_fields is not None:
        manifest["validation_protocol"] = validation_fields

    # -- FPS --
    manifest["fps"] = 30  # B1K datasets use 30 FPS; lerobot meta has fps too
    # Try to get fps from data_config or dataset metadata
    try:
        if hasattr(data_config, "fps") and data_config.fps is not None:
            manifest["fps"] = int(data_config.fps)
    except Exception:
        pass

    # -- Norm info --
    norm_stats = getattr(data_config, "norm_stats", None)
    if norm_stats is not None:
        manifest["norm_info"] = {
            "available": True,
            "num_keys": len(norm_stats),
            "keys": sorted(list(norm_stats.keys()))[:20],  # first 20 keys for preview
        }
        # Include a summary of first norm stat's shape
        if norm_stats:
            first_key = sorted(norm_stats.keys())[0]
            ns = norm_stats[first_key]
            try:
                manifest["norm_info"]["sample_key"] = first_key
                manifest["norm_info"]["sample_loc_shape"] = list(np.asarray(ns.loc).shape) if hasattr(ns, "loc") else None
            except Exception:
                pass
    else:
        manifest["norm_info"] = {"available": False}

    # -- Data fingerprint (SHA256 of data selection params) --
    manifest["data_sha"] = _build_data_fingerprint(config, data_config)

    # -- Probe control flag --
    manifest["probe_skipped"] = num_probe_batches <= 0

    # -- Streaming dataset validation: probe actual batches --
    if num_probe_batches > 0:
        probe_stats = _probe_data_loader(
            train_loader,
            num_batches=num_probe_batches,
            label="train",
        )
        manifest["train_probe"] = probe_stats

        # Action dim and state dim from probe
        manifest["action_dim"] = probe_stats.get("action_dim")
        manifest["state_dim"] = probe_stats.get("state_dim")
        manifest["image_keys"] = probe_stats.get("image_keys", [])
        manifest["has_subtask_tokens"] = probe_stats.get("has_subtask_tokens", False)
        manifest["subtask_max_len"] = probe_stats.get("subtask_max_len")
        manifest["padding_stats"] = probe_stats.get("padding_stats", {})

        # Sample count estimates (microbatches * batch_size per rank * world_size)
        train_bs = probe_stats.get("batch_size")
        if train_len_micro is not None and train_bs is not None and world_size > 0:
            manifest["n_train_samples_per_rank_estimate"] = train_len_micro * train_bs
            manifest["n_train_samples_total_estimate"] = train_len_micro * train_bs * world_size
            # n_train_frames = total samples across all ranks (spec naming)
            manifest["n_train_frames"] = train_len_micro * train_bs * world_size
        else:
            manifest["n_train_samples_per_rank_estimate"] = None
            manifest["n_train_samples_total_estimate"] = None
            manifest["n_train_frames"] = None
    else:
        # Metadata-only mode: skip data loader probing entirely.
        # This avoids spawning chunk-streaming video readers, which is the
        # dominant cost of manifest computation for B1K-scale datasets.
        manifest["train_probe"] = {
            "num_batches_requested": 0,
            "num_batches_sampled": 0,
            "action_dim": None,
            "state_dim": None,
            "image_keys": [],
            "has_subtask_tokens": False,
            "subtask_max_len": None,
            "padding_stats": {
                "action_is_pad_available": False,
                "pad_ratio_mean": None,
                "pad_ratio_min": None,
                "pad_ratio_max": None,
            },
            "episode_ids_sample": [],
            "task_ids_sample": [],
            "batch_size": None,
            "error": None,
            "probe_skipped": True,
        }
        manifest["action_dim"] = None
        manifest["state_dim"] = None
        manifest["image_keys"] = []
        manifest["has_subtask_tokens"] = False
        manifest["subtask_max_len"] = None
        manifest["padding_stats"] = {
            "action_is_pad_available": False,
            "pad_ratio_mean": None,
            "pad_ratio_min": None,
            "pad_ratio_max": None,
        }
        manifest["n_train_samples_per_rank_estimate"] = None
        manifest["n_train_samples_total_estimate"] = None
        manifest["n_train_frames"] = None

    # Val probe
    if val_loader is not None and num_probe_batches > 0:
        val_probe_stats = _probe_data_loader(
            val_loader,
            num_batches=min(num_probe_batches, manifest.get("n_val_microbatches_per_rank") or num_probe_batches),
            label="val",
        )
        manifest["val_probe"] = val_probe_stats

        # Val sample count estimates
        val_bs = val_probe_stats.get("batch_size")
        val_len = manifest.get("n_val_microbatches_per_rank")
        if val_len is not None and val_bs is not None and world_size > 0:
            manifest["n_val_samples_per_rank_estimate"] = val_len * val_bs
            manifest["n_val_samples_total_estimate"] = val_len * val_bs * world_size
            manifest["n_val_frames"] = val_len * val_bs * world_size
        else:
            manifest["n_val_samples_per_rank_estimate"] = None
            manifest["n_val_samples_total_estimate"] = None
            manifest["n_val_frames"] = None
    elif val_loader is not None and num_probe_batches <= 0:
        # Metadata-only mode: no val probing
        manifest["val_probe"] = {
            "num_batches_requested": 0,
            "num_batches_sampled": 0,
            "action_dim": None,
            "state_dim": None,
            "image_keys": [],
            "has_subtask_tokens": False,
            "subtask_max_len": None,
            "padding_stats": {
                "action_is_pad_available": False,
                "pad_ratio_mean": None,
                "pad_ratio_min": None,
                "pad_ratio_max": None,
            },
            "episode_ids_sample": [],
            "task_ids_sample": [],
            "batch_size": None,
            "error": None,
            "probe_skipped": True,
        }
        manifest["n_val_samples_per_rank_estimate"] = None
        manifest["n_val_samples_total_estimate"] = None
        manifest["n_val_frames"] = None
    else:
        manifest["val_probe"] = None
        manifest["n_val_samples_per_rank_estimate"] = None
        manifest["n_val_samples_total_estimate"] = None
        manifest["n_val_frames"] = None

    # -- Streaming dataset semantics (honest assessment) --
    # The BehaviorLeRobotDataset is a streaming/chunking dataset that may not
    # guarantee full coverage of all episodes in one "epoch" of __len__ batches.
    is_streaming = True  # B1K datasets use chunk streaming
    manifest["streaming_dataset"] = {
        "is_streaming": is_streaming,
        "shuffle_train": train_shuffle,
        "shuffle_val": val_shuffle,
        "train_seed": int(seed),
        "val_seed": int(seed),
        "epoch_semantics_note": (
            "Streaming/chunking dataset: len(loader) micro-batches per rank does NOT "
            "guarantee full coverage of all configured episodes in one pass. "
            "Episode coverage depends on chunk size, keyframe stride, and shuffling. "
            "Use train_probe.episode_ids_sample to verify which episodes appear."
        ),
        "action_horizon_boundary_note": (
            "Action horizon windows near episode boundaries may include padded frames. "
            "If action_is_pad is available, padding_stats reports the fraction of "
            "padded action positions across sampled batches."
        ),
        "episode_id_probe_note": (
            "episode_ids_sample may be empty because the Observation.from_dict() wrapper "
            "strips raw dataset fields like episode_index and task_index. The streaming "
            "dataset still respects the episodes_index filter in data_config; use "
            "n_train_episodes / n_val_episodes from data_config for the configured count."
        ),
    }

    # Validation loader determinism note
    manifest["val_determinism"] = {
        "shuffle_enabled": val_shuffle,
        "fixed_seed": True,
        "seed": int(seed),
        "note": (
            "Validation loader uses shuffle=False on the outer DataLoader with a fixed seed. "
            "Note: the inner BehaviorLeRobotDataset may have chunk-level shuffling enabled; "
            "the outer DataLoader's sequential index access should still produce deterministic "
            "iteration order for the same seed and dataset configuration."
        ),
    }

    return manifest


def _probe_data_loader(loader, num_batches: int, label: str = "train") -> dict:
    """Probe a data loader by sampling ``num_batches`` batches and collecting stats.

    Returns a dict with:
    - action_dim, state_dim, image_keys
    - has_subtask_tokens, subtask_max_len
    - padding_stats (action_is_pad ratio if available)
    - episode_ids_sample (list of unique episode indices seen)
    - task_ids_sample (list of unique task indices seen)
    - timestamp_range (min/max of timestamps if available)
    - batch_size (observed local batch size)
    - num_batches_sampled (how many batches were actually iterated)
    - error (if probing failed)
    """
    stats: dict = {
        "num_batches_requested": num_batches,
        "num_batches_sampled": 0,
        "action_dim": None,
        "state_dim": None,
        "image_keys": [],
        "has_subtask_tokens": False,
        "subtask_max_len": None,
        "padding_stats": {
            "action_is_pad_available": False,
            "pad_ratio_mean": None,
            "pad_ratio_min": None,
            "pad_ratio_max": None,
        },
        "episode_ids_sample": [],
        "task_ids_sample": [],
        "batch_size": None,
        "error": None,
    }

    pad_ratios = []
    episode_ids_set: set = set()
    task_ids_set: set = set()

    try:
        iterator = iter(loader)
        for i in range(num_batches):
            try:
                observation, actions = next(iterator)
            except StopIteration:
                break

            stats["num_batches_sampled"] = i + 1

            # Action shape: [B, action_horizon, action_dim]
            if hasattr(actions, "shape"):
                stats["batch_size"] = int(actions.shape[0])
                if len(actions.shape) >= 3:
                    stats["action_dim"] = int(actions.shape[-1])

            # State shape: [B, state_dim]
            if hasattr(observation, "state") and observation.state is not None:
                state = observation.state
                if hasattr(state, "shape") and len(state.shape) >= 2:
                    stats["state_dim"] = int(state.shape[-1])

            # Image keys
            if hasattr(observation, "images") and observation.images is not None:
                if isinstance(observation.images, dict):
                    stats["image_keys"] = sorted(list(observation.images.keys()))

            # Subtask tokens
            if hasattr(observation, "subtask_tokens") and observation.subtask_tokens is not None:
                stats["has_subtask_tokens"] = True
                st = observation.subtask_tokens
                if hasattr(st, "shape") and len(st.shape) >= 2:
                    stats["subtask_max_len"] = int(st.shape[-1])

            # Padding stats (action_is_pad on observation)
            action_is_pad = getattr(observation, "action_is_pad", None)
            if action_is_pad is not None and hasattr(action_is_pad, "float"):
                stats["padding_stats"]["action_is_pad_available"] = True
                pad_ratio = float(action_is_pad.float().mean().item())
                pad_ratios.append(pad_ratio)

            # Episode / task IDs (if available in raw dict)
            # Try to get from the batch dict before Observation wrapping
            # Some datasets include episode_index and task_index in the item
            for attr_name in ("episode_index", "episode_id"):
                val = getattr(observation, attr_name, None)
                if val is not None and hasattr(val, "tolist"):
                    episode_ids_set.update(int(v) for v in val.tolist())
                    break

            for attr_name in ("task_index", "task_id"):
                val = getattr(observation, attr_name, None)
                if val is not None and hasattr(val, "tolist"):
                    task_ids_set.update(int(v) for v in val.tolist())
                    break

    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        logging.warning("Data manifest probe failed for %s loader: %s", label, stats["error"])

    # Aggregate padding stats
    if pad_ratios:
        stats["padding_stats"]["pad_ratio_mean"] = float(np.mean(pad_ratios))
        stats["padding_stats"]["pad_ratio_min"] = float(np.min(pad_ratios))
        stats["padding_stats"]["pad_ratio_max"] = float(np.max(pad_ratios))

    # Episode / task ID samples (sorted, limited to first 20 for preview)
    stats["episode_ids_sample"] = sorted(list(episode_ids_set))[:20]
    stats["n_unique_episodes_observed"] = len(episode_ids_set)
    stats["task_ids_sample"] = sorted(list(task_ids_set))[:20]
    stats["n_unique_tasks_observed"] = len(task_ids_set)

    return stats


def _build_checkpoint_manifest(
    *,
    config,
    data_config,
    global_step: int,
    accelerator: Accelerator,
    precision: str,
    data_manifest: dict | None = None,
    checkpoint_kind: str | None = None,
    checkpoint_epoch: int | None = None,
    sample_progress: _SampleProgress | None = None,
) -> dict:
    """Build the checkpoint manifest.json dict with metadata."""
    git_info = _get_git_info()

    try:
        gpu_type = ""
        if torch.cuda.is_available():
            gpu_type = torch.cuda.get_device_name(0)
    except Exception:
        gpu_type = ""

    manifest = {
        "git": {
            "commit": git_info["commit"],
            "branch": git_info["branch"],
        },
        "config": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else str(config),
        "data_fingerprint": _build_data_fingerprint(config, data_config),
        "run_metadata": {
            "global_step": global_step,
            "timestamp": time.time(),
            "timestamp_iso": datetime.datetime.now().isoformat(),
            "hostname": platform.node(),
        },
        "hardware": {
            "num_gpus": accelerator.num_processes,
            "gpu_type": gpu_type,
            "precision": precision,
            "strategy": str(accelerator.distributed_type),
        },
    }
    if sample_progress is not None:
        manifest["run_metadata"]["sample_progress"] = sample_progress.checkpoint_payload()
    if checkpoint_kind is not None:
        manifest["run_metadata"]["checkpoint_kind"] = checkpoint_kind
    if checkpoint_epoch is not None:
        manifest["run_metadata"]["epoch"] = checkpoint_epoch
    if data_manifest is not None:
        manifest["data_manifest"] = data_manifest
    return manifest


def _atomic_write_checkpoint_dir(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)


def _move_observation_to_device(observation, device: torch.device):
    """Move every tensor in an Observation, including Variant A targets."""

    def _maybe_to(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        return x.to(device, non_blocking=True)

    # Observation is a flax.struct.dataclass, not a dm-tree container, so its
    # optional fields must be routed explicitly. Keep this list centralized for
    # both training and validation to prevent newly added fields from drifting.
    return observation.replace(
        images={k: v.to(device, non_blocking=True) for k, v in observation.images.items()},
        image_masks={k: v.to(device, non_blocking=True) for k, v in observation.image_masks.items()},
        state=observation.state.to(device, non_blocking=True),
        tokenized_prompt=_maybe_to(observation.tokenized_prompt),
        tokenized_prompt_mask=_maybe_to(observation.tokenized_prompt_mask),
        token_ar_mask=_maybe_to(observation.token_ar_mask),
        token_loss_mask=_maybe_to(observation.token_loss_mask),
        subtask_tokens=_maybe_to(observation.subtask_tokens),
        subtask_mask=_maybe_to(observation.subtask_mask),
        subtask_loss_mask=_maybe_to(observation.subtask_loss_mask),
        subtask_ar_mask=_maybe_to(observation.subtask_ar_mask),
        action_tokens=_maybe_to(observation.action_tokens),
        action_token_mask=_maybe_to(observation.action_token_mask),
        action_token_loss_mask=_maybe_to(observation.action_token_loss_mask),
        action_token_ar_mask=_maybe_to(observation.action_token_ar_mask),
        pcd_xyz=_maybe_to(observation.pcd_xyz),
    )


def _validation_metric_totals(
    batch_metrics: list[dict[str, float]],
    *,
    batch_task_ids: list[list[int]],
    batch_unique_counts: list[int],
    per_task_metric: str,
) -> tuple[dict[str, list[float | int]], dict[int, list[float | int]]]:
    """Convert scalar batch means into sample-weighted unique-anchor totals."""
    if not (
        len(batch_metrics) == len(batch_task_ids) == len(batch_unique_counts)
    ):
        raise ValueError("validation batch metric metadata lengths disagree")
    metric_totals: dict[str, list[float | int]] = {}
    task_totals: dict[int, list[float | int]] = {}
    for metrics, task_ids, unique_count in zip(
        batch_metrics, batch_task_ids, batch_unique_counts, strict=True
    ):
        unique_count = int(unique_count)
        if unique_count <= 0:
            continue
        for key, value in metrics.items():
            if not np.isfinite(value):
                continue
            total = metric_totals.setdefault(key, [0.0, 0])
            total[0] = float(total[0]) + float(value) * unique_count
            total[1] = int(total[1]) + unique_count
        unique_tasks = set(int(task_id) for task_id in task_ids[:unique_count])
        if (
            len(unique_tasks) == 1
            and per_task_metric in metrics
            and np.isfinite(metrics[per_task_metric])
        ):
            task_id = unique_tasks.pop()
            task_total = task_totals.setdefault(task_id, [0.0, 0])
            task_total[0] = float(task_total[0]) + float(metrics[per_task_metric]) * unique_count
            task_total[1] = int(task_total[1]) + unique_count
    return metric_totals, task_totals


def _validation_manifest_fields(val_loader, *, world_size: int) -> dict | None:
    coverage = getattr(val_loader, "coverage", None)
    if not coverage:
        return None
    required = {
        "raw_episodes": "n_raw_episodes",
        "unique_anchors": "n_unique_anchors",
        "duplicate_anchors": "n_duplicated",
        "executed_anchors": "n_padded",
        "batches_per_rank": "n_batches_per_rank",
        "global_batch": "val_global_batch",
    }
    if not all(source in coverage for source in required.values()):
        return None
    fields = {target: int(coverage[source]) for target, source in required.items()}
    if fields["global_batch"] % int(world_size) != 0:
        raise ValueError(f"validation global batch is not divisible by world size: {fields}")
    return fields


def run_validation(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    val_loader,
    config: _config.TrainConfig,
    global_step: int,
    steps_per_epoch: int,
    is_pi05_ki_joint: bool,
    use_vlm2: bool,
    use_autocast: bool,
    autocast_dtype: torch.dtype,
    metrics_file,
    sample_progress: _SampleProgress | None = None,
    slow_metrics: bool = False,
    val_label: str = "val",
) -> dict[str, float]:
    """Run validation on val_loader and return aggregated metrics.

    Runs ``val_num_batches`` batches from the val loader in eval mode with
    ``torch.no_grad()``.  Metrics are averaged across batches and gathered
    across DDP ranks.

    Args:
        accelerator: Accelerator instance
        model: the model (already wrapped by accelerator)
        val_loader: validation data loader
        config: training config
        global_step: current training step (for logging)
        steps_per_epoch: steps per epoch (for W&B logging)
        is_pi05_ki_joint: whether model is PI05KIJointQueryPytorch
        use_vlm2: whether model is VLM2
        use_autocast: whether to use autocast
        autocast_dtype: autocast dtype
        metrics_file: rank-0 metrics.jsonl file handle (may be None)
        slow_metrics: if True, compute slow-path metrics (flow_l1 via Euler
            integration).  For epoch-end validation only.
        val_label: label for this validation run (e.g. "val", "val_epoch_end")
        autocast_dtype: autocast dtype
        metrics_file: rank-0 metrics.jsonl file handle (may be None)

    Returns:
        dict of aggregated validation metrics (scalar floats, rank 0 values)
    """
    formal_h20 = getattr(config, "name", None) in _H20_FORMAL_B1K_CONFIGS
    if val_loader is None:
        if formal_h20:
            raise ValueError("Formal H20 validation requires a non-empty validation loader")
        return {}

    is_main = accelerator.is_main_process
    configured_val_batches = int(getattr(config, "val_num_batches", 10))
    actual_val_batches = len(val_loader)
    val_num_batches = min(configured_val_batches, actual_val_batches)
    if formal_h20:
        protocol = _validation_manifest_fields(
            val_loader, world_size=int(accelerator.num_processes)
        )
        expected_protocol = {
            "raw_episodes": 1_000,
            "unique_anchors": 4_000,
            "duplicate_anchors": 96,
            "executed_anchors": 4_096,
            "batches_per_rank": 16,
            "global_batch": 256,
        }
        if protocol != expected_protocol:
            raise ValueError(
                f"Formal H20 validation protocol mismatch: expected={expected_protocol} actual={protocol}"
            )
        if actual_val_batches != 16 or configured_val_batches < actual_val_batches:
            raise ValueError(
                "Formal H20 validation requires exactly 16 loader batches/rank and "
                f"val_num_batches>=16; got loader={actual_val_batches} configured={configured_val_batches}"
            )

    if is_main:
        logging.info("Running validation at step %s (%s batches)...", global_step, val_num_batches)

    # Switch to eval mode
    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    unwrapped.eval()

    batch_metrics_list: list[dict[str, float]] = []
    per_task_metric = "flow_l1" if slow_metrics else "flow_mse"
    per_task_enabled = bool(getattr(config, "val_log_per_task", False)) and hasattr(
        val_loader, "batch_task_ids"
    )

    try:
        with torch.no_grad():
            for batch_idx, (observation, actions) in enumerate(val_loader):
                if batch_idx >= val_num_batches:
                    break

                # Move data to device
                if _model is not None and isinstance(observation, _model.Observation):
                    observation = _move_observation_to_device(observation, accelerator.device)
                else:
                    observation = tree.map_structure(
                        lambda x: x.to(accelerator.device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
                        observation,
                    )
                actions = actions.to(device=accelerator.device, dtype=torch.float32, non_blocking=True)

                if is_pi05_ki_joint:
                    with torch.autocast(
                        device_type=accelerator.device.type,
                        dtype=autocast_dtype,
                        enabled=use_autocast,
                    ):
                        eval_metrics = unwrapped.compute_eval_metrics(
                            observation, actions,
                            compute_flow_l1=slow_metrics,
                            num_denoise_steps=10,
                            flow_l1_seed=int(config.seed) + 9999,
                            deterministic_flow=bool(
                                getattr(config, "val_deterministic_flow", False)
                            ),
                        )
                elif use_vlm2:
                    # TODO: VLM2 validation path
                    continue
                else:
                    # Standard single-forward path
                    with torch.autocast(
                        device_type=accelerator.device.type,
                        dtype=autocast_dtype,
                        enabled=use_autocast,
                    ):
                        losses = model(observation, actions)
                        if isinstance(losses, dict):
                            eval_metrics = {
                                "total_loss": losses["loss"].detach(),
                                **{k: v.detach() for k, v in losses.items() if k != "loss" and isinstance(v, torch.Tensor) and v.numel() == 1},
                            }
                        else:
                            eval_metrics = {"total_loss": losses.mean().detach()}

                # Convert to float dict for this batch
                batch_metrics = {
                    k: float(v.detach().float().item())
                    for k, v in eval_metrics.items()
                    if isinstance(v, torch.Tensor) and v.numel() == 1
                }
                batch_metrics_list.append(batch_metrics)

    finally:
        # Restore training mode
        if was_training:
            unwrapped.train()

    if not batch_metrics_list:
        if formal_h20:
            raise ValueError("Formal H20 validation produced zero metric batches")
        if is_main:
            logging.warning("Validation produced no batches; skipping val logging.")
        return {}
    if formal_h20 and len(batch_metrics_list) != 16:
        raise ValueError(
            f"Formal H20 validation processed {len(batch_metrics_list)} batches; expected 16"
        )

    all_keys = sorted({key for metrics in batch_metrics_list for key in metrics})
    has_unique_metadata = hasattr(val_loader, "batch_unique_counts")
    task_metadata = list(getattr(val_loader, "batch_task_ids", []))
    unique_counts = list(getattr(val_loader, "batch_unique_counts", []))
    if len(task_metadata) < len(batch_metrics_list):
        task_metadata.extend([[]] * (len(batch_metrics_list) - len(task_metadata)))
    if len(unique_counts) < len(batch_metrics_list):
        if has_unique_metadata:
            raise ValueError("validation unique-count metadata is shorter than processed batches")
        # Legacy/non-formal loaders have no attribution metadata. Preserve their
        # historical equal-batch aggregation by weighting each batch once.
        unique_counts.extend([1] * (len(batch_metrics_list) - len(unique_counts)))
    local_metric_totals, local_task_totals = _validation_metric_totals(
        batch_metrics_list,
        batch_task_ids=task_metadata[: len(batch_metrics_list)],
        batch_unique_counts=unique_counts[: len(batch_metrics_list)],
        per_task_metric=per_task_metric,
    )

    # Reduce sums and counts, rather than averaging rank means. This excludes
    # formal padding batches (unique_count=0) and remains exact if rank counts
    # differ in a future protocol.
    global_means: dict[str, float] = {}
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    global_metric_counts: dict[str, int] = {}
    for key in all_keys:
        local_sum, local_count = local_metric_totals.get(key, [0.0, 0])
        pair = torch.tensor([float(local_sum), float(local_count)], device=accelerator.device)
        if distributed:
            torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.SUM)
        count = float(pair[1].item())
        if count > 0:
            global_means[key] = float(pair[0].item()) / count
            global_metric_counts[key] = int(count)

    if formal_h20:
        required_metrics = {"total_loss", "flow_mse"}
        invalid_metrics = {
            key: {
                "count": global_metric_counts.get(key, 0),
                "value": global_means.get(key),
            }
            for key in required_metrics
            if global_metric_counts.get(key) != 4_000
            or not np.isfinite(global_means.get(key, float("nan")))
        }
        if invalid_metrics:
            raise ValueError(
                f"Formal H20 validation core metrics are incomplete/non-finite: {invalid_metrics}"
            )
        local_unique_count = sum(int(value) for value in unique_counts[: len(batch_metrics_list)])
        unique_tensor = torch.tensor(float(local_unique_count), device=accelerator.device)
        if distributed:
            torch.distributed.all_reduce(unique_tensor, op=torch.distributed.ReduceOp.SUM)
        if int(unique_tensor.item()) != 4_000:
            raise ValueError(
                f"Formal H20 validation reduced {int(unique_tensor.item())} unique anchors; expected 4000"
            )

    per_task_global: dict[int, float] = {}
    if per_task_enabled:
        n_tasks_max = 1024
        sums = torch.zeros(n_tasks_max, device=accelerator.device)
        cnts = torch.zeros(n_tasks_max, device=accelerator.device)
        for task_id, (task_sum, task_count) in local_task_totals.items():
            if 0 <= task_id < n_tasks_max:
                sums[task_id] = float(task_sum)
                cnts[task_id] = int(task_count)
        if distributed:
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(cnts, op=torch.distributed.ReduceOp.SUM)
        for task_id in range(n_tasks_max):
            count = float(cnts[task_id].item())
            if count > 0:
                per_task_global[task_id] = float(sums[task_id].item()) / count
    if formal_h20:
        invalid_tasks = {
            task_id: {
                "count": int(cnts[task_id].item()),
                "value": per_task_global.get(task_id),
            }
            for task_id in range(50)
            if int(cnts[task_id].item()) != 80
            or not np.isfinite(per_task_global.get(task_id, float("nan")))
        }
        if invalid_tasks:
            raise ValueError(
                f"Formal H20 validation per-task metrics are incomplete/non-finite: {invalid_tasks}"
            )

    if is_main:
        epoch = (global_step // steps_per_epoch) + 1 if steps_per_epoch > 0 else 1
        epoch_step = (global_step % steps_per_epoch) + 1 if steps_per_epoch > 0 else global_step

        # Log to metrics.jsonl
        if metrics_file is not None:
            val_record = {
                "step": int(global_step),
                "epoch": int(epoch),
                "type": "validation",
            }
            for key in sorted(global_means.keys()):
                val_record[f"val_{key}"] = global_means[key]
            if per_task_global:
                val_record["val_per_task_metric"] = per_task_metric
                val_record["val_per_task"] = {
                    f"task-{tk:04d}": v for tk, v in sorted(per_task_global.items())
                }
                vals = list(per_task_global.values())
                val_record["val_per_task_min"] = min(vals)
                val_record["val_per_task_max"] = max(vals)
                val_record["val_per_task_n"] = len(vals)
            if getattr(val_loader, "coverage", None):
                val_record["val_subset"] = val_loader.coverage
            if sample_progress is not None:
                val_record.update(sample_progress.metrics())
            _metrics_buffer_write_boundary(val_record)

        # Log to W&B
        if config.wandb_enabled:
            try:
                wandb = _get_wandb()
                log_payload: dict[str, float] = {
                    "step": float(global_step),
                    "epoch": float(epoch),
                    "epoch_step": float(epoch_step),
                    "val/steps_per_epoch": float(steps_per_epoch),
                }
                for key, val in global_means.items():
                    log_payload[f"val/{key}"] = val
                if sample_progress is not None:
                    log_payload.update(sample_progress.metrics())
                wandb.log(log_payload, step=global_step)
            except Exception:
                logging.warning("wandb val log failed; continuing without wandb", exc_info=True)

        # Log summary line
        loss_total = global_means.get("total_loss", float("nan"))
        subtask_acc = global_means.get("subtask_accuracy", float("nan"))
        flow_mse = global_means.get("flow_mse", float("nan"))
        query_l1 = global_means.get("query_l1", float("nan"))
        logging.info(
            "  [VAL] step=%s loss_total=%.4f subtask_acc=%.4f flow_mse=%.6f query_l1=%.6f",
            global_step, loss_total, subtask_acc, flow_mse, query_l1,
        )

    return global_means


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    global_step: int,
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    data_manifest: dict | None = None,
    steps_per_epoch: int | None = None,
    sample_progress: _SampleProgress | None = None,
) -> None:
    if os.environ.get("OPENPI_DISABLE_CHECKPOINT", "0") in {"1", "true", "TRUE", "True"}:
        return
    # `global_step` is 1-based here: pass the post-update optimizer step so checkpoint directories
    # line up with the visible training step count.
    checkpoint_kind = _checkpoint_save_kind(
        config,
        global_step=global_step,
        steps_per_epoch=steps_per_epoch,
    )
    if checkpoint_kind is None:
        return

    epoch_checkpointing = checkpoint_kind in {"epoch", "rolling"}
    checkpoint_epoch = None
    if epoch_checkpointing:
        assert steps_per_epoch is not None  # Validated by _checkpoint_save_kind.
        checkpoint_epoch = (
            global_step // steps_per_epoch
            if checkpoint_kind == "epoch"
            else ((global_step - 1) // steps_per_epoch) + 1
        )

    if checkpoint_kind == "rolling":
        final_ckpt_dir = config.checkpoint_dir / (
            f"{_ROLLING_CHECKPOINT_DIR_PREFIX}{global_step:012d}"
        )
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_rolling_{global_step:012d}"
    elif checkpoint_kind == "epoch":
        # Keep numeric step directory compatibility for model loading while
        # recording explicit epoch metadata in metadata.pt and manifest.json.
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / (
            f"tmp_epoch_{checkpoint_epoch:04d}_{global_step}"
        )
    else:
        # Historical step-policy paths are intentionally unchanged.
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
    h20_formal_checkpoint = getattr(config, "name", None) in _H20_FORMAL_B1K_CONFIGS
    save_acc_state = os.environ.get("OPENPI_SAVE_ACCELERATE_STATE", "1") != "0"
    if h20_formal_checkpoint and not save_acc_state:
        raise ValueError("Formal H20 checkpoints require OPENPI_SAVE_ACCELERATE_STATE=1")
    if save_acc_state:
        acc_state_dir = tmp_ckpt_dir / "accelerate_state"
        t0 = time.time()
        if accelerator.is_main_process and _should_profile_memory_step(global_step):
            _reset_peak_memory_stats(accelerator)
        if h20_formal_checkpoint:
            # A failed collective state save must abort before the old published
            # rolling checkpoint is replaced.
            accelerator.save_state(str(acc_state_dir))
        else:
            try:
                accelerator.save_state(str(acc_state_dir))
            except Exception as exc:
                logging.warning(
                    "accelerator.save_state failed (resume may not work for sharded optimizers): %s", exc
                )
        if accelerator.is_main_process and acc_state_dir.exists():
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
        if sample_progress is not None:
            metadata["sample_progress"] = sample_progress.checkpoint_payload()
        if epoch_checkpointing:
            metadata["checkpoint_kind"] = checkpoint_kind
            metadata["epoch"] = checkpoint_epoch
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # Save manifest.json (human-readable checkpoint metadata fingerprint).
        def _write_manifest() -> None:
            manifest = _build_checkpoint_manifest(
                config=config,
                data_config=data_config,
                global_step=global_step,
                accelerator=accelerator,
                precision=config.pytorch_training_precision,
                data_manifest=data_manifest,
                checkpoint_kind=checkpoint_kind if epoch_checkpointing else None,
                checkpoint_epoch=checkpoint_epoch,
                sample_progress=sample_progress,
            )
            manifest_path = tmp_ckpt_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, default=str)

        if h20_formal_checkpoint:
            _write_manifest()
        else:
            try:
                _write_manifest()
            except Exception:
                logging.warning("Failed to write manifest.json", exc_info=True)

        if h20_formal_checkpoint:
            _validate_accelerate_state_artifacts(
                tmp_ckpt_dir / "accelerate_state",
                world_size=int(accelerator.num_processes),
                require_h20_deepspeed=True,
            )
            metadata_check = torch.load(
                tmp_ckpt_dir / "metadata.pt", map_location="cpu", weights_only=False
            )
            manifest_check = json.loads((tmp_ckpt_dir / "manifest.json").read_text())
            metadata_progress = metadata_check.get("sample_progress")
            manifest_progress = (manifest_check.get("run_metadata") or {}).get("sample_progress")
            if metadata_progress != manifest_progress:
                raise ValueError(
                    "Formal H20 checkpoint metadata/manifest sample progress mismatch"
                )

        # Save norm stats.
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        _atomic_write_checkpoint_dir(tmp_ckpt_dir, final_ckpt_dir)
        if epoch_checkpointing:
            # Publish only after the complete directory is visible. At an epoch
            # boundary the rolling pointer targets the durable directory and old
            # hidden rolling data is removed without duplicating the checkpoint.
            _publish_rolling_checkpoint(config.checkpoint_dir, final_ckpt_dir)
        if epoch_checkpointing:
            logging.info(
                "Saved %s checkpoint for epoch %s at step %s -> %s",
                checkpoint_kind,
                checkpoint_epoch,
                global_step,
                final_ckpt_dir,
            )
        else:
            logging.info("Saved checkpoint at step %s -> %s", global_step, final_ckpt_dir)

        if accelerator.is_main_process and config.wandb_enabled:
            try:
                wandb = _get_wandb()
                wandb.log({"checkpoint_step": global_step}, step=global_step)
            except Exception:
                logging.warning("wandb log failed; continuing without wandb", exc_info=True)
                try:
                    object.__setattr__(config, "wandb_enabled", False)
                except Exception:
                    pass

    accelerator.wait_for_everyone()


def _validate_runtime_config(config: _config.TrainConfig) -> None:
    """Fail fast on runtime options that would otherwise fail mid-training."""
    _validate_formal_b1k_contract(config)
    if config.prepare_hf_cache_only and config.force_load_cache:
        raise ValueError(
            "prepare_hf_cache_only and force_load_cache are mutually exclusive. "
            "prepare_hf_cache_only builds the cache from scratch; "
            "force_load_cache requires an already-built cache and fails if missing."
        )
    if int(config.log_interval) <= 0:
        raise ValueError("--log-interval must be a positive integer.")
    if int(config.val_log_interval) <= 0:
        raise ValueError("--val-log-interval must be a positive integer.")


def _resolved_gradient_checkpointing_state(model) -> bool | None:
    state_getter = getattr(model, "is_gradient_checkpointing_enabled", None)
    return state_getter() if state_getter is not None else getattr(model, "gradient_checkpointing_enabled", None)


def _configure_gradient_checkpointing(model, *, enabled: bool) -> None:
    """Apply the requested GC policy and fail closed if the model disagrees."""

    if enabled:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("Model does not support requested gradient checkpointing policy: enabled=True")
        model.gradient_checkpointing_enable()
    else:
        if not hasattr(model, "gradient_checkpointing_disable"):
            raise RuntimeError("Model does not support requested gradient checkpointing policy: enabled=False")
        model.gradient_checkpointing_disable()

    resolved = _resolved_gradient_checkpointing_state(model)
    if not isinstance(resolved, bool) or resolved is not enabled:
        raise RuntimeError(
            "Gradient checkpointing state mismatch: "
            f"requested={enabled} resolved={resolved!r}"
        )


# The kwargs the validation call site passes to compute_eval_metrics on BOTH KI
# variants, classified by what a missing one actually costs. The two classes are
# invoked through one shared `is_pi05_ki_joint` branch but are maintained
# separately, so this surface is where they silently drift apart.
#
#   metric selectors  -- choose WHICH metrics are produced. A variant may
#                        legitimately not emit one; consumers must then use
#                        `in`/`.get()` rather than unconditional indexing.
#   metric parameters -- only meaningful when their metric is enabled; they
#                        travel with it.
#   behaviour flags   -- change the STATISTICAL PROPERTIES of metrics that are
#                        reported. These must be honoured or hard-fail. Silently
#                        ignoring one is the worst case: with deterministic_flow
#                        off, flow_loss / expert_loss / total_loss carry a random
#                        component that does NOT shrink as the validation subset
#                        grows, so an arm that ignores it has an irreducible noise
#                        floor on exactly the metrics an A/B compares -- invisible,
#                        and fatal to the conclusion rather than to the run.
_KI_EVAL_METRIC_SELECTORS = ("compute_flow_l1",)
_KI_EVAL_METRIC_PARAMETERS = ("num_denoise_steps", "flow_l1_seed")
_KI_EVAL_BEHAVIOUR_FLAGS = ("deterministic_flow",)


def _validate_ki_eval_capability(model, config, *, is_main: bool) -> None:
    """Fail at step 0 if the model cannot honour the validation call the trainer makes.

    Motivation (concrete): the trainer passes ``deterministic_flow`` to both KI
    variants unconditionally, but Variant A's override of ``compute_eval_metrics``
    did not accept it. The result was ``TypeError: compute_eval_metrics() got an
    unexpected keyword argument 'deterministic_flow'`` raised at the FIRST
    validation -- i.e. after ~2h40m of training on 32 GPUs, not at startup. A
    bounded smoke shorter than one validation interval cannot catch it either.

    This check resolves what will be requested against what the model actually
    supports, before the training loop, so the same defect costs seconds.
    """
    import inspect

    if config.pytorch_model_name not in ("pi05_ki_joint_query", "pi05_ki_joint_fast"):
        return

    fn = getattr(type(model), "compute_eval_metrics", None)
    if fn is None:
        raise RuntimeError(
            f"model {type(model).__name__} is routed through the KI validation branch "
            "but does not implement compute_eval_metrics"
        )
    params = inspect.signature(fn).parameters
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    requested = (
        _KI_EVAL_METRIC_SELECTORS + _KI_EVAL_METRIC_PARAMETERS + _KI_EVAL_BEHAVIOUR_FLAGS
    )
    problems: list[str] = []

    for name in requested:
        if name in params:
            continue
        if accepts_var_kw:
            # Absorbed by **kwargs: no TypeError, and therefore no signal at all.
            problems.append(
                f"{name!r} is not an explicit parameter and would be silently swallowed "
                f"by **kwargs in {type(model).__name__}.compute_eval_metrics"
            )
        else:
            problems.append(
                f"{name!r} is not accepted by {type(model).__name__}.compute_eval_metrics "
                "and would raise TypeError at the first validation"
            )

    # A behaviour flag that is requested ON must be genuinely implemented, not
    # merely present in the signature. Signature parity with a no-op body is a
    # silent failure, so require the flag to be referenced in the body as well as
    # the signature (>= 2 occurrences: the parameter plus at least one use).
    #
    # This is a heuristic, deliberately a weak one: it catches a blatant no-op but
    # a body that merely mentions the name in a comment would satisfy it. It is a
    # backstop for the signature check, not a proof of correct semantics -- that is
    # what tests/test_pi05_ki_h20_bf16_two_arm.py asserts, by requiring both
    # variants to contain the actual mechanism (manual_seed, train_preprocess=False,
    # set_rng_state_all).
    source_available = True
    try:
        body = inspect.getsource(fn)
    except (OSError, TypeError):
        body = ""
        source_available = False
    for name in _KI_EVAL_BEHAVIOUR_FLAGS:
        if not bool(getattr(config, f"val_{name}", False)):
            continue
        if not source_available:
            # Do NOT skip silently: an unverifiable behaviour flag is a known gap,
            # not a pass. Real model classes are file-backed so this should not
            # occur in production.
            if is_main:
                logging.warning(
                    "KI eval capability: could not read the source of %s.compute_eval_metrics, "
                    "so whether behaviour flag %r is actually honoured is UNVERIFIED "
                    "(signature accepts it)",
                    type(model).__name__,
                    name,
                )
            continue
        if name in params and body.count(name) < 2:
            problems.append(
                f"behaviour flag {name!r} is accepted by {type(model).__name__}"
                ".compute_eval_metrics but never referenced in its body, so it would be "
                "accepted and ignored -- this changes the statistical properties of "
                "reported metrics rather than which metrics appear"
            )

    if problems:
        raise RuntimeError(
            "KI validation capability check failed for "
            f"{config.pytorch_model_name!r} ({type(model).__name__}):\n  - "
            + "\n  - ".join(problems)
        )

    if is_main:
        logging.info(
            "KI eval capability OK model=%s selectors=%s parameters=%s "
            "behaviour_flags=%s (val_deterministic_flow=%s)",
            type(model).__name__,
            list(_KI_EVAL_METRIC_SELECTORS),
            list(_KI_EVAL_METRIC_PARAMETERS),
            list(_KI_EVAL_BEHAVIOUR_FLAGS),
            bool(getattr(config, "val_deterministic_flow", False)),
        )
        # Variant-specific metric keys, logged once so an absent column later is
        # visible here rather than inferred from a gap in the metrics file.
        if config.pytorch_model_name == "pi05_ki_joint_fast":
            logging.info(
                "KI eval metric surface: Variant A emits action_ce_loss / "
                "action_token_accuracy and deliberately OMITS query_mse_loss / "
                "query_l1 (no learned queries, no query head). Consumers must use "
                "`in` or .get() for those keys; `query_l1=nan` in the [VAL] line is "
                "expected for this variant, not a failure."
            )


def _validate_a100_optimizer_offload_policy(config_name: str, ds_config: dict) -> None:
    if config_name in _A100_BF16_NO_OPTIMIZER_OFFLOAD_CONFIGS:
        family = "A100 formal"
    elif config_name in _H20_BF16_NO_OPTIMIZER_OFFLOAD_CONFIGS:
        family = "H20"
    else:
        return
    zero_config = ds_config.get("zero_optimization", {})
    if "offload_optimizer" in zero_config:
        raise ValueError(
            f"{family} config {config_name!r} requires optimizer offload disabled; "
            f"got {zero_config.get('offload_optimizer')!r}"
        )


def train_loop(config: _config.TrainConfig, *, formatter: logging.Formatter) -> None:
    _validate_runtime_config(config)

    kwargs_handlers = []
    if config.pytorch_model_name in ("pi05_ki_joint_query", "pi05_ki_joint_fast"):
        # Each KI optimizer step uses two distinct wrapped forwards.  DDP must
        # discover the parameters unused by each phase so both reducer passes
        # complete and synchronize the correct parameter subset.
        kwargs_handlers.append(
            DistributedDataParallelKwargs(find_unused_parameters=True)
        )

    accelerator = Accelerator(
        mixed_precision=_infer_accelerate_mixed_precision(config),
        gradient_accumulation_steps=int(getattr(config, "gradient_accumulation_steps", 1)),
        kwargs_handlers=kwargs_handlers,
    )
    if _is_formal_b1k_mode(config):
        from openpi.training.launcher_profile import materialize_effective_recipe

        # CLI parsing produces raw registered authorities. Materialize the shared
        # pure effective recipe exactly once after world size is known, then
        # validate the same effective object that the launcher resolved.
        config = materialize_effective_recipe(
            config, world_size=int(accelerator.num_processes)
        )
    _validate_formal_b1k_contract(config, accelerator=accelerator)

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
            latest = _latest_step_dir(
                config.checkpoint_dir,
                checkpoint_policy=getattr(config, "checkpoint_policy", _CHECKPOINT_POLICY_STEP),
            )
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
        if (
            getattr(config, "checkpoint_policy", _CHECKPOINT_POLICY_STEP)
            == _CHECKPOINT_POLICY_EPOCH_WITH_ROLLING
        ):
            _cleanup_rolling_checkpoint_artifacts(
                config.checkpoint_dir,
                keep_target=_rolling_checkpoint_target(config.checkpoint_dir),
            )
    accelerator.wait_for_everyone()
    if not is_main:
        _wait_for_path(config.log_dir, what="log_dir")

    add_file_logging(str(config.log_dir / f"rank{accelerator.process_index}.log"), formatter)
    install_excepthook()

    # Rank-0 metrics.jsonl writer: one JSON line per optimizer step.
    # Opened in append mode for resume safety; writes are buffered and
    # flushed only at log_interval / validation / checkpoint boundaries.
    _metrics_file = None
    if is_main:
        _metrics_path = config.log_dir / "metrics.jsonl"
        _metrics_file = open(_metrics_path, "a")
        _metrics_buffer_init(_metrics_file)
        logging.info("Writing per-step metrics to %s", _metrics_path)

    configure_hf_cache(config, accelerator=accelerator)
    os.environ["OPENPI_FORCE_LOAD_CACHE"] = "1" if config.force_load_cache else "0"
    if is_main:
        logging.info("prepare_hf_cache_only=%s", config.prepare_hf_cache_only)
        logging.info("force_load_cache=%s", config.force_load_cache)

    if not config.prepare_hf_cache_only:
        if _is_formal_b1k_mode(config):
            _init_formal_b1k_wandb(config, accelerator=accelerator, resuming=resuming)
        elif is_main:
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
        precision = str(config.pytorch_training_precision)
        _patch_deepspeed_config(
            ds_config,
            effective_batch_size=int(effective_batch_size),
            grad_accum_steps=int(accelerator.gradient_accumulation_steps),
            world_size=int(world_size),
            precision=precision,
            clip_grad_norm=float(config.optimizer.clip_gradient_norm),
        )
        _validate_deepspeed_precision_config(accelerator, ds_config, precision=precision)
        _validate_a100_optimizer_offload_policy(config.name, ds_config)
        if is_main:
            fp16_config = ds_config.get("fp16", {}) if isinstance(ds_config.get("fp16", {}), dict) else {}
            zero_config = (
                ds_config.get("zero_optimization", {})
                if isinstance(ds_config.get("zero_optimization", {}), dict)
                else {}
            )
            offload_config = (
                zero_config.get("offload_optimizer", {})
                if isinstance(zero_config.get("offload_optimizer", {}), dict)
                else {}
            )
            logging.info(
                "Patched DeepSpeed config: micro_bs=%s grad_accum=%s train_bs=%s bf16=%s fp16=%s "
                "grad_clip=%s fp16_initial_scale_power=%s fp16_loss_scale_window=%s fp16_hysteresis=%s "
                "fp16_min_loss_scale=%s zero_stage=%s reduce_bucket=%s allgather_bucket=%s "
                "overlap_comm=%s offload_optimizer_device=%s offload_optimizer_pin_memory=%s",
                ds_config.get("train_micro_batch_size_per_gpu"),
                ds_config.get("gradient_accumulation_steps"),
                ds_config.get("train_batch_size"),
                ds_config.get("bf16", {}).get("enabled", False),
                fp16_config.get("enabled", False),
                ds_config.get("gradient_clipping"),
                fp16_config.get("initial_scale_power"),
                fp16_config.get("loss_scale_window"),
                fp16_config.get("hysteresis"),
                fp16_config.get("min_loss_scale"),
                zero_config.get("stage"),
                zero_config.get("reduce_bucket_size"),
                zero_config.get("allgather_bucket_size"),
                zero_config.get("overlap_comm"),
                offload_config.get("device"),
                offload_config.get("pin_memory"),
            )

    # Build the training loader under the config's streaming anchor stride.
    # The stride always comes from config.streaming_anchor_stride. Epoch-anchor-
    # offset configs build the first epoch with offset[0] and rebuild at each
    # epoch boundary; the formal B1K path sets no offsets and so builds once.
    epoch_offsets = getattr(config, "epoch_anchor_offsets", None)
    initial_offset = int(epoch_offsets[0]) if epoch_offsets else 0
    with _train_b1k_anchor_env(
        getattr(config, "streaming_anchor_stride", 1),
        initial_offset,
    ):
        loader, data_config = build_datasets(config)

    # Build validation data loader (if val_data is configured)
    val_loader = None
    val_data_config = None
    if config.val_data:
        # Validation always uses the baseline stride-1 / no-drop contract so
        # that metrics are computed on the full-resolution data, independent
        # of any training-side streaming anchor stride (e.g. stride-4 skill
        # bridge or the stride-12 formal passes).
        with _baseline_b1k_dataset_env():
            val_loader, val_data_config = build_val_datasets(config)
        if is_main:
            val_eps = getattr(val_data_config, "episodes_index", None)
            val_tasks = getattr(val_data_config, "tasks", None)
            logging.info(
                "Validation data: tasks=%s episodes=%s val_num_batches=%s val_log_interval=%s",
                val_tasks,
                len(val_eps) if val_eps is not None else "N/A",
                config.val_num_batches,
                config.val_log_interval,
            )

    if config.prepare_hf_cache_only:
        if is_main:
            logging.info("PREPARE_HF_CACHE_ONLY mode: cache build complete for train+val; exiting.")
        _metrics_buffer_close()
        return

    # Epoch accounting: len(loader) is per-rank micro-batch count.
    steps_per_epoch_micro = len(loader)
    steps_per_epoch = max(1, steps_per_epoch_micro // accelerator.gradient_accumulation_steps)
    if steps_per_epoch <= 0:
        raise RuntimeError(f"Computed steps_per_epoch={steps_per_epoch}, expected a positive value.")

    # Streaming anchor stride: the B1K streaming dataset advances its cursor by
    # ``config.streaming_anchor_stride`` frames per sample, so a single pass over
    # the unique anchors spans ``len(loader) // stride`` micro-batches, not the
    # full raw frame count. Without this correction, ``num_train_epochs=N``
    # would iterate the anchor cycle ~``stride`` times per "epoch". We floor to
    # the nearest complete anchor (incomplete trailing horizons are dropped,
    # matching ``_streaming_drop_incomplete_horizon`` semantics). The grad-accum
    # division is applied AFTER the stride reduction, so the effective optimizer
    # steps per epoch == (microbatches // stride) // grad_accum.
    #
    # KNOWN OVERSHOOT, quantified. ``len(dataset)`` is inherited from
    # LeRobotDataset and is the RAW frame count: it is unaware of both the stride
    # and of ``drop_incomplete_horizon``. Dividing by the stride recovers the
    # stride reduction but NOT the horizon rejection, so this estimate exceeds
    # the true count of horizon-eligible anchors. For the B1K train population
    # (107,696,389 raw frames, action_horizon 32, first 180 sorted episodes of
    # each of the 50 tasks) the exact eligible count at stride 4 is 26,857,712
    # versus 26,924,097 for raw//4 -- an overshoot of 66,385 anchors, 0.247%.
    # At global batch 1024 that is 26,293 estimated versus 26,228 exact, i.e.
    # the last ~65 steps re-read the head of the sweep rather than extending it.
    # This is a duplicate-exposure rounding effect, not a coverage loss, and it
    # is bounded by one stride-pass worth of horizon tails. Deriving the exact
    # value would require replicating the dataset's eligibility predicate here,
    # which would drift from the dataset itself; the estimate is used instead and
    # the discrepancy is logged so it can never be silent.
    streaming_stride = int(getattr(config, "streaming_anchor_stride", 1))
    if streaming_stride > 1 and config.num_train_epochs is not None:
        effective_micro = max(1, steps_per_epoch_micro // streaming_stride)
        effective_steps_per_epoch = max(
            1, effective_micro // accelerator.gradient_accumulation_steps
        )
        if is_main:
            logging.info(
                "Streaming anchor stride=%s: steps_per_epoch reduced from %s to %s "
                "(raw_micro=%s, effective_micro=%s//%s=%s, grad_accum=%s) "
                "so one epoch == one pass over unique anchors; this is an UPPER BOUND "
                "because len(dataset) is horizon-unaware (see comment above)",
                streaming_stride,
                steps_per_epoch,
                effective_steps_per_epoch,
                steps_per_epoch_micro,
                steps_per_epoch_micro,
                streaming_stride,
                effective_micro,
                accelerator.gradient_accumulation_steps,
            )
        steps_per_epoch = effective_steps_per_epoch

    # ---- Batch-invariant cadences ----
    # Formal profiles were materialized once from the shared pure resolver after
    # world size became known. Non-formal configs retain their historical local
    # conversion because they are not launcher-profile contracts.
    global_batch = (
        int(config.batch_size_per_gpu)
        * int(accelerator.num_processes)
        * int(accelerator.gradient_accumulation_steps)
    )
    if not _is_formal_b1k_mode(config):
        for samples_field, steps_field in (
            ("val_interval_samples", "val_log_interval"),
            ("save_interval_samples", "save_interval"),
        ):
            samples = getattr(config, samples_field, None)
            if samples is None:
                continue
            if int(samples) <= 0:
                raise ValueError(f"{samples_field} must be positive when set; got {samples!r}")
            object.__setattr__(
                config, steps_field, max(1, int(samples) // global_batch)
            )
        if getattr(config, "save_interval_samples", None) is not None:
            object.__setattr__(
                config, "rolling_checkpoint_interval", int(config.save_interval)
            )

    if config.num_train_epochs is not None:
        if config.num_train_epochs <= 0:
            raise ValueError("--num_train_epochs must be a positive integer when provided.")
        if epoch_offsets is not None and len(epoch_offsets) != int(config.num_train_epochs):
            raise ValueError(
                f"epoch_anchor_offsets length ({len(epoch_offsets)}) must equal "
                f"num_train_epochs ({config.num_train_epochs}); got offsets={epoch_offsets}"
            )
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
        if (
            config.save_at_epoch_end_only
            and getattr(config, "checkpoint_policy", _CHECKPOINT_POLICY_STEP) == _CHECKPOINT_POLICY_STEP
        ):
            object.__setattr__(config, "save_interval", target_steps)
            if is_main:
                logging.info("save_at_epoch_end_only enabled: save_interval=%s", target_steps)

    # ---- Resolved recipe: derivation consistency + one auditable log line ----
    # These are consistency assertions, not remembered values. They are the
    # replacement for the deleted literal gates (warmup/peak_lr/decay_steps/
    # num_train_steps == fixed tuple), which had to be hand-edited on every
    # batch or stride change and so froze a stale budget back in each time.
    resolved_epochs = getattr(config, "num_train_epochs", None)
    resolved_steps = int(config.num_train_steps)
    resolved_decay = int(config.lr_schedule.decay_steps) or resolved_steps
    resolved_warmup = int(config.lr_schedule.warmup_steps)
    if steps_per_epoch <= 0:
        raise RuntimeError(f"steps_per_epoch must be positive; got {steps_per_epoch}")
    if resolved_steps <= 0:
        raise RuntimeError(f"num_train_steps must be positive after derivation; got {resolved_steps}")
    if resolved_epochs is not None and resolved_steps != int(resolved_epochs) * steps_per_epoch:
        raise RuntimeError(
            f"Budget inconsistency: num_train_steps={resolved_steps} != "
            f"num_train_epochs={resolved_epochs} x steps_per_epoch={steps_per_epoch}"
        )
    # The strict schedule-span assertions are scoped to the formal B1K protocol.
    # Elsewhere a decay_steps that differs from num_train_steps is a tolerated
    # (warned + auto-corrected) condition further down, and several exploratory
    # configs rely on that -- e.g. the long-baseline pair uses
    # num_train_steps=200 with decay_steps=500. Promoting that to a hard failure
    # here would break configs this refactor is not about.
    if _is_formal_b1k_mode(config):
        if resolved_decay != resolved_steps:
            raise RuntimeError(
                f"Formal B1K: decay_steps={resolved_decay} must equal "
                f"num_train_steps={resolved_steps} so the cosine schedule spans exactly the budget"
            )
        if resolved_warmup >= resolved_steps:
            raise RuntimeError(
                f"Formal B1K: warmup_steps={resolved_warmup} must be < "
                f"num_train_steps={resolved_steps}"
            )
    if is_main:
        # Single line so the whole recipe can be audited from the log without
        # reading the config. num_train_epochs and steps_per_epoch are both
        # present on purpose: without epochs, the discriminating relation
        # num_train_steps == epochs x steps_per_epoch cannot be evaluated, and a
        # checker that silently skips it would certify a hardcoded budget as
        # "derived". samples_per_step is included so throughput (samples/s) can
        # be computed from any two log timestamps.
        logging.info(
            "RESOLVED_RECIPE config=%s stride=%s batch_size_per_gpu=%s world=%s grad_accum=%s "
            "global_batch=%s samples_per_step=%s num_train_epochs=%s steps_per_epoch=%s "
            "num_train_steps=%s decay_steps=%s warmup_steps=%s peak_lr=%.6g "
            "val_log_interval=%s save_interval=%s total_samples=%s",
            config.name,
            streaming_stride,
            int(config.batch_size_per_gpu),
            int(accelerator.num_processes),
            int(accelerator.gradient_accumulation_steps),
            global_batch,
            global_batch,
            resolved_epochs,
            steps_per_epoch,
            resolved_steps,
            resolved_decay,
            resolved_warmup,
            float(config.lr_schedule.peak_lr),
            int(config.val_log_interval),
            int(config.save_interval),
            resolved_steps * global_batch,
        )

    checkpoint_policy = getattr(config, "checkpoint_policy", _CHECKPOINT_POLICY_STEP)
    if checkpoint_policy == _CHECKPOINT_POLICY_EPOCH_WITH_ROLLING:
        rolling_interval = int(getattr(config, "rolling_checkpoint_interval", 0))
        if rolling_interval <= 0:
            raise ValueError(
                "epoch_with_rolling checkpoint policy requires --rolling-checkpoint-interval > 0"
            )
        if is_main:
            logging.info(
                "Checkpoint policy: durable at every %s-step epoch boundary; "
                "rolling recovery every %s optimizer steps via %s",
                steps_per_epoch,
                rolling_interval,
                config.checkpoint_dir / _ROLLING_CHECKPOINT_LINK_NAME,
            )
    elif checkpoint_policy != _CHECKPOINT_POLICY_STEP:
        raise ValueError(f"Unsupported checkpoint_policy: {checkpoint_policy}")

    # ---- Data manifest: compute after data loaders are built, before training starts ----
    # Computed on all ranks for symmetry, but only logged/saved on rank 0.
    _data_manifest: dict | None = None
    try:
        _data_manifest = _compute_data_manifest(
            config=config,
            data_config=data_config,
            train_loader=loader,
            val_loader=val_loader,
            val_data_config=val_data_config,
            steps_per_epoch=steps_per_epoch,
            world_size=world_size,
            grad_accum_steps=accelerator.gradient_accumulation_steps,
            seed=int(config.seed),
            train_shuffle=True,
            val_shuffle=False,
            num_probe_batches=int(os.environ.get("OPENPI_DATA_MANIFEST_PROBE_BATCHES", "0")),
        )
        if is_main:
            # Write data_manifest.json to log dir
            manifest_path = config.log_dir / "data_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(_data_manifest, f, indent=2, default=str)
            logging.info("Data manifest written to %s", manifest_path)

            # Also log a "manifest" record to metrics.jsonl at step 0
            if _metrics_file is not None:
                manifest_record = {
                    "step": 0,
                    "epoch": 1,
                    "type": "manifest",
                    "action_horizon": _data_manifest.get("action_horizon"),
                    "action_dim": _data_manifest.get("action_dim"),
                    "state_dim": _data_manifest.get("state_dim"),
                    "image_keys": _data_manifest.get("image_keys", []),
                    "n_train_episodes": _data_manifest.get("n_train_episodes"),
                    "n_val_episodes": _data_manifest.get("n_val_episodes"),
                    "steps_per_epoch": _data_manifest.get("steps_per_epoch"),
                    "has_val_data": _data_manifest.get("has_val_data"),
                    "has_subtask_tokens": _data_manifest.get("has_subtask_tokens"),
                    "train_sha256": (_data_manifest.get("data_sha") or {}).get("sha256"),
                    "is_streaming": (_data_manifest.get("streaming_dataset") or {}).get("is_streaming"),
                }
                _metrics_buffer_append(manifest_record)

            train_probe = _data_manifest.get("train_probe", {}) or {}
            val_probe = _data_manifest.get("val_probe", {}) or {}
            logging.info(
                "Data manifest: train_eps=%s val_eps=%s action_dim=%s state_dim=%s "
                "action_horizon=%s steps_per_epoch=%s train_batches_sampled=%s val_batches_sampled=%s",
                _data_manifest.get("n_train_episodes"),
                _data_manifest.get("n_val_episodes"),
                _data_manifest.get("action_dim"),
                _data_manifest.get("state_dim"),
                _data_manifest.get("action_horizon"),
                _data_manifest.get("steps_per_epoch"),
                train_probe.get("num_batches_sampled", 0),
                val_probe.get("num_batches_sampled", 0),
            )
    except Exception:
        if getattr(config, "name", None) in _H20_FORMAL_B1K_CONFIGS:
            raise RuntimeError("Formal H20 data manifest validation failed")
        logging.warning("Failed to compute data manifest; continuing without it", exc_info=True)
        _data_manifest = None

    # Build model (same logic as train_pytorch.py).
    import openpi.models.pi0_config as _pi0_config
    import openpi.models.pi05_subtask_config as _pi05_subtask_config
    import openpi.models.vlm2_vla_config as _vlm2_vla_config

    if isinstance(config.model, _vlm2_vla_config.VLM2VLAConfig):
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    elif isinstance(config.model, _pi05_subtask_config.Pi05SubtaskConfig):
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    elif not isinstance(config.model, _pi0_config.Pi0Config):
        model_cfg = _pi0_config.Pi0Config(
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
    # Both KI variants share the two-phase (backbone then expert) training loop;
    # they differ only in the backbone action objective.
    is_pi05_ki_joint = config.pytorch_model_name in ("pi05_ki_joint_query", "pi05_ki_joint_fast")
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
    elif config.pytorch_model_name == "pi0_hamlet":
        import openpi.models_pytorch.pi0_hamlet as _pi0_hamlet

        model = _pi0_hamlet.Pi05WithHamlet(model_cfg)
    elif config.pytorch_model_name == "pi0_memoryvla":
        import openpi.models_pytorch.pi0_memoryvla as _pi0_memoryvla

        model = _pi0_memoryvla.Pi05WithMemoryVLA(model_cfg)
    elif config.pytorch_model_name == "subtask":
        import openpi.models_pytorch.pi05_subtask as _pi05_subtask

        alpha = getattr(model_cfg, "alpha", 10.0)
        model = _pi05_subtask.PI05SubtaskPytorch(
            model_cfg,
            alpha=alpha,
            action_expert_name="subtask",
        )
    elif config.pytorch_model_name in ("pi05_ki_joint_query", "pi05_ki_joint_fast"):
        if config.pytorch_model_name == "pi05_ki_joint_fast":
            # Variant A: discrete FAST action tokens + cross-entropy backbone
            # objective (paper-accurate Knowledge Insulation).
            import openpi.models_pytorch.pi05_ki_joint_fast as _pi05_ki_joint_fast

            model = _pi05_ki_joint_fast.PI05KIJointFastPytorch(model_cfg)
            variant_label = "FAST action-token CE variant"
            objective_weight_name = "beta_action"
        else:
            # Variant B: learned action queries + MSE backbone objective.
            import openpi.models_pytorch.pi05_ki_joint_query as _pi05_ki_joint_query

            model = _pi05_ki_joint_query.PI05KIJointQueryPytorch(model_cfg)
            variant_label = "query-MSE variant"
            objective_weight_name = "beta_query"
        if is_main:
            ki = bool(getattr(model, "knowledge_insulation", False))
            logging.info(
                "π0.5-KI joint %s model loaded (knowledge_insulation=%s, beta_text=%.3f, %s=%.3f)",
                variant_label,
                ki,
                getattr(model, "beta_text", 1.0),
                objective_weight_name,
                getattr(model, objective_weight_name, 1.0),
            )
    else:
        import openpi.models_pytorch.pi0_pytorch as _pi0_pytorch

        model = _pi0_pytorch.PI0Pytorch(model_cfg)

    gradient_checkpointing = bool(config.gradient_checkpointing)
    _configure_gradient_checkpointing(model, enabled=gradient_checkpointing)
    if is_main:
        logging.info(
            "Gradient checkpointing runtime policy resolved: requested=%s enabled=%s",
            gradient_checkpointing,
            _resolved_gradient_checkpointing_state(model),
        )

    # Step-0 gate: prove the model can honour the validation call the trainer will
    # make, before spending any training time. See _validate_ki_eval_capability.
    _validate_ki_eval_capability(model, config, is_main=is_main)

    # Memory/perf knobs (keep same behavior as train_pytorch.py).
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # A formal launcher may set this before Python imports torch, which is
        # when the allocator configuration must take effect. Never overwrite an
        # explicitly validated launch contract after CUDA/model initialization.
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True"
        )
        if is_main:
            logging.info(
                "Enabled memory optimizations for 8+ GPU training "
                "(PYTORCH_CUDA_ALLOC_CONF=%s)",
                os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            )

    # Weight loading for fine-tuning.
    if config.pytorch_weight_path is not None:
        if is_main:
            logging.info("Loading weights from: %s", config.pytorch_weight_path)
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        if not os.path.exists(model_path):
            if is_main:
                logging.warning("Model checkpoint not found at %s. Skipping weight loading.", model_path)
        else:
            load_strict = config.pytorch_model_name not in (
                "pi05_ki_joint_fast",
                "vlm2",
                "vlm2_subtask",
                "subtask",
                "pi0_hamlet",
                "pi0_memoryvla",
                "pi05_ki_joint_query",
            )
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

    optim_params = _trainable_parameters(model)
    if len(optim_params) == 0:
        raise RuntimeError("No trainable parameters found (all parameters are frozen).")

    # Per-group peak/end LRs for pi05_ki_joint_query joint model (backbone + expert).
    # Read from config if available, otherwise fall back to global schedule values.
    bb_peak_lr = float(getattr(config, "backbone_lr", peak_lr))
    ex_peak_lr = float(getattr(config, "expert_lr", peak_lr))
    bb_end_lr = float(getattr(config, "backbone_end_lr", end_lr))
    ex_end_lr = float(getattr(config, "expert_end_lr", end_lr))

    # LR schedule factory — builds an independent cosine schedule for each param group.
    def _make_lr_schedule(peak: float, end_val: float):
        def _schedule(step: int) -> float:
            return _cosine_lr_value(
                step,
                warmup_steps=warmup_steps,
                peak_lr=peak,
                decay_steps=decay_steps,
                end_lr=end_val,
            )

        return _schedule

    lr_schedule_bb = _make_lr_schedule(bb_peak_lr, bb_end_lr)
    lr_schedule_ex = _make_lr_schedule(ex_peak_lr, ex_end_lr)

    use_8bit_optim = os.environ.get("USE_8BIT_OPTIM", "0") == "1"
    if is_pi05_ki_joint:
        # π0.5-KI joint query: single AdamW with two param groups (backbone + expert).
        # DeepSpeed ZeRO-2 shards one optimizer state across ranks → memory efficient.
        bb_params = list(model.get_backbone_params())
        ex_params = list(model.get_expert_params())

        # Verify coverage (non-zero, no overlap with trainable).
        bb_ids = {id(p) for p in bb_params}
        ex_ids = {id(p) for p in ex_params}
        if bb_ids & ex_ids:
            raise ValueError("Backbone and expert param groups overlap!")
        all_ids = bb_ids | ex_ids
        trainable_ids = {id(p) for p in optim_params}
        missing = trainable_ids - all_ids
        if missing:
            raise ValueError(f"{len(missing)} trainable params not in backbone or expert groups.")

        if use_8bit_optim:
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(
                    [
                        {"params": bb_params, "lr": bb_peak_lr, "name": "backbone"},
                        {"params": ex_params, "lr": ex_peak_lr, "name": "expert"},
                    ],
                    betas=(config.optimizer.b1, config.optimizer.b2),
                    eps=config.optimizer.eps,
                    weight_decay=config.optimizer.weight_decay,
                )
                if is_main:
                    logging.info("Using 8-bit AdamW with 2 param groups (backbone/expert)")
            except ImportError:
                if is_main:
                    logging.warning("bitsandbytes not found, falling back to standard AdamW")
                optimizer = torch.optim.AdamW(
                    [
                        {"params": bb_params, "lr": bb_peak_lr, "name": "backbone"},
                        {"params": ex_params, "lr": ex_peak_lr, "name": "expert"},
                    ],
                    betas=(config.optimizer.b1, config.optimizer.b2),
                    eps=config.optimizer.eps,
                    weight_decay=config.optimizer.weight_decay,
                )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": bb_params, "lr": bb_peak_lr, "name": "backbone"},
                    {"params": ex_params, "lr": ex_peak_lr, "name": "expert"},
                ],
                betas=(config.optimizer.b1, config.optimizer.b2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
            )

        if is_main:
            bb_count = sum(p.numel() for p in bb_params)
            ex_count = sum(p.numel() for p in ex_params)
            logging.info(
                "π0.5-KI joint query param-group optimizer: backbone=%d params (%.2fM, lr=%.2e), "
                "expert=%d params (%.2fM, lr=%.2e)",
                len(bb_params), bb_count / 1e6, bb_peak_lr,
                len(ex_params), ex_count / 1e6, ex_peak_lr,
            )
    elif use_8bit_optim:
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
        return _cosine_lr_value(
            step,
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            decay_steps=decay_steps,
            end_lr=end_lr,
        )

    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        _patch_deepspeed_loss_scaler()
        if _fast_grad_norm_enabled():
            _patch_deepspeed_grad_norm()

    # Prepare with Accelerator (DDP or DeepSpeed).
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    two_phase_update = _TwoPhaseUpdateController(accelerator) if is_pi05_ki_joint else None
    # `accelerator.prepare()` may wrap/replace the model parameters (especially for DeepSpeed).
    # Keep a post-prepare view for gradient clipping/debugging; otherwise debug stats can report
    # grad_tensors=0 even when DeepSpeed computed a non-zero grad_norm.
    optim_params = _trainable_parameters(model)

    # Patch DeepSpeed engine to not disable our outer torch.autocast context.
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        _patch_deepspeed_autocast(accelerator)

    # Resume (after prepare so that accelerator can restore distributed states).
    global_step = 0
    resume_metadata: dict = {}
    if resuming:
        global_step, resume_metadata, _ = _load_resume_state(
            accelerator,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_policy=getattr(config, "checkpoint_policy", _CHECKPOINT_POLICY_STEP),
            formal=getattr(config, "name", None) in _H20_FORMAL_B1K_CONFIGS,
            selected=latest,
            config=config,
            data_config=data_config,
        )
        if is_main:
            logging.info("Resumed training from step %s", global_step)

    epoch_num_samples = int(steps_per_epoch) * int(global_batch)
    sample_progress = (
        _restore_sample_progress(
            resume_metadata,
            current_samples_per_update=global_batch,
            default_epoch_num_samples=epoch_num_samples,
        )
        if resuming
        else _SampleProgress.fresh(
            samples_per_update=global_batch,
            epoch_num_samples=epoch_num_samples,
        )
    )

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

    # Pre-compute autocast settings (used in both training and validation)
    ds_plugin_val = accelerator.state.deepspeed_plugin if accelerator.distributed_type == DistributedType.DEEPSPEED else None
    ds_uses_torch_autocast_val = bool(
        ds_plugin_val is not None
        and ds_plugin_val.deepspeed_config.get("torch_autocast", {}).get("enabled", False)
    )
    use_autocast = (
        accelerator.device.type == "cuda"
        and config.pytorch_training_precision in ("bfloat16", "float16")
        and not ds_uses_torch_autocast_val
    )
    autocast_dtype = (
        torch.bfloat16 if config.pytorch_training_precision == "bfloat16" else torch.float16
    )

    start_time = time.time()
    infos: list[dict[str, float]] = []
    consecutive_skipped_updates = 0
    consecutive_nonfinite_losses = 0
    total_nonfinite_loss_batches = 0
    total_nonfinite_grad_updates = 0
    total_ds_overflow_skipped_updates = 0
    max_consecutive_skipped_updates = int(os.environ.get("OPENPI_MAX_CONSECUTIVE_SKIPPED_UPDATES", "50"))
    max_consecutive_nonfinite_losses = int(os.environ.get("OPENPI_MAX_CONSECUTIVE_NONFINITE_LOSSES", "50"))
    let_deepspeed_handle_nonfinite_grad = _env_flag("OPENPI_DS_HANDLE_NONFINITE_GRAD", True)
    if is_main:
        logging.info(
            "FP16 stability guards: max_consecutive_skipped_updates=%s "
            "max_consecutive_nonfinite_losses=%s ds_handle_nonfinite_grad=%s debug_overflow=%s",
            max_consecutive_skipped_updates,
            max_consecutive_nonfinite_losses,
            let_deepspeed_handle_nonfinite_grad,
            _debug_overflow_enabled(config),
        )

    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    last_epoch_logged = None
    train_iterator = None
    epoch_anchor_index = 0 if epoch_offsets is not None else None
    while global_step < int(config.num_train_steps):
        # Epoch-anchor-offset mode: rebuild the loader when a new epoch starts
        # so the dataset captures the next offset from the env. The streaming
        # dataset reads OPENPI_B1K_ANCHOR_OFFSET at construction time; merely
        # calling iter(loader) would replay the same offset for every epoch.
        #
        # The formal B1K path deliberately does NOT set epoch_anchor_offsets and
        # therefore never rebuilds. Its three stride-12 offset passes were
        # removed because they union to exactly one stride-4 sweep, and each
        # boundary performed a full mid-training
        # `del loader; build_datasets(); accelerator.prepare()` -- the same
        # fork-while-holding-flock path that has deadlocked in practice. That
        # rebuild's 3-retry protection lives in `except` clauses and a deadlock
        # never raises, so the retry was structurally blind to the failure it
        # appeared to cover. One sweep, no mid-run rebuilds.
        if epoch_offsets is not None:
            current_epoch = global_step // steps_per_epoch
            if current_epoch != epoch_anchor_index:
                if current_epoch >= len(epoch_offsets):
                    break
                accelerator.wait_for_everyone()
                _close_training_iterator(train_iterator)
                train_iterator = None
                del loader
                gc.collect()
                next_offset = int(epoch_offsets[current_epoch])
                with _train_b1k_anchor_env(
                    int(getattr(config, "streaming_anchor_stride", 1)),
                    next_offset,
                ):
                    loader, _ = build_datasets(config)
                loader = accelerator.prepare(loader)
                epoch_anchor_index = current_epoch
                if is_main:
                    logging.info(
                        "Epoch anchor offset rotated: epoch=%s offset=%s stride=%s",
                        current_epoch + 1,
                        next_offset,
                        getattr(config, "streaming_anchor_stride", 1),
                    )
        train_iterator = iter(loader)

        for observation, actions in train_iterator:
            if global_step >= int(config.num_train_steps):
                break
            if epoch_anchor_index is not None and global_step >= (epoch_anchor_index + 1) * steps_per_epoch:
                break

            profile_memory = is_main and _should_profile_memory_step(global_step)

            # Move data to device.
            # NOTE: Observation is a flax.struct.dataclass, which is *not* a dm-tree container.
            # tree.map_structure would treat it as a leaf and leave nested tensors on CPU.
            if _model is not None and isinstance(observation, _model.Observation):
                observation = _move_observation_to_device(observation, accelerator.device)
            else:
                observation = tree.map_structure(
                    lambda x: x.to(accelerator.device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
                    observation,
                )
            actions = actions.to(device=accelerator.device, dtype=torch.float32, non_blocking=True)

            with accelerator.accumulate(model):
                # Update LR per optimizer step (only when syncing grads).
                if accelerator.sync_gradients:
                    lr = lr_schedule(global_step)
                    if is_pi05_ki_joint:
                        # Independent cosine schedules per param group.
                        lr_bb = lr_schedule_bb(global_step)
                        lr_ex = lr_schedule_ex(global_step)
                        for pg in optimizer.param_groups:
                            if pg.get("name") == "backbone":
                                pg["lr"] = lr_bb
                            elif pg.get("name") == "expert":
                                pg["lr"] = lr_ex
                            else:
                                pg["lr"] = lr
                    else:
                        for pg in optimizer.param_groups:
                            pg["lr"] = lr

                extra_metrics: dict[str, float] = {}
                ds_plugin = accelerator.state.deepspeed_plugin if accelerator.distributed_type == DistributedType.DEEPSPEED else None
                ds_uses_torch_autocast = bool(
                    ds_plugin is not None
                    and ds_plugin.deepspeed_config.get("torch_autocast", {}).get("enabled", False)
                )
                # If DeepSpeed has torch_autocast enabled, do not wrap another autocast context.
                # Otherwise, we enable autocast for half-precision modes to avoid dtype mismatch
                # (e.g., fp16 weights with fp32 activations).
                use_autocast = (
                    accelerator.device.type == "cuda"
                    and config.pytorch_training_precision in ("bfloat16", "float16")
                    and not ds_uses_torch_autocast
                )
                autocast_dtype = (
                    torch.bfloat16 if config.pytorch_training_precision == "bfloat16" else torch.float16
                )
                if is_pi05_ki_joint:
                    # ================================================================
                    # π0.5-KI joint query: two-phase forward/backward (one graph at a time)
                    # Phase 1: backbone (CE + query MSE) → backward → free graph
                    # Phase 2: expert (flow matching) → backward → free graph
                    # Single optimizer with 2 param groups; one step at the end.
                    # Memory: at most one 3.6B graph in memory at once.
                    # KI: structural (detached KV in expert forward); when KI=ON,
                    #   phase-2 backward produces zero backbone grads.
                    # ================================================================
                    # -- Diagnostic optimization: cheap per-step finite check,
                    #    detailed _gather_scalar_stats only at log_interval or on error.
                    bb_loss_stats = None
                    ex_loss_stats = None
                    # Whether we need detailed scalar stats (min/max/mean/std) this step.
                    need_detailed_stats = (
                        (accelerator.sync_gradients and global_step % int(config.log_interval) == 0)
                        or _debug_overflow_enabled(config)
                    )

                    if profile_memory:
                        _reset_peak_memory_stats(accelerator)

                    # ---- Phase 1: backbone forward + loss check + backward ----
                    with torch.autocast(
                        device_type=accelerator.device.type,
                        dtype=autocast_dtype,
                        enabled=use_autocast,
                    ):
                        # Use the wrapper-visible forward path so DDP reducer
                        # and DeepSpeed/autocast hooks run for this phase.
                        bb_losses = model(observation, actions, phase="backbone")
                        bb_loss = bb_losses["backbone_loss"]

                    # Cheap cross-rank finiteness check for backbone loss
                    # (1 all-reduce instead of 5 from _gather_scalar_stats).
                    bb_all_finite = (
                        _gather_finite_consensus(accelerator, bb_loss)
                        if _loss_finite_check_enabled()
                        else True
                    )
                    bb_nonfinite = not bb_all_finite

                    if bb_nonfinite:
                        # Non-finite detected: compute detailed stats for error diagnostics.
                        bb_loss_stats = _gather_scalar_stats(accelerator, bb_loss)
                        if is_main:
                            logging.error(
                                "Non-finite backbone loss at step=%s: "
                                "finite=%d/%d bad_rank=%d min=%s max=%s mean=%s",
                                global_step,
                                int(bb_loss_stats["finite_count"]),
                                int(bb_loss_stats["total_count"]),
                                int(bb_loss_stats["bad_rank"]),
                                bb_loss_stats["min"],
                                bb_loss_stats["max"],
                                bb_loss_stats["mean"],
                            )
                    elif need_detailed_stats:
                        # Log interval or debug mode: compute detailed stats.
                        bb_loss_stats = _gather_scalar_stats(accelerator, bb_loss)

                    if bb_nonfinite:
                        if not torch.isfinite(bb_loss.detach()):
                            _log_nonfinite_batch_state(loss=bb_loss, actions=actions, observation=observation)
                            _save_nonfinite_debug_dump(
                                output_dir=config.log_dir,
                                global_step=global_step,
                                accelerator=accelerator,
                                loss=bb_loss,
                                actions=actions,
                                observation=observation,
                            )
                        consecutive_nonfinite_losses += 1
                        total_nonfinite_loss_batches += 1
                        two_phase_update.clear_gradients(optimizer)
                        accelerator.wait_for_everyone()
                        # Flush metrics before raising / skipping on error.
                        _metrics_buffer_flush()
                        if consecutive_nonfinite_losses >= max_consecutive_nonfinite_losses:
                            raise FloatingPointError(
                                "Too many consecutive non-finite backbone losses. "
                                f"Reached {consecutive_nonfinite_losses} skipped batches."
                            )
                        if is_main:
                            logging.warning(
                                "Skipping non-finite backbone loss at step=%s; "
                                "consecutive_nonfinite_losses=%s/%s",
                                global_step,
                                consecutive_nonfinite_losses,
                                max_consecutive_nonfinite_losses,
                            )
                        continue

                    # The measured no-optimizer-offload policy requires the
                    # first phase not to declare the outer update boundary.
                    two_phase_update.backward_first_phase(bb_loss)

                    # Capture scalar metrics only on rank 0. The old code called
                    # .item() for every component on all 64 ranks even though only
                    # rank 0 writes metrics, creating unnecessary device syncs.
                    if is_main:
                        bb_loss_val = float(bb_loss.detach().float().item())
                        extra_metrics = {
                            k: float(v.detach().float().item())
                            for k, v in bb_losses.items()
                            if k != "backbone_loss" and isinstance(v, torch.Tensor) and v.numel() == 1
                        }
                        # Structured loss keys are arm-specific: Variant A reports
                        # action-token CE, while Variant B reports query MSE.
                        _add_pi05_ki_structured_backbone_metrics(extra_metrics, bb_loss_val)
                    else:
                        bb_loss_val = float("nan")
                    del bb_losses, bb_loss

                    if profile_memory:
                        log_memory_usage(accelerator, global_step, "after_backbone_backward")
                        _reset_peak_memory_stats(accelerator)

                    # ---- Phase 2: expert forward + loss check + backward ----
                    with torch.autocast(
                        device_type=accelerator.device.type,
                        dtype=autocast_dtype,
                        enabled=use_autocast,
                    ):
                        ex_losses = model(observation, actions, phase="expert")
                        ex_loss = ex_losses["expert_loss"]

                    # Cheap cross-rank finiteness check for expert loss
                    # (1 all-reduce instead of 5 from _gather_scalar_stats).
                    ex_all_finite = (
                        _gather_finite_consensus(accelerator, ex_loss)
                        if _loss_finite_check_enabled()
                        else True
                    )
                    ex_nonfinite = not ex_all_finite

                    if ex_nonfinite:
                        # Non-finite detected: compute detailed stats for error diagnostics.
                        ex_loss_stats = _gather_scalar_stats(accelerator, ex_loss)
                        if is_main:
                            logging.error(
                                "Non-finite expert loss at step=%s: "
                                "finite=%d/%d bad_rank=%d min=%s max=%s mean=%s",
                                global_step,
                                int(ex_loss_stats["finite_count"]),
                                int(ex_loss_stats["total_count"]),
                                int(ex_loss_stats["bad_rank"]),
                                ex_loss_stats["min"],
                                ex_loss_stats["max"],
                                ex_loss_stats["mean"],
                            )
                    elif need_detailed_stats:
                        # Log interval or debug mode: compute detailed stats.
                        ex_loss_stats = _gather_scalar_stats(accelerator, ex_loss)

                    if ex_nonfinite:
                        if not torch.isfinite(ex_loss.detach()):
                            _log_nonfinite_batch_state(loss=ex_loss, actions=actions, observation=observation)
                            _save_nonfinite_debug_dump(
                                output_dir=config.log_dir,
                                global_step=global_step,
                                accelerator=accelerator,
                                loss=ex_loss,
                                actions=actions,
                                observation=observation,
                            )
                        consecutive_nonfinite_losses += 1
                        total_nonfinite_loss_batches += 1
                        two_phase_update.clear_gradients(optimizer)
                        accelerator.wait_for_everyone()
                        # Flush metrics before raising / skipping on error.
                        _metrics_buffer_flush()
                        if consecutive_nonfinite_losses >= max_consecutive_nonfinite_losses:
                            raise FloatingPointError(
                                "Too many consecutive non-finite expert losses. "
                                f"Reached {consecutive_nonfinite_losses} skipped batches."
                            )
                        if is_main:
                            logging.warning(
                                "Skipping non-finite expert loss at step=%s; "
                                "consecutive_nonfinite_losses=%s/%s",
                                global_step,
                                consecutive_nonfinite_losses,
                                max_consecutive_nonfinite_losses,
                            )
                        continue

                    # Expert backward accumulates expert gradients (and
                    # flow→backbone gradients with KI=OFF). As the final phase,
                    # it always inherits the actual outer accumulation boundary;
                    # the engine update remains deferred until both graphs have
                    # been freed below.
                    two_phase_update.backward(ex_loss)

                    # Capture scalar metrics only on rank 0; all other ranks keep
                    # the training path free of logging-only .item() synchronizations.
                    if is_main:
                        ex_loss_val = float(ex_loss.detach().float().item())
                        for k, v in ex_losses.items():
                            if k != "expert_loss" and isinstance(v, torch.Tensor) and v.numel() == 1:
                                extra_metrics[k] = float(v.detach().float().item())
                        # Structured expert loss keys shared by both π0.5-KI arms.
                        # loss_expert = weighted expert loss (alpha * flow_loss)
                        # loss_flow_raw = raw flow matching loss (pre-alpha weighting)
                        extra_metrics["loss_expert"] = ex_loss_val
                        extra_metrics["loss_flow_raw"] = extra_metrics.get("flow_loss", float("nan"))
                    else:
                        ex_loss_val = float("nan")
                    del ex_losses, ex_loss

                    # ---- Per-param-group gradient norms (before clipping) ----
                    # NOTE: Under DeepSpeed ZeRO-2, grads are partitioned per-rank.
                    # We use safe_get_local_grad + all-reduce sum-of-squares to
                    # recover the global norm per group.  If gradient data is not
                    # yet available (e.g. before engine.step()), the value is NaN
                    # and the ``_available`` flag is False.
                    # The total grad_norm from accelerator.clip_grad_norm_ is always
                    # the authoritative total.
                    #
                    # DIAGNOSTIC OPTIMIZATION: per-group grad norms are computed
                    # only at log_interval cadence or when debug overflow mode is
                    # enabled, since they require 2 additional all-reduces per step.
                    # The total gradient clipping math (clip_grad_norm_) is unaffected.
                    unwrapped_model = accelerator.unwrap_model(model)
                    bb_params_group = list(unwrapped_model.get_backbone_params())
                    ex_params_group = list(unwrapped_model.get_expert_params())

                    if is_main:
                        # ---- Loss-based expert fraction (logging only) ----
                        total_loss_for_fraction = bb_loss_val + ex_loss_val
                        if total_loss_for_fraction > 0:
                            extra_metrics["expert_loss_fraction"] = ex_loss_val / total_loss_for_fraction
                        else:
                            extra_metrics["expert_loss_fraction"] = 0.0

                    # Per-group grad norms: only compute on sync steps AND at log
                    # cadence (or debug overflow mode).  Pure diagnostics.
                    _compute_group_norms = (
                        accelerator.sync_gradients
                        and (
                            global_step % int(config.log_interval) == 0
                            or _debug_overflow_enabled(config)
                        )
                    )
                    if _compute_group_norms:
                        gn_backbone, bb_gn_available = _compute_param_group_grad_norm(bb_params_group, accelerator)
                        gn_expert, ex_gn_available = _compute_param_group_grad_norm(ex_params_group, accelerator)
                        extra_metrics["grad_norm_backbone"] = gn_backbone
                        extra_metrics["grad_norm_expert"] = gn_expert
                        extra_metrics["grad_norm_backbone_available"] = bb_gn_available
                        extra_metrics["grad_norm_expert_available"] = ex_gn_available
                    elif accelerator.sync_gradients:
                        # Sync step but not at log cadence: mark as not computed this step.
                        extra_metrics["grad_norm_backbone"] = float("nan")
                        extra_metrics["grad_norm_expert"] = float("nan")
                        extra_metrics["grad_norm_backbone_available"] = False
                        extra_metrics["grad_norm_expert_available"] = False
                    else:
                        # Non-sync steps: no meaningful grad norm across ranks
                        extra_metrics["grad_norm_backbone"] = float("nan")
                        extra_metrics["grad_norm_expert"] = float("nan")
                        extra_metrics["grad_norm_backbone_available"] = False
                        extra_metrics["grad_norm_expert_available"] = False

                    # ---- KI heuristic diagnostic (loss-based, always available when losses > 0) ----
                    # This is a *heuristic*, not a proof of KI correctness.
                    #
                    # Idea: the ratio of expert loss magnitude vs total loss
                    # serves as a proxy for how much gradient contribution
                    # the expert phase could potentially make to backbone params.
                    # When KI=ON, flow gradients should not leak to backbone,
                    # so the actual expert→backbone contribution should be near zero
                    # even when this loss ratio is high.
                    #
                    # This is a loss-magnitude proxy for KI, not a direct measurement.
                    # A proper KI verification requires unit tests with
                    # controlled single-phase backwards (see test_ki_integration_*.py).
                    if is_main and accelerator.sync_gradients:
                        total_loss_for_ratio = bb_loss_val + ex_loss_val
                        if total_loss_for_ratio > 0:
                            # Heuristic: expert_loss / (backbone_loss + expert_loss)
                            # When KI=ON, this ratio estimates the maximum possible leak;
                            # actual leak should be near 0.
                            extra_metrics["ki_heuristic_loss_ratio"] = ex_loss_val / total_loss_for_ratio
                        else:
                            extra_metrics["ki_heuristic_loss_ratio"] = 0.0
                    elif is_main:
                        extra_metrics["ki_heuristic_loss_ratio"] = float("nan")

                    # Both phases passed — reset counter
                    consecutive_nonfinite_losses = 0

                    if profile_memory:
                        log_memory_usage(accelerator, global_step, "after_expert_backward")

                    # Combined loss is required only for rank-0 logging and
                    # diagnostics; non-main ranks avoid scalar materialization.
                    loss_for_log = bb_loss_val + ex_loss_val if is_main else float("nan")
                    if is_main:
                        extra_metrics["loss_total"] = loss_for_log
                    # Use expert loss stats as the primary loss_rank_stats for downstream code.
                    # On non-log-interval steps without errors, ex_loss_stats may be None;
                    # supply a stub with all_finite=True (we already passed the cheap check).
                    if ex_loss_stats is not None:
                        loss_rank_stats = ex_loss_stats
                    else:
                        loss_rank_stats = {
                            "all_finite": 1.0,
                            "finite_count": float(accelerator.num_processes),
                            "total_count": float(accelerator.num_processes),
                            "bad_rank": -1.0,
                            "min": float("nan"),
                            "max": float("nan"),
                            "mean": float("nan"),
                            "std": float("nan"),
                        }

                else:
                    # ---- Standard single-forward path (PI0, PI05, VLM2, etc.) ----
                    with torch.autocast(
                        device_type=accelerator.device.type,
                        dtype=autocast_dtype,
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
                                k: v.item()
                                for k, v in losses.items()
                                if k != "loss" and isinstance(v, torch.Tensor) and v.numel() == 1
                            }
                            loss = losses["loss"]
                        elif isinstance(losses, (list, tuple)):
                            loss = torch.stack(list(losses)).mean()
                        elif not isinstance(losses, torch.Tensor):
                            loss = torch.tensor(losses, device=accelerator.device, dtype=torch.float32)
                        else:
                            loss = losses.mean()

                    if _debug_overflow_enabled(config):
                        extra_metrics.update(_loss_tensor_debug_metrics(losses, actions))

                    # Cheap cross-rank finiteness check (1 all-reduce).
                    # Detailed _gather_scalar_stats only at log_interval, on error,
                    # or when debug overflow mode is enabled.
                    _need_detailed_stats = (
                        (accelerator.sync_gradients and global_step % int(config.log_interval) == 0)
                        or _debug_overflow_enabled(config)
                    )
                    loss_all_finite = (
                        _gather_finite_consensus(accelerator, loss)
                        if _loss_finite_check_enabled()
                        else True
                    )
                    any_rank_has_nonfinite_loss = not loss_all_finite

                    if any_rank_has_nonfinite_loss or _need_detailed_stats:
                        # Compute detailed stats for error diagnostics or log cadence.
                        loss_rank_stats = _gather_scalar_stats(accelerator, loss)
                    else:
                        # Lightweight stub: all ranks agreed finite, no stats needed.
                        loss_rank_stats = {
                            "all_finite": 1.0,
                            "finite_count": float(accelerator.num_processes),
                            "total_count": float(accelerator.num_processes),
                            "bad_rank": -1.0,
                            "min": float("nan"),
                            "max": float("nan"),
                            "mean": float("nan"),
                            "std": float("nan"),
                        }

                    if any_rank_has_nonfinite_loss and is_main:
                        logging.error(
                            "Non-finite loss detected on at least one rank at step=%s: "
                            "finite=%d/%d bad_rank=%d min=%s max=%s mean=%s std=%s",
                            global_step,
                            int(loss_rank_stats["finite_count"]),
                            int(loss_rank_stats["total_count"]),
                            int(loss_rank_stats["bad_rank"]),
                            loss_rank_stats["min"],
                            loss_rank_stats["max"],
                            loss_rank_stats["mean"],
                            loss_rank_stats["std"],
                        )

                    if not torch.isfinite(loss.detach()):
                        _log_nonfinite_batch_state(loss=loss, actions=actions, observation=observation)
                        _save_nonfinite_debug_dump(
                            output_dir=config.log_dir,
                            global_step=global_step,
                            accelerator=accelerator,
                            loss=loss,
                            actions=actions,
                            observation=observation,
                        )
                    if any_rank_has_nonfinite_loss:
                        consecutive_nonfinite_losses += 1
                        total_nonfinite_loss_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        accelerator.wait_for_everyone()
                        # Flush metrics before raising on error.
                        _metrics_buffer_flush()
                        if consecutive_nonfinite_losses >= max_consecutive_nonfinite_losses:
                            raise FloatingPointError(
                                "Too many consecutive non-finite losses before backward. "
                                f"Reached {consecutive_nonfinite_losses} skipped batches."
                            )
                        if is_main:
                            logging.warning(
                                "Skipping non-finite-loss batch at global_step=%s; "
                                "consecutive_nonfinite_losses=%s/%s",
                                global_step,
                                consecutive_nonfinite_losses,
                                max_consecutive_nonfinite_losses,
                            )
                        continue

                    consecutive_nonfinite_losses = 0

                    if profile_memory:
                        log_memory_usage(accelerator, global_step, "after_forward")

                    if profile_memory:
                        _reset_peak_memory_stats(accelerator)
                    accelerator.backward(loss)
                    if profile_memory:
                        log_memory_usage(accelerator, global_step, "after_backward")

                    loss_for_log = float(loss.detach().float().item())
                loss_scale = _get_deepspeed_loss_scale(accelerator)
                if accelerator.sync_gradients:
                    if global_step < 5 and is_main and torch.cuda.is_available() and not profile_memory:
                        log_memory_usage(accelerator, global_step, "after_backward")

                    grad_stats_pre = None
                    grad_stats_post = None
                    clip_threshold_override = _env_float("OPENPI_TRAIN_LOOP_GRADIENT_CLIPPING")
                    if clip_threshold_override is None:
                        clip_threshold_override = _env_float("OPENPI_DS_GRADIENT_CLIPPING")
                    default_clip_threshold = (
                        0.5
                        if _fp16_stability_profile_enabled() and config.pytorch_training_precision == "float16"
                        else config.optimizer.clip_gradient_norm
                    )
                    clip_threshold = float(
                        default_clip_threshold if clip_threshold_override is None else clip_threshold_override
                    )
                    deepspeed_two_phase_update = is_pi05_ki_joint and two_phase_update.is_deepspeed
                    if _debug_overflow_enabled(config) and not deepspeed_two_phase_update:
                        grad_stats_pre = _collect_grad_debug_stats(optim_params, accelerator)

                    # DeepSpeed's cached global norm is unavailable before the
                    # first engine step. For two-phase KI, clipping and norm
                    # computation are owned by the single engine.step below.
                    if deepspeed_two_phase_update:
                        grad_norm = None
                    elif is_pi05_ki_joint:
                        grad_norm = two_phase_update.clip_grad_norm_before_step(
                            optim_params, max_norm=clip_threshold
                        )
                    else:
                        grad_norm = accelerator.clip_grad_norm_(optim_params, max_norm=clip_threshold)
                    grad_norm_value = _grad_norm_to_float(grad_norm)
                    if not deepspeed_two_phase_update and not np.isfinite(grad_norm_value):
                        total_nonfinite_grad_updates += 1
                        if accelerator.distributed_type == DistributedType.DEEPSPEED and let_deepspeed_handle_nonfinite_grad:
                            if is_main:
                                logging.warning(
                                    "Non-finite grad_norm at global_step=%s loss=%.6f grad_norm=%s; "
                                    "letting DeepSpeed optimizer.step() handle overflow/loss-scale update "
                                    "instead of pre-step skipping. total_nonfinite_grad_updates=%s",
                                    global_step,
                                    loss_for_log,
                                    grad_norm_value,
                                    total_nonfinite_grad_updates,
                                )
                        else:
                            consecutive_skipped_updates += 1
                            if is_pi05_ki_joint:
                                two_phase_update.clear_gradients(optimizer)
                            else:
                                optimizer.zero_grad(set_to_none=True)
                            accelerator.wait_for_everyone()
                            if is_main:
                                logging.warning(
                                    "Skipping optimizer update because grad_norm is non-finite at global_step=%s "
                                    "loss=%.6f grad_norm=%s consecutive_skipped_updates=%s/%s "
                                    "total_nonfinite_grad_updates=%s",
                                    global_step,
                                    loss_for_log,
                                    grad_norm_value,
                                    consecutive_skipped_updates,
                                    max_consecutive_skipped_updates,
                                    total_nonfinite_grad_updates,
                                )
                            if consecutive_skipped_updates >= max_consecutive_skipped_updates:
                                raise RuntimeError(
                                    "Too many consecutive optimizer updates were skipped due to non-finite gradients. "
                                    f"Reached {consecutive_skipped_updates} skipped updates."
                                )
                            continue

                    if _debug_overflow_enabled(config) and not deepspeed_two_phase_update:
                        grad_stats_post = _collect_grad_debug_stats(optim_params, accelerator)
                        if is_main:
                            clipping_triggered = grad_norm_value > clip_threshold
                            logging.info(
                                "overflow_debug step=%s loss=%.6f loss_scale=%s clip_threshold=%.4f "
                                "loss_all_min=%.6f loss_all_max=%.6f loss_all_mean=%.6f loss_all_std=%.6f "
                                "loss_all_finite=%d/%d "
                                "per_sample_loss_max=%.6f per_sample_loss_mean=%.6f per_sample_loss_std=%.6f "
                                "target_action_abs_max=%.6f "
                                "grad_norm_pre=%.6f grad_norm_api=%.6f grad_norm_post=%.6f "
                                "max_abs_pre=%.6f max_abs_post=%.6f nonfinite_pre=%d nonfinite_post=%d "
                                "finite_ratio_pre=%.8f finite_ratio_post=%.8f grad_tensors=%.0f clipped=%s",
                                global_step,
                                loss_for_log,
                                f"{loss_scale:.1f}" if loss_scale is not None else "n/a",
                                clip_threshold,
                                loss_rank_stats["min"],
                                loss_rank_stats["max"],
                                loss_rank_stats["mean"],
                                loss_rank_stats["std"],
                                int(loss_rank_stats["finite_count"]),
                                int(loss_rank_stats["total_count"]),
                                extra_metrics.get("per_sample_loss_max", float("nan")),
                                extra_metrics.get("per_sample_loss_mean", float("nan")),
                                extra_metrics.get("per_sample_loss_std", float("nan")),
                                extra_metrics.get("target_action_abs_max", float("nan")),
                                grad_stats_pre["global_norm"],
                                grad_norm_value,
                                grad_stats_post["global_norm"],
                                grad_stats_pre["max_abs_grad"],
                                grad_stats_post["max_abs_grad"],
                                int(grad_stats_pre["nonfinite_count"]),
                                int(grad_stats_post["nonfinite_count"]),
                                grad_stats_pre["finite_ratio"],
                                grad_stats_post["finite_ratio"],
                                grad_stats_pre["grad_tensors"],
                                clipping_triggered,
                            )
                            if grad_stats_pre["nonfinite_count"] > 0 or grad_stats_post["nonfinite_count"] > 0:
                                logging.warning(
                                    "overflow_debug detected non-finite gradients at step=%s (pre=%d post=%d)",
                                    global_step,
                                    int(grad_stats_pre["nonfinite_count"]),
                                    int(grad_stats_post["nonfinite_count"]),
                                )
                            if loss_scale is not None and loss_scale <= 1.0:
                                logging.warning("overflow_debug loss scale collapsed to %.1f at step=%s", loss_scale, global_step)
                    if profile_memory:
                        _reset_peak_memory_stats(accelerator)
                    if is_pi05_ki_joint:
                        post_step_grad_norm = two_phase_update.step_and_zero_grad(optimizer)
                        if deepspeed_two_phase_update:
                            # DeepSpeed populates this cache inside
                            # `_take_model_step`, after its clipping/optimizer
                            # logic and before it clears the partitioned grads.
                            grad_norm_value = _grad_norm_to_float(post_step_grad_norm)
                            if not np.isfinite(grad_norm_value):
                                total_nonfinite_grad_updates += 1
                                if is_main:
                                    logging.warning(
                                        "DeepSpeed reported a non-finite or unavailable post-step grad_norm at "
                                        "global_step=%s loss=%.6f grad_norm=%s; the engine already handled "
                                        "clipping/overflow. total_nonfinite_grad_updates=%s",
                                        global_step,
                                        loss_for_log,
                                        grad_norm_value,
                                        total_nonfinite_grad_updates,
                                    )
                    else:
                        optimizer.step()
                    step_was_skipped = accelerator.optimizer_step_was_skipped
                    if profile_memory:
                        log_memory_usage(accelerator, global_step, "after_optimizer_step")
                    if not is_pi05_ki_joint:
                        optimizer.zero_grad(set_to_none=True)

                    if step_was_skipped:
                        consecutive_skipped_updates += 1
                        total_ds_overflow_skipped_updates += 1
                        if is_main:
                            logging.warning(
                                "Optimizer step skipped due to overflow at global_step=%s loss=%.6f "
                                "loss_scale=%s consecutive_skipped_updates=%s/%s total_ds_overflow_skipped_updates=%s "
                                "total_nonfinite_loss_batches=%s total_nonfinite_grad_updates=%s",
                                global_step,
                                loss_for_log,
                                f"{loss_scale:.1f}" if loss_scale is not None else "n/a",
                                consecutive_skipped_updates,
                                max_consecutive_skipped_updates,
                                total_ds_overflow_skipped_updates,
                                total_nonfinite_loss_batches,
                                total_nonfinite_grad_updates,
                            )
                        if consecutive_skipped_updates >= max_consecutive_skipped_updates:
                            raise RuntimeError(
                                "Too many consecutive optimizer steps were skipped due to overflow. "
                                f"Reached {consecutive_skipped_updates} skipped updates."
                            )
                        continue

                    consecutive_skipped_updates = 0
                    sample_progress.record_update(committed=True)

                    # stats/logging use optimizer-step granularity
                    if is_main:
                        info_dict = {
                            "loss": loss_for_log,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "grad_norm": grad_norm_value,
                            "grad_norm_total": grad_norm_value,
                            **sample_progress.metrics(),
                            **extra_metrics,
                        }
                        # Per-param-group LRs for π0.5-KI joint query model.
                        if is_pi05_ki_joint:
                            for pg in optimizer.param_groups:
                                name = pg.get("name")
                                if name:
                                    info_dict[f"lr_{name}"] = float(pg["lr"])
                        infos.append(info_dict)
                        if loss_scale is not None:
                            infos[-1]["loss_scale"] = loss_scale
                            # Always warn on dangerously low loss scale (regardless of debug mode)
                            if loss_scale < 1.0:
                                logging.warning(
                                    "loss_scale=%.2e at step=%s — training may be numerically unstable",
                                    loss_scale, global_step,
                                )

                        # Write per-step metrics to metrics.jsonl (rank 0 only).
                        # Flattened info_dict + step + epoch for offline analysis.
                        if _metrics_file is not None:
                            _epoch = (global_step // steps_per_epoch) + 1
                            _record = {
                                "step": int(global_step),
                                "epoch": int(_epoch),
                                **info_dict,
                            }
                            # Buffered: flushed at log_interval / val / checkpoint boundaries.
                            _metrics_buffer_append(_record)

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
                        avg_loss_scale = None
                        loss_scale_values = [info["loss_scale"] for info in infos if "loss_scale" in info]
                        if loss_scale_values:
                            avg_loss_scale = sum(loss_scale_values) / len(loss_scale_values)

                        # π0.5-KI joint query: per-component loss averages for log line
                        avg_bb_loss = None
                        avg_ex_loss = None
                        avg_ce_loss = None
                        avg_flow_raw = None
                        if is_pi05_ki_joint:
                            bb_vals = [info["loss_backbone"] for info in infos if "loss_backbone" in info]
                            ex_vals = [info["loss_expert"] for info in infos if "loss_expert" in info]
                            ce_vals = [info["loss_ce"] for info in infos if "loss_ce" in info]
                            flow_vals = [info["loss_flow_raw"] for info in infos if "loss_flow_raw" in info]
                            if bb_vals:
                                avg_bb_loss = sum(bb_vals) / len(bb_vals)
                            if ex_vals:
                                avg_ex_loss = sum(ex_vals) / len(ex_vals)
                            if ce_vals:
                                avg_ce_loss = sum(ce_vals) / len(ce_vals)
                            if flow_vals:
                                avg_flow_raw = sum(flow_vals) / len(flow_vals)

                        if is_pi05_ki_joint and avg_bb_loss is not None and avg_ex_loss is not None:
                            logging.info(
                                "step=%s epoch=%s epoch_step=%s/%s loss=%.4f (bb=%.4f ex=%.4f ce=%.4f flow=%.4f) "
                                "lr=%.2e grad_norm=%.2f loss_scale=%s time=%.1fs",
                                global_step,
                                epoch,
                                epoch_step,
                                steps_per_epoch,
                                avg_loss,
                                avg_bb_loss,
                                avg_ex_loss,
                                avg_ce_loss if avg_ce_loss is not None else float("nan"),
                                avg_flow_raw if avg_flow_raw is not None else float("nan"),
                                avg_lr,
                                avg_grad_norm,
                                f"{avg_loss_scale:.1f}" if avg_loss_scale is not None else "n/a",
                                elapsed,
                            )
                        else:
                            logging.info(
                                "step=%s epoch=%s epoch_step=%s/%s loss=%.4f lr=%.2e grad_norm=%.2f loss_scale=%s time=%.1fs",
                                global_step,
                                epoch,
                                epoch_step,
                                steps_per_epoch,
                                avg_loss,
                                avg_lr,
                                avg_grad_norm,
                                f"{avg_loss_scale:.1f}" if avg_loss_scale is not None else "n/a",
                                elapsed,
                            )

                        if is_main and config.wandb_enabled and len(infos) > 0:
                            try:
                                wandb = _get_wandb()
                                log_payload: dict[str, float] = {
                                    "loss": avg_loss,
                                    "loss/total": avg_loss,
                                    "learning_rate": avg_lr,
                                    "grad_norm": avg_grad_norm,
                                    "grad_norm/total": avg_grad_norm,
                                    "step": float(global_step),
                                    "epoch": float(epoch),
                                    "epoch_step": float(epoch_step),
                                    "steps_per_epoch": float(steps_per_epoch),
                                    "time_per_step": elapsed / max(1, int(config.log_interval)),
                                    **sample_progress.metrics(),
                                }
                                if avg_loss_scale is not None:
                                    log_payload["loss_scale"] = avg_loss_scale

                                # --- π0.5-KI structured metrics ---
                                if is_pi05_ki_joint:
                                    # Loss components use arm-correct objective names.
                                    _update_pi05_ki_wandb_loss_metrics(log_payload, infos)

                                    # Per-param-group LRs
                                    for pg_name in ("backbone", "expert"):
                                        lr_key = f"lr_{pg_name}"
                                        vals = [info[lr_key] for info in infos if lr_key in info]
                                        if vals:
                                            log_payload[f"lr/{pg_name}"] = sum(vals) / len(vals)

                                    # Per-param-group gradient norms
                                    for pg_name in ("backbone", "expert"):
                                        gn_key = f"grad_norm_{pg_name}"
                                        vals = [info[gn_key] for info in infos if gn_key in info and np.isfinite(info[gn_key])]
                                        if vals:
                                            log_payload[f"grad_norm/{pg_name}"] = sum(vals) / len(vals)

                                    # KI heuristic diagnostic (clearly labeled as heuristic)
                                    ki_key = "ki_heuristic_loss_ratio"
                                    ki_vals = [info[ki_key] for info in infos if ki_key in info and np.isfinite(info[ki_key])]
                                    if ki_vals:
                                        log_payload["ki/heuristic_loss_ratio"] = sum(ki_vals) / len(ki_vals)

                                # Backward-compat: legacy subtask/ prefix
                                for metric_key in ("flow_loss", "ce_loss"):
                                    vals = [info[metric_key] for info in infos if metric_key in info]
                                    if vals:
                                        log_payload[f"subtask/{metric_key}"] = sum(vals) / len(vals)

                                wandb.log(log_payload, step=global_step)
                            except Exception:
                                # Formal mode requires init/auth consensus only; individual
                                # runtime log calls intentionally remain best-effort.
                                logging.warning("wandb log failed; continuing without wandb", exc_info=True)
                                try:
                                    object.__setattr__(config, "wandb_enabled", False)
                                except Exception:
                                    pass

                        start_time = time.time()
                        infos = []

                    current_step = global_step + 1

                    # Flush buffered metrics.jsonl at log_interval boundaries.
                    if accelerator.sync_gradients and global_step % int(config.log_interval) == 0:
                        _metrics_buffer_flush()

                    # Flush only when a checkpoint is actually due. ``save_checkpoint``
                    # is invoked every optimizer step but internally uses the same
                    # policy helper to no-op on non-checkpoint steps.
                    if _checkpoint_save_kind(
                        config,
                        global_step=current_step,
                        steps_per_epoch=steps_per_epoch,
                    ) is not None:
                        _metrics_buffer_flush()

                    # checkpoint/save + progress bar update
                    save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        global_step=current_step,
                        config=config,
                        data_config=data_config,
                        data_manifest=_data_manifest,
                        steps_per_epoch=steps_per_epoch,
                        sample_progress=sample_progress,
                    )

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "loss": f"{loss_for_log:.4f}",
                                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                                "step": current_step,
                            }
                        )

                    global_step = current_step

                    # ---- Validation (at val_log_interval, after optimizer step) ----
                    if val_loader is not None and global_step % int(config.val_log_interval) == 0 and global_step > 0:
                        # Flush training metrics so file is consistent before validation writes.
                        _metrics_buffer_flush()
                        # `flow_l1` (Euler-integrated ACTION error) is the offline
                        # metric that tracks real policy performance, so allow it
                        # on a cadence instead of only at epoch end / final.
                        _slow_every = int(getattr(config, "val_slow_metrics_every", 0) or 0)
                        _val_idx = global_step // int(config.val_log_interval)
                        _slow_now = _slow_every > 0 and (_val_idx % _slow_every == 0)
                        run_validation(
                            accelerator=accelerator,
                            model=model,
                            val_loader=val_loader,
                            config=config,
                            global_step=global_step,
                            steps_per_epoch=steps_per_epoch,
                            is_pi05_ki_joint=is_pi05_ki_joint,
                            use_vlm2=use_vlm2,
                            use_autocast=use_autocast,
                            autocast_dtype=autocast_dtype,
                            metrics_file=_metrics_file,
                            sample_progress=sample_progress,
                            slow_metrics=_slow_now,
                        )

                    # ---- Epoch-end validation with slow metrics (flow_l1) ----
                    # Restricted to epoch-expressed budgets, and skipped on the
                    # final step because the post-loop "final validation" below
                    # already covers it with slow metrics.
                    #
                    # Both conditions preserve pre-refactor behaviour exactly.
                    # The old guard was `not formal_b1k_mode`, which suppressed
                    # this for ALL formal configs. The legacy fixed-budget formal
                    # configs (on_bf16 / on_v100_fp32) have num_train_epochs=None
                    # and so are still suppressed by the first condition; the
                    # single-epoch H20 formal run's only boundary is its last
                    # step and so is suppressed by the second. Multi-epoch
                    # non-formal runs (e.g. a100, 4 epochs) keep firing at
                    # epochs 1..n-1 as before.
                    if (
                        val_loader is not None
                        and config.num_train_epochs is not None
                        and steps_per_epoch > 0
                        and global_step % steps_per_epoch == 0
                        and global_step > 0
                        and global_step < int(config.num_train_steps)
                    ):
                        # Flush training metrics before epoch-end validation writes.
                        _metrics_buffer_flush()
                        run_validation(
                            accelerator=accelerator,
                            model=model,
                            val_loader=val_loader,
                            config=config,
                            global_step=global_step,
                            steps_per_epoch=steps_per_epoch,
                            is_pi05_ki_joint=is_pi05_ki_joint,
                            use_vlm2=use_vlm2,
                            use_autocast=use_autocast,
                            autocast_dtype=autocast_dtype,
                            metrics_file=_metrics_file,
                            sample_progress=sample_progress,
                            slow_metrics=True,
                            val_label="val_epoch_end",
                        )

    _close_training_iterator(train_iterator)

    # Final validation at end of training (if val data configured)
    # Includes slow metrics (flow_l1) for final evaluation.
    if val_loader is not None:
        # Flush training metrics before final validation writes.
        _metrics_buffer_flush()
        run_validation(
            accelerator=accelerator,
            model=model,
            val_loader=val_loader,
            config=config,
            global_step=int(config.num_train_steps),
            steps_per_epoch=steps_per_epoch,
            is_pi05_ki_joint=is_pi05_ki_joint,
            use_vlm2=use_vlm2,
            use_autocast=use_autocast,
            autocast_dtype=autocast_dtype,
            metrics_file=_metrics_file,
            sample_progress=sample_progress,
            slow_metrics=True,
            val_label="val_final",
        )

    if pbar is not None:
        pbar.close()
    # Close metrics.jsonl file (rank 0 only).
    # Flushes pending buffer before closing.
    if _metrics_file is not None:
        _metrics_buffer_close()
    if is_main and config.wandb_enabled:
        try:
            wandb = _get_wandb()
            wandb.finish()
        except Exception:
            logging.warning("wandb finish failed", exc_info=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            logging.warning("destroy_process_group failed during shutdown", exc_info=True)


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

    # Import OpenPI modules lazily to keep this file spawn-safe for DataLoader workers.
    global _config, _data_loader, _model, _normalize
    if _config is None:
        import openpi.models.model as _model_mod
        import openpi.shared.normalize as _normalize_mod
        import openpi.training.config as _config_mod
        import openpi.training.data_loader as _data_loader_mod

        importlib.import_module("openpi.models_pytorch.pi0_hamlet")
        importlib.import_module("openpi.models_pytorch.pi0_memoryvla")
        importlib.import_module("openpi.models_pytorch.pi0_pytorch")
        importlib.import_module("openpi.models_pytorch.pi05_subtask")

        _model = _model_mod
        _normalize = _normalize_mod
        _config = _config_mod
        _data_loader = _data_loader_mod

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
