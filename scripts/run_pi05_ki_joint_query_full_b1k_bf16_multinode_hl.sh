#!/usr/bin/env bash
set -euo pipefail

# π0.5-KI joint-query full B1K task-set BF16 training for Merlin/Arnold on HL.
# Contract: every B1K challenge task (no tasks filter), per-task episodes 0-179
# train / 180-199 validation, KI enabled, five complete epochs with no step cap,
# validation every 1000 steps, durable checkpoint at every epoch end, one rolling
# recovery checkpoint every 10000 steps, batch size 4 per GPU (global 128 on 32
# GPUs), Accelerate BF16 + DeepSpeed ZeRO-2 with no optimizer offload.

# ---- Cache mode flags (mutually exclusive) ----
# PREPARE_HF_CACHE_ONLY=1: build train+val Arrow cache then exit (no training)
# FORCE_LOAD_CACHE=1:     require existing cache; fail fast if missing/incomplete
# Both unset:             normal training (build cache if missing, reuse if present)
PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"
FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-0}"

if [[ "${PREPARE_HF_CACHE_ONLY}" == "1" && "${FORCE_LOAD_CACHE}" == "1" ]]; then
  echo "ERROR: PREPARE_HF_CACHE_ONLY=1 and FORCE_LOAD_CACHE=1 are mutually exclusive." >&2
  echo "  PREPARE_HF_CACHE_ONLY=1 builds the cache from scratch." >&2
  echo "  FORCE_LOAD_CACHE=1 requires an already-built cache and fails if missing." >&2
  echo "Use only one of these flags, or neither for normal mode." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME

CONFIG_NAME="${CONFIG_NAME:-pi05_ki_joint_query_b1k-full_task-ki_on_bf16}"
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/${CONFIG_NAME}}"

# Managed job/task IDs are shared by all nodes in one run, while /tmp itself is
# node-local. Manual runs must provide a rank-consistent OPENPI_HF_CACHE_RUN_ID,
# or opt into explicit cross-job reuse with an absolute LOCAL_CACHE_ROOT plus
# FORCE_LOAD_CACHE=1. Reusing a cache across jobs is intentionally read-only.
# Component-level cache variables injected by Merlin are replaced below so they
# cannot redirect Arrow/Triton/temp I/O back to NAS/home.
_MANAGED_CACHE_RUN_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-}}"
_EXPLICIT_CACHE_RUN_ID="${OPENPI_HF_CACHE_RUN_ID:-}"
_EXPLICIT_LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-}"
if [[ -n "${_EXPLICIT_CACHE_RUN_ID}" && -z "${_EXPLICIT_CACHE_RUN_ID//[[:space:]]/}" ]]; then
  echo "ERROR: OPENPI_HF_CACHE_RUN_ID must contain non-whitespace characters." >&2
  exit 1
fi
if [[ -n "${_EXPLICIT_CACHE_RUN_ID}" ]]; then
  CACHE_RUN_ID="${_EXPLICIT_CACHE_RUN_ID}"
elif [[ -n "${_MANAGED_CACHE_RUN_ID}" ]]; then
  CACHE_RUN_ID="${_MANAGED_CACHE_RUN_ID}"
elif [[ -n "${_EXPLICIT_LOCAL_CACHE_ROOT}" ]]; then
  if [[ "${FORCE_LOAD_CACHE}" != "1" ]]; then
    echo "ERROR: manual LOCAL_CACHE_ROOT reuse requires FORCE_LOAD_CACHE=1." >&2
    echo "For a new manual run, set a unique non-empty OPENPI_HF_CACHE_RUN_ID instead." >&2
    exit 1
  fi
  if [[ "${_EXPLICIT_LOCAL_CACHE_ROOT}" != /* ]]; then
    echo "ERROR: LOCAL_CACHE_ROOT must be an absolute node-local path: ${_EXPLICIT_LOCAL_CACHE_ROOT}" >&2
    exit 1
  fi
  _cache_root_digest="$(printf '%s' "${_EXPLICIT_LOCAL_CACHE_ROOT}" | sha256sum)"
  _cache_root_digest="${_cache_root_digest%% *}"
  CACHE_RUN_ID="manual-cache-root-${_cache_root_digest:0:24}"
else
  echo "ERROR: manual runs require a unique non-empty OPENPI_HF_CACHE_RUN_ID." >&2
  echo "For explicit cross-job reuse, set absolute LOCAL_CACHE_ROOT with FORCE_LOAD_CACHE=1." >&2
  exit 1
fi
export OPENPI_HF_CACHE_RUN_ID="${CACHE_RUN_ID}"
CACHE_RUN_KEY="${CACHE_RUN_ID//\//_}"
CACHE_RUN_KEY="${CACHE_RUN_KEY//:/_}"
CACHE_RUN_KEY="${CACHE_RUN_KEY//,/_}"
CACHE_RUN_KEY="${CACHE_RUN_KEY// /_}"
LOCAL_CACHE_ROOT="${_EXPLICIT_LOCAL_CACHE_ROOT:-/tmp/openpi-comet/${USER:-tiger}/${CONFIG_NAME}/${CACHE_RUN_KEY}}"
if [[ "${LOCAL_CACHE_ROOT}" != /* ]]; then
  echo "ERROR: LOCAL_CACHE_ROOT must be an absolute node-local path: ${LOCAL_CACHE_ROOT}" >&2
  exit 1
fi

# The no-training test mode skips conda and all model/data/dependency checks, but
# runs the exact same local-cache binding and directory validation as a real job.
if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" != "1" ]]; then
  CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/navigation-hl/mlx/users/chenjunting/miniconda3}"
  CONDA_ENV="${CONDA_ENV:-openpi-comet-nas}"
  CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: conda initialization script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Bind every bulk cache authoritatively *after* conda activation so neither the
# task environment nor a conda activation hook can restore an inherited path.
export OPENPI_DATA_HOME="${LOCAL_CACHE_ROOT}/openpi"
export HF_HOME="${LOCAL_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_XET_CACHE="${HF_HOME}/xet"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TRITON_CACHE_DIR="${LOCAL_CACHE_ROOT}/triton/autotune"
export XDG_CACHE_HOME="${LOCAL_CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${LOCAL_CACHE_ROOT}/matplotlib"
export TORCH_HOME="${LOCAL_CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_HOME}/inductor"
export TORCH_EXTENSIONS_DIR="${TORCH_HOME}/extensions"
LOCAL_TMP_BACKING="${LOCAL_CACHE_ROOT}/tmp"
# Despite its historical name, value 1 selects one shared Arrow cache per node
# (node<N>/), not one copy per GPU rank. Local rank 0 builds while node peers
# poll atomic ready/failure markers; no NCCL collective waits on Arrow I/O.
export OPENPI_HF_DATASETS_CACHE_PER_RANK=1
export OPENPI_HF_LOCAL_SYNC_TIMEOUT_S="${OPENPI_HF_LOCAL_SYNC_TIMEOUT_S:-7200}"
export OPENPI_HF_LOCAL_SYNC_POLL_S="${OPENPI_HF_LOCAL_SYNC_POLL_S:-2}"

# HF_DATASETS_CACHE is intentionally absent from all launcher filesystem
# preflights. Strict existence validation or normal mkdir first occurs inside
# load_hf_dataset after generation-scoped c10d failure coordination is active.
_LOCAL_CACHE_DIRS=(
  "${LOCAL_CACHE_ROOT}"
  "${OPENPI_DATA_HOME}"
  "${HF_HUB_CACHE}"
  "${HF_MODULES_CACHE}"
  "${HF_ASSETS_CACHE}"
  "${HF_XET_CACHE}"
  "${TRANSFORMERS_CACHE}"
  "${TRITON_CACHE_DIR}"
  "${XDG_CACHE_HOME}"
  "${MPLCONFIGDIR}"
  "${TORCH_HOME}"
  "${TORCHINDUCTOR_CACHE_DIR}"
  "${TORCH_EXTENSIONS_DIR}"
  "${LOCAL_TMP_BACKING}"
)
mkdir -p "${_LOCAL_CACHE_DIRS[@]}"
for _local_cache_dir in "${_LOCAL_CACHE_DIRS[@]}"; do
  if [[ ! -d "${_local_cache_dir}" || ! -w "${_local_cache_dir}" ]]; then
    echo "ERROR: local cache directory is not writable: ${_local_cache_dir}" >&2
    exit 1
  fi
done
_cache_write_probe="${LOCAL_CACHE_ROOT}/.cache_preflight.$$"
if ! (umask 077 && : > "${_cache_write_probe}"); then
  echo "ERROR: failed to create local cache write probe: ${_cache_write_probe}" >&2
  exit 1
fi
rm -f "${_cache_write_probe}"

# Python's multiprocess.Manager appends pymp-*/listener-* below TMPDIR. Keep the
# storage inside LOCAL_CACHE_ROOT, but expose it through a short node-local alias
# so the AF_UNIX pathname remains below Linux sockaddr_un.sun_path's 107 usable
# bytes. A 96-bit SHA-256 prefix makes aliases collision-safe without embedding
# the potentially long config/job key in the socket pathname.
_tmp_alias_digest="$(printf '%s' "${LOCAL_CACHE_ROOT}" | sha256sum)"
_tmp_alias_digest="${_tmp_alias_digest%% *}"
TMP_ALIAS="/tmp/openpi-tmp-${UID:-$(id -u)}-${_tmp_alias_digest:0:24}"
if [[ -e "${TMP_ALIAS}" && ! -L "${TMP_ALIAS}" ]]; then
  echo "ERROR: refusing to replace non-symlink TMP alias path: ${TMP_ALIAS}" >&2
  exit 1
fi
if [[ -L "${TMP_ALIAS}" ]]; then
  _tmp_alias_target="$(readlink -- "${TMP_ALIAS}")"
  if [[ "${_tmp_alias_target}" != "${LOCAL_TMP_BACKING}" ]]; then
    rm -f -- "${TMP_ALIAS}"
  fi
fi
if [[ ! -L "${TMP_ALIAS}" ]]; then
  if ! ln -s -- "${LOCAL_TMP_BACKING}" "${TMP_ALIAS}"; then
    if [[ ! -L "${TMP_ALIAS}" || "$(readlink -- "${TMP_ALIAS}")" != "${LOCAL_TMP_BACKING}" ]]; then
      echo "ERROR: failed to create TMP alias ${TMP_ALIAS} -> ${LOCAL_TMP_BACKING}" >&2
      exit 1
    fi
  fi
fi
if [[ "$(readlink -- "${TMP_ALIAS}")" != "${LOCAL_TMP_BACKING}" ]]; then
  echo "ERROR: TMP alias target mismatch: ${TMP_ALIAS}" >&2
  exit 1
fi
export TMPDIR="${TMP_ALIAS}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
if [[ ! -d "${TMPDIR}" || ! -w "${TMPDIR}" ]]; then
  echo "ERROR: short TMPDIR alias is not writable: ${TMPDIR}" >&2
  exit 1
fi

# DeepSpeed's Triton initialization runs `df` on TRITON_CACHE_DIR. Validate the
# already-created exact directory now so startup cannot emit the previous ENOENT.
if ! df -P "${TRITON_CACHE_DIR}" >/dev/null; then
  echo "ERROR: df preflight failed for TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" >&2
  exit 1
fi

# Exercise the exact lightweight primitive used by datasets Arrow preparation.
# This intentionally imports neither the trainer nor any model/data module.
python - \
  "${TMPDIR}" \
  "${LOCAL_TMP_BACKING}" \
  "${OPENPI_HF_LOCAL_SYNC_TIMEOUT_S}" \
  "${OPENPI_HF_LOCAL_SYNC_POLL_S}" <<'PY'
import math
import os
from pathlib import Path
import sys
import tempfile

import multiprocess

expected_tmpdir = Path(sys.argv[1])
expected_backing = Path(sys.argv[2])
try:
    local_sync_timeout_s = float(sys.argv[3])
    local_sync_poll_s = float(sys.argv[4])
except ValueError as exc:
    raise SystemExit(
        "ERROR: OPENPI_HF_LOCAL_SYNC_TIMEOUT_S and OPENPI_HF_LOCAL_SYNC_POLL_S "
        "must be positive finite numbers"
    ) from exc
if not math.isfinite(local_sync_timeout_s) or local_sync_timeout_s <= 0:
    raise SystemExit(
        "ERROR: OPENPI_HF_LOCAL_SYNC_TIMEOUT_S must be a positive finite number: "
        f"{sys.argv[3]}"
    )
if not math.isfinite(local_sync_poll_s) or local_sync_poll_s <= 0:
    raise SystemExit(
        "ERROR: OPENPI_HF_LOCAL_SYNC_POLL_S must be a positive finite number: "
        f"{sys.argv[4]}"
    )
if local_sync_poll_s > local_sync_timeout_s:
    raise SystemExit(
        "ERROR: OPENPI_HF_LOCAL_SYNC_POLL_S must not exceed "
        "OPENPI_HF_LOCAL_SYNC_TIMEOUT_S: "
        f"{local_sync_poll_s} > {local_sync_timeout_s}"
    )

actual_tmpdir = Path(tempfile.gettempdir())
if actual_tmpdir != expected_tmpdir:
    raise SystemExit(
        f"ERROR: tempfile.gettempdir()={actual_tmpdir} does not equal intended short alias {expected_tmpdir}"
    )
if actual_tmpdir.resolve() != expected_backing.resolve():
    raise SystemExit(
        f"ERROR: short TMPDIR {actual_tmpdir} does not resolve to backing directory {expected_backing}"
    )
if not os.access(actual_tmpdir, os.W_OK):
    raise SystemExit(f"ERROR: tempfile.gettempdir() is not writable: {actual_tmpdir}")

# Linux sockaddr_un.sun_path has 108 bytes including its trailing NUL.
af_unix_path_max_bytes = 107
representative_socket = actual_tmpdir / "pymp-12345678" / "listener-12345678"
representative_socket_bytes = len(os.fsencode(representative_socket))
if representative_socket_bytes > af_unix_path_max_bytes:
    raise SystemExit(
        "ERROR: representative multiprocess Manager socket path is too long: "
        f"{representative_socket_bytes} > {af_unix_path_max_bytes}: {representative_socket}"
    )

with multiprocess.Manager() as manager:
    probe = manager.dict()
    probe["ready"] = True
    if probe.get("ready") is not True:
        raise SystemExit("ERROR: multiprocess.Manager proxy round-trip failed")
    manager_socket = Path(os.fsdecode(manager._address))
    manager_socket_bytes = len(os.fsencode(manager_socket))
    if manager_socket_bytes > af_unix_path_max_bytes:
        raise SystemExit(
            "ERROR: live multiprocess Manager socket path is too long: "
            f"{manager_socket_bytes} > {af_unix_path_max_bytes}: {manager_socket}"
        )

print(f"TEMPFILE_GETTEMPDIR={actual_tmpdir}")
print(f"TMPDIR_REALPATH={actual_tmpdir.resolve()}")
print(f"MANAGER_SOCKET_REPRESENTATIVE_BYTES={representative_socket_bytes}")
print(f"AF_UNIX_PATH_MAX_BYTES={af_unix_path_max_bytes}")
print(f"MULTIPROCESS_MANAGER_SOCKET_BYTES={manager_socket_bytes}")
print(f"OPENPI_HF_LOCAL_SYNC_TIMEOUT_S={local_sync_timeout_s:g}")
print(f"OPENPI_HF_LOCAL_SYNC_POLL_S={local_sync_poll_s:g}")
print("HF_LOCAL_SYNC_PREFLIGHT_OK")
print("MULTIPROCESS_MANAGER_PREFLIGHT_OK")
PY

print_local_cache_paths() {
  printf '%s\n' \
    "LOCAL_CACHE_ROOT=${LOCAL_CACHE_ROOT}" \
    "OPENPI_DATA_HOME=${OPENPI_DATA_HOME}" \
    "HF_HOME=${HF_HOME}" \
    "HF_HUB_CACHE=${HF_HUB_CACHE}" \
    "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}" \
    "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
    "HF_MODULES_CACHE=${HF_MODULES_CACHE}" \
    "HF_ASSETS_CACHE=${HF_ASSETS_CACHE}" \
    "HF_XET_CACHE=${HF_XET_CACHE}" \
    "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
    "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
    "MPLCONFIGDIR=${MPLCONFIGDIR}" \
    "TORCH_HOME=${TORCH_HOME}" \
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}" \
    "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}" \
    "LOCAL_TMP_BACKING=${LOCAL_TMP_BACKING}" \
    "TMP_ALIAS=${TMP_ALIAS}" \
    "TMPDIR=${TMPDIR}" \
    "TMP=${TMP}" \
    "TEMP=${TEMP}" \
    "OPENPI_HF_DATASETS_CACHE_PER_RANK=${OPENPI_HF_DATASETS_CACHE_PER_RANK}" \
    "OPENPI_HF_CACHE_RUN_ID=${OPENPI_HF_CACHE_RUN_ID}" \
    "OPENPI_HF_LOCAL_SYNC_TIMEOUT_S=${OPENPI_HF_LOCAL_SYNC_TIMEOUT_S}" \
    "OPENPI_HF_LOCAL_SYNC_POLL_S=${OPENPI_HF_LOCAL_SYNC_POLL_S}" \
    "PERSISTENT_OUTPUT_ROOT=${PERSISTENT_OUTPUT_ROOT}"
}

if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  print_local_cache_paths
  echo "LOCAL_CACHE_PREFLIGHT_OK; distributed launch skipped"
  exit 0
fi

BASE_PI05_CKPT="${BASE_PI05_CKPT:-${REPO_ROOT}/checkpoints/pi05_base_pytorch}"
B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/}"
# Keep model and normalization assets coherent when BASE_PI05_CKPT is overridden.
B1K_ASSETS_DIR="${BASE_PI05_CKPT}/assets/behavior-1k/2025-challenge-demos"
NORM_STATS_PATH="${B1K_ASSETS_DIR}/norm_stats.json"
ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/configs/accelerate_ds_zero2.yaml}"
DEEPSPEED_CONFIG="${REPO_ROOT}/configs/deepspeed_zero2.json"
TRAINER="${TRAINER:-${REPO_ROOT}/scripts/train_accelerate.py}"

# Offline OpenPI tokenizer bootstrap. The model itself stays on shared NAS, while
# its small metadata is copied locally for offline consumers that expect a cache.
REPO_OPENPI_CACHE="${REPO_OPENPI_CACHE:-${REPO_ROOT}/.cache/openpi}"
TOKENIZER_REL="big_vision/paligemma_tokenizer.model"
TOKENIZER_SOURCE="${REPO_OPENPI_CACHE}/${TOKENIZER_REL}"
TOKENIZER_LOCAL="${OPENPI_DATA_HOME}/${TOKENIZER_REL}"
if [[ ! -s "${TOKENIZER_LOCAL}" ]]; then
  if [[ ! -s "${TOKENIZER_SOURCE}" ]]; then
    echo "ERROR: offline tokenizer cache is missing: ${TOKENIZER_SOURCE}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${TOKENIZER_LOCAL}")"
  _tokenizer_tmp="${TOKENIZER_LOCAL}.tmp.$$"
  cp -f "${TOKENIZER_SOURCE}" "${_tokenizer_tmp}"
  mv -f "${_tokenizer_tmp}" "${TOKENIZER_LOCAL}"
fi
LOCAL_MODEL_METADATA_DIR="${HF_HOME}/local-models/pi05_base_pytorch"
mkdir -p "${LOCAL_MODEL_METADATA_DIR}"
if [[ -f "${BASE_PI05_CKPT}/config.json" && ! -s "${LOCAL_MODEL_METADATA_DIR}/config.json" ]]; then
  _model_config_tmp="${LOCAL_MODEL_METADATA_DIR}/config.json.tmp.$$"
  cp -f "${BASE_PI05_CKPT}/config.json" "${_model_config_tmp}"
  mv -f "${_model_config_tmp}" "${LOCAL_MODEL_METADATA_DIR}/config.json"
fi

export OPENPI_BEHAVIOR_DATASET_ROOT="${OPENPI_BEHAVIOR_DATASET_ROOT:-${B1K_DATASET_ROOT}}"
export B1K_VIDEO_BACKEND="${B1K_VIDEO_BACKEND:-video_reader}"
export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Manifest probe batches: 0 = metadata-only (skip loader probing for faster startup)
# Set > 0 to re-enable shape/padding/episode inspection from real batches.
export OPENPI_DATA_MANIFEST_PROBE_BATCHES="${OPENPI_DATA_MANIFEST_PROBE_BATCHES:-0}"
if [[ "${OPENPI_OFFLINE}" == "1" ]]; then
  export WANDB_DISABLED="${WANDB_DISABLED:-1}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"
fi

export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-1}"
export OPENPI_DATALOADER_TIMEOUT_S="${OPENPI_DATALOADER_TIMEOUT_S:-600}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}"
export OPENPI_DATALOADER_PIN_MEMORY="${OPENPI_DATALOADER_PIN_MEMORY:-1}"
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"
export OPENPI_DDP_TIMEOUT_MIN="${OPENPI_DDP_TIMEOUT_MIN:-120}"

# PyTorch 2.7 warns on the deprecated NCCL_ASYNC_ERROR_HANDLING spelling.
# Honor an inherited legacy value once, then expose only the supported name.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-${NCCL_ASYNC_ERROR_HANDLING:-1}}"
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

# Merlin exposes these ARNOLD_* values on every node. Explicit standard names
# remain available as overrides for manual launches and debugging.
NUM_NODES="${NUM_NODES:-${NNODES:-${ARNOLD_WORKER_NUM:-1}}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-29514}}"
MASTER_PORT="${MASTER_PORT%%,*}"
if [[ -z "${MASTER_ADDR}" && "${NUM_NODES}" == "1" ]]; then
  MASTER_ADDR="127.0.0.1"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "ERROR: MASTER_ADDR/ARNOLD_WORKER_0_HOST is required for multi-node launch." >&2
  exit 2
fi
if ! [[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ && "${NODE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid topology NUM_NODES=${NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE} NODE_RANK=${NODE_RANK}" >&2
  exit 2
fi
if (( NODE_RANK >= NUM_NODES )); then
  echo "ERROR: NODE_RANK=${NODE_RANK} must be less than NUM_NODES=${NUM_NODES}." >&2
  exit 2
fi
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
export MASTER_ADDR MASTER_PORT NUM_NODES GPUS_PER_NODE NODE_RANK

# A non-positive step count disables the cap; train_accelerate.py derives the
# exact five-epoch target from the full-task dataloader at runtime.
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-epoch_with_rolling}"
# Rolling recovery writes are collective and I/O-heavy. 10000 optimizer steps is
# the default compromise; override per run with ROLLING_CHECKPOINT_INTERVAL.
ROLLING_CHECKPOINT_INTERVAL="${ROLLING_CHECKPOINT_INTERVAL:-10000}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-bfloat16}"
if [[ "${PYTORCH_TRAINING_PRECISION}" != "bfloat16" ]]; then
  echo "ERROR: this launcher is BF16-only; PYTORCH_TRAINING_PRECISION must be bfloat16." >&2
  exit 2
fi
if [[ "${CHECKPOINT_POLICY}" != "epoch_with_rolling" ]]; then
  echo "ERROR: this launcher requires CHECKPOINT_POLICY=epoch_with_rolling." >&2
  exit 2
fi
if ! [[ "${ROLLING_CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ROLLING_CHECKPOINT_INTERVAL must be a positive integer." >&2
  exit 2
fi

# Check all default HL dependencies before any distributed process is started.
for _required_file in \
  "${TRAINER}" \
  "${ACCEL_CONFIG}" \
  "${DEEPSPEED_CONFIG}" \
  "${BASE_PI05_CKPT}/model.safetensors" \
  "${BASE_PI05_CKPT}/config.json" \
  "${NORM_STATS_PATH}" \
  "${TOKENIZER_LOCAL}"; do
  if [[ ! -f "${_required_file}" ]]; then
    echo "ERROR: required file not found: ${_required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${OPENPI_BEHAVIOR_DATASET_ROOT}" ]]; then
  echo "ERROR: B1K dataset root not found: ${OPENPI_BEHAVIOR_DATASET_ROOT}" >&2
  exit 1
fi
TRAINING_DEP_ISSUES="$(python - "${REPO_ROOT}/environment.yml" <<'PY'
from importlib import metadata
import importlib.util
from pathlib import Path
import re
import sys

required_packages = ("accelerate", "deepspeed")
environment_text = Path(sys.argv[1]).read_text()
pins = dict(
    re.findall(
        r"^\s*-\s+(accelerate|deepspeed)==([^\s#]+)",
        environment_text,
        flags=re.MULTILINE,
    )
)
for package in required_packages:
    expected_version = pins.get(package)
    if expected_version is None:
        print(f"{package}: pin missing from environment.yml")
        continue
    if importlib.util.find_spec(package) is None:
        print(f"{package}=={expected_version}: missing")
        continue
    try:
        installed_version = metadata.version(package)
    except metadata.PackageNotFoundError:
        print(f"{package}=={expected_version}: package metadata missing")
        continue
    if installed_version != expected_version:
        print(f"{package}=={expected_version}: found {installed_version}")
PY
)"
if [[ -n "${TRAINING_DEP_ISSUES}" ]]; then
  echo "ERROR: conda env ${CONDA_ENV} does not match pinned training dependencies:" >&2
  while IFS= read -r _dep_issue; do
    echo "       - ${_dep_issue}" >&2
  done <<< "${TRAINING_DEP_ISSUES}"
  echo "       Resolve against ${REPO_ROOT}/environment.yml before launching." >&2
  exit 1
fi

# All durable artifacts stay under the shared repository/NAS checkout.
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/checkpoints}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/logs}"
EXP_NAME_SYNC_DIR="${PERSISTENT_OUTPUT_ROOT}/_exp_name_sync"

if [[ "${NODE_RANK}" == "0" ]]; then
  mkdir -p "${EXP_NAME_SYNC_DIR}" "${ASSETS_BASE_DIR}" "${CHECKPOINT_BASE_DIR}" "${LOG_BASE_DIR}"
else
  for _wait_i in $(seq 1 600); do
    [[ -d "${EXP_NAME_SYNC_DIR}" ]] && break
    sleep 1
  done
  if [[ ! -d "${EXP_NAME_SYNC_DIR}" ]]; then
    echo "ERROR: timed out waiting for shared output root: ${EXP_NAME_SYNC_DIR}" >&2
    exit 1
  fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${EXP_NAME:-}" ]]; then
  RUN_KEY="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-${MASTER_ADDR}_${MASTER_PORT}_${NUM_NODES}x${GPUS_PER_NODE}}}"
  RUN_KEY="${RUN_KEY//\//_}"
  RUN_KEY="${RUN_KEY//:/_}"
  RUN_KEY="${RUN_KEY// /_}"
  EXP_NAME_FILE="${EXP_NAME_SYNC_DIR}/full_b1k_bf16_${RUN_KEY}.txt"
  _script_start_sentinel="${EXP_NAME_SYNC_DIR}/.node${NODE_RANK}.start_sentinel.$$"
  : > "${_script_start_sentinel}"
  _script_start_ts="$(stat -c %Y "${_script_start_sentinel}")"
  trap 'rm -f "${_script_start_sentinel}"' EXIT
  if [[ "${NODE_RANK}" == "0" ]]; then
    if [[ "${RESUME:-0}" == "1" && -s "${EXP_NAME_FILE}" ]]; then
      EXP_NAME="$(<"${EXP_NAME_FILE}")"
      touch "${EXP_NAME_FILE}"
    else
      EXP_NAME="pi05_ki_joint_query_full_b1k_bf16_${NUM_NODES}n${GPUS_PER_NODE}g_${TIMESTAMP}"
      _exp_name_tmp="${EXP_NAME_FILE}.tmp.$$"
      printf '%s\n' "${EXP_NAME}" > "${_exp_name_tmp}"
      mv -f "${_exp_name_tmp}" "${EXP_NAME_FILE}"
    fi
  else
    for _wait_i in $(seq 1 600); do
      if [[ -s "${EXP_NAME_FILE}" ]]; then
        _exp_name_mtime="$(stat -c %Y "${EXP_NAME_FILE}" 2>/dev/null || echo 0)"
        [[ "${_exp_name_mtime}" -ge "${_script_start_ts}" ]] && break
      fi
      sleep 1
    done
    _exp_name_mtime="$(stat -c %Y "${EXP_NAME_FILE}" 2>/dev/null || echo 0)"
    if [[ ! -s "${EXP_NAME_FILE}" || "${_exp_name_mtime}" -lt "${_script_start_ts}" ]]; then
      echo "ERROR: timed out waiting for fresh experiment name: ${EXP_NAME_FILE}" >&2
      exit 1
    fi
    EXP_NAME="$(<"${EXP_NAME_FILE}")"
  fi
fi

CONSOLE_LOG_DIR="${PERSISTENT_OUTPUT_ROOT}/console_logs/${EXP_NAME}"
if [[ "${NODE_RANK}" == "0" ]]; then
  mkdir -p "${CONSOLE_LOG_DIR}"
else
  for _wait_i in $(seq 1 600); do
    [[ -d "${CONSOLE_LOG_DIR}" ]] && break
    sleep 1
  done
  if [[ ! -d "${CONSOLE_LOG_DIR}" ]]; then
    echo "ERROR: timed out waiting for console log directory: ${CONSOLE_LOG_DIR}" >&2
    exit 1
  fi
fi
CONSOLE_LOG="${CONSOLE_LOG_DIR}/node${NODE_RANK}.log"

EXTRA_ARGS=()
if [[ "${WANDB_DISABLED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-wandb-enabled)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
elif [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "${PREPARE_HF_CACHE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--prepare-hf-cache-only)
fi
if [[ "${FORCE_LOAD_CACHE}" == "1" ]]; then
  EXTRA_ARGS+=(--force-load-cache)
fi

GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS))
echo "============================================================"
echo "π0.5-KI joint-query full B1K task-set BF16 HL launch"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "TOPOLOGY=${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs (rank ${NODE_RANK}, world ${TOTAL_GPUS})"
echo "RENDEZVOUS=${MASTER_ADDR}:${MASTER_PORT}"
echo "BASE_PI05_CKPT=${BASE_PI05_CKPT}"
echo "NORM_STATS_PATH=${NORM_STATS_PATH}"
echo "OPENPI_BEHAVIOR_DATASET_ROOT=${OPENPI_BEHAVIOR_DATASET_ROOT}"
echo "ACCEL_CONFIG=${ACCEL_CONFIG}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG} (ZeRO-2, GPU-resident optimizer)"
echo "PRECISION=${PYTORCH_TRAINING_PRECISION} / Accelerate bf16"
echo "BUDGET=${NUM_TRAIN_EPOCHS} epoch(s); step cap=${NUM_TRAIN_STEPS} (0 disables cap)"
echo "CHECKPOINTS=${CHECKPOINT_POLICY}; durable every epoch, rolling every ${ROLLING_CHECKPOINT_INTERVAL} steps"
echo "VALIDATION_INTERVAL=${VAL_LOG_INTERVAL}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "PREPARE_HF_CACHE_ONLY=${PREPARE_HF_CACHE_ONLY}"
echo "FORCE_LOAD_CACHE=${FORCE_LOAD_CACHE}"
echo "MANIFEST_PROBE_BATCHES=${OPENPI_DATA_MANIFEST_PROBE_BATCHES}"
print_local_cache_paths
echo "CONSOLE_LOG=${CONSOLE_LOG}"
echo "============================================================"

python -m accelerate.commands.launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${TOTAL_GPUS}" \
  --num_machines "${NUM_NODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --same_network \
  "${TRAINER}" \
  "${CONFIG_NAME}" \
  --pytorch-weight-path "${BASE_PI05_CKPT}" \
  --exp-name "${EXP_NAME}" \
  --pytorch-training-precision "${PYTORCH_TRAINING_PRECISION}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --checkpoint-policy "${CHECKPOINT_POLICY}" \
  --rolling-checkpoint-interval "${ROLLING_CHECKPOINT_INTERVAL}" \
  --val-log-interval "${VAL_LOG_INTERVAL}" \
  --val-num-batches "${VAL_NUM_BATCHES}" \
  --batch-size-per-gpu "${BATCH_SIZE_PER_GPU}" \
  --num-workers "${NUM_WORKERS}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  --log-base-dir "${LOG_BASE_DIR}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"
