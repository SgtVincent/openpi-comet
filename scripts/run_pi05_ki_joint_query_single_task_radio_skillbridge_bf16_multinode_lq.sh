#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# π0.5-KI joint-query Skill Bridge single-task BF16 training
# for Merlin/Arnold on LQ (cloudnative-lq) cluster.
#
# Topology: 4 nodes × 8 A100-SXM4-40GB = 32 GPUs
# Config  : pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_bf16
# Task    : turning_on_radio
# Budget  : min(2000 steps, 1 epoch)
# Precision: BF16 (DeepSpeed ZeRO-2, GPU-resident optimizer)
# ============================================================

# REPO_ROOT may be explicitly set (e.g. /opt/tiger/openpi-comet for web
# FULL_SCRIPT mounts). If unset, derive from script location.
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME

# ---- LQ cluster paths ----
CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3}"
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

CONFIG_NAME="${CONFIG_NAME:-pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_bf16}"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch}"
B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/data/2025-challenge-demos/}"
B1K_ASSETS_DIR="${BASE_PI05_CKPT}/assets/behavior-1k/2025-challenge-demos"
NORM_STATS_PATH="${B1K_ASSETS_DIR}/norm_stats.json"
ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/configs/accelerate_ds_zero2.yaml}"
DEEPSPEED_CONFIG="${REPO_ROOT}/configs/deepspeed_zero2.json"
TRAINER="${TRAINER:-${REPO_ROOT}/scripts/train_accelerate.py}"

# ---- Local per-node cache (avoid NAS lock contention) ----
CACHE_RUN_KEY="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-${ARNOLD_WORKER_0_HOST:-manual}_${ARNOLD_WORKER_0_PORT:-local}}}"
CACHE_RUN_KEY="${CACHE_RUN_KEY//\//_}"
CACHE_RUN_KEY="${CACHE_RUN_KEY//:/_}"
CACHE_RUN_KEY="${CACHE_RUN_KEY//,/_}"
CACHE_RUN_KEY="${CACHE_RUN_KEY// /_}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/openpi-comet/${USER:-tiger}/${CONFIG_NAME}/${CACHE_RUN_KEY}}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${LOCAL_CACHE_ROOT}/openpi}"
export HF_HOME="${HF_HOME:-${LOCAL_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_CACHE_ROOT}/xdg}"
export TORCH_HOME="${TORCH_HOME:-${LOCAL_CACHE_ROOT}/torch}"
LOCAL_TMP_BACKING="${LOCAL_CACHE_ROOT}/tmp"
mkdir -p \
  "${OPENPI_DATA_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${LOCAL_TMP_BACKING}"

# Python's multiprocess.Manager appends pymp-*/listener-* below TMPDIR. Keep the
# storage inside LOCAL_CACHE_ROOT, but expose it through a short node-local alias
# so the AF_UNIX pathname remains below Linux sockaddr_un.sun_path's 107 usable
# bytes. A 96-bit SHA-256 prefix makes aliases collision-safe without embedding
# the potentially long config/job key in the socket pathname.
#
# TMPDIR is bound authoritatively (not "${TMPDIR:-...}"): an inherited TMPDIR
# from the Merlin task environment is exactly how the overflow reached training
# before, so inheritance must not silently bypass this fix. Deliberate opt-out
# stays available through OPENPI_ALLOW_EXTERNAL_TMPDIR=1.
_tmp_alias_digest="$(printf '%s' "${LOCAL_CACHE_ROOT}" | sha256sum)"
_tmp_alias_digest="${_tmp_alias_digest%% *}"
TMP_ALIAS="/tmp/openpi-tmp-${UID:-$(id -u)}-${_tmp_alias_digest:0:24}"
if [[ "${OPENPI_ALLOW_EXTERNAL_TMPDIR:-0}" == "1" && -n "${TMPDIR:-}" ]]; then
  echo "WARNING: honouring externally provided TMPDIR=${TMPDIR} (OPENPI_ALLOW_EXTERNAL_TMPDIR=1);" >&2
  echo "         AF_UNIX socket length is now the caller's responsibility." >&2
  _TMP_ALIAS_MANAGED=0
else
  _TMP_ALIAS_MANAGED=1
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
    # A concurrent rank may win this race; treat an equivalent symlink as success.
    if ! ln -s -- "${LOCAL_TMP_BACKING}" "${TMP_ALIAS}" 2>/dev/null; then
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
fi
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
if [[ ! -d "${TMPDIR}" || ! -w "${TMPDIR}" ]]; then
  echo "ERROR: TMPDIR is not a writable directory: ${TMPDIR}" >&2
  exit 1
fi

# Exercise the exact lightweight primitive that overflowed sun_path before, so a
# regression fails here instead of mid-training. This intentionally imports
# neither the trainer nor any model/data module.
python - "${TMPDIR}" "${LOCAL_TMP_BACKING}" "${_TMP_ALIAS_MANAGED}" <<'PY'
import os
from pathlib import Path
import sys
import tempfile

import multiprocess

expected_tmpdir = Path(sys.argv[1])
expected_backing = Path(sys.argv[2])
# Alias identity is only guaranteed when this launcher owns the binding; the
# sun_path budget and the Manager round-trip are verified either way.
alias_managed = sys.argv[3] == "1"

actual_tmpdir = Path(tempfile.gettempdir())
if actual_tmpdir != expected_tmpdir:
    raise SystemExit(
        f"ERROR: tempfile.gettempdir()={actual_tmpdir} does not equal intended TMPDIR {expected_tmpdir}"
    )
if alias_managed and actual_tmpdir.resolve() != expected_backing.resolve():
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
print("MULTIPROCESS_MANAGER_PREFLIGHT_OK")
PY

# Offline tokenizer bootstrap.
# NOTE: the tokenizer cache is NOT part of the Git checkout (untracked).
# Default to the canonical LQ NAS path; allow env override for custom setups.
REPO_OPENPI_CACHE="${REPO_OPENPI_CACHE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/.cache/openpi}"
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

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

# ============================================================
# Topology: strict 4×8 = 32 GPUs for formal LQ trial
# ============================================================
NUM_NODES="${NUM_NODES:-${NNODES:-${ARNOLD_WORKER_NUM:-}}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-}}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-}}"
MASTER_PORT="${MASTER_PORT%%,*}"

# Formal script: fail fast if critical Arnold vars are missing
if [[ -z "${NUM_NODES}" ]]; then
  echo "ERROR: NUM_NODES / ARNOLD_WORKER_NUM is required for formal LQ launch." >&2
  echo "       (use ARNOLD_WORKER_NUM or set NUM_NODES explicitly)" >&2
  exit 2
fi
if [[ -z "${GPUS_PER_NODE}" ]]; then
  echo "ERROR: GPUS_PER_NODE / ARNOLD_WORKER_GPU is required for formal LQ launch." >&2
  echo "       (use ARNOLD_WORKER_GPU or set GPUS_PER_NODE explicitly)" >&2
  exit 2
fi
if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: NODE_RANK / ARNOLD_ID is required for formal LQ launch." >&2
  echo "       (use ARNOLD_ID or set NODE_RANK explicitly)" >&2
  exit 2
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "ERROR: MASTER_ADDR / ARNOLD_WORKER_0_HOST is required for formal LQ multi-node launch." >&2
  echo "       (use ARNOLD_WORKER_0_HOST or set MASTER_ADDR explicitly)" >&2
  exit 2
fi
if [[ -z "${MASTER_PORT}" ]]; then
  echo "ERROR: MASTER_PORT / ARNOLD_WORKER_0_PORT is required for formal LQ multi-node launch." >&2
  echo "       (use ARNOLD_WORKER_0_PORT or set MASTER_PORT explicitly)" >&2
  exit 2
fi

# Validate topology values
if ! [[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ && "${NODE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid topology NUM_NODES=${NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE} NODE_RANK=${NODE_RANK}" >&2
  exit 2
fi
if (( NODE_RANK >= NUM_NODES )); then
  echo "ERROR: NODE_RANK=${NODE_RANK} must be less than NUM_NODES=${NUM_NODES}." >&2
  exit 2
fi

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Strict 4×8 = 32 GPU constraint for this formal trial script
if [[ "${NUM_NODES}" != "4" ]]; then
  echo "ERROR: this LQ formal script requires NUM_NODES=4, got ${NUM_NODES}." >&2
  exit 2
fi
if [[ "${GPUS_PER_NODE}" != "8" ]]; then
  echo "ERROR: this LQ formal script requires GPUS_PER_NODE=8, got ${GPUS_PER_NODE}." >&2
  exit 2
fi
if [[ "${TOTAL_GPUS}" != "32" ]]; then
  echo "ERROR: this LQ formal script requires TOTAL_GPUS=32, got ${TOTAL_GPUS}." >&2
  exit 2
fi

export MASTER_ADDR MASTER_PORT NUM_NODES GPUS_PER_NODE NODE_RANK

# ---- Training hyperparameters ----
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-2000}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-100}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-bfloat16}"
if [[ "${PYTORCH_TRAINING_PRECISION}" != "bfloat16" ]]; then
  echo "ERROR: this launcher is BF16-only; PYTORCH_TRAINING_PRECISION must be bfloat16." >&2
  exit 2
fi

# ---- Preflight checks (filesystem + deps) ----
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

# ---- Persistent output on shared NAS (not local /opt/tiger) ----
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_skillbridge_a100_lq_bf16}"
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

# ---- EXP_NAME sync across nodes (on shared NAS) ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${EXP_NAME:-}" ]]; then
  RUN_KEY="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-${MASTER_ADDR}_${MASTER_PORT}_${NUM_NODES}x${GPUS_PER_NODE}}}"
  RUN_KEY="${RUN_KEY//\//_}"
  RUN_KEY="${RUN_KEY//:/_}"
  RUN_KEY="${RUN_KEY// /_}"
  EXP_NAME_FILE="${EXP_NAME_SYNC_DIR}/skillbridge_radio_bf16_lq_${RUN_KEY}.txt"
  _script_start_sentinel="${EXP_NAME_SYNC_DIR}/.node${NODE_RANK}.start_sentinel.$$"
  : > "${_script_start_sentinel}"
  _script_start_ts="$(stat -c %Y "${_script_start_sentinel}")"
  trap 'rm -f "${_script_start_sentinel}"' EXIT
  if [[ "${NODE_RANK}" == "0" ]]; then
    if [[ "${RESUME:-0}" == "1" && -s "${EXP_NAME_FILE}" ]]; then
      EXP_NAME="$(<"${EXP_NAME_FILE}")"
      touch "${EXP_NAME_FILE}"
    else
      EXP_NAME="pi05_ki_joint_query_single_task_radio_skillbridge_bf16_lq_${NUM_NODES}n${GPUS_PER_NODE}g_${TIMESTAMP}"
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

# Per-node console logs on shared NAS
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

GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS))
echo "============================================================"
echo "π0.5-KI Skill Bridge single-task BF16 LQ launch"
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
echo "BUDGET=min(${NUM_TRAIN_STEPS} steps, ${NUM_TRAIN_EPOCHS} epoch)"
echo "INTERVALS=validation ${VAL_LOG_INTERVAL}, save ${SAVE_INTERVAL}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "LOCAL_CACHE_ROOT=${LOCAL_CACHE_ROOT}"
echo "PERSISTENT_OUTPUT_ROOT=${PERSISTENT_OUTPUT_ROOT}"
echo "CONSOLE_LOG=${CONSOLE_LOG}"
echo "============================================================"

# ---- Preflight-only mode: print command and exit without launching training ----
if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo ""
  echo "PREFLIGHT MODE (OPENPI_LAUNCH_PREFLIGHT_ONLY=1) — would run:"
  echo "  python -m accelerate.commands.launch \\"
  echo "    --config_file ${ACCEL_CONFIG} \\"
  echo "    --num_processes ${TOTAL_GPUS} \\"
  echo "    --num_machines ${NUM_NODES} \\"
  echo "    --machine_rank ${NODE_RANK} \\"
  echo "    --main_process_ip ${MASTER_ADDR} \\"
  echo "    --main_process_port ${MASTER_PORT} \\"
  echo "    --same_network \\"
  echo "    ${TRAINER} \\"
  echo "    ${CONFIG_NAME} \\"
  echo "    --pytorch-weight-path ${BASE_PI05_CKPT} \\"
  echo "    --exp-name ${EXP_NAME} \\"
  echo "    ... (remaining CLI args)"
  echo ""
  echo "Preflight OK — all checks passed. Exiting without launching training."
  exit 0
fi

# ---- Single accelerate launch ----
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
  --save-interval "${SAVE_INTERVAL}" \
  --val-log-interval "${VAL_LOG_INTERVAL}" \
  --val-num-batches "${VAL_NUM_BATCHES}" \
  --keep-period "${KEEP_PERIOD}" \
  --batch-size-per-gpu "${BATCH_SIZE_PER_GPU}" \
  --num-workers "${NUM_WORKERS}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  --log-base-dir "${LOG_BASE_DIR}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"
