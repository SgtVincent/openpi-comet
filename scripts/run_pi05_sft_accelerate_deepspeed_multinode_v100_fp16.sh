#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# 4-Node Accelerate + DeepSpeed ZeRO-2 Training Script (V100 FP16 Best Practice)
# Total: 4 nodes × 8 GPUs = 32 GPUs
# Maintains 13324 steps (same as baseline) by keeping
# global batch size = 288 (3 × 32 × 3)
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_PATH="${CONDA_PATH:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3}"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate openpi-comet-nas
export LD_LIBRARY_PATH="${CONDA_PATH}/envs/openpi-comet-nas/lib:$LD_LIBRARY_PATH"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.cache/openpi}"
export B1K_VIDEO_BACKEND="${B1K_VIDEO_BACKEND:-video_reader}"

B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/}"
export OPENPI_BEHAVIOR_DATASET_ROOT="${OPENPI_BEHAVIOR_DATASET_ROOT:-${B1K_DATASET_ROOT}}"

export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-1}"
export OPENPI_DATALOADER_TIMEOUT_S="${OPENPI_DATALOADER_TIMEOUT_S:-600}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}"
export OPENPI_DATALOADER_PIN_MEMORY="${OPENPI_DATALOADER_PIN_MEMORY:-1}"
# Increase num_procs for multi-node dataset loading
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"

# ============================================================
# Multi-node configuration
# ============================================================
# Arnold compatibility: if ARNOLD_* env vars exist, prefer them.
# This matches the proven torchrun multi-node scripts in this repo.
if [[ -n "${ARNOLD_WORKER_NUM:-}" ]]; then
  NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM}}"
else
  NUM_NODES="${NUM_NODES:-4}"
fi
if [[ -n "${ARNOLD_WORKER_GPU:-}" ]]; then
  GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU}}"
else
  GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
fi
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# IMPORTANT: torchrun/accelerate expects a SINGLE hostname/IP for rendezvous.
# On many machines `hostname -i` can return multiple IPs (e.g., IPv6 + IPv4) separated by spaces,
# which breaks parsing like: "fdbd:... 10.x.x.x:29514".
# Prefer an explicit MASTER_ADDR from the scheduler; otherwise, pick the first non-loopback IPv4.
if [[ -n "${ARNOLD_WORKER_0_HOST:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST}}"
else
  MASTER_ADDR="${MASTER_ADDR:-}"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="$(
    hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+(\.[0-9]+){3}$' | grep -v '^127\.' | head -n1 || true
  )"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="$(
    hostname -i 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+(\.[0-9]+){3}$' | grep -v '^127\.' | head -n1 || true
  )"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "ERROR: MASTER_ADDR is empty. Please export MASTER_ADDR=<rank0 IPv4> explicitly." >&2
  exit 2
fi
if [[ "${MASTER_ADDR}" == *" "* ]]; then
  echo "ERROR: MASTER_ADDR must be a single host/IP, got: '${MASTER_ADDR}'" >&2
  echo "Please export MASTER_ADDR=<rank0 IPv4>, for example: export MASTER_ADDR=10.147.193.66" >&2
  exit 2
fi

if [[ -n "${ARNOLD_WORKER_0_PORT:-}" ]]; then
  # ARNOLD_WORKER_0_PORT can be a comma-separated port list; use the first.
  MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT%%,*}}"
else
  MASTER_PORT="${MASTER_PORT:-29514}"
fi
if [[ -n "${ARNOLD_ID:-}" ]]; then
  NODE_RANK="${NODE_RANK:-${ARNOLD_ID}}"
else
  NODE_RANK="${NODE_RANK:-0}"
fi

echo "============================================================"
echo "Multi-node training configuration:"
echo "  NUM_NODES: ${NUM_NODES}"
echo "  GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "  TOTAL_GPUS: ${TOTAL_GPUS}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "============================================================"

# ============================================================
# Training hyperparameters
# Keep the SAME effective global batch size per optimizer step as the single-node run,
# so that `num_train_steps` stays comparable for loss-curve comparison.
#
# Formula: effective_global_batch = batch_size_per_gpu × total_gpus × gradient_accumulation_steps
# Example (single-node baseline): 12 × 8 × 3 = 288
# Example (4 nodes × 8 GPUs, V100): 3 × 32 × 3 = 288
# ============================================================
TASK_NAME="${TASK_NAME:-make_pizza}"
CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-make_pizza_lr1e-4_5ep_sft}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${REPO_ROOT}/checkpoints/pi05_base_pytorch}"
DEFAULT_B1K_ASSETS_DIR="${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${DEFAULT_B1K_ASSETS_DIR}}"

SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
SAVE_AT_EPOCH_END_ONLY="${SAVE_AT_EPOCH_END_ONLY:-0}"
FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-0}"
PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"

# Reference (single-node) run hyperparams for fair comparison.
# Override these if your single-node run used different values.
REF_NUM_NODES="${REF_NUM_NODES:-1}"
REF_GPUS_PER_NODE="${REF_GPUS_PER_NODE:-8}"
REF_BATCH_SIZE_PER_GPU="${REF_BATCH_SIZE_PER_GPU:-12}"
REF_GRADIENT_ACCUMULATION_STEPS="${REF_GRADIENT_ACCUMULATION_STEPS:-3}"
REF_TOTAL_GPUS=$((REF_NUM_NODES * REF_GPUS_PER_NODE))
TARGET_GLOBAL_BATCH_SIZE=$((REF_BATCH_SIZE_PER_GPU * REF_TOTAL_GPUS * REF_GRADIENT_ACCUMULATION_STEPS))

NUM_WORKERS="${NUM_WORKERS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-float16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-3}"

denom=$((TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS))
if [[ "${denom}" -le 0 ]]; then
  echo "Invalid denom for batch computation: TOTAL_GPUS=${TOTAL_GPUS} GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}" >&2
  exit 1
fi

DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-2}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# HF datasets cache: prefer per-node local SSD (/opt/tiger) to avoid NAS lock contention.
# Allow external override via env.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/}"

# EXP_NAME must be identical across nodes. Arnold scripts synchronize this via a shared file.
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${REPO_ROOT}/checkpoints}"
EXP_NAME_SYNC_DIR="${CHECKPOINTS_ROOT}/_exp_name_sync"
if [[ -z "${EXP_NAME:-}" ]]; then
  RUN_KEY="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-}}"
  if [[ -z "${RUN_KEY}" ]]; then
    RUN_KEY="${MASTER_ADDR}_${MASTER_PORT}_${NUM_NODES}x${GPUS_PER_NODE}"
  fi
  RUN_KEY="${RUN_KEY//\//_}"
  RUN_KEY="${RUN_KEY//:/_}"
  RUN_KEY="${RUN_KEY// /_}"

  EXP_NAME_FILE="${EXP_NAME_SYNC_DIR}/pi05_accel_make_pizza_${RUN_KEY}.txt"
  # Record the wall-clock "script start" boundary so non-rank0 can reject any EXP_NAME_FILE
  # that was left behind by a previous run. We compare against file mtime below.
  # NOTE: touching this sentinel is critical to fix the cross-run race where non-rank0
  # reads a stale EXP_NAME_FILE from a prior (crashed) run before rank0 overwrites it.
  _script_start_sentinel="${EXP_NAME_SYNC_DIR}/.node${NODE_RANK}.start_sentinel.$$"
  mkdir -p "${EXP_NAME_SYNC_DIR}"
  : > "${_script_start_sentinel}"
  trap 'rm -f "${_script_start_sentinel}"' EXIT
  if [[ "${NODE_RANK}" == "0" ]]; then
    if [[ "${RESUME:-0}" == "1" && -s "${EXP_NAME_FILE}" ]]; then
      EXP_NAME="$(cat "${EXP_NAME_FILE}")"
    else
      # Proactively remove any stale EXP_NAME_FILE from a prior run so that non-rank0
      # cannot accidentally read it (race condition fix).
      rm -f "${EXP_NAME_FILE}"
      EXP_NAME="${CONFIG_NAME}_accel_ds_z${DEEPSPEED_STAGE}_v100fp16_${NUM_NODES}n${GPUS_PER_NODE}g_${TIMESTAMP}"
      _tmp_exp_name_file="${EXP_NAME_FILE}.$$.$RANDOM.tmp"
      printf "%s\n" "${EXP_NAME}" > "${_tmp_exp_name_file}"
      mv -f "${_tmp_exp_name_file}" "${EXP_NAME_FILE}"
      # Bump mtime so non-rank0 can detect that this file belongs to the current run.
      touch "${EXP_NAME_FILE}"
    fi
  else
    # Wait for an EXP_NAME_FILE that was WRITTEN AFTER this node's start sentinel.
    # This rejects stale files from crashed prior runs that share the same RUN_KEY.
    _start_ts=$(stat -c %Y "${_script_start_sentinel}" 2>/dev/null || echo 0)
    for _i in $(seq 1 600); do
      if [[ -s "${EXP_NAME_FILE}" ]]; then
        _file_ts=$(stat -c %Y "${EXP_NAME_FILE}" 2>/dev/null || echo 0)
        if [[ "${_file_ts}" -ge "${_start_ts}" ]]; then
          break
        fi
      fi
      sleep 1
    done
    _file_ts=$(stat -c %Y "${EXP_NAME_FILE}" 2>/dev/null || echo 0)
    if [[ ! -s "${EXP_NAME_FILE}" || "${_file_ts}" -lt "${_start_ts}" ]]; then
      echo "Timed out waiting for fresh EXP_NAME_FILE (>= ${_start_ts}): ${EXP_NAME_FILE} (mtime=${_file_ts})" >&2
      exit 1
    fi
    EXP_NAME="$(cat "${EXP_NAME_FILE}")"
  fi
else
  EXP_NAME="${EXP_NAME}"
fi

if [[ ! -f "${BASE_PI05_CKPT}/model.safetensors" ]]; then
  echo "Missing base checkpoint: ${BASE_PI05_CKPT}/model.safetensors" >&2
  exit 1
fi

if [[ ! -d "${B1K_ASSETS_DIR}" ]]; then
  echo "Missing B1K assets dir: ${B1K_ASSETS_DIR}" >&2
  exit 1
fi

if [[ "${B1K_ASSETS_DIR}" != "${DEFAULT_B1K_ASSETS_DIR}" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    mkdir -p "$(dirname "${DEFAULT_B1K_ASSETS_DIR}")"
    if [[ ! -e "${DEFAULT_B1K_ASSETS_DIR}" ]]; then
      ln -s "${B1K_ASSETS_DIR}" "${DEFAULT_B1K_ASSETS_DIR}"
    fi
  else
    for _i in $(seq 1 600); do
      if [[ -e "${DEFAULT_B1K_ASSETS_DIR}" ]]; then
        break
      fi
      sleep 1
    done
    if [[ ! -e "${DEFAULT_B1K_ASSETS_DIR}" ]]; then
      echo "Timed out waiting for DEFAULT_B1K_ASSETS_DIR: ${DEFAULT_B1K_ASSETS_DIR}" >&2
      exit 1
    fi
  fi
fi

# Avoid multi-node concurrent writes to the same log file on NAS.
CONSOLE_LOG_DIR="checkpoints/console_logs/${EXP_NAME}"
if [[ "${NODE_RANK}" == "0" ]]; then
  mkdir -p "${CONSOLE_LOG_DIR}"
else
  for _i in $(seq 1 600); do
    if [[ -d "${CONSOLE_LOG_DIR}" ]]; then
      break
    fi
    sleep 1
  done
  if [[ ! -d "${CONSOLE_LOG_DIR}" ]]; then
    echo "Timed out waiting for CONSOLE_LOG_DIR: ${CONSOLE_LOG_DIR}" >&2
    exit 1
  fi
fi
CONSOLE_LOG="${CONSOLE_LOG_DIR}/node${NODE_RANK}.log"

ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_ds_zero2_v100_fp16.yaml}"
if [[ ! -f "${ACCEL_CONFIG}" ]]; then
  echo "Missing accelerate config: ${ACCEL_CONFIG}" >&2
  exit 1
fi

# ============================================================
# Compute effective batch size for logging
# ============================================================
GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS))
EFFECTIVE_BATCH_PER_GPU=$((BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))

echo "Starting PI0.5 make_pizza SFT with Accelerate + DeepSpeed ZeRO-${DEEPSPEED_STAGE} (Multi-Node)"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${TASK_NAME}"
echo "Exp Name: ${EXP_NAME}"
echo "Base PyTorch checkpoint: ${BASE_PI05_CKPT}"
echo "B1K assets dir: ${B1K_ASSETS_DIR}"
echo "OPENPI_BEHAVIOR_DATASET_ROOT: ${OPENPI_BEHAVIOR_DATASET_ROOT:-<config_default>}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "Multi-node: ${NUM_NODES} nodes × ${GPUS_PER_NODE} GPUs = ${TOTAL_GPUS} GPUs"
echo "NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS}"
echo "NUM_TRAIN_STEPS: ${NUM_TRAIN_STEPS:-<auto>}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "KEEP_PERIOD: ${KEEP_PERIOD}"
echo "SAVE_AT_EPOCH_END_ONLY: ${SAVE_AT_EPOCH_END_ONLY}"
echo "FORCE_LOAD_CACHE: ${FORCE_LOAD_CACHE}"
echo "PREPARE_HF_CACHE_ONLY: ${PREPARE_HF_CACHE_ONLY}"
echo "BATCH_SIZE_PER_GPU: ${BATCH_SIZE_PER_GPU}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "PYTORCH_TRAINING_PRECISION: ${PYTORCH_TRAINING_PRECISION}"
echo "DeepSpeed Stage: ${DEEPSPEED_STAGE}"
echo "Accelerate config: ${ACCEL_CONFIG}"
echo "---"
echo "Global batch size: ${GLOBAL_BATCH_SIZE} (${BATCH_SIZE_PER_GPU} × ${TOTAL_GPUS} × ${GRADIENT_ACCUMULATION_STEPS})"
echo "Effective batch size per GPU: ${EFFECTIVE_BATCH_PER_GPU}"
echo "Reference global batch size (single-node): ${TARGET_GLOBAL_BATCH_SIZE} (${REF_BATCH_SIZE_PER_GPU} × ${REF_TOTAL_GPUS} × ${REF_GRADIENT_ACCUMULATION_STEPS})"
echo "Console log: ${CONSOLE_LOG}"

EXTRA_ARGS=()
if [[ "${WANDB_DISABLED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-wandb-enabled)
fi
case "${SAVE_AT_EPOCH_END_ONLY}" in
  1|true|TRUE|True|yes|YES|y|Y)
    EXTRA_ARGS+=(--save_at_epoch_end_only)
    ;;
esac
if [[ "${FORCE_LOAD_CACHE}" == "1" ]]; then
  EXTRA_ARGS+=(--force-load-cache)
fi
if [[ "${PREPARE_HF_CACHE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--prepare-hf-cache-only)
fi
if [[ -n "${BATCH_SIZE_PER_GPU}" ]]; then
  EXTRA_ARGS+=(--batch_size_per_gpu "${BATCH_SIZE_PER_GPU}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  EXTRA_ARGS+=(--num_workers "${NUM_WORKERS}")
fi
if [[ -n "${NUM_TRAIN_STEPS}" ]]; then
  EXTRA_ARGS+=(--num_train_steps "${NUM_TRAIN_STEPS}")
fi
if [[ "${GRADIENT_ACCUMULATION_STEPS}" != "1" ]]; then
  EXTRA_ARGS+=(--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}")
fi
if [[ -n "${PYTORCH_TRAINING_PRECISION}" ]]; then
  EXTRA_ARGS+=(--pytorch-training-precision "${PYTORCH_TRAINING_PRECISION}")
fi
case "${DEBUG_OVERFLOW:-0}" in
  1|true|TRUE|True|yes|YES|y|Y)
    EXTRA_ARGS+=(--debug-overflow)
    ;;
esac

# ============================================================
# Multi-node accelerate launch
# ============================================================
# Note: `--same_network` is a boolean CLI flag for accelerate.
# Do not pass `true` after it, otherwise `true` is parsed as the training script path.
# Note: For accelerate multi-node, `--num_processes` is the GLOBAL world size.
# Using per-node GPU count here would be divided by `--num_machines`,
# resulting in only (GPUS_PER_NODE / NUM_NODES) workers per node.
accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${TOTAL_GPUS}" \
  --num_machines "${NUM_NODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --same_network \
  scripts/train_accelerate.py \
  "${CONFIG_NAME}" \
  --pytorch-weight-path "${BASE_PI05_CKPT}" \
  --exp_name "${EXP_NAME}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Training finished"
