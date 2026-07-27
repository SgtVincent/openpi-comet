#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# Multi-node Accelerate + DeepSpeed ZeRO-2 Training Script (V100 FP16)
# Target: 8 nodes x 8 GPUs = 64 GPUs (defaults; can be overridden)
# Config default: pi05_memoryvla_b1k-pt12_cs32_bs64_lr1e-4_step50k
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Arnold/cluster environments sometimes inject user-site / PYTHONHOME that can shadow
# conda deps (e.g. requests -> urllib3). Force-import isolation to prefer conda.
export PYTHONNOUSERSITE=1
unset PYTHONHOME

CONDA_PATH="${CONDA_PATH:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3}"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate openpi-comet-nas
export LD_LIBRARY_PATH="${CONDA_PATH}/envs/openpi-comet-nas/lib:$LD_LIBRARY_PATH"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.cache/openpi}"
export B1K_VIDEO_BACKEND="${B1K_VIDEO_BACKEND:-video_reader}"

# Default to a path that exists alongside this repo checkout.
B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/data/2025-challenge-demos/}"
export OPENPI_BEHAVIOR_DATASET_ROOT="${OPENPI_BEHAVIOR_DATASET_ROOT:-${B1K_DATASET_ROOT}}"

export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# In offline mode we default to disabling wandb to avoid optional logging deps breaking training.
if [[ "${OPENPI_OFFLINE}" == "1" ]]; then
  WANDB_DISABLED="${WANDB_DISABLED:-1}"
  export WANDB_DISABLED
  export WANDB_MODE="${WANDB_MODE:-disabled}"
fi

export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-1}"
export OPENPI_DATALOADER_TIMEOUT_S="${OPENPI_DATALOADER_TIMEOUT_S:-600}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}"
export OPENPI_DATALOADER_PIN_MEMORY="${OPENPI_DATALOADER_PIN_MEMORY:-1}"
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"

# Conservative V100 fp16 defaults shared with the single-node launcher. Override
# explicitly for throughput A/B; set OPENPI_FP16_STABILITY_PROFILE=0 to leave the
# JSON DeepSpeed fp16/ZeRO values unchanged except for batch-size patching.
export OPENPI_FP16_STABILITY_PROFILE="${OPENPI_FP16_STABILITY_PROFILE:-1}"
if [[ "${OPENPI_FP16_STABILITY_PROFILE}" == "1" ]]; then
  export OPENPI_FP16_INITIAL_SCALE_POWER="${OPENPI_FP16_INITIAL_SCALE_POWER:-10}"
  export OPENPI_FP16_LOSS_SCALE_WINDOW="${OPENPI_FP16_LOSS_SCALE_WINDOW:-1000}"
  export OPENPI_FP16_HYSTERESIS="${OPENPI_FP16_HYSTERESIS:-2}"
  export OPENPI_FP16_MIN_LOSS_SCALE="${OPENPI_FP16_MIN_LOSS_SCALE:-1}"
  export OPENPI_DS_GRADIENT_CLIPPING="${OPENPI_DS_GRADIENT_CLIPPING:-0.5}"
  export OPENPI_DS_REDUCE_BUCKET_SIZE="${OPENPI_DS_REDUCE_BUCKET_SIZE:-50000000}"
  export OPENPI_DS_ALLGATHER_BUCKET_SIZE="${OPENPI_DS_ALLGATHER_BUCKET_SIZE:-50000000}"
  export OPENPI_DS_OFFLOAD_PIN_MEMORY="${OPENPI_DS_OFFLOAD_PIN_MEMORY:-0}"
fi

# Keep distributed failures observable without flooding normal logs.
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ============================================================
# Multi-node configuration
# ============================================================
# Arnold compatibility: if ARNOLD_* env vars exist, prefer them.
if [[ -n "${ARNOLD_WORKER_NUM:-}" ]]; then
  NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM}}"
else
  NUM_NODES="${NUM_NODES:-8}"
fi
if [[ -n "${ARNOLD_WORKER_GPU:-}" ]]; then
  GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU}}"
else
  GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
fi
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

if [[ -n "${ARNOLD_WORKER_0_HOST:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST}}"
else
  MASTER_ADDR="${MASTER_ADDR:-}"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+(\.[0-9]+){3}$' | grep -v '^127\.' | head -n1 || true)"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="$(hostname -i 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+(\.[0-9]+){3}$' | grep -v '^127\.' | head -n1 || true)"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "ERROR: MASTER_ADDR is empty. Please export MASTER_ADDR=<rank0 IPv4> explicitly." >&2
  exit 2
fi
if [[ "${MASTER_ADDR}" == *" "* ]]; then
  echo "ERROR: MASTER_ADDR must be a single host/IP, got: '${MASTER_ADDR}'" >&2
  exit 2
fi

if [[ -n "${ARNOLD_WORKER_0_PORT:-}" ]]; then
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
# ============================================================
CONFIG_NAME="${CONFIG_NAME:-pi05_memoryvla_b1k-pt12_cs32_bs64_lr1e-4_step50k}"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${REPO_ROOT}/checkpoints/pi05_base_pytorch}"

DEFAULT_B1K_ASSETS_DIR_PT50="${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets"
DEFAULT_B1K_ASSETS_DIR_PT12="${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt12-cs32/assets"
if [[ "${CONFIG_NAME}" == *"pt12"* || "${CONFIG_NAME}" == *"b1kpt12"* ]]; then
  DEFAULT_B1K_ASSETS_DIR="${DEFAULT_B1K_ASSETS_DIR_PT12}"
else
  DEFAULT_B1K_ASSETS_DIR="${DEFAULT_B1K_ASSETS_DIR_PT50}"
fi
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${DEFAULT_B1K_ASSETS_DIR}}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"

SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
SAVE_AT_EPOCH_END_ONLY="${SAVE_AT_EPOCH_END_ONLY:-0}"
FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-0}"
PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"

NUM_WORKERS="${NUM_WORKERS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-float16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"

DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-2}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# HF datasets cache: prefer per-node local SSD (/opt/tiger) to avoid NAS lock contention.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/}"

# EXP_NAME must be identical across nodes.
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

  EXP_NAME_FILE="${EXP_NAME_SYNC_DIR}/pi05_accel_memoryvla_${RUN_KEY}.txt"
  _script_start_sentinel="${EXP_NAME_SYNC_DIR}/.node${NODE_RANK}.start_sentinel.$$"
  mkdir -p "${EXP_NAME_SYNC_DIR}"
  : > "${_script_start_sentinel}"
  trap 'rm -f "${_script_start_sentinel}"' EXIT

  if [[ "${NODE_RANK}" == "0" ]]; then
    if [[ "${RESUME:-0}" == "1" && -s "${EXP_NAME_FILE}" ]]; then
      EXP_NAME="$(cat "${EXP_NAME_FILE}")"
    else
      rm -f "${EXP_NAME_FILE}"
      EXP_NAME="${CONFIG_NAME}_accel_ds_z${DEEPSPEED_STAGE}_v100fp16_${NUM_NODES}n${GPUS_PER_NODE}g_${TIMESTAMP}"
      _tmp_exp_name_file="${EXP_NAME_FILE}.$$.$RANDOM.tmp"
      printf "%s\n" "${EXP_NAME}" > "${_tmp_exp_name_file}"
      mv -f "${_tmp_exp_name_file}" "${EXP_NAME_FILE}"
      touch "${EXP_NAME_FILE}"
    fi
  else
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
fi

if [[ ! -f "${BASE_PI05_CKPT}/model.safetensors" ]]; then
  echo "Missing base checkpoint: ${BASE_PI05_CKPT}/model.safetensors" >&2
  exit 1
fi
if [[ ! -d "${B1K_ASSETS_DIR}" ]]; then
  echo "Missing B1K assets dir: ${B1K_ASSETS_DIR}" >&2
  exit 1
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

GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS))

echo "Starting OpenPi MemoryVLA pretrain with Accelerate + DeepSpeed ZeRO-${DEEPSPEED_STAGE} (Multi-Node)"
echo "Config: ${CONFIG_NAME}"
echo "Exp Name: ${EXP_NAME}"
echo "Base PyTorch checkpoint: ${BASE_PI05_CKPT}"
echo "B1K assets dir: ${B1K_ASSETS_DIR}"
echo "OPENPI_BEHAVIOR_DATASET_ROOT: ${OPENPI_BEHAVIOR_DATASET_ROOT:-<config_default>}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "Multi-node: ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs = ${TOTAL_GPUS} GPUs"
echo "NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS:-<unset>}"
echo "NUM_TRAIN_STEPS: ${NUM_TRAIN_STEPS:-<unset>}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "KEEP_PERIOD: ${KEEP_PERIOD}"
echo "SAVE_AT_EPOCH_END_ONLY: ${SAVE_AT_EPOCH_END_ONLY}"
echo "FORCE_LOAD_CACHE: ${FORCE_LOAD_CACHE}"
echo "PREPARE_HF_CACHE_ONLY: ${PREPARE_HF_CACHE_ONLY}"
echo "BATCH_SIZE_PER_GPU: ${BATCH_SIZE_PER_GPU}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "PYTORCH_TRAINING_PRECISION: ${PYTORCH_TRAINING_PRECISION}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE} (${BATCH_SIZE_PER_GPU} x ${TOTAL_GPUS} x ${GRADIENT_ACCUMULATION_STEPS})"
echo "OPENPI_FP16_STABILITY_PROFILE: ${OPENPI_FP16_STABILITY_PROFILE}"
echo "OPENPI_FP16_INITIAL_SCALE_POWER: ${OPENPI_FP16_INITIAL_SCALE_POWER:-<unset>}"
echo "OPENPI_FP16_LOSS_SCALE_WINDOW: ${OPENPI_FP16_LOSS_SCALE_WINDOW:-<unset>}"
echo "OPENPI_FP16_HYSTERESIS: ${OPENPI_FP16_HYSTERESIS:-<unset>}"
echo "OPENPI_DS_GRADIENT_CLIPPING: ${OPENPI_DS_GRADIENT_CLIPPING:-<unset>}"
echo "OPENPI_DS_REDUCE_BUCKET_SIZE: ${OPENPI_DS_REDUCE_BUCKET_SIZE:-<unset>}"
echo "OPENPI_DS_ALLGATHER_BUCKET_SIZE: ${OPENPI_DS_ALLGATHER_BUCKET_SIZE:-<unset>}"
echo "OPENPI_DS_OFFLOAD_PIN_MEMORY: ${OPENPI_DS_OFFLOAD_PIN_MEMORY:-<unset>}"
echo "NCCL_ASYNC_ERROR_HANDLING: ${NCCL_ASYNC_ERROR_HANDLING}"
echo "NCCL_DEBUG: ${NCCL_DEBUG}"
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
if [[ -n "${NUM_TRAIN_EPOCHS}" ]]; then
  EXTRA_ARGS+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
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

# IMPORTANT:
# - `--same_network` is a boolean flag; do NOT pass `true` after it.
# - `--num_processes` is GLOBAL world size.
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
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Training finished"
