#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# Single-node Accelerate + DeepSpeed ZeRO-2 Training Script
# Target: V100 FP16
# Config: pi05_memoryvla_b1k-pt12_cs32_bs64_lr1e-4_step50k
#
# Notes
# - Override dataset root via OPENPI_BEHAVIOR_DATASET_ROOT (recommended across clusters).
# - Override base checkpoint via BASE_PI05_CKPT (must contain model.safetensors).
# - For smoke test, set: NUM_TRAIN_STEPS=2 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

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

# Conservative V100 fp16 defaults.  Override any of these variables explicitly
# for throughput A/B runs; set OPENPI_FP16_STABILITY_PROFILE=0 to keep the JSON
# DeepSpeed config untouched except for batch-size patching.
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

CONFIG_NAME="${CONFIG_NAME:-pi05_memoryvla_b1k-pt12_cs32_bs64_lr1e-4_step50k}"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${REPO_ROOT}/checkpoints/pi05_base_pytorch}"

DEFAULT_B1K_ASSETS_DIR_PT12="${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt12-cs32/assets"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${DEFAULT_B1K_ASSETS_DIR_PT12}}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-3}"
NUM_WORKERS="${NUM_WORKERS:-10}"
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-float16}"

SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
SAVE_AT_EPOCH_END_ONLY="${SAVE_AT_EPOCH_END_ONLY:-0}"
FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-0}"
PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"

ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_ds_zero2_v100_fp16.yaml}"
if [[ -n "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="${MASTER_PORT}"
else
  # Pick an available TCP port for torch distributed rendezvous.
  MASTER_PORT="$(python - <<'PY'
import socket
sock = socket.socket()
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"
fi

if [[ ! -f "${ACCEL_CONFIG}" ]]; then
  echo "Missing accelerate config: ${ACCEL_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${BASE_PI05_CKPT}/model.safetensors" ]]; then
  echo "Missing base checkpoint: ${BASE_PI05_CKPT}/model.safetensors" >&2
  exit 1
fi
if [[ ! -d "${B1K_ASSETS_DIR}" ]]; then
  echo "Missing B1K assets dir: ${B1K_ASSETS_DIR}" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_accel_ds_z2_v100fp16_${NUM_GPUS}g_${TIMESTAMP}}"

# Prefer per-node local SSD to reduce NAS lock contention.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/}"

CONSOLE_LOG_DIR="checkpoints/console_logs/${EXP_NAME}"
mkdir -p "${CONSOLE_LOG_DIR}"
CONSOLE_LOG="${CONSOLE_LOG_DIR}/node0.log"

echo "============================================================"
echo "Single-node training configuration:"
echo "  CONFIG_NAME: ${CONFIG_NAME}"
echo "  EXP_NAME: ${EXP_NAME}"
echo "  NUM_GPUS: ${NUM_GPUS}"
echo "  BASE_PI05_CKPT: ${BASE_PI05_CKPT}"
echo "  B1K_ASSETS_DIR: ${B1K_ASSETS_DIR}"
echo "  OPENPI_BEHAVIOR_DATASET_ROOT: ${OPENPI_BEHAVIOR_DATASET_ROOT}"
echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "  NUM_WORKERS: ${NUM_WORKERS}"
echo "  BATCH_SIZE_PER_GPU: ${BATCH_SIZE_PER_GPU}"
echo "  GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  PYTORCH_TRAINING_PRECISION: ${PYTORCH_TRAINING_PRECISION}"
echo "  NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS:-<unset>}"
echo "  NUM_TRAIN_STEPS: ${NUM_TRAIN_STEPS:-<unset>}"
echo "  SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "  KEEP_PERIOD: ${KEEP_PERIOD}"
echo "  SAVE_AT_EPOCH_END_ONLY: ${SAVE_AT_EPOCH_END_ONLY}"
echo "  FORCE_LOAD_CACHE: ${FORCE_LOAD_CACHE}"
echo "  PREPARE_HF_CACHE_ONLY: ${PREPARE_HF_CACHE_ONLY}"
echo "  ACCEL_CONFIG: ${ACCEL_CONFIG}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  OPENPI_FP16_STABILITY_PROFILE: ${OPENPI_FP16_STABILITY_PROFILE}"
echo "  OPENPI_FP16_INITIAL_SCALE_POWER: ${OPENPI_FP16_INITIAL_SCALE_POWER:-<unset>}"
echo "  OPENPI_FP16_LOSS_SCALE_WINDOW: ${OPENPI_FP16_LOSS_SCALE_WINDOW:-<unset>}"
echo "  OPENPI_FP16_HYSTERESIS: ${OPENPI_FP16_HYSTERESIS:-<unset>}"
echo "  OPENPI_DS_GRADIENT_CLIPPING: ${OPENPI_DS_GRADIENT_CLIPPING:-<unset>}"
echo "  OPENPI_DS_REDUCE_BUCKET_SIZE: ${OPENPI_DS_REDUCE_BUCKET_SIZE:-<unset>}"
echo "  OPENPI_DS_ALLGATHER_BUCKET_SIZE: ${OPENPI_DS_ALLGATHER_BUCKET_SIZE:-<unset>}"
echo "  OPENPI_DS_OFFLOAD_PIN_MEMORY: ${OPENPI_DS_OFFLOAD_PIN_MEMORY:-<unset>}"
echo "  CONSOLE_LOG: ${CONSOLE_LOG}"
echo "============================================================"

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

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_accelerate.py \
  "${CONFIG_NAME}" \
  --pytorch-weight-path "${BASE_PI05_CKPT}" \
  --exp_name "${EXP_NAME}" \
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Training finished"
