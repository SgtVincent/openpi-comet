#!/bin/bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.cache/openpi}"
export B1K_VIDEO_BACKEND="${B1K_VIDEO_BACKEND:-video_reader}"
export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-1}"
export OPENPI_DATALOADER_TIMEOUT_S="${OPENPI_DATALOADER_TIMEOUT_S:-600}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}"
export OPENPI_DATALOADER_PIN_MEMORY="${OPENPI_DATALOADER_PIN_MEMORY:-1}"
export OPENPI_DDP_FIND_UNUSED_PARAMETERS="${OPENPI_DDP_FIND_UNUSED_PARAMETERS:-0}"
export OPENPI_DDP_STATIC_GRAPH="${OPENPI_DDP_STATIC_GRAPH:-1}"
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_NAME="${CONFIG_NAME:-pi05_hybrid_b1k-make_pizza_lr2.5e-6_5ep_sft}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_step${NUM_TRAIN_STEPS}_${TIMESTAMP}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/}"

mkdir -p checkpoints/console_logs
CONSOLE_LOG="checkpoints/console_logs/${EXP_NAME}.log"

EXTRA_ARGS=()
if [[ "${WANDB_DISABLED:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--no-wandb-enabled)
fi

python -u scripts/train_pytorch.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --log-interval "${LOG_INTERVAL}" \
  --save-interval "${SAVE_INTERVAL}" \
  --overwrite \
  --batch-size-per-gpu "${BATCH_SIZE_PER_GPU}" \
  --num-workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${CONSOLE_LOG}"