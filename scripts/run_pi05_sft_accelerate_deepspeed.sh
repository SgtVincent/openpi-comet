#!/bin/bash
set -euo pipefail
set -x

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
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-${#_GPU_IDS[@]}}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
fi

TASK_NAME="${TASK_NAME:-make_pizza}"
CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-make_pizza_lr1e-4_5ep_sft}"
MASTER_PORT="${MASTER_PORT:-29514}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${REPO_ROOT}/checkpoints/pi05_base_pytorch}"
DEFAULT_B1K_ASSETS_DIR="${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${DEFAULT_B1K_ASSETS_DIR}}"

SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-100000}"
SAVE_AT_EPOCH_END_ONLY="${SAVE_AT_EPOCH_END_ONLY:-1}"
FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-0}"
PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-12}"
NUM_WORKERS="${NUM_WORKERS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-2}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_accel_ds_z${DEEPSPEED_STAGE}_${TIMESTAMP}}"
export HF_DATASETS_CACHE="/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/"

if [[ ! -f "${BASE_PI05_CKPT}/model.safetensors" ]]; then
  echo "Missing base checkpoint: ${BASE_PI05_CKPT}/model.safetensors" >&2
  exit 1
fi

if [[ ! -d "${B1K_ASSETS_DIR}" ]]; then
  echo "Missing B1K assets dir: ${B1K_ASSETS_DIR}" >&2
  exit 1
fi

if [[ "${B1K_ASSETS_DIR}" != "${DEFAULT_B1K_ASSETS_DIR}" ]]; then
  mkdir -p "$(dirname "${DEFAULT_B1K_ASSETS_DIR}")"
  if [[ ! -e "${DEFAULT_B1K_ASSETS_DIR}" ]]; then
    ln -s "${B1K_ASSETS_DIR}" "${DEFAULT_B1K_ASSETS_DIR}"
  fi
fi

CONSOLE_LOG="checkpoints/console_logs/${EXP_NAME}.log"
mkdir -p "$(dirname "${CONSOLE_LOG}")"

ACCEL_CONFIG="configs/accelerate_ds_zero${DEEPSPEED_STAGE}.yaml"
if [[ ! -f "${ACCEL_CONFIG}" ]]; then
  echo "Missing accelerate config: ${ACCEL_CONFIG}" >&2
  exit 1
fi

echo "Starting PI0.5 make_pizza SFT with Accelerate + DeepSpeed ZeRO-${DEEPSPEED_STAGE}"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${TASK_NAME}"
echo "Exp Name: ${EXP_NAME}"
echo "Base PyTorch checkpoint: ${BASE_PI05_CKPT}"
echo "B1K assets dir: ${B1K_ASSETS_DIR}"
echo "OPENPI_BEHAVIOR_DATASET_ROOT: ${OPENPI_BEHAVIOR_DATASET_ROOT:-<config_default>}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "KEEP_PERIOD: ${KEEP_PERIOD}"
echo "SAVE_AT_EPOCH_END_ONLY: ${SAVE_AT_EPOCH_END_ONLY}"
echo "FORCE_LOAD_CACHE: ${FORCE_LOAD_CACHE}"
echo "PREPARE_HF_CACHE_ONLY: ${PREPARE_HF_CACHE_ONLY}"
echo "BATCH_SIZE_PER_GPU: ${BATCH_SIZE_PER_GPU}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "DeepSpeed Stage: ${DEEPSPEED_STAGE}"
echo "Accelerate config: ${ACCEL_CONFIG}"

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
if [[ "${GRADIENT_ACCUMULATION_STEPS}" != "1" ]]; then
  EXTRA_ARGS+=(--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}")
fi

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NPROC_PER_NODE}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_accelerate.py \
  "${CONFIG_NAME}" \
  --pytorch-weight-path "${BASE_PI05_CKPT}" \
  --exp_name "${EXP_NAME}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Training finished"
