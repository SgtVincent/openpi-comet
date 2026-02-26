#!/bin/bash
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
export PYTHONPATH="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python:$PYTHONPATH"
export LD_LIBRARY_PATH="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/lib:$LD_LIBRARY_PATH"


export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.cache/openpi}"
export B1K_VIDEO_BACKEND="${B1K_VIDEO_BACKEND:-video_reader}"

export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-1}"  # keep workers alive across epochs to avoid per-epoch spawn overhead
export OPENPI_DATALOADER_TIMEOUT_S="${OPENPI_DATALOADER_TIMEOUT_S:-600}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}"
export OPENPI_DATALOADER_PIN_MEMORY="${OPENPI_DATALOADER_PIN_MEMORY:-1}"
export OPENPI_DDP_FIND_UNUSED_PARAMETERS="${OPENPI_DDP_FIND_UNUSED_PARAMETERS:-0}"
export OPENPI_DDP_STATIC_GRAPH="${OPENPI_DDP_STATIC_GRAPH:-1}"
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi

IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#_GPU_IDS[@]}}"
if [[ "${NPROC_PER_NODE}" -le 0 ]]; then
  NPROC_PER_NODE=1
fi

TASK_NAME="${TASK_NAME:-make_pizza}"
CONFIG_NAME="${CONFIG_NAME:-vlm2_b1k-make_pizza_lr2.5e-6_5ep_sft}"
MASTER_PORT="${MASTER_PORT:-29512}"

SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-100000}"
SAVE_AT_EPOCH_END_ONLY="${SAVE_AT_EPOCH_END_ONLY:-0}"
FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-0}"
PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-}"
NUM_WORKERS="${NUM_WORKERS:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_${TIMESTAMP}}"
# default to config-exclusive cache, but allow external override for cache reuse
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/hf_datasets_cache/${CONFIG_NAME}}"

CONSOLE_LOG="checkpoints/console_logs/${EXP_NAME}.log"
mkdir -p "$(dirname "${CONSOLE_LOG}")"
TORCHRUN_LOG_DIR="checkpoints/torchrun_logs/${EXP_NAME}"
mkdir -p "${TORCHRUN_LOG_DIR}"

echo "Starting VLM2 SFT (5 epochs)"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${TASK_NAME}"
echo "Exp Name: ${EXP_NAME}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "KEEP_PERIOD: ${KEEP_PERIOD}"
echo "SAVE_AT_EPOCH_END_ONLY: ${SAVE_AT_EPOCH_END_ONLY}"
echo "FORCE_LOAD_CACHE: ${FORCE_LOAD_CACHE}"
echo "PREPARE_HF_CACHE_ONLY: ${PREPARE_HF_CACHE_ONLY}"
echo "OPENPI_DATALOADER_PREFETCH_FACTOR: ${OPENPI_DATALOADER_PREFETCH_FACTOR}"
echo "OPENPI_DATALOADER_PIN_MEMORY: ${OPENPI_DATALOADER_PIN_MEMORY}"
echo "OPENPI_DDP_FIND_UNUSED_PARAMETERS: ${OPENPI_DDP_FIND_UNUSED_PARAMETERS}"
echo "OPENPI_DDP_STATIC_GRAPH: ${OPENPI_DDP_STATIC_GRAPH}"
echo "OPENPI_LOAD_DATASET_NUM_PROC_CAP: ${OPENPI_LOAD_DATASET_NUM_PROC_CAP}"
echo "BATCH_SIZE_PER_GPU: ${BATCH_SIZE_PER_GPU:-<config_default>}"
echo "NUM_WORKERS: ${NUM_WORKERS:-<config_default>}"

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

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  --log_dir "${TORCHRUN_LOG_DIR}" --redirects 3 --tee 3 \
  scripts/train_pytorch.py \
  "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Training finished"
        