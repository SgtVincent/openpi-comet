#!/bin/bash
set -euo pipefail
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
export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-1}"
export OPENPI_DATALOADER_TIMEOUT_S="${OPENPI_DATALOADER_TIMEOUT_S:-600}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}"
export OPENPI_DATALOADER_PIN_MEMORY="${OPENPI_DATALOADER_PIN_MEMORY:-1}"
export OPENPI_DDP_FIND_UNUSED_PARAMETERS="${OPENPI_DDP_FIND_UNUSED_PARAMETERS:-0}"
export OPENPI_DDP_STATIC_GRAPH="${OPENPI_DDP_STATIC_GRAPH:-1}"
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-${#_GPU_IDS[@]}}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
fi

if [[ "${NPROC_PER_NODE}" -le 0 ]]; then
  NPROC_PER_NODE=1
fi

CONFIG_NAME="${CONFIG_NAME:-vlm2_subtask_b1k-make_pizza_ann-skill_lr1e-4_5ep_sft}"
MASTER_PORT="${MASTER_PORT:-29533}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-100000}"
SAVE_AT_EPOCH_END_ONLY="${SAVE_AT_EPOCH_END_ONLY:-1}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${REPO_ROOT}/checkpoints/pi05_base_pytorch}"
DEFAULT_B1K_ASSETS_DIR="${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${DEFAULT_B1K_ASSETS_DIR}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_baseckpt_1ep_${TIMESTAMP}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/}"

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

mkdir -p checkpoints/console_logs checkpoints/torchrun_logs outputs/checkpoints outputs/logs
CONSOLE_LOG="checkpoints/console_logs/${EXP_NAME}.log"
TORCHRUN_LOG_DIR="checkpoints/torchrun_logs/${EXP_NAME}"

EXTRA_ARGS=()
if [[ "${WANDB_DISABLED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-wandb-enabled)
fi
case "${SAVE_AT_EPOCH_END_ONLY}" in
  1|true|TRUE|True|yes|YES|y|Y)
    EXTRA_ARGS+=(--save_at_epoch_end_only)
    ;;
esac
if [[ -n "${NUM_TRAIN_STEPS}" ]]; then
  EXTRA_ARGS+=(--num_train_steps "${NUM_TRAIN_STEPS}")
fi

echo "Starting VLM2+Subtask make_pizza SFT (annotations_skill supervision)"
echo "Config: ${CONFIG_NAME}"
echo "Exp Name: ${EXP_NAME}"
echo "Base PyTorch checkpoint: ${BASE_PI05_CKPT}"
echo "B1K assets dir: ${B1K_ASSETS_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "num_train_epochs: ${NUM_TRAIN_EPOCHS}"
echo "num_train_steps: ${NUM_TRAIN_STEPS:-<use_epochs>}"
echo "save_interval: ${SAVE_INTERVAL}"
echo "keep_period: ${KEEP_PERIOD}"
echo "batch_size_per_gpu: ${BATCH_SIZE_PER_GPU}"
echo "num_workers: ${NUM_WORKERS}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  --log_dir "${TORCHRUN_LOG_DIR}" --redirects 3 --tee 3 \
  scripts/train_pytorch.py \
  "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --pytorch-weight-path "${BASE_PI05_CKPT}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --log_interval "${LOG_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
  --overwrite \
  --batch_size_per_gpu "${BATCH_SIZE_PER_GPU}" \
  --num_workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${CONSOLE_LOG}"
