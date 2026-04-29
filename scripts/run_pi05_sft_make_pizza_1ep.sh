#!/bin/bash
# Usage examples:
#   CONFIG_NAME=pi05_b1k-make_pizza_lr1e-4_5ep_sft bash scripts/run_pi05_sft_make_pizza_1ep.sh
#   CONFIG_NAME=pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft bash scripts/run_pi05_sft_make_pizza_1ep.sh
#   CONFIG_NAME=pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft bash scripts/run_pi05_sft_make_pizza_1ep.sh
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"


source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
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
export OPENPI_DDP_TIMEOUT_MIN="${OPENPI_DDP_TIMEOUT_MIN:-120}"
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"
export OPENPI_HF_LOCAL_SYNC_TIMEOUT_S="${OPENPI_HF_LOCAL_SYNC_TIMEOUT_S:-7200}"
export OPENPI_HF_LOCAL_SYNC_POLL_S="${OPENPI_HF_LOCAL_SYNC_POLL_S:-2}"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

TASK_NAME="${TASK_NAME:-make_pizza}"
CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-make_pizza_lr1e-4_5ep_sft}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${REPO_ROOT}/checkpoints}"
USER_MASTER_PORT="${MASTER_PORT:-}"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${REPO_ROOT}/checkpoints/pi05_base_pytorch}"
DEFAULT_B1K_ASSETS_DIR="${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${DEFAULT_B1K_ASSETS_DIR}}"

SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-100000}"
SAVE_AT_EPOCH_END_ONLY="${SAVE_AT_EPOCH_END_ONLY:-1}"
FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-0}"
PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-36}"
NUM_WORKERS="${NUM_WORKERS:-2}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME_SYNC_DIR="${CHECKPOINTS_ROOT}/_exp_name_sync"
# default to config-exclusive cache, but allow external override for cache reuse
# export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/hf_datasets_cache/${CONFIG_NAME}}"
export HF_DATASETS_CACHE="/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/" #

IS_MULTINODE=0
if [[ -n "${ARNOLD_WORKER_0_HOST:-}" && -n "${ARNOLD_WORKER_GPU:-}" && -n "${ARNOLD_WORKER_NUM:-}" && -n "${ARNOLD_ID:-}" ]]; then
  IS_MULTINODE=1
  MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST}}"
  if [[ -n "${USER_MASTER_PORT}" ]]; then
    MASTER_PORT="${USER_MASTER_PORT}"
  else
    MASTER_PORT="${ARNOLD_WORKER_0_PORT%%,*}"
  fi
  NPROC_PER_NODE="${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU}}"
  NNODES="${NNODES:-${ARNOLD_WORKER_NUM}}"
  NODE_RANK="${NODE_RANK:-${ARNOLD_ID}}"
else
  MASTER_PORT="${USER_MASTER_PORT:-29513}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-${#_GPU_IDS[@]}}"
  else
    NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
  fi
  NNODES=1
  NODE_RANK=0
fi
WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"

if [[ -z "${EXP_NAME:-}" ]]; then
  if [[ "${IS_MULTINODE}" == "1" ]]; then
    RUN_KEY="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-}}"
    if [[ -z "${RUN_KEY}" ]]; then
      RUN_KEY="${MASTER_ADDR}_${MASTER_PORT}_${NNODES}x${NPROC_PER_NODE}"
    fi
    RUN_KEY="${RUN_KEY//\//_}"
    RUN_KEY="${RUN_KEY//:/_}"
    RUN_KEY="${RUN_KEY// /_}"
    EXP_NAME_FILE="${EXP_NAME_SYNC_DIR}/make_pizza_1ep_${RUN_KEY}.txt"
    if [[ "${NODE_RANK}" == "0" ]]; then
      mkdir -p "${EXP_NAME_SYNC_DIR}"
      if [[ "${RESUME:-0}" == "1" && -s "${EXP_NAME_FILE}" ]]; then
        EXP_NAME="$(cat "${EXP_NAME_FILE}")"
      else
        EXP_NAME="${CONFIG_NAME}_baseckpt_1ep_${TIMESTAMP}"
        _tmp_exp_name_file="${EXP_NAME_FILE}.$$.$RANDOM.tmp"
        printf "%s\n" "${EXP_NAME}" > "${_tmp_exp_name_file}"
        mv -f "${_tmp_exp_name_file}" "${EXP_NAME_FILE}"
      fi
    else
      for _i in $(seq 1 600); do
        if [[ -s "${EXP_NAME_FILE}" ]]; then
          break
        fi
        sleep 1
      done
      if [[ ! -s "${EXP_NAME_FILE}" ]]; then
        echo "Timed out waiting for EXP_NAME_FILE: ${EXP_NAME_FILE}" >&2
        exit 1
      fi
      EXP_NAME="$(cat "${EXP_NAME_FILE}")"
    fi
  else
    EXP_NAME="${CONFIG_NAME}_baseckpt_1ep_${TIMESTAMP}"
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
  mkdir -p "$(dirname "${DEFAULT_B1K_ASSETS_DIR}")"
  if [[ ! -e "${DEFAULT_B1K_ASSETS_DIR}" ]]; then
    ln -s "${B1K_ASSETS_DIR}" "${DEFAULT_B1K_ASSETS_DIR}"
  fi
fi

if [[ "${IS_MULTINODE}" == "1" ]]; then
  TORCHRUN_LOG_PARENT="${CHECKPOINTS_ROOT}/torchrun_logs/${EXP_NAME}"
  TORCHRUN_LOG_DIR="${TORCHRUN_LOG_PARENT}/node${NODE_RANK}"
  if [[ "${NODE_RANK}" == "0" ]]; then
    mkdir -p "${TORCHRUN_LOG_PARENT}"
  else
    for _i in $(seq 1 600); do
      if [[ -d "${TORCHRUN_LOG_PARENT}" ]]; then
        break
      fi
      ls -la "${CHECKPOINTS_ROOT}" >/dev/null 2>&1 || true
      sleep 1
    done
    if [[ ! -d "${TORCHRUN_LOG_PARENT}" ]]; then
      echo "Timed out waiting for TORCHRUN_LOG_PARENT: ${TORCHRUN_LOG_PARENT}" >&2
      exit 1
    fi
  fi
  mkdir -p "${TORCHRUN_LOG_DIR}"
  CONSOLE_LOG="${TORCHRUN_LOG_DIR}/console.log"
else
  CONSOLE_LOG="${CHECKPOINTS_ROOT}/console_logs/${EXP_NAME}.log"
  mkdir -p "$(dirname "${CONSOLE_LOG}")"
  TORCHRUN_LOG_DIR="${CHECKPOINTS_ROOT}/torchrun_logs/${EXP_NAME}"
  mkdir -p "${TORCHRUN_LOG_DIR}"
fi

echo "Starting PI0.5-family make_pizza SFT"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${TASK_NAME}"
echo "Exp Name: ${EXP_NAME}"
echo "Base PyTorch checkpoint: ${BASE_PI05_CKPT}"
echo "B1K assets dir: ${B1K_ASSETS_DIR}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "Launch mode: $([[ "${IS_MULTINODE}" == "1" ]] && echo multinode || echo single-node)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "MASTER_ADDR: ${MASTER_ADDR:-<standalone>}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS}"
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
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

TORCHRUN_ARGS=(--log_dir "${TORCHRUN_LOG_DIR}" --redirects 3 --tee 3)
if [[ "${IS_MULTINODE}" == "1" ]]; then
  TORCHRUN_ARGS=(
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    "${TORCHRUN_ARGS[@]}"
  )
else
  TORCHRUN_ARGS=(
    --standalone
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port "${MASTER_PORT}"
    "${TORCHRUN_ARGS[@]}"
  )
fi

torchrun "${TORCHRUN_ARGS[@]}" \
  scripts/train_pytorch.py \
  "${CONFIG_NAME}" \
  --pytorch-weight-path "${BASE_PI05_CKPT}" \
  --exp_name "${EXP_NAME}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Training finished"
