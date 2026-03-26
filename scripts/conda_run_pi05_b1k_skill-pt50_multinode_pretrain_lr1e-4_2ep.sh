#!/bin/bash
set -euo pipefail
set -x # DEBUG PRINT

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
# Per-node local SSD cache to avoid NAS lock contention
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/${CONFIG_NAME}/}"
export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-1}"
export OPENPI_DATALOADER_TIMEOUT_S="${OPENPI_DATALOADER_TIMEOUT_S:-600}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-2}"
export OPENPI_DDP_TIMEOUT_MIN="${OPENPI_DDP_TIMEOUT_MIN:-120}"
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"
export OPENPI_HF_LOCAL_SYNC_TIMEOUT_S="${OPENPI_HF_LOCAL_SYNC_TIMEOUT_S:-7200}"
export OPENPI_HF_LOCAL_SYNC_POLL_S="${OPENPI_HF_LOCAL_SYNC_POLL_S:-2}"

# NCCL diagnostics: increase heartbeat timeout (default 480s is too short for multi-node init),
# enable debug info dump on timeout, and set async error handling for graceful failure.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Arnold environment variables auto set by merlin task
MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
# ARNOLD_WORKER_0_PORT 可能是逗号分隔的端口列表，这里只取第一个作为 master_port
MASTER_PORT="${ARNOLD_WORKER_0_PORT%%,*}"
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}
WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
FORCE_LOAD_CACHE="${FORCE_LOAD_CACHE:-1}"
PREPARE_HF_CACHE_ONLY="${PREPARE_HF_CACHE_ONLY:-0}"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"

LOG_INTERVAL="${LOG_INTERVAL:-100}"
PRECISION="${PRECISION:-bfloat16}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${REPO_ROOT}/checkpoints}"
EXP_NAME_SYNC_DIR="${CHECKPOINTS_ROOT}/_exp_name_sync"

if [[ -z "${EXP_NAME:-}" ]]; then
  RUN_KEY="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-}}"
  if [[ -z "${RUN_KEY}" ]]; then
    RUN_KEY="${MASTER_ADDR}_${MASTER_PORT}_${NNODES}x${NPROC_PER_NODE}"
  fi
  RUN_KEY="${RUN_KEY//\//_}"
  RUN_KEY="${RUN_KEY//:/_}"
  RUN_KEY="${RUN_KEY// /_}"

  EXP_NAME_FILE="${EXP_NAME_SYNC_DIR}/pi05_skill_pt50_pretrain_${RUN_KEY}.txt"
  if [[ "${NODE_RANK}" == "0" ]]; then
    mkdir -p "${EXP_NAME_SYNC_DIR}"
    if [[ "${RESUME:-0}" == "1" && -s "${EXP_NAME_FILE}" ]]; then
      EXP_NAME="$(cat "${EXP_NAME_FILE}")"
    else
      EXP_NAME="pi05_skill_pt50_multinode_pretrain_${NNODES}x${NPROC_PER_NODE}_${TIMESTAMP}"
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
  EXP_NAME="${EXP_NAME}"
fi

TORCHRUN_LOG_PARENT="${CHECKPOINTS_ROOT}/torchrun_logs/${EXP_NAME}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_PARENT}/node${NODE_RANK}"
if [[ "${NODE_RANK}" == "0" ]]; then
  mkdir -p "${TORCHRUN_LOG_PARENT}"
else
  for _i in $(seq 1 600); do
    if [[ -d "${TORCHRUN_LOG_PARENT}" ]]; then
      break
    fi
    sleep 1
  done
  if [[ ! -d "${TORCHRUN_LOG_PARENT}" ]]; then
    echo "Timed out waiting for TORCHRUN_LOG_PARENT: ${TORCHRUN_LOG_PARENT}" >&2
    exit 1
  fi
fi
mkdir -p "${TORCHRUN_LOG_DIR}"
CONSOLE_LOG="${TORCHRUN_LOG_DIR}/console.log"
if [[ "${NODE_RANK}" == "0" ]]; then
  mkdir -p "$(dirname "${CONSOLE_LOG}")"
fi

EXTRA_ARGS=()
if [[ "${WANDB_DISABLED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-wandb-enabled)
fi
EXTRA_ARGS+=(--overwrite)
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ "${FORCE_LOAD_CACHE}" == "1" ]]; then
  EXTRA_ARGS+=(--force-load-cache)
fi
if [[ "${PREPARE_HF_CACHE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--prepare-hf-cache-only)
fi

echo "Starting PI05 subtask skill pt50 multinode pretrain (Arnold multi-node)"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo "EXP_NAME: ${EXP_NAME}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "PER_GPU_BATCH_SIZE: ${PER_GPU_BATCH_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "NUM_EPOCHS: ${NUM_EPOCHS}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "KEEP_PERIOD: ${KEEP_PERIOD}"
echo "FORCE_LOAD_CACHE: ${FORCE_LOAD_CACHE}"
echo "PREPARE_HF_CACHE_ONLY: ${PREPARE_HF_CACHE_ONLY}"
echo "OPENPI_DDP_TIMEOUT_MIN: ${OPENPI_DDP_TIMEOUT_MIN}"
echo "OPENPI_LOAD_DATASET_NUM_PROC_CAP: ${OPENPI_LOAD_DATASET_NUM_PROC_CAP}"
echo "OPENPI_HF_LOCAL_SYNC_TIMEOUT_S: ${OPENPI_HF_LOCAL_SYNC_TIMEOUT_S}"
echo "OPENPI_HF_LOCAL_SYNC_POLL_S: ${OPENPI_HF_LOCAL_SYNC_POLL_S}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

torchrun \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --log_dir "${TORCHRUN_LOG_DIR}" --redirects 3 --tee 3 \
  scripts/train_pytorch.py \
  "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --log_interval "${LOG_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
  --batch_size_per_gpu "${PER_GPU_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --pytorch-training-precision "${PRECISION}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Training finished"