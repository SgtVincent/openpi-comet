#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# Skill Bridge Serial A/B Experiment (V100 FP32, 8 nodes × 8 = 64 GPUs)
#
# Phase 1: Control (bridge disabled)  → 2000 steps
# Phase 2: Bridge  (bridge enabled)   → 2000 steps
# Both phases run on the SAME allocation (64 V100 GPUs).
#
# Based on proven pattern: run_pi05_ki_joint_query_single_task_radio_fp32_multinode_v100.sh
# ============================================================

# Force JAX to CPU only (no GPU JAX)
export JAX_PLATFORMS=cpu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Set PYTHONPATH to src
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Arnold/cluster environments sometimes inject user-site / PYTHONHOME that can shadow
# conda deps. Force-import isolation to prefer conda.
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

# NCCL: conservative timeouts for multi-node
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

# ============================================================
# Multi-node configuration (Arnold-compatible)
# ============================================================
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

# ============================================================
# Experiment configuration
# ============================================================
# Phase 1: control (bridge disabled)
CONFIG_NAME_1="${CONFIG_NAME_1:-pi05_ki_joint_query_b1k-single_task-radio-ki_on_control_fp32}"
# Phase 2: bridge (bridge enabled)
CONFIG_NAME_2="${CONFIG_NAME_2:-pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_fp32}"

TASK_NAME="${TASK_NAME:-turning_on_radio}"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch}"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${BASE_PI05_CKPT}/assets}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-2000}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"  # steps takes priority

SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-100}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"

NUM_WORKERS="${NUM_WORKERS:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-float32}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"

DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-2}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_ds_zero2_v100_fp32.yaml}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# HF datasets cache: prefer per-node local SSD (/opt/tiger) to avoid NAS lock contention.
# Use a shared cache name so both phases share the same dataset cache.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/skill_bridge_serial_v100_fp32/}"

# EXP_NAME must be identical across nodes for both phases.
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_ROOT}/outputs}"
EXP_NAME_SYNC_DIR="${OUTPUTS_ROOT}/_exp_name_sync"
if [[ -z "${EXP_NAME:-}" ]]; then
  RUN_KEY="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-}}"
  if [[ -z "${RUN_KEY}" ]]; then
    RUN_KEY="${MASTER_ADDR}_${MASTER_PORT}_${NUM_NODES}x${GPUS_PER_NODE}"
  fi
  RUN_KEY="${RUN_KEY//\//_}"
  RUN_KEY="${RUN_KEY//:/_}"
  RUN_KEY="${RUN_KEY// /_}"

  EXP_NAME_FILE="${EXP_NAME_SYNC_DIR}/skill_bridge_serial_${RUN_KEY}.txt"
  _script_start_sentinel="${EXP_NAME_SYNC_DIR}/.node${NODE_RANK}.start_sentinel.$$"
  mkdir -p "${EXP_NAME_SYNC_DIR}"
  : > "${_script_start_sentinel}"
  trap 'rm -f "${_script_start_sentinel}"' EXIT

  if [[ "${NODE_RANK}" == "0" ]]; then
    rm -f "${EXP_NAME_FILE}"
    EXP_NAME="skill_bridge_serial_${NUM_NODES}n${GPUS_PER_NODE}g_${TIMESTAMP}"
    _tmp_exp_name_file="${EXP_NAME_FILE}.$$.$RANDOM.tmp"
    printf "%s\n" "${EXP_NAME}" > "${_tmp_exp_name_file}"
    mv -f "${_tmp_exp_name_file}" "${EXP_NAME_FILE}"
    touch "${EXP_NAME_FILE}"
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

# Verify base checkpoint and assets exist
if [[ ! -f "${BASE_PI05_CKPT}/model.safetensors" ]]; then
  echo "Missing base checkpoint: ${BASE_PI05_CKPT}/model.safetensors" >&2
  exit 1
fi
if [[ ! -d "${B1K_ASSETS_DIR}" ]]; then
  echo "Missing B1K assets dir: ${B1K_ASSETS_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ACCEL_CONFIG}" ]]; then
  echo "Missing accelerate config: ${ACCEL_CONFIG}" >&2
  exit 1
fi

# Per-node console log to avoid NAS concurrent write issues.
CONSOLE_LOG_DIR="${OUTPUTS_ROOT}/console_logs/${EXP_NAME}"
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

GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS))

echo "============================================================" | tee -a "${CONSOLE_LOG}"
echo "Skill Bridge Serial A/B Experiment — V100 FP32 64 GPU" | tee -a "${CONSOLE_LOG}"
echo "============================================================" | tee -a "${CONSOLE_LOG}"
echo "Phase 1 (control): ${CONFIG_NAME_1}" | tee -a "${CONSOLE_LOG}"
echo "Phase 2 (bridge):  ${CONFIG_NAME_2}" | tee -a "${CONSOLE_LOG}"
echo "Exp Name: ${EXP_NAME}" | tee -a "${CONSOLE_LOG}"
echo "Base PyTorch checkpoint: ${BASE_PI05_CKPT}" | tee -a "${CONSOLE_LOG}"
echo "B1K assets dir: ${B1K_ASSETS_DIR}" | tee -a "${CONSOLE_LOG}"
echo "Multi-node: ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs = ${TOTAL_GPUS} GPUs" | tee -a "${CONSOLE_LOG}"
echo "NODE_RANK: ${NODE_RANK}" | tee -a "${CONSOLE_LOG}"
echo "MASTER_ADDR: ${MASTER_ADDR}" | tee -a "${CONSOLE_LOG}"
echo "MASTER_PORT: ${MASTER_PORT}" | tee -a "${CONSOLE_LOG}"
echo "NUM_TRAIN_STEPS: ${NUM_TRAIN_STEPS}" | tee -a "${CONSOLE_LOG}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}" | tee -a "${CONSOLE_LOG}"
echo "VAL_LOG_INTERVAL: ${VAL_LOG_INTERVAL}" | tee -a "${CONSOLE_LOG}"
echo "BATCH_SIZE_PER_GPU: ${BATCH_SIZE_PER_GPU}" | tee -a "${CONSOLE_LOG}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE}" | tee -a "${CONSOLE_LOG}"
echo "Accelerate config: ${ACCEL_CONFIG}" | tee -a "${CONSOLE_LOG}"
echo "Console log: ${CONSOLE_LOG}" | tee -a "${CONSOLE_LOG}"
echo "============================================================" | tee -a "${CONSOLE_LOG}"

# ============================================================
# Helper: run one training phase
# ============================================================
run_phase() {
  local phase_num=$1
  local config_name=$2
  local phase_exp_name="${EXP_NAME}_phase${phase_num}"

  echo "" | tee -a "${CONSOLE_LOG}"
  echo "====== PHASE ${phase_num}: ${config_name} ======" | tee -a "${CONSOLE_LOG}"
  echo "Phase exp name: ${phase_exp_name}" | tee -a "${CONSOLE_LOG}"

  EXTRA_ARGS=()
  if [[ "${WANDB_DISABLED:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-wandb-enabled)
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
  if [[ -n "${VAL_LOG_INTERVAL}" ]]; then
    EXTRA_ARGS+=(--val_log_interval "${VAL_LOG_INTERVAL}")
  fi

  # IMPORTANT:
  # - `--same_network` is a boolean flag; do NOT pass `true` after it.
  # - `--num_processes` is GLOBAL world size (total GPUs across all nodes).
  # - Config name is positional, NOT --config-name
  accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${TOTAL_GPUS}" \
    --num_machines "${NUM_NODES}" \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --same_network \
    scripts/train_accelerate.py \
    "${config_name}" \
    --pytorch-weight-path "${BASE_PI05_CKPT}" \
    --exp_name "${phase_exp_name}" \
    --save_interval "${SAVE_INTERVAL}" \
    --keep_period "${KEEP_PERIOD}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

  local rc=${PIPESTATUS[0]}
  echo "Phase ${phase_num} finished with exit code ${rc}" | tee -a "${CONSOLE_LOG}"
  return ${rc}
}

# ============================================================
# Phase 1: Control (bridge disabled)
# ============================================================
run_phase 1 "${CONFIG_NAME_1}"
PHASE1_RC=$?

# Cooldown between phases — let NCCL fully teardown before next init
echo "" | tee -a "${CONSOLE_LOG}"
echo "Inter-phase cooldown: 90s (let NCCL teardown fully)..." | tee -a "${CONSOLE_LOG}"
sleep 90

# ============================================================
# Phase 2: Bridge (bridge enabled)
# ============================================================
run_phase 2 "${CONFIG_NAME_2}"
PHASE2_RC=$?

echo "" | tee -a "${CONSOLE_LOG}"
echo "============================================================" | tee -a "${CONSOLE_LOG}"
echo "Skill Bridge Serial A/B — Complete" | tee -a "${CONSOLE_LOG}"
echo "  Phase 1 (control) exit code: ${PHASE1_RC}" | tee -a "${CONSOLE_LOG}"
echo "  Phase 2 (bridge)  exit code: ${PHASE2_RC}" | tee -a "${CONSOLE_LOG}"
echo "============================================================" | tee -a "${CONSOLE_LOG}"

# Exit with non-zero if either phase failed
if [[ "${PHASE1_RC}" -ne 0 || "${PHASE2_RC}" -ne 0 ]]; then
  echo "One or both phases failed!" >&2
  exit 1
fi
