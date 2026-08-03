#!/bin/bash
# Serial skill bridge A/B experiment: control (bridge OFF) → bridge (bridge ON)
# 8 nodes × 8 V100 = 64 ranks, FP32, same allocation
#
# Usage (Arnold multi-node):
#   CONFIG_NAME_1=pi05_ki_joint_query_b1k-single_task-radio-ki_on_control_fp32 \
#   CONFIG_NAME_2=pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_fp32 \
#   NUM_EPOCHS_1=2000 NUM_EPOCHS_2=2000 \
#   bash scripts/run_skill_bridge_v100_64rank_serial.sh
#
# NAS-mounted entrypoint — no git clone needed; runs directly from
# feat-skill-bridge worktree.

set -euo pipefail
set -x

REPO_ROOT="/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/feat-skill-bridge"
cd "${REPO_ROOT}"

source /mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas

# Ensure PyYAML is available (Arnold env may not have it by default)
ensure_yaml_available() {
  if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("yaml") else 1)
PY
  then
    return 0
  fi
  echo "[bootstrap] installing PyYAML" >&2
  conda install -y pyyaml 2>/dev/null || pip install PyYAML
}
ensure_yaml_available

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
export LD_LIBRARY_PATH="/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/lib/libstdc++.so.6"

# Offline mode — no internet access on compute nodes
export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Per-node local SSD cache for HF datasets (avoid NAS lock contention)
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/tiger/hf_datasets_cache/skill_bridge_v100_64rank/}"
export OPENPI_PERSISTENT_WORKERS="${OPENPI_PERSISTENT_WORKERS:-1}"
export OPENPI_DATALOADER_TIMEOUT_S="${OPENPI_DATALOADER_TIMEOUT_S:-600}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-2}"
export OPENPI_DDP_TIMEOUT_MIN="${OPENPI_DDP_TIMEOUT_MIN:-120}"
export OPENPI_LOAD_DATASET_NUM_PROC_CAP="${OPENPI_LOAD_DATASET_NUM_PROC_CAP:-8}"

# NCCL diagnostics
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_AF_INET6_ADDR_ENABLE=0
export NCCL_SOCKET_IFNAME=eth0

# Arnold environment variables → standard DDP vars
MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT="${ARNOLD_WORKER_0_PORT%%,*}"
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}
WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"

# Configs and step counts for the two serial phases
CONFIG_NAME_1="${CONFIG_NAME_1:-pi05_ki_joint_query_b1k-single_task-radio-ki_on_control_fp32}"
CONFIG_NAME_2="${CONFIG_NAME_2:-pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_fp32}"
NUM_EPOCHS_1="${NUM_EPOCHS_1:-2000}"
NUM_EPOCHS_2="${NUM_EPOCHS_2:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-float32}"

# Output root for both phases
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_ROOT}/outputs}"
LOGS_ROOT="${OUTPUTS_ROOT}/skill_bridge_v100_64rank_serial/logs"
mkdir -p "${LOGS_ROOT}"

RUN_KEY="${ARNOLD_JOB_ID:-skill_bridge_serial_$(date +%Y%m%d_%H%M%S)}"
RUN_KEY="${RUN_KEY//\//_}"
RUN_KEY="${RUN_KEY//:/_}"

CONSOLE_LOG="${LOGS_ROOT}/node${NODE_RANK}_${RUN_KEY}.log"

echo "=== Skill Bridge Serial A/B Experiment ==="
echo "Phase 1 (control):  ${CONFIG_NAME_1}  (${NUM_EPOCHS_1} epochs)"
echo "Phase 2 (bridge):   ${CONFIG_NAME_2}  (${NUM_EPOCHS_2} epochs)"
echo "Nodes: ${NNODES}  Ranks/node: ${NPROC_PER_NODE}  World size: ${WORLD_SIZE}"
echo "Node rank: ${NODE_RANK}  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Precision: ${PRECISION}"
echo "Outputs root: ${OUTPUTS_ROOT}"
echo "Log: ${CONSOLE_LOG}"

run_phase() {
  local phase_num=$1
  local config_name=$2
  local num_epochs=$3

  echo "--- Phase ${phase_num}: ${config_name} (${num_epochs} epochs) ---"

  accelerate launch \
    --multi_gpu \
    --num_machines "${NNODES}" \
    --machine_rank "${NODE_RANK}" \
    --num_processes "${WORLD_SIZE}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --dynamo_backend "no" \
    --mixed_precision "no" \
    scripts/train_accelerate.py \
    --config-name "${config_name}" \
    --num_train_epochs "${num_epochs}" \
    --log_interval "${LOG_INTERVAL}" \
    --save_interval "${SAVE_INTERVAL}" \
    --batch_size_per_gpu "${PER_GPU_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --pytorch-training-precision "${PRECISION}" \
    --overwrite \
    2>&1 | tee -a "${CONSOLE_LOG}"

  local exit_code=$?
  echo "--- Phase ${phase_num} finished with exit code ${exit_code} ---"
  return ${exit_code}
}

# Phase 1: Control (bridge disabled)
run_phase 1 "${CONFIG_NAME_1}" "${NUM_EPOCHS_1}"

# Cooldown between phases (let NCCL fully teardown)
echo "Cooldown: 90s between phases..."
sleep 90

# Phase 2: Bridge (bridge enabled) — same allocation, fresh run
run_phase 2 "${CONFIG_NAME_2}" "${NUM_EPOCHS_2}"

echo "=== Both phases complete ==="
