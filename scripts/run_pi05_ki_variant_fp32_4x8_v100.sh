#!/usr/bin/env bash
# Launch one π0.5-KI debug arm on its own 4-node x 8-GPU V100 allocation.
#
# Variant A: FAST action tokens + backbone cross-entropy.
# Variant B: learned action queries + backbone MSE.
#
# Select exactly one arm with the first argument or OPENPI_KI_ARM:
#   bash scripts/run_pi05_ki_variant_fp32_4x8_v100.sh A
#   OPENPI_KI_ARM=B bash scripts/run_pi05_ki_variant_fp32_4x8_v100.sh
#
# CPU-only preflight (never invokes nvidia-smi, Accelerate, or the keepalive wrapper):
#   OPENPI_LAUNCH_PREFLIGHT_ONLY=1 OPENPI_KI_ARM=A \
#     OPENPI_FAST_TOKENIZER_PATH=/absolute/path/to/physical-intelligence--fast \
#     OPENPI_PREFLIGHT_PYTHON=/absolute/path/to/python \
#     bash scripts/run_pi05_ki_variant_fp32_4x8_v100.sh
#
# The normal path performs fail-fast validation before handing training to the
# existing keepalive-on-failure wrapper. That wrapper deliberately has no
# `set -e`: after training exits, it records the result and holds the allocation.

set -euo pipefail

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
info() { printf '[pi05-ki-v100][%s] %s\n' "$(_ts)" "$*"; }
die() {
  printf '[pi05-ki-v100][%s] FATAL: %s\n' "$(_ts)" "$1" >&2
  shift || true
  for line in "$@"; do
    printf '[pi05-ki-v100]                  %s\n' "${line}" >&2
  done
  exit 2
}

if (( $# > 1 )); then
  die "expected at most one positional arm argument (A or B), got $#"
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)}"
KEEPALIVE_WRAPPER="${KEEPALIVE_WRAPPER:-${REPO_ROOT}/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh}"
TRAINER="${TRAINER:-${REPO_ROOT}/scripts/train_accelerate.py}"
ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/configs/accelerate_ds_zero2_v100_fp32.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2_v100_fp32.json}"

# Arm selection is explicit. Never infer an objective from a config-name typo.
ARG_ARM="${1:-}"
ENV_ARM="${OPENPI_KI_ARM:-}"
if [[ -n "${ARG_ARM}" && -n "${ENV_ARM}" && "${ARG_ARM,,}" != "${ENV_ARM,,}" ]]; then
  die "arm argument '${ARG_ARM}' conflicts with OPENPI_KI_ARM='${ENV_ARM}'"
fi
ARM_INPUT="${ARG_ARM:-${ENV_ARM}}"
case "${ARM_INPUT,,}" in
  a | variant-a | fast | fast-ce | fast_ce)
    ARM="A"
    ARM_LABEL="variantA_fast_ce"
    EXPECTED_CONFIG="pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32_debug"
    EXPECTED_MODEL="pi05_ki_joint_fast"
    ;;
  b | variant-b | query | query-mse | query_mse)
    ARM="B"
    ARM_LABEL="variantB_query_mse"
    EXPECTED_CONFIG="pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32_debug"
    EXPECTED_MODEL="pi05_ki_joint_query"
    ;;
  "")
    die "arm selection is required" \
        "Pass A/B as the first argument or set OPENPI_KI_ARM=A/B."
    ;;
  *)
    die "unknown arm '${ARM_INPUT}'" \
        "Expected A (FAST-CE) or B (query-MSE)."
    ;;
esac
export OPENPI_KI_ARM="${ARM}"

CONFIG_NAME="${CONFIG_NAME:-${EXPECTED_CONFIG}}"
if [[ "${CONFIG_NAME}" != "${EXPECTED_CONFIG}" ]]; then
  die "arm ${ARM} is restricted to config '${EXPECTED_CONFIG}', got '${CONFIG_NAME}'" \
      "Unknown names can silently fall back to pi05_b1k-base; refusing any override."
fi
export CONFIG_NAME

# Strict persistent-allocation topology: each arm owns an independent 4x8 task.
NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM:-}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES/ARNOLD_WORKER_NUM must be a positive integer"
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_NODE/ARNOLD_WORKER_GPU must be a positive integer"
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "NODE_RANK/ARNOLD_ID must be a non-negative integer"
(( NUM_NODES == 4 )) || die "expected exactly 4 nodes for one arm, got ${NUM_NODES}"
(( GPUS_PER_NODE == 8 )) || die "expected exactly 8 GPUs per node, got ${GPUS_PER_NODE}"
(( NODE_RANK < NUM_NODES )) || die "NODE_RANK=${NODE_RANK} must be less than NUM_NODES=${NUM_NODES}"
TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))

GPU_MODEL="${GPU_MODEL:-${ARNOLD_WORKER_GPU_TYPE:-${ARNOLD_GPU_TYPE:-}}}"
if [[ -n "${GPU_MODEL}" && "${GPU_MODEL^^}" != *V100* ]]; then
  die "this FP32 debug launcher is V100-only, but GPU model is '${GPU_MODEL}'"
fi

MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
_MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-29514}}"
MASTER_PORT="${_MASTER_PORT%%,*}"
[[ -n "${MASTER_ADDR}" ]] || die "MASTER_ADDR/ARNOLD_WORKER_0_HOST is required"
[[ "${MASTER_ADDR}" != *" "* ]] || die "MASTER_ADDR must be one host/IP, got '${MASTER_ADDR}'"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be numeric, got '${MASTER_PORT}'"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || die "MASTER_PORT must be in [1, 65535], got ${MASTER_PORT}"

# FP32 is mandatory on this path. The launcher does not accept precision drift.
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-float32}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
[[ "${PYTORCH_TRAINING_PRECISION}" == "float32" ]] \
  || die "V100 debug requires PYTORCH_TRAINING_PRECISION=float32; got '${PYTORCH_TRAINING_PRECISION}'"
[[ "${ACCELERATE_MIXED_PRECISION}" == "no" ]] \
  || die "V100 debug requires ACCELERATE_MIXED_PRECISION=no; got '${ACCELERATE_MIXED_PRECISION}'"
if [[ -n "${OPENPI_REUSE_PREFIX_KV:-}" && "${OPENPI_REUSE_PREFIX_KV}" != "0" ]]; then
  die "OPENPI_REUSE_PREFIX_KV must remain disabled for these debug runs"
fi
unset OPENPI_REUSE_PREFIX_KV
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLED="${WANDB_DISABLED:-0}"
[[ "${WANDB_MODE}" == "online" ]] || die "debug runs require WANDB_MODE=online; got '${WANDB_MODE}'"
[[ "${WANDB_DISABLED}" == "0" ]] || die "debug runs require WANDB_DISABLED=0; got '${WANDB_DISABLED}'"
export WANDB_MODE WANDB_DISABLED

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-5}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-1}"
for pair in \
  "BATCH_SIZE_PER_GPU:${BATCH_SIZE_PER_GPU}" \
  "GRADIENT_ACCUMULATION_STEPS:${GRADIENT_ACCUMULATION_STEPS}" \
  "NUM_TRAIN_STEPS:${NUM_TRAIN_STEPS}" \
  "NUM_WORKERS:${NUM_WORKERS}" \
  "SAVE_INTERVAL:${SAVE_INTERVAL}" \
  "VAL_LOG_INTERVAL:${VAL_LOG_INTERVAL}" \
  "VAL_NUM_BATCHES:${VAL_NUM_BATCHES}"; do
  key="${pair%%:*}"
  value="${pair#*:}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${key} must be a positive integer, got '${value}'"
done
(( BATCH_SIZE_PER_GPU == 1 )) \
  || die "this 32GB V100 smoke launcher is locked to BATCH_SIZE_PER_GPU=1; got ${BATCH_SIZE_PER_GPU}"
(( NUM_TRAIN_STEPS <= 100 )) \
  || die "debug budget must be at most 100 steps; got ${NUM_TRAIN_STEPS}"

BASE_PI05_CKPT="${BASE_PI05_CKPT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch}"
B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/data/2025-challenge-demos/}"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${BASE_PI05_CKPT}/assets/behavior-1k/2025-challenge-demos}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${B1K_ASSETS_DIR}/norm_stats.json}"
REPO_OPENPI_CACHE="${REPO_OPENPI_CACHE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/.cache/openpi}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-${REPO_OPENPI_CACHE}/big_vision/paligemma_tokenizer.model}"

[[ -s "${BASE_PI05_CKPT}/model.safetensors" ]] \
  || die "base π0.5 weights are missing: ${BASE_PI05_CKPT}/model.safetensors"
[[ -d "${B1K_DATASET_ROOT}" ]] || die "B1K dataset root is missing: ${B1K_DATASET_ROOT}"
[[ -s "${NORM_STATS_PATH}" ]] || die "B1K norm stats are missing: ${NORM_STATS_PATH}"
[[ -s "${PALIGEMMA_TOKENIZER}" ]] \
  || die "offline PaliGemma tokenizer is missing: ${PALIGEMMA_TOKENIZER}"
[[ -f "${TRAINER}" ]] || die "trainer not found: ${TRAINER}"
[[ -f "${ACCEL_CONFIG}" ]] || die "FP32 Accelerate config not found: ${ACCEL_CONFIG}"
[[ -f "${DEEPSPEED_CONFIG}" ]] || die "FP32 DeepSpeed config not found: ${DEEPSPEED_CONFIG}"
[[ -f "${KEEPALIVE_WRAPPER}" ]] || die "keepalive wrapper not found: ${KEEPALIVE_WRAPPER}"

OPENPI_PREFLIGHT_PYTHON="${OPENPI_PREFLIGHT_PYTHON:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python}"
[[ "${OPENPI_PREFLIGHT_PYTHON}" == /* && -x "${OPENPI_PREFLIGHT_PYTHON}" ]] \
  || die "OPENPI_PREFLIGHT_PYTHON must be an absolute executable path: ${OPENPI_PREFLIGHT_PYTHON}"

# With an external deepspeed_config_file, Accelerate 1.13 rejects precision
# fields duplicated at the top level. FP32 is enforced by DeepSpeed here and
# by the TrainConfig validation below, not by the Accelerate YAML.
"${OPENPI_PREFLIGHT_PYTHON}" - "${ACCEL_CONFIG}" "${DEEPSPEED_CONFIG}" <<'PY'
import json
from pathlib import Path
import re
import sys

accelerate_path = Path(sys.argv[1])
deepspeed_path = Path(sys.argv[2])
accelerate_text = accelerate_path.read_text()

if re.search(r'^mixed_precision\s*:', accelerate_text, re.MULTILINE):
    raise SystemExit(
        f"ERROR: {accelerate_path} must not define top-level mixed_precision "
        "when deepspeed_config_file is used"
    )
if "deepspeed_config_file: configs/deepspeed_zero2_v100_fp32.json" not in accelerate_text:
    raise SystemExit(f"ERROR: {accelerate_path} does not reference the FP32 DeepSpeed config")

ds = json.loads(deepspeed_path.read_text())
checks = {
    "fp16.enabled": ds.get("fp16", {}).get("enabled") is False,
    "bf16.enabled": ds.get("bf16", {}).get("enabled") is False,
    "torch_autocast.enabled": ds.get("torch_autocast", {}).get("enabled") is False,
    "torch_autocast.dtype=float32": ds.get("torch_autocast", {}).get("dtype") == "float32",
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"ERROR: {deepspeed_path} violates FP32-only settings: {', '.join(failed)}")
print("FP32_DISTRIBUTED_CONFIG_PREFLIGHT_OK accelerate_mixed_precision_key=absent")
PY

# get_config() silently returns pi05_b1k-base for unknown names. Compare the
# resolved name and every safety-critical field instead of trusting success.
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${OPENPI_PREFLIGHT_PYTHON}" - "${CONFIG_NAME}" "${EXPECTED_MODEL}" "${NUM_TRAIN_STEPS}" <<'PY'
import sys

from openpi.training.train_config import get_config

name, expected_model, expected_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
config = get_config(name)
checks = {
    "registered exact name": config.name == name,
    "expected model": config.pytorch_model_name == expected_model,
    "pytorch float32": config.pytorch_training_precision == "float32",
    "Accelerate mixed precision disabled": config.accelerate_mixed_precision == "no",
    "model dtype float32": config.model.dtype == "float32",
    "knowledge insulation enabled": config.model.knowledge_insulation is True,
    "expert KV truncation enabled": config.model.truncate_expert_kv is True,
    "Skill Bridge disabled (train)": config.data[0].base_config.skill_bridge.enabled is False,
    "Skill Bridge disabled (validation)": config.val_data[0].base_config.skill_bridge.enabled is False,
    "B1 per GPU": config.batch_size_per_gpu == 1,
    "config smoke budget": config.num_train_steps == 5,
}
failed = [label for label, ok in checks.items() if not ok]
if failed:
    raise SystemExit(
        f"ERROR: config {name!r} failed strict validation: " + "; ".join(failed)
    )
if expected_steps > 100:
    raise SystemExit(f"ERROR: requested runtime budget {expected_steps} exceeds debug limit")
print(f"CONFIG_PREFLIGHT_OK name={name} model={expected_model}")
PY

# Variant A must prove that the pre-cached FAST remote-code processor can load
# with network access disabled. Variant B intentionally does not need it.
export OPENPI_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="${REPO_OPENPI_CACHE}"
export OPENPI_BEHAVIOR_DATASET_ROOT="${B1K_DATASET_ROOT}"
if [[ "${ARM}" == "A" ]]; then
  OPENPI_FAST_TOKENIZER_PATH="${OPENPI_FAST_TOKENIZER_PATH:-}"
  [[ -n "${OPENPI_FAST_TOKENIZER_PATH}" ]] \
    || die "Variant A requires OPENPI_FAST_TOKENIZER_PATH" \
        "Set it to an absolute, pre-cached physical-intelligence/fast processor directory."
  [[ "${OPENPI_FAST_TOKENIZER_PATH}" == /* && -d "${OPENPI_FAST_TOKENIZER_PATH}" ]] \
    || die "OPENPI_FAST_TOKENIZER_PATH must be an existing absolute directory: ${OPENPI_FAST_TOKENIZER_PATH}"
  export OPENPI_FAST_TOKENIZER_PATH
  if ! "${OPENPI_PREFLIGHT_PYTHON}" - "${OPENPI_FAST_TOKENIZER_PATH}" <<'PY'
import sys
from transformers import AutoProcessor

path = sys.argv[1]
try:
    AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
except Exception as exc:
    raise SystemExit(
        f"ERROR: FAST processor cache at {path!r} cannot be loaded fully offline: {exc}\n"
        "Re-cache physical-intelligence/fast (including remote-code files) and retry."
    ) from exc
print(f"FAST_OFFLINE_PROCESSOR_PREFLIGHT_OK path={path}")
PY
  then
    die "Variant A FAST offline-cache preflight failed" \
        "No network fallback is permitted during the multi-node run."
  fi
else
  unset OPENPI_FAST_TOKENIZER_PATH
  info "Variant B selected: FAST processor cache is not required."
fi

GLOBAL_BATCH_SIZE=$(( BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS ))
JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-manual}}"
JOB_ID_SAFE="${JOB_ID//[^A-Za-z0-9_.-]/_}"
PERSISTENT_OUTPUT_BASE="${PERSISTENT_OUTPUT_BASE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_ki_v100_fp32_debug}"
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${PERSISTENT_OUTPUT_BASE}/${ARM_LABEL}/${JOB_ID_SAFE}}"
EXP_NAME="${EXP_NAME:-pi05_ki_${ARM_LABEL}_v100_fp32_4n8g_${JOB_ID_SAFE}}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/checkpoints}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/logs}"
CONSOLE_LOG_DIR="${PERSISTENT_OUTPUT_ROOT}/console_logs/${EXP_NAME}"
CONSOLE_LOG="${CONSOLE_LOG_DIR}/node${NODE_RANK}.log"

info "============================================================"
info "π0.5-KI 4x8 V100 FP32 debug arm ${ARM}"
info "objective=${ARM_LABEL} config=${CONFIG_NAME} model=${EXPECTED_MODEL}"
info "topology=${NUM_NODES}x${GPUS_PER_NODE} rank=${NODE_RANK} world=${TOTAL_GPUS}"
info "rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
info "precision=float32 / Accelerate no / DeepSpeed fp16=false,bf16=false"
info "budget=${NUM_TRAIN_STEPS} steps batch/GPU=${BATCH_SIZE_PER_GPU} global_batch=${GLOBAL_BATCH_SIZE}"
info "base_checkpoint=${BASE_PI05_CKPT}"
info "output_root=${PERSISTENT_OUTPUT_ROOT}"
info "prefix_kv_reuse=DISABLED"
info "============================================================"

# Exit before making output directories, inspecting GPU processes, invoking the
# wrapper, or launching Accelerate. This is a pure CPU/read-only preflight.
if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  info "PREFLIGHT_OK: no GPU, training, output-directory, or occupier action was performed."
  exit 0
fi

assert_no_occupiers() {
  local matches
  matches="$(ps -eo pid=,args= 2>/dev/null \
    | awk '/__GPU_OCCUPY__torch_mm_512/ && /gpu_occupy_(torch_mm[.]py|stub[.]sh)/ {print}' || true)"
  [[ -z "${matches}" ]] || die "GPU keepalive occupiers are still running on this node" \
      "Stop them with this allocation's node-local STOP file, verify they are absent, then retry." \
      "${matches}"
}
assert_no_occupiers

export NUM_NODES GPUS_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
export CONFIG_NAME PYTORCH_TRAINING_PRECISION ACCELERATE_MIXED_PRECISION
export BATCH_SIZE_PER_GPU GRADIENT_ACCUMULATION_STEPS NUM_TRAIN_STEPS NUM_WORKERS
export SAVE_INTERVAL VAL_LOG_INTERVAL VAL_NUM_BATCHES
export BASE_PI05_CKPT B1K_DATASET_ROOT B1K_ASSETS_DIR NORM_STATS_PATH REPO_OPENPI_CACHE
export PERSISTENT_OUTPUT_ROOT EXP_NAME ASSETS_BASE_DIR CHECKPOINT_BASE_DIR LOG_BASE_DIR
export ACCEL_CONFIG DEEPSPEED_CONFIG TRAINER

# The wrapper re-enters this script with OPENPI_KI_TRAINING_INNER=1. The inner
# path is the real fail-fast training launcher; the wrapper captures its rc.
if [[ "${OPENPI_KI_TRAINING_INNER:-0}" != "1" ]]; then
  mkdir -p "${PERSISTENT_OUTPUT_ROOT}"
  export OPENPI_KI_TRAINING_INNER=1
  export LAUNCHER="${SCRIPT_PATH}"
  export KEEPALIVE_STATE_DIR="${PERSISTENT_OUTPUT_ROOT}/keepalive"
  export OCCUPY_RUNTIME_DIR="${OCCUPY_RUNTIME_DIR:-/tmp/pi05_ki_${ARM_LABEL}_gpu_occupy}"
  export EXPECTED_GPUS_PER_NODE=8
  export KEEPALIVE_DISABLE="${KEEPALIVE_DISABLE:-0}"
  export KEEPALIVE_ON_SUCCESS="${KEEPALIVE_ON_SUCCESS:-1}"
  export STRICT_GPU_COUNT="${STRICT_GPU_COUNT:-0}"
  unset TRAIN_COMMAND
  info "handing training to keepalive wrapper: ${KEEPALIVE_WRAPPER}"
  exec bash "${KEEPALIVE_WRAPPER}"
fi

# Inner training path. Failures are expected to propagate to the non-fail-fast
# wrapper above, which records them and restarts idle keepalives.
CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3}"
CONDA_ENV="${CONDA_ENV:-openpi-comet-nas}"
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
[[ -f "${CONDA_SH}" ]] || die "conda initialization script not found: ${CONDA_SH}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export JAX_PLATFORMS=cpu

mkdir -p "${CONSOLE_LOG_DIR}" "${ASSETS_BASE_DIR}" "${CHECKPOINT_BASE_DIR}" "${LOG_BASE_DIR}"
cd "${REPO_ROOT}"

info "launching Accelerate; durable console log: ${CONSOLE_LOG}"
python -m accelerate.commands.launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${TOTAL_GPUS}" \
  --num_machines "${NUM_NODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --same_network \
  "${TRAINER}" \
  "${CONFIG_NAME}" \
  --pytorch-weight-path "${BASE_PI05_CKPT}" \
  --exp-name "${EXP_NAME}" \
  --pytorch-training-precision "${PYTORCH_TRAINING_PRECISION}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --batch-size-per-gpu "${BATCH_SIZE_PER_GPU}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --val-log-interval "${VAL_LOG_INTERVAL}" \
  --val-num-batches "${VAL_NUM_BATCHES}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  --log-base-dir "${LOG_BASE_DIR}" \
  2>&1 | tee -a "${CONSOLE_LOG}"
