#!/usr/bin/env bash
# Launch one formal π0.5-KI A/B arm on an independent 4-node x 8-GPU V100 allocation.
#
# The launcher is intentionally distinct from the five-step debug launcher. It
# is locked to the formal 104,912 optimizer-step, global-batch-256 contract:
# B1/GPU x world32 x GA8, FP32, DeepSpeed ZeRO-2 with CPU optimizer offload.
#
# Required immutable provenance hook:
#   OPENPI_EXPECTED_CODE_COMMIT=<40-character final commit SHA>
#
# CPU/read-only preflight:
#   OPENPI_LAUNCH_PREFLIGHT_ONLY=1 OPENPI_KI_ARM=B \
#   OPENPI_EXPECTED_CODE_COMMIT=$(git rev-parse HEAD) \
#   bash scripts/run_pi05_ki_formal_fp32_4x8_v100.sh
#
# FORMAL MERLIN ENTRYPOINT (required; do not point Merlin at this file):
#   OPENPI_KI_TRAINING_INNER=1 \
#   LAUNCHER=<repo>/scripts/run_pi05_ki_formal_fp32_4x8_v100.sh \
#   KEEPALIVE_DISABLE=0 KEEPALIVE_ON_SUCCESS=1 STRICT_GPU_COUNT=0 \
#   bash <repo>/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh
#
# The keepalive wrapper must be the outermost process so it captures launcher
# preflight failures as the training rc and keeps holding the allocation. This
# launcher refuses a normal run without OPENPI_KI_TRAINING_INNER=1, preventing
# the old nested-wrapper path where a preflight exit could release the job.

set -euo pipefail

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
info() { printf '[pi05-ki-formal-v100][%s] %s\n' "$(_ts)" "$*"; }
die() {
  printf '[pi05-ki-formal-v100][%s] FATAL: %s\n' "$(_ts)" "$1" >&2
  shift || true
  for line in "$@"; do
    printf '[pi05-ki-formal-v100]                         %s\n' "${line}" >&2
  done
  exit 2
}

if (( $# > 1 )); then
  die "expected at most one positional arm argument (A or B), got $#"
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)}"
TRAINER="${TRAINER:-${REPO_ROOT}/scripts/train_accelerate.py}"
ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/configs/accelerate_ds_zero2_v100_fp32.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2_v100_fp32.json}"
OPENPI_PREFLIGHT_PYTHON="${OPENPI_PREFLIGHT_PYTHON:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python}"

# This must be exported before any Python process imports torch. It supplements
# recursive activation checkpointing by reducing allocator fragmentation; it is
# not the primary OOM fix. Reject inherited alternatives so the formal runtime
# cannot silently drift to a configuration that lacks expandable segments.
FORMAL_CUDA_ALLOC_CONF="expandable_segments:True"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${FORMAL_CUDA_ALLOC_CONF}}"
[[ "${PYTORCH_CUDA_ALLOC_CONF}" == "${FORMAL_CUDA_ALLOC_CONF}" ]] \
  || die "formal V100 requires PYTORCH_CUDA_ALLOC_CONF=${FORMAL_CUDA_ALLOC_CONF}, got '${PYTORCH_CUDA_ALLOC_CONF}'"
export PYTORCH_CUDA_ALLOC_CONF

[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || die "REPO_ROOT is not a git worktree: ${REPO_ROOT}"
[[ -f "${TRAINER}" ]] || die "trainer not found: ${TRAINER}"
[[ -f "${ACCEL_CONFIG}" ]] || die "FP32 Accelerate config not found: ${ACCEL_CONFIG}"
[[ -f "${DEEPSPEED_CONFIG}" ]] || die "FP32 DeepSpeed config not found: ${DEEPSPEED_CONFIG}"
[[ "${OPENPI_PREFLIGHT_PYTHON}" == /* && -x "${OPENPI_PREFLIGHT_PYTHON}" ]] \
  || die "OPENPI_PREFLIGHT_PYTHON must be an absolute executable path: ${OPENPI_PREFLIGHT_PYTHON}"

# Immutable code identity is mandatory: no branch names, dirty trees or short
# hashes. The final commit is supplied by the runtime job definition because a
# tracked launcher cannot contain the SHA of the commit that contains itself.
EXPECTED_COMMIT="${OPENPI_EXPECTED_CODE_COMMIT:-}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || die "OPENPI_EXPECTED_CODE_COMMIT must be the exact 40-character final commit SHA"
ACTUAL_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] \
  || die "code provenance mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}"
DIRTY_STATUS="$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)"
[[ -z "${DIRTY_STATUS}" ]] \
  || die "formal launch requires a clean worktree at ${EXPECTED_COMMIT}" "${DIRTY_STATUS}"
export OPENPI_EXPECTED_CODE_COMMIT="${EXPECTED_COMMIT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
OPENPI_IMPORT_PATH="$("${OPENPI_PREFLIGHT_PYTHON}" - <<'PY'
import openpi
print(openpi.__file__)
PY
)"
[[ "${OPENPI_IMPORT_PATH}" == "${REPO_ROOT}/src/openpi/"* ]] \
  || die "openpi import does not resolve inside the pinned tree: ${OPENPI_IMPORT_PATH}"

ARG_ARM="${1:-}"
ENV_ARM="${OPENPI_KI_ARM:-}"
if [[ -n "${ARG_ARM}" && -n "${ENV_ARM}" && "${ARG_ARM,,}" != "${ENV_ARM,,}" ]]; then
  die "arm argument '${ARG_ARM}' conflicts with OPENPI_KI_ARM='${ENV_ARM}'"
fi
case "${ARG_ARM:-${ENV_ARM}}" in
  A | a | variant-a | fast | fast-ce | fast_ce)
    ARM="A"
    ARM_LABEL="variantA_fast_ce"
    EXPECTED_CONFIG="pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32"
    EXPECTED_MODEL="pi05_ki_joint_fast"
    ;;
  B | b | variant-b | query | query-mse | query_mse)
    ARM="B"
    ARM_LABEL="variantB_query_mse"
    EXPECTED_CONFIG="pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32"
    EXPECTED_MODEL="pi05_ki_joint_query"
    ;;
  "")
    die "arm selection is required" "Pass A/B or set OPENPI_KI_ARM=A/B."
    ;;
  *)
    die "unknown arm '${ARG_ARM:-${ENV_ARM}}'" "Expected A (FAST-CE) or B (query-MSE)."
    ;;
esac
export OPENPI_KI_ARM="${ARM}"
CONFIG_NAME="${CONFIG_NAME:-${EXPECTED_CONFIG}}"
[[ "${CONFIG_NAME}" == "${EXPECTED_CONFIG}" ]] \
  || die "arm ${ARM} is restricted to formal config '${EXPECTED_CONFIG}', got '${CONFIG_NAME}'" \
      "No debug config or unknown-name fallback is permitted."
export CONFIG_NAME

NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM:-}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES/ARNOLD_WORKER_NUM must be a positive integer"
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_NODE/ARNOLD_WORKER_GPU must be a positive integer"
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "NODE_RANK/ARNOLD_ID must be a non-negative integer"
(( NUM_NODES == 4 )) || die "formal arm requires exactly 4 nodes, got ${NUM_NODES}"
(( GPUS_PER_NODE == 8 )) || die "formal arm requires exactly 8 GPUs per node, got ${GPUS_PER_NODE}"
(( NODE_RANK < NUM_NODES )) || die "NODE_RANK=${NODE_RANK} must be less than NUM_NODES=${NUM_NODES}"
TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))
GPU_MODEL="${GPU_MODEL:-${ARNOLD_WORKER_GPU_TYPE:-${ARNOLD_GPU_TYPE:-}}}"
[[ -z "${GPU_MODEL}" || "${GPU_MODEL^^}" == *V100* ]] \
  || die "formal FP32 launcher is V100-only, but GPU model is '${GPU_MODEL}'"

MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
_MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-29514}}"
MASTER_PORT="${_MASTER_PORT%%,*}"
[[ -n "${MASTER_ADDR}" && "${MASTER_ADDR}" != *" "* ]] || die "MASTER_ADDR must be one non-empty host/IP"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be numeric, got '${MASTER_PORT}'"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || die "MASTER_PORT must be in [1, 65535]"

PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-float32}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-104912}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
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
[[ "${PYTORCH_TRAINING_PRECISION}" == "float32" ]] || die "formal V100 requires FP32"
[[ "${ACCELERATE_MIXED_PRECISION}" == "no" ]] || die "formal V100 requires Accelerate mixed precision disabled"
(( BATCH_SIZE_PER_GPU == 1 )) || die "formal V100 requires BATCH_SIZE_PER_GPU=1"
(( GRADIENT_ACCUMULATION_STEPS == 8 )) || die "formal V100 requires GRADIENT_ACCUMULATION_STEPS=8"
(( NUM_TRAIN_STEPS == 104912 )) || die "formal V100 requires NUM_TRAIN_STEPS=104912"
(( SAVE_INTERVAL == 10000 )) || die "formal V100 requires SAVE_INTERVAL=10000"
(( VAL_LOG_INTERVAL == 1000 )) || die "formal V100 requires VAL_LOG_INTERVAL=1000"
(( VAL_NUM_BATCHES == 20 )) || die "formal V100 requires VAL_NUM_BATCHES=20"
GLOBAL_BATCH_SIZE=$(( BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS ))
(( GLOBAL_BATCH_SIZE == 256 )) || die "formal effective global batch must be 256, got ${GLOBAL_BATCH_SIZE}"
if [[ -n "${OPENPI_REUSE_PREFIX_KV:-}" && "${OPENPI_REUSE_PREFIX_KV}" != "0" ]]; then
  die "OPENPI_REUSE_PREFIX_KV remains HOLD and must be disabled"
fi
unset OPENPI_REUSE_PREFIX_KV
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLED="${WANDB_DISABLED:-0}"
[[ "${WANDB_MODE}" == "online" ]] || die "formal runs require WANDB_MODE=online"
[[ "${WANDB_DISABLED}" == "0" ]] || die "formal runs require WANDB_DISABLED=0"
export WANDB_MODE WANDB_DISABLED

FORMAL_BASE_PI05_CKPT="/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch"
FORMAL_B1K_DATASET_ROOT="/mnt/bn/saiwenresearch/mlx/users/chenjunting/data/2025-challenge-demos/"
FORMAL_B1K_ASSETS_DIR="${FORMAL_BASE_PI05_CKPT}/assets/behavior-1k/2025-challenge-demos"
FORMAL_NORM_STATS_PATH="${FORMAL_B1K_ASSETS_DIR}/norm_stats.json"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${FORMAL_BASE_PI05_CKPT}}"
B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-${FORMAL_B1K_DATASET_ROOT}}"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${FORMAL_B1K_ASSETS_DIR}}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${FORMAL_NORM_STATS_PATH}}"
REPO_OPENPI_CACHE="${REPO_OPENPI_CACHE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/.cache/openpi}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-${REPO_OPENPI_CACHE}/big_vision/paligemma_tokenizer.model}"
[[ -s "${BASE_PI05_CKPT}/model.safetensors" ]] || die "base weights missing: ${BASE_PI05_CKPT}/model.safetensors"
[[ -d "${B1K_DATASET_ROOT}" ]] || die "B1K dataset root missing: ${B1K_DATASET_ROOT}"
[[ -s "${NORM_STATS_PATH}" ]] || die "B1K norm stats missing: ${NORM_STATS_PATH}"
[[ -s "${PALIGEMMA_TOKENIZER}" ]] || die "offline PaliGemma tokenizer missing: ${PALIGEMMA_TOKENIZER}"

"${OPENPI_PREFLIGHT_PYTHON}" - "${ACCEL_CONFIG}" "${DEEPSPEED_CONFIG}" <<'PY'
import json
from pathlib import Path
import re
import sys

accelerate_path = Path(sys.argv[1])
deep_path = Path(sys.argv[2])
accelerate_text = accelerate_path.read_text()
if re.search(r"^mixed_precision\s*:", accelerate_text, re.MULTILINE):
    raise SystemExit("ERROR: Accelerate config duplicates mixed_precision with deepspeed_config_file")
if "deepspeed_config_file: configs/deepspeed_zero2_v100_fp32.json" not in accelerate_text:
    raise SystemExit("ERROR: Accelerate config does not reference the formal FP32 DeepSpeed file")
ds = json.loads(deep_path.read_text())
checks = {
    "ZeRO stage 2": ds.get("zero_optimization", {}).get("stage") == 2,
    "CPU optimizer offload": ds.get("zero_optimization", {}).get("offload_optimizer", {}).get("device") == "cpu",
    "fp16 disabled": ds.get("fp16", {}).get("enabled") is False,
    "bf16 disabled": ds.get("bf16", {}).get("enabled") is False,
    "torch autocast disabled": ds.get("torch_autocast", {}).get("enabled") is False,
    "GA is auto": ds.get("gradient_accumulation_steps") == "auto",
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("ERROR: formal DeepSpeed contract failed: " + ", ".join(failed))
print("FORMAL_FP32_ZERO2_PREFLIGHT_OK stage=2 offload_optimizer=cpu zero3=disabled")
PY

"${OPENPI_PREFLIGHT_PYTHON}" - "${CONFIG_NAME}" "${EXPECTED_MODEL}" <<'PY'
import sys
from openpi.training.train_config import get_config

name, expected_model = sys.argv[1:]
config = get_config(name)
checks = {
    "registered exact name": config.name == name,
    "expected model": config.pytorch_model_name == expected_model,
    "float32": config.pytorch_training_precision == "float32",
    "Accelerate precision no": config.accelerate_mixed_precision == "no",
    "model float32": config.model.dtype == "float32",
    "KI enabled": config.model.knowledge_insulation is True,
    "expert KV truncated": config.model.truncate_expert_kv is True,
    "Skill Bridge disabled train": config.data[0].base_config.skill_bridge.enabled is False,
    "Skill Bridge disabled val": config.val_data[0].base_config.skill_bridge.enabled is False,
    "B1": config.batch_size_per_gpu == 1,
    "GA8": config.gradient_accumulation_steps == 8,
    "104912 steps": config.num_train_steps == 104_912,
    "fixed-step mode": config.num_train_epochs is None,
    "stride12": config.streaming_anchor_stride == 12,
    "save10k": config.save_interval == 10_000,
    "val1k": config.val_log_interval == 1_000,
    "val20": config.val_num_batches == 20,
    "warmup1000": config.lr_schedule.warmup_steps == 1_000,
    "peak1e-5": config.lr_schedule.peak_lr == 1e-5,
    "decay104912": config.lr_schedule.decay_steps == 104_912,
    "decay0": config.lr_schedule.decay_lr == 0.0,
    "online W&B": config.wandb_enabled is True and config.project_name == "pi05_ki",
}
if expected_model == "pi05_ki_joint_fast":
    checks["FAST action capacity 208"] = config.model.action_token_max_len == 208
else:
    checks["query arm has no FAST target"] = not hasattr(config.model, "action_token_max_len")
failed = [label for label, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"ERROR: formal config {name!r} failed: " + "; ".join(failed))
print(f"FORMAL_CONFIG_PREFLIGHT_OK name={name} model={expected_model} B1xW32xGA8=256")
PY

export OPENPI_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="${REPO_OPENPI_CACHE}"
export OPENPI_BEHAVIOR_DATASET_ROOT="${B1K_DATASET_ROOT}"
if [[ "${ARM}" == "A" ]]; then
  OPENPI_FAST_TOKENIZER_PATH="${OPENPI_FAST_TOKENIZER_PATH:-}"
  [[ "${OPENPI_FAST_TOKENIZER_PATH}" == /* && -d "${OPENPI_FAST_TOKENIZER_PATH}" ]] \
    || die "Variant A requires an absolute pre-cached OPENPI_FAST_TOKENIZER_PATH"
  export OPENPI_FAST_TOKENIZER_PATH
  "${OPENPI_PREFLIGHT_PYTHON}" - "${OPENPI_FAST_TOKENIZER_PATH}" <<'PY'
import sys
from transformers import AutoProcessor
path = sys.argv[1]
AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
print(f"FAST_OFFLINE_PROCESSOR_PREFLIGHT_OK path={path}")
PY
else
  unset OPENPI_FAST_TOKENIZER_PATH
  info "Variant B selected: FAST processor cache is not required."
fi

JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-manual}}"
JOB_ID_SAFE="${JOB_ID//[^A-Za-z0-9_.-]/_}"
PERSISTENT_OUTPUT_BASE="${PERSISTENT_OUTPUT_BASE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_ki_v100_fp32_formal}"
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${PERSISTENT_OUTPUT_BASE}/${ARM_LABEL}/${JOB_ID_SAFE}}"
EXP_NAME="${EXP_NAME:-pi05_ki_formal_${ARM_LABEL}_v100_fp32_4n8g_${JOB_ID_SAFE}}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/checkpoints}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/logs}"
CONSOLE_LOG_DIR="${PERSISTENT_OUTPUT_ROOT}/console_logs/${EXP_NAME}"
CONSOLE_LOG="${CONSOLE_LOG_DIR}/node${NODE_RANK}.log"

info "============================================================"
info "formal arm=${ARM} objective=${ARM_LABEL} config=${CONFIG_NAME}"
info "code_commit=${ACTUAL_COMMIT} openpi.__file__=${OPENPI_IMPORT_PATH}"
info "topology=${NUM_NODES}x${GPUS_PER_NODE} world=${TOTAL_GPUS} rank=${NODE_RANK}"
info "FP32 ZeRO-2 CPU-offload B1/GPU GA8 global_batch=${GLOBAL_BATCH_SIZE}"
info "steps=104912 passes=34982/34971/34959 boundaries=34982/69953/104912"
info "save=10000 validation=1000x20 W&B=online project=pi05_ki"
info "output_root=${PERSISTENT_OUTPUT_ROOT} prefix_kv_reuse=DISABLED"
info "============================================================"

if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  info "PREFLIGHT_OK: CPU/read-only checks passed; no GPU, output or occupier action was performed."
  exit 0
fi

[[ "${OPENPI_KI_TRAINING_INNER:-0}" == "1" ]] \
  || die "formal Merlin entrypoint must be the outer keepalive wrapper" \
      "Set OPENPI_KI_TRAINING_INNER=1 and LAUNCHER=${SCRIPT_PATH} on the wrapper."
[[ "${KEEPALIVE_DISABLE:-}" == "0" ]] \
  || die "outer keepalive wrapper requires KEEPALIVE_DISABLE=0"
[[ "${KEEPALIVE_ON_SUCCESS:-}" == "1" ]] \
  || die "outer keepalive wrapper requires KEEPALIVE_ON_SUCCESS=1"
[[ "${STRICT_GPU_COUNT:-}" == "0" ]] \
  || die "outer keepalive wrapper requires STRICT_GPU_COUNT=0"

# The formal configs embed these verified LQ paths. Runtime path substitution is
# intentionally rejected until it can update every coupled data/assets field;
# accepting a partial override would silently break A/B input identity.
[[ "${BASE_PI05_CKPT}" == "${FORMAL_BASE_PI05_CKPT}" ]] \
  || die "formal runtime base checkpoint is pinned to ${FORMAL_BASE_PI05_CKPT}"
[[ "${B1K_DATASET_ROOT}" == "${FORMAL_B1K_DATASET_ROOT}" ]] \
  || die "formal runtime dataset root is pinned to ${FORMAL_B1K_DATASET_ROOT}"
[[ "${B1K_ASSETS_DIR}" == "${FORMAL_B1K_ASSETS_DIR}" ]] \
  || die "formal runtime assets are pinned to ${FORMAL_B1K_ASSETS_DIR}"
[[ "${NORM_STATS_PATH}" == "${FORMAL_NORM_STATS_PATH}" ]] \
  || die "formal runtime norm stats are pinned to ${FORMAL_NORM_STATS_PATH}"

assert_no_occupiers() {
  local matches
  matches="$(ps -eo pid=,args= 2>/dev/null \
    | awk '/__GPU_OCCUPY__torch_mm_512/ && /gpu_occupy_(torch_mm[.]py|stub[.]sh)/ {print}' || true)"
  [[ -z "${matches}" ]] || die "GPU keepalive occupiers are still running" \
    "Stop this allocation's exact alias and verify absence before launch." "${matches}"
}
assert_no_occupiers

export NUM_NODES GPUS_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
export CONFIG_NAME PYTORCH_TRAINING_PRECISION ACCELERATE_MIXED_PRECISION
export BATCH_SIZE_PER_GPU GRADIENT_ACCUMULATION_STEPS NUM_TRAIN_STEPS NUM_WORKERS
export SAVE_INTERVAL VAL_LOG_INTERVAL VAL_NUM_BATCHES
export BASE_PI05_CKPT B1K_DATASET_ROOT B1K_ASSETS_DIR NORM_STATS_PATH REPO_OPENPI_CACHE
export PERSISTENT_OUTPUT_ROOT EXP_NAME ASSETS_BASE_DIR CHECKPOINT_BASE_DIR LOG_BASE_DIR
export ACCEL_CONFIG DEEPSPEED_CONFIG TRAINER REPO_ROOT

CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3}"
CONDA_ENV="${CONDA_ENV:-openpi-comet-nas}"
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
[[ -f "${CONDA_SH}" ]] || die "conda initialization script not found: ${CONDA_SH}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
unset PYTHONHOME
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${OPENPI_EXPECTED_CODE_COMMIT}" ]] || die "commit changed before inner launch"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree became dirty before inner launch"
python - <<'PY'
import os
import openpi
print(f"CODE_PROVENANCE commit={os.environ['OPENPI_EXPECTED_CODE_COMMIT']} openpi.__file__={openpi.__file__}")
PY
mkdir -p "${CONSOLE_LOG_DIR}" "${ASSETS_BASE_DIR}" "${CHECKPOINT_BASE_DIR}" "${LOG_BASE_DIR}"

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
  --pytorch-training-precision float32 \
  --num-train-steps 104912 \
  --batch-size-per-gpu 1 \
  --gradient-accumulation-steps 8 \
  --num-workers "${NUM_WORKERS}" \
  --save-interval 10000 \
  --val-log-interval 1000 \
  --val-num-batches 20 \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  --log-base-dir "${LOG_BASE_DIR}" \
  2>&1 | tee -a "${CONSOLE_LOG}"
