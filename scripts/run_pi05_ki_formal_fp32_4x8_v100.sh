#!/usr/bin/env bash
# Launch one registered formal/validation π0.5-KI profile on 4 x 8 V100 GPUs.
# TrainConfig owns the training recipe; this shell owns topology, paths, code
# identity, backend policy validation, and the outer keepalive contract.

set -euo pipefail

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
info() { printf '[pi05-ki-formal-v100][%s] %s\n' "$(_ts)" "$*"; }
die() {
  printf '[pi05-ki-formal-v100][%s] FATAL: %s\n' "$(_ts)" "$1" >&2
  shift || true
  for line in "$@"; do printf '[pi05-ki-formal-v100] %s\n' "$line" >&2; done
  exit 2
}

if (( $# > 1 )); then die "expected at most one positional arm argument (A or B), got $#"; fi
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)}"
TRAINER="${TRAINER:-${REPO_ROOT}/scripts/train_accelerate.py}"
ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/configs/accelerate_ds_zero2_v100_fp32.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2_v100_fp32.json}"
OPENPI_PREFLIGHT_PYTHON="${OPENPI_PREFLIGHT_PYTHON:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python}"

FORMAL_CUDA_ALLOC_CONF="expandable_segments:True"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${FORMAL_CUDA_ALLOC_CONF}}"
[[ "${PYTORCH_CUDA_ALLOC_CONF}" == "${FORMAL_CUDA_ALLOC_CONF}" ]] \
  || die "formal V100 requires PYTORCH_CUDA_ALLOC_CONF=${FORMAL_CUDA_ALLOC_CONF}, got '${PYTORCH_CUDA_ALLOC_CONF}'"
export PYTORCH_CUDA_ALLOC_CONF

[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || die "REPO_ROOT is not a git worktree: ${REPO_ROOT}"
for path in "${TRAINER}" "${ACCEL_CONFIG}" "${DEEPSPEED_CONFIG}"; do [[ -f "$path" ]] || die "required file not found: $path"; done
[[ "${OPENPI_PREFLIGHT_PYTHON}" == /* && -x "${OPENPI_PREFLIGHT_PYTHON}" ]] \
  || die "OPENPI_PREFLIGHT_PYTHON must be an absolute executable path: ${OPENPI_PREFLIGHT_PYTHON}"

EXPECTED_COMMIT="${OPENPI_EXPECTED_CODE_COMMIT:-}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "OPENPI_EXPECTED_CODE_COMMIT must be the exact 40-character final commit SHA"
ACTUAL_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || die "code provenance mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}"
DIRTY_STATUS="$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)"
[[ -z "${DIRTY_STATUS}" ]] || die "formal launch requires a clean worktree at ${EXPECTED_COMMIT}" "${DIRTY_STATUS}"
export OPENPI_EXPECTED_CODE_COMMIT="${EXPECTED_COMMIT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
OPENPI_IMPORT_PATH="$("${OPENPI_PREFLIGHT_PYTHON}" - <<'PY'
import openpi
print(openpi.__file__)
PY
)"
[[ "${OPENPI_IMPORT_PATH}" == "${REPO_ROOT}/src/openpi/"* ]] || die "openpi import does not resolve inside the pinned tree: ${OPENPI_IMPORT_PATH}"

ARG_ARM="${1:-}"
ENV_ARM="${OPENPI_KI_ARM:-}"
if [[ -n "${ARG_ARM}" && -n "${ENV_ARM}" && "${ARG_ARM,,}" != "${ENV_ARM,,}" ]]; then
  die "arm argument '${ARG_ARM}' conflicts with OPENPI_KI_ARM='${ENV_ARM}'"
fi
case "${ARG_ARM:-${ENV_ARM}}" in
  A|a|variant-a|fast|fast-ce|fast_ce)
    ARM=A; ARM_LABEL=variantA_fast_ce; DEFAULT_CONFIG=pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32; EXPECTED_MODEL=pi05_ki_joint_fast ;;
  B|b|variant-b|query|query-mse|query_mse)
    ARM=B; ARM_LABEL=variantB_query_mse; DEFAULT_CONFIG=pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32; EXPECTED_MODEL=pi05_ki_joint_query ;;
  "") die "arm selection is required" "Pass A/B or set OPENPI_KI_ARM=A/B." ;;
  *) die "unknown arm '${ARG_ARM:-${ENV_ARM}}'" ;;
esac
export OPENPI_KI_ARM="${ARM}"
CONFIG_NAME="${CONFIG_NAME:-${DEFAULT_CONFIG}}"
case "${ARM}:${CONFIG_NAME}" in
  A:pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32|A:pi05_ki_joint_fast_b1k-full_task-ki_on_v100_fp32_validation10|B:pi05_ki_joint_query_b1k-full_task-ki_on_v100_fp32) ;;
  *) die "arm ${ARM} does not allow config '${CONFIG_NAME}'; refusing unknown/mismatched profile" ;;
esac
export CONFIG_NAME

eval "$("${OPENPI_PREFLIGHT_PYTHON}" -m openpi.training.launcher_profile "${CONFIG_NAME}" --expected-model "${EXPECTED_MODEL}" --world-size 32 --format shell)" \
  || die "failed to resolve exact registered TrainConfig ${CONFIG_NAME}"
[[ "${CFG_NAME}" == "${CONFIG_NAME}" ]] || die "resolved config identity mismatch: ${CFG_NAME} != ${CONFIG_NAME}"

NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM:-}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ && "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "invalid topology"
(( NUM_NODES == 4 && GPUS_PER_NODE == 8 && NODE_RANK < NUM_NODES )) || die "V100 profile requires 4 nodes x 8 GPUs"
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
GPU_MODEL="${GPU_MODEL:-${ARNOLD_WORKER_GPU_TYPE:-${ARNOLD_GPU_TYPE:-}}}"
[[ -z "${GPU_MODEL}" || "${GPU_MODEL^^}" == *V100* ]] || die "formal FP32 launcher is V100-only, got '${GPU_MODEL}'"
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
_MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-29514}}"; MASTER_PORT="${_MASTER_PORT%%,*}"
[[ -n "${MASTER_ADDR}" && "${MASTER_ADDR}" != *" "* && "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "invalid rendezvous"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || die "MASTER_PORT must be in [1,65535]"
GLOBAL_BATCH_SIZE=$((CFG_BATCH_SIZE_PER_GPU * TOTAL_GPUS * CFG_GRADIENT_ACCUMULATION_STEPS))
[[ -n "${CFG_EXPECTED_GLOBAL_BATCH}" && "${GLOBAL_BATCH_SIZE}" -eq "${CFG_EXPECTED_GLOBAL_BATCH}" ]] \
  || die "profile ${CFG_NAME} requires global batch ${CFG_EXPECTED_GLOBAL_BATCH}, got ${GLOBAL_BATCH_SIZE}"

if [[ -n "${OPENPI_REUSE_PREFIX_KV:-}" && "${OPENPI_REUSE_PREFIX_KV}" != 0 ]]; then die "OPENPI_REUSE_PREFIX_KV remains HOLD and must be disabled"; fi
unset OPENPI_REUSE_PREFIX_KV
WANDB_MODE="${WANDB_MODE:-online}"; WANDB_DISABLED="${WANDB_DISABLED:-0}"
[[ "${WANDB_MODE}" == online && "${WANDB_DISABLED}" == 0 && "${CFG_WANDB_ENABLED}" == 1 ]] || die "registered profile requires online W&B"
export WANDB_MODE WANDB_DISABLED

FORMAL_BASE_PI05_CKPT="/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch"
FORMAL_B1K_DATASET_ROOT="/mnt/bn/saiwenresearch/mlx/users/chenjunting/data/2025-challenge-demos/"
FORMAL_B1K_ASSETS_DIR="${FORMAL_BASE_PI05_CKPT}/assets/behavior-1k/2025-challenge-demos"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${FORMAL_BASE_PI05_CKPT}}"
B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-${FORMAL_B1K_DATASET_ROOT}}"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${FORMAL_B1K_ASSETS_DIR}}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${B1K_ASSETS_DIR}/norm_stats.json}"
REPO_OPENPI_CACHE="${REPO_OPENPI_CACHE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/.cache/openpi}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-${REPO_OPENPI_CACHE}/big_vision/paligemma_tokenizer.model}"
[[ "${CFG_WEIGHT_PATH}" == "${FORMAL_BASE_PI05_CKPT}" && "${CFG_TRAIN_DATA_ROOT}" == "${FORMAL_B1K_DATASET_ROOT}" && "${CFG_TRAIN_ASSETS_DIR}" == "${FORMAL_BASE_PI05_CKPT}/assets" ]] \
  || die "registered V100 profile changed pinned input identity"
for path in "${BASE_PI05_CKPT}/model.safetensors" "${NORM_STATS_PATH}" "${PALIGEMMA_TOKENIZER}"; do [[ -s "$path" ]] || die "required input missing: $path"; done

"${OPENPI_PREFLIGHT_PYTHON}" - "${ACCEL_CONFIG}" "${DEEPSPEED_CONFIG}" <<'PY'
import json, re, sys
from pathlib import Path
acc = Path(sys.argv[1]).read_text()
if re.search(r"^mixed_precision\s*:", acc, re.MULTILINE): raise SystemExit("ERROR: duplicate Accelerate precision")
if "deepspeed_config_file: configs/deepspeed_zero2_v100_fp32.json" not in acc: raise SystemExit("ERROR: wrong DeepSpeed file")
ds = json.loads(Path(sys.argv[2]).read_text())
checks = {
    "ZeRO stage 2": ds.get("zero_optimization", {}).get("stage") == 2,
    "CPU optimizer offload": ds.get("zero_optimization", {}).get("offload_optimizer", {}).get("device") == "cpu",
    "fp16 disabled": ds.get("fp16", {}).get("enabled") is False,
    "bf16 disabled": ds.get("bf16", {}).get("enabled") is False,
    "torch autocast disabled": ds.get("torch_autocast", {}).get("enabled") is False,
    "batch auto": ds.get("train_batch_size") == "auto" and ds.get("train_micro_batch_size_per_gpu") == "auto",
    "GA auto": ds.get("gradient_accumulation_steps") == "auto",
}
failed = [name for name, ok in checks.items() if not ok]
if failed: raise SystemExit("ERROR: V100 backend contract failed: " + ", ".join(failed))
print("FORMAL_FP32_ZERO2_PREFLIGHT_OK stage=2 offload_optimizer=cpu batch=auto ga=auto")
PY

export OPENPI_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="${REPO_OPENPI_CACHE}" OPENPI_BEHAVIOR_DATASET_ROOT="${B1K_DATASET_ROOT}"
if [[ "${ARM}" == A ]]; then
  OPENPI_FAST_TOKENIZER_PATH="${OPENPI_FAST_TOKENIZER_PATH:-}"
  [[ "${OPENPI_FAST_TOKENIZER_PATH}" == /* && -d "${OPENPI_FAST_TOKENIZER_PATH}" ]] || die "Variant A requires an absolute pre-cached OPENPI_FAST_TOKENIZER_PATH"
  export OPENPI_FAST_TOKENIZER_PATH
  "${OPENPI_PREFLIGHT_PYTHON}" - "${OPENPI_FAST_TOKENIZER_PATH}" <<'PY'
import sys
from transformers import AutoProcessor
AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
print(f"FAST_OFFLINE_PROCESSOR_PREFLIGHT_OK path={sys.argv[1]}")
PY
else
  unset OPENPI_FAST_TOKENIZER_PATH
fi

JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-manual}}"; JOB_ID_SAFE="${JOB_ID//[^A-Za-z0-9_.-]/_}"
PERSISTENT_OUTPUT_BASE="${PERSISTENT_OUTPUT_BASE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_ki_v100_fp32_formal}"
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${PERSISTENT_OUTPUT_BASE}/${ARM_LABEL}/${JOB_ID_SAFE}}"
EXP_NAME="${EXP_NAME:-pi05_ki_${CONFIG_NAME}_${JOB_ID_SAFE}}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/checkpoints}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/logs}"
CONSOLE_LOG_DIR="${PERSISTENT_OUTPUT_ROOT}/console_logs/${EXP_NAME}"; CONSOLE_LOG="${CONSOLE_LOG_DIR}/node${NODE_RANK}.log"

info "profile=${CFG_NAME} model=${CFG_MODEL_NAME} arm=${ARM}"
info "code_commit=${ACTUAL_COMMIT} openpi=${OPENPI_IMPORT_PATH} topology=${NUM_NODES}x${GPUS_PER_NODE} world=${TOTAL_GPUS} rank=${NODE_RANK}"
info "precision=${CFG_PYTORCH_TRAINING_PRECISION}/accelerate-${CFG_ACCELERATE_MIXED_PRECISION} B${CFG_BATCH_SIZE_PER_GPU}xW${TOTAL_GPUS}xGA${CFG_GRADIENT_ACCUMULATION_STEPS}=${GLOBAL_BATCH_SIZE}"
info "budget_steps=${CFG_NUM_TRAIN_STEPS} epochs=${CFG_NUM_TRAIN_EPOCHS:-None} stride=${CFG_STREAMING_ANCHOR_STRIDE} warmup=${CFG_WARMUP_STEPS} peak_lr=${CFG_PEAK_LR} decay=${CFG_DECAY_STEPS}->${CFG_DECAY_LR}"
info "save=${CFG_SAVE_INTERVAL} val=${CFG_VAL_LOG_INTERVAL}x${CFG_VAL_NUM_BATCHES} workers=${CFG_NUM_WORKERS} cap=${CFG_ACTION_TOKEN_MAX_LEN:-N/A} project=${CFG_PROJECT_NAME}"
info "output_root=${PERSISTENT_OUTPUT_ROOT} prefix_kv_reuse=DISABLED"

if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  info "PREFLIGHT_OK: resolved registered profile; no GPU, output, or occupier action."
  exit 0
fi
[[ "${OPENPI_KI_TRAINING_INNER:-0}" == 1 ]] || die "formal Merlin entrypoint must be the outer keepalive wrapper" "Set OPENPI_KI_TRAINING_INNER=1 and LAUNCHER=${SCRIPT_PATH}."
[[ "${KEEPALIVE_DISABLE:-}" == 0 && "${KEEPALIVE_ON_SUCCESS:-}" == 1 && "${STRICT_GPU_COUNT:-}" == 0 ]] || die "outer wrapper contract mismatch"
assert_no_occupiers() {
  local matches
  matches="$(ps -eo pid=,args= 2>/dev/null | awk '/__GPU_OCCUPY__torch_mm_512/ && /gpu_occupy_(torch_mm[.]py|stub[.]sh)/ {print}' || true)"
  [[ -z "${matches}" ]] || die "GPU keepalive occupiers are still running" "${matches}"
}
assert_no_occupiers

export NUM_NODES GPUS_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT CONFIG_NAME
export BASE_PI05_CKPT B1K_DATASET_ROOT B1K_ASSETS_DIR NORM_STATS_PATH REPO_OPENPI_CACHE
export PERSISTENT_OUTPUT_ROOT EXP_NAME ASSETS_BASE_DIR CHECKPOINT_BASE_DIR LOG_BASE_DIR ACCEL_CONFIG DEEPSPEED_CONFIG TRAINER REPO_ROOT
CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3}"; CONDA_ENV="${CONDA_ENV:-openpi-comet-nas}"; CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
[[ -f "${CONDA_SH}" ]] || die "conda initialization script not found: ${CONDA_SH}"
# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
unset PYTHONHOME
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${OPENPI_EXPECTED_CODE_COMMIT}" && -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree identity changed before launch"
python - <<'PY'
import os, openpi
print(f"CODE_PROVENANCE commit={os.environ['OPENPI_EXPECTED_CODE_COMMIT']} openpi.__file__={openpi.__file__}")
PY
mkdir -p "${CONSOLE_LOG_DIR}" "${ASSETS_BASE_DIR}" "${CHECKPOINT_BASE_DIR}" "${LOG_BASE_DIR}"

TRAIN_ARGS=(
  "${CONFIG_NAME}"
  --pytorch-weight-path "${BASE_PI05_CKPT}"
  --exp-name "${EXP_NAME}"
  --pytorch-training-precision "${CFG_PYTORCH_TRAINING_PRECISION}"
  --num-train-steps "${CFG_NUM_TRAIN_STEPS}"
  --batch-size-per-gpu "${CFG_BATCH_SIZE_PER_GPU}"
  --gradient-accumulation-steps "${CFG_GRADIENT_ACCUMULATION_STEPS}"
  --num-workers "${CFG_NUM_WORKERS}"
  --save-interval "${CFG_SAVE_INTERVAL}"
  --val-log-interval "${CFG_VAL_LOG_INTERVAL}"
  --val-num-batches "${CFG_VAL_NUM_BATCHES}"
  --assets-base-dir "${ASSETS_BASE_DIR}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --log-base-dir "${LOG_BASE_DIR}"
)
if [[ -n "${CFG_NUM_TRAIN_EPOCHS}" ]]; then TRAIN_ARGS+=(--num-train-epochs "${CFG_NUM_TRAIN_EPOCHS}"); fi
python -m accelerate.commands.launch \
  --config_file "${ACCEL_CONFIG}" --num_processes "${TOTAL_GPUS}" --num_machines "${NUM_NODES}" \
  --machine_rank "${NODE_RANK}" --main_process_ip "${MASTER_ADDR}" --main_process_port "${MASTER_PORT}" --same_network \
  "${TRAINER}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"
