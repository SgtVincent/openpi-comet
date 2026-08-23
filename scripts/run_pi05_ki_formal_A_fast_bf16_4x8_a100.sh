#!/usr/bin/env bash
# ============================================================================
# Formal π0.5-KI **Variant A** (FAST action-token CE) — 4-node × 8 A100 BF16.
#
# Contract:
#   * B4/GPU × world 32 × GA2 = global batch 256
#   * stride=4, 4 epochs, anchor offsets 0/1/2/3
#   * BF16, DeepSpeed ZeRO-2, optimizer state kept on GPU (no CPU offload)
#   * action_token_max_len=208 (exhaustive bound: max(train 199, val 190)=199)
#   * W&B project pi05_ki_a100, fresh run, NO resume
#
# This script is Variant-A only. Variant B lives in
# run_pi05_ki_formal_B_query_bf16_4x8_a100.sh with a disjoint output tree and
# disjoint W&B/exp_name identity.
#
# CPU/read-only preflight:
#   OPENPI_LAUNCH_PREFLIGHT_ONLY=1 \
#   OPENPI_EXPECTED_CODE_COMMIT=$(git rev-parse HEAD) \
#   bash scripts/run_pi05_ki_formal_A_fast_bf16_4x8_a100.sh
#
# FORMAL MERLIN ENTRYPOINT (the keepalive wrapper must be outermost):
#   OPENPI_KI_TRAINING_INNER=1 \
#   LAUNCHER=<repo>/scripts/run_pi05_ki_formal_A_fast_bf16_4x8_a100.sh \
#   KEEPALIVE_DISABLE=0 KEEPALIVE_ON_SUCCESS=1 STRICT_GPU_COUNT=0 \
#   bash <repo>/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh
# WARNING: This new no-offload codebase is not a drop-in relaunch base for
# the existing offload-ON A100 run. World-32 B4/GA2 A100-40GB peak memory is
# UNMEASURED; require a bounded memory smoke and explicit launch authorization.
# Do not combine with the separate B8/GA1 change without another bounded smoke.
# ============================================================================

set -euo pipefail

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
info() { printf '[pi05-ki-A-a100][%s] %s\n' "$(_ts)" "$*"; }
die() {
  printf '[pi05-ki-A-a100][%s] FATAL: %s\n' "$(_ts)" "$1" >&2
  shift || true
  for line in "$@"; do
    printf '[pi05-ki-A-a100]                         %s\n' "${line}" >&2
  done
  exit 2
}

# ---- Arm identity (hard-wired; this script is A-only) -----------------------
ARM="A"
ARM_LABEL="variantA_fast_ce"
EXPECTED_CONFIG="pi05_ki_joint_fast_b1k-full_task-ki_on_a100_bf16"
EXPECTED_MODEL="pi05_ki_joint_fast"
WANDB_PROJECT="pi05_ki_a100"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)}"
TRAINER="${TRAINER:-${REPO_ROOT}/scripts/train_accelerate.py}"
ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/configs/accelerate_ds_zero2_a100_bf16.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2_a100_bf16.json}"
CUDA_PREFLIGHT="${CUDA_PREFLIGHT:-${REPO_ROOT}/scripts/cuda_preflight.py}"
OPENPI_PREFLIGHT_PYTHON="${OPENPI_PREFLIGHT_PYTHON:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python}"

[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || die "REPO_ROOT is not a git worktree: ${REPO_ROOT}"
[[ -f "${TRAINER}" ]] || die "trainer not found: ${TRAINER}"
[[ -f "${ACCEL_CONFIG}" ]] || die "Accelerate config not found: ${ACCEL_CONFIG}"
[[ -f "${DEEPSPEED_CONFIG}" ]] || die "DeepSpeed config not found: ${DEEPSPEED_CONFIG}"
[[ -f "${CUDA_PREFLIGHT}" ]] || die "CUDA preflight script not found: ${CUDA_PREFLIGHT}"
[[ "${OPENPI_PREFLIGHT_PYTHON}" == /* && -x "${OPENPI_PREFLIGHT_PYTHON}" ]] \
  || die "OPENPI_PREFLIGHT_PYTHON must be an absolute executable path: ${OPENPI_PREFLIGHT_PYTHON}"

# ---- Immutable code identity ------------------------------------------------
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

# ---- Topology (4 x 8 A100) ------------------------------------------------
NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM:-}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES/ARNOLD_WORKER_NUM must be a positive integer"
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_NODE/ARNOLD_WORKER_GPU must be a positive integer"
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "NODE_RANK/ARNOLD_ID must be a non-negative integer"
(( NUM_NODES == 4 )) || die "formal A100 requires exactly 4 nodes, got ${NUM_NODES}"
(( GPUS_PER_NODE == 8 )) || die "formal A100 requires exactly 8 GPUs per node, got ${GPUS_PER_NODE}"
(( NODE_RANK < NUM_NODES )) || die "NODE_RANK=${NODE_RANK} must be < NUM_NODES=${NUM_NODES}"
TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))

GPU_MODEL="${GPU_MODEL:-${ARNOLD_WORKER_GPU_TYPE:-${ARNOLD_GPU_TYPE:-}}}"
[[ -z "${GPU_MODEL}" || "${GPU_MODEL^^}" == *A100* ]] \
  || die "this BF16 launcher is A100-only, but GPU model is '${GPU_MODEL}'"

MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
_MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-29514}}"
MASTER_PORT="${_MASTER_PORT%%,*}"
[[ -n "${MASTER_ADDR}" && "${MASTER_ADDR}" != *" "* ]] || die "MASTER_ADDR must be one non-empty host/IP"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be numeric"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || die "MASTER_PORT in [1,65535]"

# ---- Precision & batch contract --------------------------------------------
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-bfloat16}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
STREAMING_ANCHOR_STRIDE="${STREAMING_ANCHOR_STRIDE:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"

for pair in \
  "BATCH_SIZE_PER_GPU:${BATCH_SIZE_PER_GPU}" \
  "GRADIENT_ACCUMULATION_STEPS:${GRADIENT_ACCUMULATION_STEPS}" \
  "NUM_TRAIN_EPOCHS:${NUM_TRAIN_EPOCHS}" \
  "STREAMING_ANCHOR_STRIDE:${STREAMING_ANCHOR_STRIDE}" \
  "SAVE_INTERVAL:${SAVE_INTERVAL}" \
  "VAL_LOG_INTERVAL:${VAL_LOG_INTERVAL}" \
  "VAL_NUM_BATCHES:${VAL_NUM_BATCHES}" \
  "NUM_WORKERS:${NUM_WORKERS}"; do
  key="${pair%%:*}"; value="${pair#*:}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${key} must be a positive integer, got '${value}'"
done

[[ "${PYTORCH_TRAINING_PRECISION}" == "bfloat16" ]] || die "A100 formal requires BF16"
[[ "${ACCELERATE_MIXED_PRECISION}" == "bf16" ]] || die "A100 formal requires Accelerate bf16"
(( BATCH_SIZE_PER_GPU == 4 )) || die "A100 formal requires BATCH_SIZE_PER_GPU=4"
(( GRADIENT_ACCUMULATION_STEPS == 2 )) || die "A100 formal requires GRADIENT_ACCUMULATION_STEPS=2"
(( NUM_TRAIN_EPOCHS == 4 )) || die "A100 formal requires NUM_TRAIN_EPOCHS=4"
(( STREAMING_ANCHOR_STRIDE == 4 )) || die "A100 formal requires stride=4"
(( SAVE_INTERVAL == 10000 )) || die "SAVE_INTERVAL=10000"
(( VAL_LOG_INTERVAL == 1000 )) || die "VAL_LOG_INTERVAL=1000"
(( VAL_NUM_BATCHES == 20 )) || die "VAL_NUM_BATCHES=20"

GLOBAL_BATCH_SIZE=$(( BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS ))
(( GLOBAL_BATCH_SIZE == 256 )) || die "global batch must be 256 (B4×W32×GA2), got ${GLOBAL_BATCH_SIZE}"

# ---- W&B (fresh, no resume) ------------------------------------------------
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLED="${WANDB_DISABLED:-0}"
[[ "${WANDB_MODE}" == "online" ]] || die "formal runs require WANDB_MODE=online"
[[ "${WANDB_DISABLED}" == "0" ]] || die "formal runs require WANDB_DISABLED=0"
export WANDB_MODE WANDB_DISABLED
# Prevent accidental resume.
unset WANDB_RESUME WANDB_RUN_ID

# ---- Allocator config ------------------------------------------------------
FORMAL_CUDA_ALLOC_CONF="expandable_segments:True"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${FORMAL_CUDA_ALLOC_CONF}}"
[[ "${PYTORCH_CUDA_ALLOC_CONF}" == "${FORMAL_CUDA_ALLOC_CONF}" ]] \
  || die "A100 formal requires PYTORCH_CUDA_ALLOC_CONF=${FORMAL_CUDA_ALLOC_CONF}"
export PYTORCH_CUDA_ALLOC_CONF

# ---- Pinned data paths -----------------------------------------------------
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
[[ -s "${NORM_STATS_PATH}" ]] || die "norm stats missing: ${NORM_STATS_PATH}"
[[ -s "${PALIGEMMA_TOKENIZER}" ]] || die "PaliGemma tokenizer missing: ${PALIGEMMA_TOKENIZER}"

# ---- DeepSpeed config preflight --------------------------------------------
"${OPENPI_PREFLIGHT_PYTHON}" - "${ACCEL_CONFIG}" "${DEEPSPEED_CONFIG}" <<'PY'
import json, re, sys
from pathlib import Path
acc = Path(sys.argv[1]).read_text()
if re.search(r"^mixed_precision\s*:", acc, re.MULTILINE):
    raise SystemExit("ERROR: Accelerate config must not define top-level mixed_precision with deepspeed_config_file")
if "deepspeed_config_file: configs/deepspeed_zero2_a100_bf16.json" not in acc:
    raise SystemExit("ERROR: Accelerate config does not reference deepspeed_zero2_a100_bf16.json")
ds = json.loads(Path(sys.argv[2]).read_text())
checks = {
    "ZeRO stage 2": ds.get("zero_optimization", {}).get("stage") == 2,
    "optimizer offload disabled": "offload_optimizer" not in ds.get("zero_optimization", {}),
    "bf16 enabled": ds.get("bf16", {}).get("enabled") is True,
    "fp16 disabled": ds.get("fp16", {}).get("enabled") is False,
    "GA auto": ds.get("gradient_accumulation_steps") == "auto",
}
failed = [n for n, ok in checks.items() if not ok]
if failed:
    raise SystemExit("ERROR: A100 BF16 DeepSpeed contract failed: " + ", ".join(failed))
print("A100_BF16_ZERO2_PREFLIGHT_OK stage=2 offload=none bf16=true memory=UNMEASURED")
PY

# ---- TrainConfig preflight -------------------------------------------------
"${OPENPI_PREFLIGHT_PYTHON}" - "${EXPECTED_CONFIG}" "${EXPECTED_MODEL}" <<'PY'
import sys
from openpi.training.train_config import get_config
name, model_name = sys.argv[1:]
cfg = get_config(name)
checks = {
    "registered exact name": cfg.name == name,
    "expected model": cfg.pytorch_model_name == model_name,
    "bf16 precision": cfg.pytorch_training_precision == "bfloat16",
    "Accelerate bf16": cfg.accelerate_mixed_precision == "bf16",
    "model dtype bfloat16": cfg.model.dtype == "bfloat16",
    "KI enabled": cfg.model.knowledge_insulation is True,
    "expert KV truncated": cfg.model.truncate_expert_kv is True,
    "B4": cfg.batch_size_per_gpu == 4,
    "GA2": cfg.gradient_accumulation_steps == 2,
    "gradient checkpointing default on": cfg.gradient_checkpointing is True,
    "4 epochs": cfg.num_train_epochs == 4,
    "stride4": cfg.streaming_anchor_stride == 4,
    "offsets [0,1,2,3]": cfg.epoch_anchor_offsets == [0, 1, 2, 3],
    "save10k": cfg.save_interval == 10_000,
    "val1k": cfg.val_log_interval == 1_000,
    "val20": cfg.val_num_batches == 20,
    "warmup1000": cfg.lr_schedule.warmup_steps == 1_000,
    "peak1e-5": cfg.lr_schedule.peak_lr == 1e-5,
    "wandb enabled": cfg.wandb_enabled is True,
}
if model_name == "pi05_ki_joint_fast":
    checks["FAST cap 208"] = cfg.model.action_token_max_len == 208
else:
    checks["query arm no FAST target"] = not hasattr(cfg.model, "action_token_max_len")
failed = [label for label, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"ERROR: config {name!r} failed: " + "; ".join(failed))
print(f"A100_CONFIG_PREFLIGHT_OK name={name} model={model_name} B4xW32xGA2=256 epochs=4 stride=4 offsets=0,1,2,3")
PY

# ---- FAST tokenizer preflight (A only) -------------------------------------
export OPENPI_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="${REPO_OPENPI_CACHE}"
export OPENPI_BEHAVIOR_DATASET_ROOT="${B1K_DATASET_ROOT}"
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

# ---- Output identity (arm-first, unique Arnold job id before truncation) ----
JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-manual}}"
JOB_ID_SAFE="${JOB_ID//[^A-Za-z0-9_.-]/_}"
PERSISTENT_OUTPUT_BASE="${PERSISTENT_OUTPUT_BASE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_ki_a100_bf16_formal}"
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${PERSISTENT_OUTPUT_BASE}/${ARM_LABEL}/${JOB_ID_SAFE}}"
# Arm discriminator first; job id placed immediately so even a truncated name
# (e.g. Arnold UI truncating at ~40 chars) retains arm + unique short id.
EXP_NAME="${EXP_NAME:-A_fast_ce_a100_bf16_4n8g_${JOB_ID_SAFE}}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/checkpoints}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/logs}"
CONSOLE_LOG_DIR="${PERSISTENT_OUTPUT_ROOT}/console_logs/${EXP_NAME}"
CONSOLE_LOG="${CONSOLE_LOG_DIR}/node${NODE_RANK}.log"

info "============================================================"
info "formal A100 arm=${ARM} objective=${ARM_LABEL} config=${EXPECTED_CONFIG}"
info "code_commit=${ACTUAL_COMMIT} openpi=${OPENPI_IMPORT_PATH}"
info "topology=${NUM_NODES}x${GPUS_PER_NODE} world=${TOTAL_GPUS} rank=${NODE_RANK}"
info "BF16 ZeRO-2 no-optimizer-offload B4/GPU GA2 global_batch=${GLOBAL_BATCH_SIZE} gradient_checkpointing=on"
info "WARNING: world32 B4/GA2 no-offload peak on A100-40GB is UNMEASURED; bounded smoke required before long run"
info "epochs=4 stride=4 offsets=0,1,2,3 wandb_project=${WANDB_PROJECT}"
info "output_root=${PERSISTENT_OUTPUT_ROOT}"
info "============================================================"

if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  info "PREFLIGHT_OK: CPU/read-only checks passed; no GPU, output, or occupier action."
  exit 0
fi

# ---- CUDA/driver preflight on every node -----------------------------------
# This is the fail-fast gate that prevents a 30-minute c10d bootstrap hang on a
# host with an incompatible driver. It must run before accelerate launch.
"${OPENPI_PREFLIGHT_PYTHON}" "${CUDA_PREFLIGHT}" --min-gpus "${GPUS_PER_NODE}" --min-driver-major 525 \
  || die "CUDA/driver preflight failed on node rank ${NODE_RANK}"

# ---- Wrapper contract ------------------------------------------------------
[[ "${OPENPI_KI_TRAINING_INNER:-0}" == "1" ]] \
  || die "formal Merlin entrypoint must be the outer keepalive wrapper" \
      "Set OPENPI_KI_TRAINING_INNER=1 and LAUNCHER=${SCRIPT_PATH} on the wrapper."
[[ "${KEEPALIVE_DISABLE:-}" == "0" ]] || die "outer wrapper requires KEEPALIVE_DISABLE=0"
[[ "${KEEPALIVE_ON_SUCCESS:-}" == "1" ]] || die "outer wrapper requires KEEPALIVE_ON_SUCCESS=1"
[[ "${STRICT_GPU_COUNT:-}" == "0" ]] || die "outer wrapper requires STRICT_GPU_COUNT=0"

[[ "${BASE_PI05_CKPT}" == "${FORMAL_BASE_PI05_CKPT}" ]] || die "base checkpoint pinned to ${FORMAL_BASE_PI05_CKPT}"
[[ "${B1K_DATASET_ROOT}" == "${FORMAL_B1K_DATASET_ROOT}" ]] || die "dataset root pinned"
[[ "${B1K_ASSETS_DIR}" == "${FORMAL_B1K_ASSETS_DIR}" ]] || die "assets dir pinned"
[[ "${NORM_STATS_PATH}" == "${FORMAL_NORM_STATS_PATH}" ]] || die "norm stats pinned"

assert_no_occupiers() {
  local matches
  matches="$(ps -eo pid=,args= 2>/dev/null \
    | awk '/__GPU_OCCUPY__torch_mm_512/ && /gpu_occupy_(torch_mm[.]py|stub[.]sh)/ {print}' || true)"
  [[ -z "${matches}" ]] || die "GPU keepalive occupiers still running" "${matches}"
}
assert_no_occupiers

export NUM_NODES GPUS_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
export CONFIG_NAME="${EXPECTED_CONFIG}"
export PYTORCH_TRAINING_PRECISION ACCELERATE_MIXED_PRECISION
export BATCH_SIZE_PER_GPU GRADIENT_ACCUMULATION_STEPS NUM_WORKERS
export SAVE_INTERVAL VAL_LOG_INTERVAL VAL_NUM_BATCHES
export BASE_PI05_CKPT B1K_DATASET_ROOT B1K_ASSETS_DIR NORM_STATS_PATH REPO_OPENPI_CACHE
export PERSISTENT_OUTPUT_ROOT EXP_NAME ASSETS_BASE_DIR CHECKPOINT_BASE_DIR LOG_BASE_DIR
export ACCEL_CONFIG DEEPSPEED_CONFIG TRAINER REPO_ROOT
export WANDB_PROJECT

CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3}"
CONDA_ENV="${CONDA_ENV:-openpi-comet-nas}"
CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
[[ -f "${CONDA_SH}" ]] || die "conda init not found: ${CONDA_SH}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
unset PYTHONHOME
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${OPENPI_EXPECTED_CODE_COMMIT}" ]] || die "commit changed before inner launch"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "tree became dirty before inner launch"
python - <<'PY'
import os, openpi
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
  --pytorch-training-precision bfloat16 \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --batch-size-per-gpu 4 \
  --gradient-accumulation-steps 2 \
  --gradient-checkpointing \
  --num-workers "${NUM_WORKERS}" \
  --save-interval 10000 \
  --val-log-interval 1000 \
  --val-num-batches 20 \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  --log-base-dir "${LOG_BASE_DIR}" \
  2>&1 | tee -a "${CONSOLE_LOG}"
