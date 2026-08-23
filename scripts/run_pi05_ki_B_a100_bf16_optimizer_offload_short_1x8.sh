#!/usr/bin/env bash
# ============================================================================
# Bounded 1-node x 8 A100 optimizer-offload A/B experiment for production
# pi0.5-KI Variant B (query-MSE).
#
# The selected arms share the exact production model/data TrainConfig and use:
#   * production B4/GPU x GA2 on world 8 = global batch 64
#   * BF16, DeepSpeed ZeRO-2, seed 42, exactly 100 optimizer steps
#   * validation and model-checkpoint saves disabled for bounded measurement
#   * fresh, disjoint output and W&B run identities
#
# Validation is disabled by the experiment-only config's empty val_data. The
# shared trainer remains bit-identical to 7019917. This experiment branch must
# never be used as a V100 or formal-production relaunch base.
#
# Required arm selector:
#   OPTIMIZER_OFFLOAD_MODE=on   -> CPU optimizer offload (baseline JSON)
#   OPTIMIZER_OFFLOAD_MODE=off  -> identical JSON minus offload_optimizer
#
# CPU/read-only preflight example:
#   OPTIMIZER_OFFLOAD_MODE=on MAX_TRAIN_STEPS=100 \
#   ARNOLD_JOB_ID=preflight-offload-on NUM_NODES=1 GPUS_PER_NODE=8 NODE_RANK=0 \
#   OPENPI_LAUNCH_PREFLIGHT_ONLY=1 \
#   OPENPI_EXPECTED_CODE_COMMIT=$(git rev-parse HEAD) \
#   bash scripts/run_pi05_ki_B_a100_bf16_optimizer_offload_short_1x8.sh
#
# MERLIN ENTRYPOINT (the keepalive wrapper remains outermost):
#   OPTIMIZER_OFFLOAD_MODE=<on|off> MAX_TRAIN_STEPS=100 \
#   OPENPI_EXPECTED_CODE_COMMIT=<full-commit-sha> OPENPI_KI_TRAINING_INNER=1 \
#   LAUNCHER=<repo>/scripts/run_pi05_ki_B_a100_bf16_optimizer_offload_short_1x8.sh \
#   KEEPALIVE_DISABLE=0 KEEPALIVE_ON_SUCCESS=0 STRICT_GPU_COUNT=0 \
#   bash <repo>/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh
# ============================================================================

set -euo pipefail

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
info() { printf '[pi05-ki-B-offload-short][%s] %s\n' "$(_ts)" "$*"; }
die() {
  printf '[pi05-ki-B-offload-short][%s] FATAL: %s\n' "$(_ts)" "$1" >&2
  shift || true
  for line in "$@"; do
    printf '[pi05-ki-B-offload-short]                               %s\n' "${line}" >&2
  done
  exit 2
}

# ---- Single-variable arm identity ------------------------------------------
OPTIMIZER_OFFLOAD_MODE="${OPTIMIZER_OFFLOAD_MODE:-}"
case "${OPTIMIZER_OFFLOAD_MODE}" in
  on)
    RUN_LABEL="A100-B4GA2-offload-on-1x8-100step"
    DS_CONFIG_BASENAME="deepspeed_zero2_a100_bf16_offload_on_short.json"
    ACCEL_CONFIG_BASENAME="accelerate_ds_zero2_a100_bf16_offload_on_short.yaml"
    ;;
  off)
    RUN_LABEL="A100-B4GA2-offload-off-1x8-100step"
    DS_CONFIG_BASENAME="deepspeed_zero2_a100_bf16_offload_off_short.json"
    ACCEL_CONFIG_BASENAME="accelerate_ds_zero2_a100_bf16_offload_off_short.yaml"
    ;;
  *)
    die "OPTIMIZER_OFFLOAD_MODE must be exactly 'on' or 'off'; got '${OPTIMIZER_OFFLOAD_MODE}'"
    ;;
esac

EXPECTED_CONFIG="pi05_ki_joint_query_b1k-full_task-ki_on_a100_bf16_offload_short"
PRODUCTION_CONFIG="pi05_ki_joint_query_b1k-full_task-ki_on_a100_bf16"
EXPECTED_MODEL="pi05_ki_joint_query"
WANDB_PROJECT="pi05_ki_a100_offload_short"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)}"
TRAINER="${REPO_ROOT}/scripts/train_accelerate.py"
ACCEL_CONFIG="${REPO_ROOT}/configs/${ACCEL_CONFIG_BASENAME}"
DEEPSPEED_CONFIG="${REPO_ROOT}/configs/${DS_CONFIG_BASENAME}"
BASELINE_ACCEL_CONFIG="${REPO_ROOT}/configs/accelerate_ds_zero2_a100_bf16.yaml"
BASELINE_DEEPSPEED_CONFIG="${REPO_ROOT}/configs/deepspeed_zero2_a100_bf16.json"
CUDA_PREFLIGHT="${REPO_ROOT}/scripts/cuda_preflight.py"
OPENPI_PREFLIGHT_PYTHON="${OPENPI_PREFLIGHT_PYTHON:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python}"

[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || die "REPO_ROOT is not a git worktree: ${REPO_ROOT}"
for required in \
  "${TRAINER}" \
  "${ACCEL_CONFIG}" \
  "${DEEPSPEED_CONFIG}" \
  "${BASELINE_ACCEL_CONFIG}" \
  "${BASELINE_DEEPSPEED_CONFIG}" \
  "${CUDA_PREFLIGHT}"; do
  [[ -f "${required}" ]] || die "required file not found: ${required}"
done
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
  || die "bounded launch requires a clean worktree at ${EXPECTED_COMMIT}" "${DIRTY_STATUS}"
export OPENPI_EXPECTED_CODE_COMMIT="${EXPECTED_COMMIT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
OPENPI_IMPORT_PATH="$("${OPENPI_PREFLIGHT_PYTHON}" - <<'PY'
import openpi
print(openpi.__file__)
PY
)"
[[ "${OPENPI_IMPORT_PATH}" == "${REPO_ROOT}/src/openpi/"* ]] \
  || die "openpi import does not resolve inside the pinned tree: ${OPENPI_IMPORT_PATH}"

# ---- Fixed 1 x 8 A100 topology ---------------------------------------------
NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM:-}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES/ARNOLD_WORKER_NUM must be a positive integer"
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_NODE/ARNOLD_WORKER_GPU must be a positive integer"
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "NODE_RANK/ARNOLD_ID must be a non-negative integer"
(( NUM_NODES == 1 )) || die "offload short experiment requires exactly 1 node, got ${NUM_NODES}"
(( GPUS_PER_NODE == 8 )) || die "offload short experiment requires exactly 8 GPUs, got ${GPUS_PER_NODE}"
(( NODE_RANK == 0 )) || die "single-node experiment requires NODE_RANK=0, got ${NODE_RANK}"
TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))

GPU_MODEL="${GPU_MODEL:-${ARNOLD_WORKER_GPU_TYPE:-${ARNOLD_GPU_TYPE:-}}}"
[[ -z "${GPU_MODEL}" || "${GPU_MODEL^^}" == *A100* ]] \
  || die "this BF16 experiment is A100-only, but GPU model is '${GPU_MODEL}'"

MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"
_MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-29514}}"
MASTER_PORT="${_MASTER_PORT%%,*}"
[[ -n "${MASTER_ADDR}" && "${MASTER_ADDR}" != *" "* ]] || die "MASTER_ADDR must be one non-empty host/IP"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be numeric"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || die "MASTER_PORT must be in [1,65535]"

# ---- Matched short-run contract --------------------------------------------
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-bfloat16}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
STREAMING_ANCHOR_STRIDE="${STREAMING_ANCHOR_STRIDE:-4}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OPENPI_DISABLE_CHECKPOINT="${OPENPI_DISABLE_CHECKPOINT:-1}"

for pair in \
  "BATCH_SIZE_PER_GPU:${BATCH_SIZE_PER_GPU}" \
  "GRADIENT_ACCUMULATION_STEPS:${GRADIENT_ACCUMULATION_STEPS}" \
  "MAX_TRAIN_STEPS:${MAX_TRAIN_STEPS}" \
  "NUM_TRAIN_EPOCHS:${NUM_TRAIN_EPOCHS}" \
  "STREAMING_ANCHOR_STRIDE:${STREAMING_ANCHOR_STRIDE}" \
  "SEED:${SEED}" \
  "LOG_INTERVAL:${LOG_INTERVAL}" \
  "SAVE_INTERVAL:${SAVE_INTERVAL}" \
  "VAL_LOG_INTERVAL:${VAL_LOG_INTERVAL}" \
  "VAL_NUM_BATCHES:${VAL_NUM_BATCHES}" \
  "NUM_WORKERS:${NUM_WORKERS}"; do
  key="${pair%%:*}"; value="${pair#*:}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${key} must be a positive integer, got '${value}'"
done

[[ "${PYTORCH_TRAINING_PRECISION}" == "bfloat16" ]] || die "offload short experiment requires BF16"
[[ "${ACCELERATE_MIXED_PRECISION}" == "bf16" ]] || die "offload short experiment requires Accelerate bf16"
(( BATCH_SIZE_PER_GPU == 4 )) || die "offload short experiment requires BATCH_SIZE_PER_GPU=4"
(( GRADIENT_ACCUMULATION_STEPS == 2 )) || die "offload short experiment requires GRADIENT_ACCUMULATION_STEPS=2"
(( MAX_TRAIN_STEPS == 100 )) || die "this submitted experiment requires MAX_TRAIN_STEPS=100"
(( NUM_TRAIN_EPOCHS == 4 )) || die "offload short experiment preserves NUM_TRAIN_EPOCHS=4"
(( STREAMING_ANCHOR_STRIDE == 4 )) || die "offload short experiment preserves stride=4"
(( SEED == 42 )) || die "offload short experiment requires SEED=42"
(( LOG_INTERVAL == 1 )) || die "offload short experiment requires LOG_INTERVAL=1"
(( SAVE_INTERVAL == 10000 )) || die "offload short experiment preserves SAVE_INTERVAL=10000"
(( VAL_LOG_INTERVAL == 1000 )) || die "offload short experiment preserves VAL_LOG_INTERVAL=1000"
(( VAL_NUM_BATCHES == 20 )) || die "offload short experiment preserves VAL_NUM_BATCHES=20"
[[ "${OPENPI_DISABLE_CHECKPOINT}" == "1" ]] || die "OPENPI_DISABLE_CHECKPOINT must be exactly 1"
export OPENPI_DISABLE_CHECKPOINT

GLOBAL_BATCH_SIZE=$(( BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS ))
(( GLOBAL_BATCH_SIZE == 64 )) || die "global batch must be 64 (B4xW8xGA2), got ${GLOBAL_BATCH_SIZE}"
(( MAX_TRAIN_STEPS < SAVE_INTERVAL )) || die "bounded run must finish before SAVE_INTERVAL"
(( MAX_TRAIN_STEPS < VAL_LOG_INTERVAL )) || die "bounded run must finish before VAL_LOG_INTERVAL"

# ---- Fresh, disjoint W&B/output identity -----------------------------------
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLED="${WANDB_DISABLED:-0}"
[[ "${WANDB_MODE}" == "online" ]] || die "measurement requires WANDB_MODE=online"
[[ "${WANDB_DISABLED}" == "0" ]] || die "measurement requires WANDB_DISABLED=0"
export WANDB_MODE WANDB_DISABLED
unset WANDB_RESUME WANDB_RUN_ID

JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-}}"
[[ -n "${JOB_ID}" ]] || die "ARNOLD_JOB_ID or ARNOLD_TASK_ID is required for fresh output identity"
JOB_ID_SAFE="${JOB_ID//[^A-Za-z0-9_.-]/_}"
PERSISTENT_OUTPUT_BASE="${PERSISTENT_OUTPUT_BASE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_ki_a100_optimizer_offload_short}"
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_BASE}/${RUN_LABEL}/${JOB_ID_SAFE}"
EXP_NAME="${RUN_LABEL}_${JOB_ID_SAFE}"
ASSETS_BASE_DIR="${PERSISTENT_OUTPUT_ROOT}/assets"
CHECKPOINT_BASE_DIR="${PERSISTENT_OUTPUT_ROOT}/checkpoints"
LOG_BASE_DIR="${PERSISTENT_OUTPUT_ROOT}/logs"
CONSOLE_LOG_DIR="${PERSISTENT_OUTPUT_ROOT}/console_logs/${EXP_NAME}"
CONSOLE_LOG="${CONSOLE_LOG_DIR}/node0.log"
[[ ! -e "${PERSISTENT_OUTPUT_ROOT}" ]] \
  || die "fresh-run output already exists; refusing to reuse: ${PERSISTENT_OUTPUT_ROOT}"

# ---- Allocator and pinned input contract -----------------------------------
FORMAL_CUDA_ALLOC_CONF="expandable_segments:True"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${FORMAL_CUDA_ALLOC_CONF}}"
[[ "${PYTORCH_CUDA_ALLOC_CONF}" == "${FORMAL_CUDA_ALLOC_CONF}" ]] \
  || die "experiment requires PYTORCH_CUDA_ALLOC_CONF=${FORMAL_CUDA_ALLOC_CONF}"
export PYTORCH_CUDA_ALLOC_CONF

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
[[ "${BASE_PI05_CKPT}" == "${FORMAL_BASE_PI05_CKPT}" ]] || die "base checkpoint must remain pinned"
[[ "${B1K_DATASET_ROOT}" == "${FORMAL_B1K_DATASET_ROOT}" ]] || die "dataset root must remain pinned"
[[ "${B1K_ASSETS_DIR}" == "${FORMAL_B1K_ASSETS_DIR}" ]] || die "assets dir must remain pinned"
[[ "${NORM_STATS_PATH}" == "${FORMAL_NORM_STATS_PATH}" ]] || die "norm stats must remain pinned"
[[ -s "${BASE_PI05_CKPT}/model.safetensors" ]] || die "base weights missing: ${BASE_PI05_CKPT}/model.safetensors"
[[ -d "${B1K_DATASET_ROOT}" ]] || die "B1K dataset root missing: ${B1K_DATASET_ROOT}"
[[ -s "${NORM_STATS_PATH}" ]] || die "norm stats missing: ${NORM_STATS_PATH}"
[[ -s "${PALIGEMMA_TOKENIZER}" ]] || die "PaliGemma tokenizer missing: ${PALIGEMMA_TOKENIZER}"

# ---- Effect-based DeepSpeed/Accelerate contract ----------------------------
"${OPENPI_PREFLIGHT_PYTHON}" - \
  "${BASELINE_ACCEL_CONFIG}" "${ACCEL_CONFIG}" \
  "${BASELINE_DEEPSPEED_CONFIG}" "${DEEPSPEED_CONFIG}" \
  "${OPTIMIZER_OFFLOAD_MODE}" "${DS_CONFIG_BASENAME}" <<'PY'
import copy
import json
import sys
from pathlib import Path

baseline_accel_path, selected_accel_path, baseline_ds_path, selected_ds_path, mode, selected_ds_name = sys.argv[1:]
baseline_accel = Path(baseline_accel_path).read_text()
selected_accel = Path(selected_accel_path).read_text()
expected_accel = baseline_accel.replace(
    "configs/deepspeed_zero2_a100_bf16.json",
    f"configs/{selected_ds_name}",
)
if selected_accel != expected_accel:
    raise SystemExit("ERROR: selected Accelerate config differs from baseline beyond its DeepSpeed config reference")

baseline_ds = json.loads(Path(baseline_ds_path).read_text())
selected_ds = json.loads(Path(selected_ds_path).read_text())
expected_ds = copy.deepcopy(baseline_ds)
if mode == "off":
    del expected_ds["zero_optimization"]["offload_optimizer"]
if selected_ds != expected_ds:
    raise SystemExit("ERROR: selected DeepSpeed config is not the declared single-variable baseline transform")

zero = selected_ds.get("zero_optimization", {})
offload = zero.get("offload_optimizer")
checks = {
    "ZeRO stage 2": zero.get("stage") == 2,
    "optimizer offload mode": (
        offload == {"device": "cpu", "pin_memory": True} if mode == "on" else "offload_optimizer" not in zero
    ),
    "bf16 enabled": selected_ds.get("bf16", {}).get("enabled") is True,
    "fp16 disabled": selected_ds.get("fp16", {}).get("enabled") is False,
    "micro batch auto": selected_ds.get("train_micro_batch_size_per_gpu") == "auto",
    "GA auto": selected_ds.get("gradient_accumulation_steps") == "auto",
}
failed = [label for label, ok in checks.items() if not ok]
if failed:
    raise SystemExit("ERROR: DeepSpeed short-experiment contract failed: " + ", ".join(failed))
device = offload["device"] if isinstance(offload, dict) else "none"
print(
    f"A100_OFFLOAD_DS_PREFLIGHT_OK mode={mode} offload_state={mode} "
    f"offload_optimizer_device={device} stage=2 bf16=true single_delta=true"
)
PY

# ---- Production-derived TrainConfig contract -------------------------------
"${OPENPI_PREFLIGHT_PYTHON}" - \
  "${PRODUCTION_CONFIG}" "${EXPECTED_CONFIG}" "${EXPECTED_MODEL}" \
  "${BATCH_SIZE_PER_GPU}" "${GRADIENT_ACCUMULATION_STEPS}" \
  "${MAX_TRAIN_STEPS}" "${SEED}" "${LOG_INTERVAL}" <<'PY'
import dataclasses
import sys
from openpi.training.train_config import get_config

prod_name, name, model_name, micro, ga, max_steps, seed, log_interval = sys.argv[1:]
micro, ga, max_steps, seed, log_interval = map(int, (micro, ga, max_steps, seed, log_interval))
prod = get_config(prod_name)
cfg = get_config(name)
checks = {
    "registered exact names": prod.name == prod_name and cfg.name == name,
    "production Variant B": cfg.pytorch_model_name == model_name,
    "bf16 precision": cfg.pytorch_training_precision == "bfloat16",
    "Accelerate bf16": cfg.accelerate_mixed_precision == "bf16",
    "model dtype bfloat16": cfg.model.dtype == "bfloat16",
    "KI enabled": cfg.model.knowledge_insulation is True,
    "expert KV truncated": cfg.model.truncate_expert_kv is True,
    "B4": cfg.batch_size_per_gpu == micro == 4,
    "GA2": cfg.gradient_accumulation_steps == ga == 2,
    "100 optimizer steps": cfg.num_train_steps == max_steps == 100,
    "4-epoch ceiling": cfg.num_train_epochs == 4,
    "per-step timing logs": cfg.log_interval == log_interval == 1,
    "validation disabled by empty val_data": cfg.val_data == [],
    "stride4": cfg.streaming_anchor_stride == 4,
    "offsets [0,1,2,3]": cfg.epoch_anchor_offsets == [0, 1, 2, 3],
    "seed42": cfg.seed == seed == 42,
    "save10k": cfg.save_interval == 10_000,
    "val1k retained": cfg.val_log_interval == 1_000,
    "warmup1000": cfg.lr_schedule.warmup_steps == 1_000,
    "peak1e-5": cfg.lr_schedule.peak_lr == 1e-5,
    "wandb enabled": cfg.wandb_enabled is True,
    "query arm no FAST target": not hasattr(cfg.model, "action_token_max_len"),
}
failed = [label for label, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"ERROR: short config {name!r} failed: " + "; ".join(failed))

allowed = {"name", "exp_name", "num_train_steps", "log_interval", "val_data"}
changed = {
    field.name
    for field in dataclasses.fields(prod)
    if getattr(prod, field.name) != getattr(cfg, field.name)
}
if changed != allowed:
    raise SystemExit(f"ERROR: short config drift outside bounded controls: {sorted(changed)}")
if cfg.batch_size_per_gpu * 8 * cfg.gradient_accumulation_steps != 64:
    raise SystemExit("ERROR: runtime global batch is not 64")
print(
    "A100_OFFLOAD_RUNTIME_CONTRACT_OK "
    f"name={name} micro_bs=4 grad_accum=2 world_size=8 global_batch=64 "
    "max_steps=100 log_interval=1 validation=disabled(empty-val-data) seed=42"
)
PY

# Variant B does not need the FAST tokenizer.
export OPENPI_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="${REPO_OPENPI_CACHE}"
export OPENPI_BEHAVIOR_DATASET_ROOT="${B1K_DATASET_ROOT}"
unset OPENPI_FAST_TOKENIZER_PATH

info "============================================================"
info "run_label=${RUN_LABEL} mode=${OPTIMIZER_OFFLOAD_MODE} config=${EXPECTED_CONFIG}"
info "code_commit=${ACTUAL_COMMIT} openpi=${OPENPI_IMPORT_PATH}"
info "topology=1x8 world_size=8 rank=0 BF16 ZeRO-2 micro_bs=4 grad_accum=2 global_batch=${GLOBAL_BATCH_SIZE}"
info "offload_state=${OPTIMIZER_OFFLOAD_MODE} max_steps=${MAX_TRAIN_STEPS} log_interval=${LOG_INTERVAL} validation=disabled(empty-val-data) checkpoint=disabled seed=${SEED}"
info "timing_analysis=drop_step_0_warmup"
info "wandb_project=${WANDB_PROJECT} exp_name=${EXP_NAME}"
info "deepspeed_config=${DEEPSPEED_CONFIG}"
info "output_root=${PERSISTENT_OUTPUT_ROOT}"
info "============================================================"

if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  info "PREFLIGHT_OK: CPU/read-only checks passed; no GPU, output, or occupier action."
  exit 0
fi

# ---- CUDA/driver and outer-wrapper contract --------------------------------
"${OPENPI_PREFLIGHT_PYTHON}" "${CUDA_PREFLIGHT}" --min-gpus 8 --min-driver-major 525 \
  || die "CUDA/driver preflight failed"
[[ "${OPENPI_KI_TRAINING_INNER:-0}" == "1" ]] \
  || die "Merlin entrypoint must be the outer keepalive wrapper" \
      "Set OPENPI_KI_TRAINING_INNER=1 and LAUNCHER=${SCRIPT_PATH} on the wrapper."
[[ "${KEEPALIVE_DISABLE:-}" == "0" ]] || die "outer wrapper requires KEEPALIVE_DISABLE=0"
[[ "${KEEPALIVE_ON_SUCCESS:-}" == "0" ]] \
  || die "bounded run requires KEEPALIVE_ON_SUCCESS=0 so successful jobs release GPUs"
[[ "${STRICT_GPU_COUNT:-}" == "0" ]] || die "outer wrapper requires STRICT_GPU_COUNT=0"

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
export BATCH_SIZE_PER_GPU GRADIENT_ACCUMULATION_STEPS MAX_TRAIN_STEPS NUM_WORKERS SEED
export LOG_INTERVAL SAVE_INTERVAL VAL_LOG_INTERVAL VAL_NUM_BATCHES
export BASE_PI05_CKPT B1K_DATASET_ROOT B1K_ASSETS_DIR NORM_STATS_PATH REPO_OPENPI_CACHE
export PERSISTENT_OUTPUT_ROOT EXP_NAME ASSETS_BASE_DIR CHECKPOINT_BASE_DIR LOG_BASE_DIR
export ACCEL_CONFIG DEEPSPEED_CONFIG TRAINER REPO_ROOT WANDB_PROJECT

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
[[ ! -e "${PERSISTENT_OUTPUT_ROOT}" ]] || die "output appeared before inner launch: ${PERSISTENT_OUTPUT_ROOT}"
python - <<'PY'
import os
import openpi
print(f"CODE_PROVENANCE commit={os.environ['OPENPI_EXPECTED_CODE_COMMIT']} openpi.__file__={openpi.__file__}")
PY
mkdir -p "${CONSOLE_LOG_DIR}" "${ASSETS_BASE_DIR}" "${LOG_BASE_DIR}"

python -m accelerate.commands.launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes 8 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --same_network \
  "${TRAINER}" \
  "${CONFIG_NAME}" \
  --project-name "${WANDB_PROJECT}" \
  --pytorch-weight-path "${BASE_PI05_CKPT}" \
  --exp-name "${EXP_NAME}" \
  --pytorch-training-precision bfloat16 \
  --seed 42 \
  --num-train-steps "${MAX_TRAIN_STEPS}" \
  --num-train-epochs 4 \
  --batch-size-per-gpu 4 \
  --gradient-accumulation-steps 2 \
  --num-workers "${NUM_WORKERS}" \
  --log-interval 1 \
  --save-interval 10000 \
  --val-log-interval 1000 \
  --val-num-batches 20 \
  --no-resume \
  --no-overwrite \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  --log-base-dir "${LOG_BASE_DIR}" \
  2>&1 | tee -a "${CONSOLE_LOG}"
