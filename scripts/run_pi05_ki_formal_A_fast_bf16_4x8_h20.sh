#!/usr/bin/env bash
# ============================================================================
# Formal π0.5-KI **Variant A** (FAST action-token CE) — 4-node × 8 NVIDIA_H20 BF16.
#
# Two-arm controlled experiment. Both arms are identical except the warm-start
# package; select with OPENPI_H20_ARM:
#
#   ARM=A  "comet base"  config pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16
#          weights checkpoints/openpi_comet/pi05-b1kpt50-cs32 (fp32)
#          assets  <that ckpt>/assets      norm_stats sha256 d66ed168…  6368 B
#   ARM=B  "pi05 base"   config pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16
#          weights checkpoints/pi05_base_pytorch (bf16)
#          assets  <that ckpt>/assets      norm_stats sha256 4dde119e…  6361 B
#
# Each arm pairs weights with the normalization they were fit under. That is
# deliberate, and it means the arms differ in TWO coupled variables (weights AND
# action normalization): the comparison ranks base *packages*, it does not
# isolate the weights.
#
# Contract (identical in both arms):
#   * B8/GPU × world 32 × GA1 = global batch 256   (matches the cont2 contract
#     already proven on this exact 4×8 H20 / soil-hl / group 947 hardware)
#   * stride 12, three passes, anchor offsets 0/4/8, fixed 104,912-step budget
#   * BF16, DeepSpeed ZeRO-2, optimizer NOT offloaded, gradient checkpointing ON
#   * action_token_max_len=256 — shared by both arms so capacity can never be a
#     third confound. SAMPLED (not exhaustive) bound under d66ed168; see the
#     _H20_FAST_ACTION_TOKEN_MAX_LEN comment in pi05_ki_joint_query_config.py.
#   * W&B project pi05_ki, fresh run, NO resume
#
# H20 mounts behavior-data-hl / navigation-hl / robot-mllm-data-hl and does NOT
# mount saiwenresearch, so every pinned path here is HL-side.
#
# MODES
#   OPENPI_H20_MODE=smoke   bounded 8-optimizer-step run on the *_smoke config,
#                           same B8/GA1 so the measured memory peak is real.
#   OPENPI_H20_MODE=formal  the 104,912-step budget (default).
#
# CPU/read-only preflight (no GPU, no output, no occupier action):
#   OPENPI_LAUNCH_PREFLIGHT_ONLY=1 OPENPI_H20_ARM=A \
#   OPENPI_EXPECTED_CODE_COMMIT=<40-char sha> \
#   bash scripts/run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh
#
# FORMAL MERLIN ENTRYPOINT (the keepalive wrapper must be outermost):
#   OPENPI_KI_TRAINING_INNER=1 OPENPI_H20_ARM=A \
#   LAUNCHER=<repo>/scripts/run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh \
#   KEEPALIVE_DISABLE=0 KEEPALIVE_ON_SUCCESS=1 STRICT_GPU_COUNT=0 \
#   bash <repo>/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh
# ============================================================================

set -euo pipefail

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
info() { printf '[pi05-ki-A-h20][%s] %s\n' "$(_ts)" "$*"; }
die() {
  printf '[pi05-ki-A-h20][%s] FATAL: %s\n' "$(_ts)" "$1" >&2
  shift || true
  for line in "$@"; do
    printf '[pi05-ki-A-h20]                        %s\n' "${line}" >&2
  done
  exit 2
}

# ---- Arm identity ----------------------------------------------------------
ARM="${OPENPI_H20_ARM:-}"
[[ "${ARM}" == "A" || "${ARM}" == "B" ]] \
  || die "OPENPI_H20_ARM must be exactly A (comet base) or B (pi05 base); got '${ARM}'"
MODE="${OPENPI_H20_MODE:-formal}"
[[ "${MODE}" == "smoke" || "${MODE}" == "formal" ]] \
  || die "OPENPI_H20_MODE must be 'smoke' or 'formal'; got '${MODE}'"

EXPECTED_MODEL="pi05_ki_joint_fast"
WANDB_PROJECT="pi05_ki"

# Per-arm warm-start package. The norm_stats digest is asserted below: it is the
# cheapest hard proof that the arm is running the normalization that belongs to
# its own weights, which is the entire premise of the comparison.
if [[ "${ARM}" == "A" ]]; then
  ARM_LABEL="variantA_fast_ce_cometbase"
  BASE_CONFIG="pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16"
  FORMAL_BASE_CKPT="/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32"
  EXPECTED_NORM_STATS_SHA256="d66ed16830a98f90dde8a315058b4a0df59f5e05734c1686d8b3f66787d0a929"
  EXPECTED_WEIGHTS_BYTES=14467165872
else
  ARM_LABEL="variantA_fast_ce_pi05base"
  BASE_CONFIG="pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16"
  FORMAL_BASE_CKPT="/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch"
  EXPECTED_NORM_STATS_SHA256="4dde119e69123ed865072c71a714095ae746c6d294fefba910a842757a7083ce"
  EXPECTED_WEIGHTS_BYTES=7233650408
fi
if [[ "${MODE}" == "smoke" ]]; then
  EXPECTED_CONFIG="${BASE_CONFIG}_smoke"
  ARM_LABEL="${ARM_LABEL}_smoke"
else
  EXPECTED_CONFIG="${BASE_CONFIG}"
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)}"
TRAINER="${TRAINER:-${REPO_ROOT}/scripts/train_accelerate.py}"
ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/configs/accelerate_ds_zero2_h20_bf16.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2_h20_bf16.json}"
CUDA_PREFLIGHT="${CUDA_PREFLIGHT:-${REPO_ROOT}/scripts/cuda_preflight.py}"
CUDA_PREFLIGHT_ALL="${CUDA_PREFLIGHT_ALL:-${REPO_ROOT}/scripts/cuda_preflight_all_devices.py}"
# NOTE: the navigation-hl conda env is NOT usable — its transformers install
# cannot resolve `GemmaForCausalLM`, so `openpi.models_pytorch.gemma_pytorch`
# fails to import there and no model can be built. behavior-data-hl is the
# verified working env. Do not "simplify" this back to navigation-hl.
OPENPI_PREFLIGHT_PYTHON="${OPENPI_PREFLIGHT_PYTHON:-/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python}"

[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || die "REPO_ROOT is not a git worktree: ${REPO_ROOT}"
[[ -f "${TRAINER}" ]] || die "trainer not found: ${TRAINER}"
[[ -f "${ACCEL_CONFIG}" ]] || die "Accelerate config not found: ${ACCEL_CONFIG}"
[[ -f "${DEEPSPEED_CONFIG}" ]] || die "DeepSpeed config not found: ${DEEPSPEED_CONFIG}"
[[ -f "${CUDA_PREFLIGHT}" ]] || die "CUDA preflight script not found: ${CUDA_PREFLIGHT}"
[[ -f "${CUDA_PREFLIGHT_ALL}" ]] || die "per-device CUDA preflight not found: ${CUDA_PREFLIGHT_ALL}"
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

# ---- Topology (4 x 8 H20) ---------------------------------------------------
NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM:-}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES/ARNOLD_WORKER_NUM must be a positive integer"
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_NODE/ARNOLD_WORKER_GPU must be a positive integer"
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "NODE_RANK/ARNOLD_ID must be a non-negative integer"
(( NUM_NODES == 4 )) || die "formal H20 requires exactly 4 nodes, got ${NUM_NODES}"
(( GPUS_PER_NODE == 8 )) || die "formal H20 requires exactly 8 GPUs per node, got ${GPUS_PER_NODE}"
(( NODE_RANK < NUM_NODES )) || die "NODE_RANK=${NODE_RANK} must be < NUM_NODES=${NUM_NODES}"
TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))

# H20 gate (the A100 launcher's equivalent line gates on *A100*; this one must
# not silently accept a different accelerator).
GPU_MODEL="${GPU_MODEL:-${ARNOLD_WORKER_GPU_TYPE:-${ARNOLD_GPU_TYPE:-}}}"
[[ -z "${GPU_MODEL}" || "${GPU_MODEL^^}" == *H20* ]] \
  || die "this BF16 launcher is H20-only, but GPU model is '${GPU_MODEL}'"

MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
_MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-29517}}"
MASTER_PORT="${_MASTER_PORT%%,*}"
[[ -n "${MASTER_ADDR}" && "${MASTER_ADDR}" != *" "* ]] || die "MASTER_ADDR must be one non-empty host/IP"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be numeric"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || die "MASTER_PORT in [1,65535]"

# ---- Precision & batch contract --------------------------------------------
PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-bfloat16}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
if [[ "${MODE}" == "smoke" ]]; then
  SAVE_INTERVAL="${SAVE_INTERVAL:-8}"
  VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-4}"
  VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-1}"
else
  SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
  VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-1000}"
  VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
fi

for pair in \
  "BATCH_SIZE_PER_GPU:${BATCH_SIZE_PER_GPU}" \
  "GRADIENT_ACCUMULATION_STEPS:${GRADIENT_ACCUMULATION_STEPS}" \
  "SAVE_INTERVAL:${SAVE_INTERVAL}" \
  "VAL_LOG_INTERVAL:${VAL_LOG_INTERVAL}" \
  "VAL_NUM_BATCHES:${VAL_NUM_BATCHES}" \
  "NUM_WORKERS:${NUM_WORKERS}"; do
  key="${pair%%:*}"; value="${pair#*:}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${key} must be a positive integer, got '${value}'"
done

[[ "${PYTORCH_TRAINING_PRECISION}" == "bfloat16" ]] || die "H20 formal requires BF16"
[[ "${ACCELERATE_MIXED_PRECISION}" == "bf16" ]] || die "H20 formal requires Accelerate bf16"
(( BATCH_SIZE_PER_GPU == 8 )) || die "H20 formal requires BATCH_SIZE_PER_GPU=8"
(( GRADIENT_ACCUMULATION_STEPS == 1 )) || die "H20 formal requires GRADIENT_ACCUMULATION_STEPS=1"

GLOBAL_BATCH_SIZE=$(( BATCH_SIZE_PER_GPU * TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS ))
(( GLOBAL_BATCH_SIZE == 256 )) || die "global batch must be 256 (B8×W32×GA1), got ${GLOBAL_BATCH_SIZE}"

# ---- W&B (fresh, no resume) ------------------------------------------------
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLED="${WANDB_DISABLED:-0}"
[[ "${WANDB_MODE}" == "online" ]] || die "formal runs require WANDB_MODE=online"
[[ "${WANDB_DISABLED}" == "0" ]] || die "formal runs require WANDB_DISABLED=0"
export WANDB_MODE WANDB_DISABLED
unset WANDB_RESUME WANDB_RUN_ID

# ---- Allocator config ------------------------------------------------------
FORMAL_CUDA_ALLOC_CONF="expandable_segments:True"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${FORMAL_CUDA_ALLOC_CONF}}"
[[ "${PYTORCH_CUDA_ALLOC_CONF}" == "${FORMAL_CUDA_ALLOC_CONF}" ]] \
  || die "H20 formal requires PYTORCH_CUDA_ALLOC_CONF=${FORMAL_CUDA_ALLOC_CONF}"
export PYTORCH_CUDA_ALLOC_CONF

# ---- Pinned HL data paths --------------------------------------------------
FORMAL_B1K_DATASET_ROOT="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/"
FORMAL_B1K_ASSETS_DIR="${FORMAL_BASE_CKPT}/assets/behavior-1k/2025-challenge-demos"
FORMAL_NORM_STATS_PATH="${FORMAL_B1K_ASSETS_DIR}/norm_stats.json"
BASE_PI05_CKPT="${BASE_PI05_CKPT:-${FORMAL_BASE_CKPT}}"
B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-${FORMAL_B1K_DATASET_ROOT}}"
B1K_ASSETS_DIR="${B1K_ASSETS_DIR:-${FORMAL_B1K_ASSETS_DIR}}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${FORMAL_NORM_STATS_PATH}}"
# H20-reachable caches. The A100 launcher points these at saiwenresearch, which
# this pool cannot mount.
REPO_OPENPI_CACHE="${REPO_OPENPI_CACHE:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/.cache/openpi}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-${REPO_OPENPI_CACHE}/big_vision/paligemma_tokenizer.model}"
[[ -s "${BASE_PI05_CKPT}/model.safetensors" ]] || die "base weights missing: ${BASE_PI05_CKPT}/model.safetensors"
[[ -d "${B1K_DATASET_ROOT}" ]] || die "B1K dataset root missing: ${B1K_DATASET_ROOT}"
[[ -s "${NORM_STATS_PATH}" ]] || die "norm stats missing: ${NORM_STATS_PATH}"
[[ -s "${PALIGEMMA_TOKENIZER}" ]] || die "PaliGemma tokenizer missing: ${PALIGEMMA_TOKENIZER}"

# Arm/normalization pairing proof. Getting this wrong is the single most damaging
# silent error available here — it would train on the wrong action statistics and
# also void the sampled token-length bound.
ACTUAL_WEIGHTS_BYTES="$(stat -c%s "${BASE_PI05_CKPT}/model.safetensors")"
(( ACTUAL_WEIGHTS_BYTES == EXPECTED_WEIGHTS_BYTES )) \
  || die "arm ${ARM} weights size mismatch" \
        "expected ${EXPECTED_WEIGHTS_BYTES} B, got ${ACTUAL_WEIGHTS_BYTES} B at ${BASE_PI05_CKPT}/model.safetensors"
ACTUAL_NORM_STATS_SHA256="$(sha256sum "${NORM_STATS_PATH}" | awk '{print $1}')"
[[ "${ACTUAL_NORM_STATS_SHA256}" == "${EXPECTED_NORM_STATS_SHA256}" ]] \
  || die "arm ${ARM} norm_stats digest mismatch — wrong normalization for these weights" \
        "path     ${NORM_STATS_PATH}" \
        "expected ${EXPECTED_NORM_STATS_SHA256}" \
        "actual   ${ACTUAL_NORM_STATS_SHA256}"

# ---- DeepSpeed config preflight --------------------------------------------
"${OPENPI_PREFLIGHT_PYTHON}" - "${ACCEL_CONFIG}" "${DEEPSPEED_CONFIG}" <<'PY'
import json, re, sys
from pathlib import Path
acc = Path(sys.argv[1]).read_text()
if re.search(r"^mixed_precision\s*:", acc, re.MULTILINE):
    raise SystemExit("ERROR: Accelerate config must not define top-level mixed_precision with deepspeed_config_file")
if "deepspeed_config_file: configs/deepspeed_zero2_h20_bf16.json" not in acc:
    raise SystemExit("ERROR: Accelerate config does not reference deepspeed_zero2_h20_bf16.json")
ds = json.loads(Path(sys.argv[2]).read_text())
checks = {
    "ZeRO stage 2": ds.get("zero_optimization", {}).get("stage") == 2,
    "optimizer offload disabled": "offload_optimizer" not in ds.get("zero_optimization", {}),
    "no parameter offload": "offload_param" not in ds.get("zero_optimization", {}),
    "bf16 enabled": ds.get("bf16", {}).get("enabled") is True,
    "fp16 disabled": ds.get("fp16", {}).get("enabled") is False,
    "GA auto": ds.get("gradient_accumulation_steps") == "auto",
}
failed = [n for n, ok in checks.items() if not ok]
if failed:
    raise SystemExit("ERROR: H20 BF16 DeepSpeed contract failed: " + ", ".join(failed))
print("H20_BF16_ZERO2_PREFLIGHT_OK stage=2 offload=none bf16=true memory=UNMEASURED")
PY

# ---- TrainConfig preflight -------------------------------------------------
"${OPENPI_PREFLIGHT_PYTHON}" - "${EXPECTED_CONFIG}" "${EXPECTED_MODEL}" "${MODE}" \
  "${BASE_PI05_CKPT}" "${B1K_DATASET_ROOT}" <<'PY'
import sys
from openpi.training.train_config import get_config
name, model_name, mode, base_ckpt, data_root = sys.argv[1:]
cfg = get_config(name)
data_cfg = cfg.data[0]
checks = {
    "registered exact name": cfg.name == name,
    "expected model": cfg.pytorch_model_name == model_name,
    "bf16 precision": cfg.pytorch_training_precision == "bfloat16",
    "Accelerate bf16": cfg.accelerate_mixed_precision == "bf16",
    "model dtype bfloat16": cfg.model.dtype == "bfloat16",
    "KI enabled": cfg.model.knowledge_insulation is True,
    "expert KV truncated": cfg.model.truncate_expert_kv is True,
    "B8": cfg.batch_size_per_gpu == 8,
    "GA1": cfg.gradient_accumulation_steps == 1,
    "gradient checkpointing on": cfg.gradient_checkpointing is True,
    "FAST cap 256 (shared by both arms)": cfg.model.action_token_max_len == 256,
    "warm start points at this arm's package": str(cfg.pytorch_weight_path) == base_ckpt,
    "assets come from the same package": str(data_cfg.assets.assets_dir) == f"{base_ckpt}/assets",
    "HL dataset root": str(data_cfg.base_config.behavior_dataset_root) == data_root,
    "no LQ path in weights": "saiwenresearch" not in str(cfg.pytorch_weight_path),
    "no LQ path in assets": "saiwenresearch" not in str(data_cfg.assets.assets_dir),
    "no LQ path in data root": "saiwenresearch" not in str(data_cfg.base_config.behavior_dataset_root),
    "wandb enabled": cfg.wandb_enabled is True,
    "project pi05_ki": cfg.project_name == "pi05_ki",
    "peak1e-5": cfg.lr_schedule.peak_lr == 1e-5,
    "no epoch budget": cfg.num_train_epochs is None,
}
if mode == "formal":
    checks.update({
        "104912 steps": cfg.num_train_steps == 104_912,
        "stride12": cfg.streaming_anchor_stride == 12,
        "save10k": cfg.save_interval == 10_000,
        "val1k": cfg.val_log_interval == 1_000,
        "val20": cfg.val_num_batches == 20,
        "warmup1000": cfg.lr_schedule.warmup_steps == 1_000,
        "decay104912": cfg.lr_schedule.decay_steps == 104_912,
    })
else:
    checks.update({
        "bounded smoke budget": 0 < cfg.num_train_steps <= 16,
        "smoke stride1": cfg.streaming_anchor_stride == 1,
        # HARD REQUIREMENT. Validation fires on
        # `global_step % val_log_interval == 0 and global_step > 0`, so the smoke
        # must complete at least one FULL validation pass strictly before its
        # final step. Variant A's known historical failure is inside
        # compute_eval_metrics (the trainer passes deterministic_flow to both KI
        # variants unconditionally and Variant A's override lacked it, killing an
        # A100 FAST run at its first validation). A smoke that never reaches
        # validation would pass and then let us promote a broken run to the
        # formal budget.
        "smoke reaches validation before its last step":
            cfg.val_log_interval < cfg.num_train_steps
            and cfg.num_train_steps // cfg.val_log_interval >= 2,
    })
failed = [label for label, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"ERROR: config {name!r} failed: " + "; ".join(failed))
print(
    f"H20_CONFIG_PREFLIGHT_OK name={name} model={model_name} mode={mode} "
    f"B8xW32xGA1=256 steps={cfg.num_train_steps} stride={cfg.streaming_anchor_stride} cap=256"
)
PY

# ---- FAST tokenizer preflight ----------------------------------------------
export OPENPI_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME="${REPO_OPENPI_CACHE}"
export OPENPI_BEHAVIOR_DATASET_ROOT="${B1K_DATASET_ROOT}"
# No FAST processor was cached on any H20-mounted volume, so one is staged here.
# Variant A cannot tokenize actions without it and the pool runs offline.
OPENPI_FAST_TOKENIZER_PATH="${OPENPI_FAST_TOKENIZER_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/h20_fastce/fast_tokenizer}"
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

# ---- Warm-start key-mapping preflight (fail closed) ------------------------
# train_accelerate.py loads pi05_ki_joint_fast weights with strict=False, so a
# warm start that matched nothing would still log success and train from noise.
# Prove the mapping before spending GPUs.
"${OPENPI_PREFLIGHT_PYTHON}" "${REPO_ROOT}/scripts/verify_warm_start_keymap.py" \
  --config "${EXPECTED_CONFIG}" \
  --checkpoint "${BASE_PI05_CKPT}" \
  --max-unexpected 0 \
  || die "warm-start key mapping preflight failed for arm ${ARM} (${BASE_PI05_CKPT})"

# ---- Output identity (arm-first, unique Arnold job id before truncation) ----
JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-manual}}"
JOB_ID_SAFE="${JOB_ID//[^A-Za-z0-9_.-]/_}"
PERSISTENT_OUTPUT_BASE="${PERSISTENT_OUTPUT_BASE:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/outputs/pi05_ki_h20_bf16_formal}"
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${PERSISTENT_OUTPUT_BASE}/${ARM_LABEL}/${JOB_ID_SAFE}}"
EXP_NAME="${EXP_NAME:-A_fast_ce_h20_arm${ARM}_${MODE}_4n8g_${JOB_ID_SAFE}}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/checkpoints}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PERSISTENT_OUTPUT_ROOT}/logs}"
CONSOLE_LOG_DIR="${PERSISTENT_OUTPUT_ROOT}/console_logs/${EXP_NAME}"
CONSOLE_LOG="${CONSOLE_LOG_DIR}/node${NODE_RANK}.log"

info "============================================================"
info "formal H20 arm=${ARM} (${ARM_LABEL}) mode=${MODE} config=${EXPECTED_CONFIG}"
info "code_commit=${ACTUAL_COMMIT} openpi=${OPENPI_IMPORT_PATH}"
info "topology=${NUM_NODES}x${GPUS_PER_NODE} world=${TOTAL_GPUS} rank=${NODE_RANK} gpu_model=${GPU_MODEL:-unset}"
info "BF16 ZeRO-2 no-optimizer-offload B8/GPU GA1 global_batch=${GLOBAL_BATCH_SIZE} gradient_checkpointing=on"
info "warm_start=${BASE_PI05_CKPT} (${ACTUAL_WEIGHTS_BYTES} B)"
info "norm_stats=${NORM_STATS_PATH} sha256=${ACTUAL_NORM_STATS_SHA256:0:16}…"
info "FAST cap=256 (SAMPLED bound under d66ed168; exhaustive gate required before the long run)"
info "WARNING: world32 B8/GA1 no-offload peak on H20 is UNMEASURED; bounded smoke required first"
info "wandb_project=${WANDB_PROJECT} output_root=${PERSISTENT_OUTPUT_ROOT}"
info "============================================================"

if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  info "PREFLIGHT_OK: CPU/read-only checks passed; no GPU, output, or occupier action."
  exit 0
fi

# ---- CUDA/driver preflight on every node -----------------------------------
# Two layers. cuda_preflight.py is the established per-node gate (cuInit, driver
# floor, device count, GPU-0 context). cuda_preflight_all_devices.py then touches
# EVERY device and proves native BF16 per device — the GPU-0-only probe would let
# a node with one bad or already-held GPU through and it would fail later inside
# c10d bootstrap, which is not watchdog-protected.
"${OPENPI_PREFLIGHT_PYTHON}" "${CUDA_PREFLIGHT}" --min-gpus "${GPUS_PER_NODE}" --min-driver-major 525 \
  || die "CUDA/driver preflight failed on node rank ${NODE_RANK}"
"${OPENPI_PREFLIGHT_PYTHON}" "${CUDA_PREFLIGHT_ALL}" --expect-gpus "${GPUS_PER_NODE}" --require-bf16 \
  || die "per-device CUDA/BF16 preflight failed on node rank ${NODE_RANK}"

# ---- Wrapper contract ------------------------------------------------------
[[ "${OPENPI_KI_TRAINING_INNER:-0}" == "1" ]] \
  || die "formal Merlin entrypoint must be the outer keepalive wrapper" \
      "Set OPENPI_KI_TRAINING_INNER=1 and LAUNCHER=${SCRIPT_PATH} on the wrapper."
[[ "${KEEPALIVE_DISABLE:-}" == "0" ]] || die "outer wrapper requires KEEPALIVE_DISABLE=0"
[[ "${KEEPALIVE_ON_SUCCESS:-}" == "1" ]] || die "outer wrapper requires KEEPALIVE_ON_SUCCESS=1"
[[ "${STRICT_GPU_COUNT:-}" == "0" ]] || die "outer wrapper requires STRICT_GPU_COUNT=0"

[[ "${BASE_PI05_CKPT}" == "${FORMAL_BASE_CKPT}" ]] || die "base checkpoint pinned to ${FORMAL_BASE_CKPT}"
[[ "${B1K_DATASET_ROOT}" == "${FORMAL_B1K_DATASET_ROOT}" ]] || die "dataset root pinned"
[[ "${B1K_ASSETS_DIR}" == "${FORMAL_B1K_ASSETS_DIR}" ]] || die "assets dir pinned"
[[ "${NORM_STATS_PATH}" == "${FORMAL_NORM_STATS_PATH}" ]] || die "norm stats pinned"

# The hold job keeps tagged matmul occupiers on every GPU to stay above the
# platform utilization floor. Training needs exclusive devices, so the occupiers
# for these GPUs must already have been stopped by the entrypoint before this
# point. Verified from live process state, never from a shared PID file.
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

# behavior-data-hl conda root: navigation-hl's env cannot import the model.
CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/behavior-data-hl/chenjunting/miniconda3}"
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
  --batch-size-per-gpu 8 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --num-workers "${NUM_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --val-log-interval "${VAL_LOG_INTERVAL}" \
  --val-num-batches "${VAL_NUM_BATCHES}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  --log-base-dir "${LOG_BASE_DIR}" \
  2>&1 | tee -a "${CONSOLE_LOG}"
