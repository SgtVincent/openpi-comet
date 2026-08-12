#!/usr/bin/env bash
# ============================================================================
# π0.5-KI joint-query Skill Bridge BF16 — 8 nodes × 8 A100 (64 GPUs), LQ.
#
# ALL-IN-ONE MERLIN ENTRYPOINT. The whole Merlin entrypoint is one line:
#
#   exec bash scripts/run_pi05_skillbridge_bf16_multinode_lq_8x8.sh
#
# TRAINING CONTRACT (current)
# ---------------------------
#   * BATCH_SIZE_PER_GPU = 4  ->  global batch 4 × 64 = 256 (grad-accum 1)
#   * NUM_TRAIN_EPOCHS   = 3  ->  three FULL stride-1 passes over Full B1K
#   * NUM_TRAIN_STEPS    = 0  ->  step cap DISABLED; the epoch budget governs
#
# NOT a strict Tracking Run 2 A/B. Run 2 (the non-Skill-Bridge control)
# warmstarted from a 360k-step checkpoint that lives only on HL NAS and is
# unavailable here; this run warmstarts from the LQ base pi05 checkpoint.
# Run 2 also used a fixed 104,912-step budget across three stride-12 passes
# (offsets 0/4/8); this run uses 3 real stride-1 epochs with the standard
# loader. The only intended algorithmic A/B difference vs the control is
# skill_bridge.enabled=True.
#
# Expect roughly 420,689 steps/epoch and ~1,262,067 optimizer steps in total
# at global batch 256, but those are ESTIMATES: the authoritative
# steps_per_epoch is computed at runtime from the actual dataloader length.
# See the "Frozen training schedule" block below for exact trainer semantics.
#
# WHY THIS EXISTS
# ---------------
# The previous 8×8 hot-update entrypoint called the keepalive wrapper directly
# and relied on the wrapper's *default* LAUNCHER, which points at
#   scripts/run_pi05_ki_joint_query_single_task_radio_skillbridge_bf16_multinode_lq.sh
# That LQ launcher hard-asserts NUM_NODES=4 / GPUS_PER_NODE=8 / TOTAL_GPUS=32,
# so an 8-node job died instantly with
#   "ERROR: this LQ formal script requires NUM_NODES=4, got 8."
# and the wrapper then (correctly, by design) fell through to GPU occupation.
#
# This script fixes that class of bug structurally:
#   * It pins LAUNCHER EXPLICITLY to the HL launcher, whose topology check is
#     generic (reads ARNOLD_WORKER_NUM, validates only sanity + rank<nodes and
#     has NO node-count hard-code), so it scales to 8 nodes unchanged.
#     Never rely on the wrapper's default LAUNCHER — that default IS the bug.
#   * It overrides every HL-flavoured default path. The HL launcher defaults
#     CONDA_ROOT and B1K_DATASET_ROOT to /mnt/bn/navigation-hl/..., and
#     /mnt/bn/navigation-hl DOES NOT EXIST on LQ, so without these overrides
#     the run fails 100% of the time.
#   * It freezes every stable knob (config / data / checkpoint / schedule /
#     W&B / keepalive) so the Merlin entrypoint carries no parameters at all
#     and cannot drift between hot-updates.
#
# LAYERING (no recursion)
# -----------------------
#   this script  --exec-->  keepalive wrapper  --bash-->  HL launcher
#                                                          --> accelerate launch
# We only ever `exec` the wrapper once, with LAUNCHER already resolved, so the
# wrapper never re-enters this script.
#
# ---------------------------------------------------------------------------
# SHELL-OPTION SEMANTICS (deliberate, please read before "hardening" this)
# ---------------------------------------------------------------------------
# `set -e` is intentionally NOT used. A Merlin entrypoint that exits releases
# the whole GPU allocation, which is exactly what the keepalive design exists
# to prevent. `set -uo pipefail` is safe here and only covers THIS script's own
# pre-flight/configuration phase.
#
# Because the final step is `exec`, the wrapper REPLACES this process image:
# shell options set here (-u / -o pipefail) are process-local shell state and
# are NOT inherited by the wrapper. The wrapper keeps its own, already-audited
# option handling and its own non-fail-fast hold path. Only exported
# environment variables cross the exec boundary.
#
# The one place we do exit non-zero on purpose is the 8×8 topology assertion.
# That is a misconfiguration detected BEFORE any GPU work starts, where
# fail-fast is correct: holding 64 GPUs for a job that can never train wastes
# the allocation instead of protecting it.
# ============================================================================

set -uo pipefail

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

die() {
  echo "ERROR[${SCRIPT_NAME}]: $*" >&2
  exit 2
}

info() {
  echo "[${SCRIPT_NAME}] $*"
}

# ---------------------------------------------------------------------------
# Repo root. Overridable for web FULL_SCRIPT mounts (e.g. /opt/tiger/...).
# ---------------------------------------------------------------------------
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || die "cannot resolve REPO_ROOT"
fi
export REPO_ROOT
cd "${REPO_ROOT}" || die "cannot cd into REPO_ROOT=${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Underlying scripts. Both paths are asserted so a rename/move fails here with
# a clear message instead of deep inside the wrapper.
#
# HL launcher is the correct base for ANY node count: its topology block reads
# ARNOLD_WORKER_NUM / ARNOLD_WORKER_GPU / ARNOLD_ID and validates only that the
# values are positive integers and NODE_RANK < NUM_NODES. It also already
# carries the short-TMPDIR alias + live multiprocess.Manager() AF_UNIX preflight
# (the 107-byte sun_path fix). Do not swap it for the 4-node-locked LQ script.
# ---------------------------------------------------------------------------
LAUNCHER="${REPO_ROOT}/scripts/run_pi05_ki_joint_query_single_task_radio_bf16_multinode_hl.sh"
KEEPALIVE_WRAPPER="${REPO_ROOT}/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh"

[[ -f "${LAUNCHER}" ]] || die "training launcher not found: ${LAUNCHER}"
[[ -f "${KEEPALIVE_WRAPPER}" ]] || die "keepalive wrapper not found: ${KEEPALIVE_WRAPPER}"

# Exported so the wrapper uses THIS launcher and never its own 4-node default.
export LAUNCHER

# ---------------------------------------------------------------------------
# Topology: strict 8 nodes × 8 GPUs = 64, read dynamically from ARNOLD_*.
# Standard names stay honoured first so manual/debug launches can override.
# ---------------------------------------------------------------------------
EXPECT_NUM_NODES=8
EXPECT_GPUS_PER_NODE=8
EXPECT_TOTAL_GPUS=$((EXPECT_NUM_NODES * EXPECT_GPUS_PER_NODE))

NUM_NODES="${NUM_NODES:-${NNODES:-${ARNOLD_WORKER_NUM:-}}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-}}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
MASTER_PORT="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-29514}}"
# Merlin may publish a comma-separated port list; take the first entry.
MASTER_PORT="${MASTER_PORT%%,*}"
JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-manual}}"

[[ -n "${NUM_NODES}" ]] || die "NUM_NODES/ARNOLD_WORKER_NUM is not set; cannot verify the 8×8 topology"
[[ -n "${GPUS_PER_NODE}" ]] || die "GPUS_PER_NODE/ARNOLD_WORKER_GPU is not set; cannot verify the 8×8 topology"
[[ -n "${NODE_RANK}" ]] || die "NODE_RANK/ARNOLD_ID is not set; cannot verify the 8×8 topology"

[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES is not a positive integer: ${NUM_NODES}"
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_NODE is not a positive integer: ${GPUS_PER_NODE}"
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "NODE_RANK is not a non-negative integer: ${NODE_RANK}"

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# This is the assertion that the broken 8×8 entrypoint tripped, inverted: this
# script is the 8×8 script, so 4×8 (and anything else) must be rejected here.
if [[ "${NUM_NODES}" != "${EXPECT_NUM_NODES}" ]]; then
  die "this LQ 8×8 entrypoint requires NUM_NODES=${EXPECT_NUM_NODES}, got ${NUM_NODES}." \
      "For a ${NUM_NODES}-node run use the launcher matching that topology."
fi
if [[ "${GPUS_PER_NODE}" != "${EXPECT_GPUS_PER_NODE}" ]]; then
  die "this LQ 8×8 entrypoint requires GPUS_PER_NODE=${EXPECT_GPUS_PER_NODE}, got ${GPUS_PER_NODE}."
fi
if [[ "${TOTAL_GPUS}" != "${EXPECT_TOTAL_GPUS}" ]]; then
  die "this LQ 8×8 entrypoint requires TOTAL_GPUS=${EXPECT_TOTAL_GPUS}, got ${TOTAL_GPUS}."
fi
if (( NODE_RANK >= NUM_NODES )); then
  die "NODE_RANK=${NODE_RANK} must be less than NUM_NODES=${NUM_NODES}."
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  die "MASTER_ADDR/ARNOLD_WORKER_0_HOST is required for the multi-node rendezvous."
fi

export NUM_NODES GPUS_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT

# ---------------------------------------------------------------------------
# Frozen LQ paths.
#
# Every value below overrides an HL-flavoured default in the HL launcher.
# Authoritative source: the existing 4×8 LQ launcher (verified to exist on LQ).
# /mnt/bn/navigation-hl is NOT mounted on LQ, so omitting any of these is a
# guaranteed failure, not a graceful degradation.
# ---------------------------------------------------------------------------
LQ_NAS_USER_ROOT="/mnt/bn/saiwenresearch/mlx/users/chenjunting"
LQ_CANONICAL_REPO="${LQ_NAS_USER_ROOT}/repo/openpi-comet"

export CONDA_ROOT="${CONDA_ROOT:-${LQ_NAS_USER_ROOT}/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-openpi-comet-nas}"

# Base π0.5 checkpoint + its coherent normalization assets live in the CANONICAL
# repo. Git worktrees do NOT carry checkpoints/ (it is untracked there), so this
# must be the canonical absolute path and not ${REPO_ROOT}/checkpoints.
export BASE_PI05_CKPT="${BASE_PI05_CKPT:-${LQ_CANONICAL_REPO}/checkpoints/pi05_base_pytorch}"
# NOTE: the HL launcher derives B1K_ASSETS_DIR and NORM_STATS_PATH from
# BASE_PI05_CKPT, so pinning BASE_PI05_CKPT keeps weights and norm stats
# coherent automatically. Do not set them separately.

export B1K_DATASET_ROOT="${B1K_DATASET_ROOT:-${LQ_NAS_USER_ROOT}/data/2025-challenge-demos/}"
# Offline tokenizer source cache, also taken from the canonical repo.
export REPO_OPENPI_CACHE="${REPO_OPENPI_CACHE:-${LQ_CANONICAL_REPO}/.cache/openpi}"

# Skill Bridge BF16 training config (registered in
# src/openpi/training/pi05_ki_joint_query_config.py).
#
# Full-B1K Skill Bridge variant (50 tasks, 180 train / 180-199 val episodes
# per task, stride-1 loader). NOT a strict Run 2 A/B: warmstarts from the LQ
# base pi05 checkpoint (Run 2's 360k-step warmstart lives on HL NAS and is
# unavailable on LQ), runs 3 real stride-1 epochs instead of Run 2's fixed
# 104,912-step three stride-12 passes, and uses B4 on 64 GPUs for the same
# global batch 256. The only intended algorithmic A/B difference vs the
# non-Skill-Bridge control is skill_bridge.enabled=True.
export CONFIG_NAME="${CONFIG_NAME:-pi05_ki_joint_query_b1k-full_task-ki_on_skillbridge_bf16}"

# Dedicated 8×8 output root, kept separate from the 4×8 root so the two
# topologies never share checkpoints or experiment-name sync state.
# This root may already contain artifacts from earlier 8×8 attempts. That is
# safe because the HL launcher mints a fresh timestamped EXP_NAME per run
# (unless RESUME=1), so each run gets its own checkpoint/log subtree and no
# stale wandb_id.txt is reachable. Point PERSISTENT_OUTPUT_ROOT elsewhere if a
# fully pristine tree is ever required.
export PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${LQ_NAS_USER_ROOT}/repo/outputs/pi05_skillbridge_a100_lq_bf16_8x8}"

# IMPORTANT: EXP_NAME is deliberately left UNSET. The HL launcher elects the
# experiment name on rank 0 and publishes it to the other ranks through a NAS
# sync file. Setting it here would embed a per-node `date` value and give the 8
# nodes divergent names / checkpoint dirs.

# ---------------------------------------------------------------------------
# Frozen training schedule: B4 × 64 GPUs = global batch 256, 3 Full-B1K epochs.
#
# NUM_TRAIN_STEPS=0 DISABLES the step cap on purpose. In
# scripts/train_accelerate.py the budget resolves as
#     computed_steps = num_train_epochs * steps_per_epoch
#     target_steps   = computed_steps if num_train_steps <= 0
#                      else min(num_train_steps, computed_steps)
# so a non-positive NUM_TRAIN_STEPS selects computed_steps outright with no
# min() clamp. That is the only way to guarantee all 3 epochs actually run.
#
# THE STEP COUNT IS APPROXIMATE. steps_per_epoch is computed at RUNTIME from
# the real dataloader length, never here. For Full B1K (50 tasks × 180 train
# episodes, stride-1) at global batch 256 the estimate is ~420,689 steps/epoch
# and ~1,262,067 optimizer steps across 3 epochs. Estimates only — not a
# contract; the trainer prints the authoritative computed value at startup.
#
# LR SCHEDULE: warmup 10,000 (~1% of the ~1.26M-step budget), peak_lr 1e-5,
# cosine decay to 0. The new Full-B1K Skill Bridge config sets
# decay_steps=0, which the trainer (train_accelerate.py) treats as "use
# num_train_steps" via its decay_steps<=0 branch (info-level log, no warning).
# decay therefore spans the whole 3-epoch run and reaches 0 exactly at the end.
#
# BATCH_SIZE_PER_GPU=4 is a deliberate, user-mandated choice. It roughly
# quadruples activation memory versus the validated B1 run and so carries real
# OOM risk on A100. There is intentionally NO protective fallback or
# auto-downgrade: if it OOMs, that outcome is itself the signal being sought.
# The keepalive wrapper still holds the 64-GPU allocation on failure, so an OOM
# does not cost the batch.
# ---------------------------------------------------------------------------
export PYTORCH_TRAINING_PRECISION="${PYTORCH_TRAINING_PRECISION:-bfloat16}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-0}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
export VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-100}"

# Validate the two schedule knobs that feed shell arithmetic below, so a typo'd
# env override fails here with a clear message instead of as a cryptic
# arithmetic error (or, worse, reaching the trainer).
[[ "${NUM_TRAIN_STEPS}" =~ ^-?[0-9]+$ ]] || die "NUM_TRAIN_STEPS must be an integer, got '${NUM_TRAIN_STEPS}'"
[[ "${NUM_TRAIN_EPOCHS}" =~ ^[0-9]+$ ]] || die "NUM_TRAIN_EPOCHS must be a non-negative integer, got '${NUM_TRAIN_EPOCHS}'"
[[ "${BATCH_SIZE_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE_PER_GPU must be a positive integer, got '${BATCH_SIZE_PER_GPU}'"

# Reporting-only derived numbers (never passed to the trainer).
# GLOBAL_BATCH is exact: micro-batch × world size, grad-accum is 1 here.
GLOBAL_BATCH=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS))
# Step estimate for Full B1K (all 50 tasks, 180 train episodes/task, stride-1)
# at global batch 256. Anchored on Tracking Run 2's measured throughput at the
# same global batch; the trainer computes the authoritative steps_per_epoch at
# runtime from the real dataloader length, so this is a reporting estimate only.
EST_STEPS_PER_EPOCH=420689
EST_TOTAL_STEPS=$((EST_STEPS_PER_EPOCH * NUM_TRAIN_EPOCHS))
# Human-readable step-cap state, so the banner stays truthful even if a caller
# re-enables the cap through the environment.
if (( NUM_TRAIN_STEPS > 0 )); then
  STEP_CAP_DESC="step cap ACTIVE at ${NUM_TRAIN_STEPS}"
else
  STEP_CAP_DESC="step cap disabled"
fi

# ---------------------------------------------------------------------------
# Online Byted-W&B.
#
# The HL launcher sets WANDB_DISABLED/WANDB_MODE from OPENPI_OFFLINE using
# `${WANDB_DISABLED:-1}` / `${WANDB_MODE:-disabled}`, i.e. it only supplies
# defaults. Exporting explicit values here therefore survives, and the
# launcher's `--no-wandb-enabled` branch (taken only when WANDB_DISABLED=1)
# stays off. HuggingFace offline controls remain independent of W&B.
# ---------------------------------------------------------------------------
export WANDB_DISABLED="${WANDB_DISABLED:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
case "${WANDB_DISABLED}" in
  1 | true | TRUE | True)
    die "this 8×8 entrypoint requires online Byted-W&B, but WANDB_DISABLED=${WANDB_DISABLED}."
    ;;
esac
case "${WANDB_MODE}" in
  disabled | DISABLED | Disabled | offline | OFFLINE | Offline | dryrun | DRYRUN | Dryrun)
    die "this 8×8 entrypoint requires WANDB_MODE=online, got ${WANDB_MODE}."
    ;;
esac

# ---------------------------------------------------------------------------
# Keepalive policy: hold the 64-GPU allocation after training terminates,
# whether it succeeded or failed, so the batch is not returned to Merlin.
# STRICT_GPU_COUNT stays 0 on purpose: a GPU-count mismatch must be recorded
# and then kept holding, never turned into an exit.
# ---------------------------------------------------------------------------
export KEEPALIVE_DISABLE="${KEEPALIVE_DISABLE:-0}"
export KEEPALIVE_ON_SUCCESS="${KEEPALIVE_ON_SUCCESS:-1}"
export EXPECTED_GPUS_PER_NODE="${EXPECTED_GPUS_PER_NODE:-${EXPECT_GPUS_PER_NODE}}"
export STRICT_GPU_COUNT="${STRICT_GPU_COUNT:-0}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
info "============================================================"
info "π0.5-KI Skill Bridge BF16 — LQ 8×8 all-in-one entrypoint"
info "PLAN=B${BATCH_SIZE_PER_GPU} × ${TOTAL_GPUS} GPUs = global batch ${GLOBAL_BATCH}, ${NUM_TRAIN_EPOCHS} full epochs, ${STEP_CAP_DESC}"
info "REPO_ROOT=${REPO_ROOT}"
info "CONFIG_NAME=${CONFIG_NAME}"
info "TOPOLOGY=${NUM_NODES} nodes × ${GPUS_PER_NODE} GPUs (rank ${NODE_RANK}, world ${TOTAL_GPUS})"
info "RENDEZVOUS=${MASTER_ADDR}:${MASTER_PORT}"
info "ARNOLD_JOB_ID=${JOB_ID}"
info "CONDA=${CONDA_ROOT} (env ${CONDA_ENV})"
info "BASE_PI05_CKPT=${BASE_PI05_CKPT}"
info "B1K_DATASET_ROOT=${B1K_DATASET_ROOT}"
info "REPO_OPENPI_CACHE=${REPO_OPENPI_CACHE}"
info "PERSISTENT_OUTPUT_ROOT=${PERSISTENT_OUTPUT_ROOT}"
if (( NUM_TRAIN_STEPS > 0 )); then
  info "BUDGET=min(${NUM_TRAIN_STEPS} steps, ${NUM_TRAIN_EPOCHS} epochs) [step cap ACTIVE - overridden from the default]"
else
  info "BUDGET=${NUM_TRAIN_EPOCHS} full epochs, step cap DISABLED (NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS})"
  info "BUDGET_ESTIMATE=~${EST_STEPS_PER_EPOCH} steps/epoch × ${NUM_TRAIN_EPOCHS} ≈ ${EST_TOTAL_STEPS} optimizer steps"
  info "BUDGET_NOTE=estimate only; steps_per_epoch is computed at runtime by the dataloader"
  info "BUDGET_NOTE=LR decay_steps=0 in config; trainer auto-aligns decay to the computed total steps (info log)"
fi
info "INTERVALS=validation ${VAL_LOG_INTERVAL}, save ${SAVE_INTERVAL}"
info "BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU} NUM_WORKERS=${NUM_WORKERS}"
info "GLOBAL_BATCH=${GLOBAL_BATCH} (${BATCH_SIZE_PER_GPU} per GPU × ${TOTAL_GPUS} GPUs)"
info "WANDB_MODE=${WANDB_MODE} (WANDB_DISABLED=${WANDB_DISABLED})"
info "KEEPALIVE_ON_SUCCESS=${KEEPALIVE_ON_SUCCESS} STRICT_GPU_COUNT=${STRICT_GPU_COUNT}"
info "LAUNCHER=${LAUNCHER}"
info "============================================================"

# ---------------------------------------------------------------------------
# Preflight-only escape hatch: bypass the keepalive wrapper entirely.
#
# WHY: with KEEPALIVE_ON_SUCCESS=1 the wrapper would treat a successful
# preflight (rc=0) as "training finished" and start real GPU occupier
# processes. That makes CPU-only verification of the short-TMPDIR /
# multiprocess.Manager() AF_UNIX fix impossible without touching GPUs.
# Exec'ing the launcher directly keeps OPENPI_LAUNCH_PREFLIGHT_ONLY=1 a pure,
# side-effect-free check. Topology validation above has already run, so this
# path still enforces 8×8.
# ---------------------------------------------------------------------------
if [[ "${OPENPI_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  info "OPENPI_LAUNCH_PREFLIGHT_ONLY=1 — bypassing the keepalive wrapper and"
  info "exec'ing the launcher directly (no GPU occupation, no training)."
  exec bash "${LAUNCHER}"
fi

# ---------------------------------------------------------------------------
# Normal path: hand over to the keepalive wrapper.
#
# `exec` keeps a single foreground process for Merlin to observe and means this
# script's shell options do not leak into the wrapper (see the header note).
# The wrapper owns failure handling from here on and must never fail-fast.
# ---------------------------------------------------------------------------
info "exec'ing keepalive wrapper: ${KEEPALIVE_WRAPPER}"
exec bash "${KEEPALIVE_WRAPPER}"
