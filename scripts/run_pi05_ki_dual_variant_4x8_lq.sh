#!/usr/bin/env bash
# =============================================================================
# pi0.5-KI dual-variant entrypoint: 8 nodes split into two independent 4x8 arms
# =============================================================================
#
# WHAT THIS DOES
#   Takes ONE 8-node x 8-GPU (64 GPU) Merlin allocation and runs TWO fully
#   independent 32-GPU trainings inside it:
#
#     nodes 0-3  -> arm A : Variant A, FAST frequency-space discrete action
#                           tokens + cross-entropy backbone objective
#     nodes 4-7  -> arm B : Variant B, learned action queries + MSE backbone
#                           objective (the current shipping model)
#
#   Both arms run the ORIGINAL skill-annotation recipe, i.e. Skill Bridge is
#   DISABLED (skill_bridge_enabled=False) so the only intended difference
#   between the arms is the backbone action representation.
#
# WHY SPLITTING WORKS AT ALL
#   Each node belongs to exactly ONE arm, so each node still runs exactly one
#   training process and the existing keepalive wrapper's single-process model
#   (TRAIN_RC="${PIPESTATUS[0]}") is preserved untouched.
#
#   What the split does require is a SEPARATE rendezvous per arm: arm B's
#   machine_rank 0 lives on node 4, so it cannot rendezvous on node 0's
#   address. See "RENDEZVOUS" below.
#
# GPU OCCUPANCY IS THE TOP PRIORITY
#   Per the operating requirement, holding the allocation outranks training
#   liveness. Both arms therefore delegate to
#   run_pi05_skillbridge_lq_keepalive_on_failure.sh with
#   KEEPALIVE_ON_SUCCESS=1 and STRICT_GPU_COUNT=0, so the wrapper starts matmul
#   occupier processes and keeps holding the GPUs when training exits, whether
#   it failed OR succeeded, and a GPU-count mismatch is recorded rather than
#   turned into an exit.
#
# USAGE
#   exec bash scripts/run_pi05_ki_dual_variant_4x8_lq.sh
#
# PREFLIGHT (no GPU touched, no training, no occupier)
#   OPENPI_DUAL_PREFLIGHT_ONLY=1 bash scripts/run_pi05_ki_dual_variant_4x8_lq.sh
# =============================================================================

set -uo pipefail

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_ts() { date '+%Y-%m-%d %H:%M:%S'; }
info() { printf '[dual][%s] %s\n' "$(_ts)" "$*"; }
die() {
  printf '[dual][%s] FATAL: %s\n' "$(_ts)" "$1" >&2
  shift || true
  for line in "$@"; do printf '[dual]        %s\n' "${line}" >&2; done
  exit 1
}

readonly REPO_ROOT="/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/kv-reuse-mask"
readonly KEEPALIVE_WRAPPER="${REPO_ROOT}/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh"
# Generic launcher: reads NUM_NODES / NODE_RANK / MASTER_ADDR / MASTER_PORT /
# CONFIG_NAME from the environment and has NO hard topology gate, which is why
# it can serve a 4-node arm. The 8x8 entrypoint cannot: it hard-asserts
# NUM_NODES=8 and would reject every arm here.
readonly ARM_LAUNCHER="${REPO_ROOT}/scripts/run_pi05_ki_joint_query_single_task_radio_bf16_multinode_hl.sh"

# -----------------------------------------------------------------------------
# Topology
# -----------------------------------------------------------------------------
NUM_NODES_TOTAL="${NUM_NODES_TOTAL:-${ARNOLD_WORKER_NUM:-}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-}}"
NODE_RANK_GLOBAL="${NODE_RANK_GLOBAL:-${ARNOLD_ID:-}}"

readonly EXPECT_TOTAL_NODES=8
readonly EXPECT_GPUS_PER_NODE=8
readonly NODES_PER_ARM=4

[[ -n "${NUM_NODES_TOTAL}" ]] || die "ARNOLD_WORKER_NUM is unset; cannot determine the allocation size."
[[ -n "${GPUS_PER_NODE}" ]]   || die "ARNOLD_WORKER_GPU is unset; cannot determine GPUs per node."
[[ -n "${NODE_RANK_GLOBAL}" ]] || die "ARNOLD_ID is unset; cannot determine this node's global rank."

[[ "${NUM_NODES_TOTAL}"  =~ ^[1-9][0-9]*$ ]] || die "NUM_NODES_TOTAL is not a positive integer: ${NUM_NODES_TOTAL}"
[[ "${GPUS_PER_NODE}"    =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_NODE is not a positive integer: ${GPUS_PER_NODE}"
[[ "${NODE_RANK_GLOBAL}" =~ ^[0-9]+$ ]]      || die "NODE_RANK_GLOBAL is not a non-negative integer: ${NODE_RANK_GLOBAL}"

if (( NUM_NODES_TOTAL != EXPECT_TOTAL_NODES )); then
  die "this dual-variant entrypoint needs exactly ${EXPECT_TOTAL_NODES} nodes, got ${NUM_NODES_TOTAL}." \
      "It splits the allocation into two ${NODES_PER_ARM}-node arms."
fi
if (( GPUS_PER_NODE != EXPECT_GPUS_PER_NODE )); then
  die "this entrypoint needs ${EXPECT_GPUS_PER_NODE} GPUs per node, got ${GPUS_PER_NODE}."
fi
if (( NODE_RANK_GLOBAL >= NUM_NODES_TOTAL )); then
  die "NODE_RANK_GLOBAL=${NODE_RANK_GLOBAL} must be < NUM_NODES_TOTAL=${NUM_NODES_TOTAL}."
fi

# Arm assignment: contiguous halves keep each arm's collectives inside a
# tighter set of hosts than an interleaved split would.
if (( NODE_RANK_GLOBAL < NODES_PER_ARM )); then
  ARM="A"
  ARM_MASTER_GLOBAL_RANK=0
else
  ARM="B"
  ARM_MASTER_GLOBAL_RANK="${NODES_PER_ARM}"
fi
ARM_NODE_RANK=$(( NODE_RANK_GLOBAL % NODES_PER_ARM ))
ARM_TOTAL_GPUS=$(( NODES_PER_ARM * GPUS_PER_NODE ))

# -----------------------------------------------------------------------------
# Per-arm configuration
# -----------------------------------------------------------------------------
# Both configs are the NON-Skill-Bridge recipe. The Variant B config is the
# existing control whose own comment records it as the "Formal lean B8/W32 run
# ... non-Skill-Bridge control", i.e. it is already sized for exactly
# 4x8 = 32 GPUs at batch 8 per GPU -> global batch 256, matching the 8x8 B4
# run it is meant to be compared against.
readonly VARIANT_B_CONFIG="${VARIANT_B_CONFIG:-pi05_ki_joint_query_b1k-full_task-ki_on_bf16}"

# Variant A is NOT implemented yet: no model registers FAST action tokens with
# a cross-entropy backbone objective. This name is the intended landing spot
# and the preflight below refuses to launch arm A until the config resolves.
readonly VARIANT_A_CONFIG="${VARIANT_A_CONFIG:-pi05_ki_joint_fast_b1k-full_task-ki_on_bf16}"

# Batch per GPU: 8 keeps global batch at 8*32 = 256 for both arms.
readonly ARM_BATCH_SIZE_PER_GPU="${ARM_BATCH_SIZE_PER_GPU:-8}"

# Distinct rendezvous ports. ARNOLD_WORKER_0_PORT may be a comma-separated
# list; take the first entry as arm A's port and derive arm B's from it so the
# two c10d stores can never collide.
_BASE_PORT="${ARNOLD_WORKER_0_PORT:-29514}"
_BASE_PORT="${_BASE_PORT%%,*}"
[[ "${_BASE_PORT}" =~ ^[0-9]+$ ]] || _BASE_PORT=29514
readonly ARM_A_PORT="${ARM_A_PORT:-${_BASE_PORT}}"
readonly ARM_B_PORT="${ARM_B_PORT:-$(( _BASE_PORT + 137 ))}"

readonly OUTPUT_BASE="${OUTPUT_BASE:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs}"
readonly RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
# RUN_TAG must be identical on every node of an arm or each node would resume
# from a different directory. Derive it from the immutable job id instead of
# per-node wall clock unless the caller pins it explicitly.
readonly JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-manual}}"
readonly SHARED_TAG="${SHARED_TAG:-job${JOB_ID}}"

if [[ "${ARM}" == "A" ]]; then
  ARM_CONFIG_NAME="${VARIANT_A_CONFIG}"
  ARM_MASTER_PORT="${ARM_A_PORT}"
  ARM_LABEL="variantA_fast_ce"
else
  ARM_CONFIG_NAME="${VARIANT_B_CONFIG}"
  ARM_MASTER_PORT="${ARM_B_PORT}"
  ARM_LABEL="variantB_query_mse"
fi
ARM_OUTPUT_ROOT="${OUTPUT_BASE}/pi05_ki_dual_${SHARED_TAG}/${ARM_LABEL}"

# -----------------------------------------------------------------------------
# RENDEZVOUS
# -----------------------------------------------------------------------------
# Arm A's master is node 0, whose address is the well-known
# ARNOLD_WORKER_0_HOST. Arm B's master is node 4 and there is NO precedent in
# this repository for a per-node ARNOLD_WORKER_<i>_HOST variable, so we cannot
# rely on one existing.
#
# Resolution order:
#   1. ARNOLD_WORKER_<master_rank>_HOST if the platform happens to export it.
#   2. Shared-filesystem handshake on the BN mount, which is already proven to
#      be shared across all nodes because every node reads and writes the same
#      checkpoint tree. The arm master publishes its own address; the other
#      nodes of that arm poll for it.
readonly RDZV_DIR="${RDZV_DIR:-${OUTPUT_BASE}/pi05_ki_dual_${SHARED_TAG}/_rendezvous}"
readonly RDZV_FILE="${RDZV_DIR}/arm_${ARM}_master_host"
readonly RDZV_TIMEOUT_S="${RDZV_TIMEOUT_S:-900}"
readonly RDZV_POLL_S="${RDZV_POLL_S:-5}"

_self_host() {
  # Resolution order matters here.
  #
  # 1. Hostname. Every working multi-node script in this repository uses
  #    ARNOLD_WORKER_0_HOST as MASTER_ADDR, and Arnold node names such as
  #    n136-080-072 resolve between peers, so a hostname is the format the
  #    rendezvous is known to accept.
  # 2. IPv4, explicitly filtered. `hostname -i` on these nodes returns the
  #    IPv6 address FIRST; handing a bare IPv6 literal to
  #    accelerate --main_process_ip breaks the c10d TCPStore because it is
  #    parsed as host:port on the colons.
  # 3. IPv6 only as a last resort, bracketed so it at least parses.
  local name ipv4 ipv6
  name="$(hostname 2>/dev/null)"
  if [[ -n "${name}" && "${name}" != "localhost" ]]; then
    printf '%s' "${name}"
    return 0
  fi

  ipv4="$(hostname -I 2>/dev/null | tr ' ' '\n' \
          | grep -E '^([0-9]{1,3}\.){3}[0-9]{1,3}$' \
          | grep -v '^127\.' | head -n1)"
  if [[ -n "${ipv4}" ]]; then
    printf '%s' "${ipv4}"
    return 0
  fi

  ipv6="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep ':' | head -n1)"
  if [[ -n "${ipv6}" ]]; then
    printf '[%s]' "${ipv6}"
    return 0
  fi
  return 1
}

resolve_arm_master() {
  local var_name="ARNOLD_WORKER_${ARM_MASTER_GLOBAL_RANK}_HOST"
  local from_env="${!var_name:-}"
  if [[ -n "${from_env}" ]]; then
    info "rendezvous: ${var_name}=${from_env} (platform-provided)"
    printf '%s' "${from_env}"
    return 0
  fi

  mkdir -p "${RDZV_DIR}" || die "cannot create rendezvous dir: ${RDZV_DIR}"

  if (( NODE_RANK_GLOBAL == ARM_MASTER_GLOBAL_RANK )); then
    local me
    me="$(_self_host)"
    [[ -n "${me}" ]] || die "cannot determine this node's own address for the rendezvous."
    # Write atomically so a peer never reads a half-written file.
    printf '%s\n' "${me}" > "${RDZV_FILE}.tmp.$$" \
      && mv -f "${RDZV_FILE}.tmp.$$" "${RDZV_FILE}" \
      || die "cannot publish rendezvous address to ${RDZV_FILE}"
    info "rendezvous: published arm ${ARM} master address ${me} -> ${RDZV_FILE}"
    printf '%s' "${me}"
    return 0
  fi

  info "rendezvous: waiting for arm ${ARM} master address at ${RDZV_FILE} (timeout ${RDZV_TIMEOUT_S}s)"
  local waited=0 addr=""
  while (( waited < RDZV_TIMEOUT_S )); do
    if [[ -s "${RDZV_FILE}" ]]; then
      addr="$(head -n1 "${RDZV_FILE}" | tr -d '[:space:]')"
      [[ -n "${addr}" ]] && break
    fi
    sleep "${RDZV_POLL_S}"
    waited=$(( waited + RDZV_POLL_S ))
  done
  [[ -n "${addr}" ]] || die \
    "rendezvous timed out after ${RDZV_TIMEOUT_S}s waiting for ${RDZV_FILE}." \
    "The arm master (global node rank ${ARM_MASTER_GLOBAL_RANK}) never published its address." \
    "Check that node ${ARM_MASTER_GLOBAL_RANK} started and that ${RDZV_DIR} is on a shared mount."
  info "rendezvous: resolved arm ${ARM} master address ${addr}"
  printf '%s' "${addr}"
}

# -----------------------------------------------------------------------------
# Config existence gate
# -----------------------------------------------------------------------------
# Fail BEFORE touching GPUs if an arm's config is not registered. Without this
# the arm would burn a rendezvous, start torchrun and only then die inside
# Python on every rank, which is far harder to read in the logs.
readonly TRAINING_PYTHON="${TRAINING_PYTHON:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python}"

assert_config_registered() {
  local cfg="$1"
  [[ -x "${TRAINING_PYTHON}" ]] || die "training python not found or not executable: ${TRAINING_PYTHON}"
  if ! PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${TRAINING_PYTHON}" - "${cfg}" <<'PYEOF'
import sys

name = sys.argv[1]
try:
    from openpi.training import config as _config
except Exception as exc:  # pragma: no cover - surfaced through the launcher log
    print(f"cannot import openpi.training.config: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    resolved = _config.get_config(name)
except Exception as exc:
    print(f"config {name!r} is not registered: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

# get_config() does NOT raise on an unknown name: it logs a warning and returns
# the default 'pi05_b1k-base'. Comparing the resolved name back is therefore the
# only reliable existence check, and it also catches a typo that would
# otherwise silently train a completely different recipe.
actual = getattr(resolved, "name", None)
if actual != name:
    print(
        f"config {name!r} is NOT registered: get_config() silently fell back to "
        f"{actual!r}. Refusing to launch.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"config {name!r} resolved")
PYEOF
  then
    if [[ "${ARM}" == "A" ]]; then
      die "arm A config '${cfg}' does not resolve." \
          "Variant A (FAST discrete action tokens + cross-entropy backbone objective) is NOT" \
          "implemented in this repository yet: no model class registers it and" \
          "'pi05_ki_joint_fast' is currently only a reserved name." \
          "" \
          "Land Variant A first, then re-run. What it needs:" \
          "  1. a public action-token helper on FASTTokenizer (currently only the private" \
          "     _act_tokens_to_paligemma_tokens maps FAST ids into the PaliGemma vocab tail)," \
          "  2. a data transform mirroring TokenizeSubtaskInputs that also emits" \
          "     action_tokens / action_token_mask / action_token_loss_mask," \
          "  3. those fields threaded through openpi.models.model.Observation," \
          "  4. a model that embeds the action tokens after the subtask segment with a" \
          "     CAUSAL ar_mask and computes cross-entropy against them instead of" \
          "     _compute_query_mse_loss," \
          "  5. the config + pytorch_model_name registration and tests." \
          "" \
          "To hold these GPUs meanwhile, run this arm with OPENPI_ARM_OCCUPY_ONLY=1."
    fi
    die "arm ${ARM} config '${cfg}' does not resolve; refusing to start." \
        "Check the name against src/openpi/training/pi05_ki_joint_query_config.py."
  fi
}

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
info "============================================================"
info "pi0.5-KI dual-variant 4x8 + 4x8 entrypoint"
info "  allocation      = ${NUM_NODES_TOTAL} nodes x ${GPUS_PER_NODE} GPUs"
info "  this node       = global rank ${NODE_RANK_GLOBAL} -> arm ${ARM} (arm-local rank ${ARM_NODE_RANK})"
info "  arm A           = nodes 0-$(( NODES_PER_ARM - 1 ))   Variant A (FAST tokens + CE)"
info "  arm B           = nodes ${NODES_PER_ARM}-$(( NUM_NODES_TOTAL - 1 ))   Variant B (query MSE)"
info "  arm size        = ${NODES_PER_ARM} nodes x ${GPUS_PER_NODE} = ${ARM_TOTAL_GPUS} GPUs"
info "  config          = ${ARM_CONFIG_NAME}"
info "  batch/GPU       = ${ARM_BATCH_SIZE_PER_GPU}  -> global batch $(( ARM_BATCH_SIZE_PER_GPU * ARM_TOTAL_GPUS ))"
info "  skill bridge    = DISABLED for both arms (original skill annotations)"
info "  master port     = ${ARM_MASTER_PORT}  (arm A ${ARM_A_PORT} / arm B ${ARM_B_PORT})"
info "  output root     = ${ARM_OUTPUT_ROOT}"
info "  launcher        = ${ARM_LAUNCHER}"
info "  keepalive       = ON for success AND failure (GPU occupancy is top priority)"
info "  job id          = ${JOB_ID}"
info "============================================================"

[[ -f "${ARM_LAUNCHER}" ]]     || die "arm launcher not found: ${ARM_LAUNCHER}"
[[ -f "${KEEPALIVE_WRAPPER}" ]] || die "keepalive wrapper not found: ${KEEPALIVE_WRAPPER}"

# -----------------------------------------------------------------------------
# Occupy-only escape hatch
# -----------------------------------------------------------------------------
# Lets one arm hold its 32 GPUs with matmul occupiers while the other arm
# trains. This is the intended way to reserve capacity for arm A until
# Variant A lands.
if [[ "${OPENPI_ARM_OCCUPY_ONLY:-0}" == "1" ]]; then
  info "OPENPI_ARM_OCCUPY_ONLY=1 -- arm ${ARM} will hold its GPUs without training."
  export KEEPALIVE_DISABLE=0
  export KEEPALIVE_ON_SUCCESS=1
  export STRICT_GPU_COUNT=0
  export NODE_RANK="${ARM_NODE_RANK}"
  export NUM_NODES="${NODES_PER_ARM}"
  export PERSISTENT_OUTPUT_ROOT="${ARM_OUTPUT_ROOT}"
  export OCCUPY_RUNTIME_DIR="/tmp/pi05_ki_dual_occupy_arm${ARM}"
  # `true` exits 0 immediately; with KEEPALIVE_ON_SUCCESS=1 the wrapper then
  # starts the occupiers and holds, which is exactly the desired behaviour.
  export TRAIN_COMMAND="true"
  info "exec'ing keepalive wrapper in occupy-only mode: ${KEEPALIVE_WRAPPER}"
  exec bash "${KEEPALIVE_WRAPPER}"
fi

# -----------------------------------------------------------------------------
# Validate the config, then resolve the rendezvous
# -----------------------------------------------------------------------------
assert_config_registered "${ARM_CONFIG_NAME}"

if [[ "${OPENPI_DUAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  info "OPENPI_DUAL_PREFLIGHT_ONLY=1 -- topology and config validated, exiting"
  info "without resolving the rendezvous, touching GPUs or starting occupiers."
  exit 0
fi

ARM_MASTER_ADDR="$(resolve_arm_master)"
[[ -n "${ARM_MASTER_ADDR}" ]] || die "failed to resolve the arm ${ARM} master address."

# -----------------------------------------------------------------------------
# Hand over to the keepalive wrapper
# -----------------------------------------------------------------------------
# The arm-scoped values below deliberately SHADOW the ARNOLD_* globals that the
# launcher and wrapper would otherwise read. Both of them resolve topology as
# "${NUM_NODES:-${ARNOLD_WORKER_NUM:-...}}" and
# "${NODE_RANK:-${ARNOLD_ID:-...}}", so exporting these makes each arm see a
# clean 4-node world with its own rank space and its own master, while the
# platform still sees a single 8-node job.
export NUM_NODES="${NODES_PER_ARM}"
export NNODES="${NODES_PER_ARM}"
export NODE_RANK="${ARM_NODE_RANK}"
export GPUS_PER_NODE="${GPUS_PER_NODE}"
export NPROC_PER_NODE="${GPUS_PER_NODE}"
export MASTER_ADDR="${ARM_MASTER_ADDR}"
export MASTER_PORT="${ARM_MASTER_PORT}"

export CONFIG_NAME="${ARM_CONFIG_NAME}"
export BATCH_SIZE_PER_GPU="${ARM_BATCH_SIZE_PER_GPU}"

# Separate output trees so the two arms cannot overwrite each other's
# checkpoints, accelerate_state or metrics.
export PERSISTENT_OUTPUT_ROOT="${ARM_OUTPUT_ROOT}"
export OUTPUT_ROOT="${ARM_OUTPUT_ROOT}"
export KEEPALIVE_STATE_DIR="${ARM_OUTPUT_ROOT}/keepalive"
export OCCUPY_RUNTIME_DIR="/tmp/pi05_ki_dual_occupy_arm${ARM}"

# Distinct Tracking / wandb identity per arm, otherwise both arms would stream
# into one run and the comparison would be meaningless.
export EXP_NAME="${EXP_NAME:-pi05_ki_dual_${ARM_LABEL}_4n8g_${SHARED_TAG}}"
export WANDB_MODE="${WANDB_MODE:-online}"

# GPU occupancy outranks training liveness: hold on success AND failure, and
# record a GPU-count mismatch instead of exiting on it.
export KEEPALIVE_DISABLE="${KEEPALIVE_DISABLE:-0}"
export KEEPALIVE_ON_SUCCESS="${KEEPALIVE_ON_SUCCESS:-1}"
export STRICT_GPU_COUNT="${STRICT_GPU_COUNT:-0}"
export EXPECTED_GPUS_PER_NODE="${GPUS_PER_NODE}"
export LAUNCHER="${ARM_LAUNCHER}"

info "arm ${ARM} handing over to the keepalive wrapper:"
info "  NUM_NODES=${NUM_NODES} NODE_RANK=${NODE_RANK} GPUS_PER_NODE=${GPUS_PER_NODE}"
info "  RENDEZVOUS=${MASTER_ADDR}:${MASTER_PORT}"
info "  CONFIG_NAME=${CONFIG_NAME} BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU}"
info "  EXP_NAME=${EXP_NAME}"
info "  PERSISTENT_OUTPUT_ROOT=${PERSISTENT_OUTPUT_ROOT}"

# `exec` keeps a single foreground process for Merlin to observe and prevents
# this script's shell options from leaking into the wrapper. The wrapper owns
# failure handling from here on and must never fail fast.
exec bash "${KEEPALIVE_WRAPPER}"
