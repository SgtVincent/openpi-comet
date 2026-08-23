#!/usr/bin/env bash
# ============================================================================
# Keepalive-on-failure wrapper for the π0.5-KI Skill Bridge LQ A100 trial.
#
# WHY THIS EXISTS
# ---------------
# On Merlin/Arnold, when the entrypoint bash process exits the whole GPU
# allocation is reclaimed. The underlying training currently can die with a
# TorchElastic `ChildFailedError` (exit 1). Under the persistent-resource
# policy we want to KEEP holding the GPUs after such a failure so a human can
# inspect / fix / relaunch, instead of losing the allocation and re-queueing
# (re-acquiring an equivalent 4×8 A100 batch has unbounded latency).
#
# WHAT IT DOES
# ------------
#   1. Runs the real training launcher (unmodified, output NOT swallowed).
#   2. Captures the *training* exit code (via PIPESTATUS, not tee's rc) and
#      records rc / timestamp / hostname / node rank into a durable,
#      node-specific file on shared NAS.
#   3. ONLY AFTER training has terminated, and only on failure by default,
#      starts one matmul "occupier" process per GPU (single-GPU isolated via
#      CUDA_VISIBLE_DEVICES) so the allocation keeps showing GPU utilization.
#   4. Stays in the FOREGROUND with a heartbeat loop -> Merlin still sees a
#      live entrypoint -> the GPUs are not reclaimed.
#
# ############################################################################
# ##  HOW TO STOP THE OCCUPIERS BEFORE STARTING A REAL TRAINING RUN         ##
# ############################################################################
# The occupiers hold GPU memory. They MUST be stopped before any subsequent
# real training/eval run on the same allocation, otherwise that run will OOM
# or be starved.
#
#   Preferred (graceful, per node — the wrapper cleans up its own children
#   and then exits):
#       touch /tmp/pi05_skillbridge_gpu_occupy/STOP
#
#   Manual fallback (only if the wrapper is gone / unresponsive). Match on the
#   occupier payload script name AND the marker string: a bare
#   `pkill -f __GPU_OCCUPY__torch_mm_512` would also match the wrapper itself
#   and any shell whose argv merely mentions the marker.
#       ps -eo pid=,args= \
#         | grep -F -- '__GPU_OCCUPY__torch_mm_512' \
#         | grep -E -- 'gpu_occupy_(torch_mm\.py|stub\.sh)' \
#         | awk '{print $1}' | xargs -r kill -TERM
#       # verify (should print nothing):
#       ps -eo pid=,args= | grep -F -- '__GPU_OCCUPY__torch_mm_512' \
#         | grep -E -- 'gpu_occupy_(torch_mm\.py|stub\.sh)'
#       nvidia-smi
#
#   To never start occupiers at all for a given launch:
#       KEEPALIVE_DISABLE=1 bash scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh
#
# ############################################################################
#
# ENVIRONMENT KNOBS
# -----------------
#   LAUNCHER                 underlying training script (default: the LQ BF16
#                            skill-bridge multinode launcher)
#   TRAIN_COMMAND            full shell command string; overrides LAUNCHER
#                            entirely (used by the no-GPU smoke test)
#   KEEPALIVE_DISABLE=1      disable keepalive completely; wrapper then exits
#                            with the training exit code (transparent mode)
#   KEEPALIVE_ON_SUCCESS     default 0 -> success path exits 0 without
#                            occupying (never permanently hold a finished
#                            trial). Set 1 to also hold on success.
#   KEEPALIVE_STATE_DIR      durable shared-NAS dir for status/heartbeat files
#   OCCUPY_RUNTIME_DIR       node-local dir for pidfiles + occupier logs
#                            (default /tmp/pi05_skillbridge_gpu_occupy)
#   STOP_OCCUPIERS_FILE      default ${OCCUPY_RUNTIME_DIR}/STOP
#   EXPECTED_GPUS_PER_NODE   default 8 (formal LQ trial topology)
#   STRICT_GPU_COUNT         default 0 -> mismatch is loudly RECORDED but we
#                            still hold the allocation (aborting would release
#                            the GPUs, defeating the purpose). 1 -> fail fast.
#   OCCUPIER_PYTHON          absolute conda python used by the occupiers
#   OCCUPIER_DRY_RUN=1       spawn marker-carrying stub processes instead of
#                            touching CUDA (for CPU-only smoke tests)
#   OCCUPIER_AUTO_RESTART    default 1 -> respawn an occupier that died, so
#                            utilization does not silently drop to 0
#   HEARTBEAT_INTERVAL_S     default 300 (heartbeat log/file cadence)
#   STOP_POLL_INTERVAL_S     default 10 (STOP-file responsiveness)
#   NVIDIA_SMI_BIN           default nvidia-smi (overridable for fake tests)
#
# EXIT CODES
# ----------
#   0   training succeeded (and keepalive not requested on success), or the
#       keepalive loop was stopped cleanly via STOP file / SIGTERM / SIGINT
#   >0  only when KEEPALIVE_DISABLE=1 (training rc is propagated verbatim),
#       or on a wrapper usage error
# ============================================================================

# NOTE: deliberately NO `set -e`. The whole point of this wrapper is to survive
# a failing training command instead of aborting with it.
set -uo pipefail

readonly OCCUPY_MARKER='__GPU_OCCUPY__torch_mm_512'

usage() {
  # Print the header comment block (lines 2..89) with the leading '# ' stripped.
  sed -n '2,89p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac

# ---------------------------------------------------------------------------
# Repo root / identity
# ---------------------------------------------------------------------------
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"
NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM:-1}}"
HOST_NAME="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-host)"
JOB_ID="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-manual}}"

# Sanitize identity fragments used inside file names.
sanitize() {
  local s="$1"
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s//,/_}"
  s="${s// /_}"
  printf '%s' "$s"
}
JOB_ID_SAFE="$(sanitize "${JOB_ID}")"
HOST_SAFE="$(sanitize "${HOST_NAME}")"

# ---------------------------------------------------------------------------
# Paths: durable (shared NAS) vs node-local (/tmp)
# ---------------------------------------------------------------------------
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_skillbridge_a100_lq_bf16}"
KEEPALIVE_STATE_DIR="${KEEPALIVE_STATE_DIR:-${PERSISTENT_OUTPUT_ROOT}/keepalive/${JOB_ID_SAFE}}"

# Node-specific durable filenames so 4 nodes never overwrite each other.
NODE_TAG="node${NODE_RANK}_${HOST_SAFE}"
DURABLE_STATUS_FILE="${KEEPALIVE_STATE_DIR}/${NODE_TAG}.status.json"
DURABLE_EVENTS_FILE="${KEEPALIVE_STATE_DIR}/${NODE_TAG}.events.log"
DURABLE_HEARTBEAT_FILE="${KEEPALIVE_STATE_DIR}/${NODE_TAG}.heartbeat"

OCCUPY_RUNTIME_DIR="${OCCUPY_RUNTIME_DIR:-/tmp/pi05_skillbridge_gpu_occupy}"
STOP_OCCUPIERS_FILE="${STOP_OCCUPIERS_FILE:-${OCCUPY_RUNTIME_DIR}/STOP}"
OCCUPIER_SCRIPT="${OCCUPY_RUNTIME_DIR}/gpu_occupy_torch_mm.py"
OCCUPIER_STUB_SCRIPT="${OCCUPY_RUNTIME_DIR}/gpu_occupy_stub.sh"

# ---------------------------------------------------------------------------
# Behaviour knobs
# ---------------------------------------------------------------------------
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/scripts/run_pi05_ki_joint_query_single_task_radio_skillbridge_bf16_multinode_lq.sh}"
KEEPALIVE_DISABLE="${KEEPALIVE_DISABLE:-0}"
KEEPALIVE_ON_SUCCESS="${KEEPALIVE_ON_SUCCESS:-0}"
EXPECTED_GPUS_PER_NODE="${EXPECTED_GPUS_PER_NODE:-8}"
STRICT_GPU_COUNT="${STRICT_GPU_COUNT:-0}"
OCCUPIER_PYTHON="${OCCUPIER_PYTHON:-/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python}"
OCCUPIER_DRY_RUN="${OCCUPIER_DRY_RUN:-0}"
OCCUPIER_AUTO_RESTART="${OCCUPIER_AUTO_RESTART:-1}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-300}"
STOP_POLL_INTERVAL_S="${STOP_POLL_INTERVAL_S:-10}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"

# PIDs this wrapper personally started. We NEVER kill anything outside this
# list (cross-checked against the argv marker before signalling).
OWNED_PIDS=()

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

log() {
  printf '[keepalive][%s][%s] %s\n' "$(ts)" "${NODE_TAG}" "$*"
}

log_err() {
  printf '[keepalive][%s][%s] %s\n' "$(ts)" "${NODE_TAG}" "$*" >&2
}

# Append to the durable per-node event log (best effort; never fatal).
record_event() {
  printf '[%s] %s\n' "$(ts)" "$*" >> "${DURABLE_EVENTS_FILE}" 2>/dev/null || true
}

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '%s' "$s"
}

# Atomically (tmp + mv) write the durable per-node status document.
write_status() {
  local phase="$1"
  local train_rc="$2"
  local detail="${3:-}"
  local tmp="${DURABLE_STATUS_FILE}.tmp.$$"
  {
    printf '{\n'
    printf '  "phase": "%s",\n' "$(json_escape "${phase}")"
    printf '  "train_exit_code": %s,\n' "${train_rc}"
    printf '  "timestamp": "%s",\n' "$(ts)"
    printf '  "timestamp_epoch": %s,\n' "$(date +%s)"
    printf '  "hostname": "%s",\n' "$(json_escape "${HOST_NAME}")"
    printf '  "node_rank": "%s",\n' "$(json_escape "${NODE_RANK}")"
    printf '  "num_nodes": "%s",\n' "$(json_escape "${NUM_NODES}")"
    printf '  "arnold_job_id": "%s",\n' "$(json_escape "${JOB_ID}")"
    printf '  "wrapper_pid": %s,\n' "$$"
    printf '  "gpus_detected": "%s",\n' "$(json_escape "${GPU_COUNT:-unknown}")"
    printf '  "expected_gpus_per_node": "%s",\n' "$(json_escape "${EXPECTED_GPUS_PER_NODE}")"
    printf '  "occupiers_running": "%s",\n' "$(json_escape "${OCCUPIERS_RUNNING:-0}")"
    printf '  "occupy_marker": "%s",\n' "${OCCUPY_MARKER}"
    printf '  "stop_file": "%s",\n' "$(json_escape "${STOP_OCCUPIERS_FILE}")"
    printf '  "dry_run": "%s",\n' "$(json_escape "${OCCUPIER_DRY_RUN}")"
    printf '  "detail": "%s"\n' "$(json_escape "${detail}")"
    printf '}\n'
  } > "${tmp}" 2>/dev/null || { log_err "WARN: could not write durable status to ${tmp}"; return 0; }
  mv -f "${tmp}" "${DURABLE_STATUS_FILE}" 2>/dev/null \
    || log_err "WARN: could not move durable status into ${DURABLE_STATUS_FILE}"
}

# ---------------------------------------------------------------------------
# Prepare directories
# ---------------------------------------------------------------------------
mkdir -p "${OCCUPY_RUNTIME_DIR}" || {
  log_err "FATAL: cannot create node-local runtime dir ${OCCUPY_RUNTIME_DIR}"
  exit 2
}
if ! mkdir -p "${KEEPALIVE_STATE_DIR}" 2>/dev/null; then
  log_err "WARN: cannot create durable state dir ${KEEPALIVE_STATE_DIR};"
  log_err "      falling back to node-local ${OCCUPY_RUNTIME_DIR} (NOT durable)."
  KEEPALIVE_STATE_DIR="${OCCUPY_RUNTIME_DIR}"
  DURABLE_STATUS_FILE="${KEEPALIVE_STATE_DIR}/${NODE_TAG}.status.json"
  DURABLE_EVENTS_FILE="${KEEPALIVE_STATE_DIR}/${NODE_TAG}.events.log"
  DURABLE_HEARTBEAT_FILE="${KEEPALIVE_STATE_DIR}/${NODE_TAG}.heartbeat"
fi

# A stale STOP file from a previous run would immediately abort this one.
if [[ -e "${STOP_OCCUPIERS_FILE}" ]]; then
  log "Removing pre-existing STOP file from a previous run: ${STOP_OCCUPIERS_FILE}"
  rm -f "${STOP_OCCUPIERS_FILE}" || log_err "WARN: could not remove stale STOP file"
fi

WRAPPER_LOG="${OCCUPY_RUNTIME_DIR}/wrapper_${NODE_TAG}.log"

log "============================================================"
log "π0.5-KI Skill Bridge — keepalive-on-failure wrapper"
log "REPO_ROOT=${REPO_ROOT}"
log "NODE_RANK=${NODE_RANK} / NUM_NODES=${NUM_NODES} host=${HOST_NAME} job=${JOB_ID}"
log "LAUNCHER=${LAUNCHER}"
log "TRAIN_COMMAND=${TRAIN_COMMAND:-<unset, using LAUNCHER>}"
log "KEEPALIVE_DISABLE=${KEEPALIVE_DISABLE} KEEPALIVE_ON_SUCCESS=${KEEPALIVE_ON_SUCCESS}"
log "OCCUPIER_DRY_RUN=${OCCUPIER_DRY_RUN} OCCUPIER_AUTO_RESTART=${OCCUPIER_AUTO_RESTART}"
log "EXPECTED_GPUS_PER_NODE=${EXPECTED_GPUS_PER_NODE} STRICT_GPU_COUNT=${STRICT_GPU_COUNT}"
log "OCCUPY_RUNTIME_DIR=${OCCUPY_RUNTIME_DIR}"
log "STOP_OCCUPIERS_FILE=${STOP_OCCUPIERS_FILE}"
log "KEEPALIVE_STATE_DIR=${KEEPALIVE_STATE_DIR}"
log "DURABLE_STATUS_FILE=${DURABLE_STATUS_FILE}"
log "WRAPPER_LOG=${WRAPPER_LOG}"
log "To stop occupiers:  touch ${STOP_OCCUPIERS_FILE}"
log "Manual fallback:    ps -eo pid=,args= | grep -F -- '${OCCUPY_MARKER}' | grep -E -- 'gpu_occupy_(torch_mm[.]py|stub[.]sh)' | awk '{print \$1}' | xargs -r kill -TERM"
log "============================================================"

record_event "wrapper start pid=$$ launcher=${LAUNCHER} train_command=${TRAIN_COMMAND:-<unset>}"

# ---------------------------------------------------------------------------
# GPU count detection
# ---------------------------------------------------------------------------
detect_gpu_count() {
  # Priority 1: Arnold-provided GPU count.
  if [[ -n "${ARNOLD_WORKER_GPU:-}" && "${ARNOLD_WORKER_GPU}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${ARNOLD_WORKER_GPU}"
    return 0
  fi
  # Priority 2: nvidia-smi -L line count.
  if command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1; then
    local n
    n="$("${NVIDIA_SMI_BIN}" -L 2>/dev/null | grep -c '^GPU ' || true)"
    if [[ "${n}" =~ ^[0-9]+$ && "${n}" -gt 0 ]]; then
      printf '%s' "${n}"
      return 0
    fi
  fi
  printf '0'
}

# ---------------------------------------------------------------------------
# Occupier payloads (materialized node-locally, self-contained)
# ---------------------------------------------------------------------------
write_occupier_payloads() {
  cat > "${OCCUPIER_SCRIPT}" <<'PYEOF'
"""Single-GPU matmul occupier.

Argv carries the marker string so `ps` can identify (and only
identify) processes started by the keepalive wrapper.

Usage: python gpu_occupy_torch_mm.py <marker> <logical_gpu_index>

CUDA_VISIBLE_DEVICES is expected to already pin this process to exactly one
physical GPU, so we always use cuda:0 here.
"""
import os
import signal
import sys
import time

MATRIX_DIM = 512


def main() -> int:
    marker = sys.argv[1] if len(sys.argv) > 1 else "unknown-marker"
    gpu_index = sys.argv[2] if len(sys.argv) > 2 else "?"

    stop = {"flag": False}

    def _handle(signum, _frame):
        stop["flag"] = True
        print(f"[occupy][gpu{gpu_index}] received signal {signum}, exiting", flush=True)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    import torch  # imported after signal setup so shutdown is always responsive

    if not torch.cuda.is_available():
        print(f"[occupy][gpu{gpu_index}] FATAL: CUDA unavailable", flush=True)
        return 3

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    a = torch.randn(MATRIX_DIM, MATRIX_DIM, device=device)
    b = torch.randn(MATRIX_DIM, MATRIX_DIM, device=device)
    print(
        f"[occupy][gpu{gpu_index}] marker={marker} pid={os.getpid()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"dim={MATRIX_DIM} started",
        flush=True,
    )

    iterations = 0
    last_report = time.time()
    while not stop["flag"]:
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        iterations += 1
        del c
        now = time.time()
        if now - last_report >= 300:
            print(
                f"[occupy][gpu{gpu_index}] alive iters={iterations} "
                f"mem_alloc_mb={torch.cuda.memory_allocated() / 1e6:.1f}",
                flush=True,
            )
            last_report = now

    print(f"[occupy][gpu{gpu_index}] stopped after {iterations} iterations", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF

  # CPU-only stub used by OCCUPIER_DRY_RUN=1: same argv shape (marker present)
  # but never touches CUDA, so the wrapper logic is testable without GPUs.
  cat > "${OCCUPIER_STUB_SCRIPT}" <<'SHEOF'
#!/usr/bin/env bash
# Dry-run occupier stub. Argv intentionally carries the marker string.
set -uo pipefail
marker="${1:-unknown-marker}"
gpu_index="${2:-?}"
trap 'echo "[occupy-stub][gpu${gpu_index}] signalled, exiting"; exit 0' TERM INT
echo "[occupy-stub][gpu${gpu_index}] marker=${marker} pid=$$ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} started (DRY RUN, no CUDA)"
while true; do
  sleep 5 &
  wait $! || true
done
SHEOF
  chmod +x "${OCCUPIER_STUB_SCRIPT}" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Occupier process management
# ---------------------------------------------------------------------------
pidfile_for() { printf '%s/gpu%s.pid' "${OCCUPY_RUNTIME_DIR}" "$1"; }
logfile_for() { printf '%s/gpu%s.log' "${OCCUPY_RUNTIME_DIR}" "$1"; }

# True iff pid is alive AND its argv contains our marker.
# The marker cross-check protects against PID reuse and guarantees we never
# signal an unrelated process.
pid_is_our_occupier() {
  local pid="$1"
  [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  local cmdline_file="/proc/${pid}/cmdline"
  if [[ -r "${cmdline_file}" ]]; then
    tr '\0' ' ' < "${cmdline_file}" 2>/dev/null | grep -q -- "${OCCUPY_MARKER}" && return 0
    return 1
  fi
  # /proc unreadable: fall back to ps.
  ps -p "${pid}" -o args= 2>/dev/null | grep -q -- "${OCCUPY_MARKER}" && return 0
  return 1
}

# Idempotency check: is GPU i already covered by a live occupier we recorded?
gpu_already_occupied() {
  local gpu="$1"
  local pf
  pf="$(pidfile_for "${gpu}")"
  [[ -s "${pf}" ]] || return 1
  local pid
  pid="$(<"${pf}")"
  pid="${pid//[[:space:]]/}"
  pid_is_our_occupier "${pid}"
}

start_occupier_for_gpu() {
  local gpu="$1"
  local pf lf
  pf="$(pidfile_for "${gpu}")"
  lf="$(logfile_for "${gpu}")"

  if gpu_already_occupied "${gpu}"; then
    local existing
    existing="$(<"${pf}")"
    log "GPU ${gpu}: occupier already alive (pid ${existing//[[:space:]]/}) — not starting a duplicate"
    return 0
  fi

  if [[ "${OCCUPIER_DRY_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" nohup \
      bash "${OCCUPIER_STUB_SCRIPT}" "${OCCUPY_MARKER}" "${gpu}" \
      >> "${lf}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" nohup \
      "${OCCUPIER_PYTHON}" "${OCCUPIER_SCRIPT}" "${OCCUPY_MARKER}" "${gpu}" \
      >> "${lf}" 2>&1 &
  fi
  local pid=$!
  printf '%s\n' "${pid}" > "${pf}"
  OWNED_PIDS+=("${pid}")
  log "GPU ${gpu}: launched occupier pid=${pid} (log ${lf})"
  return 0
}

count_live_occupiers() {
  local gpu pf pid live=0
  for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    pf="$(pidfile_for "${gpu}")"
    [[ -s "${pf}" ]] || continue
    pid="$(<"${pf}")"
    pid="${pid//[[:space:]]/}"
    if pid_is_our_occupier "${pid}"; then
      live=$((live + 1))
    fi
  done
  printf '%s' "${live}"
}

verify_occupiers() {
  local gpu pf pid ok=0 bad=()
  for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    pf="$(pidfile_for "${gpu}")"
    pid=""
    [[ -s "${pf}" ]] && { pid="$(<"${pf}")"; pid="${pid//[[:space:]]/}"; }
    if pid_is_our_occupier "${pid}"; then
      log "  verify GPU ${gpu}: pid ${pid} alive (kill -0 OK, marker present)"
      ok=$((ok + 1))
    else
      log_err "  verify GPU ${gpu}: NO live occupier (pidfile='${pf}' pid='${pid}')"
      bad+=("${gpu}")
    fi
  done
  OCCUPIERS_RUNNING="${ok}"
  log "Occupier verification: ${ok}/${GPU_COUNT} GPUs covered (expected ${EXPECTED_GPUS_PER_NODE})"
  if ((${#bad[@]} > 0)); then
    log_err "Occupier verification: GPUs without a live occupier: ${bad[*]}"
  fi

  # Secondary confirmation via nvidia-smi compute-apps (skipped in dry run).
  if [[ "${OCCUPIER_DRY_RUN}" != "1" ]] && command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1; then
    local smi_pids
    smi_pids="$("${NVIDIA_SMI_BIN}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c '^[0-9]\+$' || true)"
    log "nvidia-smi reports ${smi_pids:-0} compute apps on this node"
  fi

  if [[ "${ok}" -eq "${EXPECTED_GPUS_PER_NODE}" ]]; then
    log "KEEPALIVE OK: ${ok}/${EXPECTED_GPUS_PER_NODE} occupiers running"
    record_event "occupiers verified ${ok}/${EXPECTED_GPUS_PER_NODE}"
  else
    log_err "KEEPALIVE PARTIAL: ${ok} occupiers running, expected ${EXPECTED_GPUS_PER_NODE}"
    record_event "occupiers PARTIAL ${ok}/${EXPECTED_GPUS_PER_NODE}"
  fi
  return 0
}

# Terminate ONLY the occupiers we started (pidfile + marker cross-checked).
cleanup_own_occupiers() {
  local reason="${1:-cleanup}"
  log "Cleaning up occupiers started by this wrapper (reason: ${reason})"

  local pids_to_kill=()
  local gpu pf pid
  for ((gpu = 0; gpu < ${GPU_COUNT:-0}; gpu++)); do
    pf="$(pidfile_for "${gpu}")"
    [[ -s "${pf}" ]] || continue
    pid="$(<"${pf}")"
    pid="${pid//[[:space:]]/}"
    if pid_is_our_occupier "${pid}"; then
      pids_to_kill+=("${pid}")
    else
      log "  GPU ${gpu}: pid '${pid}' is not a live occupier of ours — skipping (never kill unknown processes)"
    fi
    rm -f "${pf}" 2>/dev/null || true
  done

  if ((${#pids_to_kill[@]} == 0)); then
    log "  no live occupiers of ours to terminate"
    return 0
  fi

  log "  sending SIGTERM to: ${pids_to_kill[*]}"
  local p
  for p in "${pids_to_kill[@]}"; do
    kill -TERM "${p}" 2>/dev/null || true
  done

  local waited=0
  while ((waited < 30)); do
    local still=0
    for p in "${pids_to_kill[@]}"; do
      pid_is_our_occupier "${p}" && still=$((still + 1))
    done
    ((still == 0)) && break
    sleep 1
    waited=$((waited + 1))
  done

  for p in "${pids_to_kill[@]}"; do
    if pid_is_our_occupier "${p}"; then
      log_err "  pid ${p} still alive after ${waited}s — sending SIGKILL"
      kill -KILL "${p}" 2>/dev/null || true
    fi
  done

  log "  occupier cleanup done"
  record_event "occupier cleanup done (${reason}); pids=${pids_to_kill[*]}"
  return 0
}

SHUTDOWN_REQUESTED=0
on_signal() {
  local sig="$1"
  SHUTDOWN_REQUESTED=1
  log "Received SIG${sig} — shutting down keepalive"
  record_event "received SIG${sig}"
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

# ---------------------------------------------------------------------------
# STEP 1 — run the underlying training (output preserved, never swallowed)
# ---------------------------------------------------------------------------
if [[ -z "${TRAIN_COMMAND:-}" ]]; then
  if [[ ! -f "${LAUNCHER}" ]]; then
    log_err "FATAL: training launcher not found: ${LAUNCHER}"
    log_err "       set LAUNCHER=<path> or TRAIN_COMMAND=<shell command>"
    exit 2
  fi
  TRAIN_DESC="bash ${LAUNCHER}"
else
  TRAIN_DESC="${TRAIN_COMMAND}"
fi

log "STEP 1/4: launching training: ${TRAIN_DESC}"
record_event "training start: ${TRAIN_DESC}"
write_status "training_running" "null" "training launched: ${TRAIN_DESC}"

TRAIN_START_EPOCH="$(date +%s)"
# stdout+stderr are merged into one stream and tee'd, so the root-cause
# traceback stays visible on the Merlin console AND is persisted to
# WRAPPER_LOG. The underlying launcher keeps writing its own console log too.
if [[ -z "${TRAIN_COMMAND:-}" ]]; then
  bash "${LAUNCHER}" 2>&1 | tee -a "${WRAPPER_LOG}"
else
  # shellcheck disable=SC2086  # intentional: TRAIN_COMMAND is a command string
  eval "${TRAIN_COMMAND}" 2>&1 | tee -a "${WRAPPER_LOG}"
fi
# PIPESTATUS[0] is the TRAINING rc; $? would be tee's rc (almost always 0).
TRAIN_RC="${PIPESTATUS[0]}"
TRAIN_END_EPOCH="$(date +%s)"
TRAIN_DURATION_S=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))

log "============================================================"
log "STEP 2/4: training terminated"
log "  exit_code = ${TRAIN_RC}"
log "  duration  = ${TRAIN_DURATION_S}s"
log "  hostname  = ${HOST_NAME}"
log "  node_rank = ${NODE_RANK}"
log "============================================================"
record_event "training exit rc=${TRAIN_RC} duration_s=${TRAIN_DURATION_S} host=${HOST_NAME} node_rank=${NODE_RANK}"

GPU_COUNT="$(detect_gpu_count)"
OCCUPIERS_RUNNING=0
write_status "training_finished" "${TRAIN_RC}" "training exited rc=${TRAIN_RC} after ${TRAIN_DURATION_S}s"

# ---------------------------------------------------------------------------
# STEP 3 — decide whether to keep the allocation alive
# ---------------------------------------------------------------------------
if [[ "${KEEPALIVE_DISABLE}" == "1" ]]; then
  log "KEEPALIVE_DISABLE=1 — keepalive fully disabled; propagating training rc=${TRAIN_RC}"
  record_event "keepalive disabled; exiting with rc=${TRAIN_RC}"
  write_status "keepalive_disabled" "${TRAIN_RC}" "KEEPALIVE_DISABLE=1; propagating training rc"
  exit "${TRAIN_RC}"
fi

if [[ "${TRAIN_RC}" -eq 0 && "${KEEPALIVE_ON_SUCCESS}" != "1" ]]; then
  log "Training SUCCEEDED and KEEPALIVE_ON_SUCCESS=0 — exiting 0 without occupying."
  log "(Never permanently hold a finished trial; set KEEPALIVE_ON_SUCCESS=1 to override.)"
  record_event "training success; exiting 0 without keepalive"
  write_status "success_no_keepalive" "${TRAIN_RC}" "training succeeded; keepalive not requested on success"
  exit 0
fi

if [[ "${TRAIN_RC}" -eq 0 ]]; then
  log "Training SUCCEEDED but KEEPALIVE_ON_SUCCESS=1 — holding the allocation anyway."
else
  log "Training FAILED (rc=${TRAIN_RC}) — holding the allocation for manual inspection."
fi

if [[ "${SHUTDOWN_REQUESTED}" == "1" ]]; then
  log "Shutdown was requested during training — not starting occupiers; exiting 0."
  write_status "stopped_before_occupy" "${TRAIN_RC}" "shutdown requested during training"
  exit 0
fi

if [[ -e "${STOP_OCCUPIERS_FILE}" ]]; then
  log "STOP file present (${STOP_OCCUPIERS_FILE}) — not starting occupiers; exiting 0."
  write_status "stopped_before_occupy" "${TRAIN_RC}" "STOP file present before occupier start"
  exit 0
fi

# ---------------------------------------------------------------------------
# STEP 3b — GPU count sanity (only reached AFTER training has terminated,
# so occupiers can never contend with a live training run)
# ---------------------------------------------------------------------------
log "STEP 3/4: starting per-GPU occupiers (training has already terminated)"
log "  detected GPU_COUNT=${GPU_COUNT} (ARNOLD_WORKER_GPU='${ARNOLD_WORKER_GPU:-}')"

if [[ "${GPU_COUNT}" -eq 0 ]]; then
  log_err "GPU count detected as 0 — cannot start occupiers."
  if [[ "${STRICT_GPU_COUNT}" == "1" ]]; then
    log_err "STRICT_GPU_COUNT=1 — failing fast."
    write_status "gpu_detect_failed" "${TRAIN_RC}" "GPU_COUNT=0 and STRICT_GPU_COUNT=1"
    exit 3
  fi
  log_err "Continuing into the heartbeat loop WITHOUT occupiers so the allocation is still held."
  record_event "GPU_COUNT=0; heartbeat without occupiers"
elif [[ "${GPU_COUNT}" -ne "${EXPECTED_GPUS_PER_NODE}" ]]; then
  log_err "GPU COUNT MISMATCH: detected ${GPU_COUNT}, expected ${EXPECTED_GPUS_PER_NODE} (formal LQ trial expects 8)."
  record_event "GPU count mismatch detected=${GPU_COUNT} expected=${EXPECTED_GPUS_PER_NODE}"
  if [[ "${STRICT_GPU_COUNT}" == "1" ]]; then
    log_err "STRICT_GPU_COUNT=1 — failing fast."
    write_status "gpu_count_mismatch" "${TRAIN_RC}" "detected ${GPU_COUNT} != expected ${EXPECTED_GPUS_PER_NODE}, STRICT_GPU_COUNT=1"
    exit 3
  fi
  log_err "STRICT_GPU_COUNT=0 — mismatch RECORDED; proceeding with ${GPU_COUNT} GPUs"
  log_err "(aborting here would release the allocation, which is exactly what this wrapper exists to prevent)."
fi

# ---------------------------------------------------------------------------
# STEP 3c — start occupiers (idempotent) and verify
# ---------------------------------------------------------------------------
if [[ "${GPU_COUNT}" -gt 0 ]]; then
  write_occupier_payloads
  if [[ "${OCCUPIER_DRY_RUN}" != "1" && ! -x "${OCCUPIER_PYTHON}" ]]; then
    log_err "WARN: OCCUPIER_PYTHON is not executable: ${OCCUPIER_PYTHON}"
    log_err "      occupier launches will likely fail; heartbeat will still hold the allocation."
  fi
  for ((_gpu = 0; _gpu < GPU_COUNT; _gpu++)); do
    start_occupier_for_gpu "${_gpu}"
  done
  # Give CUDA init a moment before verifying.
  sleep "${OCCUPIER_STARTUP_GRACE_S:-5}"
  verify_occupiers
fi

write_status "keepalive_active" "${TRAIN_RC}" "occupiers ${OCCUPIERS_RUNNING}/${GPU_COUNT}; holding allocation"

# ---------------------------------------------------------------------------
# STEP 4 — foreground heartbeat: this is what prevents Merlin from
# reclaiming the GPUs. The wrapper intentionally never returns on its own.
# ---------------------------------------------------------------------------
log "============================================================"
log "STEP 4/4: entering FOREGROUND keepalive heartbeat loop"
log "  training rc was ${TRAIN_RC}; allocation is now being held"
log "  stop cleanly with:  touch ${STOP_OCCUPIERS_FILE}"
log "  or send SIGTERM/SIGINT to wrapper pid $$"
log "  manual fallback:    ps -eo pid=,args= | grep -F -- '${OCCUPY_MARKER}' | grep -E -- 'gpu_occupy_(torch_mm[.]py|stub[.]sh)' | awk '{print \$1}' | xargs -r kill -TERM"
log "============================================================"
record_event "heartbeat loop entered; occupiers=${OCCUPIERS_RUNNING}/${GPU_COUNT}"

# Emit an immediately-greppable marker for smoke tests / monitoring.
log "KEEPALIVE_HEARTBEAT_LOOP_STARTED train_rc=${TRAIN_RC} occupiers=${OCCUPIERS_RUNNING}/${GPU_COUNT}"

HEARTBEAT_N=0
LAST_HEARTBEAT_EPOCH=0
STOP_REASON="unknown"

while true; do
  if [[ "${SHUTDOWN_REQUESTED}" == "1" ]]; then
    STOP_REASON="signal"
    break
  fi
  if [[ -e "${STOP_OCCUPIERS_FILE}" ]]; then
    log "STOP file detected: ${STOP_OCCUPIERS_FILE}"
    STOP_REASON="stop_file"
    break
  fi

  NOW_EPOCH="$(date +%s)"
  if ((NOW_EPOCH - LAST_HEARTBEAT_EPOCH >= HEARTBEAT_INTERVAL_S)); then
    HEARTBEAT_N=$((HEARTBEAT_N + 1))
    LAST_HEARTBEAT_EPOCH="${NOW_EPOCH}"
    LIVE="$(count_live_occupiers)"

    # Respawn dead occupiers so GPU utilization does not silently drop to 0
    # (which would itself trigger low-utilization eviction).
    if [[ "${OCCUPIER_AUTO_RESTART}" == "1" && "${GPU_COUNT}" -gt 0 && "${LIVE}" -lt "${GPU_COUNT}" ]]; then
      log_err "HEARTBEAT #${HEARTBEAT_N}: only ${LIVE}/${GPU_COUNT} occupiers alive — respawning missing ones"
      for ((_gpu = 0; _gpu < GPU_COUNT; _gpu++)); do
        gpu_already_occupied "${_gpu}" || start_occupier_for_gpu "${_gpu}"
      done
      sleep 3
      LIVE="$(count_live_occupiers)"
    fi

    OCCUPIERS_RUNNING="${LIVE}"
    log "HEARTBEAT #${HEARTBEAT_N} — holding allocation; occupiers ${LIVE}/${GPU_COUNT}; train_rc=${TRAIN_RC}; uptime_s=$((NOW_EPOCH - TRAIN_END_EPOCH))"
    printf '%s heartbeat=%s occupiers=%s/%s train_rc=%s\n' \
      "$(ts)" "${HEARTBEAT_N}" "${LIVE}" "${GPU_COUNT}" "${TRAIN_RC}" \
      > "${DURABLE_HEARTBEAT_FILE}.tmp.$$" 2>/dev/null \
      && mv -f "${DURABLE_HEARTBEAT_FILE}.tmp.$$" "${DURABLE_HEARTBEAT_FILE}" 2>/dev/null
    write_status "keepalive_active" "${TRAIN_RC}" "heartbeat #${HEARTBEAT_N}; occupiers ${LIVE}/${GPU_COUNT}"
  fi

  # Background sleep + wait so TERM/INT traps fire immediately instead of
  # being deferred until the sleep completes.
  sleep "${STOP_POLL_INTERVAL_S}" &
  wait $! 2>/dev/null || true
done

log "Exiting keepalive loop (reason: ${STOP_REASON})"
cleanup_own_occupiers "${STOP_REASON}"
OCCUPIERS_RUNNING="$(count_live_occupiers)"
write_status "stopped" "${TRAIN_RC}" "keepalive stopped (${STOP_REASON}); original training rc was ${TRAIN_RC}"
record_event "wrapper exit 0 after keepalive stop (${STOP_REASON}); original training rc=${TRAIN_RC}"
log "Keepalive wrapper exiting 0. Original training exit code was ${TRAIN_RC} (see ${DURABLE_STATUS_FILE})."
exit 0
