#!/usr/bin/env bash
# ============================================================================
# CPU-only smoke test for scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh
#
# Verifies, WITHOUT any GPU and WITHOUT any real training:
#   T1 failure path  : rc=7 captured via PIPESTATUS, durable node-specific
#                      status records rc/timestamp/hostname/node rank,
#                      8/8 dry-run occupiers started, heartbeat loop entered,
#                      STOP file causes a clean exit 0, no leftover processes.
#   T2 success path  : rc=0 -> wrapper returns 0 immediately, no occupiers.
#   T3 disable path  : KEEPALIVE_DISABLE=1 propagates the training rc verbatim.
#   T4 idempotency   : a second wrapper invocation does not double-start
#                      occupiers for GPUs already covered.
#
# Uses OCCUPIER_DRY_RUN=1 (marker-carrying stub, never touches CUDA) plus a
# fake `nvidia-smi` on PATH so GPU count is deterministic (8) on any host.
#
# Run:  bash tests/scripts/test_keepalive_on_failure_wrapper.sh
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh"
MARKER='__GPU_OCCUPY__torch_mm_512'

TEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/keepalive_smoke.XXXXXX")"
FAKE_BIN="${TEST_TMP}/fakebin"
mkdir -p "${FAKE_BIN}"

FAILURES=0
pass() { printf '  [PASS] %s\n' "$*"; }
fail() {
  printf '  [FAIL] %s\n' "$*" >&2
  FAILURES=$((FAILURES + 1))
}

# NOTE: a bare `pgrep -f "${MARKER}"` also matches this test harness's own
# command line (and any shell whose argv happens to mention the marker), so we
# match on the occupier *payload script name* AND the marker. This is the same
# precise pattern documented in the wrapper as the manual-kill fallback.
OCCUPIER_ARGV_RE='gpu_occupy_(stub\.sh|torch_mm\.py)'

list_occupier_pids() {
  ps -eo pid=,args= 2>/dev/null \
    | grep -F -- "${MARKER}" \
    | grep -E -- "${OCCUPIER_ARGV_RE}" \
    | awk '{print $1}'
}

count_occupiers() {
  local n
  n="$(list_occupier_pids | wc -l | tr -d ' ')"
  printf '%s' "${n:-0}"
}

# Always leave the machine clean, even if the test aborts.
cleanup() {
  local leftovers
  leftovers="$(list_occupier_pids | tr '\n' ' ')"
  if [[ -n "${leftovers// /}" ]]; then
    printf 'cleanup: killing leftover occupier pids: %s\n' "${leftovers}" >&2
    # shellcheck disable=SC2086
    kill -TERM ${leftovers} 2>/dev/null || true
    sleep 1
    leftovers="$(list_occupier_pids | tr '\n' ' ')"
    if [[ -n "${leftovers// /}" ]]; then
      # shellcheck disable=SC2086
      kill -KILL ${leftovers} 2>/dev/null || true
    fi
  fi
  rm -rf "${TEST_TMP}"
}
trap cleanup EXIT

# ---- fake nvidia-smi: 8 GPUs, no real hardware needed ----------------------
cat > "${FAKE_BIN}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  -L)
    for i in 0 1 2 3 4 5 6 7; do
      echo "GPU ${i}: FAKE-A100-SXM4-40GB (UUID: GPU-fake-${i})"
    done
    ;;
  *) : ;;
esac
exit 0
EOF
chmod +x "${FAKE_BIN}/nvidia-smi"

echo "============================================================"
echo "keepalive-on-failure wrapper — CPU-only smoke test"
echo "WRAPPER=${WRAPPER}"
echo "TEST_TMP=${TEST_TMP}"
echo "============================================================"

# Sanity: no pre-existing occupiers (otherwise results are meaningless).
if [[ "$(count_occupiers)" != "0" ]]; then
  fail "PRECONDITION: occupier processes already running before the test"
  list_occupier_pids >&2
  exit 1
fi
pass "precondition: no pre-existing occupier processes"

# ===========================================================================
# T1 — failure path (rc=7)
# ===========================================================================
echo ""
echo "--- T1: failure path, TRAIN_COMMAND exits 7 ---"
T1_RUNTIME="${TEST_TMP}/t1_runtime"
T1_STATE="${TEST_TMP}/t1_state"
T1_OUT="${TEST_TMP}/t1_wrapper_stdout.log"
mkdir -p "${T1_RUNTIME}" "${T1_STATE}"

env PATH="${FAKE_BIN}:${PATH}" \
  TRAIN_COMMAND="bash -lc 'echo FAKE_TRAIN_ROOT_CAUSE_STDOUT; echo FAKE_TRAIN_ROOT_CAUSE_STDERR >&2; exit 7'" \
  OCCUPIER_DRY_RUN=1 \
  OCCUPY_RUNTIME_DIR="${T1_RUNTIME}" \
  KEEPALIVE_STATE_DIR="${T1_STATE}" \
  EXPECTED_GPUS_PER_NODE=8 \
  HEARTBEAT_INTERVAL_S=2 \
  STOP_POLL_INTERVAL_S=1 \
  OCCUPIER_STARTUP_GRACE_S=2 \
  NODE_RANK=3 \
  ARNOLD_JOB_ID="smoketest_job" \
  bash "${WRAPPER}" > "${T1_OUT}" 2>&1 &
T1_PID=$!

# Wait for the heartbeat loop to be entered (max ~60s).
for _ in $(seq 1 60); do
  grep -q 'KEEPALIVE_HEARTBEAT_LOOP_STARTED' "${T1_OUT}" 2>/dev/null && break
  sleep 1
done

if grep -q 'KEEPALIVE_HEARTBEAT_LOOP_STARTED train_rc=7' "${T1_OUT}"; then
  pass "training rc=7 captured and heartbeat loop entered"
else
  fail "heartbeat loop not entered with train_rc=7"
  tail -40 "${T1_OUT}" >&2
fi

grep -q 'FAKE_TRAIN_ROOT_CAUSE_STDOUT' "${T1_OUT}" \
  && pass "training stdout preserved (not swallowed)" \
  || fail "training stdout missing from wrapper output"
grep -q 'FAKE_TRAIN_ROOT_CAUSE_STDERR' "${T1_OUT}" \
  && pass "training stderr preserved (not swallowed)" \
  || fail "training stderr missing from wrapper output"
grep -q 'exit_code = 7' "${T1_OUT}" \
  && pass "wrapper reported exit_code = 7 (PIPESTATUS, not tee rc)" \
  || fail "wrapper did not report exit_code = 7"

# Durable, node-specific status file.
T1_STATUS="$(find "${T1_STATE}" -name 'node3_*.status.json' 2>/dev/null | head -n1)"
if [[ -n "${T1_STATUS}" && -s "${T1_STATUS}" ]]; then
  pass "durable node-specific status file exists: $(basename "${T1_STATUS}")"
  echo "  ---- status file contents ----"
  sed 's/^/  /' "${T1_STATUS}"
  echo "  ------------------------------"
  grep -q '"train_exit_code": 7' "${T1_STATUS}" \
    && pass "status file records train_exit_code=7" \
    || fail "status file missing train_exit_code=7"
  grep -q '"timestamp":' "${T1_STATUS}" \
    && pass "status file records timestamp" \
    || fail "status file missing timestamp"
  grep -q '"hostname":' "${T1_STATUS}" \
    && pass "status file records hostname" \
    || fail "status file missing hostname"
  grep -q '"node_rank": "3"' "${T1_STATUS}" \
    && pass "status file records node_rank=3" \
    || fail "status file missing node_rank=3"
else
  fail "durable node-specific status file not found under ${T1_STATE}"
  ls -la "${T1_STATE}" >&2 || true
fi

# Occupier coverage: 8/8 marker-carrying processes.
T1_OCC="$(count_occupiers)"
if [[ "${T1_OCC}" == "8" ]]; then
  pass "8/8 dry-run occupiers running, each argv carries ${MARKER}"
else
  fail "expected 8 occupiers, found ${T1_OCC}"
  ps -eo pid=,args= | grep -F -- "${MARKER}" | grep -E -- "${OCCUPIER_ARGV_RE}" >&2 || true
fi
grep -q 'KEEPALIVE OK: 8/8 occupiers running' "${T1_OUT}" \
  && pass "wrapper self-verified 8/8 via kill -0 + marker check" \
  || fail "wrapper did not self-verify 8/8"

# One process per GPU, isolated via CUDA_VISIBLE_DEVICES.
T1_CVD_OK=1
for gpu in 0 1 2 3 4 5 6 7; do
  grep -q "CUDA_VISIBLE_DEVICES=${gpu} " "${T1_RUNTIME}/gpu${gpu}.log" 2>/dev/null || T1_CVD_OK=0
done
[[ "${T1_CVD_OK}" == "1" ]] \
  && pass "each occupier pinned to a single GPU via CUDA_VISIBLE_DEVICES" \
  || fail "CUDA_VISIBLE_DEVICES pinning not confirmed in per-GPU logs"

# Wait for at least one heartbeat tick.
for _ in $(seq 1 30); do
  grep -q 'HEARTBEAT #1' "${T1_OUT}" 2>/dev/null && break
  sleep 1
done
grep -q 'HEARTBEAT #1 — holding allocation' "${T1_OUT}" \
  && pass "foreground heartbeat tick observed" \
  || fail "no heartbeat tick observed"

# T4 (piggybacked): idempotency — a second invocation must not duplicate.
echo ""
echo "--- T4: idempotency, second invocation must not double-start ---"
T4_OUT="${TEST_TMP}/t4_wrapper_stdout.log"
env PATH="${FAKE_BIN}:${PATH}" \
  TRAIN_COMMAND="bash -lc 'exit 9'" \
  OCCUPIER_DRY_RUN=1 \
  OCCUPY_RUNTIME_DIR="${T1_RUNTIME}" \
  KEEPALIVE_STATE_DIR="${T1_STATE}" \
  EXPECTED_GPUS_PER_NODE=8 \
  HEARTBEAT_INTERVAL_S=2 \
  STOP_POLL_INTERVAL_S=1 \
  OCCUPIER_STARTUP_GRACE_S=2 \
  NODE_RANK=4 \
  ARNOLD_JOB_ID="smoketest_job" \
  bash "${WRAPPER}" > "${T4_OUT}" 2>&1 &
T4_PID=$!
for _ in $(seq 1 60); do
  grep -q 'KEEPALIVE_HEARTBEAT_LOOP_STARTED' "${T4_OUT}" 2>/dev/null && break
  sleep 1
done
T4_OCC="$(count_occupiers)"
if [[ "${T4_OCC}" == "8" ]]; then
  pass "still 8 occupiers after second invocation (idempotent, no duplicates)"
else
  fail "expected 8 occupiers after second invocation, found ${T4_OCC}"
fi
grep -qc 'occupier already alive' "${T4_OUT}" >/dev/null 2>&1 \
  && pass "second invocation logged 'occupier already alive' skips" \
  || fail "second invocation did not report idempotent skips"

# STOP file: both wrappers share OCCUPY_RUNTIME_DIR, so one touch stops both.
echo ""
echo "--- T1/T4: STOP file shutdown ---"
touch "${T1_RUNTIME}/STOP"
wait "${T4_PID}"; T4_RC=$?
wait "${T1_PID}"; T1_RC=$?

[[ "${T1_RC}" == "0" ]] \
  && pass "T1 wrapper exited 0 after STOP file (training rc=7 not propagated)" \
  || fail "T1 wrapper exited ${T1_RC}, expected 0"
[[ "${T4_RC}" == "0" ]] \
  && pass "T4 wrapper exited 0 after STOP file" \
  || fail "T4 wrapper exited ${T4_RC}, expected 0"
grep -q 'STOP file detected' "${T1_OUT}" \
  && pass "T1 detected the STOP file" \
  || fail "T1 did not log STOP file detection"
grep -q 'occupier cleanup done' "${T1_OUT}" \
  && pass "T1 cleaned up its own occupiers" \
  || fail "T1 did not report occupier cleanup"

sleep 2
LEFT="$(count_occupiers)"
if [[ "${LEFT}" == "0" ]]; then
  pass "NO leftover occupier processes after STOP (ps clean)"
else
  fail "${LEFT} leftover occupier processes remain"
  ps -eo pid=,args= | grep -F -- "${MARKER}" | grep -E -- "${OCCUPIER_ARGV_RE}" >&2 || true
fi

# ===========================================================================
# T2 — success path
# ===========================================================================
echo ""
echo "--- T2: success path, TRAIN_COMMAND exits 0 ---"
T2_RUNTIME="${TEST_TMP}/t2_runtime"
T2_STATE="${TEST_TMP}/t2_state"
T2_OUT="${TEST_TMP}/t2_wrapper_stdout.log"
mkdir -p "${T2_RUNTIME}" "${T2_STATE}"
env PATH="${FAKE_BIN}:${PATH}" \
  TRAIN_COMMAND="bash -lc 'exit 0'" \
  OCCUPIER_DRY_RUN=1 \
  OCCUPY_RUNTIME_DIR="${T2_RUNTIME}" \
  KEEPALIVE_STATE_DIR="${T2_STATE}" \
  EXPECTED_GPUS_PER_NODE=8 \
  NODE_RANK=0 \
  ARNOLD_JOB_ID="smoketest_job" \
  timeout 120 bash "${WRAPPER}" > "${T2_OUT}" 2>&1
T2_RC=$?
[[ "${T2_RC}" == "0" ]] \
  && pass "success path returned 0" \
  || fail "success path returned ${T2_RC}, expected 0"
grep -q 'exiting 0 without occupying' "${T2_OUT}" \
  && pass "success path did not occupy (KEEPALIVE_ON_SUCCESS=0 default)" \
  || fail "success path did not log the no-occupy decision"
T2_OCC="$(count_occupiers)"
[[ "${T2_OCC}" == "0" ]] \
  && pass "success path started no occupiers" \
  || fail "success path leaked ${T2_OCC} occupiers"

# ===========================================================================
# T3 — KEEPALIVE_DISABLE=1 propagates rc
# ===========================================================================
echo ""
echo "--- T3: KEEPALIVE_DISABLE=1 propagates training rc ---"
T3_RUNTIME="${TEST_TMP}/t3_runtime"
T3_STATE="${TEST_TMP}/t3_state"
T3_OUT="${TEST_TMP}/t3_wrapper_stdout.log"
mkdir -p "${T3_RUNTIME}" "${T3_STATE}"
env PATH="${FAKE_BIN}:${PATH}" \
  TRAIN_COMMAND="bash -lc 'exit 7'" \
  KEEPALIVE_DISABLE=1 \
  OCCUPIER_DRY_RUN=1 \
  OCCUPY_RUNTIME_DIR="${T3_RUNTIME}" \
  KEEPALIVE_STATE_DIR="${T3_STATE}" \
  NODE_RANK=0 \
  ARNOLD_JOB_ID="smoketest_job" \
  timeout 120 bash "${WRAPPER}" > "${T3_OUT}" 2>&1
T3_RC=$?
[[ "${T3_RC}" == "7" ]] \
  && pass "KEEPALIVE_DISABLE=1 propagated rc=7 verbatim" \
  || fail "KEEPALIVE_DISABLE=1 returned ${T3_RC}, expected 7"
T3_OCC="$(count_occupiers)"
[[ "${T3_OCC}" == "0" ]] \
  && pass "KEEPALIVE_DISABLE=1 started no occupiers" \
  || fail "KEEPALIVE_DISABLE=1 leaked ${T3_OCC} occupiers"

# ===========================================================================
echo ""
echo "============================================================"
FINAL_LEFT="$(count_occupiers)"
[[ "${FINAL_LEFT}" == "0" ]] \
  && pass "FINAL: no occupier processes remain on this host" \
  || fail "FINAL: ${FINAL_LEFT} occupier processes remain"

if [[ "${FAILURES}" == "0" ]]; then
  echo "SMOKE TEST RESULT: ALL CHECKS PASSED"
  echo "============================================================"
  exit 0
fi
echo "SMOKE TEST RESULT: ${FAILURES} CHECK(S) FAILED" >&2
echo "============================================================"
exit 1
