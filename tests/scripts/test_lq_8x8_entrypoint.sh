#!/usr/bin/env bash
# ============================================================================
# Tests for scripts/run_pi05_skillbridge_bf16_multinode_lq_8x8.sh
#
# Scope: pure CPU, no GPU, no training, no occupier processes, no residue.
#
# Two families of checks:
#   A. Wiring checks — run the entrypoint against a STUBBED REPO_ROOT (the
#      script honours REPO_ROOT for web FULL_SCRIPT mounts). The stub replaces
#      the HL launcher and the keepalive wrapper with recorders, so we can
#      assert which layer gets exec'd and with which environment WITHOUT ever
#      starting real training or GPU occupiers.
#   B. Topology checks — run the entrypoint against the REAL repo in
#      OPENPI_LAUNCH_PREFLIGHT_ONLY=1 mode, which bypasses the wrapper and so
#      cannot occupy GPUs. Requires the conda env's python on PATH for the
#      live multiprocess.Manager() preflight; skipped with a clear message
#      when that interpreter is unavailable.
#
# Usage:
#   bash tests/scripts/test_lq_8x8_entrypoint.sh
# ============================================================================

set -uo pipefail

REPO_ROOT_REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENTRYPOINT_REL="scripts/run_pi05_skillbridge_bf16_multinode_lq_8x8.sh"
ENTRYPOINT="${REPO_ROOT_REAL}/${ENTRYPOINT_REL}"
HL_LAUNCHER_REL="scripts/run_pi05_ki_joint_query_single_task_radio_bf16_multinode_hl.sh"
WRAPPER_REL="scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh"

CONDA_BIN="/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin"

# Job id used by the real-preflight case. Deterministic so its node-local
# artifacts can be located and removed afterwards.
REAL_PREFLIGHT_JOB_ID="test_lq_8x8_entrypoint_selftest"
CONFIG_NAME_EXPECTED="pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_bf16"

PASS=0
FAIL=0
SKIP=0

ok()   { printf 'PASS  %s\n' "$*"; PASS=$((PASS + 1)); }
bad()  { printf 'FAIL  %s\n' "$*"; FAIL=$((FAIL + 1)); }
skip() { printf 'SKIP  %s\n' "$*"; SKIP=$((SKIP + 1)); }

TEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/test_lq_8x8_entrypoint.XXXXXX")" || {
  echo "FATAL: cannot create temp dir" >&2
  exit 1
}

# The real-preflight case makes the launcher create node-local state OUTSIDE
# TEST_TMP: a short TMPDIR alias symlink /tmp/openpi-tmp-<uid>-<digest> plus its
# backing tree under /tmp/openpi-comet/<user>/<config>/<job-id>. Removing only
# TEST_TMP would leave both behind, so clean up all three classes explicitly.
cleanup() {
  rm -rf -- "${TEST_TMP}"

  local backing_root="/tmp/openpi-comet/${USER:-tiger}/${CONFIG_NAME_EXPECTED}"
  local backing="${backing_root}/${REAL_PREFLIGHT_JOB_ID}"

  # Recompute the alias name exactly as the launcher does, and only remove it
  # when it is a symlink pointing at our own backing dir. Never touch aliases
  # belonging to other runs.
  local digest alias
  digest="$(printf '%s' "${backing}" | sha256sum)"
  digest="${digest%% *}"
  alias="/tmp/openpi-tmp-${UID:-$(id -u)}-${digest:0:24}"
  if [[ -L "${alias}" && "$(readlink -- "${alias}")" == "${backing}/tmp" ]]; then
    rm -f -- "${alias}"
  fi

  rm -rf -- "${backing}"
  # Drop the now-empty ancestors, but never a non-empty shared directory.
  rmdir -- "${backing_root}" 2>/dev/null || true
  rmdir -- "/tmp/openpi-comet/${USER:-tiger}" 2>/dev/null || true
  rmdir -- "/tmp/openpi-comet" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Build a stub REPO_ROOT: real entrypoint, recorder stand-ins beneath it.
# ---------------------------------------------------------------------------
STUB_ROOT="${TEST_TMP}/stub_repo"
mkdir -p "${STUB_ROOT}/scripts"
cp "${ENTRYPOINT}" "${STUB_ROOT}/${ENTRYPOINT_REL}"

RECORD_DIR="${TEST_TMP}/records"
mkdir -p "${RECORD_DIR}"

make_recorder() {
  local path="$1" tag="$2"
  cat > "${path}" <<EOF
#!/usr/bin/env bash
# Recorder stub: never trains, never touches a GPU.
{
  echo "INVOKED=${tag}"
  echo "LAUNCHER=\${LAUNCHER:-<unset>}"
  echo "CONFIG_NAME=\${CONFIG_NAME:-<unset>}"
  echo "CONDA_ROOT=\${CONDA_ROOT:-<unset>}"
  echo "B1K_DATASET_ROOT=\${B1K_DATASET_ROOT:-<unset>}"
  echo "BASE_PI05_CKPT=\${BASE_PI05_CKPT:-<unset>}"
  echo "REPO_OPENPI_CACHE=\${REPO_OPENPI_CACHE:-<unset>}"
  echo "PERSISTENT_OUTPUT_ROOT=\${PERSISTENT_OUTPUT_ROOT:-<unset>}"
  echo "WANDB_MODE=\${WANDB_MODE:-<unset>}"
  echo "WANDB_DISABLED=\${WANDB_DISABLED:-<unset>}"
  echo "KEEPALIVE_ON_SUCCESS=\${KEEPALIVE_ON_SUCCESS:-<unset>}"
  echo "STRICT_GPU_COUNT=\${STRICT_GPU_COUNT:-<unset>}"
  echo "NUM_TRAIN_STEPS=\${NUM_TRAIN_STEPS:-<unset>}"
  echo "NUM_TRAIN_EPOCHS=\${NUM_TRAIN_EPOCHS:-<unset>}"
  echo "BATCH_SIZE_PER_GPU=\${BATCH_SIZE_PER_GPU:-<unset>}"
  echo "NUM_WORKERS=\${NUM_WORKERS:-<unset>}"
  echo "SAVE_INTERVAL=\${SAVE_INTERVAL:-<unset>}"
  echo "VAL_LOG_INTERVAL=\${VAL_LOG_INTERVAL:-<unset>}"
  echo "PYTORCH_TRAINING_PRECISION=\${PYTORCH_TRAINING_PRECISION:-<unset>}"
  echo "EXP_NAME=\${EXP_NAME:-<unset>}"
  echo "NUM_NODES=\${NUM_NODES:-<unset>}"
  echo "GPUS_PER_NODE=\${GPUS_PER_NODE:-<unset>}"
  echo "NODE_RANK=\${NODE_RANK:-<unset>}"
  echo "MASTER_ADDR=\${MASTER_ADDR:-<unset>}"
  echo "MASTER_PORT=\${MASTER_PORT:-<unset>}"
} > "${RECORD_DIR}/${tag}.env"
exit 0
EOF
  chmod +x "${path}"
}

make_recorder "${STUB_ROOT}/${HL_LAUNCHER_REL}" "hl_launcher"
make_recorder "${STUB_ROOT}/${WRAPPER_REL}" "wrapper"

run_stub() {
  # run_stub <extra env assignments...> ; prints nothing, sets STUB_RC
  rm -f "${RECORD_DIR}"/*.env
  env -i \
    HOME="${HOME}" USER="${USER:-tiger}" \
    PATH="/usr/local/bin:/usr/bin:/bin" \
    REPO_ROOT="${STUB_ROOT}" \
    ARNOLD_WORKER_0_HOST=10.0.0.1 \
    ARNOLD_WORKER_0_PORT=29514 \
    ARNOLD_JOB_ID=stubtest \
    "$@" \
    bash "${STUB_ROOT}/${ENTRYPOINT_REL}" > "${TEST_TMP}/stub.out" 2> "${TEST_TMP}/stub.err"
  STUB_RC=$?
}

record_value() {
  # record_value <tag> <key>
  sed -n "s/^$2=//p" "${RECORD_DIR}/$1.env" 2>/dev/null | head -1
}

echo "=== bash -n ==="
if bash -n "${ENTRYPOINT}"; then
  ok "bash -n ${ENTRYPOINT_REL}"
else
  bad "bash -n ${ENTRYPOINT_REL}"
fi

echo
echo "=== A. wiring (stubbed REPO_ROOT, no GPU, no training) ==="

run_stub ARNOLD_WORKER_NUM=8 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0
if [[ "${STUB_RC}" -eq 0 ]]; then
  ok "8x8 normal path exits 0"
else
  bad "8x8 normal path rc=${STUB_RC} (stderr: $(head -2 "${TEST_TMP}/stub.err" | tr '\n' ' '))"
fi

if [[ -f "${RECORD_DIR}/wrapper.env" ]]; then
  ok "normal path exec's the keepalive wrapper"
else
  bad "normal path did NOT exec the keepalive wrapper"
fi
if [[ -f "${RECORD_DIR}/hl_launcher.env" ]]; then
  bad "normal path must not exec the launcher directly (wrapper does that)"
else
  ok "normal path does not bypass the wrapper"
fi

# THE regression this script exists for: the wrapper must receive an explicit
# LAUNCHER pointing at the generic HL launcher, never fall back to its own
# 4-node-locked LQ default.
got_launcher="$(record_value wrapper LAUNCHER)"
if [[ "${got_launcher}" == "${STUB_ROOT}/${HL_LAUNCHER_REL}" ]]; then
  ok "wrapper receives explicit LAUNCHER=<HL launcher>"
else
  bad "wrapper LAUNCHER wrong: ${got_launcher}"
fi
case "${got_launcher}" in
  *skillbridge_bf16_multinode_lq.sh)
    bad "LAUNCHER points at the 4-node-locked LQ launcher (the original bug)"
    ;;
  *)
    ok "LAUNCHER is not the 4-node-locked LQ launcher"
    ;;
esac

check_kv() {
  local key="$1" want="$2"
  local got
  got="$(record_value wrapper "${key}")"
  if [[ "${got}" == "${want}" ]]; then
    ok "${key}=${want}"
  else
    bad "${key}: want '${want}', got '${got}'"
  fi
}

check_kv CONFIG_NAME "${CONFIG_NAME_EXPECTED}"
check_kv CONDA_ROOT "/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3"
check_kv B1K_DATASET_ROOT "/mnt/bn/saiwenresearch/mlx/users/chenjunting/data/2025-challenge-demos/"
check_kv BASE_PI05_CKPT "/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch"
check_kv REPO_OPENPI_CACHE "/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/.cache/openpi"
check_kv WANDB_MODE "online"
check_kv WANDB_DISABLED "0"
check_kv KEEPALIVE_ON_SUCCESS "1"
check_kv STRICT_GPU_COUNT "0"
check_kv NUM_TRAIN_STEPS "2000"
check_kv NUM_TRAIN_EPOCHS "1"
check_kv BATCH_SIZE_PER_GPU "1"
check_kv NUM_WORKERS "4"
check_kv SAVE_INTERVAL "200"
check_kv VAL_LOG_INTERVAL "100"
check_kv PYTORCH_TRAINING_PRECISION "bfloat16"
check_kv NUM_NODES "8"
check_kv GPUS_PER_NODE "8"
check_kv NODE_RANK "0"

# EXP_NAME must stay unset: the HL launcher elects it on rank 0 and publishes it
# via NAS, so a per-node value here would desynchronise the 8 nodes.
got_exp="$(record_value wrapper EXP_NAME)"
if [[ "${got_exp}" == "<unset>" ]]; then
  ok "EXP_NAME left unset (rank-0 election preserved)"
else
  bad "EXP_NAME must not be pre-set, got '${got_exp}'"
fi

# Base checkpoint must come from the canonical repo, not the worktree, because
# a worktree carries no checkpoints/ tree.
if [[ "$(record_value wrapper BASE_PI05_CKPT)" == "${STUB_ROOT}"* ]]; then
  bad "BASE_PI05_CKPT resolved under REPO_ROOT instead of the canonical repo"
else
  ok "BASE_PI05_CKPT points outside REPO_ROOT (canonical repo)"
fi

# Preflight escape hatch must bypass the wrapper, otherwise a successful
# preflight plus KEEPALIVE_ON_SUCCESS=1 would start real occupiers.
run_stub ARNOLD_WORKER_NUM=8 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0 OPENPI_LAUNCH_PREFLIGHT_ONLY=1
if [[ "${STUB_RC}" -eq 0 ]]; then
  ok "preflight-only path exits 0"
else
  bad "preflight-only path rc=${STUB_RC}"
fi
if [[ -f "${RECORD_DIR}/hl_launcher.env" && ! -f "${RECORD_DIR}/wrapper.env" ]]; then
  ok "preflight-only bypasses the wrapper (no occupier possible)"
else
  bad "preflight-only did not bypass the wrapper"
fi

# Explicit caller overrides must still win over the frozen defaults.
run_stub ARNOLD_WORKER_NUM=8 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0 NUM_TRAIN_STEPS=7
if [[ "$(record_value wrapper NUM_TRAIN_STEPS)" == "7" ]]; then
  ok "explicit NUM_TRAIN_STEPS override is honoured"
else
  bad "explicit override lost: NUM_TRAIN_STEPS=$(record_value wrapper NUM_TRAIN_STEPS)"
fi

# Offline / disabled W&B must be rejected: this trial requires online tracking.
run_stub ARNOLD_WORKER_NUM=8 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0 WANDB_MODE=offline
if [[ "${STUB_RC}" -ne 0 ]] && grep -q 'WANDB_MODE=online' "${TEST_TMP}/stub.err"; then
  ok "WANDB_MODE=offline rejected"
else
  bad "WANDB_MODE=offline was not rejected (rc=${STUB_RC})"
fi

run_stub ARNOLD_WORKER_NUM=8 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0 WANDB_DISABLED=1
if [[ "${STUB_RC}" -ne 0 ]] && grep -q 'WANDB_DISABLED' "${TEST_TMP}/stub.err"; then
  ok "WANDB_DISABLED=1 rejected"
else
  bad "WANDB_DISABLED=1 was not rejected (rc=${STUB_RC})"
fi

echo
echo "=== A2. topology rejection (stubbed, cheap) ==="
assert_topology_reject() {
  local desc="$1"
  shift
  run_stub "$@"
  if [[ "${STUB_RC}" -ne 0 && ! -f "${RECORD_DIR}/wrapper.env" && ! -f "${RECORD_DIR}/hl_launcher.env" ]]; then
    ok "${desc} rejected before any launch (rc=${STUB_RC})"
  else
    bad "${desc} was NOT rejected (rc=${STUB_RC})"
  fi
}
assert_topology_reject "4x8"            ARNOLD_WORKER_NUM=4 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0
assert_topology_reject "8x4"            ARNOLD_WORKER_NUM=8 ARNOLD_WORKER_GPU=4 ARNOLD_ID=0
assert_topology_reject "1x8"            ARNOLD_WORKER_NUM=1 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0
assert_topology_reject "16x8"           ARNOLD_WORKER_NUM=16 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0
assert_topology_reject "rank==nodes"    ARNOLD_WORKER_NUM=8 ARNOLD_WORKER_GPU=8 ARNOLD_ID=8
assert_topology_reject "missing nodes"  ARNOLD_WORKER_GPU=8 ARNOLD_ID=0
assert_topology_reject "non-numeric"    ARNOLD_WORKER_NUM=eight ARNOLD_WORKER_GPU=8 ARNOLD_ID=0

echo
echo "=== B. real 8x8 preflight (short TMPDIR + live multiprocess.Manager) ==="
if [[ ! -x "${CONDA_BIN}/python" ]]; then
  skip "conda python not executable at ${CONDA_BIN}/python; live Manager preflight not run"
else
  PRE_OUT="${TEST_TMP}/real_preflight.out"
  env -i HOME="${HOME}" USER="${USER:-tiger}" \
    PATH="${CONDA_BIN}:/usr/local/bin:/usr/bin:/bin" \
    OPENPI_LAUNCH_PREFLIGHT_ONLY=1 \
    ARNOLD_WORKER_NUM=8 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0 \
    ARNOLD_WORKER_0_HOST=10.0.0.1 ARNOLD_WORKER_0_PORT=29514 \
    ARNOLD_JOB_ID="${REAL_PREFLIGHT_JOB_ID}" \
    bash "${ENTRYPOINT}" > "${PRE_OUT}" 2>&1
  REAL_RC=$?
  if [[ "${REAL_RC}" -eq 0 ]]; then
    ok "real 8x8 preflight rc=0"
  else
    bad "real 8x8 preflight rc=${REAL_RC}: $(tail -3 "${PRE_OUT}" | tr '\n' ' ')"
  fi
  for marker in MULTIPROCESS_MANAGER_PREFLIGHT_OK LOCAL_CACHE_PREFLIGHT_OK; do
    if grep -q "${marker}" "${PRE_OUT}"; then
      ok "preflight reported ${marker}"
    else
      bad "preflight missing ${marker}"
    fi
  done
  sock_bytes="$(sed -n 's/^MULTIPROCESS_MANAGER_SOCKET_BYTES=//p' "${PRE_OUT}" | head -1)"
  if [[ "${sock_bytes}" =~ ^[0-9]+$ ]] && (( sock_bytes <= 107 )); then
    ok "live Manager socket ${sock_bytes} bytes <= 107 (AF_UNIX limit)"
  else
    bad "live Manager socket byte check failed: '${sock_bytes}'"
  fi
fi

echo
echo "=== C. no GPU occupiers were started ==="
# Match BOTH the occupier marker and the occupier script name: a bare marker
# match also hits this test's own argv / the wrapper's argv (self-match hazard).
occ="$(pgrep -af '__GPU_OCCUPY__torch_mm_512' 2>/dev/null | grep -c 'gpu_occupy_torch_mm\.py' || true)"
if [[ "${occ}" == "0" ]]; then
  ok "no occupier processes running (count=0)"
else
  bad "occupier processes detected (count=${occ})"
fi

echo
printf 'RESULT: %d passed, %d failed, %d skipped\n' "${PASS}" "${FAIL}" "${SKIP}"
[[ "${FAIL}" -eq 0 ]]
