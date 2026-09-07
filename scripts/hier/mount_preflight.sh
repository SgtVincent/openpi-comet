#!/usr/bin/env bash
# ============================================================================
# Per-rank mount preflight for the MoMA-VLA training path.
#
# Runs ON EVERY RANK, BEFORE training starts, and answers one question:
#   "does THIS pod actually have every mount the training path will read,
#    with real bytes behind it?"
#
# WHY CONTENT, NOT EXISTENCE
# --------------------------
# When a bind/fuse/nfs mount fails the mount-point DIRECTORY STILL EXISTS.
# `[[ -d ... ]]` and `[[ -f ... ]]` therefore prove nothing at all.
# Worse, on this very host there are TWO INDEPENDENT recorded cases of
# "the write returned success but the file was zero bytes":
#     (a) a `cp` that produced an empty destination file, and
#     (b) git loose objects that were zeroed -- twice, and in one case the
#         object was written CORRECTLY FIRST and zeroed only later.
# So neither `rc=0` nor path existence is evidence. The only thing that is
# evidence is: READ the file and assert the number of bytes you got back is
# greater than zero. That is what every probe here does.
#
# Case (b) also means "assert non-empty immediately after writing" is not
# enough -- the zeroing can happen afterwards. That is precisely why this
# check is a DELAYED RE-READ run at job start, not a write-time assertion.
#
# WHY ONE RANK'S FAILURE MUST FAIL THE WHOLE JOB
# ----------------------------------------------
# If a single node is missing a mount and only that node dies, the job does
# not present as "a mount was missing". It presents as "training hung" or
# "a rank dropped" -- torchrun/NCCL peers block on a collective and the real
# cause is 30 log-scrolls away on one node out of four. Partial failure is
# dramatically harder to diagnose than total failure. So this script runs a
# cross-rank VETO BARRIER on shared NAS (see BARRIER below): every rank
# publishes its verdict, every rank requires ALL peers to be OK, and any FAIL
# (or any MISSING / STALE verdict) aborts every rank with a message that names
# the guilty rank. Total, uniform, self-describing failure.
#
# WHY RANK COMES FROM ARNOLD_ID AND NOWHERE ELSE
# ----------------------------------------------
# Measured on the two production H20 jobs: POD INDEX != NODE RANK, on both.
#     JOB1 (d7f6f139...): executor-0 is ARNOLD_ID=1
#     JOB2 (ef9bf869...): executor-0 is ARNOLD_ID=3, executor-3 is ARNOLD_ID=0
# Anything that infers rank from the pod name mislabels the evidence, which is
# worse than having none. Rank is read from ARNOLD_ID inside the pod, and if
# ARNOLD_ID is unset this script REFUSES to guess: it exits 11 (loud), and
# even the explicit override tags the rank as the literal string UNSET --
# never 0. (Note: the keepalive wrapper's own `NODE_RANK="${NODE_RANK:-
# ${ARNOLD_ID:-0}}"` DOES silently collapse to 0; do not copy that idiom.)
#
# HARNESS SELF-VALIDATION (why an all-red run is not acceptable)
# -------------------------------------------------------------
# A check that goes all-red on any problem carries as little information as
# one that always passes: you cannot read anything out of the red. Every run
# therefore includes two always-on self-checks, reported separately from the
# real mounts:
#   POSCTRL  a file this script writes into its own node-local temp dir with
#            known content -- MUST come back OK. If it does not, the measuring
#            apparatus is broken and the verdict is NOT MEASURED (exit 15),
#            never "your mounts are bad".
#   ABSENT   a path that cannot exist -- MUST come back FAIL. If it comes back
#            OK, the comparison is inverted and the run is void (exit 15).
# Failures are also per-probe: probing loop has no `set -e` and no early
# `break`, so ONE bad path reports FAIL while every genuine probe still
# reports its own OK with its own byte count.
#
# USAGE
#   bash scripts/hier/mount_preflight.sh
#
# ENVIRONMENT KNOBS
#   MOUNT_PREFLIGHT_BARRIER          1 (default) -> cross-rank veto barrier on
#                                    shared NAS. 0 -> local probes only
#                                    (single-node / manual runs).
#   MOUNT_PREFLIGHT_BARRIER_DIR      override the barrier directory outright
#   MOUNT_PREFLIGHT_STATE_ROOT       parent of the barrier dir
#                                    (default <handoff>/mount_preflight)
#   MOUNT_PREFLIGHT_RUN_KEY          REQUIRED when the cross-rank barrier is on.
#                                    It must be a rank-consistent, unique token
#                                    for THIS robust attempt, not ARNOLD_TRIAL_ID
#                                    (the same trial id survives restarts).
#                                    Reusing a namespace can mix rank0 from this
#                                    attempt with fresh-looking rank1-3 verdicts
#                                    from the previous attempt and release early.
#   MOUNT_PREFLIGHT_NUM_RANKS        expected rank count (default
#                                    ARNOLD_WORKER_NUM; loud if unset)
#   MOUNT_PREFLIGHT_BARRIER_TIMEOUT_S  default 300
#   MOUNT_PREFLIGHT_BARRIER_POLL_S     default 5
#   MOUNT_PREFLIGHT_STALE_S          default 900. A peer verdict older than
#                                    this counts as MISSING, not as OK, so a
#                                    leftover verdict from an earlier attempt
#                                    cannot wave a broken node through.
#   MOUNT_PREFLIGHT_ALLOW_UNSET_RANK 1 -> tolerate unset ARNOLD_ID, rank is
#                                    reported as the string UNSET (never 0)
#                                    and the barrier is force-disabled.
#   MOUNT_PREFLIGHT_MAX_BYTES        bounded prefix read per probe
#                                    (default 65536). Large shards are not
#                                    read in full; the point is a real read(),
#                                    not a checksum.
#   MOUNT_PREFLIGHT_EXTRA_PROBES     ';'-separated extra 'GROUP:/abs/path'
#                                    entries. Used by the negative control to
#                                    inject a path that cannot exist.
#   MOUNT_PREFLIGHT_TMPDIR           node-local scratch (default under /tmp).
#                                    Deliberately NOT on any NAS: the
#                                    measuring apparatus must not live on the
#                                    thing being measured.
#   MOUNT_ROOT_WORKTREE / _HANDOFF / _WT_STAGE1 / _DATASET
#                                    override the four mount roots
#
# EXIT CODES (contract)
#   0   ALL_OK  -- every local probe read >0 bytes AND (barrier on) every peer
#                 rank published OK
#   10  LOCAL_FAIL      -- this rank has at least one bad probe. The verdict is
#                          published as FAIL first, so peers abort too.
#   11  RANK_UNKNOWN    -- ARNOLD_ID unset/non-numeric and no explicit override
#   12  PEER_FAIL       -- a peer rank published FAIL; this rank aborts so the
#                          whole job fails together
#   13  BARRIER_TIMEOUT -- did not see all NUM_RANKS fresh verdicts in time;
#                          missing ranks are named
#   14  PUBLISH_FAIL    -- could not durably publish own verdict (barrier dir
#                          unwritable, or the written file read back empty --
#                          the zero-byte hazard applied to our own bookkeeping)
#   15  NOT_MEASURED    -- harness self-check failed (POSCTRL not OK, or
#                          ABSENT sentinel came back OK). Says nothing about
#                          the mounts; do not interpret as pass or fail.
#   2   usage error
# ============================================================================

# NOTE: deliberately NO `set -e`. A failing probe must NOT abort the run --
# per-probe granularity is the whole point (see HARNESS SELF-VALIDATION).
set -uo pipefail

readonly PF_TAG='mount-preflight'
readonly PF_VERSION='1.0.0'

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
readonly RC_ALL_OK=0
readonly RC_LOCAL_FAIL=10
readonly RC_RANK_UNKNOWN=11
readonly RC_PEER_FAIL=12
readonly RC_BARRIER_TIMEOUT=13
readonly RC_PUBLISH_FAIL=14
readonly RC_NOT_MEASURED=15
readonly RC_USAGE=2

case "${1:-}" in
  -h | --help)
    # Print the header comment block only. The end line is pinned by the
    # marker below, not hard-coded, so editing the header cannot silently make
    # --help spill source code into its own output.
    _hdr_end="$(grep -n '^# =\{20,\}' "${BASH_SOURCE[0]}" | sed -n '2s/:.*//p')"
    sed -n "2,${_hdr_end:-124}p" "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  '') : ;;
  *)
    printf '%s: unexpected argument %q (this script takes no positional args)\n' \
      "${PF_TAG}" "$1" >&2
    exit "${RC_USAGE}"
    ;;
esac

# ---------------------------------------------------------------------------
# Identity. RANK COMES FROM ARNOLD_ID ONLY -- never from the pod name.
# ---------------------------------------------------------------------------
HOST_NAME="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown-host)"
POD_NAME="${MY_POD_NAME:-${ARNOLD_POD_NAME:-<unset>}}"
ALLOW_UNSET_RANK="${MOUNT_PREFLIGHT_ALLOW_UNSET_RANK:-0}"

RANK_SOURCE='ARNOLD_ID'
RANK="${ARNOLD_ID-}"
RANK_KNOWN=1
if [[ -z "${RANK}" ]]; then
  RANK_KNOWN=0
elif [[ ! "${RANK}" =~ ^[0-9]+$ ]]; then
  RANK_KNOWN=0
fi

if [[ "${RANK_KNOWN}" -eq 0 ]]; then
  # LOUD. Never default to 0: a wrong rank label is worse than no rank label,
  # because every downstream "which rank saw what" answer becomes a lie.
  {
    printf '\n'
    printf '################################################################\n'
    printf '## [%s] ARNOLD_ID IS UNSET OR NON-NUMERIC (got: %s)\n' "${PF_TAG}" "${ARNOLD_ID-<unset>}"
    printf '## Rank CANNOT be inferred from the pod name: measured on two\n'
    printf '## production jobs, pod index and rank are misaligned (JOB1\n'
    printf '## executor-0 = rank 1; JOB2 executor-0 = rank 3, executor-3 =\n'
    printf '## rank 0). Refusing to guess.\n'
    printf '## host=%s pod=%s\n' "${HOST_NAME}" "${POD_NAME}"
    printf '## To run anyway (single-node / manual), set\n'
    printf '##     MOUNT_PREFLIGHT_ALLOW_UNSET_RANK=1\n'
    printf '## The rank will then be reported as the literal string UNSET and\n'
    printf '## the cross-rank barrier is force-disabled.\n'
    printf '################################################################\n'
  } >&2
  if [[ "${ALLOW_UNSET_RANK}" != "1" ]]; then
    printf '[%s] VERDICT rank=UNKNOWN host=%s result=FAIL probes_ok=0/0 failed=ARNOLD_ID_UNSET exit=%d run_key=- peers=-/-\n' \
      "${PF_TAG}" "${HOST_NAME}" "${RC_RANK_UNKNOWN}" >&2
    exit "${RC_RANK_UNKNOWN}"
  fi
  RANK='UNSET'
  RANK_SOURCE='OVERRIDE(MOUNT_PREFLIGHT_ALLOW_UNSET_RANK=1)'
fi

# ---------------------------------------------------------------------------
# Mount roots under test
# ---------------------------------------------------------------------------
# Production must pass the immutable clone root explicitly. Defaulting to the
# active worktree recreates the exact race the clone is meant to prevent (HEAD
# and uncommitted files can move between rank starts), so barrier-enabled runs
# fail closed below if this knob is omitted.
MOUNT_ROOT_WORKTREE="${MOUNT_ROOT_WORKTREE:-}"
MOUNT_ROOT_HANDOFF="${MOUNT_ROOT_HANDOFF:-/mnt/bn/behavior-data-hl/chenjunting/repo/moma_handoff_20260907}"
MOUNT_ROOT_WT_STAGE1="${MOUNT_ROOT_WT_STAGE1:-/mnt/bn/navigation-hl/mlx/users/chenjunting/h20_skillbridge/wt_stage1}"
MOUNT_ROOT_DATASET="${MOUNT_ROOT_DATASET:-/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos}"
MOUNT_ROOT_MEMORY="${MOUNT_ROOT_DATASET}/derived/fixed_compact_memory_annotations"

MAX_BYTES="${MOUNT_PREFLIGHT_MAX_BYTES:-65536}"
if [[ "${MOUNT_PREFLIGHT_BARRIER:-1}" == '1' && -z "${MOUNT_ROOT_WORKTREE}" ]]; then
  printf '[%s] FATAL: MOUNT_ROOT_WORKTREE is required and must name the immutable launch clone; refusing to probe an active worktree.\n' "${PF_TAG}" >&2
  exit "${RC_USAGE}"
fi
if [[ -z "${MOUNT_ROOT_WORKTREE}" ]]; then
  MOUNT_ROOT_WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi

# ---------------------------------------------------------------------------
# Probe table:  GROUP|absolute path|why this file is a reasonable sentinel
#
# Selection rules applied to every entry below:
#   * it is a file the TRAINING PATH ITSELF reads or executes (so a zero-byte
#     read here is a real outage, not a synthetic one), or the smallest
#     load-bearing descriptor of its subtree;
#   * it is stable -- not a log, not a lock, not an output artifact that a
#     concurrent run may rotate or truncate;
#   * it is small enough that reading a prefix on every rank at startup costs
#     nothing (the 367 MB meta/episodes_stats.jsonl is deliberately NOT used);
#   * for each root there is at least one probe BELOW the root, not only at
#     it, because a partially-materialised mount can show top-level entries
#     while the subtree underneath is empty.
# ---------------------------------------------------------------------------
PROBES=(
  # -- code worktree -------------------------------------------------------
  "WORKTREE|${MOUNT_ROOT_WORKTREE}/pyproject.toml|tracked, at the worktree root, ~3.2 KB, untouched by the in-flight MoMA work; cheapest proof that the root resolves to real content instead of an empty mount point"
  "WORKTREE|${MOUNT_ROOT_WORKTREE}/.git|the 101-byte 'gitdir:' pointer that MAKES this a linked worktree; git loose objects have been zeroed on this host twice, and if this pointer is zeroed every git call in the launcher breaks with an unrelated-looking error"
  "WORKTREE|${MOUNT_ROOT_WORKTREE}/scripts/run_pi05_ki_joint_query_single_task_radio_skillbridge_bf16_multinode_lq.sh|the exact file the keepalive wrapper runs as \${LAUNCHER}. An empty shell script EXITS 0: a zero-byte read here makes training silently never start while every rc says success. Strongest functional sentinel in the tree"
  # -- handoff directory ---------------------------------------------------
  "HANDOFF|${MOUNT_ROOT_HANDOFF}/run_momavla_keepalive_on_failure.sh|the MoMA-VLA entrypoint wrapper the launch path executes out of the handoff dir; same 'empty script exits 0' hazard as above"
  "HANDOFF|${MOUNT_ROOT_HANDOFF}/moma_doc.md|stable spec document, so the handoff group does not depend only on executables; catches a mount that lost plain data but kept scripts"
  # -- wt_stage1 (SEPARATE PHYSICAL MOUNT: navigation-hl / nfs4) -----------
  "WT_STAGE1|${MOUNT_ROOT_WT_STAGE1}/scripts/train_accelerate.py|literally the trainer this job execs (~300 KB). This is the single most load-bearing file in the launch path; it lives on navigation-hl, a DIFFERENT physical mount from every other probe, so it is the one that can fail independently"
  "WT_STAGE1|${MOUNT_ROOT_WT_STAGE1}/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh|the entrypoint recorded for the live H20 job (audit/H20_RECON.md); proves the scripts/ subtree of the frozen clone, not just its root"
  # -- dataset root --------------------------------------------------------
  "DATASET|${MOUNT_ROOT_DATASET}/meta/info.json|the LeRobot dataset descriptor the loader opens FIRST; an empty one surfaces as an obscure JSON error deep inside the data pipeline instead of 'the dataset mount is gone'"
  "DATASET|${MOUNT_ROOT_DATASET}/meta/tasks.jsonl|the task table, also loader-critical and small"
  "DATASET|${MOUNT_ROOT_DATASET}/data/task-0000/episode_00000010.parquet|one real training shard. meta/ alone can be present while the bulk data subtree is empty, so the bulk tree needs its own probe. Only a bounded prefix is read"
  # -- memory annotation subtree ------------------------------------------
  "DATASET_MEMORY|${MOUNT_ROOT_MEMORY}/manifest.json|the manifest enumerating the 10,000 annotated episodes; the memory path validates against it, so a zero-byte manifest silently changes what is trained on"
  "DATASET_MEMORY|${MOUNT_ROOT_MEMORY}/compact_memory_cache.jsonl|the compact-memory cache the memory path actually loads (~1.1 MB)"
  "DATASET_MEMORY|${MOUNT_ROOT_MEMORY}/task-0000/episode_00000010.json|one leaf per-episode annotation. The top-level files above can be materialised while the 52 per-task directories are empty; this is the only probe that rules that out"
)

# Optional extra probes (negative control injection).
if [[ -n "${MOUNT_PREFLIGHT_EXTRA_PROBES:-}" ]]; then
  _saved_ifs="${IFS}"
  IFS=';'
  read -r -a _extra <<< "${MOUNT_PREFLIGHT_EXTRA_PROBES}"
  IFS="${_saved_ifs}"
  for _e in "${_extra[@]}"; do
    [[ -z "${_e}" ]] && continue
    _g="${_e%%:*}"
    _p="${_e#*:}"
    if [[ -z "${_g}" || -z "${_p}" || "${_p}" != /* ]]; then
      printf '[%s] FATAL: bad MOUNT_PREFLIGHT_EXTRA_PROBES entry %q (want GROUP:/abs/path)\n' \
        "${PF_TAG}" "${_e}" >&2
      exit "${RC_USAGE}"
    fi
    PROBES+=("${_g}|${_p}|injected via MOUNT_PREFLIGHT_EXTRA_PROBES")
  done
fi

# ---------------------------------------------------------------------------
# Node-local scratch. NOT on any NAS: the measuring apparatus must not live on
# the thing being measured, and /tmp here is a local nvme (not a mount under
# test), so a NAS outage cannot silently zero our own scratch reads.
# ---------------------------------------------------------------------------
TMP_ROOT="${MOUNT_PREFLIGHT_TMPDIR:-/tmp/mount_preflight.$$.$(date +%s)}"
HARNESS_OK=1
if ! mkdir -p "${TMP_ROOT}" 2>/dev/null; then
  printf '[%s] FATAL: cannot create node-local scratch dir %s -- NOTHING WAS MEASURED\n' \
    "${PF_TAG}" "${TMP_ROOT}" >&2
  printf '[%s] VERDICT rank=%s host=%s result=NOT_MEASURED probes_ok=0/0 failed=HARNESS_TMPDIR exit=%d run_key=- peers=-/-\n' \
    "${PF_TAG}" "${RANK}" "${HOST_NAME}" "${RC_NOT_MEASURED}" >&2
  exit "${RC_NOT_MEASURED}"
fi
cleanup() { rm -rf "${TMP_ROOT}" 2>/dev/null || true; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
RANK_LABEL="rank=${RANK}"
log()     { printf '[%s][%s][%s] %s\n' "${PF_TAG}" "$(ts)" "${RANK_LABEL}" "$*"; }
log_err() { printf '[%s][%s][%s] %s\n' "${PF_TAG}" "$(ts)" "${RANK_LABEL}" "$*" >&2; }

# ---------------------------------------------------------------------------
# "which mount did this rank actually see" -- resolve a path to the mount that
# backs it via /proc/self/mountinfo (longest mount-point prefix wins).
# Prints: '<mountpoint> <fstype> <source>'
# ---------------------------------------------------------------------------
mount_of() {
  local p="$1"
  awk -v p="${p}" '
    {
      mp = $5
      sep = 0
      for (i = 6; i <= NF; i++) { if ($i == "-") { sep = i; break } }
      if (sep == 0) next
      fstype = $(sep + 1); src = $(sep + 2)
      pref = (mp == "/") ? "/" : mp "/"
      if (p == mp || index(p, pref) == 1) {
        if (length(mp) > bl) { bl = length(mp); out = mp " " fstype " " src }
      }
    }
    END { print (out == "") ? "<no-mount-match> - -" : out }
  ' /proc/self/mountinfo 2>/dev/null || printf '<mountinfo-unreadable> - -'
}

# ---------------------------------------------------------------------------
# THE actual assertion: read the file and count the bytes we got back.
#
# Implementation notes that matter:
#  * `head -c N -- FILE > tmp` performs a real read() of FILE. We then take the
#    byte count from the LOCAL temp. We do NOT use `wc -c < FILE`: GNU wc
#    short-circuits regular files via fstat+lseek and never reads the content,
#    which would silently downgrade this to the metadata check we are trying
#    to avoid.
#  * `head` is invoked as a SIMPLE COMMAND, not a pipeline, so the rc captured
#    into PROBE_READ_RC really is head's rc (after a pipeline `$?` belongs to
#    the last element of the pipe, which is a classic way to record rc=0 for a
#    command that failed).
#  * stat_size is recorded ALONGSIDE bytes_read, never instead of it, so the
#    smoking-gun shape from this host -- "stat says 300954, read gave 0" --
#    is visible in the log and gets its own status FAIL_ZEROED.
#
# Sets: PROBE_STATUS PROBE_BYTES PROBE_STAT_SIZE PROBE_READ_RC PROBE_ERR
# ---------------------------------------------------------------------------
probe_read() {
  local f="$1"
  local tmp="${TMP_ROOT}/read.$RANDOM$RANDOM"
  local errf="${tmp}.err"

  PROBE_STATUS='FAIL_HARNESS'
  PROBE_BYTES=-1
  PROBE_STAT_SIZE=-1
  PROBE_READ_RC=-1
  PROBE_ERR=''

  if [[ ! -e "${f}" ]]; then
    PROBE_STATUS='FAIL_MISSING'
    PROBE_BYTES=0
    PROBE_ERR='path does not exist'
    return 1
  fi
  if [[ -d "${f}" ]]; then
    PROBE_STATUS='FAIL_IS_DIR'
    PROBE_BYTES=0
    PROBE_ERR='probe target is a directory; a probe must be a readable FILE (an existing directory is exactly what a failed mount leaves behind)'
    return 1
  fi
  if [[ ! -f "${f}" ]]; then
    PROBE_STATUS='FAIL_NOT_REGULAR'
    PROBE_BYTES=0
    PROBE_ERR='not a regular file'
    return 1
  fi

  PROBE_STAT_SIZE="$(stat -c '%s' -- "${f}" 2>/dev/null)" || PROBE_STAT_SIZE=-1
  [[ -n "${PROBE_STAT_SIZE}" ]] || PROBE_STAT_SIZE=-1

  # simple command -> $? is head's own rc
  head -c "${MAX_BYTES}" -- "${f}" > "${tmp}" 2> "${errf}"
  PROBE_READ_RC=$?

  PROBE_BYTES="$(stat -c '%s' -- "${tmp}" 2>/dev/null)" || PROBE_BYTES=-1
  [[ -n "${PROBE_BYTES}" ]] || PROBE_BYTES=-1
  PROBE_ERR="$(tr -d '\n' < "${errf}" 2>/dev/null)"
  rm -f "${tmp}" "${errf}" 2>/dev/null || true

  if [[ "${PROBE_BYTES}" -lt 0 ]]; then
    PROBE_STATUS='FAIL_HARNESS'
    PROBE_ERR="${PROBE_ERR:-could not size the local copy; NOT MEASURED}"
    return 1
  fi
  if [[ "${PROBE_READ_RC}" -ne 0 ]]; then
    PROBE_STATUS='FAIL_READ_ERR'
    return 1
  fi
  if [[ "${PROBE_BYTES}" -eq 0 ]]; then
    if [[ "${PROBE_STAT_SIZE}" -gt 0 ]]; then
      # stat claims bytes, read returned none: the exact 'write said OK but the
      # file is zero bytes' shape this host has produced twice.
      PROBE_STATUS='FAIL_ZEROED'
      PROBE_ERR="${PROBE_ERR:-stat_size=${PROBE_STAT_SIZE} but read returned 0 bytes}"
    else
      PROBE_STATUS='FAIL_EMPTY'
      PROBE_ERR="${PROBE_ERR:-file is 0 bytes}"
    fi
    return 1
  fi
  PROBE_STATUS='OK'
  return 0
}

# ---------------------------------------------------------------------------
# Harness self-check: a positive control that MUST be OK and an absent
# sentinel that MUST be FAIL. Without this pair, "everything red" and
# "apparatus broken" are indistinguishable, and neither is readable.
# ---------------------------------------------------------------------------
POSCTRL_FILE="${TMP_ROOT}/POSCTRL.txt"
POSCTRL_PAYLOAD='mount-preflight positive control: 64 bytes of known content..'
ABSENT_FILE="/proc/self/__mount_preflight_absent_${RANDOM}${RANDOM}__/nope"

selfcheck() {
  local ok=1
  printf '%s\n' "${POSCTRL_PAYLOAD}" > "${POSCTRL_FILE}" 2>/dev/null || true
  probe_read "${POSCTRL_FILE}"
  log "SELFCHECK POSCTRL  expect=OK   got=${PROBE_STATUS} bytes=${PROBE_BYTES} stat_size=${PROBE_STAT_SIZE} read_rc=${PROBE_READ_RC} path=${POSCTRL_FILE}"
  if [[ "${PROBE_STATUS}" != 'OK' || "${PROBE_BYTES}" -le 0 ]]; then
    log_err "SELFCHECK POSCTRL FAILED -> the measuring apparatus cannot read a file it just wrote. NOTHING about the mounts was measured."
    ok=0
  fi

  probe_read "${ABSENT_FILE}"
  log "SELFCHECK ABSENT   expect=FAIL got=${PROBE_STATUS} bytes=${PROBE_BYTES} stat_size=${PROBE_STAT_SIZE} read_rc=${PROBE_READ_RC} path=${ABSENT_FILE}"
  if [[ "${PROBE_STATUS}" == 'OK' ]]; then
    log_err "SELFCHECK ABSENT FAILED -> a path that cannot exist reported OK. The comparison is inverted; this run is void."
    ok=0
  fi
  return $(( ok == 1 ? 0 : 1 ))
}

# ---------------------------------------------------------------------------
# Barrier configuration
# ---------------------------------------------------------------------------
RUN_KEY_SOURCE='MOUNT_PREFLIGHT_RUN_KEY'
RUN_KEY="${MOUNT_PREFLIGHT_RUN_KEY:-}"
BARRIER_ENABLED="${MOUNT_PREFLIGHT_BARRIER:-1}"

# A trial/task/job id is NOT an attempt id. Robust restart keeps the same trial,
# so a namespace derived from ARNOLD_TRIAL_ID can combine this attempt's rank0
# with still-fresh rank1-3 OK verdicts from the previous attempt. A shorter stale
# timeout does not close that race. The attempt token therefore has no fallback:
# the launcher/platform must inject one shared unique value on all ranks.
if [[ "${BARRIER_ENABLED}" == '1' && -z "${RUN_KEY}" ]]; then
  log_err 'FATAL: cross-rank barrier requires MOUNT_PREFLIGHT_RUN_KEY.'
  log_err 'It must be unique to this robust ATTEMPT and identical on all ranks; ARNOLD_TRIAL_ID is not sufficient.'
  printf '[%s] VERDICT rank=%s host=%s result=NOT_MEASURED probes_ok=0/%d failed=ATTEMPT_RUN_KEY_UNSET exit=%d run_key=- peers=-/- barrier=DISABLED\n' \
    "${PF_TAG}" "${RANK}" "${HOST_NAME}" "${#PROBES[@]}" "${RC_NOT_MEASURED}" >&2
  exit "${RC_NOT_MEASURED}"
fi
if [[ -z "${RUN_KEY}" ]]; then
  RUN_KEY='manual-local'; RUN_KEY_SOURCE='barrier-disabled'
fi

STATE_ROOT="${MOUNT_PREFLIGHT_STATE_ROOT:-${MOUNT_ROOT_HANDOFF}/mount_preflight}"
BARRIER_DIR="${MOUNT_PREFLIGHT_BARRIER_DIR:-${STATE_ROOT}/${RUN_KEY}}"
BARRIER_TIMEOUT_S="${MOUNT_PREFLIGHT_BARRIER_TIMEOUT_S:-300}"
BARRIER_POLL_S="${MOUNT_PREFLIGHT_BARRIER_POLL_S:-5}"
STALE_S="${MOUNT_PREFLIGHT_STALE_S:-900}"

NUM_RANKS="${MOUNT_PREFLIGHT_NUM_RANKS:-${ARNOLD_WORKER_NUM:-}}"
if [[ "${RANK}" == 'UNSET' && "${BARRIER_ENABLED}" == '1' ]]; then
  log_err "WARN: rank is UNSET (override in effect) -> cross-rank barrier force-disabled; this run proves nothing about peers."
  BARRIER_ENABLED=0
fi
if [[ "${BARRIER_ENABLED}" == '1' && -z "${NUM_RANKS}" ]]; then
  log_err "WARN: ARNOLD_WORKER_NUM unset and MOUNT_PREFLIGHT_NUM_RANKS not given -> cannot require a full quorum, so the cross-rank barrier is DISABLED. This rank's OK says NOTHING about its peers."
  BARRIER_ENABLED=0
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
log '============================================================'
log "MoMA-VLA per-rank mount preflight v${PF_VERSION}"
log "rank=${RANK} (source=${RANK_SOURCE})  host=${HOST_NAME}  pod=${POD_NAME}"
log "  NOTE: rank is read from ARNOLD_ID inside the pod. Pod index != rank"
log "        (measured: JOB1 executor-0=rank1; JOB2 executor-0=rank3)."
log "num_ranks=${NUM_RANKS:-<unset>}  barrier=${BARRIER_ENABLED}  run_key=${RUN_KEY} (source=${RUN_KEY_SOURCE})"
log "barrier_dir=${BARRIER_DIR}"
log "max_probe_bytes=${MAX_BYTES}  tmp_root=${TMP_ROOT}"
log '------------------------------------------------------------'

# ---------------------------------------------------------------------------
# Harness self-check FIRST -- before we let this run say anything about mounts
# ---------------------------------------------------------------------------
if ! selfcheck; then
  log_err 'HARNESS SELF-CHECK FAILED -> verdict is NOT_MEASURED (neither pass nor fail).'
  printf '[%s] VERDICT rank=%s host=%s result=NOT_MEASURED probes_ok=0/%d failed=HARNESS_SELFCHECK exit=%d run_key=%s peers=-/-\n' \
    "${PF_TAG}" "${RANK}" "${HOST_NAME}" "${#PROBES[@]}" "${RC_NOT_MEASURED}" "${RUN_KEY}" >&2
  exit "${RC_NOT_MEASURED}"
fi
log 'SELFCHECK result=PASS (positive control OK, absent sentinel FAIL)'
log '------------------------------------------------------------'

# ---------------------------------------------------------------------------
# The mounts this rank actually saw
# ---------------------------------------------------------------------------
log 'MOUNTS this rank actually saw (from /proc/self/mountinfo):'
declare -A _seen_root=()
for _rec in "${PROBES[@]}"; do
  _g="${_rec%%|*}"
  _rest="${_rec#*|}"
  _p="${_rest%%|*}"
  if [[ -z "${_seen_root[${_g}]:-}" ]]; then
    _mi="$(mount_of "${_p}")"
    log "  group=${_g} mountpoint=${_mi%% *} fstype=$(printf '%s' "${_mi}" | awk '{print $2}') source=$(printf '%s' "${_mi}" | awk '{print $3}')"
    _seen_root["${_g}"]=1
  fi
done
log '------------------------------------------------------------'

# ---------------------------------------------------------------------------
# Probe loop. NO early exit, NO `set -e`: one bad path must not suppress the
# per-probe OK of every good one.
# ---------------------------------------------------------------------------
TOTAL=${#PROBES[@]}
N_OK=0
FAILED_LIST=()
PROBE_REPORT=()

log "PROBES (content reads; a probe passes only if bytes_read > 0):"
for _rec in "${PROBES[@]}"; do
  _g="${_rec%%|*}"
  _rest="${_rec#*|}"
  _p="${_rest%%|*}"
  _why="${_rest#*|}"

  probe_read "${_p}"
  _st="${PROBE_STATUS}"
  _b="${PROBE_BYTES}"
  _ss="${PROBE_STAT_SIZE}"
  _rc="${PROBE_READ_RC}"
  _er="${PROBE_ERR}"

  if [[ "${_st}" == 'OK' ]]; then
    N_OK=$(( N_OK + 1 ))
    log "  [OK  ] group=${_g} bytes_read=${_b} stat_size=${_ss} read_rc=${_rc} path=${_p}"
  else
    FAILED_LIST+=("${_g}:${_p}(${_st})")
    log_err "  [FAIL] group=${_g} status=${_st} bytes_read=${_b} stat_size=${_ss} read_rc=${_rc} path=${_p} err='${_er}'"
    log_err "         why this file matters: ${_why}"
  fi
  PROBE_REPORT+=("$(printf '{"group":"%s","path":"%s","status":"%s","bytes_read":%s,"stat_size":%s,"read_rc":%s}' \
    "${_g}" "${_p}" "${_st}" "${_b}" "${_ss}" "${_rc}")")
done

LOCAL_RESULT='OK'
[[ "${N_OK}" -eq "${TOTAL}" ]] || LOCAL_RESULT='FAIL'
FAILED_JOINED='-'
if [[ "${#FAILED_LIST[@]}" -gt 0 ]]; then
  FAILED_JOINED="$(printf '%s,' "${FAILED_LIST[@]}")"
  FAILED_JOINED="${FAILED_JOINED%,}"
fi

log '------------------------------------------------------------'
log "LOCAL result=${LOCAL_RESULT} probes_ok=${N_OK}/${TOTAL} failed=${FAILED_JOINED}"

# ---------------------------------------------------------------------------
# Barrier: publish own verdict, then require every peer to be OK.
# ---------------------------------------------------------------------------
PEERS_OK=0
PEERS_EXPECTED="${NUM_RANKS:--}"
BARRIER_RESULT='DISABLED'

publish_verdict() {
  local result="$1"
  local now; now="$(date +%s)"
  local dst="${BARRIER_DIR}/rank${RANK}.verdict"
  local tmp="${BARRIER_DIR}/.rank${RANK}.$$.tmp"

  mkdir -p "${BARRIER_DIR}" 2>/dev/null || {
    log_err "PUBLISH: cannot create barrier dir ${BARRIER_DIR}"
    return 1
  }
  # A stale verdict from an earlier attempt must never be mistaken for ours.
  rm -f "${dst}" 2>/dev/null || true

  {
    printf 'VERDICT=%s\n' "${result}"
    printf 'rank=%s\n' "${RANK}"
    printf 'host=%s\n' "${HOST_NAME}"
    printf 'pod=%s\n' "${POD_NAME}"
    printf 'epoch=%s\n' "${now}"
    printf 'iso=%s\n' "$(ts)"
    printf 'probes_ok=%s/%s\n' "${N_OK}" "${TOTAL}"
    printf 'failed=%s\n' "${FAILED_JOINED}"
    printf 'preflight_version=%s\n' "${PF_VERSION}"
    printf 'probes_json=[%s]\n' "$(printf '%s,' "${PROBE_REPORT[@]}" | sed 's/,$//')"
  } > "${tmp}" 2>/dev/null || {
    log_err "PUBLISH: write to ${tmp} failed"
    return 1
  }

  mv -f "${tmp}" "${dst}" 2>/dev/null || {
    log_err "PUBLISH: mv ${tmp} -> ${dst} failed"
    rm -f "${tmp}" 2>/dev/null || true
    return 1
  }

  # READ IT BACK. `rc=0` from the write above proves nothing on this host --
  # it has produced zero-byte files from successful writes twice. Assert both
  # non-empty AND that the verdict token really survived.
  local back
  back="$(head -c 4096 -- "${dst}" 2>/dev/null)"
  if [[ -z "${back}" ]]; then
    log_err "PUBLISH: readback of ${dst} returned 0 bytes (write reported success) -- refusing to trust it"
    return 1
  fi
  case "${back}" in
    "VERDICT=${result}"*) : ;;
    *)
      log_err "PUBLISH: readback of ${dst} does not start with VERDICT=${result}"
      return 1
      ;;
  esac
  log "PUBLISH: ${dst} (VERDICT=${result}, readback $(printf '%s' "${back}" | wc -c) bytes)"
  return 0
}

# Reads BARRIER_DIR once. Sets _peer_ok_n, _peer_fail_list, _peer_missing_list.
scan_peers() {
  local now; now="$(date +%s)"
  _peer_ok_n=0
  _peer_fail_list=()
  _peer_missing_list=()
  local r f content v epoch age
  for (( r = 0; r < NUM_RANKS; r++ )); do
    f="${BARRIER_DIR}/rank${r}.verdict"
    if [[ ! -f "${f}" ]]; then
      _peer_missing_list+=("${r}(no-file)")
      continue
    fi
    content="$(head -c 8192 -- "${f}" 2>/dev/null)"
    if [[ -z "${content}" ]]; then
      # zero-byte verdict: NOT 'assume ok', NOT 'assume fail' -> MISSING
      _peer_missing_list+=("${r}(empty-file)")
      continue
    fi
    v="$(printf '%s\n' "${content}" | sed -n 's/^VERDICT=//p' | head -1)"
    epoch="$(printf '%s\n' "${content}" | sed -n 's/^epoch=//p' | head -1)"
    if [[ -z "${v}" || -z "${epoch}" || ! "${epoch}" =~ ^[0-9]+$ ]]; then
      _peer_missing_list+=("${r}(unparseable)")
      continue
    fi
    age=$(( now - epoch ))
    if [[ "${age}" -gt "${STALE_S}" ]]; then
      # A leftover OK from a previous attempt must not wave a broken node
      # through, so stale counts as MISSING rather than as OK.
      _peer_missing_list+=("${r}(stale-${age}s)")
      continue
    fi
    case "${v}" in
      OK)   _peer_ok_n=$(( _peer_ok_n + 1 )) ;;
      FAIL) _peer_fail_list+=("${r}") ;;
      *)    _peer_missing_list+=("${r}(verdict=${v})") ;;
    esac
  done
}

FINAL_RC="${RC_ALL_OK}"

if [[ "${BARRIER_ENABLED}" != '1' ]]; then
  log "BARRIER disabled -> this rank reports only on ITSELF. Peers unverified."
  BARRIER_RESULT='DISABLED'
  [[ "${LOCAL_RESULT}" == 'OK' ]] || FINAL_RC="${RC_LOCAL_FAIL}"
else
  log '------------------------------------------------------------'
  log "BARRIER: publishing verdict, then requiring ALL ${NUM_RANKS} ranks OK"
  if ! publish_verdict "${LOCAL_RESULT}"; then
    log_err "BARRIER: could not durably publish this rank's verdict -> aborting. Peers will see this rank as MISSING and abort too, which is the intended total failure."
    BARRIER_RESULT='PUBLISH_FAIL'
    FINAL_RC="${RC_PUBLISH_FAIL}"
  elif [[ "${LOCAL_RESULT}" != 'OK' ]]; then
    # Publish FAIL first (done above) so peers abort; then exit immediately.
    log_err "BARRIER: local FAIL published; peers will abort with ${RC_PEER_FAIL}. Not waiting."
    BARRIER_RESULT='LOCAL_FAIL_PUBLISHED'
    FINAL_RC="${RC_LOCAL_FAIL}"
  else
    _deadline=$(( $(date +%s) + BARRIER_TIMEOUT_S ))
    while : ; do
      scan_peers
      if [[ "${#_peer_fail_list[@]}" -gt 0 ]]; then
        log_err "BARRIER: peer rank(s) reported FAIL: ${_peer_fail_list[*]}"
        log_err "BARRIER: aborting THIS rank too. A mount missing on one node must fail the whole job -- a lone rank exiting presents as 'training hung', not as 'a mount was missing'."
        for _fr in "${_peer_fail_list[@]}"; do
          log_err "BARRIER:   rank${_fr} detail: $(sed -n 's/^failed=//p' "${BARRIER_DIR}/rank${_fr}.verdict" 2>/dev/null | head -1)"
          log_err "BARRIER:   rank${_fr} host:   $(sed -n 's/^host=//p' "${BARRIER_DIR}/rank${_fr}.verdict" 2>/dev/null | head -1)"
        done
        BARRIER_RESULT='PEER_FAIL'
        PEERS_OK="${_peer_ok_n}"
        FINAL_RC="${RC_PEER_FAIL}"
        break
      fi
      if [[ "${_peer_ok_n}" -ge "${NUM_RANKS}" ]]; then
        log "BARRIER: all ${NUM_RANKS}/${NUM_RANKS} ranks published OK"
        BARRIER_RESULT='ALL_OK'
        PEERS_OK="${_peer_ok_n}"
        break
      fi
      if [[ "$(date +%s)" -ge "${_deadline}" ]]; then
        log_err "BARRIER: TIMEOUT after ${BARRIER_TIMEOUT_S}s with ${_peer_ok_n}/${NUM_RANKS} OK"
        log_err "BARRIER: not accounted for: ${_peer_missing_list[*]:-<none>}"
        log_err "BARRIER: treating an incomplete quorum as failure ON PURPOSE. 'rank N never published' is a diagnosable message; letting the rest start and stall in a collective is not."
        BARRIER_RESULT='TIMEOUT'
        PEERS_OK="${_peer_ok_n}"
        FINAL_RC="${RC_BARRIER_TIMEOUT}"
        break
      fi
      log "BARRIER: ${_peer_ok_n}/${NUM_RANKS} OK so far; waiting ${BARRIER_POLL_S}s (missing: ${_peer_missing_list[*]:-<none>})"
      sleep "${BARRIER_POLL_S}"
    done
  fi
fi

# ---------------------------------------------------------------------------
# Single grep-able verdict line. Always printed, on stdout when the whole
# thing is good and on stderr otherwise.
#   [mount-preflight] VERDICT rank=<R> host=<H> result=<OK|FAIL|NOT_MEASURED>
#       probes_ok=<n>/<total> failed=<list|-> exit=<code> run_key=<K>
#       peers=<ok>/<expected> barrier=<state>
# ---------------------------------------------------------------------------
VERDICT_RESULT='OK'
[[ "${FINAL_RC}" -eq 0 ]] || VERDICT_RESULT='FAIL'
VERDICT_LINE="$(printf '[%s] VERDICT rank=%s host=%s result=%s probes_ok=%d/%d failed=%s exit=%d run_key=%s peers=%s/%s barrier=%s' \
  "${PF_TAG}" "${RANK}" "${HOST_NAME}" "${VERDICT_RESULT}" "${N_OK}" "${TOTAL}" \
  "${FAILED_JOINED}" "${FINAL_RC}" "${RUN_KEY}" "${PEERS_OK}" "${PEERS_EXPECTED}" "${BARRIER_RESULT}")"

log '============================================================'
if [[ "${FINAL_RC}" -eq 0 ]]; then
  printf '%s\n' "${VERDICT_LINE}"
else
  printf '%s\n' "${VERDICT_LINE}" >&2
fi
exit "${FINAL_RC}"
