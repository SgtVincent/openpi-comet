#!/usr/bin/env bash
# Generate one shared, unique key per robust attempt without mixing generations.
#
# Every rank publishes a fresh random request. Rank 0 waits until every request
# differs from the value consumed by the previous attempt, then publishes a
# manifest containing the new attempt key and every request. A peer accepts the
# manifest only if it contains its own exact request. Therefore a peer starting
# before rank 0 cannot consume the previous attempt's current file, even when it
# is seconds old.
set -uo pipefail

TAG=attempt-key
RANK="${ARNOLD_ID-}"
N="${ARNOLD_WORKER_NUM-}"
TRIAL="${ARNOLD_TRIAL_ID-}"
ROOT="${MOUNT_PREFLIGHT_STATE_ROOT:-/mnt/bn/behavior-data-hl/chenjunting/repo/moma_handoff_20260907/mount_preflight}"
TIMEOUT="${MOUNT_ATTEMPT_KEY_TIMEOUT_S:-300}"
POLL="${MOUNT_ATTEMPT_KEY_POLL_S:-1}"

fail() { printf '[%s] FATAL: %s\n' "$TAG" "$*" >&2; exit 16; }
[[ "$RANK" =~ ^[0-9]+$ ]] || fail "ARNOLD_ID must be numeric, got ${RANK:-<unset>}"
[[ "$N" =~ ^[1-9][0-9]*$ ]] || fail "ARNOLD_WORKER_NUM must be positive, got ${N:-<unset>}"
[[ "$TRIAL" =~ ^[A-Za-z0-9._-]+$ ]] || fail "ARNOLD_TRIAL_ID is required and must be path-safe"
(( RANK < N )) || fail "rank ${RANK} is outside worker count ${N}"

DIR="${ROOT}/attempt_handshake/trial-${TRIAL}"
REQ_DIR="${DIR}/requests"
USED_DIR="${DIR}/consumed"
CURRENT="${DIR}/current.manifest"
mkdir -p "$REQ_DIR" "$USED_DIR" || fail "cannot create shared handshake dir ${DIR}"

uuid="$(cat /proc/sys/kernel/random/uuid 2>/dev/null)" || fail "cannot read kernel UUID"
[[ -n "$uuid" ]] || fail "empty UUID"
request="rank${RANK}-${uuid}"
tmp="${REQ_DIR}/rank${RANK}.tmp.$$"
printf '%s\n' "$request" > "$tmp" || fail "cannot write request temp"
[[ -s "$tmp" ]] || fail "request temp is empty"
mv -f "$tmp" "${REQ_DIR}/rank${RANK}" || fail "cannot publish request"

start="$(date +%s)"
if [[ "$RANK" == 0 ]]; then
  while :; do
    ready=1
    requests=()
    for ((r=0; r<N; r++)); do
      f="${REQ_DIR}/rank${r}"
      [[ -s "$f" ]] || { ready=0; break; }
      q="$(sed -n '1p' "$f")"
      [[ "$q" == rank${r}-* ]] || { ready=0; break; }
      old=""
      [[ -s "${USED_DIR}/rank${r}" ]] && old="$(sed -n '1p' "${USED_DIR}/rank${r}")"
      [[ "$q" != "$old" ]] || { ready=0; break; }
      requests+=("$q")
    done
    (( ready == 1 )) && break
    (( $(date +%s) - start < TIMEOUT )) || fail "timeout waiting for fresh requests from ${N} ranks"
    sleep "$POLL"
  done

  attempt="trial-${TRIAL}-$(date +%s%N)-$(cat /proc/sys/kernel/random/uuid)"
  manifest_tmp="${CURRENT}.tmp.$$"
  {
    printf 'attempt=%s\n' "$attempt"
    for ((r=0; r<N; r++)); do printf 'rank=%s request=%s\n' "$r" "${requests[$r]}"; done
  } > "$manifest_tmp" || fail "cannot write manifest"
  [[ -s "$manifest_tmp" ]] || fail "manifest temp is empty"
  mv -f "$manifest_tmp" "$CURRENT" || fail "cannot publish manifest"
  for ((r=0; r<N; r++)); do
    printf '%s\n' "${requests[$r]}" > "${USED_DIR}/rank${r}.tmp.$$" || fail "cannot write consumed request"
    mv -f "${USED_DIR}/rank${r}.tmp.$$" "${USED_DIR}/rank${r}" || fail "cannot publish consumed request"
  done
fi

while :; do
  if [[ -s "$CURRENT" ]] && awk -v r="$RANK" -v q="$request" '$1=="rank="r && $2=="request="q {ok=1} END{exit !ok}' "$CURRENT"; then
    attempt="$(awk -F= '$1=="attempt"{print $2; exit}' "$CURRENT")"
    [[ "$attempt" =~ ^trial-${TRIAL}- ]] || fail "manifest has invalid attempt token"
    printf '%s\n' "$attempt"
    exit 0
  fi
  (( $(date +%s) - start < TIMEOUT )) || fail "timeout waiting for manifest containing this rank request"
  sleep "$POLL"
done
