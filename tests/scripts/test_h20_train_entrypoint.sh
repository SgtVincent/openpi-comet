#!/usr/bin/env bash
# ============================================================================
# Tests for scripts/h20_train_entrypoint.sh
#
# Scope: pure CPU, no GPU, no training, no occupier processes, no residue.
#
# WHY THESE ASSERTIONS ARE NOT COSMETIC
# Every property checked here corresponds to a defect that was measured, not
# imagined, while bringing this entrypoint up:
#
#   * `set -u` -> sourcing extra_bashrc.sh aborts at its line 33 (PROMPT_COMMAND
#     unbound in a non-interactive pod) BEFORE use_gpu/free_gpu are defined, this
#     script exits, and all 32 H20 are released.
#   * no LAUNCHING sentinel -> the 60 s supervisor restores occupancy during the
#     launcher's >450 s CPU preflight, and the launcher's own assert_no_occupiers
#     then kills the run.
#   * no watchdog -> if the supervisor dies, the allocation survives but
#     utilization sits at zero and the 30%/3-hour reaper takes all 32 cards.
#   * tag-only occ_count -> over-counts (measured 1 -> 2 with a live
#     `pkill -f "$GPU_OCCUPY_TAG"`, which free_gpu ends in), so the supervisor
#     concludes occupiers exist and never starts them. Same reaper outcome.
#   * any `exit` on a bad-input path -> releases 32 GPUs instead of holding them.
#
# Two families:
#   A. Static contract checks on the script text.
#   B. Behavioural checks of the extracted occ_count predicate against real
#      processes, including the free_gpu pkill case.
#
# Usage:
#   bash tests/scripts/test_h20_train_entrypoint.sh
# ============================================================================

set -uo pipefail

REPO_ROOT_REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENTRYPOINT_REL="scripts/h20_train_entrypoint.sh"
ENTRYPOINT="${REPO_ROOT_REAL}/${ENTRYPOINT_REL}"
LAUNCHER_REL="scripts/run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh"

PASS=0
FAIL=0
SKIP=0

ok()   { printf 'PASS  %s\n' "$*"; PASS=$((PASS + 1)); }
bad()  { printf 'FAIL  %s\n' "$*"; FAIL=$((FAIL + 1)); }
skip() { printf 'SKIP  %s\n' "$*"; SKIP=$((SKIP + 1)); }

have() { grep -qF -- "$1" "${ENTRYPOINT}"; }

printf '== A. static contract ==\n'

[[ -f "${ENTRYPOINT}" ]] && ok "entrypoint exists at ${ENTRYPOINT_REL}" \
  || { bad "entrypoint missing at ${ENTRYPOINT_REL}"; printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"; exit 1; }

bash -n "${ENTRYPOINT}" 2>/dev/null && ok "bash -n clean" || bad "bash -n failed"

# --- the set -u kill -------------------------------------------------------
if grep -qE '^set .*-[a-z]*u' "${ENTRYPOINT}"; then
  bad "script enables -u; sourcing extra_bashrc.sh will abort and RELEASE 32 GPUs"
else
  ok "-u absent (mandatory: -u kills the source of extra_bashrc.sh)"
fi
if grep -qE '^set .*-[a-z]*e' "${ENTRYPOINT}"; then
  bad "script enables -e; any unexpected non-zero exits and releases 32 GPUs"
else
  ok "-e absent"
fi
have 'set -o pipefail' && ok "pipefail enabled" || bad "pipefail missing"
have 'PROMPT_COMMAND="${PROMPT_COMMAND:-}"' \
  && ok "PROMPT_COMMAND guard present (belt-and-braces for the -u trap)" \
  || bad "PROMPT_COMMAND guard missing"
grep -q 'Do not "harmonise"' "${ENTRYPOINT}" \
  && ok "comment warns against harmonising strictness with tool scripts" \
  || bad "missing the rationale comment; a future cleanup will re-add -u"

# --- never exits -----------------------------------------------------------
# An `exit` anywhere at top level is a released allocation. Subshell exits are
# fine (the launcher subshell uses `exit 2` on a failed cd), so only flag
# unindented top-level ones.
if grep -nE '^exit ' "${ENTRYPOINT}"; then
  bad "top-level 'exit' found; this releases 32 GPUs"
else
  ok "no top-level exit (bad input must fall into keepalive, not exit)"
fi
have 'while true; do' && ok "terminal keepalive loop present" || bad "no keepalive loop"
[[ "$(tail -1 "${ENTRYPOINT}" | tr -d '[:space:]')" == "done" ]] \
  && ok "last line is the keepalive 'done' (paste truncation is detectable)" \
  || bad "last line is not 'done'; truncation would be silent"
have 'FATAL launcher not found' \
  && ok "missing launcher logs and falls through rather than exiting" \
  || bad "missing-launcher path does not fall through to keepalive"

# --- occupancy handoff -----------------------------------------------------
have ': > "$LAUNCHING"' && ok "launch sentinel is raised" || bad "no launch sentinel"
sent_line="$(grep -n ': > "\$LAUNCHING"' "${ENTRYPOINT}" | head -1 | cut -d: -f1)"
free_line="$(grep -n 'free_gpu >>"\$HB" 2>&1 || true' "${ENTRYPOINT}" | awk -F: -v s="${sent_line:-0}" '$1>s {print $1; exit}')"
if [[ -n "${sent_line}" && -n "${free_line}" && "${sent_line}" -lt "${free_line}" ]]; then
  ok "sentinel raised BEFORE the handoff free_gpu (line ${sent_line} < ${free_line})"
else
  bad "sentinel/handoff ordering wrong (sentinel=${sent_line:-none} handoff=${free_line:-none})"
fi
have 'PHASE="LAUNCHING"' && ok "supervisor has a LAUNCHING phase" || bad "no LAUNCHING phase"
have 'PHASE="TRAINING"' && ok "supervisor has a TRAINING phase" || bad "no TRAINING phase"
have 'PHASE="HOLDING"'  && ok "supervisor has a HOLDING phase"  || bad "no HOLDING phase"
# LAUNCHING must be tested first, otherwise HOLDING re-occupies mid-launch.
l_line="$(grep -n 'PHASE="LAUNCHING"' "${ENTRYPOINT}" | head -1 | cut -d: -f1)"
h_line="$(grep -n 'PHASE="HOLDING"' "${ENTRYPOINT}" | head -1 | cut -d: -f1)"
if [[ -n "$l_line" && -n "$h_line" && "$l_line" -lt "$h_line" ]]; then
  ok "LAUNCHING takes priority over HOLDING (line $l_line < $h_line)"
else
  bad "HOLDING is evaluated before LAUNCHING; supervisor will kill its own launch"
fi
have 'kill -0 "$SUPERVISOR_PID"' \
  && ok "supervisor watchdog present (it is the last line of defence)" \
  || bad "no watchdog; a dead supervisor means 0% util and a reaped job"
have 'run_arm${H20_ARM}' && ok "run dir is arm-scoped" || bad "run dir not arm-scoped"
have 'podfacts_' && ok "in-pod podfacts probe present (SSH/WebShell are broken)" \
  || bad "no podfacts probe"

# --- env-driven identity ---------------------------------------------------
have 'H20_ARM="${H20_ARM:-A}"'      && ok "H20_ARM from env, default A"    || bad "H20_ARM default wrong"
have 'H20_MODE="${H20_MODE:-smoke}"' && ok "H20_MODE from env, default smoke" || bad "H20_MODE default wrong"
if grep -qE '^OPENPI_EXPECTED_CODE_COMMIT=[0-9a-f]{40}' "${ENTRYPOINT}"; then
  bad "SHA is hardcoded; it must come from the job config so the two cannot drift"
else
  ok "SHA not hardcoded (comes from the environment, as the launcher expects)"
fi
# REPO_ROOT must be derived from the script's own location, which is the whole
# point of moving the entrypoint into the repo.
have 'BASH_SOURCE[0]' && ok "REPO_ROOT derived from the script's own location" \
  || bad "REPO_ROOT not self-derived; a moved worktree breaks the run"

# --- occ_count predicate ---------------------------------------------------
if grep -q 'gpu_occupy_(torch_mm\[.\]py|stub\[.\]sh)' "${ENTRYPOINT}"; then
  ok "occ_count uses the two-condition predicate (matches assert_no_occupiers)"
else
  bad "occ_count is tag-only; it over-counts and the supervisor then never occupies"
fi
if [[ -f "${REPO_ROOT_REAL}/${LAUNCHER_REL}" ]]; then
  grep -q 'gpu_occupy_(torch_mm\[.\]py|stub\[.\]sh)' "${REPO_ROOT_REAL}/${LAUNCHER_REL}" \
    && ok "launcher uses the same predicate (the two agree)" \
    || bad "launcher predicate differs from the supervisor's"
else
  skip "launcher not found; cannot cross-check the predicate"
fi

printf '\n== B. occ_count behaviour against real processes ==\n'

# Extract the real function and exercise it. Runs from a FILE so the harness's own
# argv cannot contain the unbracketed tag and self-match — a trap that produced a
# false positive during development.
OCC_FN="$(mktemp)"; trap 'rm -f "${OCC_FN}"' EXIT
sed -n '/^occ_count() {/,/^}/p' "${ENTRYPOINT}" > "${OCC_FN}"
if [[ ! -s "${OCC_FN}" ]]; then
  skip "could not extract occ_count; behavioural checks skipped"
else
  RUNNER="$(mktemp)"
  cat "${OCC_FN}" > "${RUNNER}"
  cat >> "${RUNNER}" <<'INNER'
echo "BASELINE=$(occ_count)"
# free_gpu ends in `pkill -f "$GPU_OCCUPY_TAG"`, so its argv carries the tag but
# NOT the occupier script name. A tag-only count reports this as an occupier.
bash -c 'exec -a "pkill -f __GPU_OCCUPY__torch_mm_512" sleep 4' >/dev/null 2>&1 &
P=$!; sleep 1
echo "WITH_PKILL=$(occ_count)"
kill $P 2>/dev/null; wait $P 2>/dev/null
# A genuine occupier carries BOTH the tag and the script name.
bash -c 'exec -a "python gpu_occupy_torch_mm.py __GPU_OCCUPY__torch_mm_512 0" sleep 4' >/dev/null 2>&1 &
R=$!; sleep 1
echo "WITH_REAL=$(occ_count)"
kill $R 2>/dev/null; wait $R 2>/dev/null
INNER
  OUT="$(bash "${RUNNER}" 2>/dev/null)"
  rm -f "${RUNNER}"
  base="$(sed -n 's/^BASELINE=//p' <<<"${OUT}")"
  pk="$(sed -n 's/^WITH_PKILL=//p' <<<"${OUT}")"
  rl="$(sed -n 's/^WITH_REAL=//p' <<<"${OUT}")"
  [[ "${base}" == "0" ]] && ok "baseline count is 0 (no self-match)" \
    || bad "baseline count is ${base}, expected 0"
  [[ "${pk}" == "0" ]] && ok "free_gpu's pkill is NOT counted (the real over-count path)" \
    || bad "pkill inflated the count to ${pk}; supervisor would refuse to occupy"
  [[ "${rl}" -ge 1 ]] 2>/dev/null && ok "a genuine occupier IS counted (${rl})" \
    || bad "genuine occupier not counted (got ${rl}); supervisor would double-start"
fi

printf '\n%d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
[[ "$FAIL" -eq 0 ]] || exit 1
