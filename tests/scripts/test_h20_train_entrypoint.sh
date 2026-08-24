#!/usr/bin/env bash
# ============================================================================
# Tests for scripts/h20_train_entrypoint.sh
#
# Scope: pure CPU, no GPU, no training, no occupier processes, no residue.
#
# The entrypoint is deliberately MINIMAL: set env, free_gpu, call the proven
# keepalive wrapper. These tests protect two things:
#
#   1. The properties whose absence costs 32 H20 — each corresponds to a defect
#      that was measured, not imagined.
#   2. That the removed complexity STAYS removed. An earlier version added a 60 s
#      occupancy supervisor; it raced the launcher's assert_no_occupiers, which
#      needed a launch sentinel, which needed three-way phase logic, which needed a
#      watchdog because the supervisor had become a single point of failure. Three
#      of those four existed only to manage the first. The wrapper already provides
#      keepalive-on-failure plus OCCUPIER_AUTO_RESTART, so none of it is needed.
#
# Usage:
#   bash tests/scripts/test_h20_train_entrypoint.sh
# ============================================================================

set -uo pipefail

REPO_ROOT_REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENTRYPOINT_REL="scripts/h20_train_entrypoint.sh"
ENTRYPOINT="${REPO_ROOT_REAL}/${ENTRYPOINT_REL}"
WRAPPER_REL="scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh"
LAUNCHER_REL="scripts/run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh"

PASS=0
FAIL=0
SKIP=0

ok()   { printf 'PASS  %s\n' "$*"; PASS=$((PASS + 1)); }
bad()  { printf 'FAIL  %s\n' "$*"; FAIL=$((FAIL + 1)); }
skip() { printf 'SKIP  %s\n' "$*"; SKIP=$((SKIP + 1)); }
have() { grep -qF -- "$1" "${ENTRYPOINT}"; }

printf '== existence ==\n'
if [[ -f "${ENTRYPOINT}" ]]; then
  ok "entrypoint exists at ${ENTRYPOINT_REL}"
else
  bad "entrypoint missing at ${ENTRYPOINT_REL}"
  printf '\n%d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
  exit 1
fi
bash -n "${ENTRYPOINT}" 2>/dev/null && ok "bash -n clean" || bad "bash -n failed"

printf '\n== an exit costs 32 H20 ==\n'
if grep -qE '^set .*-[a-z]*u' "${ENTRYPOINT}"; then
  bad "-u enabled; sourcing extra_bashrc.sh aborts at its line 33 on unbound PROMPT_COMMAND, before use_gpu/free_gpu exist, and RELEASES 32 GPUs"
else
  ok "-u absent (its presence kills the source of extra_bashrc.sh)"
fi
if grep -qE '^set .*-[a-z]*e' "${ENTRYPOINT}"; then
  bad "-e enabled; any unexpected non-zero exits and releases 32 GPUs"
else
  ok "-e absent"
fi
have 'set -o pipefail' && ok "pipefail enabled" || bad "pipefail missing"
have 'PROMPT_COMMAND="${PROMPT_COMMAND:-}"' \
  && ok "PROMPT_COMMAND guard present" || bad "PROMPT_COMMAND guard missing"
grep -q 'robustness beats strictness' "${ENTRYPOINT}" \
  && ok "rationale comment present (a future cleanup would otherwise re-add -u)" \
  || bad "no rationale comment explaining why -u/-e are absent"
if grep -nE '^exit ' "${ENTRYPOINT}"; then
  bad "top-level 'exit' found; this releases 32 GPUs"
else
  ok "no top-level exit"
fi
# The defensive tail must be the final construct, so paste truncation is visible.
last="$(tail -1 "${ENTRYPOINT}")"
[[ "${last}" == *done ]] \
  && ok "last line ends with 'done' (truncation is detectable): ${last}" \
  || bad "last line does not end with 'done': ${last}"
have 'while true; do sleep 300; done' \
  && ok "defensive infinite hold present" || bad "no defensive hold"

printf '\n== delegation, not reimplementation ==\n'
have 'free_gpu' \
  && ok "free_gpu present — the one genuine difference from a clean-node start" \
  || bad "no free_gpu; the launcher's assert_no_occupiers will refuse to start"
have 'KEEPALIVE_ON_SUCCESS=1' \
  && ok "wrapper asked to hold on success as well as failure" \
  || bad "wrapper not asked to hold on success"
have 'run_pi05_skillbridge_lq_keepalive_on_failure.sh' \
  && ok "delegates keepalive to the proven wrapper" \
  || bad "does not call the keepalive wrapper"
have 'use_gpu' \
  && ok "re-occupies if the wrapper ever returns (never idle at 0% util)" \
  || bad "no defensive re-occupy; a returning wrapper would be reaped at 0% util"

printf '\n== removed complexity stays removed ==\n'
for removed in 'supervisor()' 'LAUNCHING' 'kill -0' 'occ_count()'; do
  if grep -qF -- "${removed}" "${ENTRYPOINT}"; then
    bad "re-introduced '${removed}'; the wrapper already owns keepalive + OCCUPIER_AUTO_RESTART"
  else
    ok "'${removed}' stays removed"
  fi
done
# Coarse proxy for "did the subsystem come back". The precise guard is the
# removed-machinery loop above; this only catches wholesale regrowth.
# Baseline 144 lines: 110 for the simplified house-pattern entrypoint plus ~34 for
# the loud provenance check (a deliberate, reviewed addition, not creep). Raise this
# only for a reviewed addition — and if you are raising it because a supervisor came
# back, do not raise it.
lines="$(wc -l < "${ENTRYPOINT}")"
if [[ "${lines}" -le 160 ]]; then
  ok "entrypoint is ${lines} lines (house pattern: env, free_gpu, call a repo script)"
else
  bad "entrypoint has grown to ${lines} lines; a subsystem has probably crept back"
fi

printf '\n== identity comes from the environment ==\n'
have 'H20_ARM="${H20_ARM:-A}"'       && ok "H20_ARM from env, default A"     || bad "H20_ARM default wrong"
have 'H20_MODE="${H20_MODE:-smoke}"' && ok "H20_MODE from env, default smoke" || bad "H20_MODE default wrong"
if grep -qE '^(export )?OPENPI_EXPECTED_CODE_COMMIT=[0-9a-f]{40}' "${ENTRYPOINT}"; then
  bad "SHA hardcoded in the script; it must come from the job config so the two cannot drift"
else
  ok "SHA not hardcoded (supplied by the job config)"
fi
have 'BASH_SOURCE[0]' \
  && ok "REPO_ROOT derived from the script's own location (the point of living in the repo)" \
  || bad "REPO_ROOT not self-derived; a moved worktree would break the run"
have 'podfacts_' \
  && ok "in-pod podfacts probe present (SSH and WebShell are both unavailable here)" \
  || bad "no podfacts probe; the W&B credential is unobtainable any other way"

printf '\n== provenance check is loud, and covers the dirty case ==\n'
# It cannot exit (that costs 32 GPUs), so the log is its only channel and a
# mismatch must not look like a match. It must also distinguish DIRTY: a pin can be
# true about HEAD while uncommitted edits are what actually execute — exactly the
# false-assurance that nearly shipped.
have 'PROVENANCE=MISMATCH' && ok "emits PROVENANCE=MISMATCH" || bad "no MISMATCH state"
have 'PROVENANCE=DIRTY'    && ok "emits PROVENANCE=DIRTY (pin true about HEAD, false about the code)" || bad "no DIRTY state"
have 'PROVENANCE=UNSET'    && ok "emits PROVENANCE=UNSET" || bad "no UNSET state"
have 'PROVENANCE=OK'       && ok "emits PROVENANCE=OK" || bad "no OK state"
grep -q 'PROVENANCE MISMATCH -- THE PINNED COMMIT IS NOT CHECKED OUT' "${ENTRYPOINT}" \
  && ok "mismatch is unmistakable in the log, not a neutral line" \
  || bad "mismatch is not visually distinct from a match"
have 'provenance_$(hostname).txt' \
  && ok "writes a dedicated status file (monitor need not parse the log)" \
  || bad "no machine-readable provenance status file"
have 'status --porcelain --untracked-files=all' \
  && ok "checks worktree cleanliness, not just HEAD" \
  || bad "does not detect the dirty case"

printf '\n== the contract is asserted downstream, not duplicated here ==\n'
# Re-exporting B8/GA1/stride/offsets would be inert: the launcher defaults and
# asserts the batch contract, and train_accelerate.py setdefaults the anchor vars
# then validates them. Duplicating it just creates two places to drift.
for dup in 'BATCH_SIZE_PER_GPU=' 'GRADIENT_ACCUMULATION_STEPS=' 'FRAME_ANCHOR_STRIDE=' 'NUM_TRAIN_STEPS='; do
  if grep -qE "^(export )?${dup}" "${ENTRYPOINT}"; then
    bad "duplicates '${dup}' which is already asserted downstream — two places to drift"
  else
    ok "'${dup}' not duplicated (asserted by the launcher / trainer)"
  fi
done
if [[ -f "${REPO_ROOT_REAL}/${WRAPPER_REL}" ]]; then
  grep -q 'OCCUPIER_AUTO_RESTART' "${REPO_ROOT_REAL}/${WRAPPER_REL}" \
    && ok "wrapper really does restart dead occupiers (so no supervisor is needed)" \
    || bad "wrapper lacks OCCUPIER_AUTO_RESTART; delegation assumption is wrong"
else
  skip "wrapper not found; cannot verify the delegation assumption"
fi
if [[ -f "${REPO_ROOT_REAL}/${LAUNCHER_REL}" ]]; then
  grep -q 'assert_no_occupiers' "${REPO_ROOT_REAL}/${LAUNCHER_REL}" \
    && ok "launcher does gate on occupiers (so free_gpu is genuinely required)" \
    || bad "launcher has no occupier gate; free_gpu may be unnecessary"
else
  skip "launcher not found; cannot verify the free_gpu requirement"
fi

printf '\n%d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
[[ "$FAIL" -eq 0 ]] || exit 1
