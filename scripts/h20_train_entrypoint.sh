#!/usr/bin/env bash
# ============================================================================
# H20 Variant A (FAST-CE) training entrypoint — 4 x 8 NVIDIA_H20.
#
# Merlin job main command. Follows the house pattern proven by job
# 6fc66189eb6c5c88: set env, then call a repo script. Nothing more.
#
#   H20_ARM=A|B                       default A   (A = comet base, B = pi05 base)
#   H20_MODE=smoke|formal             default smoke
#   OPENPI_EXPECTED_CODE_COMMIT=<sha> required; supplied by the job config
#
# WHY THIS IS SHORT
# An earlier version added a 60 s occupancy supervisor. That supervisor raced the
# launcher's assert_no_occupiers, which needed a launch sentinel, which needed
# three-way phase logic, which needed a watchdog because the supervisor had become
# a single point of failure. Three of those four existed only to manage the first.
# The supervisor is gone and they went with it.
#
# The actual requirement — "if training fails, fall back to the matmul occupiers so
# the cards are not released" — is already met by
# run_pi05_skillbridge_lq_keepalive_on_failure.sh, which with KEEPALIVE_ON_SUCCESS=1
# holds the allocation after success or failure, restarts dead occupiers
# (OCCUPIER_AUTO_RESTART=1) and never returns. That is the keepalive, already
# written and already proven. Do not reimplement it here.
#
# The one genuine difference from a clean-node start: this job is converted from a
# hold job that currently has 32 tagged occupiers running, and the launcher
# hard-fails via assert_no_occupiers. Hence one `free_gpu` before launching.
#
# Per-arm paths, the norm_stats digest assertion, B8/GA1, stride 12 and offsets
# 0,4,8 are NOT re-exported here: the launcher already derives and asserts them
# from OPENPI_H20_ARM, and train_accelerate.py setdefaults the anchor vars and then
# validates them. Duplicating a contract that is already asserted downstream just
# creates two places to drift.
# ============================================================================

# `-u` is DELIBERATELY ABSENT and `-e` must never be added. With `-u`, sourcing
# extra_bashrc.sh aborts at its line 33 (PROMPT_COMMAND is an interactive-shell
# variable, unset in a job pod) BEFORE use_gpu/free_gpu are defined; this script
# then exits, the trial ends, and all 32 H20 are released. Verified empirically,
# and it masquerades as a platform fault. An exit here costs 32 GPUs, so
# robustness beats strictness.
set -o pipefail
export PROMPT_COMMAND="${PROMPT_COMMAND:-}"

source /mnt/bn/behavior-data-hl/chenjunting/repo/extra_bashrc.sh

export proxy="http://sys-proxy-rd-relay.byted.org:8118"
export http_proxy="$proxy"; export https_proxy="$proxy"
export HTTP_PROXY="$proxy"; export HTTPS_PROXY="$proxy"
export no_proxy=".byted.org"; export NO_PROXY=".byted.org"

export H20_ARM="${H20_ARM:-A}"
export H20_MODE="${H20_MODE:-smoke}"
export OPENPI_H20_ARM="$H20_ARM"
export OPENPI_H20_MODE="$H20_MODE"

_SELF="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${_SELF}")/.." 2>/dev/null && pwd)}"
export LAUNCHER="${REPO_ROOT}/scripts/run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh"
WRAPPER="${REPO_ROOT}/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh"

RUN_DIR="${H20_RUN_BASE:-/mnt/bn/navigation-hl/mlx/users/chenjunting/h20_fastce}/run_arm${H20_ARM}"
mkdir -p "$RUN_DIR" 2>/dev/null || true
LOG="$RUN_DIR/entrypoint_$(hostname).log"
echo "[h20] $(date -u +%FT%TZ) arm=$H20_ARM mode=$H20_MODE commit=${OPENPI_EXPECTED_CODE_COMMIT:-<unset>} repo=$REPO_ROOT" >>"$LOG" 2>&1

# ---- in-pod probe: answers two unknowns we cannot check any other way -------
# SSH and WebShell are both unavailable on this job family, so these facts are
# unobtainable from outside. The launcher mandates WANDB_MODE=online, and a
# wandb.init blocking on a missing credential is indistinguishable from a training
# hang — this way we learn it in seconds instead of misreading a stall. The FAST
# tokenizer is the one dependency with no H20 precedent. We do NOT switch to
# offline: formal runs are required to log to W&B.
{
  echo "=== pod facts $(date -u +%FT%TZ) $(hostname) arm=$H20_ARM mode=$H20_MODE ==="
  env | grep -i '^WANDB' || echo "(no WANDB_* in env)"
  for f in /home/tiger/.netrc /root/.netrc; do
    [ -f "$f" ] && echo "$f -> $(grep -c 'api.wandb.ai' "$f" 2>/dev/null || echo 0) api.wandb.ai match(es)"
  done
  ls -la "${H20_RUN_BASE:-/mnt/bn/navigation-hl/mlx/users/chenjunting/h20_fastce}/fast_tokenizer/" 2>&1 | head -8
  echo "worktree HEAD: $(git -C "$REPO_ROOT" rev-parse HEAD 2>&1)"
  echo "expected     : ${OPENPI_EXPECTED_CODE_COMMIT:-<unset>}"
  /mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python -c \
    "import transformers; from transformers import GemmaForCausalLM; print('transformers', transformers.__version__, 'GemmaForCausalLM OK')" 2>&1 | tail -2
} > "$RUN_DIR/podfacts_$(hostname).txt" 2>&1

# ---- provenance check: LOUD on mismatch ------------------------------------
# This cannot exit (an exit releases 32 H20), so logging is its only channel —
# which means a mismatch must be unmistakable rather than a neutral line that looks
# like the match case. It is also written to a dedicated status file so a monitor
# can see it without parsing the log.
#
# Note what this does and does not prove: it compares HEAD to the pinned SHA. It
# does NOT prove the working tree is clean, so a pin can be "correct" about HEAD
# while uncommitted edits are what actually execute. The launcher's own clean-tree
# gate is what closes that, and it runs before training.
_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
_DIRTY="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all 2>/dev/null | wc -l)"
PROV="$RUN_DIR/provenance_$(hostname).txt"
if [ "${OPENPI_EXPECTED_CODE_COMMIT:-}" = "" ]; then
  echo "PROVENANCE=UNSET head=$_HEAD dirty=$_DIRTY" > "$PROV" 2>&1
  echo "[h20] !!! WARN OPENPI_EXPECTED_CODE_COMMIT IS UNSET -- the launcher will refuse to start" >>"$LOG" 2>&1
elif [ "$_HEAD" != "$OPENPI_EXPECTED_CODE_COMMIT" ]; then
  echo "PROVENANCE=MISMATCH head=$_HEAD expected=$OPENPI_EXPECTED_CODE_COMMIT dirty=$_DIRTY" > "$PROV" 2>&1
  echo "[h20] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >>"$LOG" 2>&1
  echo "[h20] !!! PROVENANCE MISMATCH -- THE PINNED COMMIT IS NOT CHECKED OUT" >>"$LOG" 2>&1
  echo "[h20] !!!   expected $OPENPI_EXPECTED_CODE_COMMIT" >>"$LOG" 2>&1
  echo "[h20] !!!   HEAD     $_HEAD" >>"$LOG" 2>&1
  echo "[h20] !!! The launcher will refuse to start. Cards are retained by keepalive." >>"$LOG" 2>&1
  echo "[h20] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >>"$LOG" 2>&1
elif [ "$_DIRTY" -ne 0 ]; then
  echo "PROVENANCE=DIRTY head=$_HEAD dirty=$_DIRTY" > "$PROV" 2>&1
  echo "[h20] !!! WARN commit matches but the worktree has $_DIRTY uncommitted change(s):" >>"$LOG" 2>&1
  echo "[h20] !!!   the pin would be true about HEAD and false about the code that runs." >>"$LOG" 2>&1
  echo "[h20] !!!   The launcher's clean-tree gate will refuse to start." >>"$LOG" 2>&1
else
  echo "PROVENANCE=OK head=$_HEAD dirty=0" > "$PROV" 2>&1
  echo "[h20] provenance OK: HEAD == pinned commit, worktree clean" >>"$LOG" 2>&1
fi

# ---- hand the GPUs to training --------------------------------------------
# This job is converted from a hold job with 32 tagged occupiers running, and the
# launcher's assert_no_occupiers refuses to start while any exist. free_gpu is
# tag-scoped (its pkill matches only the occupier tag) so it cannot touch a
# business process. This is the only occupancy action the entrypoint takes; the
# wrapper owns occupancy from here on.
echo "[h20] stopping tagged occupiers so assert_no_occupiers can pass" >>"$LOG" 2>&1
free_gpu >>"$LOG" 2>&1 || true
sleep 5

# ---- launch: the wrapper owns keepalive and never returns ------------------
export OPENPI_KI_TRAINING_INNER=1
export KEEPALIVE_DISABLE=0 KEEPALIVE_ON_SUCCESS=1 STRICT_GPU_COUNT=0
echo "[h20] launching via $WRAPPER (KEEPALIVE_ON_SUCCESS=1)" >>"$LOG" 2>&1
bash "$WRAPPER" >>"$LOG" 2>&1
echo "[h20] wrapper returned rc=$? -- it normally never does; holding anyway" >>"$LOG" 2>&1

# ---- defensive tail -------------------------------------------------------
# The wrapper is not supposed to return. If it ever does, neither exit (that
# releases 32 H20) nor idle at 0% utilization (the 30% / 3-hour reaper would then
# take them). Re-occupy and hold.
use_gpu >>"$LOG" 2>&1 || true
while true; do sleep 300; done
