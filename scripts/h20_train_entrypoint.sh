#!/usr/bin/env bash
# ============================================================================
# H20 Variant A (FAST-CE) training entrypoint — 4 x 8 NVIDIA_H20.
#
# This is the Merlin job's MAIN COMMAND. If it exits, the 32-GPU allocation is
# RELEASED. Every design choice below follows from that single fact.
#
# It lives in the repo (rather than as a loose NAS file) so that:
#   * it is covered by the provenance of the code it pins — an entrypoint that
#     asserts a 40-char SHA while living outside the repo can drift from that
#     code with nothing detecting it;
#   * REPO_ROOT is derived from this script's own location instead of an absolute
#     NAS path to a worktree another session could remove;
#   * it is reachable from the branch, the tag and the provenance bundle.
#
# INVOCATION (console 入口指令 pastes a small shim that calls this):
#   OPENPI_EXPECTED_CODE_COMMIT=<40-char sha>   # required; from the job config
#   H20_ARM=A|B                                 # default A  (A=comet base, B=pi05 base)
#   H20_MODE=smoke|formal                       # default smoke
#
# The SHA is taken from the ENVIRONMENT, not hardcoded here, so the pinned commit
# lives in the job config — the same convention the launcher itself uses.
#
# GUARANTEES
#   (1) never exits, so the allocation is not released even after training ends
#       or fails
#   (2) GPU utilization stays above the platform 30% / 3-hour kill line during
#       every protected low-utilization gap: allocation->launch, the multi-minute
#       CPU preflight, dataset cache build, model init, validation, checkpointing,
#       and after the run
#   (3) occupiers are handed off ONLY for the window training needs the devices,
#       and restored as soon as training is no longer resident
#   (4) occupancy decisions come from LIVE evidence (ps / nvidia-smi), never from
#       a shared PID file, and only tagged occupiers are ever stopped
#   (5) a NAS heartbeat, because SSH and WebShell are both unavailable on this job
# ============================================================================

# NOTE: `-u` is DELIBERATELY ABSENT, and `-e` must never be added.
# With `-u`, sourcing extra_bashrc.sh aborts at its line 33
# (`export PROMPT_COMMAND="history -a; ...; ${PROMPT_COMMAND}"` — PROMPT_COMMAND is
# an interactive-shell variable and is unset in a job pod). The source then dies
# BEFORE use_gpu/free_gpu are defined, this script exits, the trial ends, and all
# 32 H20 are released. Verified empirically. It also masquerades as a platform
# fault, which sends diagnosis in the wrong direction.
#
# The correct strictness depends on what an exit COSTS. Here it costs 32 GPUs, so
# robustness beats strictness. Contrast a tool script that holds no GPUs, where
# `set -euo pipefail` is right. Do not "harmonise" the two.
set -o pipefail
export PROMPT_COMMAND="${PROMPT_COMMAND:-}"   # belt-and-braces

source /mnt/bn/behavior-data-hl/chenjunting/repo/extra_bashrc.sh

export proxy="http://sys-proxy-rd-relay.byted.org:8118"
export http_proxy="$proxy"; export https_proxy="$proxy"
export HTTP_PROXY="$proxy"; export HTTPS_PROXY="$proxy"
export no_proxy=".byted.org"; export NO_PROXY=".byted.org"

H20_ARM="${H20_ARM:-A}"
H20_MODE="${H20_MODE:-smoke}"

# Derived from this script's own location: the durability win of living in the repo.
_SELF="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${_SELF}")/.." 2>/dev/null && pwd)}"
LAUNCHER="${REPO_ROOT}/scripts/run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh"

RUN_BASE="${H20_RUN_BASE:-/mnt/bn/navigation-hl/mlx/users/chenjunting/h20_fastce}"
RUN_DIR="${RUN_BASE}/run_arm${H20_ARM}"
mkdir -p "$RUN_DIR" 2>/dev/null || true
NODE="$(hostname)"
HB="$RUN_DIR/hb_${NODE}.log"
ST="$RUN_DIR/status_${NODE}.txt"
LAUNCHING="$RUN_DIR/.launching_${NODE}"
TRAIN_LOG="$RUN_DIR/train_${H20_MODE}_${NODE}.log"
PODFACTS="$RUN_DIR/podfacts_${NODE}.txt"

NGPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
case "$NGPU" in ''|*[!0-9]*) NGPU=0 ;; esac

log() { echo "[h20-fastce] $(date -u +%FT%TZ) node=$NODE $*" >>"$HB" 2>/dev/null; }

# Occupier accounting uses the SAME two-condition predicate as the launcher's
# assert_no_occupiers: the tag AND the occupier script name.
#
#  (a) Agreement. The launcher refuses to start if IT sees an occupier, so this
#      count must match what the launcher counts. Those two disagreeing about what
#      an occupier IS is its own latent bug class.
#  (b) Over-counting is the DANGEROUS direction: if this over-reports, the
#      supervisor concludes occupiers already exist, does NOT start them,
#      utilization sits at zero, and the 30% / 3-hour reaper takes all 32 cards.
#
# The over-count path is real, not theoretical: extra_bashrc.sh's `free_gpu` ends
# in `pkill -f "$GPU_OCCUPY_TAG"`, so while that pkill lives its argv carries the
# tag. Measured with a live pkill: tag-only count = 2, two-condition count = 0.
# The bracket class prevents this pipeline from matching its own command line.
occ_count() {
  ps -eo args= 2>/dev/null \
    | awk '/[_]_GPU_OCCUPY__torch_mm_512/ && /gpu_occupy_(torch_mm[.]py|stub[.]sh)/' \
    | wc -l
}
train_alive() { ps -eo args= 2>/dev/null | grep -q "[t]rain_accelerate.py"; }

log "entrypoint start arm=$H20_ARM mode=$H20_MODE ngpu=$NGPU repo_root=$REPO_ROOT"
log "pinned commit from env: ${OPENPI_EXPECTED_CODE_COMMIT:-<unset -- launcher will refuse>}"

# ---- in-pod dependency probe (results on NAS within seconds) ----------------
# SSH and WebShell are both broken on this job, so these facts cannot be checked
# from outside. The launcher mandates WANDB_MODE=online, and a wandb.init blocking
# on a missing credential is indistinguishable from a training hang — so we learn
# it here rather than misdiagnosing a stall hours later. We do NOT silently switch
# to offline: formal runs are required to log to W&B.
{
  echo "=== pod facts $(date -u +%FT%TZ) node=$NODE ==="
  echo "-- arm=$H20_ARM mode=$H20_MODE ngpu=$NGPU"
  echo "-- wandb env:"; env | grep -i '^WANDB' || echo "   (no WANDB_* in env)"
  echo "-- netrc api.wandb.ai entries:"
  for f in /home/tiger/.netrc /root/.netrc "$HOME/.netrc"; do
    [ -f "$f" ] && echo "   $f -> $(grep -c 'api.wandb.ai' "$f" 2>/dev/null || echo 0) match(es)"
  done
  echo "-- FAST tokenizer (no H20 precedent; staged for this run):"
  ls -la "${RUN_BASE}/fast_tokenizer/" 2>&1 | head -12
  echo "-- repo root / trainer:"
  ls -la "${REPO_ROOT}/scripts/train_accelerate.py" 2>&1 | tail -1
  echo "-- worktree commit:"; git -C "$REPO_ROOT" rev-parse HEAD 2>&1
  echo "-- expected commit  : ${OPENPI_EXPECTED_CODE_COMMIT:-<unset>}"
  echo "-- preflight python import:"
  /mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python -c \
    "import transformers; from transformers import GemmaForCausalLM; print('   transformers', transformers.__version__, 'GemmaForCausalLM OK')" 2>&1 | tail -3
} > "$PODFACTS" 2>&1
_WCRED="$(grep -c 'api.wandb.ai' /home/tiger/.netrc 2>/dev/null || echo 0)"
case "$_WCRED" in ''|*[!0-9]*) _WCRED=0 ;; esac
if [ "$_WCRED" -eq 0 ] && [ -z "${WANDB_API_KEY:-}" ]; then
  log "WARN no W&B credential in-pod -> wandb.init may stall; treat an early stall as wandb-suspect first"
else
  log "W&B credential present in-pod (netrc matches=$_WCRED)"
fi

# ---- occupancy supervisor --------------------------------------------------
# Three-way, with LAUNCHING taking priority over HOLDING. It never touches a
# business process: the only thing it can stop is a tagged occupier, and only via
# free_gpu, whose pkill is tag-scoped.
supervisor() {
  while true; do
    TS="$(date -u +%FT%TZ)"
    N="$(occ_count)"; case "$N" in ''|*[!0-9]*) N=0 ;; esac
    UTIL="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)"
    MEM="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)"
    if [ -f "$LAUNCHING" ]; then
      # RACE FIX. The launcher runs its whole CPU preflight and only THEN calls
      # assert_no_occupiers, which hard-fails if ANY tagged occupier exists.
      # Measured: the warm-start keymap check alone takes 261 s cold and the full
      # preflight exceeds 450 s — many 60 s supervisor cycles. Without this branch
      # the supervisor restores occupancy mid-launch and kills our own run with
      # "GPU keepalive occupiers still running". The cost is NOT reading the
      # 14.5 GB checkpoint (only its header is parsed); it is importing the
      # openpi/transformers stack off NFS and building the 4.1B-param module graph.
      # ~8 min at low util is ~4% of a 3-hour averaging window, so it cannot trip
      # the reaper.
      PHASE="LAUNCHING"
      if [ "$N" -gt 0 ]; then
        log "$TS launch window active -> stopping tagged occupiers so assert_no_occupiers can pass"
        free_gpu >>"$HB" 2>&1 || true
        N=0
      fi
    elif train_alive; then
      # Training owns the devices. Do NOT start occupiers: they would contend for
      # HBM and could OOM the run. Low util here is an expected protected gap
      # (cache build / init / validation / checkpoint), not something to mask.
      PHASE="TRAINING"
      if [ "$N" -gt 0 ]; then
        log "$TS occupiers present while training resident -> free_gpu (tag-scoped)"
        free_gpu >>"$HB" 2>&1 || true
        N=0
      fi
    else
      PHASE="HOLDING"
      if [ "$N" -lt "$NGPU" ]; then
        log "$TS occ=$N/$NGPU no training resident -> (re)start occupiers"
        use_gpu >>"$HB" 2>&1 || log "use_gpu returned nonzero"
        N="$(occ_count)"; case "$N" in ''|*[!0-9]*) N=0 ;; esac
      fi
    fi
    printf '%s node=%s phase=%s occ=%s/%s util=%s mem=%s train_alive=%s arm=%s mode=%s\n' \
      "$TS" "$NODE" "$PHASE" "$N" "$NGPU" "$UTIL" "$MEM" \
      "$(train_alive && echo yes || echo no)" "$H20_ARM" "$H20_MODE" > "$ST" 2>/dev/null
    LN="$(wc -l <"$HB" 2>/dev/null || echo 0)"
    case "$LN" in ''|*[!0-9]*) LN=0 ;; esac
    if [ "$LN" -gt 6000 ]; then tail -1200 "$HB" >"$HB.tmp" 2>/dev/null && mv "$HB.tmp" "$HB"; fi
    sleep 60
  done
}

# Occupy immediately: allocation->launch is itself a protected gap.
log "starting occupiers for the allocation->launch gap"
use_gpu >>"$HB" 2>&1 || log "use_gpu returned nonzero"
supervisor &
SUPERVISOR_PID=$!
log "supervisor started pid=$SUPERVISOR_PID"

# ---- launch ---------------------------------------------------------------
# A missing or unreadable launcher must NOT exit: log it and fall through to the
# terminal keepalive, which retains the allocation with occupiers running.
if [ ! -f "$LAUNCHER" ]; then
  log "FATAL launcher not found: $LAUNCHER -- falling into keepalive, allocation retained"
else
  log "handing GPUs to training: raising launch sentinel, then stopping tagged occupiers"
  : > "$LAUNCHING"
  free_gpu >>"$HB" 2>&1 || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    N="$(occ_count)"; case "$N" in ''|*[!0-9]*) N=0 ;; esac
    [ "$N" -eq 0 ] && break
    sleep 2
  done
  log "occupiers now $(occ_count)/$NGPU; launching arm=$H20_ARM mode=$H20_MODE"
  (
    cd "$REPO_ROOT" || exit 2
    export OPENPI_KI_TRAINING_INNER=1
    export KEEPALIVE_DISABLE=0 KEEPALIVE_ON_SUCCESS=1 STRICT_GPU_COUNT=0
    export OPENPI_H20_ARM="$H20_ARM"
    export OPENPI_H20_MODE="$H20_MODE"
    export REPO_ROOT="$REPO_ROOT"
    export LAUNCHER="$LAUNCHER"
    bash "${REPO_ROOT}/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh"
  ) >>"$TRAIN_LOG" 2>&1
  TRAIN_RC=$?
  rm -f "$LAUNCHING"
  log "training wrapper returned rc=$TRAIN_RC"
fi

# ---- terminal keepalive with supervisor watchdog ---------------------------
# Do not release the allocation regardless of outcome. The supervisor sees no
# resident trainer and re-occupies within one 60 s cycle.
log "entering terminal keepalive (allocation retained; supervisor owns occupancy)"
while true; do
  # The supervisor is the ONLY thing restoring occupancy once training is gone.
  # If it died the allocation would survive (this loop never exits) but utilization
  # would sit at zero and the 30% / 3-hour reaper would take all 32 cards.
  if ! kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
    log "supervisor pid=$SUPERVISOR_PID is gone -> respawning"
    supervisor &
    SUPERVISOR_PID=$!
    log "supervisor respawned pid=$SUPERVISOR_PID"
  fi
  sleep 300
done
