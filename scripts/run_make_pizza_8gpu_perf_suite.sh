#!/usr/bin/env bash
set -Eeuo pipefail

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "[Warn] $*" >&2; }
die() { echo "[Error] $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUNTIME_ENV_FILE="${HOME}/.openpi_runtime_env.sh"
if [[ -f "${RUNTIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV_FILE}"
fi

CONDA_SH="${CONDA_SH:-/mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate openpi-comet-nas || warn "failed to activate openpi-comet-nas"
else
  warn "CONDA_SH not found: ${CONDA_SH}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
[[ -x "${PYTHON_BIN}" ]] || die "python not found"

PHASE="${1:-all}"
case "${PHASE}" in
  all|phase_a_smoke_filter|phase_b_2gpu_screen|phase_c_8gpu_confirm) ;;
  *)
    die "unsupported phase: ${PHASE}"
    ;;
esac

TASK_NAME="${TASK_NAME:-make_pizza}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K}"
OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft}"
CKPT_DIR="${CKPT_DIR:-/mnt/bn/behavior-data-hl/chenjunting/checkpoints/pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_005748}"
ENV_WRAPPER_RGB="omnigibson.learning.wrappers.RGBWrapper"
ENV_WRAPPER_LOWRES="omnigibson.learning.wrappers.RGBLowResWrapper"

CPU_AFFINITY_CPUS_PER_WORKER="${CPU_AFFINITY_CPUS_PER_WORKER:-12}"
DEFAULT_WARMUP_CACHE_TAR="${DEFAULT_WARMUP_CACHE_TAR:-/mnt/bn/navigation-hl/mlx/users/chenjunting/behavior1k_cache_all_gpus.tar.gz}"
WARMUP_CACHE_TAR="${WARMUP_CACHE_TAR:-$DEFAULT_WARMUP_CACHE_TAR}"
PHASE_A_GPU_IDS="${PHASE_A_GPU_IDS:-0}"
PHASE_B_GPU_IDS="${PHASE_B_GPU_IDS:-0,1}"
PHASE_C_GPU_IDS="${PHASE_C_GPU_IDS:-0,1,2,3,4,5,6,7}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_TAG="${SUITE_TAG:-make_pizza_perf_suite_${TIMESTAMP}}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/eval_logs/${SUITE_TAG}}"
mkdir -p "${SUITE_DIR}"
MANIFEST_PATH="${SUITE_DIR}/experiments_manifest.tsv"
SUMMARY_SCRIPT="${REPO_ROOT}/scripts/summarize_make_pizza_perf_suite.py"
LAUNCHER="${REPO_ROOT}/scripts/run_b1k_eval_parallel_single_task_headless.sh"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  printf 'phase\tgroup\trun_dir\tlaunch_epoch_s\tend_epoch_s\texit_code\tstatus\tnotes\n' > "${MANIFEST_PATH}"
fi

tsv_escape() {
  local text="${1:-}"
  text="${text//$'\t'/ }"
  text="${text//$'\n'/ }"
  printf '%s' "${text}"
}

append_manifest_row() {
  local phase="$1"
  local group="$2"
  local run_dir="$3"
  local launch_epoch_s="$4"
  local end_epoch_s="$5"
  local exit_code="$6"
  local status="$7"
  local notes="$8"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(tsv_escape "${phase}")" \
    "$(tsv_escape "${group}")" \
    "$(tsv_escape "${run_dir}")" \
    "$(tsv_escape "${launch_epoch_s}")" \
    "$(tsv_escape "${end_epoch_s}")" \
    "$(tsv_escape "${exit_code}")" \
    "$(tsv_escape "${status}")" \
    "$(tsv_escape "${notes}")" >> "${MANIFEST_PATH}"
}

reset_group_env() {
  export GPU_IDS=""
  export NUM_GPUS=""
  export EVAL_INSTANCE_IDS=""
  export MAX_STEPS=""
  export WRITE_VIDEO=false
  export PARTIAL_SCENE_LOAD=false
  export RENDER_VIEWER_CAMERA=true
  export GUI_VIEWPORT_ONLY=false
  export VIEWER_WIDTH=1280
  export VIEWER_HEIGHT=720
  export ENV_WRAPPER_TARGET="${ENV_WRAPPER_RGB}"
  export CPU_AFFINITY_MODE=none
  export EVAL_LAUNCH_STAGGER=20
  export RESTORE_CACHE_BEFORE_EVAL=false
  export WARMUP_CACHE_TAR="${WARMUP_CACHE_TAR}"
  export SERVER_READY_TIMEOUT=1800
  export SERVER_STARTUP_WAIT=600
  export MAX_STEPS_HUMAN_MULTIPLIER=1.2
  export VIDEO_ON_REPLAN_ONLY=true
  export STUCK_MOTION_WINDOW=2000
  export STUCK_MIN_STEPS=5000
  export STUCK_MOTION_THRESHOLD=0.75
  export CPU_AFFINITY_CPUS_PER_WORKER
}

apply_candidate_group() {
  local candidate="$1"
  reset_group_env
  case "${candidate}" in
    A0_baseline_current) ;;
    A1_partial_scene_load)
      export PARTIAL_SCENE_LOAD=true
      ;;
    A2_no_viewer_camera)
      export RENDER_VIEWER_CAMERA=false
      export GUI_VIEWPORT_ONLY=true
      export VIEWER_WIDTH=64
      export VIEWER_HEIGHT=64
      ;;
    A3_rgb_low_res)
      export ENV_WRAPPER_TARGET="${ENV_WRAPPER_LOWRES}"
      ;;
    A4_cache_restore)
      export RESTORE_CACHE_BEFORE_EVAL=true
      ;;
    A5_partial_scene_plus_no_viewer)
      export PARTIAL_SCENE_LOAD=true
      export RENDER_VIEWER_CAMERA=false
      export GUI_VIEWPORT_ONLY=true
      export VIEWER_WIDTH=64
      export VIEWER_HEIGHT=64
      ;;
    A6_partial_scene_plus_no_viewer_plus_cache)
      export PARTIAL_SCENE_LOAD=true
      export RENDER_VIEWER_CAMERA=false
      export GUI_VIEWPORT_ONLY=true
      export VIEWER_WIDTH=64
      export VIEWER_HEIGHT=64
      export RESTORE_CACHE_BEFORE_EVAL=true
      ;;
    *)
      die "unknown candidate group: ${candidate}"
      ;;
  esac
}

select_groups() {
  local phase="$1"
  local top_k="$2"
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --suite-dir "${SUITE_DIR}" \
    --select-phase "${phase}" \
    --select-top "${top_k}" \
    --print-groups
}

run_launcher_group() {
  local phase="$1"
  local group="$2"
  local notes="${3:-}"

  if [[ "${RESTORE_CACHE_BEFORE_EVAL}" == "true" && ! -f "${WARMUP_CACHE_TAR}" ]]; then
    warn "skip ${group}: warmup cache tar not found: ${WARMUP_CACHE_TAR}"
    append_manifest_row "${phase}" "${group}" "" "" "" "" "skipped" "missing warmup cache tar: ${WARMUP_CACHE_TAR}"
    return 0
  fi

  local run_tag="${SUITE_TAG}_${group}_$(date +%Y%m%d_%H%M%S)"
  local run_dir="${REPO_ROOT}/eval_logs/${run_tag}"
  local launch_epoch_s
  local end_epoch_s
  local exit_code=0
  launch_epoch_s="$(date +%s)"

  export TASK_NAME
  export BEHAVIOR_DIR
  export OPENPI_CONFIG_NAME
  export CKPT_DIR
  export RUN_TAG="${run_tag}"

  log "=== ${phase} / ${group} ==="
  log "RUN_TAG=${RUN_TAG}"
  log "GPU_IDS=${GPU_IDS} NUM_GPUS=${NUM_GPUS} EVAL_INSTANCE_IDS=${EVAL_INSTANCE_IDS} MAX_STEPS=${MAX_STEPS:-<auto>}"
  log "PARTIAL_SCENE_LOAD=${PARTIAL_SCENE_LOAD} ENV_WRAPPER_TARGET=${ENV_WRAPPER_TARGET}"
  log "RENDER_VIEWER_CAMERA=${RENDER_VIEWER_CAMERA} GUI_VIEWPORT_ONLY=${GUI_VIEWPORT_ONLY} VIEWER=${VIEWER_WIDTH}x${VIEWER_HEIGHT}"
  log "CPU_AFFINITY_MODE=${CPU_AFFINITY_MODE} EVAL_LAUNCH_STAGGER=${EVAL_LAUNCH_STAGGER}"
  log "RESTORE_CACHE_BEFORE_EVAL=${RESTORE_CACHE_BEFORE_EVAL} WARMUP_CACHE_TAR=${WARMUP_CACHE_TAR}"

  bash "${LAUNCHER}" "${CKPT_DIR}" || exit_code=$?
  end_epoch_s="$(date +%s)"

  append_manifest_row "${phase}" "${group}" "${run_dir}" "${launch_epoch_s}" "${end_epoch_s}" "${exit_code}" "done" "${notes}"
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --suite-dir "${SUITE_DIR}" >/dev/null
  return "${exit_code}"
}

run_phase_a() {
  local groups=(
    A0_baseline_current
    A1_partial_scene_load
    A2_no_viewer_camera
    A3_rgb_low_res
    A4_cache_restore
    A5_partial_scene_plus_no_viewer
    A6_partial_scene_plus_no_viewer_plus_cache
  )
  local group
  for group in "${groups[@]}"; do
    apply_candidate_group "${group}"
    export GPU_IDS="${PHASE_A_GPU_IDS}"
    export NUM_GPUS=1
    export EVAL_INSTANCE_IDS=0
    export MAX_STEPS=300
    run_launcher_group "phase_a_smoke_filter" "${group}" || true
  done
}

run_phase_b() {
  mapfile -t top_a < <(select_groups "phase_a_smoke_filter" 3)
  (( ${#top_a[@]} > 0 )) || die "phase A selection returned no groups"
  local best_a="${top_a[0]}"
  local non_baseline_count=0
  local group

  for group in "${top_a[@]}"; do
    if [[ "${group}" != "A0_baseline_current" ]]; then
      non_baseline_count=$(( non_baseline_count + 1 ))
    fi
  done

  if (( non_baseline_count == 0 )); then
    top_a=("A0_baseline_current")
  fi

  for group in "${top_a[@]}"; do
    apply_candidate_group "${group}"
    export GPU_IDS="${PHASE_B_GPU_IDS}"
    export NUM_GPUS=2
    export EVAL_INSTANCE_IDS=0,1
    export MAX_STEPS=600
    run_launcher_group "phase_b_2gpu_screen" "B_pick_${group}" "selected_from=${group}" || true
  done

  apply_candidate_group "${best_a}"
  export GPU_IDS="${PHASE_B_GPU_IDS}"
  export NUM_GPUS=2
  export EVAL_INSTANCE_IDS=0,1
  export MAX_STEPS=600
  export CPU_AFFINITY_MODE=compact
  run_launcher_group "phase_b_2gpu_screen" "B_cpu_affinity" "selected_from=${best_a}" || true

  apply_candidate_group "${best_a}"
  export GPU_IDS="${PHASE_B_GPU_IDS}"
  export NUM_GPUS=2
  export EVAL_INSTANCE_IDS=0,1
  export MAX_STEPS=600
  export EVAL_LAUNCH_STAGGER=60
  run_launcher_group "phase_b_2gpu_screen" "B_launch_stagger_60s" "selected_from=${best_a}" || true
}

run_phase_c() {
  mapfile -t top_a < <(select_groups "phase_a_smoke_filter" 1)
  mapfile -t top_b < <(select_groups "phase_b_2gpu_screen" 1)
  (( ${#top_a[@]} > 0 )) || die "phase A best group missing"
  (( ${#top_b[@]} > 0 )) || die "phase B best group missing"

  apply_candidate_group "A0_baseline_current"
  export GPU_IDS="${PHASE_C_GPU_IDS}"
  export NUM_GPUS=8
  export EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9
  export MAX_STEPS=""
  run_launcher_group "phase_c_8gpu_confirm" "C0_baseline_current" || true

  local best_a="${top_a[0]}"
  local best_b="${top_b[0]}"

  if [[ "${best_a}" == B_* ]]; then
    best_a="A0_baseline_current"
  fi

  if [[ "${best_a}" == A* ]]; then
    apply_candidate_group "${best_a}"
  else
    apply_candidate_group "A0_baseline_current"
  fi
  export GPU_IDS="${PHASE_C_GPU_IDS}"
  export NUM_GPUS=8
  export EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9
  export MAX_STEPS=""
  run_launcher_group "phase_c_8gpu_confirm" "C1_best_single_env" "selected_from=${best_a}" || true

  case "${best_b}" in
    B_pick_*)
      apply_candidate_group "${best_b#B_pick_}"
      ;;
    B_cpu_affinity)
      apply_candidate_group "${best_a}"
      export CPU_AFFINITY_MODE=compact
      ;;
    B_launch_stagger_60s)
      apply_candidate_group "${best_a}"
      export EVAL_LAUNCH_STAGGER=60
      ;;
    *)
      apply_candidate_group "A0_baseline_current"
      ;;
  esac
  export GPU_IDS="${PHASE_C_GPU_IDS}"
  export NUM_GPUS=8
  export EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9
  export MAX_STEPS=""
  run_launcher_group "phase_c_8gpu_confirm" "C2_best_under_contention" "selected_from=${best_b}" || true

  case "${best_b}" in
    B_pick_*)
      apply_candidate_group "${best_b#B_pick_}"
      ;;
    B_cpu_affinity)
      apply_candidate_group "${best_a}"
      export CPU_AFFINITY_MODE=compact
      ;;
    B_launch_stagger_60s)
      apply_candidate_group "${best_a}"
      export EVAL_LAUNCH_STAGGER=60
      ;;
    *)
      apply_candidate_group "A0_baseline_current"
      ;;
  esac
  export GPU_IDS="${PHASE_C_GPU_IDS}"
  export NUM_GPUS=8
  export EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9
  export MAX_STEPS=""
  export RESTORE_CACHE_BEFORE_EVAL=true
  run_launcher_group "phase_c_8gpu_confirm" "C3_best_under_contention_plus_cache_restore" "selected_from=${best_b}" || true
}

log "suite_dir=${SUITE_DIR}"
log "manifest=${MANIFEST_PATH}"

case "${PHASE}" in
  all)
    run_phase_a
    run_phase_b
    run_phase_c
    ;;
  phase_a_smoke_filter)
    run_phase_a
    ;;
  phase_b_2gpu_screen)
    run_phase_b
    ;;
  phase_c_8gpu_confirm)
    run_phase_c
    ;;
esac

"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --suite-dir "${SUITE_DIR}"
log "done"
