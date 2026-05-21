#!/usr/bin/env bash
set -Eeuo pipefail

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "[Warn] $*" >&2; }
die() { echo "[Error] $*" >&2; exit 1; }

find_first_executable() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  bash /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/run_quick_validation_dataset_single_node_8gpu.sh [run|eval|post]

Goal:
  - Single node, 8 GPUs parallel quick-validation dataset generation.
  - Wrap scripts/run_skill_eval_single_node_8gpu.sh and then build review/sanity artifacts.
  - Persist all review-critical outputs under one run directory so the run can be moved to a new instance.

Phases:
  run   (default): prepare + launch + merge + postprocess
  eval            : prepare + launch + merge only
  post            : build metadata + review set + sanity/audit manifests for an existing OUT_DIR

Key env overrides:
  RUN_TAG=quick_validation_dataset_8gpu_...
  OUT_DIR=/abs/path/to/segment_eval_runs/<run_tag>

  CKPT_DIR=/abs/path/to/checkpoint_dir
  CONFIG_NAME=pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep
  SKILLS="move to,open door"          # optional skill subset

  LOCAL_GPU_IDS=0,1,2,3,4,5,6,7
  GPUS_PER_NODE=8

  MAX_SAMPLES_PER_SKILL=8
  MAX_SAMPLES_PER_SKILL_TASK=2
  MAX_TOTAL_JOBS=0
  MAX_STEPS=160
  MAX_DYNAMIC_STEPS_CAP=<MAX_STEPS>      # defaults to MAX_STEPS for bounded quick validation

  WRITE_VIDEO=1                        # keep rollout videos
  SEGMENT_PREDICATE_DUMP_TRACE=1      # keep predicate traces in metrics json
  REVIEW_SAMPLES_PER_SKILL=8
  REVIEW_HOLDOUT_PER_SKILL=2
  AUDIT_SAMPLES_PER_SKILL=8
  AUDIT_HOLDOUT_PER_SKILL=2

  BEHAVIOR_DIR=/abs/path/to/BEHAVIOR-1K
  DEMO_DATA_PATH=/abs/path/to/2025-challenge-demos
  RAWDATA_PATH=/abs/path/to/2025-challenge-rawdata

  EXTRA_BASHRC=/path/to/extra_bashrc.sh
  SKIP_EXTRA_BASHRC=0|1
  CONDA_SH=/path/to/conda.sh

Outputs under OUT_DIR:
  manifest.json
  jobs/worker_*.json
  launcher_logs/
  server_logs/
  worker_results/
  raw/<task>/demo_<id>/skill_<idx>/...
  multinode_skill_results.*
  multinode_skill_summary.*
  review/review_manifest.{csv,json}
  sanity/sanity_manifest.{jsonl,csv}
  sanity/audit_manifest.csv
  run_metadata/*
EOF
}

on_err() {
  local exit_code=$?
  echo "[ERR] exit_code=${exit_code} line=${BASH_LINENO[0]:-unknown} cmd=${BASH_COMMAND}" >&2
}
trap on_err ERR

PHASE="${1:-run}"
case "${PHASE}" in
  run|eval|post) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    die "unsupported phase: ${PHASE}"
    ;;
esac

REPO_ROOT="${REPO_ROOT:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet}"
SCRIPT_SELF="${SCRIPT_SELF:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/run_quick_validation_dataset_single_node_8gpu.sh}"
BASE_LAUNCHER="${BASE_LAUNCHER:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh}"
[[ -d "${REPO_ROOT}" ]] || die "REPO_ROOT not found: ${REPO_ROOT}"
[[ -f "${BASE_LAUNCHER}" ]] || die "BASE_LAUNCHER not found: ${BASE_LAUNCHER}"
cd "${REPO_ROOT}"

EXTRA_BASHRC="${EXTRA_BASHRC:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/extra_bashrc.sh}"
SKIP_EXTRA_BASHRC="${SKIP_EXTRA_BASHRC:-0}"
if [[ "${SKIP_EXTRA_BASHRC}" == "1" ]]; then
  :
elif [[ -f "${EXTRA_BASHRC}" ]]; then
  set +u
  set +e
  source "${EXTRA_BASHRC}"
  set -e
  set -u
else
  warn "EXTRA_BASHRC not found: ${EXTRA_BASHRC}"
fi

CONDA_SH="${CONDA_SH:-/mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate openpi-comet-nas || warn "failed to activate conda env: openpi-comet-nas"
else
  warn "CONDA_SH not found: ${CONDA_SH}"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

PYTHON_BIN="$(find_first_executable \
  "/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python" \
  "/mnt/bn/navigation-hl/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python" \
  "${OPENPI_PYTHON:-}" \
  "${REPO_ROOT}/.venv/bin/python" \
  "$(command -v python3 2>/dev/null || true)" \
  "$(command -v python 2>/dev/null || true)" \
  || true)"
[[ -x "${PYTHON_BIN}" ]] || die "No executable Python found for quick validation dataset wrapper."

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${REPO_ROOT}/segment_eval_runs"
mkdir -p "${RUNS_ROOT}"

OUT_DIR_INPUT="${OUT_DIR-}"
RUN_TAG="${RUN_TAG:-quick_validation_dataset_8gpu_${TIMESTAMP}}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/${RUN_TAG}}"
if [[ "${PHASE}" == "post" && -z "${OUT_DIR_INPUT}" ]]; then
  die "post phase requires explicit OUT_DIR pointing to an existing run directory"
fi
if [[ "${PHASE}" == "post" ]]; then
  [[ -d "${OUT_DIR}" ]] || die "OUT_DIR does not exist for post phase: ${OUT_DIR}"
else
  mkdir -p "${OUT_DIR}"
fi

LOCAL_GPU_IDS="${LOCAL_GPU_IDS:-0,1,2,3,4,5,6,7}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt50-cs32}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K}"
DEMO_DATA_PATH="${DEMO_DATA_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos}"
RAWDATA_PATH="${RAWDATA_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata}"

MAX_SAMPLES_PER_SKILL="${MAX_SAMPLES_PER_SKILL:-8}"
MAX_SAMPLES_PER_SKILL_TASK="${MAX_SAMPLES_PER_SKILL_TASK:-2}"
MAX_TOTAL_JOBS="${MAX_TOTAL_JOBS:-0}"
MAX_STEPS="${MAX_STEPS:-160}"
MAX_DYNAMIC_STEPS_CAP="${MAX_DYNAMIC_STEPS_CAP:-${MAX_STEPS}}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
PREPARE_TIMEOUT="${PREPARE_TIMEOUT:-3600}"
SERVER_START_STAGGER_S="${SERVER_START_STAGGER_S:-10}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
WRITE_VIDEO="${WRITE_VIDEO:-1}"
SEGMENT_PREDICATE_DUMP_TRACE="${SEGMENT_PREDICATE_DUMP_TRACE:-1}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
SKILLS="${SKILLS:-}"

REVIEW_TARGET_SKILLS="${REVIEW_TARGET_SKILLS:-${SKILLS}}"
REVIEW_SAMPLES_PER_SKILL="${REVIEW_SAMPLES_PER_SKILL:-8}"
REVIEW_HOLDOUT_PER_SKILL="${REVIEW_HOLDOUT_PER_SKILL:-2}"
AUDIT_SAMPLES_PER_SKILL="${AUDIT_SAMPLES_PER_SKILL:-8}"
AUDIT_HOLDOUT_PER_SKILL="${AUDIT_HOLDOUT_PER_SKILL:-2}"
AUDIT_SEED="${AUDIT_SEED:-7}"
RUN_NOTES="${RUN_NOTES:-}"

RUN_METADATA_DIR="${OUT_DIR}/run_metadata"
RUN_METADATA_SCRIPTS_DIR="${RUN_METADATA_DIR}/scripts"
mkdir -p "${RUN_METADATA_SCRIPTS_DIR}"

snapshot_metadata() {
  mkdir -p "${RUN_METADATA_DIR}" "${RUN_METADATA_SCRIPTS_DIR}"
  cp -f "${SCRIPT_SELF}" "${RUN_METADATA_SCRIPTS_DIR}/" 2>/dev/null || true
  cp -f "${BASE_LAUNCHER}" "${RUN_METADATA_SCRIPTS_DIR}/" 2>/dev/null || true

  {
    echo "PHASE=${PHASE}"
    echo "RUN_TAG=${RUN_TAG}"
    echo "OUT_DIR=${OUT_DIR}"
    echo "NODE_RANK=${NODE_RANK}"
    echo "NUM_NODES=${NUM_NODES}"
    echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
    echo "LOCAL_GPU_IDS=${LOCAL_GPU_IDS}"
    echo "CONFIG_NAME=${CONFIG_NAME}"
    echo "CKPT_DIR=${CKPT_DIR}"
    echo "BEHAVIOR_DIR=${BEHAVIOR_DIR}"
    echo "DEMO_DATA_PATH=${DEMO_DATA_PATH}"
    echo "RAWDATA_PATH=${RAWDATA_PATH}"
    echo "SKILLS=${SKILLS}"
    echo "MAX_SAMPLES_PER_SKILL=${MAX_SAMPLES_PER_SKILL}"
    echo "MAX_SAMPLES_PER_SKILL_TASK=${MAX_SAMPLES_PER_SKILL_TASK}"
    echo "MAX_TOTAL_JOBS=${MAX_TOTAL_JOBS}"
    echo "MAX_STEPS=${MAX_STEPS}"
    echo "MAX_DYNAMIC_STEPS_CAP=${MAX_DYNAMIC_STEPS_CAP}"
    echo "WRITE_VIDEO=${WRITE_VIDEO}"
    echo "SEGMENT_PREDICATE_DUMP_TRACE=${SEGMENT_PREDICATE_DUMP_TRACE}"
    echo "REVIEW_TARGET_SKILLS=${REVIEW_TARGET_SKILLS}"
    echo "REVIEW_SAMPLES_PER_SKILL=${REVIEW_SAMPLES_PER_SKILL}"
    echo "REVIEW_HOLDOUT_PER_SKILL=${REVIEW_HOLDOUT_PER_SKILL}"
    echo "AUDIT_SAMPLES_PER_SKILL=${AUDIT_SAMPLES_PER_SKILL}"
    echo "AUDIT_HOLDOUT_PER_SKILL=${AUDIT_HOLDOUT_PER_SKILL}"
    echo "AUDIT_SEED=${AUDIT_SEED}"
    echo "RESUME=${RESUME}"
    echo "DRY_RUN=${DRY_RUN}"
    echo "REBUILD_MANIFEST=${REBUILD_MANIFEST}"
    echo "RUN_NOTES=${RUN_NOTES}"
  } > "${RUN_METADATA_DIR}/run_config.env"

  {
    echo "timestamp=$(date -Iseconds)"
    echo "hostname=$(hostname)"
    echo "pwd=$(pwd)"
    echo "script=${SCRIPT_SELF}"
    echo "base_launcher=${BASE_LAUNCHER}"
    echo "python_bin=${PYTHON_BIN}"
    echo "python_version=$(${PYTHON_BIN} -V 2>&1 || true)"
    echo "conda_env=${CONDA_DEFAULT_ENV:-}"
    echo "phase=${PHASE}"
  } > "${RUN_METADATA_DIR}/runtime_context.txt"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L > "${RUN_METADATA_DIR}/nvidia_smi_L.txt" 2>&1 || true
  fi

  if command -v git >/dev/null 2>&1 && git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_METADATA_DIR}/git_head.txt" 2>&1 || true
    git -C "${REPO_ROOT}" status --short > "${RUN_METADATA_DIR}/git_status.txt" 2>&1 || true
  fi
}

has_metrics() {
  compgen -G "${OUT_DIR}/raw/*/demo_*/skill_*/metrics/*.json" >/dev/null
}

write_artifact_summary() {
  local summary_path="${RUN_METADATA_DIR}/artifact_summary.txt"
  {
    echo "OUT_DIR=${OUT_DIR}"
    echo "RUN_TAG=${RUN_TAG}"
    echo ""
    echo "Core eval artifacts:"
    echo "- ${OUT_DIR}/manifest.json"
    echo "- ${OUT_DIR}/jobs/"
    echo "- ${OUT_DIR}/launcher_logs/"
    echo "- ${OUT_DIR}/server_logs/"
    echo "- ${OUT_DIR}/worker_results/"
    echo "- ${OUT_DIR}/raw/"
    echo "- ${OUT_DIR}/multinode_skill_results.csv"
    echo "- ${OUT_DIR}/multinode_skill_results.json"
    echo "- ${OUT_DIR}/multinode_skill_summary.csv"
    echo "- ${OUT_DIR}/multinode_skill_summary.json"
    echo "- ${OUT_DIR}/multinode_skill_summary.md"
    echo "- ${OUT_DIR}/multinode_skill_task_summary.csv"
    echo ""
    echo "Review artifacts:"
    echo "- ${OUT_DIR}/review/review_manifest.csv"
    echo "- ${OUT_DIR}/review/review_manifest.json"
    echo "- ${OUT_DIR}/review/segments/"
    echo ""
    echo "Sanity artifacts:"
    echo "- ${OUT_DIR}/sanity/sanity_manifest.jsonl"
    echo "- ${OUT_DIR}/sanity/sanity_manifest.csv"
    echo "- ${OUT_DIR}/sanity/missing_summary.json"
    echo "- ${OUT_DIR}/sanity/audit_manifest.csv"
    echo ""
    echo "Metadata artifacts:"
    echo "- ${RUN_METADATA_DIR}/run_config.env"
    echo "- ${RUN_METADATA_DIR}/runtime_context.txt"
    echo "- ${RUN_METADATA_DIR}/nvidia_smi_L.txt"
    echo "- ${RUN_METADATA_DIR}/git_head.txt"
    echo "- ${RUN_METADATA_DIR}/git_status.txt"
    echo "- ${RUN_METADATA_SCRIPTS_DIR}/"
  } > "${summary_path}"
}

run_eval() {
  snapshot_metadata

  log "=== Quick validation dataset eval ==="
  log "out_dir: ${OUT_DIR}"
  log "python_bin: ${PYTHON_BIN}"
  log "config_name: ${CONFIG_NAME}"
  log "ckpt_dir: ${CKPT_DIR}"
  log "skills: ${SKILLS:-<all>}"
  log "max_samples_per_skill: ${MAX_SAMPLES_PER_SKILL}"
  log "max_samples_per_skill_task: ${MAX_SAMPLES_PER_SKILL_TASK}"
  log "max_steps: ${MAX_STEPS}"
  log "max_dynamic_steps_cap: ${MAX_DYNAMIC_STEPS_CAP}"
  log "write_video: ${WRITE_VIDEO}"
  log "segment_predicate_dump_trace: ${SEGMENT_PREDICATE_DUMP_TRACE}"

  local base_env=(
    RUN_TAG="${RUN_TAG}"
    OUT_DIR="${OUT_DIR}"
    NODE_RANK="${NODE_RANK}"
    NUM_NODES="${NUM_NODES}"
    GPUS_PER_NODE="${GPUS_PER_NODE}"
    LOCAL_GPU_IDS="${LOCAL_GPU_IDS}"
    CONFIG_NAME="${CONFIG_NAME}"
    CKPT_DIR="${CKPT_DIR}"
    BEHAVIOR_DIR="${BEHAVIOR_DIR}"
    DEMO_DATA_PATH="${DEMO_DATA_PATH}"
    RAWDATA_PATH="${RAWDATA_PATH}"
    MAX_SAMPLES_PER_SKILL="${MAX_SAMPLES_PER_SKILL}"
    MAX_SAMPLES_PER_SKILL_TASK="${MAX_SAMPLES_PER_SKILL_TASK}"
    MAX_TOTAL_JOBS="${MAX_TOTAL_JOBS}"
    MAX_STEPS="${MAX_STEPS}"
    MAX_DYNAMIC_STEPS_CAP="${MAX_DYNAMIC_STEPS_CAP}"
    SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}"
    PREPARE_TIMEOUT="${PREPARE_TIMEOUT}"
    SERVER_START_STAGGER_S="${SERVER_START_STAGGER_S}"
    RESUME="${RESUME}"
    DRY_RUN="${DRY_RUN}"
    WRITE_VIDEO="${WRITE_VIDEO}"
    SEGMENT_PREDICATE_DUMP_TRACE="${SEGMENT_PREDICATE_DUMP_TRACE}"
    REBUILD_MANIFEST="${REBUILD_MANIFEST}"
    EXTRA_BASHRC="${EXTRA_BASHRC}"
    SKIP_EXTRA_BASHRC="${SKIP_EXTRA_BASHRC}"
    CONDA_SH="${CONDA_SH}"
    TEE_LAUNCHER_LOG="1"
    PYTHONNOUSERSITE="${PYTHONNOUSERSITE}"
  )
  if [[ -n "${SKILLS}" ]]; then
    base_env+=(SKILLS="${SKILLS}")
  fi

  log "Running base launcher: prepare"
  env "${base_env[@]}" bash "${BASE_LAUNCHER}" prepare
  log "Running base launcher: launch"
  env "${base_env[@]}" bash "${BASE_LAUNCHER}" launch
  log "Running base launcher: merge"
  env "${base_env[@]}" bash "${BASE_LAUNCHER}" merge
}

run_postprocess() {
  snapshot_metadata

  if ! has_metrics; then
    warn "No metrics json found under ${OUT_DIR}/raw; skipping review/sanity postprocess."
    write_artifact_summary
    return 0
  fi

  local review_args=(
    --run-dir "${OUT_DIR}"
    --samples-per-skill "${REVIEW_SAMPLES_PER_SKILL}"
    --holdout-per-skill "${REVIEW_HOLDOUT_PER_SKILL}"
  )
  if [[ -n "${REVIEW_TARGET_SKILLS}" ]]; then
    review_args+=(--skills "${REVIEW_TARGET_SKILLS}")
  fi

  log "Building review set"
  "${PYTHON_BIN}" -u scripts/build_skill_metric_review_set.py "${review_args[@]}"

  log "Building sanity manifest"
  "${PYTHON_BIN}" -u scripts/build_segment_sanity_manifest.py --run-dir "${OUT_DIR}"

  local sanity_manifest="${OUT_DIR}/sanity/sanity_manifest.jsonl"
  if [[ -f "${sanity_manifest}" ]]; then
    log "Building audit manifest"
    "${PYTHON_BIN}" -u scripts/audit_segment_sanity_manifest.py \
      --mode sample \
      --manifest "${sanity_manifest}" \
      --out "${OUT_DIR}/sanity/audit_manifest.csv" \
      --samples-per-skill "${AUDIT_SAMPLES_PER_SKILL}" \
      --holdout-per-skill "${AUDIT_HOLDOUT_PER_SKILL}" \
      --seed "${AUDIT_SEED}"
  else
    warn "sanity manifest not found: ${sanity_manifest}"
  fi

  write_artifact_summary
  log "Artifact summary written to ${RUN_METADATA_DIR}/artifact_summary.txt"
}

if [[ "${PHASE}" == "eval" ]]; then
  run_eval
  exit 0
fi

if [[ "${PHASE}" == "post" ]]; then
  run_postprocess
  exit 0
fi

run_eval
run_postprocess
log "Quick validation dataset run finished. OUT_DIR=${OUT_DIR}"
