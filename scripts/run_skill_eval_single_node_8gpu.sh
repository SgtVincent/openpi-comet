#!/usr/bin/env bash
set -Eeuo pipefail

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

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
  bash scripts/run_skill_eval_single_node_8gpu.sh [prepare|launch|merge]

Goal:
  - Single node, 8 GPUs parallel skill/segment eval using scripts/run_skill_metric_multinode_sweep.py
  - Focus on getting the single-node pipeline stable before multi-node

Default behavior:
  - No argument: run `launch` and then `merge`
  - NUM_NODES=1, GPUS_PER_NODE=8, LOCAL_GPU_IDS=0..7
  - Sampling defaults are tuned for a quick smoke/probe (not full coverage)
  - Outputs are persisted under <repo>/segment_eval_runs/<RUN_TAG>

Optional env overrides:
  EXTRA_BASHRC=/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/extra_bashrc.sh
  SKIP_EXTRA_BASHRC=0|1
  CONDA_SH=/path/to/conda.sh

  RUN_TAG=custom_tag
  OUT_DIR=/abs/path/to/output_dir

  LOCAL_GPU_IDS=0,1,2,3,4,5,6,7
  GPUS_PER_NODE=8

  CKPT_DIR=/abs/path/to/checkpoint_dir
  CONFIG_NAME=pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep

  SKILLS="move to,open door"   (optional subset)
  MAX_SAMPLES_PER_SKILL=4
  MAX_SAMPLES_PER_SKILL_TASK=0
  MAX_TOTAL_JOBS=0

  MAX_STEPS=120
  SERVER_READY_TIMEOUT=1800
  PREPARE_TIMEOUT=3600
  SERVER_START_STAGGER_S=10

  RESUME=1
  DRY_RUN=0
  WRITE_VIDEO=1
  SEGMENT_PREDICATE_DUMP_TRACE=0
  REBUILD_MANIFEST=0

  TEE_LAUNCHER_LOG=0|1
  CONSOLE_LOG=/abs/path/to/launcher_console.log
EOF
}

on_err() {
  local exit_code=$?
  echo "[ERR] exit_code=${exit_code} line=${BASH_LINENO[0]:-unknown} cmd=${BASH_COMMAND}" >&2
  echo "[ERR] hint: check OUT_DIR logs: launcher_logs/, server_logs/, worker_results/, raw/*/*/*/segment_eval.log" >&2
}
trap on_err ERR

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

REQUESTED_PHASE="${1:-${PHASE:-run}}"
case "${REQUESTED_PHASE}" in
  run|prepare|launch|merge) ;;
  *)
    echo "[Error] unsupported phase: ${REQUESTED_PHASE}" >&2
    usage
    exit 1
    ;;
esac

if [[ "${REQUESTED_PHASE}" == "run" ]]; then
  PHASE="launch+merge"
else
  PHASE="${REQUESTED_PHASE}"
fi

case "${PHASE}" in
  prepare|launch|merge) ;;
  launch+merge) ;;
esac

EXTRA_BASHRC="${EXTRA_BASHRC:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/extra_bashrc.sh}"
SKIP_EXTRA_BASHRC="${SKIP_EXTRA_BASHRC:-0}"
if [[ "${SKIP_EXTRA_BASHRC}" == "1" ]]; then
  :
elif [[ -f "${EXTRA_BASHRC}" ]]; then
  # Source into current shell to inherit proxy/cache/env settings.
  set +u
  set +e
  source "${EXTRA_BASHRC}"
  set -e
  set -u
else
  echo "[Warn] EXTRA_BASHRC not found: ${EXTRA_BASHRC}" >&2
fi

CONDA_SH="${CONDA_SH:-/mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate openpi-comet-nas || echo "[Warn] failed to activate conda env: openpi-comet-nas" >&2
else
  echo "[Warn] CONDA_SH not found: ${CONDA_SH}" >&2
fi

PYTHON_BIN="$(find_first_executable \
  "/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python" \
  "/mnt/bn/navigation-hl/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python" \
  "${OPENPI_PYTHON:-}" \
  "${REPO_ROOT}/.venv/bin/python" \
  "$(command -v python3 2>/dev/null || true)" \
  "$(command -v python 2>/dev/null || true)" \
  || true)"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "[Error] No executable Python found for single-node skill eval launcher." >&2
  exit 1
}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${REPO_ROOT}/segment_eval_runs"
mkdir -p "${RUNS_ROOT}"

RUN_TAG="${RUN_TAG:-pi05_b1kpt50_single_node_8gpu_${TIMESTAMP}}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/${RUN_TAG}}"
mkdir -p "${OUT_DIR}"

CONSOLE_LOG="${CONSOLE_LOG:-${OUT_DIR}/launcher_console_${PHASE}.log}"
if [[ "${TEE_LAUNCHER_LOG:-1}" == "1" ]]; then
  exec > >(tee -a "${CONSOLE_LOG}") 2>&1
fi

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
LOCAL_GPU_IDS="${LOCAL_GPU_IDS:-0,1,2,3,4,5,6,7}"

MAX_STEPS="${MAX_STEPS:-120}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
PREPARE_TIMEOUT="${PREPARE_TIMEOUT:-3600}"
SERVER_START_STAGGER_S="${SERVER_START_STAGGER_S:-10}"

MAX_SAMPLES_PER_SKILL="${MAX_SAMPLES_PER_SKILL:-4}"
MAX_SAMPLES_PER_SKILL_TASK="${MAX_SAMPLES_PER_SKILL_TASK:-0}"
MAX_TOTAL_JOBS="${MAX_TOTAL_JOBS:-0}"

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt50-cs32}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K}"
DEMO_DATA_PATH="${DEMO_DATA_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos}"
RAWDATA_PATH="${RAWDATA_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata}"

RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
WRITE_VIDEO="${WRITE_VIDEO:-1}"
SEGMENT_PREDICATE_DUMP_TRACE="${SEGMENT_PREDICATE_DUMP_TRACE:-0}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
SKILLS="${SKILLS:-}"

log "=== Single-node 8-GPU Skill Eval ==="
log "host: $(hostname)"
log "repo_root: ${REPO_ROOT}"
log "phase: ${PHASE}"
log "out_dir: ${OUT_DIR}"
log "console_log: ${CONSOLE_LOG}"
log "python_bin: ${PYTHON_BIN}"
log "python -V: $("${PYTHON_BIN}" -V 2>&1 || true)"
log "conda env: ${CONDA_DEFAULT_ENV:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi -L:"
  nvidia-smi -L || true
fi
log "df -h (out_dir):"
df -h "${OUT_DIR}" || true

PY_ARGS=(
  --out-dir "${OUT_DIR}"
  --node-rank "${NODE_RANK}"
  --num-nodes "${NUM_NODES}"
  --gpus-per-node "${GPUS_PER_NODE}"
  --local-gpu-ids "${LOCAL_GPU_IDS}"
  --max-steps "${MAX_STEPS}"
  --server-ready-timeout "${SERVER_READY_TIMEOUT}"
  --server-start-stagger-s "${SERVER_START_STAGGER_S}"
  --prepare-timeout "${PREPARE_TIMEOUT}"
  --config-name "${CONFIG_NAME}"
  --ckpt-dir "${CKPT_DIR}"
  --behavior-dir "${BEHAVIOR_DIR}"
  --demo-data-path "${DEMO_DATA_PATH}"
  --rawdata-path "${RAWDATA_PATH}"
  --max-samples-per-skill "${MAX_SAMPLES_PER_SKILL}"
  --max-samples-per-skill-task "${MAX_SAMPLES_PER_SKILL_TASK}"
  --max-total-jobs "${MAX_TOTAL_JOBS}"
)

if [[ -n "${SKILLS}" ]]; then
  PY_ARGS+=(--skills "${SKILLS}")
fi
if [[ "${RESUME}" == "1" ]]; then
  PY_ARGS+=(--resume)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  PY_ARGS+=(--dry-run)
fi
if [[ "${WRITE_VIDEO}" == "1" ]]; then
  PY_ARGS+=(--write-video)
fi
if [[ "${SEGMENT_PREDICATE_DUMP_TRACE}" == "1" ]]; then
  PY_ARGS+=(--segment-predicate-dump-trace)
fi
if [[ "${REBUILD_MANIFEST}" == "1" ]]; then
  PY_ARGS+=(--rebuild-manifest)
fi

log "node_rank: ${NODE_RANK} num_nodes: ${NUM_NODES}"
log "gpus_per_node: ${GPUS_PER_NODE} local_gpu_ids: ${LOCAL_GPU_IDS}"
log "skills: ${SKILLS:-<all>}"
log "max_samples_per_skill: ${MAX_SAMPLES_PER_SKILL} max_samples_per_skill_task: ${MAX_SAMPLES_PER_SKILL_TASK} max_total_jobs: ${MAX_TOTAL_JOBS}"
log "max_steps: ${MAX_STEPS} server_ready_timeout: ${SERVER_READY_TIMEOUT} prepare_timeout: ${PREPARE_TIMEOUT}"
log "server_start_stagger_s: ${SERVER_START_STAGGER_S}"
log "write_video: ${WRITE_VIDEO} segment_predicate_dump_trace: ${SEGMENT_PREDICATE_DUMP_TRACE}"
log "config_name: ${CONFIG_NAME}"
log "ckpt_dir: ${CKPT_DIR}"
log "behavior_dir: ${BEHAVIOR_DIR}"
log "demo_data_path: ${DEMO_DATA_PATH}"
log "rawdata_path: ${RAWDATA_PATH}"

if [[ "${PHASE}" == "prepare" ]]; then
  log "Running: prepare"
  "${PYTHON_BIN}" -u scripts/run_skill_metric_multinode_sweep.py --mode prepare "${PY_ARGS[@]}"
  exit 0
fi

if [[ "${PHASE}" == "merge" ]]; then
  log "Running: merge"
  "${PYTHON_BIN}" -u scripts/run_skill_metric_multinode_sweep.py --mode merge "${PY_ARGS[@]}"
  exit 0
fi

if [[ "${PHASE}" == "launch+merge" ]]; then
  log "Running: launch"
  "${PYTHON_BIN}" -u scripts/run_skill_metric_multinode_sweep.py --mode launch "${PY_ARGS[@]}"
  log "Running: merge"
  "${PYTHON_BIN}" -u scripts/run_skill_metric_multinode_sweep.py --mode merge "${PY_ARGS[@]}"
  exit 0
fi

log "Running: launch"
"${PYTHON_BIN}" -u scripts/run_skill_metric_multinode_sweep.py --mode launch "${PY_ARGS[@]}"
