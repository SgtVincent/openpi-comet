#!/bin/bash
set -Eeuo pipefail

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
  bash scripts/conda_run_pi05_b1kpt50_multinode_skill_eval.sh [prepare|launch|merge]

Environment overrides:
  EXTRA_BASHRC=/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/extra_bashrc.sh
  SWEEP_PHASE=launch|prepare|merge
  CONFIG_NAME=pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep
  CKPT_DIR=/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32
  RUN_TAG=custom_run_tag
  NUM_NODES=4
  GPUS_PER_NODE=8
  LOCAL_GPU_IDS=0,1,2,3,4,5,6,7
  SKILLS="move to,open door"
  MAX_SAMPLES_PER_SKILL=96
  MAX_SAMPLES_PER_SKILL_TASK=2
  MAX_TOTAL_JOBS=0
  MAX_STEPS=120
  SERVER_READY_TIMEOUT=1800
  PREPARE_TIMEOUT=3600
  REBUILD_MANIFEST=0|1
  RESUME=0|1
  DRY_RUN=0|1
  WRITE_VIDEO=0|1
  SEGMENT_PREDICATE_DUMP_TRACE=0|1
  TEE_LAUNCHER_LOG=0|1
  CONSOLE_LOG=/path/to/launcher_console.log

Notes:
  - 正式结果默认持久化到 repo 下的 segment_eval_runs/
  - 同一多节点任务建议所有节点使用相同 RUN_TAG
  - launch 阶段在 node0 上会自动 prepare manifest（若 manifest 不存在或 REBUILD_MANIFEST=1）
EOF
}

on_err() {
  local exit_code=$?
  echo "[ERR] exit_code=${exit_code} line=${BASH_LINENO[0]:-unknown} cmd=${BASH_COMMAND}" >&2
}

trap on_err ERR

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXTRA_BASHRC="${EXTRA_BASHRC:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/extra_bashrc.sh}"
CONDA_SH="/mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh"
RUNTIME_ENV_FILE="${HOME}/.openpi_runtime_env.sh"

if [[ -f "${EXTRA_BASHRC}" ]]; then
  # Source into the current shell so proxy/conda/path/cache settings are retained.
  # Running `bash extra_bashrc.sh` would not persist exported variables for this script.
  set +u
  set +e
  source "${EXTRA_BASHRC}"
  set -e
  set -u
else
  echo "[Warn] EXTRA_BASHRC not found: ${EXTRA_BASHRC}" >&2
fi

if [[ -f "${RUNTIME_ENV_FILE}" ]]; then
  source "${RUNTIME_ENV_FILE}"
fi

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
  echo "[Error] No executable Python found for multinode skill eval launcher." >&2
  exit 1
}

SWEEP_PHASE="${1:-${SWEEP_PHASE:-launch}}"
case "${SWEEP_PHASE}" in
  prepare|launch|merge) ;;
  *)
    echo "[Error] unsupported phase: ${SWEEP_PHASE}" >&2
    usage
    exit 1
    ;;
esac

MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-$(hostname)}}"
MASTER_PORT_RAW="${MASTER_PORT:-${ARNOLD_WORKER_0_PORT:-17000}}"
MASTER_PORT="${MASTER_PORT_RAW%%,*}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"
NUM_NODES="${NUM_NODES:-${ARNOLD_WORKER_NUM:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}"

if [[ -z "${LOCAL_GPU_IDS:-}" ]]; then
  LOCAL_GPU_IDS="$(seq 0 $((GPUS_PER_NODE - 1)) | paste -sd, -)"
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/openpi_comet/pi05-b1kpt50-cs32}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K}"
DEMO_DATA_PATH="${DEMO_DATA_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos}"
RAWDATA_PATH="${RAWDATA_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata}"

MAX_STEPS="${MAX_STEPS:-120}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
PREPARE_TIMEOUT="${PREPARE_TIMEOUT:-3600}"
MAX_SAMPLES_PER_SKILL="${MAX_SAMPLES_PER_SKILL:-64}"
MAX_SAMPLES_PER_SKILL_TASK="${MAX_SAMPLES_PER_SKILL_TASK:-2}"
MAX_TOTAL_JOBS="${MAX_TOTAL_JOBS:-0}"

REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
WRITE_VIDEO="${WRITE_VIDEO:-0}"
SEGMENT_PREDICATE_DUMP_TRACE="${SEGMENT_PREDICATE_DUMP_TRACE:-0}"
SKILLS="${SKILLS:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${REPO_ROOT}/segment_eval_runs"
SYNC_ROOT="${RUNS_ROOT}/_multinode_run_tag_sync"
mkdir -p "${RUNS_ROOT}" "${SYNC_ROOT}"

if [[ -z "${RUN_TAG:-}" ]]; then
  RUN_KEY="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-${MASTER_ADDR}_${MASTER_PORT}_${NUM_NODES}x${GPUS_PER_NODE}}}"
  RUN_KEY="${RUN_KEY//\//_}"
  RUN_KEY="${RUN_KEY//:/_}"
  RUN_KEY="${RUN_KEY// /_}"
  RUN_TAG_FILE="${SYNC_ROOT}/pi05_b1kpt50_skill_eval_${RUN_KEY}.txt"
  if [[ "${NODE_RANK}" == "0" ]]; then
    if [[ -s "${RUN_TAG_FILE}" && "${RESUME}" == "1" ]]; then
      RUN_TAG="$(cat "${RUN_TAG_FILE}")"
    else
      RUN_TAG="pi05_b1kpt50_multinode_skill_eval_${NUM_NODES}x${GPUS_PER_NODE}_${TIMESTAMP}"
      TMP_RUN_TAG_FILE="${RUN_TAG_FILE}.$$.$RANDOM.tmp"
      printf "%s\n" "${RUN_TAG}" > "${TMP_RUN_TAG_FILE}"
      mv -f "${TMP_RUN_TAG_FILE}" "${RUN_TAG_FILE}"
    fi
  else
    for _i in $(seq 1 600); do
      if [[ -s "${RUN_TAG_FILE}" ]]; then
        break
      fi
      sleep 1
    done
    if [[ ! -s "${RUN_TAG_FILE}" ]]; then
      echo "[Error] timed out waiting for RUN_TAG_FILE: ${RUN_TAG_FILE}" >&2
      exit 1
    fi
    RUN_TAG="$(cat "${RUN_TAG_FILE}")"
  fi
fi

OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/${RUN_TAG}}"
mkdir -p "${OUT_DIR}"

CONSOLE_LOG="${CONSOLE_LOG:-${OUT_DIR}/launcher_console_node${NODE_RANK}.log}"
if [[ "${TEE_LAUNCHER_LOG:-1}" == "1" ]]; then
  exec > >(tee -a "${CONSOLE_LOG}") 2>&1
fi

PY_ARGS=(
  --out-dir "${OUT_DIR}"
  --node-rank "${NODE_RANK}"
  --num-nodes "${NUM_NODES}"
  --gpus-per-node "${GPUS_PER_NODE}"
  --local-gpu-ids "${LOCAL_GPU_IDS}"
  --max-steps "${MAX_STEPS}"
  --server-ready-timeout "${SERVER_READY_TIMEOUT}"
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
if [[ "${REBUILD_MANIFEST}" == "1" ]]; then
  PY_ARGS+=(--rebuild-manifest)
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

echo "Starting PI05 B1K PT50 multinode skill evaluation"
echo "SWEEP_PHASE: ${SWEEP_PHASE}"
echo "EXTRA_BASHRC: ${EXTRA_BASHRC}"
echo "RUN_TAG: ${RUN_TAG}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CONSOLE_LOG: ${CONSOLE_LOG}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "LOCAL_GPU_IDS: ${LOCAL_GPU_IDS}"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo "CKPT_DIR: ${CKPT_DIR}"
echo "================ Runtime Python ================"
echo "OPENPI_PYTHON : ${OPENPI_PYTHON:-<unset>}"
echo "PYTHON_BIN     : ${PYTHON_BIN}"
echo "================================================"
echo "MAX_SAMPLES_PER_SKILL: ${MAX_SAMPLES_PER_SKILL}"
echo "MAX_SAMPLES_PER_SKILL_TASK: ${MAX_SAMPLES_PER_SKILL_TASK}"
echo "MAX_TOTAL_JOBS: ${MAX_TOTAL_JOBS}"
echo "RESUME: ${RESUME}"

if [[ "${SWEEP_PHASE}" == "prepare" ]]; then
  if [[ "${NODE_RANK}" != "0" ]]; then
    echo "Skip prepare on non-zero node rank: ${NODE_RANK}"
    exit 0
  fi
  "${PYTHON_BIN}" scripts/run_skill_metric_multinode_sweep.py --mode prepare "${PY_ARGS[@]}"
  exit 0
fi

if [[ "${SWEEP_PHASE}" == "merge" ]]; then
  "${PYTHON_BIN}" scripts/run_skill_metric_multinode_sweep.py --mode merge "${PY_ARGS[@]}"
  exit 0
fi

"${PYTHON_BIN}" scripts/run_skill_metric_multinode_sweep.py --mode launch "${PY_ARGS[@]}"
