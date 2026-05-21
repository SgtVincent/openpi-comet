#!/usr/bin/env bash
set -euo pipefail

# One-click warmup for BEHAVIOR-1K challenge tasks, then pack cache into a tar.gz on NAS.
#
# Usage:
#   bash /mnt/bn/behavior-data-hl/chenjunting/repo/setup_bashrc.sh
#   cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet
#   bash scripts/warmup_behavior1k_cache.sh
#
# Common overrides:
#   GPU_ID=0 PARTIAL_SCENE_LOAD=1 STEPS=1 \
#   CACHE_TAR=/mnt/bn/navigation-hl/mlx/users/chenjunting/behavior1k_cache_gpu0.tar.gz \
#   bash scripts/warmup_behavior1k_cache.sh

REPO_ROOT="/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet"
cd "${REPO_ROOT}"

# Make progress visible in nohup logs.
export PYTHONUNBUFFERED=1

RUNTIME_ENV_FILE="${HOME}/.openpi_runtime_env.sh"
if [[ -f "${RUNTIME_ENV_FILE}" ]]; then
  # Prepared by /mnt/bn/behavior-data-hl/chenjunting/repo/setup_bashrc.sh
  # Expects BEHAVIOR_PYTHON / OPENPI_PYTHON exported there.
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV_FILE}"
fi

BEHAVIOR_PYTHON="${BEHAVIOR_PYTHON:-/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/behavior/bin/python}"
[[ -x "${BEHAVIOR_PYTHON}" ]] || { echo "[Error] BEHAVIOR_PYTHON not found: ${BEHAVIOR_PYTHON}" >&2; exit 1; }

GPU_IDS="${GPU_IDS:-}"
PARTIAL_SCENE_LOAD="${PARTIAL_SCENE_LOAD:-1}"
STEPS="${STEPS:-1}"

# Keep appdata on local disk (/tmp). The tar.gz is persisted on NAS.
APPDATA_BASE="${APPDATA_BASE:-/tmp/omnigibson-appdata}"

# Persist the tar on NAS.
# Default to an "all gpus" bundle when GPU_IDS is empty (auto-detect).
CACHE_TAR_DEFAULT="/mnt/bn/navigation-hl/mlx/users/chenjunting/behavior1k_cache_all_gpus.tar.gz"
if [[ -n "${GPU_IDS}" ]]; then
  CACHE_TAR_DEFAULT="/mnt/bn/navigation-hl/mlx/users/chenjunting/behavior1k_cache_gpu_ids_${GPU_IDS//,/_}.tar.gz"
fi
CACHE_TAR="${CACHE_TAR:-$CACHE_TAR_DEFAULT}"

LOG_DIR="${LOG_DIR:-/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/warmup_logs/behavior1k_cache_$(date +%Y%m%d_%H%M%S)}"

echo "=========================================================="
echo "BEHAVIOR-1K Cache Warmup (One-Click)"
echo "=========================================================="
echo "BEHAVIOR_PYTHON     : ${BEHAVIOR_PYTHON}"
echo "GPU_IDS             : ${GPU_IDS:-auto}"
echo "PARTIAL_SCENE_LOAD  : ${PARTIAL_SCENE_LOAD}"
echo "STEPS (per task)    : ${STEPS}"
echo "APPDATA_BASE        : ${APPDATA_BASE}"
echo "CACHE_TAR           : ${CACHE_TAR}"
echo "LOG_DIR             : ${LOG_DIR}"
echo "=========================================================="
echo "[Info] Note: the final tar is written atomically. While warmup is running you may see ${CACHE_TAR}.tmp"
echo "[Info] Do not try to untar until warmup finishes and ${CACHE_TAR} exists without .tmp"

ARGS=(
  "--appdata-base" "${APPDATA_BASE}"
  "--cache-tar" "${CACHE_TAR}"
  "--log-dir" "${LOG_DIR}"
  "--steps" "${STEPS}"
  "--keep-going"
)
if [[ -n "${GPU_IDS}" ]]; then
  ARGS+=("--gpu-ids" "${GPU_IDS}")
fi
if [[ "${PARTIAL_SCENE_LOAD}" == "1" ]]; then
  ARGS+=("--partial-scene-load")
fi

echo "[Info] Warming caches for 50 challenge tasks. This can take hours on first run."
echo "[Info] Command: ${BEHAVIOR_PYTHON} scripts/warmup_behavior1k_cache.py ${ARGS[*]}"
echo ""

"${BEHAVIOR_PYTHON}" "scripts/warmup_behavior1k_cache.py" "${ARGS[@]}"

echo ""
echo "[Info] Done. Cache tar saved at: ${CACHE_TAR}"
