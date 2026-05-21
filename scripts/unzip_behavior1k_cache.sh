#!/usr/bin/env bash
set -euo pipefail

# Unpack a prebuilt BEHAVIOR-1K OmniGibson cache tarball into local /tmp.
# Intended to be inserted before evaluation commands on a fresh node/container.
#
# Usage (recommended):
#   bash /mnt/bn/behavior-data-hl/chenjunting/repo/setup_bashrc.sh
#   cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet
#   bash scripts/unzip_behavior1k_cache.sh
#   bash scripts/run_eval_make_pizza_3models.sh
#
# Overrides:
#   GPU_ID=0 CACHE_TAR=/path/to/cache.tar.gz DEST=/tmp bash scripts/unzip_behavior1k_cache.sh
#   bash scripts/unzip_behavior1k_cache.sh /path/to/cache.tar.gz

REPO_ROOT="/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
DEST="${DEST:-/tmp}"

DEFAULT_CACHE_TAR="/mnt/bn/navigation-hl/mlx/users/chenjunting/behavior1k_cache_all_gpus.tar.gz"
CACHE_TAR="${CACHE_TAR:-$DEFAULT_CACHE_TAR}"
if (( $# >= 1 )); then
  CACHE_TAR="$1"
fi

# Backward compatible fallback: if the all-gpus bundle doesn't exist, try per-gpu tar.
if [[ ! -f "${CACHE_TAR}" && "${CACHE_TAR}" == "${DEFAULT_CACHE_TAR}" ]]; then
  FALLBACK="/mnt/bn/navigation-hl/mlx/users/chenjunting/behavior1k_cache_gpu${GPU_ID}.tar.gz"
  if [[ -f "${FALLBACK}" ]]; then
    CACHE_TAR="${FALLBACK}"
  fi
fi

USER_NAME="${USER:-default_user}"
TARGET_DIR="${DEST}/omnigibson-appdata/${USER_NAME}/gpu${GPU_ID}"

echo "=========================================================="
echo "Unzip BEHAVIOR-1K Cache"
echo "=========================================================="
echo "GPU_ID     : ${GPU_ID}"
echo "CACHE_TAR  : ${CACHE_TAR}"
echo "DEST       : ${DEST}"
echo "TARGET_DIR : ${TARGET_DIR}"
echo "=========================================================="

[[ -f "${CACHE_TAR}" ]] || { echo "[Error] cache tar not found: ${CACHE_TAR}" >&2; exit 1; }
mkdir -p "${DEST}"

if [[ -d "${TARGET_DIR}/global/cache" && -d "${TARGET_DIR}/local/cache" ]]; then
  echo "[Info] Cache already present, skip extracting: ${TARGET_DIR}"
  exit 0
fi

echo "[Info] Extracting cache tar (may take a while on first use)..."
tar -xzf "${CACHE_TAR}" -C "${DEST}"

if [[ -d "${TARGET_DIR}/global/cache" ]]; then
  echo "[Info] Cache extracted successfully: ${TARGET_DIR}"
else
  echo "[Warn] Extract finished but expected directory not found: ${TARGET_DIR}"
  echo "[Warn] You may need to verify the tar structure or DEST."
fi
