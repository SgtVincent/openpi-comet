#!/usr/bin/env bash
# Environment wrapper for the offline validation tools in this directory.
#
# It reproduces the env that scripts/run_pi05_ki_joint_query_full_b1k_bf16_*.sh
# sets up for training, so that offline evaluation uses the SAME data loading,
# normalization and preprocessing as the training-time validation. Only the
# sampling differs (see verify_val_subset.py / eval_multi_ckpt.py).
#
# Usage:
#   scripts/val_eval/run_val_eval.sh <script.py> [args...]
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 scripts/val_eval/run_val_eval.sh \
#       scripts/val_eval/verify_val_subset.py
#
# Required in the environment (no defaults, they are site-specific):
#   CONDA_ROOT   root of a conda install that has the openpi env
#   CONDA_ENV    conda env name                      (default: openpi-comet-nas)
#   B1K_DATASET_ROOT  BEHAVIOR-1K challenge demos root
#
# Optional:
#   LOCAL_CACHE_ROOT  node-local scratch for HF/torch/triton caches
#                     (default: /tmp/openpi-val-eval/$USER)
set -uo pipefail

# Repo root derived from this script's location, so it works from any checkout.
REPO_ROOT="${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

CONDA_ROOT="${CONDA_ROOT:-}"
CONDA_ENV="${CONDA_ENV:-openpi-comet-nas}"
if [[ -z "$CONDA_ROOT" ]]; then
  echo "ERROR: set CONDA_ROOT to a conda install containing env '$CONDA_ENV'." >&2
  exit 2
fi
if [[ -z "${B1K_DATASET_ROOT:-}" ]]; then
  echo "ERROR: set B1K_DATASET_ROOT to the BEHAVIOR-1K challenge-demos root." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export JAX_PLATFORMS=cpu
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Node-local caches: the BN/NAS mount is shared and slow for many small writes.
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/openpi-val-eval/${USER:-tiger}}"
mkdir -p "${LOCAL_CACHE_ROOT}"
export OPENPI_DATA_HOME="${LOCAL_CACHE_ROOT}/openpi"
export HF_HOME="${LOCAL_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRITON_CACHE_DIR="${LOCAL_CACHE_ROOT}/triton/autotune"
export XDG_CACHE_HOME="${LOCAL_CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${LOCAL_CACHE_ROOT}/matplotlib"
export TORCH_HOME="${LOCAL_CACHE_ROOT}/torch"
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"; mkdir -p "${TMPDIR}"
export TMP="${TMPDIR}"; export TEMP="${TMPDIR}"

export OPENPI_BEHAVIOR_DATASET_ROOT="${B1K_DATASET_ROOT}"
export B1K_VIDEO_BACKEND="${B1K_VIDEO_BACKEND:-video_reader}"
export OPENPI_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_HF_DATASETS_CACHE_PER_RANK=1

# NOTE: OPENPI_B1K_ANCHOR_STRIDE / _OFFSET / _DROP_INCOMPLETE_HORIZON are left
# UNSET on purpose. Training-time validation is built inside
# `_baseline_b1k_dataset_env()` (see scripts/train_accelerate.py), which pops
# exactly these keys to force the baseline stride-1 / no-drop contract. Leaving
# them unset here reproduces that contract, so offline numbers stay comparable
# with the in-training val metrics.

cd "${REPO_ROOT}"
exec python "$@"
