#!/bin/bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas

echo "[pi05_hybrid_validation] START"
echo "[pi05_hybrid_validation] python=$(which python)"
python --version

python -u scripts/test_pi05_hybrid_feature.py "$@"
status=$?

echo "[pi05_hybrid_validation] EXIT_STATUS=${status}"
exit "${status}"