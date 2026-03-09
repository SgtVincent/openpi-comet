#!/usr/bin/env bash
# Training script for PI05_HYBRID model.
#
# Usage:
#   Single GPU:
#     bash scripts/run_pi05_hybrid_train.sh
#   Multi-GPU:
#     torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_hybrid_b1k-pt50_cs32_bs64_lr2.5e-5_5ep --exp_name pi05_hybrid_train
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_NAME="${1:-pi05_hybrid_b1k-pt50_cs32_bs64_lr2.5e-5_5ep}"
EXP_NAME="${2:-pi05_hybrid_train}"
NUM_GPUS="${3:-8}"

# Environment setup
export WANDB_PROJECT="${WANDB_PROJECT:-B1K}"
export OPENPI_PERSISTENT_WORKERS=1
export OPENPI_DATALOADER_TIMEOUT_S=300

echo "=== PI05 Hybrid Training ==="
echo "Config: ${CONFIG_NAME}"
echo "Exp: ${EXP_NAME}"
echo "GPUs: ${NUM_GPUS}"

cd "$ROOT_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
        scripts/train_pytorch.py \
        "$CONFIG_NAME" \
        --exp_name "$EXP_NAME" \
        --save_interval 5000
else
    python scripts/train_pytorch.py \
        "$CONFIG_NAME" \
        --exp_name "$EXP_NAME" \
        --save_interval 5000
fi
