#!/bin/bash
set -e

# Activate environment
source .venv/bin/activate

# Set cache path for NAS environment
export OPENPI_DATA_HOME=$(pwd)/.cache/openpi

# Optional debugging flags
# export PYTHONFAULTHANDLER=1
# export TORCH_SHOW_CPP_STACKTRACES=1

# Use video_reader for RGB-only training (faster/stable)
export B1K_VIDEO_BACKEND=video_reader

echo "Starting 8-GPU DDP Training..."
echo "Config: vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k"
echo "Exp Name: vlm2_vla_pretrain"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py \
  vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
  --exp_name vlm2_vla_pretrain \
  --num_train_steps 50000 \
  --log_interval 100 \
  --save_interval 5000 \
  --num_workers 10 \
  --pytorch-training-precision bfloat16

echo "Training finished!"
