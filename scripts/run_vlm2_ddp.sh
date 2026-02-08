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

# HuggingFace offline + cache (avoid network + avoid shared-cache lock contention)
export OPENPI_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OPENPI_PERSISTENT_WORKERS=0
export OPENPI_DATALOADER_TIMEOUT_S=600
export OPENPI_DATALOADER_PREFETCH_FACTOR=2

echo "Starting 8-GPU DDP Training..."
echo "Config: vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k"
echo "Exp Name: vlm2_vla_pretrain"

CONSOLE_LOG=checkpoints/console_logs/vlm2_vla_pretrain.log
mkdir -p "$(dirname "${CONSOLE_LOG}")"
TORCHRUN_LOG_DIR=checkpoints/torchrun_logs/vlm2_vla_pretrain
mkdir -p "${TORCHRUN_LOG_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port 29501 \
  --log_dir "${TORCHRUN_LOG_DIR}" --redirects 3 --tee 3 \
  scripts/train_pytorch.py \
  vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
  --exp_name vlm2_vla_pretrain \
  --num_train_steps 50000 \
  --log_interval 100 \
  --save_interval 5000 \
  --num_workers 10 \
  --pytorch-training-precision bfloat16 \
  --overwrite 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Training finished!"
