---
name: "openpi-training-helper"
description: "Assists with OpenPi VLM2 training (DDP), troubleshooting, and config. Invoke when running training, debugging errors (RoPE, dependencies), or configuring experiments."
---

# OpenPi VLM2 Training Helper

This skill assists with running and debugging VLM2 training experiments in the OpenPi Comet project.

## 🚀 Common Commands

### 1. 8-GPU DDP Pre-training
Use this for the actual training run.

```bash
source .venv/bin/activate
export OPENPI_DATA_HOME=$(pwd)/.cache/openpi
export B1K_VIDEO_BACKEND=video_reader
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py \
  vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
  --exp_name vlm2_vla_pretrain \
  --num_train_steps 50000 \
  --log_interval 100 \
  --save_interval 5000 \
  --num_workers 10 \
  --pytorch-training-precision bfloat16 \
  --overwrite
```

## 🛠️ Troubleshooting & Key Modifications

### 1. RoPE Shape Mismatch
**Symptom**: `RuntimeError: The size of tensor a (456) must match the size of tensor b (488)`
**Cause**: `position_ids` includes prefix+suffix, but `inputs_embeds` missing suffix.
**Fix**: In `vlm2_model.py`, ensure `inputs_embeds=[prefix_embs, suffix_embs]`.

### 2. Training Instability / `use_cache`
**Symptom**: Errors related to cache updating during training.
**Fix**: Ensure `use_cache=False` is passed to the model forward pass during training.

### 3. NAS Data Loading Slowness
**Symptom**: Training hangs at startup or "Loaded metadata" for a long time.
**Fix**:
- Set `check_files=False` in `BehaviorLeRobotDataset` (modified in `data_loader.py`).
- Set `check_timestamp_sync=False`.
- Use a higher `num_workers` (e.g., 10 per GPU for DDP).

### 4. Dependencies (`vggt`, `cut3r`)
**Symptom**: `ImportError` or "Missing third_party dependencies".
**Fix**: Ensure submodules are pulled and `vggt_integration.py` adds them to `sys.path`.

### 5. Checkpoint Strict Loading
**Policy**: `strict=True` is enforced. If keys mismatch, training will abort. Check model definition if this happens.

## 📊 Benchmarks & Hyperparameters
- **Batch Size**: 64 (global) -> `bs=64` in config name.
- **Learning Rate**: 2.5e-5.
- **Precision**: `bfloat16`.
- **Dataset**: `behavior-1k/2025-challenge-demos`.
