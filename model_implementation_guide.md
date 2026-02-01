# Model Implementation Guide (VLM2 + Pi-0.5)

Last updated: 2026-01-31

## Scope
This guide summarizes the VLM2 implementation in openpi-comet, lists key files/scripts and their purposes, and tracks current progress plus the next execution plan.

## File & Script Map
### Core VLM2 modules
- src/openpi/models_pytorch/vlm2/view_consistent_3d.py
  - View-consistent 3D-aware representation: adaptive 3D position injection, viewpoint-aware geometry alignment, semantic-geometric fusion.
- src/openpi/models_pytorch/vlm2/dual_memory.py
  - Dual memory system: working memory (sliding window), episodic memory (fixed capacity with τ/α gated update), memory attention and fusion.
- src/openpi/models_pytorch/vlm2/vlm2_model.py
  - VLM2WithPi05 model: integrates VLM2 perception + memory with Pi-0.5 PaliGemma backbone and Gemma expert decoder.
- src/openpi/models_pytorch/vlm2/__init__.py
  - Exports VLM2 modules.

### Training & configuration
- scripts/train_pytorch.py
  - PyTorch training entrypoint; supports VLM2 selection via pytorch_model_name=vlm2; prepares VLM2 inputs and runs DDP.
- src/openpi/training/config.py
  - TrainConfig definitions and VLM2-specific hyperparameters; includes VLM2 SFT config presets.
- scripts/compute_norm_stats.py
  - Computes normalization stats for a given config.

### Supporting references
- VLM2_Implementation_Guide.md
  - Algorithm details and design rationale used to align implementation.
- task_context.md
  - Running log of VLM2 implementation decisions, tests, and status.
- README.md
  - Setup and training guidance; dataset requirements and training/eval flows.

## Implementation Status
### A. 3D-Aware Representation
- Implemented: Adaptive 3D Position Injection, Viewpoint-Aware Geometry Alignment, Semantic-Geometric Fusion.
- Location: src/openpi/models_pytorch/vlm2/view_consistent_3d.py
- Status: Complete; used by VLM2PerceptionModule.

### B. Dual-Memory Module
- Implemented: working memory + episodic memory (τ/α gating, LRU replacement) and gated fusion.
- Location: src/openpi/models_pytorch/vlm2/dual_memory.py
- Status: Complete; integrated in VLM2WithPi05.

### C. VLM2 + Pi-0.5 Integration
- Implemented: VLM2WithPi05, 3D encoder placeholder, flow-matching action head, prompt+vision fusion.
- Location: src/openpi/models_pytorch/vlm2/vlm2_model.py
- Status: Complete; used by scripts/train_pytorch.py.

### D. Training Integration
- Implemented: VLM2 selection + input preparation in PyTorch training; VLM2 config entry.
- Locations: scripts/train_pytorch.py, src/openpi/training/config.py
- Status: Complete; smoke SFT runs initiated.

## Current Status
- OOM and in-place autograd errors resolved for the 8-GPU smoke SFT by using bf16 autocast, out-of-place memory updates, and full gradient checkpointing.
- Warnings remain (DDP grad-stride mismatch, pynvml deprecation) but training completes 5 steps.

## Next Steps
1. Run a longer SFT to validate stability (e.g., 100–500 steps) with bf16 + DDP.
2. Monitor memory and throughput; address DDP grad-stride warning if it impacts performance.
3. Proceed to evaluation or downstream fine-tuning as needed.

## Quick Run Notes
- Training uses openpi env (.venv). For OOM, prefer bf16 + DDP with 8 GPUs.
- Dataset for VLM2 SFT demo: /mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos
- VLM2 SFT config name: vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft
