# Model Implementation Guide (VLM2 + Pi-0.5)

Last updated: 2026-02-24

## Scope
This guide summarizes the VLM2 implementation in openpi-comet, lists key files/scripts and their purposes, and tracks the current status plus execution plan. It also documents differences between the original VLM2 paper (Liu et al., 2025, "Vision-Language Memory for Spatial Reasoning") and this adaptation for robot SFT with Pi-0.5.

---

## File & Script Map

### Core VLM2 modules
- `src/openpi/models_pytorch/vlm2/view_consistent_3d.py`  
  View-Consistent 3D-Aware Representation: Adaptive 3D Position Injection (A3PI), Viewpoint-Aware Geometry Alignment (VAGA), Semantic-Geometric cross-attention Fusion.
- `src/openpi/models_pytorch/vlm2/dual_memory.py`  
  Dual Memory Module: WorkingMemory (sliding window FIFO storing H_t), EpisodicMemory (fixed-capacity bank storing M_t with similarity-based replacement), GatedMemoryFusion (γ = σ(MLP(Concat[M_w;M_e]))), QueryFusion (visual+text cross-attention for retrieval query).
- `src/openpi/models_pytorch/vlm2/vlm2_model.py`  
  VLM2WithPi05: integrates VLM2 perception + memory with Pi-0.5 PaliGemma backbone and Gemma Expert action decoder. Includes `forward()` (training) and `sample_actions()` (inference).
- `src/openpi/models_pytorch/vlm2/vggt_integration.py`  
  VGGT3DEncoder: loads VGGT checkpoint, runs VGGT backbone to produce geometry tokens, view tokens, and point maps.
- `src/openpi/models_pytorch/vlm2/__init__.py`  
  Module exports.

### Training & configuration
- `scripts/train_pytorch.py`  
  PyTorch training entrypoint; supports VLM2 via `pytorch_model_name=vlm2`; handles input preparation, DDP, gradient checkpointing.
- `src/openpi/training/sft_make_pizza_config.py`  
  SFT configs for VLM2 and Pi-0.5 baseline on the `make_pizza` task.
- `src/openpi/training/train_config.py`  
  TrainConfig dataclass + all experiment configs; tyro CLI entrypoint.
- `scripts/compute_norm_stats.py`  
  Computes normalization stats for a given config.

---

## Paper vs. Implementation Mapping

### Paper architecture (VLM2 on LLaVA-Video-7B)
- Vision encoder: SigLIP (via LLaVA-Video)
- 3D foundation model: **π3** [52] → geometry tokens G_t, view tokens Z_t, point maps X_t
- Temporal input: monocular video V = {I_t}^N, N frames sampled uniformly
- Memory: true temporal — shared across time steps in a single inference pass

### This implementation (VLM2WithPi05 for robot SFT)
- Vision encoder: SigLIP (via PaliGemma, frozen)
- 3D foundation model: **VGGT** (paper reports π3 outperforms VGGT by ~0.8 pts on VSI-Bench)
- Temporal input: 3 simultaneous camera views (base, left_wrist, right_wrist) stacked as "frames"
- Memory: reset every `forward()` call → memory operates as **multi-camera aggregation**, not temporal memory

### Hyperparameters (per paper ablation Table 7, best config)
| Parameter | Paper optimal | Config |
|---|---|---|
| Working memory size L_w | 8 | 8 ✓ |
| Episodic memory capacity L_e | 32 | 32 ✓ |
| Episodic similarity τ | — | 0.7 |
| Episodic fusion α | — | 0.5 |

---

## Algorithm 1 — Paper vs. Code

Paper Algorithm 1 (Dual-Memory Module, per-frame):
```
Input: H_t, W_t, E_t  (where H_t = 3D-aware representation)
1. M_w_t = WorkingAttention(Q=H_t, KV=W_t)
2. M_e_t = EpisodicAttention(Q=H_t, KV=E_t)
3. γ_t = σ(MLP(Concat[M_w_t; M_e_t]))
4. M_t  = γ_t ⊙ M_w_t + (1 - γ_t) ⊙ M_e_t       ← output
5. W_{t+1} ← FIFO-add(W_t, H_t)                    ← store H_t
6. E_{t+1} ← similarity-replace(E_t, M_t)           ← store M_t
```

Code deviations (intentional, documented):
| Step | Paper | Code | Reason |
|---|---|---|---|
| Retrieval query | H_t | QueryFusion(H_t, text) | Robot SFT: text-guided retrieval |
| Output | M_t (gated fusion) | LN(H_t + M_t) | Residual + LayerNorm for stability |
| Working memory stores | H_t | H_t (cast dtype) | ✓ Matches paper |
| **Episodic memory stores** | **M_t** | **M_t** (fused_output) | **Fixed 2026-02-24; was incorrectly H_t** |
| Episodic replacement | argmax-similarity replace | merge-if-sim>τ / LRU-if-novel | Richer diversity policy |

---

## Implementation Status

### A. 3D-Aware Representation
- Status: **Complete**  
- Adaptive 3D Position Injection (A3PI), Viewpoint-Aware Geometry Alignment (VAGA), Semantic-Geometric cross-attention Fusion all implemented.  
- Location: `src/openpi/models_pytorch/vlm2/view_consistent_3d.py`

### B. Dual-Memory Module
- Status: **Complete (bugs fixed)**  
- Working memory (FIFO): stores H_t ✓  
- Episodic memory: now correctly stores M_t (gated fusion output) per paper Algorithm 1  
- Empty memory early-return: now applies `layer_norm` before returning for output-distribution consistency  
- Location: `src/openpi/models_pytorch/vlm2/dual_memory.py`

### C. VLM2 + Pi-0.5 Integration
- Status: **Complete (bugs fixed)**  
- Location: `src/openpi/models_pytorch/vlm2/vlm2_model.py`  
- Key fixes applied (2026-02-24):
  1. `repr_to_llm`: Identity when `visual_dim == llm_dim`, avoiding random projection of pretrained features  
  2. `perception_delta_scale` / `memory_delta_scale`: zero-initialized unconstrained scalars (ReZero-style); start at 0 so model behaves identically to Pi-0.5 baseline at step 0 and gradually learns to blend VLM2 features  
  3. `forward()`: aggregates **all 3 camera frames** (`b t n d → b (t n) d`) instead of last-frame only  
  4. `sample_actions()`: fixed to use same all-frame aggregation as `forward()` (was taking `[:, -1]`), eliminating train/inference seq-len mismatch  
  5. `sample_actions()`: fixed `Identity` crash (`self.repr_to_llm.weight` → `getattr(...)`)

### D. Training Integration
- Status: **Complete**  
- VLM2 input preparation in `scripts/train_pytorch.py`; SFT config in `src/openpi/training/sft_make_pizza_config.py`

---

## Observed Training Diagnostics

Comparing step-0 metrics from `make_pizza` 5-epoch SFT runs:

| Run | loss (step 0) | grad_norm (step 0) |
|---|---|---|
| Pi-0.5 baseline | 0.0166 | 0.71 |
| VLM2 (before fixes) | 4.0545 | 454.06 |
| VLM2 (after fixes) | **TBD (smoke test needed)** | **TBD** |

Root cause of instability (all fixed): random `repr_to_llm` projection + uninitialized residual gates + empty-memory returning `query` instead of zeros.

VGGT loading stats: `matched=1401, dropped_by_shape=396, missing=2`. 396 dropped weights indicate a shape mismatch in VGGT (possibly resolution-dependent position embeddings); VGGT backbone is frozen so this affects geometry feature quality but not training stability.

---

## Next Steps

1. **Smoke test** (priority): Run 100–300 steps of `vlm2_b1k-make_pizza_lr2.5e-6_5ep_sft` and verify step-0 `loss ≈ 0.02` and `grad_norm < 5`.
2. **Monitor gate learning**: Log `perception_delta_scale` and `memory_delta_scale` during training; confirm they grow from 0.
3. **VGGT shape mismatch**: Investigate `dropped_by_shape=396` in `vggt_integration.py` to identify which weight groups mismatch; may require adjusting VGGT input resolution or token count.
4. Proceed to full 5-epoch SFT once smoke test passes.

---

## Quick Run Notes

- Training uses `openpi-comet-nas` conda env.
- Dataset: `/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/`
- Config names: `vlm2_b1k-make_pizza_lr2.5e-6_5ep_sft`, `pi05_b1k-make_pizza_lr2.5e-6_5ep_sft`
- Launch: `bash scripts/run_vlm2_sft_make_pizza_5ep.sh`
- For OOM: use bf16 + DDP across 4 GPUs with gradient checkpointing (already enabled by default).
