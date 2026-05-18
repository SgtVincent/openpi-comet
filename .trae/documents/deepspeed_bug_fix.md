# DeepSpeed Bug-Fix Notes (Internal)

This file contains implementation details and historical bug-fix rationale for
Accelerate + DeepSpeed training in this repo. It is intentionally *not* part of
the user-facing guide.

## Scope

- Repo: openpi-comet
- Focus: Accelerate + DeepSpeed ZeRO-2 on V100 FP16

## Key Fixes (Why They Exist)

### 1) Avoid "double mixed precision" (DeepSpeed FP16 + torch_autocast)

Problem:
- Enabling DeepSpeed `fp16.enabled=true` while also enabling `torch_autocast`
  creates two competing mixed-precision systems. This can lead to unpredictable
  behavior (dtype mismatch, scaling oddities, instability).

Fix:
- Force `torch_autocast.enabled=false` when using DS fp16/bf16 engines.
- Ensure the DS config is consistent with the requested precision.

Related:
- `configs/deepspeed_zero2_v100_fp16.json`
- `scripts/train_accelerate.py` config patch + validation helpers

### 2) DeepSpeed engine autocast interaction (dtype mismatch)

Problem:
- In some DeepSpeed versions, the engine forward wrapper can detect an outer
  `torch.autocast` and disable it, causing float32-activation vs float16-weight
  mismatches.

Fix:
- Patch DeepSpeedEngine autocast helpers to reflect the active outer autocast
  context (rather than forcibly disabling it).

Related:
- `scripts/train_accelerate.py` `_patch_deepspeed_autocast()`

### 3) Keep training when dynamic loss scale reaches min (avoid fatal exit)

Problem:
- Some DS builds hard-fail when loss scale hits `min_loss_scale` via
  `raise_error_at_min_scale=True`. For long FP16 runs, a single bad batch can
  drive the scaler down and kill the entire job, losing progress.

Fix:
- Patch `DynamicLossScaler.__init__` so `raise_error_at_min_scale=False` and
  continue with the standard behavior: overflow -> skip step -> reduce scale.

Related:
- `scripts/train_accelerate.py` `_patch_deepspeed_loss_scaler()`

### 4) Flow matching loss overflow (V100 FP16)

Problem:
- Flow matching often uses an MSE on velocity-like tensors. Intermediate values
  can exceed FP16 max (~65504) before unscale, causing overflows and repeated
  skipped steps / loss scale collapse.

Fix:
- Compute the MSE in fp32 (cast inputs to float before `F.mse_loss`).

Related:
- `src/openpi/models_pytorch/pi0_pytorch.py`
- `src/openpi/models_pytorch/pi05_subtask.py`
- `src/openpi/models_pytorch/vlm2/vlm2_model.py`

### 5) Multi-node EXP_NAME sync race (Arnold retries)

Problem:
- Multi-node launch used a shared `EXP_NAME_FILE` keyed by a RUN_KEY that may be
  identical across retries. Non-rank0 could read a stale file from a crashed run
  before rank0 overwrote it, leading to split output dirs. This then cascades
  into checkpoint save timeouts (waiting for tmp dirs in a different exp path).

Fix:
- Add a per-node start-sentinel + mtime validation so non-rank0 only accepts an
  EXP_NAME file written after the current run starts; rank0 proactively removes
  stale files before writing.

Related:
- `scripts/run_pi05_sft_accelerate_deepspeed_multinode_v100_fp16.sh`

## Notes

- This file is intentionally not staged by default. Treat it as internal
  context for reviewers and future debugging.

