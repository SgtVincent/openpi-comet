"""π0.5-KI joint query training: two-phase training loop with dual optimizer.

Implements the training loop for PI05KIJointQueryPytorch with:
  - Two-phase forward/backward (backbone first, then expert) — no retain_graph
  - Separate AdamW optimizers for backbone and expert parameters
  - Checkpoint save/load with both optimizer states + RNG state + config validation
  - Per-step metrics with KI diagnostic norms (backbone_only, flow_to_backbone)
  - Single-GPU training_step (fully tested)
  - DDP-aware training_step_ddp (structured, needs 2+ rank validation)

Memory model
============
Backbone graph is built, backward is called, then all intermediate tensors
go out of scope and are freed.  Only then does the expert forward build its
graph.  This means at most one 3.6B-param graph is in memory at once,
versus two for a naive combined loss approach.

Knowledge Insulation (KI)
=========================
When ``model.knowledge_insulation=True``, the expert's prefix KV is
detached, so ``flow_loss.backward()`` produces zero backbone gradients.

Correct KI ON/OFF comparison uses unified gradient accumulation:
  1. Zero BOTH optimizers
  2. Backbone forward + backward → backbone grads accumulate
  3. Expert forward + backward:
     - KI=ON: adds zero backbone grads
     - KI=OFF: adds flow→backbone grads (cross-attn KV path)
  4. Clip, step BOTH optimizers once

This ensures both modes see the same data and same backbone-loss gradient —
only the flow→backbone routing differs, making KI ON/OFF a clean comparison.

DDP status
==========
Single-GPU: fully tested, production-ready.

Multi-GPU DDP: ``training_step_ddp`` is implemented with ``no_sync()`` for
phase 1 + normal sync for phase 2 (one allreduce per step).  Not yet
validated on 2+ ranks — known considerations:
  - ``find_unused_parameters=True`` may be needed (backbone params unused
    by expert forward when KI=ON)
  - ``static_graph=True`` not safe with two distinct forward passes
  - DDP wraps ``model.forward()``; phase-specific methods
    (``compute_backbone_losses`` / ``compute_expert_loss``) are called on
    the inner model and bypass DDP's forward hooks — the ``no_sync()``
    pattern compensates by deferring allreduce to the final backward
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
from torch import optim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Dual optimizer setup
# ---------------------------------------------------------------------------


def setup_dual_optimizers(
    model: nn.Module,
    *,
    lr_backbone: float = 2.5e-5,
    lr_expert: float = 5e-5,
    weight_decay: float = 1e-10,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> tuple[optim.AdamW, optim.AdamW]:
    """Create two separate AdamW optimizers for backbone and expert.

    Uses :meth:`PI05KIJointQueryPytorch.get_backbone_params` and
    :meth:`PI05KIJointQueryPytorch.get_expert_params` to partition parameters.

    Args:
        model: a PI05KIJointQueryPytorch instance (or anything with
            ``get_backbone_params()`` / ``get_expert_params()`` methods).
        lr_backbone: learning rate for backbone optimizer.
        lr_expert: learning rate for expert optimizer.
        weight_decay: AdamW weight decay (used for both).
        betas: Adam beta1, beta2 (used for both).
        eps: Adam epsilon (used for both).

    Returns:
        ``(optimizer_backbone, optimizer_expert)``

    Raises:
        ValueError: if parameter sets overlap or are incomplete.
    """
    if not hasattr(model, "get_backbone_params") or not hasattr(model, "get_expert_params"):
        raise TypeError(
            "setup_dual_optimizers requires a model with get_backbone_params() "
            "and get_expert_params() methods (e.g. PI05KIJointQueryPytorch)."
        )

    bb_params = list(model.get_backbone_params())
    ex_params = list(model.get_expert_params())

    # Verify zero overlap by id()
    bb_ids = {id(p) for p in bb_params}
    ex_ids = {id(p) for p in ex_params}
    overlap = bb_ids & ex_ids
    if overlap:
        raise ValueError(
            f"Backbone and expert parameter sets overlap by {len(overlap)} "
            f"parameters. KI requires disjoint parameter sets."
        )

    # Verify full coverage of requires_grad params
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    all_ids = bb_ids | ex_ids
    trainable_ids = {id(p) for p in all_trainable}
    missing = trainable_ids - all_ids
    if missing:
        raise ValueError(
            f"{len(missing)} trainable parameters are not assigned to either "
            f"backbone or expert optimizer."
        )

    bb_count = sum(p.numel() for p in bb_params)
    ex_count = sum(p.numel() for p in ex_params)
    logger.info(
        "Dual optimizers: backbone=%d params (%.2fM), expert=%d params (%.2fM), "
        "lr_bb=%.2e, lr_ex=%.2e",
        len(bb_params),
        bb_count / 1e6,
        len(ex_params),
        ex_count / 1e6,
        lr_backbone,
        lr_expert,
    )

    optim_bb = optim.AdamW(
        bb_params,
        lr=lr_backbone,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    optim_ex = optim.AdamW(
        ex_params,
        lr=lr_expert,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return optim_bb, optim_ex


# ---------------------------------------------------------------------------
#  Single optimizer with two param groups (ZeRO/Accelerate compatible)
# ---------------------------------------------------------------------------


def setup_param_group_optimizer(
    model: nn.Module,
    *,
    lr_backbone: float = 2.5e-5,
    lr_expert: float = 5e-5,
    weight_decay: float = 1e-10,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.AdamW:
    """Create a single AdamW with two param groups (backbone + expert).

    This is the ZeRO/Accelerate-compatible variant: one optimizer object,
    two parameter groups with different learning rates.  DeepSpeed ZeRO and
    FSDP shard a single optimizer state across ranks, which is much more
    memory-efficient than two separate optimizers each replicated per GPU.

    KI still works because the gradient routing is structural (detached KV
    in the expert forward), not optimizer-based.  Whether the backbone
    receives flow-loss gradients depends on ``model.knowledge_insulation``,
    not on how the optimizer is partitioned.

    Args:
        model: PI05KIJointQueryPytorch instance (or anything with
            ``get_backbone_params()`` / ``get_expert_params()``).
        lr_backbone: LR for the backbone param group.
        lr_expert: LR for the expert param group.
        weight_decay: weight decay (applied to both groups).
        betas: Adam beta1, beta2.
        eps: Adam epsilon.

    Returns:
        Single AdamW optimizer with param groups ``[backbone_group, expert_group]``.

    Raises:
        ValueError: if parameter sets overlap or have gaps.
    """
    if not hasattr(model, "get_backbone_params") or not hasattr(model, "get_expert_params"):
        raise TypeError(
            "setup_param_group_optimizer requires a model with "
            "get_backbone_params() / get_expert_params() methods."
        )

    bb_params = list(model.get_backbone_params())
    ex_params = list(model.get_expert_params())

    # Verify zero overlap
    bb_ids = {id(p) for p in bb_params}
    ex_ids = {id(p) for p in ex_params}
    overlap = bb_ids & ex_ids
    if overlap:
        raise ValueError(
            f"Backbone and expert param groups overlap by {len(overlap)} params."
        )

    # Verify full coverage
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    all_ids = bb_ids | ex_ids
    trainable_ids = {id(p) for p in all_trainable}
    missing = trainable_ids - all_ids
    if missing:
        raise ValueError(f"{len(missing)} trainable params not in any group.")

    bb_count = sum(p.numel() for p in bb_params)
    ex_count = sum(p.numel() for p in ex_params)
    logger.info(
        "Param-group optimizer: backbone=%d params (%.2fM, lr=%.2e), "
        "expert=%d params (%.2fM, lr=%.2e)",
        len(bb_params), bb_count / 1e6, lr_backbone,
        len(ex_params), ex_count / 1e6, lr_expert,
    )

    optimizer = optim.AdamW(
        [
            {"params": bb_params, "lr": lr_backbone, "name": "backbone"},
            {"params": ex_params, "lr": lr_expert, "name": "expert"},
        ],
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return optimizer


def get_backbone_param_group(optimizer: optim.Optimizer) -> dict | None:
    """Return the backbone param group from a two-group optimizer."""
    for pg in optimizer.param_groups:
        if pg.get("name") == "backbone":
            return pg
    return None


def get_expert_param_group(optimizer: optim.Optimizer) -> dict | None:
    """Return the expert param group from a two-group optimizer."""
    for pg in optimizer.param_groups:
        if pg.get("name") == "expert":
            return pg
    return None


# ---------------------------------------------------------------------------
#  Cosine LR schedule (mirrors train_pytorch.py for consistency)
# ---------------------------------------------------------------------------


def make_cosine_lr_schedule(
    *,
    peak_lr: float,
    warmup_steps: int,
    decay_steps: int,
    end_lr: float = 0.0,
):
    """Return a ``step -> lr`` callable matching JAX warmup-cosine decay.

    Warmup is linear from ``peak_lr / (warmup_steps + 1)`` to ``peak_lr``.
    Decay is cosine from ``peak_lr`` to ``end_lr`` over ``decay_steps - warmup_steps``.
    """

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    return lr_schedule


# ---------------------------------------------------------------------------
#  Training step
# ---------------------------------------------------------------------------


def training_step(
    model: nn.Module,
    optimizer_bb: optim.Optimizer,
    optimizer_ex: optim.Optimizer,
    observation: Any,
    actions: torch.Tensor,
    *,
    step_idx: int,
    lr_schedule_bb,
    lr_schedule_ex,
    grad_clip_norm: float = 1.0,
    use_autocast: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
    autocast_device_type: str = "cuda",
) -> dict[str, float]:
    """Single two-phase training step with unified gradient accumulation.

    Correct KI ON/OFF comparison requires both backward passes to complete
    before either optimizer steps.  Order:

    1. Zero BOTH optimizers
    2. Backbone forward + backward → backbone grads accumulate on backbone params
       (backbone graph freed after backward; expert params have zero grad so far)
    3. Measure ``backbone_only_grad_norm`` (contribution from CE + query MSE only)
    4. Expert forward + backward:
       - KI=ON: expert grads only on expert params; zero additional on backbone
       - KI=OFF: flow grads ALSO flow back to backbone, adding to existing grads
       (expert graph freed after backward)
    5. Measure ``flow_to_backbone_grad_norm`` (delta added by flow → backbone path;
       zero for KI=ON, > 0 for KI=OFF — proves KI routing)
    6. Clip both param groups
    7. Step BOTH optimizers, then zero

    Memory model: still only one graph in memory at a time.  Backbone graph is
    freed (``del bb_losses``) before expert forward.  No ``retain_graph``.

    Why this order (not step-after-each-backward)
    ==============================================
    Step-after-each-backward is correct for KI=ON (backbone and expert are
    independent), but invalid for KI=OFF baseline comparison: with interleaved
    stepping, the flow-to-backbone gradient would never be applied because the
    backbone optimizer already stepped and zeroed.  This means KI ON vs OFF
    would differ in **both** gradient routing AND update order — you couldn't
    isolate the KI effect.

    Unified order gives apples-to-apples comparison: both modes see the same
    data, the same backbone loss grad, and only differ in whether the flow
    loss contributes additional backbone gradients.

    DDP note: repeated forward/backward on a DDP-wrapped model may need
    ``no_sync`` context manager or ``find_unused_parameters``.  Single-GPU
    is always fine.

    Args:
        model: PI05KIJointQueryPytorch instance.
        optimizer_bb: backbone optimizer.
        optimizer_ex: expert optimizer.
        observation: model-compatible observation dict/object.
        actions: ``[B, T, D]`` float32 action tensor.
        step_idx: current global step (for LR scheduling).
        lr_schedule_bb: callable ``step -> lr`` for backbone.
        lr_schedule_ex: callable ``step -> lr`` for expert.
        grad_clip_norm: max global gradient norm (0 = disabled).
        use_autocast: whether to enable torch.autocast.
        autocast_dtype: autocast precision dtype.
        autocast_device_type: autocast device type.

    Returns:
        Dict of scalar metrics for this step.
    """
    t0 = time.time()

    # --- LR update (both optimizers) ---
    lr_bb = lr_schedule_bb(step_idx)
    lr_ex = lr_schedule_ex(step_idx)
    for pg in optimizer_bb.param_groups:
        pg["lr"] = lr_bb
    for pg in optimizer_ex.param_groups:
        pg["lr"] = lr_ex

    bb_params = [p for p in model.get_backbone_params() if p.requires_grad]
    ex_params = [p for p in model.get_expert_params() if p.requires_grad]

    # ------------------------------------------------------------------
    #  Step 1: Zero both optimizers
    # ------------------------------------------------------------------
    optimizer_bb.zero_grad(set_to_none=True)
    optimizer_ex.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    #  Step 2: Backbone forward + backward (Phase 1)
    # ------------------------------------------------------------------
    with torch.autocast(
        device_type=autocast_device_type,
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        bb_losses = model.compute_backbone_losses(observation, actions)

    bb_total = bb_losses["backbone_loss"]
    bb_total = bb_total.float()  # ensure float32 for backward
    bb_total.backward()

    # Capture backbone-only grad norm (CE + query MSE contribution only)
    backbone_only_grad_norm = _compute_grad_norm(bb_params)

    # Detach scalars for logging
    ce_loss = float(bb_losses["ce_loss"].item())
    query_mse_loss = float(bb_losses["query_mse_loss"].item())
    bb_loss_val = float(bb_losses["backbone_loss"].item())

    # Free backbone graph before expert forward — only one graph in memory
    del bb_losses, bb_total

    # ------------------------------------------------------------------
    #  Step 3: Expert forward + backward (Phase 2)
    #
    #  KI=ON: expert backward adds zero backbone grads
    #  KI=OFF: flow grads flow through cross-attn KV to backbone layers
    # ------------------------------------------------------------------
    with torch.autocast(
        device_type=autocast_device_type,
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        ex_losses = model.compute_expert_loss(observation, actions)

    ex_total = ex_losses["expert_loss"]
    ex_total = ex_total.float()
    ex_total.backward()

    # Measure flow-to-backbone contribution (delta after expert backward)
    #  - KI=ON: this should be ~0 (no additional backbone grad from flow)
    #  - KI=OFF: this is > 0 (flow loss adds to backbone grads)
    total_bb_grad_norm = _compute_grad_norm(bb_params)
    flow_to_backbone_grad_norm = max(
        0.0, total_bb_grad_norm - backbone_only_grad_norm
    )

    # Expert grad norm (total after expert backward)
    expert_grad_norm = _compute_grad_norm(ex_params)

    flow_loss_val = float(ex_losses["flow_loss"].item())
    expert_loss_val = float(ex_losses["expert_loss"].item())

    # Free expert graph
    del ex_losses, ex_total

    # ------------------------------------------------------------------
    #  Step 4: Gradient clipping (both groups)
    # ------------------------------------------------------------------
    if grad_clip_norm > 0:
        # Clip each group separately to its own max norm
        # (clip_grad_norm_ already returns pre-clip norm, which we already have)
        torch.nn.utils.clip_grad_norm_(bb_params, max_norm=grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(ex_params, max_norm=grad_clip_norm)

    # ------------------------------------------------------------------
    #  Step 5: Step both optimizers, then zero
    # ------------------------------------------------------------------
    optimizer_bb.step()
    optimizer_ex.step()

    # Clean up gradients
    optimizer_bb.zero_grad(set_to_none=True)
    optimizer_ex.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    #  Metrics
    # ------------------------------------------------------------------
    step_time = time.time() - t0
    total_loss = bb_loss_val + expert_loss_val

    # KI-related flags
    ki_enabled = bool(getattr(model, "knowledge_insulation", False))
    trunc_kv = bool(getattr(model, "truncate_expert_kv", True))

    # Memory usage (CUDA only)
    mem_peak_mb = 0.0
    mem_alloc_mb = 0.0
    if torch.cuda.is_available():
        mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mem_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        # Reset peak tracker for next step
        torch.cuda.reset_peak_memory_stats()

    return {
        # Losses
        "ce_loss": ce_loss,
        "query_mse_loss": query_mse_loss,
        "flow_loss": flow_loss_val,
        "backbone_loss": bb_loss_val,
        "expert_loss": expert_loss_val,
        "total_loss": total_loss,
        # Gradient norms — total
        "backbone_grad_norm": total_bb_grad_norm,
        "expert_grad_norm": expert_grad_norm,
        # Gradient norms — KI diagnostics
        # backbone_only = CE + query MSE contribution (before expert backward)
        "backbone_only_grad_norm": backbone_only_grad_norm,
        # flow_to_backbone = delta added by flow loss → backbone path
        #   KI=ON  → ~0  (proves KI is working)
        #   KI=OFF → > 0 (flow grads reach backbone)
        "flow_to_backbone_grad_norm": flow_to_backbone_grad_norm,
        # Learning rates
        "lr_backbone": lr_bb,
        "lr_expert": lr_ex,
        # Timing
        "step_time": step_time,
        # Memory
        "mem_peak_mb": mem_peak_mb,
        "mem_alloc_mb": mem_alloc_mb,
        # Flags
        "knowledge_insulation": float(ki_enabled),
        "truncate_expert_kv": float(trunc_kv),
    }


def _compute_grad_norm(params) -> float:
    """Compute global L2 norm of gradients (for no-clip path)."""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(p.grad.detach().data.norm(2).item()) ** 2
    return total_sq ** 0.5


# ---------------------------------------------------------------------------
#  Single-optimizer training step (ZeRO / Accelerate compatible)
# ---------------------------------------------------------------------------


def training_step_single_opt(
    model: nn.Module,
    optimizer: optim.Optimizer,
    observation: Any,
    actions: torch.Tensor,
    *,
    step_idx: int,
    lr_schedule_bb,
    lr_schedule_ex,
    grad_clip_norm: float = 1.0,
    use_autocast: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
    autocast_device_type: str = "cuda",
) -> dict[str, float]:
    """Single-optimizer two-phase training step (ZeRO/Accelerate compatible).

    Uses one AdamW with two param groups (backbone + expert).  This is the
    memory-efficient path for large models on V100/A100 with DeepSpeed ZeRO
    or FSDP: one optimizer state, sharded across GPUs.

    Two-phase gradient accumulation (same correct KI ON/OFF semantics):
      1. Zero optimizer
      2. Backbone forward + backward → grads on backbone params only
         (expert params not in the backbone graph → naturally zero grad)
      3. Measure ``backbone_only_grad_norm`` (CE + query MSE contribution)
      4. Free backbone graph
      5. Expert forward + backward → grads on expert params only (KI=ON)
         or → grads on expert + additional backbone grads (KI=OFF)
      6. Measure total grad norms and flow→backbone contribution
      7. Clip per-group
      8. Step once, zero

    Memory: one graph at a time (backbone freed before expert forward).
    KI: structural (detached KV in expert forward), not optimizer-based.

    Optimizer param groups must have ``name='backbone'`` and ``name='expert'``.
    Use :func:`setup_param_group_optimizer` to create one.

    This is the recommended path for GPU runs.  The dual-optimizer variant
    (:func:`training_step`) is kept for unit tests, small models, and cases
    where per-optimizer sharding is beneficial.

    Args:
        model: PI05KIJointQueryPytorch instance.
        optimizer: single AdamW with 2 named param groups.
        observation: model-compatible observation.
        actions: [B, T, D] action tensor.
        step_idx: global step index.
        lr_schedule_bb: backbone LR schedule callable.
        lr_schedule_ex: expert LR schedule callable.
        grad_clip_norm: max gradient norm per group.
        use_autocast: whether to enable autocast.
        autocast_dtype: autocast dtype.
        autocast_device_type: autocast device type.

    Returns:
        Dict of scalar metrics (same keys as training_step).
    """
    t0 = time.time()

    # --- LR update per param group ---
    lr_bb = lr_schedule_bb(step_idx)
    lr_ex = lr_schedule_ex(step_idx)
    for pg in optimizer.param_groups:
        if pg.get("name") == "backbone":
            pg["lr"] = lr_bb
        elif pg.get("name") == "expert":
            pg["lr"] = lr_ex

    bb_params = [p for p in model.get_backbone_params() if p.requires_grad]
    ex_params = [p for p in model.get_expert_params() if p.requires_grad]

    # ------------------------------------------------------------------
    #  Step 1: Zero
    # ------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    #  Step 2: Backbone forward + backward
    # ------------------------------------------------------------------
    with torch.autocast(
        device_type=autocast_device_type,
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        bb_losses = model.compute_backbone_losses(observation, actions)

    bb_total = bb_losses["backbone_loss"].float()
    bb_total.backward()

    # Backbone-only grad norm (before expert backward)
    backbone_only_grad_norm = _compute_grad_norm(bb_params)

    # Capture scalars, free backbone graph
    ce_loss = float(bb_losses["ce_loss"].item())
    query_mse_loss = float(bb_losses["query_mse_loss"].item())
    bb_loss_val = float(bb_losses["backbone_loss"].item())
    del bb_losses, bb_total

    # ------------------------------------------------------------------
    #  Step 3: Expert forward + backward
    # ------------------------------------------------------------------
    with torch.autocast(
        device_type=autocast_device_type,
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        ex_losses = model.compute_expert_loss(observation, actions)

    ex_total = ex_losses["expert_loss"].float()
    ex_total.backward()

    # Total grad norms after expert backward
    total_bb_grad_norm = _compute_grad_norm(bb_params)
    expert_grad_norm = _compute_grad_norm(ex_params)
    flow_to_backbone_grad_norm = max(0.0, total_bb_grad_norm - backbone_only_grad_norm)

    flow_loss_val = float(ex_losses["flow_loss"].item())
    expert_loss_val = float(ex_losses["expert_loss"].item())
    del ex_losses, ex_total

    # ------------------------------------------------------------------
    #  Step 4: Clip + step
    # ------------------------------------------------------------------
    if grad_clip_norm > 0:
        # Clip each group separately
        torch.nn.utils.clip_grad_norm_(bb_params, max_norm=grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(ex_params, max_norm=grad_clip_norm)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    #  Metrics
    # ------------------------------------------------------------------
    step_time = time.time() - t0
    total_loss = bb_loss_val + expert_loss_val
    ki_enabled = bool(getattr(model, "knowledge_insulation", False))
    trunc_kv = bool(getattr(model, "truncate_expert_kv", True))

    mem_peak_mb = 0.0
    mem_alloc_mb = 0.0
    if torch.cuda.is_available():
        mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mem_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()

    return {
        "ce_loss": ce_loss,
        "query_mse_loss": query_mse_loss,
        "flow_loss": flow_loss_val,
        "backbone_loss": bb_loss_val,
        "expert_loss": expert_loss_val,
        "total_loss": total_loss,
        "backbone_grad_norm": total_bb_grad_norm,
        "expert_grad_norm": expert_grad_norm,
        "backbone_only_grad_norm": backbone_only_grad_norm,
        "flow_to_backbone_grad_norm": flow_to_backbone_grad_norm,
        "lr_backbone": lr_bb,
        "lr_expert": lr_ex,
        "step_time": step_time,
        "mem_peak_mb": mem_peak_mb,
        "mem_alloc_mb": mem_alloc_mb,
        "knowledge_insulation": float(ki_enabled),
        "truncate_expert_kv": float(trunc_kv),
    }


# ---------------------------------------------------------------------------
#  DDP-aware training step
# ---------------------------------------------------------------------------

# Distributed training status (cached, set once)
_USE_DDP = None


def is_ddp() -> bool:
    """Return True if torch.distributed is initialized."""
    global _USE_DDP
    if _USE_DDP is None:
        try:
            import torch.distributed as dist
            _USE_DDP = dist.is_available() and dist.is_initialized()
        except ImportError:
            _USE_DDP = False
    return _USE_DDP


def training_step_ddp(
    model: nn.Module,
    optimizer_bb: optim.Optimizer,
    optimizer_ex: optim.Optimizer,
    observation: Any,
    actions: torch.Tensor,
    *,
    step_idx: int,
    lr_schedule_bb,
    lr_schedule_ex,
    grad_clip_norm: float = 1.0,
    use_autocast: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
    autocast_device_type: str = "cuda",
    ddp_wrapper: Any = None,
) -> dict[str, float]:
    """DDP-aware two-phase training step.

    **Correct DDP usage**: PyTorch DDP hooks into ``model.forward()`` and
    triggers gradient allreduce after the last backward pass.  With two
    forward/backward phases per step, we must:

    1. Phase 1 (backbone): run under ``no_sync()`` — gradients accumulate
       locally, no allreduce.  The backbone-only gradients don't need
       cross-rank sync at this point.
    2. Phase 2 (expert): run normally (sync enabled) — DDP triggers
       allreduce of ALL accumulated gradients (both phases) after this
       backward completes.

    This gives one allreduce per step (same as single-phase training)
    while maintaining the two-phase memory model.

    **Important**: ``model`` must be the *inner* model (not DDP-wrapped)
    for ``compute_backbone_losses`` / ``compute_expert_loss`` to work,
    because DDP only wraps ``forward()``.  The ``ddp_wrapper`` parameter
    is the DDP-wrapped model used for ``no_sync()`` context management.

    **Unused-parameter warning**: With KI=ON, the backbone parameters are
    not used by the expert forward pass (cross-attn uses detached KV).
    This means DDP sees unused parameters in the expert phase.  Set
    ``find_unused_parameters=True`` on the DDP wrapper, or use
    ``static_graph=False`` with unused param awareness.

    Args:
        model: inner PI05KIJointQueryPytorch model (unwrapped).
        optimizer_bb: backbone optimizer.
        optimizer_ex: expert optimizer.
        observation: model-compatible observation.
        actions: [B, T, D] action tensor.
        step_idx: global step index.
        lr_schedule_bb: backbone LR schedule callable.
        lr_schedule_ex: expert LR schedule callable.
        grad_clip_norm: max global gradient norm per group.
        use_autocast: whether to enable autocast.
        autocast_dtype: autocast dtype.
        autocast_device_type: autocast device type.
        ddp_wrapper: the DDP-wrapped model (for no_sync context).
            If None, falls back to single-GPU training_step behavior.

    Returns:
        Dict of scalar metrics (same keys as training_step).
    """
    if ddp_wrapper is None or not is_ddp():
        # Single GPU / no DDP — reuse non-DDP implementation
        return training_step(
            model, optimizer_bb, optimizer_ex, observation, actions,
            step_idx=step_idx,
            lr_schedule_bb=lr_schedule_bb,
            lr_schedule_ex=lr_schedule_ex,
            grad_clip_norm=grad_clip_norm,
            use_autocast=use_autocast,
            autocast_dtype=autocast_dtype,
            autocast_device_type=autocast_device_type,
        )

    t0 = time.time()

    # --- LR update ---
    lr_bb = lr_schedule_bb(step_idx)
    lr_ex = lr_schedule_ex(step_idx)
    for pg in optimizer_bb.param_groups:
        pg["lr"] = lr_bb
    for pg in optimizer_ex.param_groups:
        pg["lr"] = lr_ex

    bb_params = [p for p in model.get_backbone_params() if p.requires_grad]
    ex_params = [p for p in model.get_expert_params() if p.requires_grad]

    # --- Zero both optimizers ---
    optimizer_bb.zero_grad(set_to_none=True)
    optimizer_ex.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    #  Phase 1: Backbone forward + backward (under no_sync)
    #  Gradients accumulate locally; no allreduce yet.
    # ------------------------------------------------------------------
    with ddp_wrapper.no_sync():
        with torch.autocast(
            device_type=autocast_device_type,
            dtype=autocast_dtype,
            enabled=use_autocast,
        ):
            bb_losses = model.compute_backbone_losses(observation, actions)

        bb_total = bb_losses["backbone_loss"].float()
        bb_total.backward()

    backbone_only_grad_norm = _compute_grad_norm(bb_params)

    # Capture metrics, free backbone graph
    ce_loss = float(bb_losses["ce_loss"].item())
    query_mse_loss = float(bb_losses["query_mse_loss"].item())
    bb_loss_val = float(bb_losses["backbone_loss"].item())
    del bb_losses, bb_total

    # ------------------------------------------------------------------
    #  Phase 2: Expert forward + backward (normal sync)
    #  DDP will allreduce ALL accumulated gradients (both phases) here.
    # ------------------------------------------------------------------
    with torch.autocast(
        device_type=autocast_device_type,
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        ex_losses = model.compute_expert_loss(observation, actions)

    ex_total = ex_losses["expert_loss"].float()
    ex_total.backward()

    # After expert backward, DDP has allreduced all gradients
    total_bb_grad_norm = _compute_grad_norm(bb_params)
    expert_grad_norm = _compute_grad_norm(ex_params)
    flow_to_backbone_grad_norm = max(0.0, total_bb_grad_norm - backbone_only_grad_norm)

    flow_loss_val = float(ex_losses["flow_loss"].item())
    expert_loss_val = float(ex_losses["expert_loss"].item())
    del ex_losses, ex_total

    # --- Gradient clipping ---
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(bb_params, max_norm=grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(ex_params, max_norm=grad_clip_norm)

    # --- Step both optimizers ---
    optimizer_bb.step()
    optimizer_ex.step()
    optimizer_bb.zero_grad(set_to_none=True)
    optimizer_ex.zero_grad(set_to_none=True)

    # --- Metrics ---
    step_time = time.time() - t0
    total_loss = bb_loss_val + expert_loss_val
    ki_enabled = bool(getattr(model, "knowledge_insulation", False))
    trunc_kv = bool(getattr(model, "truncate_expert_kv", True))

    mem_peak_mb = 0.0
    mem_alloc_mb = 0.0
    if torch.cuda.is_available():
        mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mem_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()

    return {
        "ce_loss": ce_loss,
        "query_mse_loss": query_mse_loss,
        "flow_loss": flow_loss_val,
        "backbone_loss": bb_loss_val,
        "expert_loss": expert_loss_val,
        "total_loss": total_loss,
        "backbone_grad_norm": total_bb_grad_norm,
        "expert_grad_norm": expert_grad_norm,
        "backbone_only_grad_norm": backbone_only_grad_norm,
        "flow_to_backbone_grad_norm": flow_to_backbone_grad_norm,
        "lr_backbone": lr_bb,
        "lr_expert": lr_ex,
        "step_time": step_time,
        "mem_peak_mb": mem_peak_mb,
        "mem_alloc_mb": mem_alloc_mb,
        "knowledge_insulation": float(ki_enabled),
        "truncate_expert_kv": float(trunc_kv),
    }


# ---------------------------------------------------------------------------
#  Checkpoint save / load
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CheckpointState:
    """Container for all state that goes in a checkpoint."""

    global_step: int
    model_state_dict: dict
    optimizer_bb_state_dict: dict
    optimizer_ex_state_dict: dict
    metadata: dict  # extra info (config, timestamp, etc.)


def save_checkpoint(
    model: nn.Module,
    optimizer_bb: optim.Optimizer,
    optimizer_ex: optim.Optimizer,
    global_step: int,
    checkpoint_dir: str | Path,
    *,
    metadata: dict | None = None,
    is_main: bool = True,
    save_rng_state: bool = True,
    config_snapshot: dict | None = None,
) -> Path:
    """Save a joint-training checkpoint atomically.

    Structure::

        checkpoint_dir/<step>/
            model.safetensors    model state_dict (safetensors format)
            optimizer_bb.pt      backbone optimizer state
            optimizer_ex.pt      expert optimizer state
            rng_state.pt         Python, NumPy, torch CPU, torch CUDA RNG states
            metadata.pt          global_step + config snapshot + extra metadata

    RNG state (when ``save_rng_state=True``):
      - ``random``: Python stdlib random.getstate()
      - ``numpy``: numpy.random.get_state()
      - ``torch_cpu``: torch.get_rng_state()
      - ``torch_cuda``: list of torch.cuda.get_rng_state(i) for each visible device

    Config snapshot (when ``config_snapshot`` provided):
      - Key fields (knowledge_insulation, truncate_expert_kv, architecture flags)
        are validated on load to prevent silent KI-ON ↔ KI-OFF mismatches.

    Args:
        model: PI05KIJointQueryPytorch instance (possibly DDP-wrapped).
        optimizer_bb: backbone optimizer.
        optimizer_ex: expert optimizer.
        global_step: current training step.
        checkpoint_dir: base directory for checkpoints.
        metadata: optional extra dict to save in metadata.pt.
        is_main: if False, skip the actual save (DDP rank filter).
        save_rng_state: whether to save RNG state for deterministic resume.
        config_snapshot: dict of key config fields (e.g. dataclass.asdict(config)).

    Returns:
        Path to the saved checkpoint directory (main rank only, else final_dir path).
    """
    checkpoint_dir = Path(checkpoint_dir)
    final_dir = checkpoint_dir / str(global_step)
    tmp_dir = checkpoint_dir / f"tmp_{global_step}"

    if not is_main:
        return final_dir

    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Model (safetensors, handle DDP wrapper)
    model_to_save = model.module if hasattr(model, "module") else model
    safetensors.torch.save_model(model_to_save, str(tmp_dir / "model.safetensors"))

    # Optimizers
    torch.save(optimizer_bb.state_dict(), tmp_dir / "optimizer_bb.pt")
    torch.save(optimizer_ex.state_dict(), tmp_dir / "optimizer_ex.pt")

    # RNG state
    if save_rng_state:
        rng_state = _get_rng_state_dict()
        torch.save(rng_state, tmp_dir / "rng_state.pt")

    # Config snapshot from model + provided dict
    model_config = {}
    for attr in ("knowledge_insulation", "truncate_expert_kv", "beta_text",
                 "beta_query", "flow_loss_weight", "num_query_tokens",
                 "query_emb_dim"):
        val = getattr(model_to_save, attr, None)
        if val is not None:
            model_config[attr] = val

    full_metadata = {
        "global_step": global_step,
        "timestamp": time.time(),
        "model_config": model_config,
        "pi05_ki_joint_trainer_version": 2,
        **(metadata or {}),
    }
    if config_snapshot is not None:
        full_metadata["config_snapshot"] = config_snapshot
    torch.save(full_metadata, tmp_dir / "metadata.pt")

    # Atomic rename
    if final_dir.exists():
        import shutil
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)

    logger.info("Saved checkpoint at step %d -> %s", global_step, final_dir)
    return final_dir


def load_checkpoint(
    model: nn.Module,
    optimizer_bb: optim.Optimizer,
    optimizer_ex: optim.Optimizer,
    checkpoint_dir: str | Path,
    device: torch.device | str,
    *,
    step: int | None = None,
    restore_rng_state: bool = True,
    validate_config: bool = True,
) -> int:
    """Load a joint-training checkpoint.

    Validates that the checkpoint was saved with a compatible configuration:
    if ``validate_config=True``, checks ``knowledge_insulation`` and
    ``truncate_expert_kv`` match the current model, raising ``ValueError``
    if they differ (prevents silent KI-ON ↔ KI-OFF resume bugs).

    Args:
        model: PI05KIJointQueryPytorch instance (possibly DDP-wrapped).
        optimizer_bb: backbone optimizer (state restored in-place).
        optimizer_ex: expert optimizer (state restored in-place).
        checkpoint_dir: base checkpoint directory.
        device: device to load tensors onto.
        step: specific step to load (None = latest).
        restore_rng_state: whether to restore Python/NumPy/torch RNG state.
        validate_config: whether to validate KI/architecture flags match.

    Returns:
        global_step of the loaded checkpoint.

    Raises:
        FileNotFoundError: if no valid checkpoint exists.
        ValueError: if config validation fails (KI flag mismatch, etc.).
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_to_load = model.module if hasattr(model, "module") else model

    if step is None:
        step = _find_latest_checkpoint_step(checkpoint_dir)
        if step is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    ckpt_dir = checkpoint_dir / str(step)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist")

    # Load metadata first (for validation)
    meta_path = ckpt_dir / "metadata.pt"
    metadata = {}
    if meta_path.exists():
        metadata = torch.load(meta_path, map_location=device, weights_only=False)
        saved_model_config = metadata.get("model_config", {})

        # Config validation
        if validate_config and saved_model_config:
            _validate_checkpoint_config(model_to_load, saved_model_config)
    else:
        logger.warning("No metadata.pt found at %s", ckpt_dir)

    # Load model
    model_path = ckpt_dir / "model.safetensors"
    if model_path.exists():
        safetensors.torch.load_model(model_to_load, str(model_path), device=str(device))
        logger.info("Loaded model state from %s", model_path)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load backbone optimizer
    bb_path = ckpt_dir / "optimizer_bb.pt"
    if bb_path.exists():
        bb_state = torch.load(bb_path, map_location=device, weights_only=False)
        optimizer_bb.load_state_dict(bb_state)
        del bb_state
        logger.info("Loaded backbone optimizer state from %s", bb_path)
    else:
        raise FileNotFoundError(f"Backbone optimizer checkpoint not found at {bb_path}")

    # Load expert optimizer
    ex_path = ckpt_dir / "optimizer_ex.pt"
    if ex_path.exists():
        ex_state = torch.load(ex_path, map_location=device, weights_only=False)
        optimizer_ex.load_state_dict(ex_state)
        del ex_state
        logger.info("Loaded expert optimizer state from %s", ex_path)
    else:
        raise FileNotFoundError(f"Expert optimizer checkpoint not found at {ex_path}")

    # Restore RNG state
    rng_path = ckpt_dir / "rng_state.pt"
    if restore_rng_state and rng_path.exists():
        rng_state = torch.load(rng_path, map_location=device, weights_only=False)
        _set_rng_state_dict(rng_state)
        logger.info("Restored RNG state from checkpoint")
    elif restore_rng_state and not rng_path.exists():
        logger.warning(
            "restore_rng_state=True but no rng_state.pt in checkpoint; "
            "resumed run will not be bit-exact."
        )

    loaded_step = int(metadata.get("global_step", step))
    logger.info("Loaded checkpoint at step %d", loaded_step)
    return loaded_step


def _validate_checkpoint_config(model: nn.Module, saved_config: dict) -> None:
    """Validate checkpoint config matches current model settings.

    Raises ValueError with detailed message on mismatch.
    """
    mismatches = []

    for key in ("knowledge_insulation", "truncate_expert_kv"):
        current = bool(getattr(model, key, None))
        saved = bool(saved_config.get(key))
        if key in saved_config and current != saved:
            mismatches.append(
                f"  {key}: current={current}, saved={saved}"
            )

    if mismatches:
        raise ValueError(
            "Checkpoint configuration mismatch — resuming would silently change "
            "training semantics. Mismatched fields:\n" + "\n".join(mismatches) +
            "\nSet validate_config=False to override (not recommended)."
        )


# ---------------------------------------------------------------------------
#  RNG state helpers
# ---------------------------------------------------------------------------


def _get_rng_state_dict() -> dict:
    """Capture all RNG states needed for deterministic resume.

    Returns dict with keys:
      - python: random.getstate()
      - numpy: numpy.random.get_state()
      - torch_cpu: torch.get_rng_state()
      - torch_cuda: list of states, one per CUDA device (empty if no CUDA)
      - num_cuda_devices: number of visible CUDA devices at save time
    """
    import random

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
        "num_cuda_devices": 0,
    }

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        state["num_cuda_devices"] = num_devices
        state["torch_cuda"] = [
            torch.cuda.get_rng_state(i) for i in range(num_devices)
        ]

    return state


def _set_rng_state_dict(state: dict) -> None:
    """Restore RNG state from a dict produced by _get_rng_state_dict.

    Warns (does not error) if CUDA device count differs between save and load,
    since this can happen legitimately when resuming on different hardware.
    """
    import random

    # Python stdlib
    if "python" in state:
        random.setstate(state["python"])

    # NumPy
    if "numpy" in state:
        np.random.set_state(state["numpy"])

    # Torch CPU
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])

    # Torch CUDA
    if "torch_cuda" in state and torch.cuda.is_available():
        saved_devices = state.get("num_cuda_devices", len(state["torch_cuda"]))
        current_devices = torch.cuda.device_count()

        if saved_devices != current_devices:
            logger.warning(
                "CUDA device count mismatch: checkpoint saved with %d devices, "
                "current has %d. CUDA RNG state will not be restored exactly — "
                "random-number-dependent parts of training (e.g. dropout, noise "
                "sampling) will differ.",
                saved_devices, current_devices,
            )
            return

        for i, s in enumerate(state["torch_cuda"]):
            if i < current_devices:
                torch.cuda.set_rng_state(s, i)
    elif "torch_cuda" in state and not torch.cuda.is_available():
        logger.debug(
            "Checkpoint has CUDA RNG state but CUDA not available on this host; "
            "skipping CUDA RNG restore."
        )


def _find_latest_checkpoint_step(checkpoint_dir: Path) -> int | None:
    """Find the latest step number in a checkpoint directory."""
    if not checkpoint_dir.exists():
        return None
    steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(steps) if steps else None


# ---------------------------------------------------------------------------
#  Simple metrics logger
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Accumulate per-step metrics and emit periodic averages.

    Lightweight, wandb-optional.  Keeps a rolling buffer of the last N
    steps for averaging, and also exposes the full history.
    """

    def __init__(self, log_interval: int = 10, *, wandb_enabled: bool = False):
        self.log_interval = log_interval
        self.wandb_enabled = wandb_enabled
        self._buffer: list[dict[str, float]] = []
        self._history: list[dict[str, float]] = []

    def log_step(self, step: int, metrics: dict[str, float]) -> dict[str, float] | None:
        """Record metrics for one step.  Returns averaged summary every log_interval steps."""
        self._buffer.append(metrics)
        self._history.append(metrics)

        if step > 0 and step % self.log_interval == 0:
            avg = self._average_buffer()
            avg["step"] = step
            self._buffer.clear()

            if self.wandb_enabled:
                try:
                    import wandb

                    wandb.log(avg, step=step)
                except ImportError:
                    pass

            return avg
        return None

    def _average_buffer(self) -> dict[str, float]:
        if not self._buffer:
            return {}
        keys = self._buffer[0].keys()
        return {k: sum(d[k] for d in self._buffer) / len(self._buffer) for k in keys}

    @property
    def history(self) -> list[dict[str, float]]:
        return list(self._history)
