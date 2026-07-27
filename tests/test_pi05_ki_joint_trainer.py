"""Tests for π0.5-KI joint query training loop (dual optimizer + checkpoint + metrics).

Tests use the same _MiniJointModel architecture from test_pi05_ki_joint_query.py
to verify training loop correctness without loading HF models.

Covers:
1. Dual optimizer setup (partitioning, no overlap, full coverage)
2. Training step: two-phase forward/backward, no retain_graph, loss decreases
3. Knowledge insulation: expert backward → zero backbone grad
4. Checkpoint save / load round-trip
5. Resume idempotency: same seed + same data → identical loss after resume
6. Metrics collection
7. LR schedule correctness
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# Reuse the mini model from the sibling test file
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TEST_DIR))

from test_pi05_ki_joint_query import _MiniJointModel  # noqa: E402


# ===========================================================================
#  Helpers
# ===========================================================================


def _make_model(
    hidden_dim=32,
    num_heads=2,
    prefix_len=8,
    num_query_tokens=6,
    action_horizon=6,
    action_dim=4,
    knowledge_insulation=True,
    truncate_expert_kv=True,
    beta_text=1.0,
    beta_query=1.0,
    flow_loss_weight=10.0,
):
    """Create a mini joint model with get_backbone_params/get_expert_params."""
    model = _MiniJointModel(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        prefix_len=prefix_len,
        num_query_tokens=num_query_tokens,
        action_horizon=action_horizon,
        action_dim=action_dim,
        knowledge_insulation=knowledge_insulation,
        truncate_expert_kv=truncate_expert_kv,
        beta_text=beta_text,
        beta_query=beta_query,
        flow_loss_weight=flow_loss_weight,
    )
    # Add the helper methods expected by the trainer
    # Backbone: everything except expert transformer + action_in_proj
    model.get_backbone_params = lambda: [
        p for n, p in model.named_parameters()
        if not (n.startswith("expert.") or n.startswith("action_in_proj."))
    ]
    model.get_expert_params = lambda: [
        p for n, p in model.named_parameters()
        if n.startswith("expert.") or n.startswith("action_in_proj.")
    ]
    return model


def _make_batch(batch_size=4, prefix_len=8, action_horizon=6, action_dim=4, subtask_vocab=16):
    """Create a random batch.

    Returns (prefix_tokens, actions, subtask_targets).
    prefix_tokens indices are bounded by prefix_len (matches nn.Embedding size).
    subtask_targets indices are bounded by subtask_vocab (matches subtask_head output dim).
    """
    prefix_tokens = torch.randint(0, prefix_len, (batch_size, prefix_len))
    actions = torch.randn(batch_size, action_horizon, action_dim)
    subtask_targets = torch.randint(0, subtask_vocab, (batch_size, prefix_len))
    return prefix_tokens, actions, subtask_targets


def _total_grad_norm(params):
    """Compute global L2 norm of gradients for an iterable of parameters."""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(p.grad.detach().norm(2).item()) ** 2
    return total_sq ** 0.5


class _ObsWrapper:
    """Wraps batch tensors into an observation-like object.

    The real PI05KIJointQueryPytorch takes an observation object/namedtuple; for
    the mini model we intercept compute_backbone_losses / compute_expert_loss
    to unpack from this wrapper.
    """
    def __init__(self, prefix_tokens, subtask_targets):
        self.prefix_tokens = prefix_tokens
        self.subtask_targets = subtask_targets


def _wrap_model_for_trainer(model):
    """Monkey-patch compute_backbone_losses / compute_expert_loss to accept obs dict.

    The real model takes (observation, actions); the mini model takes
    (prefix_tokens, actions, subtask_targets).  We adapt the interface.
    """
    original_bb = model.compute_backbone_losses
    original_ex = model.compute_expert_loss

    def bb_forward(observation, actions, **kw):
        return original_bb(observation.prefix_tokens, actions, observation.subtask_targets)

    def ex_forward(observation, actions, noise=None, time=None, **kw):
        return original_ex(observation.prefix_tokens, actions, noise=noise, time=time)

    model.compute_backbone_losses = bb_forward
    model.compute_expert_loss = ex_forward
    return model


# ===========================================================================
#  Test 1: Dual optimizer partitioning
# ===========================================================================


class TestDualOptimizerSetup:
    def test_zero_overlap(self):
        """Backbone and expert param sets must be disjoint."""
        model = _make_model()
        bb_ids = {id(p) for p in model.get_backbone_params()}
        ex_ids = {id(p) for p in model.get_expert_params()}
        assert len(bb_ids & ex_ids) == 0, "Backbone and expert params overlap!"

    def test_full_coverage(self):
        """All model params must be in either backbone or expert set."""
        model = _make_model()
        all_ids = {id(p) for p in model.parameters()}
        bb_ids = {id(p) for p in model.get_backbone_params()}
        ex_ids = {id(p) for p in model.get_expert_params()}
        assert (bb_ids | ex_ids) == all_ids, (
            f"Missing params: {len(all_ids - bb_ids - ex_ids)}"
        )

    def test_query_tokens_in_backbone(self):
        """Query embeddings and query_action_head must be in backbone group."""
        model = _make_model()
        bb_ids = {id(p) for p in model.get_backbone_params()}
        assert id(model.query_embeddings) in bb_ids, "query_embeddings should be backbone param"
        assert id(model.query_action_head.weight) in bb_ids, "query_action_head.weight should be backbone param"

    def test_expert_params_in_expert(self):
        """Expert transformer params must be in expert group."""
        model = _make_model()
        ex_ids = {id(p) for p in model.get_expert_params()}
        assert id(model.expert.q_proj.weight) in ex_ids, "expert q_proj should be expert param"
        assert id(model.action_in_proj.weight) in ex_ids, "action_in_proj should be expert param"

    def test_setup_dual_optimizers(self):
        """setup_dual_optimizers creates two AdamW optimizers."""
        from openpi.training.pi05_ki_joint_trainer import setup_dual_optimizers

        model = _make_model()
        optim_bb, optim_ex = setup_dual_optimizers(
            model, lr_backbone=1e-4, lr_expert=2e-4,
        )
        assert isinstance(optim_bb, torch.optim.AdamW)
        assert isinstance(optim_ex, torch.optim.AdamW)
        assert optim_bb.param_groups[0]["lr"] == 1e-4
        assert optim_ex.param_groups[0]["lr"] == 2e-4
        assert len(optim_bb.param_groups) == 1
        assert len(optim_ex.param_groups) == 1

    def test_setup_fails_with_overlap(self):
        """setup_dual_optimizers raises ValueError if params overlap."""
        from openpi.training.pi05_ki_joint_trainer import setup_dual_optimizers

        model = _make_model()
        # Make them overlap
        model.get_backbone_params = lambda: list(model.parameters())
        model.get_expert_params = lambda: list(model.parameters())

        with pytest.raises(ValueError, match="overlap"):
            setup_dual_optimizers(model, lr_backbone=1e-4, lr_expert=2e-4)

    def test_setup_fails_with_missing(self):
        """setup_dual_optimizers raises ValueError if trainable params are missing."""
        from openpi.training.pi05_ki_joint_trainer import setup_dual_optimizers

        model = _make_model()
        # Only return subset
        all_params = list(model.parameters())
        model.get_backbone_params = lambda: [all_params[0]]
        model.get_expert_params = lambda: []

        with pytest.raises(ValueError, match="not assigned"):
            setup_dual_optimizers(model, lr_backbone=1e-4, lr_expert=2e-4)


# ===========================================================================
#  Test 2: Training step
# ===========================================================================


class TestTrainingStep:
    def _run_step(self, model, optim_bb, optim_ex, batch, step_idx=0, lr_schedule=None, grad_clip=1.0):
        """Run one training step via the trainer's training_step function.

        Note: model must already be wrapped via _wrap_model_for_trainer before
        the first call (calling it repeatedly would double-wrap).  We check
        for a sentinel attribute.
        """
        from openpi.training.pi05_ki_joint_trainer import training_step

        prefix_tokens, actions, subtask_targets = batch
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        # Wrap only once
        if not getattr(model, "_trainer_wrapped", False):
            _wrap_model_for_trainer(model)
            model._trainer_wrapped = True

        if lr_schedule is None:
            lr_schedule = lambda s: 1e-3

        metrics = training_step(
            model, optim_bb, optim_ex, obs, actions,
            step_idx=step_idx,
            lr_schedule_bb=lr_schedule,
            lr_schedule_ex=lr_schedule,
            grad_clip_norm=grad_clip,
            use_autocast=False,
            autocast_device_type="cpu",
        )
        return metrics

    def test_loss_decreases(self):
        """Loss should decrease after a few training steps."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)

        batch = _make_batch(batch_size=4)

        # Save original methods for evaluation
        original_bb = model.compute_backbone_losses
        original_ex = model.compute_expert_loss

        # Step 0 loss (before any update)
        with torch.no_grad():
            bb_losses_0 = original_bb(batch[0], batch[1], batch[2])
            ex_losses_0 = original_ex(batch[0], batch[1])
            total_0 = bb_losses_0["backbone_loss"].item() + ex_losses_0["expert_loss"].item()

        # Run 5 steps (this wraps the model)
        for i in range(5):
            self._run_step(model, optim_bb, optim_ex, batch, step_idx=i)

        # Step 5 loss — use original (unwrapped) methods for evaluation
        with torch.no_grad():
            bb_losses_5 = original_bb(batch[0], batch[1], batch[2])
            ex_losses_5 = original_ex(batch[0], batch[1])
            total_5 = bb_losses_5["backbone_loss"].item() + ex_losses_5["expert_loss"].item()

        assert total_5 < total_0, f"Loss did not decrease: {total_0:.4f} -> {total_5:.4f}"

    def test_no_retain_graph(self):
        """Training step should work without retain_graph=True (no errors)."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)

        batch = _make_batch(batch_size=2)
        metrics = self._run_step(model, optim_bb, optim_ex, batch, step_idx=0)
        assert "total_loss" in metrics

    def test_metrics_keys(self):
        """Metrics dict should contain all expected keys."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)
        batch = _make_batch(batch_size=2)

        metrics = self._run_step(model, optim_bb, optim_ex, batch, step_idx=0)

        expected_keys = [
            "ce_loss", "query_mse_loss", "flow_loss",
            "backbone_loss", "expert_loss", "total_loss",
            "backbone_grad_norm", "expert_grad_norm",
            "backbone_only_grad_norm", "flow_to_backbone_grad_norm",
            "lr_backbone", "lr_expert",
            "step_time", "mem_peak_mb", "mem_alloc_mb",
            "knowledge_insulation", "truncate_expert_kv",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: {key}"

    def test_ki_zero_backbone_grad_from_expert(self):
        """With KI=True, expert backward should produce zero backbone grads."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        batch = _make_batch(batch_size=2)

        # Zero all grads first
        for p in model.parameters():
            p.grad = None

        ex_losses = model.compute_expert_loss(batch[0], batch[1])
        ex_losses["expert_loss"].backward()

        # Check all backbone params have zero grad
        for p in model.get_backbone_params():
            if p.grad is not None:
                assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-9), (
                    f"Backbone param has non-zero grad from expert loss (KI=True)"
                )

    def test_no_ki_backbone_has_grad(self):
        """With KI=False, expert backward should produce non-zero backbone grads."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=False, truncate_expert_kv=False)
        batch = _make_batch(batch_size=2)

        for p in model.parameters():
            p.grad = None

        ex_losses = model.compute_expert_loss(batch[0], batch[1])
        ex_losses["expert_loss"].backward()

        # At least some backbone params should have non-zero grad
        has_grad = False
        for p in model.get_backbone_params():
            if p.grad is not None and p.grad.abs().sum() > 1e-9:
                has_grad = True
                break
        assert has_grad, "Expected non-zero backbone grads when KI=False"

    def test_separate_optimizers_step_separately(self):
        """Backbone optimizer only updates backbone params, expert only expert."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)

        # Save initial state
        bb_initial = {id(p): p.clone().detach() for p in model.get_backbone_params()}
        ex_initial = {id(p): p.clone().detach() for p in model.get_expert_params()}

        batch = _make_batch(batch_size=2)

        # Only step backbone optimizer
        optim_bb.zero_grad()
        bb_losses = model.compute_backbone_losses(batch[0], batch[1], batch[2])
        bb_losses["backbone_loss"].backward()
        optim_bb.step()

        # Backbone params should change
        bb_changed = False
        for p in model.get_backbone_params():
            if not torch.allclose(p, bb_initial[id(p)], atol=1e-9):
                bb_changed = True
                break
        assert bb_changed, "Backbone params should change after backbone step"

        # Expert params should NOT change
        for p in model.get_expert_params():
            assert torch.allclose(p, ex_initial[id(p)], atol=1e-9), (
                "Expert params should NOT change after only backbone step"
            )

        # Save current backbone state (after bb step) for comparison
        bb_after_bb_step = {id(p): p.clone().detach() for p in model.get_backbone_params()}

        # Now step expert optimizer
        optim_ex.zero_grad()
        ex_losses = model.compute_expert_loss(batch[0], batch[1])
        ex_losses["expert_loss"].backward()
        optim_ex.step()

        # Expert params should change
        ex_changed = False
        for p in model.get_expert_params():
            if not torch.allclose(p, ex_initial[id(p)], atol=1e-9):
                ex_changed = True
                break
        assert ex_changed, "Expert params should change after expert step"

        # Backbone params should NOT change from expert step (KI=True)
        for p in model.get_backbone_params():
            assert torch.allclose(p, bb_after_bb_step[id(p)], atol=1e-9), (
                "Backbone params should NOT change after expert step (KI=True)"
            )

    def test_gradient_clipping(self):
        """Gradient clipping should cap parameter update magnitude.

        We verify clipping by comparing param changes between clipped and
        unclipped runs — clipped run should have smaller update magnitudes.
        clip_grad_norm_ returns the total norm *before* clipping, so we
        verify the effect by looking at parameter deltas.
        """
        # Run with small clip value
        torch.manual_seed(42)
        model_clip = _make_model(knowledge_insulation=True)
        optim_clip_bb = torch.optim.AdamW(model_clip.get_backbone_params(), lr=1e-3)
        optim_clip_ex = torch.optim.AdamW(model_clip.get_expert_params(), lr=1e-3)

        # Run with no clip
        torch.manual_seed(42)
        model_noclip = _make_model(knowledge_insulation=True)
        optim_noclip_bb = torch.optim.AdamW(model_noclip.get_backbone_params(), lr=1e-3)
        optim_noclip_ex = torch.optim.AdamW(model_noclip.get_expert_params(), lr=1e-3)

        batch = _make_batch(batch_size=2)

        # Save initial params
        bb_initial_clip = {id(p): p.clone().detach() for p in model_clip.get_backbone_params()}
        bb_initial_noclip = {id(p): p.clone().detach() for p in model_noclip.get_backbone_params()}

        # Run clipped step
        metrics_clip = self._run_step(model_clip, optim_clip_bb, optim_clip_ex, batch, step_idx=0, grad_clip=0.001)

        # Run unclipped step
        metrics_noclip = self._run_step(model_noclip, optim_noclip_bb, optim_noclip_ex, batch, step_idx=0, grad_clip=0.0)

        # Compute total update magnitude for backbone params
        def total_delta(model, initial):
            total = 0.0
            for p in model.get_backbone_params():
                total += (p - initial[id(p)]).norm(2).item() ** 2
            return total ** 0.5

        delta_clip = total_delta(model_clip, bb_initial_clip)
        delta_noclip = total_delta(model_noclip, bb_initial_noclip)

        # Clipped update should be smaller than unclipped
        assert delta_clip < delta_noclip, (
            f"Clipped delta ({delta_clip}) should be < unclipped delta ({delta_noclip})"
        )

        # clip_grad_norm_ returns pre-clip norm, which should be > clip value for this test
        assert metrics_clip["backbone_grad_norm"] > 0.001, (
            "Expected pre-clip grad norm to exceed clip value"
        )

    def test_ki_on_flow_to_backbone_zero(self):
        """With KI=True, flow_to_backbone_grad_norm metric should be ~0."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True, truncate_expert_kv=True)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)
        batch = _make_batch(batch_size=4)

        metrics = self._run_step(model, optim_bb, optim_ex, batch, step_idx=0)

        # flow_to_backbone_grad_norm should be exactly zero (or extremely close)
        # when KI is enabled — the detached KV blocks all flow→backbone grads
        assert metrics["flow_to_backbone_grad_norm"] < 1e-6, (
            f"KI=ON but flow_to_backbone_grad_norm={metrics['flow_to_backbone_grad_norm']}"
        )
        # backbone_only_grad_norm should equal total backbone_grad_norm
        assert abs(metrics["backbone_grad_norm"] - metrics["backbone_only_grad_norm"]) < 1e-6, (
            "KI=ON but total backbone grad != backbone-only grad"
        )

    def test_ki_off_flow_to_backbone_positive(self):
        """With KI=False, flow_to_backbone_grad_norm metric should be > 0."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=False, truncate_expert_kv=False)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)
        batch = _make_batch(batch_size=4)

        metrics = self._run_step(model, optim_bb, optim_ex, batch, step_idx=0)

        # flow→backbone should contribute measurable gradient when KI is off
        assert metrics["flow_to_backbone_grad_norm"] > 1e-3, (
            f"KI=OFF but flow_to_backbone_grad_norm={metrics['flow_to_backbone_grad_norm']:.6f}"
        )
        # Total backbone grad should be > backbone-only grad
        assert metrics["backbone_grad_norm"] > metrics["backbone_only_grad_norm"], (
            "KI=OFF but total backbone grad not > backbone-only grad"
        )

    def test_ki_on_off_backbone_update_differs(self):
        """KI ON vs OFF should produce different backbone gradients.

        Both modes see the same data and same backbone loss gradient, but
        KI=OFF adds flow-loss gradients to backbone params.  The unified
        step order (both backwards, then both steps) ensures this is a
        clean comparison — only the gradient routing differs.

        We verify via gradient norms (before clipping/stepping) rather than
        parameter deltas, because AdamW's adaptive per-parameter learning
        rate can make total update norm non-monotonic with total gradient
        norm when gradient directions vary between modes.
        """
        batch = _make_batch(batch_size=4)

        # Create both models from the same seed
        torch.manual_seed(42)
        model_on = _make_model(knowledge_insulation=True, truncate_expert_kv=True)
        torch.manual_seed(42)
        model_off = _make_model(knowledge_insulation=False, truncate_expert_kv=False)

        # Verify identical initial state by state_dict
        sd_on = model_on.state_dict()
        sd_off = model_off.state_dict()
        for key in sd_on:
            assert torch.allclose(sd_on[key], sd_off[key]), (
                f"Initial params differ at {key}"
            )

        # Manual two-phase backward to compare backbone grad norms directly
        # KI=ON
        model_on.zero_grad(set_to_none=True)
        bb_on_losses = model_on.compute_backbone_losses(batch[0], batch[1], batch[2])
        bb_on_losses["backbone_loss"].backward()
        # Measure backbone-only grad norm
        bb_on_grad_before = _total_grad_norm(model_on.get_backbone_params())

        ex_on_losses = model_on.compute_expert_loss(batch[0], batch[1])
        ex_on_losses["expert_loss"].backward()
        # Measure total backbone grad after expert backward
        bb_on_grad_after = _total_grad_norm(model_on.get_backbone_params())

        # KI=OFF
        model_off.zero_grad(set_to_none=True)
        bb_off_losses = model_off.compute_backbone_losses(batch[0], batch[1], batch[2])
        bb_off_losses["backbone_loss"].backward()
        bb_off_grad_before = _total_grad_norm(model_off.get_backbone_params())

        ex_off_losses = model_off.compute_expert_loss(batch[0], batch[1])
        ex_off_losses["expert_loss"].backward()
        bb_off_grad_after = _total_grad_norm(model_off.get_backbone_params())

        # 1. Backbone-only grad (before expert backward) should match exactly
        assert abs(bb_on_grad_before - bb_off_grad_before) / max(bb_on_grad_before, 1e-8) < 1e-5, (
            f"Backbone-only grad differs: KI_ON={bb_on_grad_before:.6f}, "
            f"KI_OFF={bb_off_grad_before:.6f}"
        )

        # 2. KI=ON: after expert backward, backbone grad should be unchanged (~zero flow contribution)
        assert abs(bb_on_grad_after - bb_on_grad_before) / max(bb_on_grad_before, 1e-8) < 1e-5, (
            f"KI=ON: backbone grad changed after expert backward: "
            f"before={bb_on_grad_before:.6f}, after={bb_on_grad_after:.6f}"
        )

        # 3. KI=OFF: after expert backward, backbone grad should increase
        assert bb_off_grad_after > bb_off_grad_before + 1e-3, (
            f"KI=OFF: backbone grad should increase after expert backward: "
            f"before={bb_off_grad_before:.6f}, after={bb_off_grad_after:.6f}"
        )

        # 4. Final total backbone grad: KI=OFF > KI=ON
        assert bb_off_grad_after > bb_on_grad_after + 1e-3, (
            f"Total backbone grad: KI_OFF={bb_off_grad_after:.6f} should be > "
            f"KI_ON={bb_on_grad_after:.6f}"
        )

    def test_expert_update_same_ki_on_off(self):
        """Expert parameter update should be the same for KI ON vs OFF.

        KI only affects flow→backbone gradient flow.  The expert gets its
        gradients from the flow loss directly, which is computed the same
        way regardless of KI mode (the KV detach is invisible to the
        expert's own gradients — it only affects what reaches backbone).
        """
        batch = _make_batch(batch_size=4)

        torch.manual_seed(42)
        model_on = _make_model(knowledge_insulation=True, truncate_expert_kv=True)
        torch.manual_seed(42)
        model_off = _make_model(knowledge_insulation=False, truncate_expert_kv=False)

        # Verify identical init
        sd_on = model_on.state_dict()
        sd_off = model_off.state_dict()
        for key in sd_on:
            assert torch.allclose(sd_on[key], sd_off[key]), f"Init differs at {key}"

        optim_on_bb = torch.optim.AdamW(model_on.get_backbone_params(), lr=1e-3)
        optim_on_ex = torch.optim.AdamW(model_on.get_expert_params(), lr=1e-3)
        optim_off_bb = torch.optim.AdamW(model_off.get_backbone_params(), lr=1e-3)
        optim_off_ex = torch.optim.AdamW(model_off.get_expert_params(), lr=1e-3)

        ex_on_initial = {k: v.clone() for k, v in sd_on.items()
                         if k.startswith("expert.") or k.startswith("action_in_proj.")}
        ex_off_initial = {k: v.clone() for k, v in sd_off.items()
                          if k.startswith("expert.") or k.startswith("action_in_proj.")}

        self._run_step(model_on, optim_on_bb, optim_on_ex, batch, step_idx=0)
        self._run_step(model_off, optim_off_bb, optim_off_ex, batch, step_idx=0)

        def total_delta(model, initial):
            sd = model.state_dict()
            total_sq = 0.0
            for k in initial:
                total_sq += (sd[k] - initial[k]).norm(2).item() ** 2
            return total_sq ** 0.5

        delta_on = total_delta(model_on, ex_on_initial)
        delta_off = total_delta(model_off, ex_off_initial)

        # Expert updates should be essentially identical
        # (Adam state starts same, gradients are same, so updates match)
        assert abs(delta_on - delta_off) / max(delta_on, 1e-8) < 0.01, (
            f"Expert update differs between KI ON ({delta_on:.6f}) and "
            f"OFF ({delta_off:.6f}) — should be same"
        )

    def test_multi_step_stability(self):
        """10+ steps should run without errors or NaN."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)
        batch = _make_batch(batch_size=4)

        for i in range(10):
            metrics = self._run_step(model, optim_bb, optim_ex, batch, step_idx=i)
            assert not math.isnan(metrics["total_loss"]), f"NaN loss at step {i}"
            assert not math.isinf(metrics["total_loss"]), f"Inf loss at step {i}"
            assert metrics["step_time"] > 0


# ===========================================================================
#  Test 3: Checkpoint round-trip
# ===========================================================================


class TestCheckpoint:
    def test_save_and_load_roundtrip(self):
        """Checkpoint save/load should restore all state."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)

        # Do a few steps so optimizers have state
        batch = _make_batch(batch_size=2)
        for _ in range(3):
            optim_bb.zero_grad()
            bb_losses = model.compute_backbone_losses(batch[0], batch[1], batch[2])
            bb_losses["backbone_loss"].backward()
            optim_bb.step()

            optim_ex.zero_grad()
            ex_losses = model.compute_expert_loss(batch[0], batch[1])
            ex_losses["expert_loss"].backward()
            optim_ex.step()

        # Save at step 10
        with tempfile.TemporaryDirectory() as tmpdir:
            from openpi.training.pi05_ki_joint_trainer import save_checkpoint, load_checkpoint

            step = 10
            save_checkpoint(
                model, optim_bb, optim_ex, step, tmpdir,
                metadata={"extra": "info"},
            )

            # Save current state for comparison
            model_state_before = {k: v.clone() for k, v in model.state_dict().items()}
            bb_opt_state_before = optim_bb.state_dict()
            ex_opt_state_before = optim_ex.state_dict()

            # Create a fresh model and optimizers
            torch.manual_seed(99)
            model2 = _make_model(knowledge_insulation=True)
            optim_bb2 = torch.optim.AdamW(model2.get_backbone_params(), lr=1e-3)
            optim_ex2 = torch.optim.AdamW(model2.get_expert_params(), lr=1e-3)

            # Verify they start different
            assert not torch.allclose(
                next(model2.parameters()), next(model.parameters())
            ), "Models should start different"

            # Load checkpoint
            loaded_step = load_checkpoint(
                model2, optim_bb2, optim_ex2, tmpdir,
                device=torch.device("cpu"),
            )
            assert loaded_step == step

            # Verify model params match exactly
            for k, v in model2.state_dict().items():
                assert torch.allclose(v, model_state_before[k]), f"Mismatch in {k}"

            # Verify optimizer state sizes match
            bb_opt_state_after = optim_bb2.state_dict()
            ex_opt_state_after = optim_ex2.state_dict()

            assert len(bb_opt_state_after["state"]) == len(bb_opt_state_before["state"])
            assert len(ex_opt_state_after["state"]) == len(ex_opt_state_before["state"])

            # Check optimizer momentum state (exp_avg) matches for first param
            for key in bb_opt_state_before["state"]:
                if "exp_avg" in bb_opt_state_before["state"][key]:
                    assert torch.allclose(
                        bb_opt_state_after["state"][key]["exp_avg"],
                        bb_opt_state_before["state"][key]["exp_avg"],
                    ), f"Backbone optimizer exp_avg mismatch for param {key}"
                    break

    def test_resume_idempotency(self):
        """Same seed + same data + same step count = same loss after resume.

        We pre-sample all noise and time tensors to guarantee deterministic
        expert loss computation regardless of RNG state transitions across
        save/load boundaries.
        """
        seed = 42
        num_steps = 5
        mid_step = 2  # save after step index 1 (0-based), resume from step 2

        # Pre-generate everything deterministically
        torch.manual_seed(seed + 1000)
        all_batches = [_make_batch(batch_size=2) for _ in range(num_steps)]
        all_noise = [torch.randn(2, 6, 4) for _ in range(num_steps)]  # [B, T, D]
        all_time = [torch.rand(2) * 0.998 + 0.001 for _ in range(num_steps)]

        def run_range(model, optim_bb, optim_ex, start, end):
            """Run steps [start, end) with pre-sampled noise/time."""
            for i in range(start, end):
                batch = all_batches[i]
                noise = all_noise[i]
                time = all_time[i]
                optim_bb.zero_grad()
                bb_losses = model.compute_backbone_losses(batch[0], batch[1], batch[2])
                bb_losses["backbone_loss"].backward()
                optim_bb.step()

                optim_ex.zero_grad()
                ex_losses = model.compute_expert_loss(batch[0], batch[1], noise=noise, time=time)
                ex_losses["expert_loss"].backward()
                optim_ex.step()

        def eval_model(model):
            torch.manual_seed(seed + 9999)
            test_batch = _make_batch(batch_size=4)
            test_noise = torch.randn(4, 6, 4)
            test_time = torch.rand(4) * 0.998 + 0.001
            with torch.no_grad():
                bb_final = model.compute_backbone_losses(test_batch[0], test_batch[1], test_batch[2])
                ex_final = model.compute_expert_loss(test_batch[0], test_batch[1], noise=test_noise, time=test_time)
            return bb_final["backbone_loss"].item() + ex_final["expert_loss"].item()

        # ---- Full run ----
        torch.manual_seed(seed)
        model_full = _make_model(knowledge_insulation=True)
        optim_bb_full = torch.optim.AdamW(model_full.get_backbone_params(), lr=1e-3)
        optim_ex_full = torch.optim.AdamW(model_full.get_expert_params(), lr=1e-3)
        run_range(model_full, optim_bb_full, optim_ex_full, 0, num_steps)
        loss_full = eval_model(model_full)

        # ---- Resumed run (half + save + load + half) ----
        with tempfile.TemporaryDirectory() as tmpdir:
            from openpi.training.pi05_ki_joint_trainer import save_checkpoint, load_checkpoint

            # Train first half
            torch.manual_seed(seed)
            model1 = _make_model(knowledge_insulation=True)
            optim_bb1 = torch.optim.AdamW(model1.get_backbone_params(), lr=1e-3)
            optim_ex1 = torch.optim.AdamW(model1.get_expert_params(), lr=1e-3)
            run_range(model1, optim_bb1, optim_ex1, 0, mid_step)

            # Save at step index = mid_step - 1
            save_step = mid_step - 1
            save_checkpoint(model1, optim_bb1, optim_ex1, save_step, tmpdir)

            # Load into fresh model
            torch.manual_seed(9999)  # different seed to ensure load restores state
            model2 = _make_model(knowledge_insulation=True)
            optim_bb2 = torch.optim.AdamW(model2.get_backbone_params(), lr=1e-3)
            optim_ex2 = torch.optim.AdamW(model2.get_expert_params(), lr=1e-3)
            loaded_step = load_checkpoint(
                model2, optim_bb2, optim_ex2, tmpdir, device=torch.device("cpu")
            )
            assert loaded_step == save_step

            # Run remaining steps
            run_range(model2, optim_bb2, optim_ex2, mid_step, num_steps)
            loss_resumed = eval_model(model2)

        assert abs(loss_full - loss_resumed) < 1e-5, (
            f"Resume idempotency failed: full={loss_full:.6f}, resumed={loss_resumed:.6f}, "
            f"diff={abs(loss_full - loss_resumed):.6f}"
        )

    def test_no_checkpoint_file_error(self):
        """Loading from empty directory should raise FileNotFoundError."""
        from openpi.training.pi05_ki_joint_trainer import load_checkpoint

        model = _make_model()
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_checkpoint(model, optim_bb, optim_ex, tmpdir, device=torch.device("cpu"))

    def test_save_checkpoint_is_main_false(self):
        """Save with is_main=False should not create directory."""
        from openpi.training.pi05_ki_joint_trainer import save_checkpoint

        model = _make_model()
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optim_bb, optim_ex, 0, tmpdir, is_main=False)
            # No step directory should exist
            step_dir = Path(tmpdir) / "0"
            assert not step_dir.exists()

    def test_rng_state_save_restore(self):
        """Saving and restoring RNG state should produce identical next random values."""
        import random
        from openpi.training.pi05_ki_joint_trainer import (
            _get_rng_state_dict, _set_rng_state_dict,
        )

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Capture state after some random operations
        torch.randn(10)
        np.random.randn(10)
        random.random()

        # Save state
        saved_state = _get_rng_state_dict()

        # Get "next" values after saved state
        next_torch_before = torch.randn(5).clone()
        next_numpy_before = np.random.randn(5).copy()
        next_python_before = random.random()

        # Restore and verify identical next values
        _set_rng_state_dict(saved_state)
        next_torch_after = torch.randn(5).clone()
        next_numpy_after = np.random.randn(5).copy()
        next_python_after = random.random()

        assert torch.allclose(next_torch_before, next_torch_after)
        assert np.allclose(next_numpy_before, next_numpy_after)
        assert next_python_before == pytest.approx(next_python_after)

    def test_checkpoint_rng_roundtrip(self):
        """Checkpoint save/load should restore RNG state for deterministic resume."""
        import random
        from openpi.training.pi05_ki_joint_trainer import save_checkpoint, load_checkpoint

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        model = _make_model(knowledge_insulation=True)
        optim_bb = torch.optim.AdamW(model.get_backbone_params(), lr=1e-3)
        optim_ex = torch.optim.AdamW(model.get_expert_params(), lr=1e-3)

        # Do a few random operations to advance RNG
        torch.randn(10)
        np.random.randn(5)
        random.random()

        # Save checkpoint at this RNG state
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optim_bb, optim_ex, 10, tmpdir)

            # Advance RNG after save
            torch_then = torch.randn(3).clone()
            np_then = np.random.randn(3).copy()
            py_then = random.random()

            # Load checkpoint (should restore RNG state)
            torch.manual_seed(99)
            np.random.seed(99)
            random.seed(99)
            load_checkpoint(
                model, optim_bb, optim_ex, tmpdir,
                device=torch.device("cpu"),
                restore_rng_state=True,
            )

            # Next random values should match "then" values
            torch_now = torch.randn(3).clone()
            np_now = np.random.randn(3).copy()
            py_now = random.random()

            assert torch.allclose(torch_then, torch_now), "Torch RNG not restored"
            assert np.allclose(np_then, np_now), "NumPy RNG not restored"
            assert py_then == pytest.approx(py_now), "Python RNG not restored"

    def test_config_validation_ki_mismatch(self):
        """Loading a KI-ON checkpoint into a KI-OFF model should raise ValueError."""
        from openpi.training.pi05_ki_joint_trainer import save_checkpoint, load_checkpoint

        # Save with KI=ON
        model_on = _make_model(knowledge_insulation=True, truncate_expert_kv=True)
        optim_bb_on = torch.optim.AdamW(model_on.get_backbone_params(), lr=1e-3)
        optim_ex_on = torch.optim.AdamW(model_on.get_expert_params(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model_on, optim_bb_on, optim_ex_on, 5, tmpdir)

            # Try to load into KI=OFF model
            model_off = _make_model(knowledge_insulation=False, truncate_expert_kv=False)
            optim_bb_off = torch.optim.AdamW(model_off.get_backbone_params(), lr=1e-3)
            optim_ex_off = torch.optim.AdamW(model_off.get_expert_params(), lr=1e-3)

            with pytest.raises(ValueError, match="mismatch"):
                load_checkpoint(
                    model_off, optim_bb_off, optim_ex_off, tmpdir,
                    device=torch.device("cpu"),
                    validate_config=True,
                )

            # Should work with validate_config=False
            loaded_step = load_checkpoint(
                model_off, optim_bb_off, optim_ex_off, tmpdir,
                device=torch.device("cpu"),
                validate_config=False,
            )
            assert loaded_step == 5

    def test_config_validation_truncate_mismatch(self):
        """truncate_expert_kv mismatch should also be caught."""
        from openpi.training.pi05_ki_joint_trainer import save_checkpoint, load_checkpoint

        # Save with truncate=True, KI=True
        model_save = _make_model(knowledge_insulation=True, truncate_expert_kv=True)
        optim_bb_save = torch.optim.AdamW(model_save.get_backbone_params(), lr=1e-3)
        optim_ex_save = torch.optim.AdamW(model_save.get_expert_params(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model_save, optim_bb_save, optim_ex_save, 3, tmpdir)

            # Load with same KI but different truncate
            model_load = _make_model(knowledge_insulation=True, truncate_expert_kv=False)
            optim_bb_load = torch.optim.AdamW(model_load.get_backbone_params(), lr=1e-3)
            optim_ex_load = torch.optim.AdamW(model_load.get_expert_params(), lr=1e-3)

            with pytest.raises(ValueError, match="mismatch"):
                load_checkpoint(
                    model_load, optim_bb_load, optim_ex_load, tmpdir,
                    device=torch.device("cpu"),
                    validate_config=True,
                )

    def test_resume_idempotency_with_rng(self):
        """Full resume idempotency: RNG state restored → next step produces same loss."""
        import random
        from openpi.training.pi05_ki_joint_trainer import save_checkpoint, load_checkpoint

        seed = 42
        num_steps = 5
        mid_step = 2

        def train_range(model, optim_bb, optim_ex, batches, start, end):
            """Run training steps with pre-sampled data+noise."""
            for i in range(start, end):
                batch, noise, t = batches[i]
                optim_bb.zero_grad()
                bb_losses = model.compute_backbone_losses(batch[0], batch[1], batch[2])
                bb_losses["backbone_loss"].backward()
                optim_bb.step()

                optim_ex.zero_grad()
                ex_losses = model.compute_expert_loss(batch[0], batch[1], noise=noise, time=t)
                ex_losses["expert_loss"].backward()
                optim_ex.step()

        def eval_model(model, test_batch, test_noise, test_time):
            with torch.no_grad():
                bb = model.compute_backbone_losses(test_batch[0], test_batch[1], test_batch[2])
                ex = model.compute_expert_loss(test_batch[0], test_batch[1], noise=test_noise, time=test_time)
            return bb["backbone_loss"].item() + ex["expert_loss"].item()

        # ---- Full run ----
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model_full = _make_model(knowledge_insulation=True)
        optim_bb_full = torch.optim.AdamW(model_full.get_backbone_params(), lr=1e-3)
        optim_ex_full = torch.optim.AdamW(model_full.get_expert_params(), lr=1e-3)

        # Pre-generate all batches deterministically
        torch.manual_seed(seed + 1000)
        batches = []
        for i in range(num_steps):
            batch = _make_batch(batch_size=2)
            noise = torch.randn(2, 6, 4)
            t = torch.rand(2) * 0.998 + 0.001
            batches.append((batch, noise, t))

        # Test batch for evaluation
        torch.manual_seed(seed + 9999)
        test_batch = _make_batch(batch_size=4)
        test_noise = torch.randn(4, 6, 4)
        test_time = torch.rand(4) * 0.998 + 0.001

        train_range(model_full, optim_bb_full, optim_ex_full, batches, 0, num_steps)
        loss_full = eval_model(model_full, test_batch, test_noise, test_time)

        # ---- Resumed run ----
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            model_res = _make_model(knowledge_insulation=True)
            optim_bb_res = torch.optim.AdamW(model_res.get_backbone_params(), lr=1e-3)
            optim_ex_res = torch.optim.AdamW(model_res.get_expert_params(), lr=1e-3)

            # Train first half
            train_range(model_res, optim_bb_res, optim_ex_res, batches, 0, mid_step)

            # Save checkpoint (captures RNG state after mid_step updates)
            save_checkpoint(
                model_res, optim_bb_res, optim_ex_res, mid_step - 1, tmpdir,
                save_rng_state=True,
            )

            # Load into fresh model
            torch.manual_seed(9999)
            np.random.seed(9999)
            random.seed(9999)

            model_load = _make_model(knowledge_insulation=True)
            optim_bb_load = torch.optim.AdamW(model_load.get_backbone_params(), lr=1e-3)
            optim_ex_load = torch.optim.AdamW(model_load.get_expert_params(), lr=1e-3)

            load_checkpoint(
                model_load, optim_bb_load, optim_ex_load, tmpdir,
                device=torch.device("cpu"),
                restore_rng_state=True,
                validate_config=True,
            )

            # Continue training
            train_range(model_load, optim_bb_load, optim_ex_load, batches, mid_step, num_steps)
            loss_resumed = eval_model(model_load, test_batch, test_noise, test_time)

        assert abs(loss_full - loss_resumed) < 1e-5, (
            f"Full-resume mismatch: full={loss_full:.6f}, resumed={loss_resumed:.6f}"
        )


# ===========================================================================
#  Test 4: LR schedule
# ===========================================================================



# ===========================================================================
#  Test 4: Single-optimizer two-param-group (ZeRO / Accelerate compatible)
# ===========================================================================


class TestSingleOptimizer:
    def _run_single_opt_step(self, model, optimizer, batch, step_idx=0,
                             lr_sched_bb=None, lr_sched_ex=None, grad_clip=1.0):
        """Run one step via training_step_single_opt."""
        from openpi.training.pi05_ki_joint_trainer import training_step_single_opt

        prefix_tokens, actions, subtask_targets = batch
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        # Wrap only once
        if not getattr(model, "_trainer_wrapped", False):
            _wrap_model_for_trainer(model)
            model._trainer_wrapped = True

        if lr_sched_bb is None:
            lr_sched_bb = lambda s: 1e-3
        if lr_sched_ex is None:
            lr_sched_ex = lambda s: 1e-3

        return training_step_single_opt(
            model, optimizer, obs, actions,
            step_idx=step_idx,
            lr_schedule_bb=lr_sched_bb,
            lr_schedule_ex=lr_sched_ex,
            grad_clip_norm=grad_clip,
            use_autocast=False,
            autocast_device_type="cpu",
        )

    def test_setup_param_group_optimizer(self):
        """Single optimizer with 2 named param groups."""
        from openpi.training.pi05_ki_joint_trainer import (
            setup_param_group_optimizer,
            get_backbone_param_group, get_expert_param_group,
        )

        model = _make_model(knowledge_insulation=True)
        optim = setup_param_group_optimizer(
            model, lr_backbone=1e-4, lr_expert=5e-4,
        )

        assert isinstance(optim, torch.optim.AdamW)
        assert len(optim.param_groups) == 2

        bb_group = get_backbone_param_group(optim)
        ex_group = get_expert_param_group(optim)
        assert bb_group is not None
        assert ex_group is not None
        assert bb_group["lr"] == 1e-4
        assert ex_group["lr"] == 5e-4
        assert bb_group["name"] == "backbone"
        assert ex_group["name"] == "expert"

    def test_param_groups_no_overlap_full_coverage(self):
        """Two param groups should be disjoint and cover all trainable params."""
        from openpi.training.pi05_ki_joint_trainer import (
            setup_param_group_optimizer,
            get_backbone_param_group, get_expert_param_group,
        )

        model = _make_model(knowledge_insulation=True)
        optim = setup_param_group_optimizer(model)

        bb_params = set(id(p) for p in get_backbone_param_group(optim)["params"])
        ex_params = set(id(p) for p in get_expert_param_group(optim)["params"])
        all_params = set(id(p) for p in model.parameters() if p.requires_grad)

        assert len(bb_params & ex_params) == 0, "Param groups overlap!"
        assert bb_params | ex_params == all_params, "Param groups don't cover all params"

    def test_training_step_single_opt_loss_decreases(self):
        """Single-optimizer training step should decrease loss."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        from openpi.training.pi05_ki_joint_trainer import setup_param_group_optimizer
        optim = setup_param_group_optimizer(model)

        batch = _make_batch(batch_size=4)

        # Save original methods for evaluation
        original_bb = model.compute_backbone_losses
        original_ex = model.compute_expert_loss

        with torch.no_grad():
            bb_0 = original_bb(batch[0], batch[1], batch[2])
            ex_0 = original_ex(batch[0], batch[1])
            total_0 = bb_0["backbone_loss"].item() + ex_0["expert_loss"].item()

        for i in range(5):
            self._run_single_opt_step(model, optim, batch, step_idx=i)

        with torch.no_grad():
            bb_5 = original_bb(batch[0], batch[1], batch[2])
            ex_5 = original_ex(batch[0], batch[1])
            total_5 = bb_5["backbone_loss"].item() + ex_5["expert_loss"].item()

        assert total_5 < total_0, f"Loss did not decrease: {total_0:.4f} -> {total_5:.4f}"

    def test_single_opt_ki_on_flow_to_backbone_zero(self):
        """KI=ON + single opt: flow_to_backbone_grad_norm ~ 0."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True, truncate_expert_kv=True)
        from openpi.training.pi05_ki_joint_trainer import setup_param_group_optimizer
        optim = setup_param_group_optimizer(model)
        batch = _make_batch(batch_size=4)

        metrics = self._run_single_opt_step(model, optim, batch, step_idx=0)

        assert metrics["flow_to_backbone_grad_norm"] < 1e-6
        assert abs(metrics["backbone_grad_norm"] - metrics["backbone_only_grad_norm"]) < 1e-6

    def test_single_opt_ki_off_flow_to_backbone_positive(self):
        """KI=OFF + single opt: flow_to_backbone_grad_norm > 0."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=False, truncate_expert_kv=False)
        from openpi.training.pi05_ki_joint_trainer import setup_param_group_optimizer
        optim = setup_param_group_optimizer(model)
        batch = _make_batch(batch_size=4)

        metrics = self._run_single_opt_step(model, optim, batch, step_idx=0)

        assert metrics["flow_to_backbone_grad_norm"] > 1e-3
        assert metrics["backbone_grad_norm"] > metrics["backbone_only_grad_norm"]

    def test_single_opt_metrics_keys(self):
        """Single-opt metrics should have all the same keys as dual-opt."""
        torch.manual_seed(42)
        model = _make_model(knowledge_insulation=True)
        from openpi.training.pi05_ki_joint_trainer import setup_param_group_optimizer
        optim = setup_param_group_optimizer(model)
        batch = _make_batch(batch_size=2)

        metrics = self._run_single_opt_step(model, optim, batch, step_idx=0)

        expected_keys = [
            "ce_loss", "query_mse_loss", "flow_loss",
            "backbone_loss", "expert_loss", "total_loss",
            "backbone_grad_norm", "expert_grad_norm",
            "backbone_only_grad_norm", "flow_to_backbone_grad_norm",
            "lr_backbone", "lr_expert",
            "step_time", "mem_peak_mb", "mem_alloc_mb",
            "knowledge_insulation", "truncate_expert_kv",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: {key}"

    def test_single_opt_same_loss_decrease(self):
        """Single-opt path should decrease loss similarly to dual-opt path.

        The two paths are NOT numerically identical because dual-opt steps
        the backbone optimizer before the expert backward (which changes
        the weights used in the expert forward), while single-opt steps
        both after both backwards.  But they should both decrease loss
        and produce similar-magnitude updates.
        """
        from openpi.training.pi05_ki_joint_trainer import (
            setup_dual_optimizers, training_step,
            setup_param_group_optimizer, training_step_single_opt,
        )

        lr = 1e-3

        def lr_sched(s):
            return lr

        # Dual-opt run (deterministic init)
        torch.manual_seed(42)
        model_dual = _make_model(knowledge_insulation=True)
        optim_bb, optim_ex = setup_dual_optimizers(
            model_dual, lr_backbone=lr, lr_expert=lr,
        )
        # Save original (unwrapped) methods for evaluation BEFORE wrapping
        original_bb_dual = model_dual.compute_backbone_losses
        original_ex_dual = model_dual.compute_expert_loss
        _wrap_model_for_trainer(model_dual)
        model_dual._trainer_wrapped = True

        # Single-opt run (identical deterministic init)
        torch.manual_seed(42)
        model_single = _make_model(knowledge_insulation=True)
        optim_single = setup_param_group_optimizer(
            model_single, lr_backbone=lr, lr_expert=lr,
        )
        # Save original methods BEFORE wrapping
        original_bb_single = model_single.compute_backbone_losses
        original_ex_single = model_single.compute_expert_loss
        _wrap_model_for_trainer(model_single)
        model_single._trainer_wrapped = True

        # Verify identical init
        sd_dual = model_dual.state_dict()
        sd_single = model_single.state_dict()
        for k in sd_dual:
            assert torch.allclose(sd_dual[k], sd_single[k]), f"Init differs at {k}"

        # Generate batch + noise/time deterministically AFTER both models exist
        torch.manual_seed(123)
        batch = _make_batch(batch_size=4)
        prefix_tokens, actions, subtask_targets = batch
        obs = _ObsWrapper(prefix_tokens, subtask_targets)
        # Pre-generate noise/time for expert loss so evaluations are deterministic
        eval_noise = torch.randn_like(actions)
        eval_time = torch.rand(actions.shape[0]) * 0.998 + 0.001

        # Helper: compute total loss with deterministic expert noise/time
        def eval_total_loss(bb_fn, ex_fn):
            bb = bb_fn(batch[0], batch[1], batch[2])["backbone_loss"]
            ex = ex_fn(batch[0], batch[1], noise=eval_noise, time=eval_time)["expert_loss"]
            return (bb + ex).item()

        # Initial loss (both same since weights identical + same noise/time)
        with torch.no_grad():
            loss_0_dual = eval_total_loss(original_bb_dual, original_ex_dual)
            loss_0_single = eval_total_loss(original_bb_single, original_ex_single)
        # Weights are identical + same noise/time, so losses should be identical
        assert abs(loss_0_dual - loss_0_single) < 1e-5, (
            f"Initial losses differ: {loss_0_dual:.4f} vs {loss_0_single:.4f}"
        )
        loss_0 = loss_0_dual

        # Run 3 steps each — each step samples its own noise/time internally,
        # so the two paths will diverge slightly (different RNG consumption
        # patterns between dual-opt and single-opt implementations).  That's
        # fine — we only check that both decrease and stay similar in magnitude.
        for i in range(3):
            training_step(
                model_dual, optim_bb, optim_ex, obs, actions,
                step_idx=i, lr_schedule_bb=lr_sched, lr_schedule_ex=lr_sched,
                grad_clip_norm=1.0, use_autocast=False, autocast_device_type="cpu",
            )
            training_step_single_opt(
                model_single, optim_single, obs, actions,
                step_idx=i, lr_schedule_bb=lr_sched, lr_schedule_ex=lr_sched,
                grad_clip_norm=1.0, use_autocast=False, autocast_device_type="cpu",
            )

        # Both should decrease loss (eval with same deterministic noise/time)
        with torch.no_grad():
            loss_f_dual = eval_total_loss(original_bb_dual, original_ex_dual)
            loss_f_single = eval_total_loss(original_bb_single, original_ex_single)

        assert loss_f_dual < loss_0_dual, "Dual-opt loss should decrease"
        assert loss_f_single < loss_0_single, "Single-opt loss should decrease"
        # Both should be within reasonable range (same order of magnitude decrease)
        relative_diff = abs(loss_f_dual - loss_f_single) / max(loss_f_dual, loss_f_single)
        assert relative_diff < 0.5, (
            f"Final losses differ by {relative_diff*100:.1f}% — both paths "
            "should produce similar training trajectories"
        )


class TestLRSchedule:
    def test_cosine_schedule_warmup(self):
        """Cosine schedule should increase linearly during warmup."""
        from openpi.training.pi05_ki_joint_trainer import make_cosine_lr_schedule

        peak_lr = 1e-3
        warmup = 100
        decay = 1000
        sched = make_cosine_lr_schedule(peak_lr=peak_lr, warmup_steps=warmup, decay_steps=decay)

        # Step 0
        lr0 = sched(0)
        assert lr0 == pytest.approx(peak_lr / (warmup + 1), rel=1e-6)

        # Step at peak (end of warmup)
        lr_peak = sched(warmup)
        assert lr_peak == pytest.approx(peak_lr, rel=1e-6)

        # Warmup is monotonically increasing
        lrs = [sched(i) for i in range(warmup)]
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR should increase during warmup (step {i})"

    def test_cosine_schedule_decay(self):
        """Cosine schedule should decay after warmup."""
        from openpi.training.pi05_ki_joint_trainer import make_cosine_lr_schedule

        peak_lr = 1e-3
        warmup = 100
        decay = 1000
        sched = make_cosine_lr_schedule(peak_lr=peak_lr, warmup_steps=warmup, decay_steps=decay)

        lr_mid = sched(500)
        lr_end = sched(999)
        lr_final = sched(1000)

        assert lr_mid > lr_end, "LR should decrease during decay"
        assert lr_final == pytest.approx(0.0, abs=1e-6)

    def test_separate_lr_schedules(self):
        """Backbone and expert can have different LR schedules."""
        from openpi.training.pi05_ki_joint_trainer import make_cosine_lr_schedule

        sched_bb = make_cosine_lr_schedule(peak_lr=1e-4, warmup_steps=100, decay_steps=1000)
        sched_ex = make_cosine_lr_schedule(peak_lr=5e-4, warmup_steps=50, decay_steps=800)

        # At step 0, expert LR should be higher
        assert sched_ex(0) > sched_bb(0)
        # At step 500, both should be decaying but expert still higher
        assert sched_ex(500) > sched_bb(500)


# ===========================================================================
#  Test 5: MetricsLogger
# ===========================================================================


class TestMetricsLogger:
    def test_log_step_emits_averages(self):
        """MetricsLogger should emit averages every log_interval steps."""
        from openpi.training.pi05_ki_joint_trainer import MetricsLogger

        logger = MetricsLogger(log_interval=5, wandb_enabled=False)

        for i in range(1, 11):
            metrics = {"loss": float(i), "grad_norm": float(i * 0.1)}
            result = logger.log_step(i, metrics)
            if i % 5 == 0:
                assert result is not None
                assert "loss" in result
                assert result["step"] == i
                # Average of 1-5 = 3.0, 6-10 = 8.0
                if i == 5:
                    assert result["loss"] == pytest.approx(3.0, rel=1e-6)
                elif i == 10:
                    assert result["loss"] == pytest.approx(8.0, rel=1e-6)
            else:
                assert result is None

    def test_history_records_all(self):
        """MetricsLogger history should record every step."""
        from openpi.training.pi05_ki_joint_trainer import MetricsLogger

        logger = MetricsLogger(log_interval=100, wandb_enabled=False)

        for i in range(50):
            logger.log_step(i, {"loss": float(i)})

        assert len(logger.history) == 50
        assert logger.history[0]["loss"] == 0.0
        assert logger.history[49]["loss"] == 49.0


# ===========================================================================
#  Validation / eval metrics tests
# ===========================================================================


class TestComputeEvalMetrics:
    """Tests for compute_eval_metrics (fast-path validation metrics)."""

    def _make_wrapped_model_with_eval(self):
        """Create a mini model with compute_eval_metrics added."""
        model = _make_model()
        model = _wrap_model_for_trainer(model)

        # Add compute_eval_metrics that follows the real model's pattern:
        # one backbone forward producing ce_loss, query_mse, subtask_accuracy, query_l1
        # plus expert forward for flow loss
        def compute_eval_metrics(observation, actions):
            bb_losses = model.compute_backbone_losses(observation, actions)
            ex_losses = model.compute_expert_loss(observation, actions)

            # Fake subtask accuracy (for test: random between 0 and 1)
            # Real model computes argmax vs GT; here we derive from CE loss
            ce_val = bb_losses["ce_loss"].detach()
            # Heuristic: higher CE → lower accuracy (just for shape/type testing)
            subtask_acc = torch.clamp(1.0 - ce_val / 10.0, min=0.0)

            # Fake query L1: sqrt(MSE) as a rough proxy
            query_mse_val = bb_losses["query_mse_loss"].detach()
            query_l1 = torch.sqrt(query_mse_val + 1e-8)

            total_loss = bb_losses["backbone_loss"].detach() + ex_losses["expert_loss"].detach()

            return {
                "total_loss": total_loss,
                "backbone_loss": bb_losses["backbone_loss"].detach(),
                "expert_loss": ex_losses["expert_loss"].detach(),
                "ce_loss": bb_losses["ce_loss"].detach(),
                "query_mse_loss": bb_losses["query_mse_loss"].detach(),
                "flow_loss": ex_losses["flow_loss"].detach(),
                "subtask_accuracy": subtask_acc,
                "query_l1": query_l1,
                "flow_mse": ex_losses["flow_loss"].detach(),
            }

        model.compute_eval_metrics = compute_eval_metrics
        return model

    def test_compute_eval_metrics_returns_expected_keys(self):
        """compute_eval_metrics should return all expected metric keys."""
        model = self._make_wrapped_model_with_eval()
        prefix_tokens, actions, subtask_targets = _make_batch(batch_size=4)
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        model.eval()
        with torch.no_grad():
            metrics = model.compute_eval_metrics(obs, actions)

        expected_keys = {
            "total_loss", "backbone_loss", "expert_loss",
            "ce_loss", "query_mse_loss", "flow_loss",
            "subtask_accuracy", "query_l1", "flow_mse",
        }
        assert set(metrics.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(metrics.keys())}, "
            f"Extra keys: {set(metrics.keys()) - expected_keys}"
        )

    def test_compute_eval_metrics_all_scalar(self):
        """All compute_eval_metrics values should be scalar tensors."""
        model = self._make_wrapped_model_with_eval()
        prefix_tokens, actions, subtask_targets = _make_batch(batch_size=4)
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        model.eval()
        with torch.no_grad():
            metrics = model.compute_eval_metrics(obs, actions)

        for key, val in metrics.items():
            assert isinstance(val, torch.Tensor), f"{key}: not a tensor (type={type(val)})"
            assert val.numel() == 1, f"{key}: not scalar (shape={val.shape})"
            assert val.dtype in (torch.float32, torch.float64), (
                f"{key}: not float dtype (dtype={val.dtype})"
            )

    def test_compute_eval_metrics_all_finite(self):
        """All compute_eval_metrics values should be finite."""
        torch.manual_seed(42)
        model = self._make_wrapped_model_with_eval()
        prefix_tokens, actions, subtask_targets = _make_batch(batch_size=4)
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        model.eval()
        with torch.no_grad():
            metrics = model.compute_eval_metrics(obs, actions)

        for key, val in metrics.items():
            assert torch.isfinite(val).all(), (
                f"{key}: non-finite value: {val.item()}"
            )

    def test_subtask_accuracy_in_01_range(self):
        """subtask_accuracy should be in [0, 1] range."""
        model = self._make_wrapped_model_with_eval()
        prefix_tokens, actions, subtask_targets = _make_batch(batch_size=4)
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        model.eval()
        with torch.no_grad():
            metrics = model.compute_eval_metrics(obs, actions)

        acc = metrics["subtask_accuracy"].item()
        assert 0.0 <= acc <= 1.0, f"subtask_accuracy={acc} out of [0, 1] range"

    def test_compute_eval_metrics_no_grad(self):
        """compute_eval_metrics should not produce gradients (eval mode)."""
        model = self._make_wrapped_model_with_eval()
        prefix_tokens, actions, subtask_targets = _make_batch(batch_size=4)
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        model.train()
        # Even in train mode, values should be detached (no grad on returned dict values)
        metrics = model.compute_eval_metrics(obs, actions)

        for key, val in metrics.items():
            assert not val.requires_grad, f"{key}: has requires_grad=True in eval metrics"

    def test_compute_eval_metrics_train_vs_eval_mode_consistency(self):
        """compute_eval_metrics should produce same values in train and eval mode
        (since it's just forward passes with no dropout/etc. in mini model)."""
        model = self._make_wrapped_model_with_eval()
        prefix_tokens, actions, subtask_targets = _make_batch(batch_size=4, prefix_len=8, action_horizon=6)
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        torch.manual_seed(0)
        model.eval()
        with torch.no_grad():
            metrics_eval = model.compute_eval_metrics(obs, actions)

        torch.manual_seed(0)
        model.train()
        with torch.no_grad():
            metrics_train = model.compute_eval_metrics(obs, actions)

        for key in metrics_eval:
            assert torch.allclose(metrics_eval[key], metrics_train[key], atol=1e-6), (
                f"{key}: train/eval mismatch: eval={metrics_eval[key].item()}, train={metrics_train[key].item()}"
            )

    def test_flow_mse_equals_flow_loss(self):
        """flow_mse should be an alias for flow_loss."""
        model = self._make_wrapped_model_with_eval()
        prefix_tokens, actions, subtask_targets = _make_batch(batch_size=4)
        obs = _ObsWrapper(prefix_tokens, subtask_targets)

        model.eval()
        with torch.no_grad():
            metrics = model.compute_eval_metrics(obs, actions)

        assert torch.allclose(metrics["flow_mse"], metrics["flow_loss"]), (
            f"flow_mse={metrics['flow_mse'].item()} != flow_loss={metrics['flow_loss'].item()}"
        )
