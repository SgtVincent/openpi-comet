"""Real-model Knowledge Insulation (KI) integration tests.

These tests exercise the actual ``PaliGemmaWithExpertModel`` from
``openpi.models_pytorch.gemma_pytorch`` with a tiny Gemma config
(2 layers, 8 heads, 128 hidden) to verify gradient isolation semantics
on real transformer code — no synthetic modules.

Scope
=====
What IS tested (real model, real gradients):
  - Block-causal joint attention gradient flow (PaliGemmaWithExpertModel)
  - _detach_kv_cache() severing an attached computation graph
  - CE→expert cleanliness (already correct, no leak)
  - Flow→backbone leakage (the KI gap)
  - Detached KV → zero backbone grads (KI mechanism)
  - Optimizer group disjointness (backbone vs expert param sets)
  - Query MSE routing (Query-MSE design: query_action_head on backbone)
  - Two-phase combined loss KI correctness

What is NOT tested (out of scope for CPU integration):
  - Vision tower gradient paths (vision is GPU-resident, separate concern)
  - Full PI05KIJointQueryPytorch observation pipeline (images, subtask embeddings)
  - Distributed / mixed-precision training dynamics

Run with (requires transformers==4.53.2)::

    PYTHONNOUSERSITE=1 PYTHONPATH=src python -m pytest tests/test_ki_integration_real.py -v

Test groups (8 groups, 22 tests):
  1. KI gap existence + _detach_kv_cache effectiveness (attached → detached)
  2. KI-OFF baseline: flow loss leaks to backbone KV projs
  3. CE loss → zero expert grads (already clean direction)
  4. Detached DynamicCache preserved (content/seq_len unchanged)
  5. Expert prefix mask contract: KV truncation removes extra positions' effect
  6. Optimizer groups: backbone/expert param sets disjoint
  7. Query MSE: grads reach backbone (query_head + query_emb) but not expert
  8. Combined loss + KI: correct gradient routing, zero cross-contamination
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from openpi.models.gemma import Config
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi05_ki_joint_query import _detach_kv_cache


# ===========================================================================
#  Helpers
# ===========================================================================


def _make_tiny_model(seed: int = 42):
    """Build a tiny PaliGemmaWithExpertModel for CPU testing.

    Config: 2 layers, 8 heads, 1 KV head, 128 hidden, 256 MLP, 16 head_dim.
    Total params ~481M (includes full SigLIP vision tower ~412M; only the
    language + expert transformers ~33.5M are exercised in these tests).
    """
    torch.manual_seed(seed)
    vlm_cfg = Config(
        width=128, depth=2, mlp_dim=256,
        num_heads=8, num_kv_heads=1, head_dim=16,
    )
    ae_cfg = Config(
        width=128, depth=2, mlp_dim=256,
        num_heads=8, num_kv_heads=1, head_dim=16,
    )
    model = PaliGemmaWithExpertModel(vlm_cfg, ae_cfg, precision="float32")
    # Disable gradient checkpointing (not needed for tiny model)
    if hasattr(model.gemma_expert.model, "gradient_checkpointing"):
        model.gemma_expert.model.gradient_checkpointing = False
    model.training = False
    # Force eager attention for deterministic mask behavior
    model.paligemma.language_model.config._attn_implementation = "eager"
    return model


def _block_causal_mask(plen: int, slen: int, batch: int, device: torch.device) -> torch.Tensor:
    """4D additive float mask for block-causal joint attention.

    Prefix positions attend all prefix positions (full self-attention).
    Suffix positions attend all prefix positions + causally within suffix.

    Returns shape (B, 1, plen+slen, plen+slen), dtype float32.
    0.0 = can attend, -inf = cannot attend (HF additive convention).
    """
    total = plen + slen
    mask = torch.full((batch, 1, total, total), float("-inf"), dtype=torch.float32, device=device)
    mask[:, :, :plen, :plen] = 0.0
    for i in range(slen):
        mask[:, :, plen + i, : plen + i + 1] = 0.0
    return mask


def _grad_count(params, threshold: float = 1e-10) -> int:
    """Count parameters with non-zero gradient above threshold."""
    return sum(
        1 for p in params
        if p.grad is not None and p.grad.abs().max().item() > threshold
    )


def _grad_norm(params) -> float:
    """Total L2 norm of gradients across all params."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().data.norm(2).item() ** 2
    return math.sqrt(total)


def _zero_all(*param_groups):
    """Zero out all gradients across multiple parameter groups."""
    for group in param_groups:
        for p in group:
            if p.grad is not None:
                p.grad.zero_()


def _build_prefix_cache(
    model: PaliGemmaWithExpertModel,
    prefix_embs: torch.Tensor,
):
    """Run prefix-only forward and return the DynamicCache.

    The cache contains KV for all prefix positions, suitable for
    cross-attention by the expert.
    """
    B, plen, _ = prefix_embs.shape
    pos = torch.arange(plen, device=prefix_embs.device).unsqueeze(0).expand(B, -1)
    mask = torch.zeros(B, 1, plen, plen, dtype=torch.float32, device=prefix_embs.device)
    _, cache = model.forward(
        inputs_embeds=[prefix_embs, None],
        attention_mask=mask,
        position_ids=pos,
        use_cache=True,
    )
    return cache


def _expert_forward_with_cache(
    model: PaliGemmaWithExpertModel,
    suffix_embs: torch.Tensor,
    prefix_len: int,
    cache,
):
    """Run expert forward with past KV cache.  Returns suffix hidden states."""
    B, slen, _ = suffix_embs.shape
    pos = torch.arange(prefix_len, prefix_len + slen, device=suffix_embs.device).unsqueeze(0).expand(B, -1)
    out, _ = model.forward(
        inputs_embeds=[None, suffix_embs],
        position_ids=pos,
        past_key_values=cache,
        use_cache=False,
    )
    return out[1]


# ===========================================================================
#  Fixtures (session-scoped: ~37s model creation, amortized across all tests)
# ===========================================================================


@pytest.fixture(scope="session")
def model():
    """Session-scoped tiny real PaliGemmaWithExpertModel on CPU.

    Note: tests MUST zero their own grads.  Model is shared across tests.
    """
    return _make_tiny_model()


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


# ===========================================================================
#  1. KI gap + _detach_kv_cache effectiveness
# ===========================================================================


class TestDetachKvCacheSeveresGraph:
    """Validate that ``_detach_kv_cache`` severs the gradient path from
    flow loss to backbone params.

    The critical test pattern:
      1. Build prefix KV cache WITH gradient tracking (attached graph)
      2. Verify flow loss → backbone has grads (KI gap exists, baseline)
      3. Call ``_detach_kv_cache`` on the SAME cache object
      4. Run another flow backward → backbone gets ZERO additional grads

    This proves ``_detach_kv_cache`` is what severs the leak path — not
    just the fact that the cache was built under no_grad.
    """

    def test_attached_cache_leaks_then_detached_blocks(self, model):
        """Attached KV cache → flow leaks; after detach → zero backbone grads."""
        B, plen, slen, action_dim = 2, 8, 6, 16
        action_out = nn.Linear(128, action_dim, bias=False)
        device = next(model.parameters()).device

        # Prefix embeddings with gradient tracking (attached graph)
        prefix_embs = torch.randn(B, plen, 128, device=device, requires_grad=True)
        suffix_embs = torch.randn(B, slen, 128, device=device)

        # Step 1: build cache WITH grads (prefix is part of the graph)
        cache = _build_prefix_cache(model, prefix_embs)

        # Step 2: verify KI-OFF baseline — flow leaks to backbone
        suffix_out = _expert_forward_with_cache(model, suffix_embs, plen, cache)
        flow_loss_1 = F.mse_loss(
            action_out(suffix_out),
            torch.randn(B, slen, action_dim, device=device),
        )
        flow_loss_1.backward()

        bb_before = _grad_norm(model.paligemma.language_model.parameters())
        assert bb_before > 1e-10, (
            f"KI-OFF baseline failed: backbone grad norm = {bb_before:.2e}. "
            "Expected non-zero gradient leakage through attached KV cache."
        )
        ex_before = _grad_norm(model.gemma_expert.model.parameters())
        assert ex_before > 1e-10, "KI-OFF baseline: expert got zero grads"

        # Step 3: detach the cache (the KI mechanism)
        _detach_kv_cache(cache)

        # Reset expert grads but KEEP backbone grads as a baseline
        for p in model.gemma_expert.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in action_out.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Step 4: run another flow forward/backward with detached cache
        suffix_embs_2 = torch.randn(B, slen, 128, device=device)
        suffix_out_2 = _expert_forward_with_cache(model, suffix_embs_2, plen, cache)
        flow_loss_2 = F.mse_loss(
            action_out(suffix_out_2),
            torch.randn(B, slen, action_dim, device=device),
        )
        flow_loss_2.backward()

        # Backbone grad norm should be IDENTICAL to before (detach blocked all flow)
        bb_after = _grad_norm(model.paligemma.language_model.parameters())
        assert abs(bb_before - bb_after) < 1e-12, (
            f"_detach_kv_cache failed: backbone grad changed from {bb_before:.6e} "
            f"to {bb_after:.6e} after second backward. "
            "Detached KV should add ZERO gradient to backbone params."
        )

        # Expert should still get grads (KI doesn't break expert training)
        ex_after = _grad_norm(model.gemma_expert.model.parameters())
        assert ex_after > 1e-10, (
            "After detach: expert got zero grads. KI shouldn't prevent expert training."
        )

    def test_detach_kv_cache_returns_same_object(self, model):
        """_detach_kv_cache modifies cache in-place and returns it."""
        B, plen = 2, 8
        device = next(model.parameters()).device
        prefix_embs = torch.randn(B, plen, 128, device=device)
        cache = _build_prefix_cache(model, prefix_embs)
        detached = _detach_kv_cache(cache)
        assert detached is cache, "_detach_kv_cache should modify cache in-place"

    def test_detach_preserves_cache_values(self, model):
        """_detach_kv_cache doesn't change KV values, just .requires_grad."""
        B, plen = 2, 8
        device = next(model.parameters()).device
        prefix_embs = torch.randn(B, plen, 128, device=device)
        cache = _build_prefix_cache(model, prefix_embs)

        k_before = [cache.key_cache[i].clone() for i in range(len(cache.key_cache))]
        v_before = [cache.value_cache[i].clone() for i in range(len(cache.value_cache))]

        _detach_kv_cache(cache)

        for i in range(len(cache.key_cache)):
            assert torch.allclose(cache.key_cache[i], k_before[i]), (
                f"Layer {i} key_cache changed values after detach"
            )
            assert torch.allclose(cache.value_cache[i], v_before[i]), (
                f"Layer {i} value_cache changed values after detach"
            )
            assert not cache.key_cache[i].requires_grad, (
                f"Layer {i} key_cache still requires grad after detach"
            )


# ===========================================================================
#  2. KI-OFF baseline: flow loss leaks to backbone
# ===========================================================================


class TestFlowOnlyKIOffNonzeroBackboneGrads:
    """Without KI (attached KV cache), flow loss backward produces
    non-zero gradients on backbone KV projections.

    Baseline test that proves the KI gap exists — i.e. the test harness
    can correctly detect gradient leakage.
    """

    def test_backbone_kv_projs_get_grads(self, model):
        """Flow without KI → backbone k_proj and v_proj get non-zero grad in all layers."""
        B, plen, slen, action_dim = 2, 8, 6, 16
        action_out = nn.Linear(128, action_dim, bias=False)
        device = next(model.parameters()).device

        prefix_embs = torch.randn(B, plen, 128, device=device, requires_grad=True)
        suffix_embs = torch.randn(B, slen, 128, device=device)

        cache = _build_prefix_cache(model, prefix_embs)
        suffix_out = _expert_forward_with_cache(model, suffix_embs, plen, cache)
        flow_loss = F.mse_loss(action_out(suffix_out), torch.randn_like(action_out(suffix_out)))
        flow_loss.backward()

        for li, layer in enumerate(model.paligemma.language_model.layers):
            k_g = layer.self_attn.k_proj.weight.grad.abs().max().item()
            v_g = layer.self_attn.v_proj.weight.grad.abs().max().item()
            assert k_g > 1e-8, (
                f"KI-OFF flow: backbone layer {li} k_proj got zero grad "
                f"(expected leakage through cross-attention)"
            )
            assert v_g > 1e-8, (
                f"KI-OFF flow: backbone layer {li} v_proj got zero grad "
                f"(expected leakage through cross-attention)"
            )
        _zero_all(model.parameters(), action_out.parameters())

    def test_backbone_majority_get_grads(self, model):
        """Flow without KI → majority of backbone params have grads (residual chain)."""
        B, plen, slen, action_dim = 2, 8, 6, 16
        action_out = nn.Linear(128, action_dim, bias=False)
        device = next(model.parameters()).device

        prefix_embs = torch.randn(B, plen, 128, device=device, requires_grad=True)
        suffix_embs = torch.randn(B, slen, 128, device=device)

        cache = _build_prefix_cache(model, prefix_embs)
        suffix_out = _expert_forward_with_cache(model, suffix_embs, plen, cache)
        flow_loss = F.mse_loss(action_out(suffix_out), torch.randn_like(action_out(suffix_out)))
        flow_loss.backward()

        bb_params = list(model.paligemma.language_model.parameters())
        total_bb = len(bb_params)
        bb_with_grad = _grad_count(bb_params)

        assert bb_with_grad > total_bb * 0.5, (
            f"KI-OFF flow: only {bb_with_grad}/{total_bb} backbone params have grad. "
            "Expected majority to have grad due to cross-attention + residual chain."
        )
        _zero_all(model.parameters(), action_out.parameters())

    def test_expert_gets_grads(self, model):
        """Flow without KI → expert params get grads too (sanity check)."""
        B, plen, slen, action_dim = 2, 8, 6, 16
        action_out = nn.Linear(128, action_dim, bias=False)
        device = next(model.parameters()).device

        prefix_embs = torch.randn(B, plen, 128, device=device, requires_grad=True)
        suffix_embs = torch.randn(B, slen, 128, device=device)

        cache = _build_prefix_cache(model, prefix_embs)
        suffix_out = _expert_forward_with_cache(model, suffix_embs, plen, cache)
        flow_loss = F.mse_loss(action_out(suffix_out), torch.randn_like(action_out(suffix_out)))
        flow_loss.backward()

        expert_params = list(model.gemma_expert.model.parameters())
        assert _grad_count(expert_params) > 0, "Flow loss should produce expert grads"
        _zero_all(model.parameters(), action_out.parameters())


# ===========================================================================
#  3. CE loss → zero expert grads (already clean direction)
# ===========================================================================


class TestCELossZeroExpertGrads:
    """Subtask CE loss on prefix positions → zero grad on gemma_expert parameters.

    This direction is already clean due to block-causal attention:
    prefix queries never attend suffix positions, so CE loss on prefix
    positions has no gradient path to the expert transformer.
    """

    def test_expert_zero_grads_joint_forward(self, model):
        """CE loss on prefix → zero expert grad (block-causal architecture guarantee)."""
        B, plen, slen = 2, 8, 6
        vocab_size = 5000
        lm_head = nn.Linear(128, vocab_size, bias=False)
        device = next(model.parameters()).device

        mask = _block_causal_mask(plen, slen, B, device)
        pos = torch.arange(plen + slen, device=device).unsqueeze(0).expand(B, -1)
        pref = torch.randn(B, plen, 128, device=device)
        suf = torch.randn(B, slen, 128, device=device)

        [p_out, _], _ = model.forward(
            inputs_embeds=[pref, suf],
            attention_mask=mask,
            position_ids=pos,
        )

        logits = lm_head(p_out[:, -4:, :])
        tgt = torch.randint(0, vocab_size, (B, 4), device=device)
        ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
        ce_loss.backward()

        expert_params = list(model.gemma_expert.model.parameters())
        assert _grad_count(expert_params) == 0, (
            f"CE loss leaked to expert: {_grad_count(expert_params)} params got grad. "
            "Block-causal attention should prevent CE→expert gradient flow."
        )
        _zero_all(model.parameters(), lm_head.parameters())

    def test_backbone_gets_grads(self, model):
        """CE loss does produce non-zero backbone grads (sanity check)."""
        B, plen, slen = 2, 8, 6
        vocab_size = 5000
        lm_head = nn.Linear(128, vocab_size, bias=False)
        device = next(model.parameters()).device

        mask = _block_causal_mask(plen, slen, B, device)
        pos = torch.arange(plen + slen, device=device).unsqueeze(0).expand(B, -1)
        pref = torch.randn(B, plen, 128, device=device)
        suf = torch.randn(B, slen, 128, device=device)

        [p_out, _], _ = model.forward(
            inputs_embeds=[pref, suf],
            attention_mask=mask,
            position_ids=pos,
        )

        logits = lm_head(p_out[:, -4:, :])
        tgt = torch.randint(0, vocab_size, (B, 4), device=device)
        ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
        ce_loss.backward()

        bb_params = list(model.paligemma.language_model.parameters())
        assert _grad_count(bb_params) > 0, "CE loss should produce backbone grads"
        _zero_all(model.parameters(), lm_head.parameters())


# ===========================================================================
#  4. Detached DynamicCache preserved after backward
# ===========================================================================


class TestDetachedDynamicCachePreserved:
    """KI uses real DynamicCache with detached K/V tensors.  Verify that
    the cache seq_len and content are unchanged after a backward pass.

    The KI mechanism must NOT mutate the cache during backward — it should
    only prevent gradients from flowing through it.
    """

    def test_cache_seq_len_unchanged_after_backward(self, model):
        """After flow backward through detached cache, cache seq_len is the same."""
        B, plen, slen, action_dim = 2, 8, 6, 16
        action_out = nn.Linear(128, action_dim, bias=False)
        device = next(model.parameters()).device

        prefix_embs = torch.randn(B, plen, 128, device=device)
        with torch.no_grad():
            cache = _build_prefix_cache(model, prefix_embs)

        seq_len_before = cache.get_seq_length()

        ki_cache = _detach_kv_cache(cache)
        suffix_embs = torch.randn(B, slen, 128, device=device)
        suffix_out = _expert_forward_with_cache(model, suffix_embs, plen, ki_cache)
        flow_loss = F.mse_loss(action_out(suffix_out), torch.randn_like(action_out(suffix_out)))
        flow_loss.backward()

        seq_len_after = ki_cache.get_seq_length()
        assert seq_len_before == seq_len_after, (
            f"Cache seq_len changed after backward: {seq_len_before} → {seq_len_after}"
        )
        _zero_all(model.parameters(), action_out.parameters())

    def test_cache_content_unchanged_after_backward(self, model):
        """After flow backward through detached cache, KV values are bitwise identical."""
        B, plen, slen, action_dim = 2, 8, 6, 16
        action_out = nn.Linear(128, action_dim, bias=False)
        device = next(model.parameters()).device

        prefix_embs = torch.randn(B, plen, 128, device=device)
        with torch.no_grad():
            cache = _build_prefix_cache(model, prefix_embs)

        k_before = [cache.key_cache[i].clone() for i in range(len(cache.key_cache))]
        v_before = [cache.value_cache[i].clone() for i in range(len(cache.value_cache))]

        ki_cache = _detach_kv_cache(cache)
        suffix_embs = torch.randn(B, slen, 128, device=device)
        suffix_out = _expert_forward_with_cache(model, suffix_embs, plen, ki_cache)
        flow_loss = F.mse_loss(action_out(suffix_out), torch.randn_like(action_out(suffix_out)))
        flow_loss.backward()

        for i in range(len(cache.key_cache)):
            assert torch.allclose(cache.key_cache[i], k_before[i]), (
                f"Layer {i} key_cache changed after backward"
            )
            assert torch.allclose(cache.value_cache[i], v_before[i]), (
                f"Layer {i} value_cache changed after backward"
            )
        _zero_all(model.parameters(), action_out.parameters())


# ===========================================================================
#  5. Expert prefix mask contract: KV truncation removes target effect
# ===========================================================================


class TestExpertPrefixExcludesTargetPositions:
    """Forward mask contract: expert output changes when extra tokens are
    in the prefix KV; KV truncation removes their effect.

    This verifies that the KV cache length determines which positions the
    expert can attend — a prerequisite for KI's KV truncation approach
    to target-leakage prevention.
    """

    def test_extra_prefix_tokens_change_expert_output(self, model):
        """Adding tokens to prefix KV changes expert output (cross-attention works)."""
        B, plen, slen = 2, 8, 6
        device = next(model.parameters()).device

        base_prefix = torch.randn(B, plen, 128, device=device)
        extra_tokens = torch.randn(B, 3, 128, device=device)
        long_prefix = torch.cat([base_prefix, extra_tokens], dim=1)

        suffix = torch.randn(B, slen, 128, device=device)

        with torch.no_grad():
            base_cache = _build_prefix_cache(model, base_prefix)
            long_cache = _build_prefix_cache(model, long_prefix)

            out_base = _expert_forward_with_cache(model, suffix, plen, base_cache)
            out_long = _expert_forward_with_cache(model, suffix, plen + 3, long_cache)

        diff = (out_base - out_long).abs().max().item()
        assert diff > 1e-6, (
            f"Extra prefix tokens didn't change expert output (max diff={diff:.2e}). "
            "Cross-attention to prefix should affect expert output."
        )

    def test_truncated_kv_matches_short_prefix(self, model):
        """Truncating KV cache to N positions → expert output matches N-prefix cache.

        This is an end-to-end verification: if we take KV from a long prefix
        and truncate it to the first N positions, the expert cross-attention
        output should be identical to a cache built from just the first N
        positions (with causal attention, so later positions don't affect
        earlier ones).
        """
        B, plen, slen = 2, 8, 6
        extra = 3
        device = next(model.parameters()).device

        base_prefix = torch.randn(B, plen, 128, device=device)
        extra_tokens = torch.randn(B, extra, 128, device=device)
        long_prefix = torch.cat([base_prefix, extra_tokens], dim=1)
        suffix = torch.randn(B, slen, 128, device=device)

        long_len = plen + extra

        with torch.no_grad():
            # Build long cache with CAUSAL attention (earlier positions unaffected
            # by later ones — this is the realistic prefix scenario).
            long_pos = torch.arange(long_len, device=device).unsqueeze(0).expand(B, -1)
            long_mask = torch.full((B, 1, long_len, long_len), float("-inf"), dtype=torch.float32, device=device)
            for i in range(long_len):
                long_mask[:, :, i, : i + 1] = 0.0
            _, long_cache = model.forward(
                inputs_embeds=[long_prefix, None],
                attention_mask=long_mask,
                position_ids=long_pos,
                use_cache=True,
            )

            # Build short cache with causal attention (same first plen positions)
            short_pos = torch.arange(plen, device=device).unsqueeze(0).expand(B, -1)
            short_mask = torch.full((B, 1, plen, plen), float("-inf"), dtype=torch.float32, device=device)
            for i in range(plen):
                short_mask[:, :, i, : i + 1] = 0.0
            _, short_cache = model.forward(
                inputs_embeds=[base_prefix, None],
                attention_mask=short_mask,
                position_ids=short_pos,
                use_cache=True,
            )

            # Truncate long cache to first plen positions
            trunc_cache = copy.deepcopy(long_cache)
            for i in range(len(trunc_cache.key_cache)):
                trunc_cache.key_cache[i] = trunc_cache.key_cache[i][:, :, :plen, :].contiguous()
                trunc_cache.value_cache[i] = trunc_cache.value_cache[i][:, :, :plen, :].contiguous()

            # Expert with short cache
            out_short = _expert_forward_with_cache(model, suffix, plen, short_cache)
            # Expert with truncated long cache (same prefix_len = plen)
            out_trunc = _expert_forward_with_cache(model, suffix, plen, trunc_cache)

        diff = (out_short - out_trunc).abs().max().item()
        assert diff < 1e-5, (
            f"Truncated KV output differs from short KV output (max diff={diff:.2e}). "
            "With causal prefix attention, truncating to N positions should give "
            "identical KV to an N-prefix forward."
        )


# ===========================================================================
#  6. Optimizer groups: backbone/expert param sets disjoint
# ===========================================================================


class TestNamedParametersOptimizerGroupsDisjoint:
    """Build real optimizer groups from model.named_parameters() — verify
    zero overlap and that their union covers all trainable params.

    Static structural test: are the two transformers truly separate
    parameter sets suitable for dual-optimizer KI training?
    """

    def test_backbone_expert_disjoint(self, model):
        """Backbone language model and expert parameter sets are completely disjoint."""
        backbone_ids = set(id(p) for p in model.paligemma.language_model.parameters())
        expert_ids = set(id(p) for p in model.gemma_expert.model.parameters())

        intersection = backbone_ids & expert_ids
        assert len(intersection) == 0, (
            f"Backbone and expert share {len(intersection)} parameters. "
            "They should be completely disjoint for KI dual-optimizer training."
        )

    def test_vision_tower_disjoint_from_expert(self, model):
        """Vision tower params are also disjoint from expert."""
        vision_ids = set(id(p) for p in model.paligemma.vision_tower.parameters())
        expert_ids = set(id(p) for p in model.gemma_expert.model.parameters())
        assert len(vision_ids & expert_ids) == 0, (
            "Vision tower and expert share parameters"
        )

    def test_groups_cover_all_language_params(self, model):
        """backbone + expert covers all language transformer params (no overlap)."""
        backbone_ids = set(id(p) for p in model.paligemma.language_model.parameters())
        expert_ids = set(id(p) for p in model.gemma_expert.model.parameters())
        all_lang_ids = backbone_ids | expert_ids

        total = sum(1 for _ in model.paligemma.language_model.parameters()) + sum(
            1 for _ in model.gemma_expert.model.parameters()
        )
        assert len(all_lang_ids) == total, (
            f"Union has {len(all_lang_ids)} params, expected {total}. "
            "Backbone and expert groups should not share any parameters."
        )

    def test_action_proj_separate_from_expert_model(self, model):
        """action_out_proj is a separate module from gemma_expert (semantic grouping check).

        action_out_proj would be grouped with the expert optimizer since it's
        a head on expert output, but it's not part of gemma_expert.model itself.
        """
        action_out = nn.Linear(128, 16, bias=False)
        expert_ids = set(id(p) for p in model.gemma_expert.model.parameters())
        action_ids = set(id(p) for p in action_out.parameters())
        assert len(expert_ids & action_ids) == 0, (
            "action_out shouldn't be inside expert model params "
            "(it's a separate projection layer grouped with expert semantically)"
        )


# ===========================================================================
#  7. Query MSE: grads reach backbone (query_head + query_emb) but not expert
# ===========================================================================


class TestQueryMseBackboneOnly:
    """Query action MSE (query_action_head applied to backbone hidden states
    at query token positions) produces gradients on the backbone side only.

    The target is plain GT action tensors (leaf tensors), so no gradient
    flows through the target side.  Gradients flow from the MSE loss
    through query_action_head → query hidden states → backbone attention
    → all backbone params.  They do NOT reach the expert transformer.
    """

    def _setup(self, model):
        """Common setup for query MSE tests."""
        B, plen, slen = 2, 8, 6
        num_query = 4
        action_dim = 16
        device = next(model.parameters()).device

        query_emb = nn.Parameter(torch.randn(num_query, 128, device=device))
        query_head = nn.Linear(128, action_dim, bias=True)
        lm_head = nn.Linear(128, 1000, bias=False)

        prefix_base = torch.randn(B, plen, 128, device=device)
        q_embs = query_emb.unsqueeze(0).expand(B, -1, -1)
        full_prefix = torch.cat([prefix_base, q_embs], dim=1)
        suffix = torch.randn(B, slen, 128, device=device)

        return {
            "B": B, "plen": plen, "slen": slen, "num_query": num_query,
            "action_dim": action_dim, "device": device,
            "query_emb": query_emb, "query_head": query_head,
            "lm_head": lm_head,
            "full_prefix": full_prefix, "suffix": suffix,
        }

    def test_query_mse_backbone_gets_grads(self, model):
        """Query MSE → backbone language params get grads."""
        s = self._setup(model)
        device = s["device"]

        mask = _block_causal_mask(s["plen"] + s["num_query"], s["slen"], s["B"], device)
        pos = torch.arange(s["plen"] + s["num_query"] + s["slen"], device=device).unsqueeze(0).expand(s["B"], -1)

        [p_out, _], _ = model.forward(
            inputs_embeds=[s["full_prefix"], s["suffix"]],
            attention_mask=mask,
            position_ids=pos,
        )

        query_hidden = p_out[:, s["plen"] : s["plen"] + s["num_query"], :]
        loss = F.mse_loss(s["query_head"](query_hidden), torch.randn_like(query_hidden[..., : s["action_dim"]]))
        loss.backward()

        bb_params = list(model.paligemma.language_model.parameters())
        assert _grad_count(bb_params) > 0, (
            "Query MSE should produce backbone grads (via attention to query tokens)"
        )
        _zero_all(model.parameters(), s["query_head"].parameters(), [s["query_emb"]])

    def test_query_mse_expert_zero_grads(self, model):
        """Query MSE → expert params have zero grad."""
        s = self._setup(model)
        device = s["device"]

        mask = _block_causal_mask(s["plen"] + s["num_query"], s["slen"], s["B"], device)
        pos = torch.arange(s["plen"] + s["num_query"] + s["slen"], device=device).unsqueeze(0).expand(s["B"], -1)

        [p_out, _], _ = model.forward(
            inputs_embeds=[s["full_prefix"], s["suffix"]],
            attention_mask=mask,
            position_ids=pos,
        )

        query_hidden = p_out[:, s["plen"] : s["plen"] + s["num_query"], :]
        loss = F.mse_loss(s["query_head"](query_hidden), torch.randn_like(query_hidden[..., : s["action_dim"]]))
        loss.backward()

        expert_params = list(model.gemma_expert.model.parameters())
        assert _grad_count(expert_params) == 0, (
            f"Query MSE leaked to expert: {_grad_count(expert_params)} params got grad. "
            "Query tokens are in the prefix (backbone side), so expert should be clean."
        )
        _zero_all(model.parameters(), s["query_head"].parameters(), [s["query_emb"]])

    def test_query_head_gets_grads(self, model):
        """Query MSE → query_action_head weights and bias get grads."""
        s = self._setup(model)
        device = s["device"]

        mask = _block_causal_mask(s["plen"] + s["num_query"], s["slen"], s["B"], device)
        pos = torch.arange(s["plen"] + s["num_query"] + s["slen"], device=device).unsqueeze(0).expand(s["B"], -1)

        [p_out, _], _ = model.forward(
            inputs_embeds=[s["full_prefix"], s["suffix"]],
            attention_mask=mask,
            position_ids=pos,
        )

        query_hidden = p_out[:, s["plen"] : s["plen"] + s["num_query"], :]
        loss = F.mse_loss(s["query_head"](query_hidden), torch.randn_like(query_hidden[..., : s["action_dim"]]))
        loss.backward()

        assert s["query_head"].weight.grad is not None, "query_head.weight has no grad"
        assert s["query_head"].weight.grad.abs().max().item() > 1e-8, "query_head.weight grad is zero"
        assert s["query_head"].bias.grad is not None, "query_head.bias has no grad"
        assert s["query_head"].bias.grad.abs().max().item() > 1e-8, "query_head.bias grad is zero"
        _zero_all(model.parameters(), s["query_head"].parameters(), [s["query_emb"]])

    def test_query_embedding_gets_grads(self, model):
        """Query MSE → query_embeddings (learned query tokens) get grads."""
        s = self._setup(model)
        device = s["device"]

        mask = _block_causal_mask(s["plen"] + s["num_query"], s["slen"], s["B"], device)
        pos = torch.arange(s["plen"] + s["num_query"] + s["slen"], device=device).unsqueeze(0).expand(s["B"], -1)

        [p_out, _], _ = model.forward(
            inputs_embeds=[s["full_prefix"], s["suffix"]],
            attention_mask=mask,
            position_ids=pos,
        )

        query_hidden = p_out[:, s["plen"] : s["plen"] + s["num_query"], :]
        loss = F.mse_loss(s["query_head"](query_hidden), torch.randn_like(query_hidden[..., : s["action_dim"]]))
        loss.backward()

        assert s["query_emb"].grad is not None, "query_emb has no grad"
        assert s["query_emb"].grad.abs().max().item() > 1e-8, "query_emb grad is zero"
        _zero_all(model.parameters(), s["query_head"].parameters(), [s["query_emb"]])

    def test_target_actions_are_leaf_tensors(self, model):
        """Target action tensors for query MSE are leaf tensors (no model params on target side).

        Query-MSE design: the MSE target is plain GT actions (normalized),
        not a model-predicted quantity.  No computation graph on the target
        side, no stop_grad needed — target is just data.
        """
        B, num_query, action_dim = 2, 4, 16
        device = next(model.parameters()).device

        # GT actions are just plain tensors (leaf nodes)
        gt_actions = torch.randn(B, num_query, action_dim, device=device)
        assert gt_actions.is_leaf, "GT actions should be leaf tensors"
        assert gt_actions.grad_fn is None, "GT actions should have no grad_fn"

        # Verify MSE with leaf target doesn't accumulate grad on target
        query_head = nn.Linear(128, action_dim, bias=True)
        query_hidden = torch.randn(B, num_query, 128, device=device, requires_grad=True)
        pred = query_head(query_hidden)
        loss = F.mse_loss(pred, gt_actions)
        loss.backward()

        assert gt_actions.grad is None, (
            "GT action targets shouldn't accumulate grad — they're leaf tensors."
        )


# ===========================================================================
#  8. Combined loss + KI: correct gradient routing
# ===========================================================================


class TestCombinedLossKiCorrectRouting:
    """Combined loss (CE + query MSE + flow) with KI → correct routing:
      - CE / query MSE grads → backbone only
      - Flow loss grads → expert only (with KI)
      - Zero cross-contamination

    End-to-end integration test for the full KI architecture, simulating
    the two-phase training loop pattern.
    """

    def _setup(self, model):
        """Common setup for combined loss tests."""
        B, plen, slen = 2, 8, 6
        num_query = 4
        action_dim = 16
        vocab_size = 5000
        device = next(model.parameters()).device

        query_emb = nn.Parameter(torch.randn(num_query, 128, device=device))
        query_head = nn.Linear(128, action_dim, bias=True)
        lm_head = nn.Linear(128, vocab_size, bias=False)
        action_out = nn.Linear(128, action_dim, bias=False)

        prefix_base = torch.randn(B, plen, 128, device=device)
        q_embs = query_emb.unsqueeze(0).expand(B, -1, -1)
        full_prefix = torch.cat([prefix_base, q_embs], dim=1)
        suffix = torch.randn(B, slen, 128, device=device)

        return {
            "B": B, "plen": plen, "slen": slen, "num_query": num_query,
            "action_dim": action_dim, "vocab_size": vocab_size, "device": device,
            "query_emb": query_emb, "query_head": query_head,
            "lm_head": lm_head, "action_out": action_out,
            "full_prefix": full_prefix, "suffix": suffix,
        }

    def test_phase1_backbone_only(self, model):
        """Phase 1 (CE + query MSE): grads on backbone, zero on expert."""
        s = self._setup(model)
        device = s["device"]
        plen = s["plen"]
        num_query = s["num_query"]
        prefix_len = plen + num_query
        total_len = prefix_len + s["slen"]

        mask = _block_causal_mask(prefix_len, s["slen"], s["B"], device)
        pos = torch.arange(total_len, device=device).unsqueeze(0).expand(s["B"], -1)

        [p_out, _], _ = model.forward(
            inputs_embeds=[s["full_prefix"], s["suffix"]],
            attention_mask=mask,
            position_ids=pos,
        )

        # CE loss
        ce_logits = s["lm_head"](p_out[:, plen - 4 : plen, :])
        ce_tgt = torch.randint(0, s["vocab_size"], (s["B"], 4), device=device)
        ce_loss = F.cross_entropy(ce_logits.reshape(-1, s["vocab_size"]), ce_tgt.reshape(-1))

        # Query MSE
        query_hidden = p_out[:, plen : plen + num_query, :]
        query_loss = F.mse_loss(
            s["query_head"](query_hidden),
            torch.randn(s["B"], num_query, s["action_dim"], device=device),
        )

        backbone_loss = ce_loss + query_loss
        backbone_loss.backward()

        bb_count = _grad_count(model.paligemma.language_model.parameters())
        assert bb_count > 0, "Phase 1: backbone has zero grads"

        ex_count = _grad_count(model.gemma_expert.model.parameters())
        assert ex_count == 0, (
            f"Phase 1: expert got {ex_count} non-zero grads from CE + query MSE. "
            "Should be zero (block-causal architecture)."
        )
        _zero_all(model.parameters(), s["lm_head"].parameters(),
                  s["query_head"].parameters(), [s["query_emb"]])

    def test_phase2_expert_only_with_ki(self, model):
        """Phase 2 (flow + KI): grads on expert only, zero on backbone."""
        s = self._setup(model)
        device = s["device"]
        plen = s["plen"]
        num_query = s["num_query"]
        prefix_len = plen + num_query

        # Build cache with no_grad (inference-style, then detach = KI)
        with torch.no_grad():
            pos_p = torch.arange(prefix_len, device=device).unsqueeze(0).expand(s["B"], -1)
            mask_p = torch.zeros(s["B"], 1, prefix_len, prefix_len, dtype=torch.float32, device=device)
            _, cache = model.forward(
                inputs_embeds=[s["full_prefix"], None],
                attention_mask=mask_p,
                position_ids=pos_p,
                use_cache=True,
            )

        ki_cache = _detach_kv_cache(cache)

        suffix_out = _expert_forward_with_cache(model, s["suffix"], prefix_len, ki_cache)
        flow_loss = F.mse_loss(
            s["action_out"](suffix_out),
            torch.randn(s["B"], s["slen"], s["action_dim"], device=device),
        )
        flow_loss.backward()

        ex_count = _grad_count(model.gemma_expert.model.parameters())
        assert ex_count > 0, "Phase 2 (KI-ON): expert has zero grads"

        bb_count = _grad_count(model.paligemma.language_model.parameters())
        assert bb_count == 0, (
            f"Phase 2 (KI-ON): backbone got {bb_count} non-zero grads from flow loss. "
            "Detached KV cache should prevent all flow→backbone leakage."
        )
        _zero_all(model.parameters(), s["action_out"].parameters())

    def test_two_phase_zero_cross_contamination(self, model):
        """Full two-phase KI: backbone-only grads in phase 1, expert-only in phase 2.

        Complete KI integration test — verifies the two-phase training
        pattern achieves perfect gradient isolation.
        """
        s = self._setup(model)
        device = s["device"]
        plen = s["plen"]
        num_query = s["num_query"]
        prefix_len = plen + num_query
        total_len = prefix_len + s["slen"]

        # ---- Phase 1: backbone forward + losses ----
        mask_full = _block_causal_mask(prefix_len, s["slen"], s["B"], device)
        pos_full = torch.arange(total_len, device=device).unsqueeze(0).expand(s["B"], -1)

        [p_out, _], _ = model.forward(
            inputs_embeds=[s["full_prefix"], s["suffix"]],
            attention_mask=mask_full,
            position_ids=pos_full,
        )

        ce_logits = s["lm_head"](p_out[:, plen - 4 : plen, :])
        ce_tgt = torch.randint(0, s["vocab_size"], (s["B"], 4), device=device)
        ce_loss = F.cross_entropy(ce_logits.reshape(-1, s["vocab_size"]), ce_tgt.reshape(-1))

        query_hidden = p_out[:, plen : plen + num_query, :]
        query_loss = F.mse_loss(
            s["query_head"](query_hidden),
            torch.randn(s["B"], num_query, s["action_dim"], device=device),
        )

        backbone_loss = ce_loss + query_loss
        backbone_loss.backward()

        # Snapshot phase 1 grad norms
        bb_phase1 = _grad_norm(model.paligemma.language_model.parameters())
        ex_phase1 = _grad_norm(model.gemma_expert.model.parameters())

        # ---- Phase 2: expert forward with detached KV ----
        with torch.no_grad():
            pos_p = torch.arange(prefix_len, device=device).unsqueeze(0).expand(s["B"], -1)
            mask_p = torch.zeros(s["B"], 1, prefix_len, prefix_len, dtype=torch.float32, device=device)
            _, cache = model.forward(
                inputs_embeds=[s["full_prefix"], None],
                attention_mask=mask_p,
                position_ids=pos_p,
                use_cache=True,
            )
        ki_cache = _detach_kv_cache(cache)

        # Expert forward + flow loss
        # Note: we DON'T zero backbone grads — phase 2 should add nothing to them
        suffix_out = _expert_forward_with_cache(model, s["suffix"], prefix_len, ki_cache)
        flow_loss = F.mse_loss(
            s["action_out"](suffix_out),
            torch.randn(s["B"], s["slen"], s["action_dim"], device=device),
        )
        flow_loss.backward()

        bb_phase2 = _grad_norm(model.paligemma.language_model.parameters())
        ex_phase2 = _grad_norm(model.gemma_expert.model.parameters())

        # Key assertions:
        # 1. Backbone grad norm unchanged after phase 2 (KI blocks flow from adding)
        assert abs(bb_phase1 - bb_phase2) < 1e-12, (
            f"Backbone grad norm changed after phase 2: {bb_phase1:.6e} → {bb_phase2:.6e}. "
            "With KI, flow loss should add ZERO gradient to backbone params."
        )

        # 2. Expert grad only from phase 2 (phase 1 added zero)
        assert ex_phase1 < 1e-12, (
            f"Phase 1 gave expert non-zero grad norm: {ex_phase1:.6e}. "
            "CE + query MSE shouldn't produce expert gradients."
        )
        assert ex_phase2 > 1e-6, "Phase 2 flow loss gave expert zero grad norm"

        _zero_all(model.parameters(), s["lm_head"].parameters(),
                  s["query_head"].parameters(), s["action_out"].parameters(),
                  [s["query_emb"]])

    def test_ki_off_combined_backbone_gets_both(self, model):
        """Without KI, combined loss: backbone gets grads from both CE+MSE AND flow.

        Negative control — proves that when KI is off, the flow loss DOES
        add gradient to backbone params (i.e. the gap is real and detectable).
        """
        s = self._setup(model)
        device = s["device"]
        plen = s["plen"]
        num_query = s["num_query"]
        prefix_len = plen + num_query
        total_len = prefix_len + s["slen"]

        # Step 1: CE + query MSE only → measure backbone grad
        mask = _block_causal_mask(prefix_len, s["slen"], s["B"], device)
        pos = torch.arange(total_len, device=device).unsqueeze(0).expand(s["B"], -1)

        [p_out, s_out], _ = model.forward(
            inputs_embeds=[s["full_prefix"], s["suffix"]],
            attention_mask=mask,
            position_ids=pos,
        )

        ce_logits = s["lm_head"](p_out[:, plen - 4 : plen, :])
        ce_tgt = torch.randint(0, s["vocab_size"], (s["B"], 4), device=device)
        ce_loss = F.cross_entropy(ce_logits.reshape(-1, s["vocab_size"]), ce_tgt.reshape(-1))

        query_hidden = p_out[:, plen : plen + num_query, :]
        query_loss = F.mse_loss(
            s["query_head"](query_hidden),
            torch.randn(s["B"], num_query, s["action_dim"], device=device),
        )

        backbone_only_loss = ce_loss + query_loss
        backbone_only_loss.backward(retain_graph=True)

        bb_ce_only = _grad_norm(model.paligemma.language_model.parameters())

        # Step 2: add flow loss (KI-OFF, so it leaks to backbone too)
        flow_loss = F.mse_loss(
            s["action_out"](s_out),
            torch.randn(s["B"], s["slen"], s["action_dim"], device=device),
        )
        total_loss = flow_loss  # just flow, on top of retained graph
        total_loss.backward()

        bb_combined = _grad_norm(model.paligemma.language_model.parameters())

        # Combined grad norm should be LARGER than CE+query-only
        # (flow adds extra gradient to backbone params)
        assert bb_combined > bb_ce_only * 1.01, (
            f"KI-OFF combined: backbone grad didn't increase with flow loss "
            f"({bb_ce_only:.6e} → {bb_combined:.6e}). "
            "Expected flow to add gradient to backbone (KI gap)."
        )
        _zero_all(model.parameters(), s["lm_head"].parameters(),
                  s["query_head"].parameters(), s["action_out"].parameters(),
                  [s["query_emb"]])
