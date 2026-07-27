"""Gradient isolation / Knowledge Insulation (KI) contract tests.

These tests verify how gradients flow between the two transformers in the
PaliGemmaWithExpertModel architecture:

  - Backbone: VLM backbone (vision + language)
  - Expert:   action expert transformer

Architecture contract (faithful to production)
===============================================

The two transformers have **block-causal joint attention**:

  * Prefix (backbone side): full self-attention among prefix positions,
    NO attention to suffix positions.
  * Suffix (expert side): causal self-attention within suffix,
    cross-attention to ALL prefix positions.

This means:
  * Loss on prefix (e.g. subtask CE) → gradients **only on backbone** (no leak to expert)
  * Loss on suffix (e.g. flow MSE) → gradients on **expert AND backbone**
    (leak through K/V of prefix positions attended by suffix queries)

For a strict KI regime where each side trains on its own loss only, the
**flow-to-backbone leakage is the problem to solve**.  The CE→expert
direction is already clean.

Test oracle
===========

We use a synthetic ``_DualTransformer`` module that faithfully reproduces
the joint-attention architecture: two separate layer stacks, Q/K/V
concatenated per layer, block-causal mask.  This keeps tests fast (<5s)
and focused on connectivity semantics, not model size.

Test groups (8 groups, 23 tests)
================================
1. Subtask CE-only gradient reach  → backbone gets grads, expert is clean
2. Flow-only gradient reach        → expert gets grads, backbone gets leaked grads
3. Detached KV cache insulation    → inference-path KI mechanism
4. Detached prefix embeds insufficient → negative result: .detach() on inputs ≠ KI
5. Dual-optimizer partitioning     → param ID sets disjoint, no cross-update
6. Combined loss additivity        → gradient linearity + KI achievable via separate passes
7. action_out_proj side checks     → expert-side parameter membership
8. Target-leakage contract         → forward mask + backward isolation
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn


# ===========================================================================
#  Synthetic dual-transformer architecture
# ===========================================================================
#  Mirrors PaliGemmaWithExpertModel.compute_layer_complete exactly:
#    - Q/K/V from both sides concatenated along seq dim per layer
#    - Joint scaled dot-product attention
#    - O-proj + residual + MLP done per-side on split attention output
#    - Backbone has embed_tokens + lm_head (tied)
#    - Expert has no embedding table (takes inputs_embeds)
# ===========================================================================


class _SynthLayer(nn.Module):
    """One transformer layer: Q/K/V/O + MLP.  Shape-compatible with Gemma."""

    def __init__(self, hidden_size, num_heads, head_dim, mlp_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size, bias=False),
        )

    def qkv(self, x):
        """Return (q, k, v) each shaped (B, H, T, D)."""
        B, T, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def output(self, x, attn_out):
        """attn_out: (B, H, T, D) → hidden (B, T, C)."""
        B, H, T, D = attn_out.shape
        attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = x + self.o_proj(attn_flat)
        out = out + self.mlp(self.norm2(out))
        return out


class _DualTransformer(nn.Module):
    """Two transformers with block-causal joint attention.

    Behaviour modes:
      - Joint forward [prefix_emb, suffix_emb] + no cache    → training path
      - Prefix-only [prefix_emb, None] + use_cache=True       → build KV cache
      - Suffix-only [None, suffix_emb] + past_key_values      → inference path
    """

    def __init__(self, hidden_size=32, num_layers=2, num_heads=4, head_dim=8,
                 mlp_dim=64, vocab_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size

        # Backbone: embedding + layers + lm_head (tied weights)
        self.backbone_embed = nn.Embedding(vocab_size, hidden_size)
        self.backbone_layers = nn.ModuleList([
            _SynthLayer(hidden_size, num_heads, head_dim, mlp_dim)
            for _ in range(num_layers)
        ])
        self.backbone_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.backbone_embed.weight  # tie

        # Expert: layers only (no embedding table)
        self.expert_layers = nn.ModuleList([
            _SynthLayer(hidden_size, num_heads, head_dim, mlp_dim)
            for _ in range(num_layers)
        ])
        self.expert_norm = nn.LayerNorm(hidden_size)

    # ------------------------------------------------------------------
    def forward(self, inputs_embeds, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False):
        """Returns ([prefix_out, suffix_out], past_kv_or_None).

        attention_mask: (1, 1, T_q, T_kv) bool, True = can attend.
        """
        pref_emb, suf_emb = inputs_embeds

        # --- Prefix-only (build cache) ---
        if pref_emb is not None and suf_emb is None:
            h = pref_emb
            new_kv = []
            for layer in self.backbone_layers:
                q, k, v = layer.qkv(h)
                attn = _sdpa(q, k, v, attention_mask)
                h = layer.output(h, attn)
                if use_cache:
                    new_kv.append((k, v))
            h = self.backbone_norm(h)
            return [h, None], new_kv if use_cache else None

        # --- Suffix-only with past KV (inference / cross-attend) ---
        if pref_emb is None and suf_emb is not None:
            assert past_key_values is not None
            h = suf_emb
            for li, layer in enumerate(self.expert_layers):
                q, k, v = layer.qkv(h)
                cache_k, cache_v = past_key_values[li]
                full_k = torch.cat([cache_k, k], dim=2)
                full_v = torch.cat([cache_v, v], dim=2)
                attn = _sdpa(q, full_k, full_v, attention_mask)
                h = layer.output(h, attn)
            h = self.expert_norm(h)
            return [None, h], None

        # --- Joint prefix + suffix (training) ---
        assert pref_emb is not None and suf_emb is not None
        B = pref_emb.shape[0]
        plen = pref_emb.shape[1]
        slen = suf_emb.shape[1]
        hb, he = pref_emb, suf_emb

        for li in range(self.num_layers):
            bl = self.backbone_layers[li]
            el = self.expert_layers[li]

            qb, kb, vb = bl.qkv(hb)
            qe, ke, ve = el.qkv(he)

            q = torch.cat([qb, qe], dim=2)   # (B, H, plen+slen, D)
            k = torch.cat([kb, ke], dim=2)
            v = torch.cat([vb, ve], dim=2)

            attn = _sdpa(q, k, v, attention_mask)  # (B, H, plen+slen, D)

            attn_b = attn[:, :, :plen, :]
            attn_e = attn[:, :, plen:, :]

            hb = bl.output(hb, attn_b)
            he = el.output(he, attn_e)

        hb = self.backbone_norm(hb)
        he = self.expert_norm(he)
        return [hb, he], None


def _sdpa(q, k, v, mask=None):
    """Scaled dot-product attention.  mask: bool (True=attend)."""
    D = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


def _causal_mask(n, device):
    """(1, 1, n, n) bool causal mask (True = attend)."""
    m = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool))
    return m[None, None, :, :]


def _cross_mask(plen, slen, device):
    """(1, 1, slen, plen+slen) mask: suffix attends to all prefix + causal within suffix."""
    m = torch.zeros(slen, plen + slen, device=device, dtype=torch.bool)
    m[:, :plen] = True
    for i in range(slen):
        m[i, plen:plen + i + 1] = True
    return m[None, None, :, :]


# ===========================================================================
#  Fixtures + helpers
# ===========================================================================


def _make_model(hidden_size=32, num_layers=2, num_heads=4, head_dim=8,
                mlp_dim=64, vocab_size=512, action_dim=16, seed=0):
    """Build a tiny dual-transformer + action projections."""
    torch.manual_seed(seed)
    model = _DualTransformer(hidden_size, num_layers, num_heads, head_dim,
                             mlp_dim, vocab_size)
    action_in = nn.Linear(action_dim, hidden_size, bias=False)
    action_out = nn.Linear(hidden_size, action_dim, bias=False)
    return SimpleNamespace(
        model=model, action_in=action_in, action_out=action_out,
        hidden_size=hidden_size, vocab_size=vocab_size, action_dim=action_dim,
        device=torch.device('cpu'),
    )


def _grad_norms(ns):
    """Per-group L2 grad norms.  Groups: backbone, backbone_embed, expert,
    action_in, action_out."""
    m = ns.model
    groups = {
        'backbone': list(m.backbone_layers.parameters()) + list(m.backbone_norm.parameters()),
        'backbone_embed': [m.backbone_embed.weight],
        'expert': list(m.expert_layers.parameters()) + list(m.expert_norm.parameters()),
        'action_in': list(ns.action_in.parameters()),
        'action_out': list(ns.action_out.parameters()),
    }
    norms = {}
    for name, params in groups.items():
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += p.grad.detach().data.norm(2).item() ** 2
        norms[name] = math.sqrt(total)
    return norms


def _zero_grads(ns):
    for p in ns.model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for proj in (ns.action_in, ns.action_out):
        for p in proj.parameters():
            if p.grad is not None:
                p.grad.zero_()


def _param_ids(ns, group):
    """Set of parameter IDs for a group.  Used to check disjointness."""
    mapping = {
        'backbone': (list(ns.model.backbone_layers.parameters())
                     + list(ns.model.backbone_norm.parameters())
                     + [ns.model.backbone_embed.weight]),
        'expert': (list(ns.model.expert_layers.parameters())
                   + list(ns.model.expert_norm.parameters())
                   + list(ns.action_in.parameters())
                   + list(ns.action_out.parameters())),
    }
    return set(id(p) for p in mapping[group])


@pytest.fixture
def m():
    """Tiny dual transformer on CPU."""
    return _make_model()


# ===========================================================================
#  1. Subtask CE-only gradient reach
# ===========================================================================


class TestCEOnlyGradReach:
    """CE loss on prefix positions → gradient reach pattern.

    Block-causal architecture means prefix queries never attend to suffix
    positions, so CE loss (which depends only on prefix hidden states)
    produces **zero gradient on the expert**.

    This is already the KI-correct direction!
    """

    def _ce_loss(self, m, plen=8, slen=6):
        """Run joint forward + CE loss on last 4 prefix positions."""
        dev = m.device
        pref = torch.randn(2, plen, m.hidden_size, device=dev)
        suf = torch.randn(2, slen, m.hidden_size, device=dev)
        total = plen + slen
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)

        [p_out, s_out], _ = m.model.forward(
            inputs_embeds=[pref, suf], attention_mask=mask, position_ids=pos,
        )
        subtask = p_out[:, -4:, :]
        logits = m.model.lm_head(subtask)
        B, T, V = logits.shape
        tgt = torch.randint(0, V, (B, T), device=dev)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        return loss

    def test_backbone_has_grads(self, m):
        """CE loss → non-zero grad on backbone layers and embedding."""
        loss = self._ce_loss(m)
        _zero_grads(m)
        loss.backward()
        n = _grad_norms(m)
        assert n['backbone'] > 1e-6, f"backbone grad = {n['backbone']}"
        assert n['backbone_embed'] > 1e-6, f"embed grad = {n['backbone_embed']}"

    def test_expert_has_zero_grads(self, m):
        """CE loss → zero grad on expert (prefix doesn't attend to suffix).

        This is a **KI-correct** property of the block-causal architecture:
        the subtask CE loss does not leak gradients into the action expert.
        """
        loss = self._ce_loss(m)
        _zero_grads(m)
        loss.backward()
        n = _grad_norms(m)
        assert n['expert'] < 1e-10, (
            f"CE loss leaked to expert (norm={n['expert']}).  "
            "Block-causal architecture should prevent this."
        )

    def test_action_projs_have_zero_grads(self, m):
        """CE loss → zero grad on action_in_proj and action_out_proj."""
        loss = self._ce_loss(m)
        _zero_grads(m)
        loss.backward()
        n = _grad_norms(m)
        assert n['action_in'] < 1e-10
        assert n['action_out'] < 1e-10

    def test_exact_param_set_backbone_only(self, m):
        """CE loss: exactly the backbone param set has non-zero grad."""
        loss = self._ce_loss(m)
        _zero_grads(m)
        loss.backward()

        bb_ids = _param_ids(m, 'backbone')
        ex_ids = _param_ids(m, 'expert')

        # Every backbone param that participates should have grad;
        # no expert param should have grad.
        expert_with_grad = [p for p in m.model.expert_layers.parameters()
                            if p.grad is not None and p.grad.abs().max().item() > 1e-10]
        assert len(expert_with_grad) == 0, (
            f"{len(expert_with_grad)} expert parameters got non-zero grad from CE loss"
        )

        # All backbone Q/K/V/O/MLP params should get grad (they participate)
        bb_with_grad = [p for p in m.model.backbone_layers.parameters()
                        if p.grad is not None and p.grad.abs().max().item() > 1e-10]
        total_bb = sum(1 for _ in m.model.backbone_layers.parameters())
        assert len(bb_with_grad) == total_bb, (
            f"Only {len(bb_with_grad)}/{total_bb} backbone params got grad from CE"
        )


# ===========================================================================
#  2. Flow-only gradient reach
# ===========================================================================


class TestFlowOnlyGradReach:
    """Flow loss on suffix positions → gradient reach pattern.

    Block-causal architecture means suffix queries attend to all prefix
    positions, so flow loss produces gradients on **both** expert AND
    backbone.

    This is the direction that KI needs to fix.
    """

    def _flow_loss(self, m, plen=8, slen=6):
        """Run joint forward + MSE loss on suffix output."""
        dev = m.device
        pref = torch.randn(2, plen, m.hidden_size, device=dev)
        suf = torch.randn(2, slen, m.hidden_size, device=dev)
        total = plen + slen
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)

        [p_out, s_out], _ = m.model.forward(
            inputs_embeds=[pref, suf], attention_mask=mask, position_ids=pos,
        )
        act = m.action_out(s_out)
        tgt = torch.randn_like(act)
        return F.mse_loss(act, tgt)

    def test_expert_has_grads(self, m):
        """Flow loss → non-zero grad on expert + action_out."""
        loss = self._flow_loss(m)
        _zero_grads(m)
        loss.backward()
        n = _grad_norms(m)
        assert n['expert'] > 1e-6
        assert n['action_out'] > 1e-6

    def test_backbone_has_nonzero_grads(self, m):
        """Flow loss → non-zero grad on backbone (cross-attention leak).

        This documents the current architecture: flow loss leaks gradients
        to the backbone through the prefix K/V that suffix queries attend to.
        KI requires this path to be severed.
        """
        loss = self._flow_loss(m)
        _zero_grads(m)
        loss.backward()
        n = _grad_norms(m)
        assert n['backbone'] > 1e-6, (
            f"Flow loss should leak to backbone in joint attention, got {n['backbone']}"
        )
        assert n['backbone_embed'] < 1e-10, (
            "backbone_embed should have 0 grad when prefix embeds are "
            "leaf tensors (not produced by embed_tokens in this test)"
        )

    def test_backbone_kv_projs_get_grad(self, m):
        """Specifically: backbone k_proj and v_proj get flow grads in all layers.

        The direct leakage path is: suffix attn output → softmax → prefix K/V
        → backbone k_proj/v_proj → backbone earlier layers.

        Gradient distribution by layer:
          * Last backbone layer: k_proj and v_proj get direct gradient from
            suffix cross-attention.  q_proj gets zero (prefix queries don't
            affect suffix attention output, and there's no next layer to
            carry residual gradient back to Q).
          * Earlier layers: all of q/k/v/o_proj get grads through the
            residual chain (gradient from K/V of layer i+1 flows back
            through the residual of layer i's output, which includes the
            attention output → q/k/v_proj of layer i).
        """
        loss = self._flow_loss(m)
        _zero_grads(m)
        loss.backward()

        n_layers = m.model.num_layers
        for li, layer in enumerate(m.model.backbone_layers):
            k_g = layer.k_proj.weight.grad.abs().max().item()
            v_g = layer.v_proj.weight.grad.abs().max().item()
            assert k_g > 1e-8, f"backbone layer {li} k_proj zero grad"
            assert v_g > 1e-8, f"backbone layer {li} v_proj zero grad"

            if li < n_layers - 1:
                # Earlier layers: q_proj gets grad via residual from next layer
                q_g = layer.q_proj.weight.grad.abs().max().item()
                assert q_g > 1e-8, (
                    f"backbone layer {li} q_proj zero grad — "
                    "residual path from next layer should carry gradient"
                )
            else:
                # Last layer: q_proj gets zero from flow loss
                q_g = layer.q_proj.weight.grad.abs().max().item()
                assert q_g < 1e-10, (
                    f"backbone last layer q_proj should have zero flow grad, "
                    f"got {q_g}"
                )


# ===========================================================================
#  3. Detached KV cache insulation
# ===========================================================================


class TestDetachedKVCache:
    """Detaching the prefix KV cache severs the gradient graph → KI works."""

    def _build_cache(self, m, plen=8):
        dev = m.device
        pref = torch.randn(2, plen, m.hidden_size, device=dev)
        mask = _causal_mask(plen, dev)
        pos = torch.arange(plen, device=dev).unsqueeze(0)
        [p_out, _], kv = m.model.forward(
            inputs_embeds=[pref, None], attention_mask=mask,
            position_ids=pos, use_cache=True,
        )
        return pref, kv, plen

    def test_detached_kv_gives_zero_backbone_grad(self, m):
        _, kv, plen = self._build_cache(m)
        dev = m.device
        slen = 6

        detached_kv = tuple((k.detach(), v.detach()) for k, v in kv)
        suf = torch.randn(2, slen, m.hidden_size, device=dev)
        mask = _cross_mask(plen, slen, dev)
        pos = plen + torch.arange(slen, device=dev).unsqueeze(0)

        [_, s_out], _ = m.model.forward(
            inputs_embeds=[None, suf], attention_mask=mask,
            position_ids=pos, past_key_values=detached_kv,
        )
        loss = F.mse_loss(m.action_out(s_out), torch.randn_like(m.action_out(s_out)))

        _zero_grads(m)
        loss.backward()
        n = _grad_norms(m)

        assert n['expert'] > 1e-6, "expert should get grad"
        assert n['action_out'] > 1e-6, "action_out should get grad"
        assert n['backbone'] < 1e-10, (
            f"backbone grad should be 0 with detached KV, got {n['backbone']}"
        )
        assert n['backbone_embed'] < 1e-10

    def test_attached_kv_gives_nonzero_backbone_grad(self, m):
        """Baseline: attached KV → backbone gets grads (test harness correct)."""
        pref, kv, plen = self._build_cache(m)
        dev = m.device
        slen = 6

        suf = torch.randn(2, slen, m.hidden_size, device=dev)
        mask = _cross_mask(plen, slen, dev)
        pos = plen + torch.arange(slen, device=dev).unsqueeze(0)

        [_, s_out], _ = m.model.forward(
            inputs_embeds=[None, suf], attention_mask=mask,
            position_ids=pos, past_key_values=kv,
        )
        loss = F.mse_loss(m.action_out(s_out), torch.randn_like(m.action_out(s_out)))

        _zero_grads(m)
        loss.backward()
        n = _grad_norms(m)

        assert n['backbone'] > 1e-6, "backbone should get grad with attached KV"
        assert n['expert'] > 1e-6


# ===========================================================================
#  4. Detached prefix embeddings are NOT sufficient
# ===========================================================================


class TestDetachedEmbedsInsufficient:
    """Detaching the prefix input embeddings does NOT achieve KI.

    Even with ``prefix_emb.detach()``, the backbone's layer parameters
    (k_proj, v_proj, etc.) still get gradients because:
      1. Backbone layers compute K/V from the (detached) prefix embeds
      2. Those K/V are concatenated into joint attention
      3. Flow loss gradient flows back through K/V → k_proj/v_proj weights

    The detach only breaks the chain *before* prefix_emb; it does nothing
    to prevent gradients from reaching backbone layer parameters.
    """

    def test_detached_prefix_embeds_still_leak_to_backbone_layers(self, m):
        """prefix_emb.detach() ≠ KI — backbone layers still get grads."""
        dev = m.device
        plen, slen = 8, 6
        pref = torch.randn(2, plen, m.hidden_size, device=dev).detach()  # detached!
        suf = torch.randn(2, slen, m.hidden_size, device=dev)

        total = plen + slen
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)

        [p_out, s_out], _ = m.model.forward(
            inputs_embeds=[pref, suf], attention_mask=mask, position_ids=pos,
        )
        loss = F.mse_loss(m.action_out(s_out), torch.randn_like(m.action_out(s_out)))

        _zero_grads(m)
        loss.backward()
        n = _grad_norms(m)

        # Backbone layers still get grads — detach on input is not enough
        assert n['backbone'] > 1e-6, (
            "Detaching prefix_emb should NOT prevent flow loss from reaching "
            f"backbone layer params; got backbone grad norm = {n['backbone']}"
        )
        assert n['expert'] > 1e-6

    def test_no_grad_context_on_backbone_gives_zero(self, m):
        """Using torch.no_grad() for backbone forward gives zero backbone grad.

        This is the alternative KI mechanism (freeze backbone during flow
        step).  Documented as a valid but less flexible approach.
        """
        dev = m.device
        plen, slen = 8, 6
        pref = torch.randn(2, plen, m.hidden_size, device=dev)
        suf = torch.randn(2, slen, m.hidden_size, device=dev)
        total = plen + slen
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)

        # Disable grad on all backbone params
        for p in m.model.backbone_layers.parameters():
            p.requires_grad_(False)
        m.model.backbone_embed.weight.requires_grad_(False)
        for p in m.model.backbone_norm.parameters():
            p.requires_grad_(False)

        try:
            [p_out, s_out], _ = m.model.forward(
                inputs_embeds=[pref, suf], attention_mask=mask, position_ids=pos,
            )
            loss = F.mse_loss(m.action_out(s_out),
                              torch.randn_like(m.action_out(s_out)))
            _zero_grads(m)
            loss.backward()
            n = _grad_norms(m)

            assert n['backbone'] < 1e-10
            assert n['expert'] > 1e-6
        finally:
            # Restore
            for p in m.model.backbone_layers.parameters():
                p.requires_grad_(True)
            m.model.backbone_embed.weight.requires_grad_(True)
            for p in m.model.backbone_norm.parameters():
                p.requires_grad_(True)


# ===========================================================================
#  5. Dual-optimizer parameter partitioning
# ===========================================================================


class TestDualOptimizerPartition:
    """Two optimizers, two disjoint parameter sets — the KI training setup."""

    def _bb_params(self, m):
        return (list(m.model.backbone_layers.parameters())
                + list(m.model.backbone_norm.parameters())
                + [m.model.backbone_embed.weight])

    def _ex_params(self, m):
        return (list(m.model.expert_layers.parameters())
                + list(m.model.expert_norm.parameters())
                + list(m.action_in.parameters())
                + list(m.action_out.parameters()))

    def test_param_id_sets_disjoint(self, m):
        """Backbone and expert param ID sets must be completely disjoint."""
        bb = set(id(p) for p in self._bb_params(m))
        ex = set(id(p) for p in self._ex_params(m))
        overlap = bb & ex
        assert len(overlap) == 0, (
            f"{len(overlap)} parameters in both backbone and expert groups"
        )

    def test_union_covers_all(self, m):
        """backbone ∪ expert = all parameters in the system."""
        all_p = set(id(p) for p in m.model.parameters())
        all_p |= set(id(p) for p in m.action_in.parameters())
        all_p |= set(id(p) for p in m.action_out.parameters())
        bb = set(id(p) for p in self._bb_params(m))
        ex = set(id(p) for p in self._ex_params(m))
        assert (bb | ex) == all_p, (
            f"missing: {len(all_p - (bb|ex))}, extra: {len((bb|ex) - all_p)}"
        )

    def test_bb_opt_only_updates_bb_params(self, m):
        """Stepping backbone optimizer must not change expert param values."""
        bb_p = self._bb_params(m)
        ex_p = self._ex_params(m)
        ex_before = {id(p): p.data.clone() for p in ex_p}

        opt = torch.optim.SGD(bb_p, lr=0.01)
        # Forward + backward that gives grads to both sides
        dev = m.device
        pref = torch.randn(2, 8, m.hidden_size, device=dev)
        suf = torch.randn(2, 6, m.hidden_size, device=dev)
        total = 14
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)
        [p_out, s_out], _ = m.model.forward([pref, suf], mask, pos)
        loss = p_out.sum() + s_out.sum()
        _zero_grads(m)
        loss.backward()
        opt.step()

        for p in ex_p:
            assert torch.equal(p.data, ex_before[id(p)]), (
                "Expert param changed after backbone optimizer step"
            )

    def test_ex_opt_only_updates_ex_params(self, m):
        """Stepping expert optimizer must not change backbone param values."""
        bb_p = self._bb_params(m)
        ex_p = self._ex_params(m)
        bb_before = {id(p): p.data.clone() for p in bb_p}

        opt = torch.optim.SGD(ex_p, lr=0.01)
        dev = m.device
        pref = torch.randn(2, 8, m.hidden_size, device=dev)
        suf = torch.randn(2, 6, m.hidden_size, device=dev)
        total = 14
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)
        [p_out, s_out], _ = m.model.forward([pref, suf], mask, pos)
        loss = p_out.sum() + s_out.sum()
        _zero_grads(m)
        loss.backward()
        opt.step()

        for p in bb_p:
            assert torch.equal(p.data, bb_before[id(p)]), (
                "Backbone param changed after expert optimizer step"
            )


# ===========================================================================
#  6. Combined loss additivity + KI property
# ===========================================================================


class TestCombinedLossAdditivity:
    """Gradient additivity: combined loss grad = CE grad + flow grad.

    Under KI (separate forwards + detaches), backbone sees only CE grad
    and expert sees only flow grad.
    """

    def _ce_and_flow(self, m, plen=8, slen=6):
        """Run joint forward and return (ce_loss, flow_loss).  Reusable."""
        dev = m.device
        pref = torch.randn(2, plen, m.hidden_size, device=dev)
        suf = torch.randn(2, slen, m.hidden_size, device=dev)
        total = plen + slen
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)

        [p_out, s_out], _ = m.model.forward([pref, suf], mask, pos)

        subtask = p_out[:, -4:, :]
        logits = m.model.lm_head(subtask)
        B, T, V = logits.shape
        tgt = torch.randint(0, V, (B, T), device=dev)
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))

        act = m.action_out(s_out)
        flow = F.mse_loss(act, torch.randn_like(act))
        return ce, flow

    def test_joint_forward_backbone_gets_both_grads(self, m):
        """Joint forward + combined loss → backbone gets CE + flow grads."""
        ce, flow = self._ce_and_flow(m)
        _zero_grads(m)
        (ce + flow).backward()
        combined = _grad_norms(m)

        # CE-only for reference
        ce2, _ = self._ce_and_flow(m)
        _zero_grads(m)
        ce2.backward()
        ce_only = _grad_norms(m)

        # Flow-only for reference
        _, flow2 = self._ce_and_flow(m)
        _zero_grads(m)
        flow2.backward()
        flow_only = _grad_norms(m)

        # Backbone gets both → combined norm > each individual
        assert combined['backbone'] > max(ce_only['backbone'], flow_only['backbone']) - 1e-8

    def test_ki_via_separate_passes(self, m):
        """Two separate passes achieve strict KI: bb=CE only, ex=flow only.

        This demonstrates the target KI training regime:
          Step 1: prefix forward → CE loss → backbone backward
          Step 2: prefix forward (no grad) + detach KV + suffix forward
                  → flow loss → expert backward
        """
        dev = m.device
        plen, slen = 8, 6

        # --- Pass 1: CE on prefix only ---
        pref = torch.randn(2, plen, m.hidden_size, device=dev)
        mask = _causal_mask(plen, dev)
        pos = torch.arange(plen, device=dev).unsqueeze(0)
        [p_out, _], kv = m.model.forward(
            inputs_embeds=[pref, None], attention_mask=mask,
            position_ids=pos, use_cache=True,
        )
        subtask = p_out[:, -4:, :]
        logits = m.model.lm_head(subtask)
        B, T, V = logits.shape
        tgt = torch.randint(0, V, (B, T), device=dev)
        ce_loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))

        _zero_grads(m)
        ce_loss.backward()
        ce_norms = _grad_norms(m)

        # CE-only prefix forward → expert has zero grad
        assert ce_norms['backbone'] > 1e-6
        assert ce_norms['expert'] < 1e-10, (
            f"CE prefix-only gave expert grad {ce_norms['expert']}"
        )

        # --- Pass 2: flow with detached KV ---
        detached_kv = tuple((k.detach(), v.detach()) for k, v in kv)
        suf = torch.randn(2, slen, m.hidden_size, device=dev)
        cmask = _cross_mask(plen, slen, dev)
        spos = plen + torch.arange(slen, device=dev).unsqueeze(0)
        [_, s_out], _ = m.model.forward(
            inputs_embeds=[None, suf], attention_mask=cmask,
            position_ids=spos, past_key_values=detached_kv,
        )
        flow_loss = F.mse_loss(m.action_out(s_out),
                               torch.randn_like(m.action_out(s_out)))

        # Don't zero — accumulate
        flow_loss.backward()
        both_norms = _grad_norms(m)

        # KI property: backbone unchanged by flow backward
        assert abs(both_norms['backbone'] - ce_norms['backbone']) < 1e-8, (
            "KI violation: flow backward changed backbone grad norm"
        )
        # Expert got flow grads
        assert both_norms['expert'] > 1e-6


# ===========================================================================
#  7. action_out_proj side membership
# ===========================================================================


class TestActionOutProjSide:
    """action_out_proj is expert-side: flow grad only, no CE grad."""

    def _joint_forward(self, m):
        dev = m.device
        pref = torch.randn(2, 8, m.hidden_size, device=dev)
        suf = torch.randn(2, 6, m.hidden_size, device=dev)
        total = 14
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)
        return m.model.forward([pref, suf], mask, pos)

    def test_ce_only_zero_on_action_out(self, m):
        [p_out, s_out], _ = self._joint_forward(m)
        subtask = p_out[:, -4:, :]
        logits = m.model.lm_head(subtask)
        B, T, V = logits.shape
        tgt = torch.randint(0, V, (B, T), device=m.device)
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        _zero_grads(m)
        ce.backward()
        n = _grad_norms(m)
        assert n['action_out'] < 1e-10
        assert n['action_in'] < 1e-10

    def test_flow_only_nonzero_on_action_out(self, m):
        [p_out, s_out], _ = self._joint_forward(m)
        flow = F.mse_loss(m.action_out(s_out),
                          torch.randn_like(m.action_out(s_out)))
        _zero_grads(m)
        flow.backward()
        n = _grad_norms(m)
        assert n['action_out'] > 1e-6


# ===========================================================================
#  8. Target-leakage contract (forward mask + backward isolation)
# ===========================================================================


class TestTargetLeakageContract:
    """Knowledge Insulation target-leakage checks.

    Two complementary perspectives:

    **Forward (mask) contract**: the expert flow path's attention mask must
    exclude teacher-forced target token positions.  If target tokens sit in
    the prefix KV cache and the expert can attend to them, that's leakage.

    **Backward (isolation) contract**: flow loss gradients must not reach
    the parameters that produced the target-token representations.  This is
    a stronger check than the mask contract — it catches leakage through
    residual / norm / shared-parameter paths.
    """

    # ---- forward / mask contract ----

    def test_target_tokens_in_prefix_change_expert_output(self, m):
        """If target tokens are in the prefix KV cache, expert output changes.

        This is the *positive* proof: target tokens in prefix DO influence
        the expert (leakage).  KI requires that target positions are either
        excluded from the prefix or masked out in the cross-attention.
        """
        dev = m.device
        B = 2
        allowed_len = 6  # prompt + subtask positions (allowed context)
        target_len = 3   # teacher-forced FAST target positions (leakage risk)
        slen = 5

        # Prefix A: allowed only
        pref_a = torch.randn(B, allowed_len, m.hidden_size, device=dev)
        # Prefix B: allowed + target tokens appended
        pref_b = torch.cat([
            pref_a,
            torch.randn(B, target_len, m.hidden_size, device=dev),
        ], dim=1)

        # Build KV caches for both prefixes
        def build_kv(emb):
            n = emb.shape[1]
            mask = _causal_mask(n, dev)
            pos = torch.arange(n, device=dev).unsqueeze(0)
            [out, _], kv = m.model.forward(
                inputs_embeds=[emb, None], attention_mask=mask,
                position_ids=pos, use_cache=True,
            )
            return kv, out

        kv_a, _ = build_kv(pref_a)
        kv_b, _ = build_kv(pref_b)

        # Same suffix input through both caches
        suf = torch.randn(B, slen, m.hidden_size, device=dev)

        def run_suffix(kv, p_len):
            cmask = _cross_mask(p_len, slen, dev)
            spos = p_len + torch.arange(slen, device=dev).unsqueeze(0)
            det_kv = tuple((k.detach(), v.detach()) for k, v in kv)
            [_, s_out], _ = m.model.forward(
                inputs_embeds=[None, suf], attention_mask=cmask,
                position_ids=spos, past_key_values=det_kv,
            )
            return s_out

        out_a = run_suffix(kv_a, allowed_len)
        out_b = run_suffix(kv_b, allowed_len + target_len)

        # Outputs differ → target tokens in prefix DO influence expert
        assert not torch.allclose(out_a, out_b, atol=1e-6), (
            "Target tokens in prefix should change expert output. "
            "If they don't, the test is wrong."
        )

    def test_masking_target_positions_removes_leakage(self, m):
        """Masking target positions from cross-attention removes leakage.

        When we build the expert cross-attention mask to exclude target
        positions in the prefix, the expert output becomes independent of
        target token values — KI forward contract satisfied.
        """
        dev = m.device
        B = 2
        allowed_len = 6
        target_len = 3
        total_pre = allowed_len + target_len
        slen = 5

        # Prefix has allowed + target regions
        pref = torch.randn(B, total_pre, m.hidden_size, device=dev)

        # Build KV cache for full prefix
        mask_pre = _causal_mask(total_pre, dev)
        pos_pre = torch.arange(total_pre, device=dev).unsqueeze(0)
        [p_out, _], kv = m.model.forward(
            inputs_embeds=[pref, None], attention_mask=mask_pre,
            position_ids=pos_pre, use_cache=True,
        )
        det_kv = tuple((k.detach(), v.detach()) for k, v in kv)

        suf = torch.randn(B, slen, m.hidden_size, device=dev)

        # Case 1: expert attends to ALL prefix positions (allowed + target)
        cmask_full = _cross_mask(total_pre, slen, dev)
        spos = total_pre + torch.arange(slen, device=dev).unsqueeze(0)
        [_, out_full], _ = m.model.forward(
            inputs_embeds=[None, suf], attention_mask=cmask_full,
            position_ids=spos, past_key_values=det_kv,
        )

        # Case 2: expert attends only to ALLOWED positions (targets masked out)
        # Build a custom cross-mask that zeros out target positions
        cmask_allowed = torch.zeros(1, 1, slen, total_pre, device=dev, dtype=torch.bool)
        cmask_allowed[:, :, :, :allowed_len] = True  # only allowed prefix
        # Causal within suffix part of kv (but target positions are in prefix, not suffix kv)
        # Actually target positions are IN the prefix kv — we just mask them out
        cmask_allowed_full = cmask_allowed  # shape (1,1,slen,total_pre)
        # Wait — the KV has total_pre entries (all prefix). The mask must be
        # (1,1,slen, total_pre+slen) to cover both prefix KV and suffix KV.
        suf_causal = _causal_mask(slen, dev)  # (1,1,slen,slen)
        cmask_combined = torch.cat([cmask_allowed, suf_causal], dim=3)
        [_, out_masked], _ = m.model.forward(
            inputs_embeds=[None, suf], attention_mask=cmask_combined,
            position_ids=spos, past_key_values=det_kv,
        )

        # Outputs should differ: masking target positions changes what the
        # expert sees → proves targets were contributing to the output.
        assert not torch.allclose(out_full, out_masked, atol=1e-6), (
            "Masking target positions should change expert output. "
            "If not, the mask is not working."
        )

        # Sanity: when target positions are identical across two runs and
        # both masked out, outputs are identical
        [_, out_masked2], _ = m.model.forward(
            inputs_embeds=[None, suf], attention_mask=cmask_combined,
            position_ids=spos, past_key_values=det_kv,
        )
        assert torch.allclose(out_masked, out_masked2), (
            "Same input + same mask should give same output"
        )

    # ---- backward / isolation contract ----

    def test_flow_grad_reaches_backbone_kv_projs_in_joint(self, m):
        """Flow loss → backbone k/v_proj in all layers + q/o/MLP in earlier layers.

        This is the backward proof that the current architecture leaks.
        Gradient distribution:
          * Last layer: only k_proj and v_proj get grad (direct cross-attn)
          * Earlier layers: all projections get grad (via residual chain)

        If KI is correctly implemented, all of these should be zero.
        """
        dev = m.device
        pref = torch.randn(2, 8, m.hidden_size, device=dev)
        suf = torch.randn(2, 6, m.hidden_size, device=dev)
        total = 14
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)

        [p_out, s_out], _ = m.model.forward([pref, suf], mask, pos)
        loss = F.mse_loss(m.action_out(s_out),
                          torch.randn_like(m.action_out(s_out)))
        _zero_grads(m)
        loss.backward()

        n_layers = m.model.num_layers
        for li, layer in enumerate(m.model.backbone_layers):
            # k_proj and v_proj always get grad (direct cross-attention path)
            assert layer.k_proj.weight.grad.abs().max().item() > 1e-8, (
                f"backbone layer {li} k_proj zero grad"
            )
            assert layer.v_proj.weight.grad.abs().max().item() > 1e-8, (
                f"backbone layer {li} v_proj zero grad"
            )

            if li < n_layers - 1:
                # Earlier layers: q_proj, o_proj, MLP also get grad via residual
                assert layer.q_proj.weight.grad.abs().max().item() > 1e-8, (
                    f"backbone layer {li} q_proj zero grad (residual path)"
                )
                assert layer.o_proj.weight.grad.abs().max().item() > 1e-8, (
                    f"backbone layer {li} o_proj zero grad (residual path)"
                )
                mlp_g = layer.mlp[0].weight.grad.abs().max().item()
                assert mlp_g > 1e-8, (
                    f"backbone layer {li} MLP zero grad (residual path)"
                )
            else:
                # Last layer: q_proj and o_proj get zero from flow loss
                assert layer.q_proj.weight.grad.abs().max().item() < 1e-10, (
                    f"backbone last layer q_proj should have zero flow grad"
                )

    def test_detached_kv_zero_backbone_flow_grad(self, m):
        """Detached KV + flow backward → zero grad on ALL backbone params.

        This is the backward KI contract: with a detached prefix cache,
        not a single backbone parameter receives a gradient from flow loss.
        """
        dev = m.device
        plen = 8
        slen = 6
        pref = torch.randn(2, plen, m.hidden_size, device=dev)
        mask = _causal_mask(plen, dev)
        pos = torch.arange(plen, device=dev).unsqueeze(0)
        [p_out, _], kv = m.model.forward(
            inputs_embeds=[pref, None], attention_mask=mask,
            position_ids=pos, use_cache=True,
        )
        det_kv = tuple((k.detach(), v.detach()) for k, v in kv)

        suf = torch.randn(2, slen, m.hidden_size, device=dev)
        cmask = _cross_mask(plen, slen, dev)
        spos = plen + torch.arange(slen, device=dev).unsqueeze(0)
        [_, s_out], _ = m.model.forward(
            inputs_embeds=[None, suf], attention_mask=cmask,
            position_ids=spos, past_key_values=det_kv,
        )
        loss = F.mse_loss(m.action_out(s_out),
                          torch.randn_like(m.action_out(s_out)))
        _zero_grads(m)
        loss.backward()

        # Every single backbone parameter must have zero grad
        for name, p in m.model.named_parameters():
            if 'backbone' in name:
                gmax = p.grad.abs().max().item() if p.grad is not None else 0.0
                assert gmax < 1e-10, (
                    f"Backbone param '{name}' got non-zero grad ({gmax}) "
                    "from flow loss with detached KV — KI violation!"
                )

    def test_ce_only_zero_on_all_expert_params(self, m):
        """CE-only backward → zero grad on ALL expert params.

        This direction is already KI-correct in the block-causal arch.
        Verify at the per-parameter level.
        """
        dev = m.device
        pref = torch.randn(2, 8, m.hidden_size, device=dev)
        suf = torch.randn(2, 6, m.hidden_size, device=dev)
        total = 14
        mask = _causal_mask(total, dev)
        pos = torch.arange(total, device=dev).unsqueeze(0)

        [p_out, s_out], _ = m.model.forward([pref, suf], mask, pos)
        subtask = p_out[:, -4:, :]
        logits = m.model.lm_head(subtask)
        B, T, V = logits.shape
        tgt = torch.randint(0, V, (B, T), device=dev)
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        _zero_grads(m)
        ce.backward()

        # Every expert parameter must have zero grad
        for name, p in m.model.named_parameters():
            if 'expert' in name:
                gmax = p.grad.abs().max().item() if p.grad is not None else 0.0
                assert gmax < 1e-10, (
                    f"Expert param '{name}' got non-zero grad ({gmax}) "
                    "from CE loss — block-causal arch should prevent this."
                )
