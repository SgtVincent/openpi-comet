"""Attention-implementation equivalence test: eager vs sdpa.

Validates the ``OPENPI_ATTN_IMPL`` switch exposed by
``openpi.models_pytorch.attn_impl.resolve_attn_impl``.

WHY THIS EXISTS
===============
Production hard-coded ``config._attn_implementation = "eager"`` at four sites.
``eager`` materializes the full O(L^2) attention matrix, which on A100 shows up
as "GPU util 100% but power only ~90W/400W" — memory-bandwidth bound, not
compute bound.  Switching to ``sdpa`` avoids that materialization.

Before that switch can be trusted, we must prove SDPA consumes this model's
*custom 4D additive mask* with identical semantics.  That is exactly what this
file checks, on the REAL ``PaliGemmaWithExpertModel`` transformer code (tiny
config, CPU) rather than a synthetic stand-in.

MASK CONVENTION (must mirror production)
========================================
``pi0_pytorch._prepare_attention_masks_4d`` returns an **additive float32**
mask of shape ``(B, 1, L, L)`` where allowed positions are ``0.0`` and masked
positions are ``-1e4`` (a deliberately fp16-representable sentinel, NOT
``-inf``).  We reproduce that convention exactly here; using ``-inf`` instead
would test a mask this model never actually emits.

SCOPE
=====
Covered: real block-causal joint attention forward + backward, prefix-only
forward, KV-cache path, and the ``resolve_attn_impl`` env contract.
Not covered: fp16/bf16 kernel selection and wall-clock speedup — both are
architecture dependent (SDPA falls back to the ``math`` backend on sm_70 and
FlashAttention needs sm_80+), so performance must be measured on A100.

Run with::

    PYTHONNOUSERSITE=1 PYTHONPATH=src python -m pytest \
        tests/test_attn_impl_equivalence.py -v
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

from openpi.models.gemma import Config
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

# Production sentinel for masked positions (see _prepare_attention_masks_4d).
MASK_SENTINEL = -1e4

# fp32 equivalence tolerance.  eager and sdpa reassociate the same reductions
# differently, so bitwise identity is not expected; ~1e-6 is float32 noise.
FP32_ATOL = 1e-5


def _make_tiny_model(seed: int = 42) -> PaliGemmaWithExpertModel:
    """Tiny but REAL PaliGemmaWithExpertModel on CPU (mirrors test_ki_integration_real)."""
    torch.manual_seed(seed)
    cfg = Config(width=128, depth=2, mlp_dim=256, num_heads=8, num_kv_heads=1, head_dim=16)
    model = PaliGemmaWithExpertModel(cfg, cfg, precision="float32")
    if hasattr(model.gemma_expert.model, "gradient_checkpointing"):
        model.gemma_expert.model.gradient_checkpointing = False
    model.training = False
    return model


def _production_block_causal_mask(prefix_len: int, query_len: int, batch: int) -> torch.Tensor:
    """Additive 4D mask matching production semantics.

    Reproduces the pi05-KI joint-query layout documented in
    ``pi05_ki_joint_query``: the prefix block is causal, the trailing query
    block is bidirectional *within itself* and may attend all earlier prefix
    tokens, while prefix tokens must NOT attend query tokens.
    """
    total = prefix_len + query_len
    allowed = torch.zeros(batch, 1, total, total, dtype=torch.bool)
    # Prefix rows: causal over prefix only.
    for i in range(prefix_len):
        allowed[:, :, i, : i + 1] = True
    # Query rows: all prefix + the entire query block (bidirectional).
    allowed[:, :, prefix_len:, :prefix_len] = True
    allowed[:, :, prefix_len:, prefix_len:] = True
    return torch.where(
        allowed,
        torch.zeros((), dtype=torch.float32),
        torch.full((), MASK_SENTINEL, dtype=torch.float32),
    )


def _forward_with_impl(model, impl, embs, mask, pos, *, use_cache=False):
    """Run the joint forward under a specific attention implementation."""
    model.paligemma.language_model.config._attn_implementation = impl
    out, cache = model.forward(
        inputs_embeds=[embs, None],
        attention_mask=mask,
        position_ids=pos,
        use_cache=use_cache,
    )
    return out[0], cache


@pytest.fixture(scope="module")
def model():
    return _make_tiny_model()


@pytest.fixture(scope="module")
def batch(model):
    """Deterministic input batch shared by all comparisons."""
    torch.manual_seed(0)
    b, prefix_len, query_len = 2, 24, 8
    width = model.paligemma.language_model.embed_tokens.weight.shape[1]
    embs = torch.randn(b, prefix_len + query_len, width, dtype=torch.float32)
    mask = _production_block_causal_mask(prefix_len, query_len, b)
    pos = torch.arange(prefix_len + query_len).unsqueeze(0).expand(b, -1)
    return embs, mask, pos, prefix_len, query_len


# ---------------------------------------------------------------------------
#  resolve_attn_impl env contract
# ---------------------------------------------------------------------------


class TestResolveAttnImpl:
    @staticmethod
    def _resolve():
        mod = importlib.import_module("openpi.models_pytorch.attn_impl")
        return mod.resolve_attn_impl

    def test_defaults_to_eager(self, monkeypatch):
        """Unset env MUST preserve the historical hard-coded behaviour."""
        monkeypatch.delenv("OPENPI_ATTN_IMPL", raising=False)
        assert self._resolve()() == "eager"

    @pytest.mark.parametrize("value,expected", [("sdpa", "sdpa"), ("SDPA", "sdpa"), ("EAGER", "eager")])
    def test_accepts_known_values_case_insensitively(self, monkeypatch, value, expected):
        monkeypatch.setenv("OPENPI_ATTN_IMPL", value)
        assert self._resolve()() == expected

    def test_rejects_unknown_value(self, monkeypatch):
        """A typo must fail loudly rather than silently degrade to eager."""
        monkeypatch.setenv("OPENPI_ATTN_IMPL", "flash_attention_2")
        with pytest.raises(ValueError, match="OPENPI_ATTN_IMPL"):
            self._resolve()()


# ---------------------------------------------------------------------------
#  Numerical equivalence on the real transformer
# ---------------------------------------------------------------------------


class TestForwardEquivalence:
    def test_joint_forward_outputs_match(self, model, batch):
        """eager and sdpa must agree on the block-causal joint forward."""
        embs, mask, pos, _, _ = batch
        with torch.no_grad():
            out_eager, _ = _forward_with_impl(model, "eager", embs, mask, pos)
            out_sdpa, _ = _forward_with_impl(model, "sdpa", embs, mask, pos)
        diff = (out_eager - out_sdpa).abs().max().item()
        assert diff < FP32_ATOL, f"eager vs sdpa max_abs_diff={diff:.3e} exceeds {FP32_ATOL:.0e}"

    def test_prefix_only_forward_with_kv_cache_matches(self, model, batch):
        """The phase-2 path (prefix-only, use_cache=True) must also agree.

        This is the forward whose KV cache feeds the action expert, so a mask
        misinterpretation here would silently corrupt expert conditioning.
        """
        embs, _, _, prefix_len, _ = batch
        prefix = embs[:, :prefix_len, :]
        b = prefix.shape[0]
        # Prefix-only causal mask, production sentinel convention.
        allowed = torch.zeros(b, 1, prefix_len, prefix_len, dtype=torch.bool)
        for i in range(prefix_len):
            allowed[:, :, i, : i + 1] = True
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=torch.float32),
            torch.full((), MASK_SENTINEL, dtype=torch.float32),
        )
        pos = torch.arange(prefix_len).unsqueeze(0).expand(b, -1)

        with torch.no_grad():
            out_eager, cache_eager = _forward_with_impl(
                model, "eager", prefix, mask, pos, use_cache=True
            )
            out_sdpa, cache_sdpa = _forward_with_impl(
                model, "sdpa", prefix, mask, pos, use_cache=True
            )

        diff = (out_eager - out_sdpa).abs().max().item()
        assert diff < FP32_ATOL, f"prefix-only max_abs_diff={diff:.3e}"

        # The cached K/V that the expert cross-attends must match too.
        assert cache_eager is not None and cache_sdpa is not None
        for layer_idx in range(len(cache_eager.key_cache)):
            k_diff = (cache_eager.key_cache[layer_idx] - cache_sdpa.key_cache[layer_idx]).abs().max().item()
            v_diff = (cache_eager.value_cache[layer_idx] - cache_sdpa.value_cache[layer_idx]).abs().max().item()
            assert k_diff < FP32_ATOL, f"layer {layer_idx} key cache diff={k_diff:.3e}"
            assert v_diff < FP32_ATOL, f"layer {layer_idx} value cache diff={v_diff:.3e}"

    def test_prefix_tokens_cannot_attend_query_tokens(self, model, batch):
        """Guard the invariant that makes phase-1 KV reuse valid.

        The module docstring states prefix tokens never attend query tokens.
        That is what allows phase-1's prefix KV to be reused by phase-2.  If
        this ever regresses, KV reuse becomes mathematically unsound, so pin
        it here: perturbing ONLY the query tokens must leave prefix outputs
        unchanged under both attention implementations.
        """
        embs, mask, pos, prefix_len, _ = batch
        perturbed = embs.clone()
        perturbed[:, prefix_len:, :] += 10.0  # large perturbation, query block only

        for impl in ("eager", "sdpa"):
            with torch.no_grad():
                base, _ = _forward_with_impl(model, impl, embs, mask, pos)
                pert, _ = _forward_with_impl(model, impl, perturbed, mask, pos)
            prefix_diff = (base[:, :prefix_len, :] - pert[:, :prefix_len, :]).abs().max().item()
            assert prefix_diff < FP32_ATOL, (
                f"[{impl}] prefix output changed by {prefix_diff:.3e} when only query "
                "tokens were perturbed — block-causal invariant violated, phase-1 KV "
                "reuse would be unsound"
            )


class TestBackwardEquivalence:
    def test_gradients_match(self, model, batch):
        """Equal forward values are not enough — gradients drive training."""
        embs, mask, pos, _, _ = batch

        grads = {}
        for impl in ("eager", "sdpa"):
            model.zero_grad(set_to_none=True)
            out, _ = _forward_with_impl(model, impl, embs, mask, pos)
            out.square().mean().backward()
            grads[impl] = {
                name: p.grad.detach().clone()
                for name, p in model.paligemma.language_model.named_parameters()
                if p.grad is not None
            }

        assert grads["eager"], "no gradients captured — test is vacuous"
        assert set(grads["eager"]) == set(grads["sdpa"])

        worst_name, worst = None, 0.0
        for name, g_eager in grads["eager"].items():
            d = (g_eager - grads["sdpa"][name]).abs().max().item()
            if d > worst:
                worst_name, worst = name, d
        assert worst < FP32_ATOL, f"largest grad divergence {worst:.3e} at {worst_name}"


def test_masked_positions_are_actually_suppressed(model, batch):
    """Sanity-check the -1e4 sentinel really suppresses attention.

    ``-1e4`` is finite (chosen to stay fp16-representable), so unlike ``-inf``
    it does not *structurally* zero the softmax term.  Confirm it is
    numerically sufficient: perturbing tokens that a position is masked
    against must not move that position's output.
    """
    embs, mask, pos, prefix_len, _ = batch
    # Position 0 is masked against everything except itself (causal prefix).
    perturbed = embs.clone()
    perturbed[:, 1:prefix_len, :] += 25.0

    for impl in ("eager", "sdpa"):
        with torch.no_grad():
            base, _ = _forward_with_impl(model, impl, embs, mask, pos)
            pert, _ = _forward_with_impl(model, impl, perturbed, mask, pos)
        d = (base[:, 0, :] - pert[:, 0, :]).abs().max().item()
        assert d < FP32_ATOL, f"[{impl}] position 0 moved {d:.3e}; sentinel {MASK_SENTINEL} insufficient"
