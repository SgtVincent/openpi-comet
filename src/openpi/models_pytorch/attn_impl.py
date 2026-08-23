"""Attention-implementation selection for PaliGemma-based models.

The historical code hardcoded ``_attn_implementation = "eager"`` at every
forward site. Eager attention materializes the full ``[B, H, L, L]`` score
matrix, which dominates activation memory for the ~1000-token prefix used by
the pi05-KI joint-query model.

This module centralizes the choice so it can be switched per run through the
``OPENPI_ATTN_IMPL`` environment variable without editing model code.

The default stays ``eager`` so existing runs keep their exact numerics unless
the variable is set explicitly.

Notes on backend applicability for this model family:

* ``eager`` always works and is the historical default.
* ``sdpa`` accepts the additive float mask produced by
  ``_prepare_attention_masks_4d`` and avoids materializing the score matrix.
  Because an explicit mask is supplied, PyTorch cannot select the FLASH
  backend and falls back to the memory-efficient kernel, which is still a
  large activation-memory win over eager.
* ``flash_attention_2`` / ``flash_attention_3`` cannot express the arbitrary
  block-causal mask this model builds, so they are rejected here.
* ``flex_attention`` could exploit the block structure, but only if the mask
  is rewritten as a boolean ``mask_mod``; passing the existing additive mask
  yields no block-sparsity benefit. It is therefore not accepted yet.
"""

from __future__ import annotations

import os

_SUPPORTED_ATTN_IMPLS = ("eager", "sdpa")


def resolve_attn_impl() -> str:
    """Return the attention implementation requested for this process.

    Reads ``OPENPI_ATTN_IMPL`` and defaults to ``eager`` so unset environments
    behave exactly as before. An unsupported value raises instead of silently
    degrading, so a typo cannot quietly change training numerics.
    """
    impl = os.environ.get("OPENPI_ATTN_IMPL", "eager").strip().lower()
    if impl not in _SUPPORTED_ATTN_IMPLS:
        raise ValueError(
            f"OPENPI_ATTN_IMPL must be one of {_SUPPORTED_ATTN_IMPLS}, got {impl!r}"
        )
    return impl
