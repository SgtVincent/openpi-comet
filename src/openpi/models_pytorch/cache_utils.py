"""KV cache utilities for inference-time prefix cache management.

The :class:`_PreserveCacheLen` context manager prevents shared prefix KV caches
from growing across repeated denoising steps (flow-matching inference), which
would otherwise cause attention mask size mismatches and corrupt conditioning.

HuggingFace attention layers mutate ``past_key_values`` in-place even when
``use_cache=False``, so every suffix forward pass extends the cache by the
suffix length.  Truncating back to the original prefix length after each step
keeps the cache stable across iterations.
"""

from __future__ import annotations

from typing import Any


def get_cache_seq_len(past_key_values: Any, layer_idx: int = 0) -> int:
    """Return the sequence length of a KV cache.

    Supports both HuggingFace :class:`~transformers.cache_utils.DynamicCache`
    objects (via ``get_seq_length``) and legacy ``list[tuple[Tensor, Tensor]]``
    tuple-of-tensors formats.

    Args:
        past_key_values: KV cache object (DynamicCache, list, or tuple).
        layer_idx: Which layer to query (default 0; callers should verify
            all layers have the same length when necessary).

    Returns:
        Sequence length (number of key/value positions) of the cache.
    """
    if hasattr(past_key_values, "get_seq_length"):
        # HuggingFace DynamicCache / Cache object
        return past_key_values.get_seq_length(layer_idx=layer_idx)
    if isinstance(past_key_values, (list, tuple)):
        # Legacy tuple format: past_key_values[layer_idx] = (key, value)
        layer = past_key_values[layer_idx]
        # key shape: (batch, num_heads, seq_len, head_dim)
        return layer[0].shape[2]
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def truncate_cache_to_len(past_key_values: Any, target_len: int) -> None:
    """Truncate a KV cache in-place to the given sequence length.

    Supports both HuggingFace :class:`~transformers.cache_utils.DynamicCache`
    (using its native ``crop`` method) and legacy tuple-of-tensors formats.

    Args:
        past_key_values: KV cache object to truncate in-place.
        target_len: Number of positions to keep (counted from position 0).

    Raises:
        ValueError: If *target_len* is negative.
        TypeError: If *past_key_values* is not a recognized cache type.
    """
    if target_len < 0:
        raise ValueError(f"target_len must be >= 0, got {target_len}")

    if hasattr(past_key_values, "crop"):
        # HuggingFace DynamicCache has a crop method
        past_key_values.crop(target_len)
        return

    if isinstance(past_key_values, list):
        # Legacy mutable format: manually truncate each layer's key/value tensors
        for i in range(len(past_key_values)):
            k, v = past_key_values[i]
            past_key_values[i] = (k[:, :, :target_len, :], v[:, :, :target_len, :])
        return

    if isinstance(past_key_values, tuple):
        # Immutable tuple — convert to list, truncate, and re-wrap
        raise TypeError(
            "Cannot truncate a tuple-format cache in-place. "
            "Convert to list first, or use a DynamicCache."
        )

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


class PreserveCacheLen:
    """Context manager that saves and restores a KV cache's sequence length.

    Use this around any forward pass that receives a *shared* prefix KV cache
    (e.g.  each step of flow-matching action generation) to prevent the cache
    from growing across iterations.  The original length is captured on entry
    and restored on exit, even if the forward pass raises an exception.

    Example::

        cache = prefix_ctx["past_key_values"]
        with PreserveCacheLen(cache):
            outputs = model.forward(..., past_key_values=cache)
        # cache is back to its original prefix length here

    Args:
        past_key_values: KV cache object to protect.
    """

    def __init__(self, past_key_values: Any):
        self.past_key_values = past_key_values
        self._saved_len: int | None = None

    def __enter__(self) -> "PreserveCacheLen":
        self._saved_len = get_cache_seq_len(self.past_key_values)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._saved_len is not None:
            current_len = get_cache_seq_len(self.past_key_values)
            if current_len != self._saved_len:
                truncate_cache_to_len(self.past_key_values, self._saved_len)
        return False  # Re-raise any exception
