"""Hierarchy token cache and prefix-KV staleness guard (design P1, item 2).

Two separate things live here, and keeping them separate is the point.

1. :class:`HierarchyTokenCache` -- the *allowed* cache.  It holds hierarchy
   **token ids** and nothing else, so that a Planner tick's output can be held
   across K action chunks (design section 2.2).

2. :func:`assert_prefix_fresh` -- the guard against the *disallowed* cache.
   Design section 4.5 forbids reusing a full prefix KV across chunks because
   the prefix encodes the observation.  In this codebase that risk is currently
   absent only by accident: ``subtask_expert.py`` hardcodes
   ``past_key_values=None`` in ``encode_prefix``, so the prefix is rebuilt every
   call.  Introducing a cache is what brings the risk back, so the guard ships
   with it.

Why the staleness surface is wider than the design says
-------------------------------------------------------
Design section 4.5 says a reused prefix KV would carry a stale image.  It also
carries a stale **state**, for a reason the doc does not give: for pi05 the
state is not a prefix tensor at all.  ``embed_suffix`` skips the state
projection entirely (``pi0_pytorch.py:340`` ``if not self.pi05:``) and the state
is instead discretised into 256 bins and interpolated into the prompt *string*
by ``SubtaskTokenizer.tokenize_prompt`` (``tokenizer.py:504-507``).  So the
state lives inside the cached text tokens.  The fingerprint below therefore
covers the language tokens as well as the images -- covering only the images
would leave a stale state undetected.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# Elements sampled per tensor when fingerprinting.  Large enough that an
# observation change is overwhelmingly likely to move the digest, small enough
# that the guard costs nothing next to a VLM forward pass.
_FINGERPRINT_SAMPLES = 512


class StalePrefixKVError(RuntimeError):
    """Raised when a prefix KV built from one observation is used with another.

    This is design section 4.5 turned into something CI can catch: it fires
    exactly when a caller reuses a cached prefix across action chunks instead of
    recomputing it from the latest observation.
    """


class HierarchyCacheError(RuntimeError):
    """Raised when the hierarchy cache is misused (wrong payload, or read while empty)."""


def _tensor_digest(value: Any) -> tuple:
    """Cheap, deterministic, content-sensitive digest of one tensor.

    Returns a hashable tuple.  Uses a fixed stride so the sampled elements --
    and hence the reduction order -- are identical between two calls on
    equal-shaped tensors, which is what makes the comparison meaningful.
    """
    if value is None:
        return ("none",)

    # torch is imported lazily: this module is also imported by data-side code
    # that must not pay for a torch import.
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        n = int(flat.numel())
        if n == 0:
            return ("t", tuple(value.shape), str(value.dtype), 0.0, 0.0)
        stride = max(1, n // _FINGERPRINT_SAMPLES)
        sample = flat[::stride].to(torch.float64)
        return (
            "t",
            tuple(int(d) for d in value.shape),
            str(value.dtype),
            float(sample.sum().item()),
            float(sample.abs().max().item()),
        )

    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        n = int(flat.size)
        if n == 0:
            return ("n", tuple(value.shape), str(value.dtype), 0.0, 0.0)
        stride = max(1, n // _FINGERPRINT_SAMPLES)
        sample = flat[::stride].astype(np.float64)
        return ("n", tuple(value.shape), str(value.dtype), float(sample.sum()), float(np.abs(sample).max()))

    if isinstance(value, list | tuple):
        return tuple(_tensor_digest(v) for v in value)

    return ("scalar", repr(value))


def observation_fingerprint(images: Any, lang_tokens: Any = None, *, extra: Any = None) -> tuple:
    """Fingerprint the observation that a prefix KV was built from.

    Covers images *and* language tokens, because for pi05 the robot state is
    carried as discretised text inside the language tokens (see module
    docstring).  ``extra`` is available for callers that add further
    observation-derived prefix content.
    """
    return ("obs_fp_v1", _tensor_digest(images), _tensor_digest(lang_tokens), _tensor_digest(extra))


def assert_prefix_fresh(prefix_ctx: dict, expected_fingerprint: tuple | None, *, where: str = "") -> None:
    """Raise :class:`StalePrefixKVError` if ``prefix_ctx`` was built from another observation.

    A ``prefix_ctx`` without a fingerprint, or an ``expected_fingerprint`` of
    ``None``, is a no-op: this keeps every pre-existing caller working while
    still catching the specific mistake the design warns about.
    """
    if expected_fingerprint is None:
        return
    actual = prefix_ctx.get("obs_fingerprint") if isinstance(prefix_ctx, dict) else None
    if actual is None:
        return
    if actual != expected_fingerprint:
        raise StalePrefixKVError(
            "prefix KV was encoded from a different observation than the one being acted on"
            f"{' at ' + where if where else ''}. Design section 4.5: the prefix encodes the image and "
            "(for pi05) the discretised state, so it must be recomputed every action chunk. "
            "Cache the hierarchy token ids instead."
        )


@dataclasses.dataclass
class HierarchyTokenCache:
    """Holds the hierarchy **token ids** produced by the most recent Planner tick.

    This is the only thing design section 4.4 permits to persist across action
    chunks.  The cache therefore refuses any payload that is not an integer
    token array -- in particular it refuses a ``past_key_values``/``Cache``
    object, which is the mistake section 4.5 is about.

    Invalidation is explicit.  There is no TTL and no implicit expiry: a stale
    entry that nobody invalidated is exactly the failure mode we are guarding,
    so the cost of forgetting must be a loud error and not a silent stale read.
    """

    tokens: Any = None
    mask: Any = None
    text: str | None = None
    #: Monotonic count of Planner ticks stored; useful for rollout logging.
    generation: int = 0
    #: Why the cache is currently empty, if it is.  Purely diagnostic.
    invalidated_reason: str | None = "never populated"

    _KV_MARKERS = ("past_key_values", "cache", "dynamiccache", "staticcache")

    def _reject_non_token_payload(self, tokens: Any) -> None:
        type_name = type(tokens).__name__.lower()
        if any(marker in type_name for marker in self._KV_MARKERS):
            raise HierarchyCacheError(
                f"refusing to cache a {type(tokens).__name__}: the hierarchy cache holds token ids only. "
                "Caching a prefix KV would carry a stale observation across action chunks "
                "(design section 4.5)."
            )
        if hasattr(tokens, "dtype"):
            dtype = tokens.dtype
            is_int = getattr(dtype, "is_floating_point", None) is False or np.issubdtype(
                getattr(dtype, "type", np.int64) if isinstance(dtype, np.dtype) else np.int64, np.integer
            )
            # torch tensors expose dtype.is_floating_point; numpy goes via np.issubdtype.
            if isinstance(tokens, np.ndarray):
                is_int = np.issubdtype(tokens.dtype, np.integer)
            elif hasattr(dtype, "is_floating_point"):
                is_int = not bool(dtype.is_floating_point)
            if not is_int:
                raise HierarchyCacheError(
                    f"refusing to cache a non-integer tensor of dtype {dtype}: the hierarchy cache "
                    "holds token ids, not embeddings or KV tensors."
                )
        elif not isinstance(tokens, list | tuple):
            raise HierarchyCacheError(
                f"refusing to cache payload of type {type(tokens).__name__}: expected an integer token array."
            )

    @property
    def is_populated(self) -> bool:
        return self.tokens is not None

    def store(self, tokens: Any, mask: Any = None, *, text: str | None = None) -> None:
        """Commit a Planner tick's output.

        Design section 2.3: the minimal commit is an overwrite.  Empty output is
        rejected so that the rollout's documented degradation ("keep the old
        tokens when generation is empty") is a decision the caller makes
        explicitly, not something the cache does behind its back.
        """
        if tokens is None:
            raise HierarchyCacheError(
                "refusing to store None. To drop the held hierarchy call invalidate(reason=...); "
                "to keep the previous hierarchy on an empty generation, skip the store."
            )
        self._reject_non_token_payload(tokens)
        length = len(tokens) if not hasattr(tokens, "shape") else int(np.prod(tokens.shape))
        if length == 0:
            raise HierarchyCacheError(
                "refusing to store an empty token array; skip the store to keep the previous hierarchy."
            )
        self.tokens = tokens
        self.mask = mask
        self.text = text
        self.generation += 1
        self.invalidated_reason = None

    def get(self) -> tuple[Any, Any]:
        """Read the held hierarchy. Raises if the cache was invalidated or never filled."""
        if self.tokens is None:
            raise HierarchyCacheError(
                f"hierarchy cache is empty (reason: {self.invalidated_reason}). "
                "Run a Planner tick before reading, or handle the empty case explicitly."
            )
        return self.tokens, self.mask

    def get_or_none(self) -> tuple[Any, Any] | None:
        """Non-raising read, for callers that legitimately handle a cold cache."""
        if self.tokens is None:
            return None
        return self.tokens, self.mask

    def invalidate(self, reason: str) -> None:
        """Explicitly drop the held hierarchy.

        A reason is mandatory: an unexplained invalidation in a rollout log is
        indistinguishable from a bug.
        """
        if not reason:
            raise HierarchyCacheError("invalidate() requires a non-empty reason")
        self.tokens = None
        self.mask = None
        self.text = None
        self.invalidated_reason = reason
