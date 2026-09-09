"""HeldMemory-K rollout scheduling (design P1, item 3).

Implements the two paths of design section 2.3 and the Planner stride of
section 2.4.  Deliberately contains **no** model, tokenizer or label knowledge:
it schedules ticks and enforces invariants, so that the still-open choice
between plain-text and reserved-slot labels cannot reach it.

The two paths (design section 2.3)
----------------------------------
::

    Slow Planner:  Latest Image -> Task -> Latest State
                   -> Previous Memory -> AR Current Memory -> Action Query

    Fast Action:   Latest Image -> Task -> Latest State
                   -> Held Current Memory -> Action Query -> Action Chunk

The load-bearing asymmetry is that a Fast Action tick carries **only** the held
Current Memory.  The doc gives the reason explicitly: the Action Expert must
never see the old and the new plan at the same time.  :class:`TickConditioning`
makes that unrepresentable rather than merely documented.

What is intentionally NOT here
------------------------------
* Event-triggered refresh is P2 and out of scope.
* No structured parsing of generated memory, and no symbolic commit (section 2.5).
* No prefix-KV reuse of any kind.  Every chunk re-encodes from the latest
  observation; only memory token IDs persist.  This is doubly load-bearing for
  pi05, because the cached text carries a stale *state* as well as stale images.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from openpi.models.memory_cache import MemoryTokenCache
from openpi.models.memory_cache import assert_prefix_fresh

#: Design section 6.4 arm C is the main experiment. Arm A (K=1) is the
#: per-chunk-regeneration baseline, and K=1 must degenerate to exactly that.
DEFAULT_PLANNER_STRIDE: int = 5


class MemoryConditioningError(RuntimeError):
    """Raised when a tick is asked to carry a combination the design forbids."""


class PlannerScheduleError(ValueError):
    """Raised when the Planner stride or chunk index is not usable."""


@dataclasses.dataclass(frozen=True)
class PlannerSchedule:
    """Decides which action chunks are Planner ticks.

    Design section 4.4: fire at chunk 0 and every ``stride`` chunks thereafter.
    """

    stride: int = DEFAULT_PLANNER_STRIDE

    def __post_init__(self) -> None:
        if not isinstance(self.stride, int) or isinstance(self.stride, bool):
            raise PlannerScheduleError(f"planner stride must be an int, got {type(self.stride).__name__}")
        if self.stride < 1:
            raise PlannerScheduleError(
                f"planner stride must be >= 1, got {self.stride}. K=1 means regenerate every chunk "
                "(design section 6.4 arm A); there is no 'never plan' setting."
            )

    def is_planner_tick(self, chunk_index: int) -> bool:
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            raise PlannerScheduleError(f"chunk index must be an int, got {type(chunk_index).__name__}")
        if chunk_index < 0:
            raise PlannerScheduleError(f"chunk index must be >= 0, got {chunk_index}")
        return chunk_index % self.stride == 0

    def planner_ticks_within(self, num_chunks: int) -> list[int]:
        """The chunk indices that are Planner ticks, for reporting and tests."""
        return [i for i in range(num_chunks) if self.is_planner_tick(i)]


@dataclasses.dataclass(frozen=True)
class TickConditioning:
    """The memory content a single tick's prefix is allowed to carry.

    Enforces the design section 2.3 asymmetry at construction time:

    * Planner tick: carries Previous Memory, and *generates* Current Memory. It
      must not be handed a Current Memory to condition on.
    * Fast Action tick: carries the held Current Memory only, never Previous
      Memory.

    Making this a validated value object rather than two loose arguments is the
    point: the invariant then cannot be violated by a caller that simply forgets
    which path it is on.
    """

    chunk_index: int
    is_planner_tick: bool
    previous_memory_tokens: Any | None = None
    current_memory_tokens: Any | None = None

    def __post_init__(self) -> None:
        if self.is_planner_tick and self.current_memory_tokens is not None:
            raise MemoryConditioningError(
                f"chunk {self.chunk_index} is a Planner tick but was given current_memory_tokens. "
                "A Planner tick generates Current Memory autoregressively; it conditions on Previous "
                "Memory only (design section 2.3)."
            )
        if not self.is_planner_tick and self.previous_memory_tokens is not None:
            raise MemoryConditioningError(
                f"chunk {self.chunk_index} is a Fast Action tick but was given previous_memory_tokens. "
                "A Fast Action tick carries the held Current Memory only -- the Action Expert must never "
                "see the old and the new plan at once (design section 2.3)."
            )

    @property
    def conditioning_tokens(self) -> Any | None:
        """The single memory token sequence this tick puts in the prefix."""
        return self.previous_memory_tokens if self.is_planner_tick else self.current_memory_tokens


@dataclasses.dataclass
class HeldMemoryRollout:
    """Drives the HeldMemory-K schedule over action chunks.

    Owns the Planner stride and the held memory, and nothing else.  The caller
    supplies observations and runs the model; this object says which path each
    chunk takes and refuses the combinations the design forbids.

    Typical use per chunk::

        tick = rollout.begin_chunk(chunk_index)
        if tick.is_planner_tick:
            new_tokens = model.generate_memory(obs, tick.previous_memory_tokens)
            rollout.commit_planner_output(new_tokens)      # empty => keep previous
            tick = rollout.action_tick(chunk_index)        # now carries held memory
        prefix_ctx = model.encode_prefix(obs, tick.conditioning_tokens)
        rollout.check_prefix_is_fresh(chunk_index, prefix_ctx, obs_fingerprint)
    """

    stride: int = DEFAULT_PLANNER_STRIDE
    cache: MemoryTokenCache = dataclasses.field(default_factory=MemoryTokenCache)
    #: Number of Planner generations actually committed. Design section 6.4
    #: reports planner calls per episode, so this is a first-class counter.
    planner_calls: int = 0
    #: Planner ticks that produced nothing usable and fell back to held memory.
    planner_fallbacks: int = 0
    _schedule: PlannerSchedule = dataclasses.field(init=False, repr=False)
    _seen_chunks: set[int] = dataclasses.field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._schedule = PlannerSchedule(stride=self.stride)

    @classmethod
    def from_model_config(cls, model_config, **kwargs) -> "HeldMemoryRollout":
        """Build a rollout whose stride comes from the model config.

        Until now ``planner_stride`` was a config field with no reader: the whole
        worktree contained four mentions of it -- the definition, a telemetry
        label in ``stats()``, tests, and docs -- so setting it changed nothing and
        the K=1/2/5/10 sweep was not launchable. ``stride`` defaulting to
        ``DEFAULT_PLANNER_STRIDE`` is a *separate* constant from the config field,
        which is exactly how the two could silently disagree.

        This constructor makes the config the single source of truth and refuses
        to fall back: a missing field raises rather than quietly using 5, because
        "the sweep ran but every arm used the same K" is indistinguishable from a
        successful sweep in the output.
        """
        stride = getattr(model_config, "planner_stride", None)
        if stride is None:
            raise PlannerScheduleError(
                f"{type(model_config).__name__} has no planner_stride field. Refusing to "
                f"fall back to {DEFAULT_PLANNER_STRIDE}: every arm of the K sweep would "
                "then use the same stride while appearing to be configured."
            )
        return cls(stride=stride, **kwargs)

    @property
    def schedule(self) -> PlannerSchedule:
        return self._schedule

    def is_planner_tick(self, chunk_index: int) -> bool:
        return self._schedule.is_planner_tick(chunk_index)

    # -- the two paths -------------------------------------------------------

    def begin_chunk(self, chunk_index: int) -> TickConditioning:
        """Start a chunk and return what it may condition on.

        On a Planner tick this returns Previous Memory (``None`` on the very
        first tick, which has no history).  On a Fast Action tick it returns the
        held Current Memory.
        """
        planner = self._schedule.is_planner_tick(chunk_index)
        self._seen_chunks.add(chunk_index)
        held = self.cache.get_or_none()
        held_tokens = held[0] if held is not None else None
        if planner:
            return TickConditioning(
                chunk_index=chunk_index,
                is_planner_tick=True,
                previous_memory_tokens=held_tokens,
            )
        if held_tokens is None:
            raise MemoryConditioningError(
                f"chunk {chunk_index} is a Fast Action tick but no memory is held "
                f"(cache reason: {self.cache.invalidated_reason}). Chunk 0 is always a Planner tick, so "
                "reaching here means the schedule was bypassed or the cache was invalidated mid-episode "
                "without a following Planner tick."
            )
        return TickConditioning(
            chunk_index=chunk_index,
            is_planner_tick=False,
            current_memory_tokens=held_tokens,
        )

    def action_tick(self, chunk_index: int) -> TickConditioning:
        """The Fast Action conditioning for ``chunk_index``, after any Planner update.

        A Planner tick still has to produce an action chunk, and it does so on the
        Current Memory it just generated -- not on the Previous Memory it planned
        from.  This is the conditioning used for the action forward pass on every
        chunk, Planner or not.
        """
        held = self.cache.get_or_none()
        if held is None:
            raise MemoryConditioningError(
                f"chunk {chunk_index}: no memory is held, so no action tick is possible "
                f"(cache reason: {self.cache.invalidated_reason})."
            )
        return TickConditioning(
            chunk_index=chunk_index,
            is_planner_tick=False,
            current_memory_tokens=held[0],
        )

    # -- commit and degradation ---------------------------------------------

    def commit_planner_output(self, tokens: Any, mask: Any = None, *, text: str | None = None) -> bool:
        """Commit a Planner generation. Returns True if the held memory changed.

        Design section 2.5: the minimal update is an overwrite.  Design section 7:
        when the generation is empty, keep the previous tokens.  No structured
        parsing, no validation of the generated text.

        Returns False when the generation was empty and the previous memory was
        kept, so the caller can log it rather than having to infer it.
        """
        if _is_empty(tokens):
            if not self.cache.is_populated:
                raise MemoryConditioningError(
                    "Planner produced an empty generation and there is no previous memory to fall back on. "
                    "This is the first Planner tick, so there is nothing to hold over."
                )
            self.planner_fallbacks += 1
            return False
        self.cache.store(tokens, mask, text=text)
        self.planner_calls += 1
        return True

    def skip_planner_update(self, reason: str) -> None:
        """Deliberately keep the previous memory for this Planner tick.

        Design section 7's timeout degradation: when a Planner call times out,
        execute on the previous memory.  A reason is required so that a rollout
        log distinguishes a timeout from a bug.
        """
        if not reason:
            raise MemoryConditioningError("skip_planner_update() requires a non-empty reason")
        if not self.cache.is_populated:
            raise MemoryConditioningError(
                f"cannot skip the Planner update ({reason}): no previous memory is held to fall back on."
            )
        self.planner_fallbacks += 1

    # -- the invariant that must not regress --------------------------------

    def check_prefix_is_fresh(self, chunk_index: int, prefix_ctx: dict, obs_fingerprint: tuple | None) -> None:
        """Assert this chunk's prefix KV was built from this chunk's observation.

        Design section 4.5 / section 7 "动作忽略最新图像": reusing a prefix KV
        across chunks makes the Action Expert act on an old image and, for pi05,
        an old discretised state.  Raises ``StalePrefixKVError`` if so.
        """
        assert_prefix_fresh(prefix_ctx, obs_fingerprint, where=f"HeldMemory-K chunk {chunk_index}")

    def end_episode(self, reason: str = "episode boundary") -> None:
        """Drop held memory at an episode boundary.

        Memory carried across episodes is stale by construction and nothing
        downstream would report it.
        """
        self.cache.invalidate(reason)
        self._seen_chunks.clear()

    def stats(self) -> dict[str, int]:
        """Counters for design section 6.4 reporting."""
        return {
            "planner_stride": self.stride,
            "planner_calls": self.planner_calls,
            "planner_fallbacks": self.planner_fallbacks,
            "memory_generation": self.cache.generation,
            "chunks_seen": len(self._seen_chunks),
        }


def _is_empty(tokens: Any) -> bool:
    """Whether a Planner generation counts as empty (design section 7 degradation).

    Anything whose length cannot be measured is reported as NOT empty on purpose,
    so it falls through to ``MemoryTokenCache``'s type guard.  That guard is what
    explains *why* a payload is refused (e.g. "the memory cache holds token ids
    only"); raising a bare TypeError here instead would replace a diagnosis with
    a stack trace.
    """
    if tokens is None:
        return True
    if hasattr(tokens, "numel"):  # torch
        return int(tokens.numel()) == 0
    if isinstance(tokens, np.ndarray):
        return int(tokens.size) == 0
    try:
        return len(tokens) == 0
    except TypeError:
        return False
