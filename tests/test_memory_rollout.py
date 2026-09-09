"""Tests for the HeldMemory-K rollout schedule (MoMA-VLA P1, item 3).

Each test states what would make it fail.

Two of these are the load-bearing invariants of the design:

* ``test_fast_action_tick_cannot_carry_previous_memory`` -- design section 2.3:
  the Action Expert must never see the old and the new plan at once.
* ``test_prefix_kv_reused_across_chunks_raises`` -- design section 4.5: the
  prefix encodes the observation, and for pi05 that includes the discretised
  state, so it must be recomputed every chunk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.models.memory_cache import (
    MemoryCacheError,
    MemoryTokenCache,
    StalePrefixKVError,
    observation_fingerprint,
)
from openpi.policies.memory_rollout import (
    DEFAULT_PLANNER_STRIDE,
    HeldMemoryRollout,
    MemoryConditioningError,
    PlannerSchedule,
    PlannerScheduleError,
    TickConditioning,
)

DOC_ARMS = [1, 2, 5, 10]  # design section 6.4 arms A-D


# ---------------------------------------------------------------------------
# K as a configurable stride
# ---------------------------------------------------------------------------


def test_default_stride_is_five():
    """FAILS IF: the default K stops being 5.

    Design section 6.4 arm C (K=5) is the main experiment, and the user asked for
    5 as the default with config override.
    """
    assert DEFAULT_PLANNER_STRIDE == 5
    assert PlannerSchedule().stride == 5
    assert HeldMemoryRollout().stride == 5


def test_stride_comes_from_model_config_not_a_constant():
    """FAILS IF: K stops being a model-config field.

    The user was explicit that K must be a config field so the K=1/2/5/10 sweep
    is launchable without code edits. A constant buried in the rollout would make
    each arm a code change.
    """
    import dataclasses

    from openpi.models.pi05_subtask_config import Pi05SubtaskConfig

    names = {f.name for f in dataclasses.fields(Pi05SubtaskConfig)}
    assert "planner_stride" in names, "planner_stride is not a Pi05SubtaskConfig field"
    assert Pi05SubtaskConfig().planner_stride == 5
    # Overridable without touching code.
    assert dataclasses.replace(Pi05SubtaskConfig(), planner_stride=10).planner_stride == 10


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_stride_below_one_is_rejected(bad):
    """FAILS IF: K=0 or negative is accepted.

    K=0 would make `chunk_index % stride` raise deep inside a rollout, and there
    is no 'never plan' arm in the design.
    """
    with pytest.raises(PlannerScheduleError, match="must be >= 1"):
        PlannerSchedule(stride=bad)


def test_stride_must_be_an_int_not_a_bool():
    """FAILS IF: a bool sneaks through as a stride.

    `True == 1` in Python, so `PlannerSchedule(stride=True)` would silently
    behave as K=1 -- an entire experimental arm selected by a type error.
    """
    with pytest.raises(PlannerScheduleError, match="must be an int"):
        PlannerSchedule(stride=True)


# ---------------------------------------------------------------------------
# The Planner fires on exactly {0, K, 2K, ...}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", DOC_ARMS)
def test_planner_fires_on_exactly_the_multiples_of_k(k):
    """REQUIRED TEST. FAILS IF: the Planner fires on any chunk that is not a
    multiple of K, or skips one that is.

    Checked exhaustively over 40 chunks against an independently computed
    expected set, so an off-by-one (e.g. firing at K+1, or missing chunk 0) is
    caught rather than averaged away.
    """
    schedule = PlannerSchedule(stride=k)
    got = schedule.planner_ticks_within(40)
    expected = list(range(0, 40, k))
    assert got == expected, f"K={k}: planner ticks {got} != {expected}"
    # And the complement really is not a planner tick.
    for i in range(40):
        assert schedule.is_planner_tick(i) == (i in expected)


def test_chunk_zero_is_always_a_planner_tick():
    """FAILS IF: chunk 0 is not a Planner tick for some K.

    Design section 4.4 fires at chunk 0 explicitly. If it did not, the first Fast
    Action tick would have no memory to hold.
    """
    for k in range(1, 13):
        assert PlannerSchedule(stride=k).is_planner_tick(0), f"K={k} did not plan at chunk 0"


def test_negative_chunk_index_is_rejected():
    """FAILS IF: a negative chunk index is accepted. Python's `%` would make
    -1 % 5 == 4, quietly producing a wrong schedule instead of an error."""
    with pytest.raises(PlannerScheduleError, match="must be >= 0"):
        PlannerSchedule(stride=5).is_planner_tick(-1)


@pytest.mark.parametrize("k", DOC_ARMS)
def test_planner_call_count_matches_the_schedule(k):
    """FAILS IF: the number of committed Planner generations drifts from the
    schedule. Design section 6.4 reports planner calls per episode, so this
    counter is a reported metric, not just bookkeeping."""
    rollout = HeldMemoryRollout(stride=k)
    n_chunks = 20
    for i in range(n_chunks):
        tick = rollout.begin_chunk(i)
        if tick.is_planner_tick:
            rollout.commit_planner_output([1, 2, 3, i])
    assert rollout.planner_calls == len(range(0, n_chunks, k))
    assert rollout.stats()["planner_stride"] == k


# ---------------------------------------------------------------------------
# K=1 must degenerate exactly to per-chunk regeneration (arm A)
# ---------------------------------------------------------------------------


def test_k1_degenerates_to_per_chunk_regeneration():
    """REQUIRED TEST. FAILS IF: K=1 is not exactly the per-chunk baseline.

    Design section 6.4 arm A is the current baseline, and every other arm is
    measured against it. If K=1 ever skipped a chunk, or held memory for two
    chunks, the whole comparison would be against the wrong reference.
    """
    n = 12
    rollout = HeldMemoryRollout(stride=1)
    planned_on = []
    for i in range(n):
        tick = rollout.begin_chunk(i)
        if tick.is_planner_tick:
            planned_on.append(i)
            rollout.commit_planner_output([100 + i])
        # Whatever the path, the action tick must use the memory from THIS chunk.
        action = rollout.action_tick(i)
        assert list(action.current_memory_tokens) == [100 + i], f"chunk {i} acted on stale memory under K=1"
    assert planned_on == list(range(n)), "K=1 did not regenerate on every chunk"
    assert rollout.planner_calls == n
    assert rollout.planner_fallbacks == 0


@pytest.mark.parametrize("k", [2, 5, 10])
def test_memory_is_actually_held_between_planner_ticks(k):
    """FAILS IF: memory is regenerated on non-Planner chunks (which would make
    every arm behave like K=1), or held across a Planner tick that should have
    replaced it."""
    rollout = HeldMemoryRollout(stride=k)
    seen = []
    for i in range(3 * k):
        tick = rollout.begin_chunk(i)
        if tick.is_planner_tick:
            rollout.commit_planner_output([i])
        seen.append(list(rollout.action_tick(i).current_memory_tokens))
    # Every chunk acts on the memory generated at its most recent planner tick.
    expected = [[(i // k) * k] for i in range(3 * k)]
    assert seen == expected


# ---------------------------------------------------------------------------
# Design section 2.3: the two paths, and what each may carry
# ---------------------------------------------------------------------------


def test_fast_action_tick_cannot_carry_previous_memory():
    """REQUIRED TEST -- design section 2.3 as an executable invariant.

    FAILS IF: a Fast Action tick can be constructed carrying Previous Memory as
    well as Current Memory. The doc's stated reason is that the Action Expert
    must never see the old and the new plan at the same time; if this stops
    raising, that can happen silently and would look like a mildly confused
    policy rather than a wiring bug.
    """
    with pytest.raises(MemoryConditioningError, match="must never\n?\\s*see the old and the new plan|old and the new plan"):
        TickConditioning(
            chunk_index=3,
            is_planner_tick=False,
            previous_memory_tokens=[1, 2],
            current_memory_tokens=[3, 4],
        )
    # Previous Memory alone on a fast tick is equally forbidden.
    with pytest.raises(MemoryConditioningError):
        TickConditioning(chunk_index=3, is_planner_tick=False, previous_memory_tokens=[1, 2])


def test_planner_tick_cannot_be_handed_current_memory():
    """FAILS IF: a Planner tick can be given Current Memory to condition on.

    The Planner *generates* Current Memory autoregressively; feeding it in would
    be teacher-forcing the thing under test at rollout time.
    """
    with pytest.raises(MemoryConditioningError, match="generates Current Memory"):
        TickConditioning(chunk_index=0, is_planner_tick=True, current_memory_tokens=[1, 2])


def test_begin_chunk_gives_previous_memory_on_planner_and_current_on_fast():
    """FAILS IF: the two paths are swapped, or a path carries both.

    This checks the *rollout* wiring rather than the value object: begin_chunk is
    what a caller actually uses, so the invariant has to hold there too.
    """
    rollout = HeldMemoryRollout(stride=3)

    first = rollout.begin_chunk(0)
    assert first.is_planner_tick
    assert first.previous_memory_tokens is None  # no history yet
    assert first.current_memory_tokens is None
    rollout.commit_planner_output([10, 11])

    fast = rollout.begin_chunk(1)
    assert not fast.is_planner_tick
    assert list(fast.current_memory_tokens) == [10, 11]
    assert fast.previous_memory_tokens is None, "Fast Action tick leaked Previous Memory"

    second_planner = rollout.begin_chunk(3)
    assert second_planner.is_planner_tick
    assert list(second_planner.previous_memory_tokens) == [10, 11]
    assert second_planner.current_memory_tokens is None


def test_conditioning_tokens_exposes_exactly_one_sequence():
    """FAILS IF: a tick ever offers two memory sequences to the prefix builder.

    The prefix gets one memory span; if this returned both, the caller would have
    to choose, and that choice is exactly what section 2.3 removes.
    """
    planner = TickConditioning(chunk_index=0, is_planner_tick=True, previous_memory_tokens=[1])
    fast = TickConditioning(chunk_index=1, is_planner_tick=False, current_memory_tokens=[2])
    assert planner.conditioning_tokens == [1]
    assert fast.conditioning_tokens == [2]


def test_fast_tick_without_held_memory_raises():
    """FAILS IF: a Fast Action tick with a cold cache silently proceeds with no
    memory, i.e. an unconditioned action chunk that reports no error."""
    rollout = HeldMemoryRollout(stride=5)
    with pytest.raises(MemoryConditioningError, match="no memory is held"):
        rollout.begin_chunk(1)  # chunk 0 was skipped, so nothing was ever planned


# ---------------------------------------------------------------------------
# Design section 4.5: the prefix must be recomputed every chunk
# ---------------------------------------------------------------------------


def test_prefix_kv_reused_across_chunks_raises():
    """REQUIRED TEST -- extends the staleness guard to the rollout.

    Encodes a prefix at chunk 0 and then tries to use it at chunk 1, which is
    what 'cache the full KV across chunks' looks like in a rollout loop.

    FAILS IF: the guard is removed or the rollout stops checking. Then the Action
    Expert acts on chunk 0's image and (for pi05) chunk 0's discretised state
    while believing it is current.
    """
    rollout = HeldMemoryRollout(stride=5)
    rollout.begin_chunk(0)
    rollout.commit_planner_output([1, 2, 3])

    torch.manual_seed(0)
    obs0_images, obs0_lang = [torch.randn(1, 3, 8, 8)], torch.tensor([[1, 2, 3]])
    obs1_images, obs1_lang = [torch.randn(1, 3, 8, 8)], torch.tensor([[1, 2, 4]])
    prefix_from_chunk0 = {"obs_fingerprint": observation_fingerprint(obs0_images, obs0_lang)}

    # Same chunk, same observation: fine.
    rollout.check_prefix_is_fresh(0, prefix_from_chunk0, observation_fingerprint(obs0_images, obs0_lang))

    # Next chunk, new observation, reused prefix: must raise.
    with pytest.raises(StalePrefixKVError, match="different observation"):
        rollout.check_prefix_is_fresh(1, prefix_from_chunk0, observation_fingerprint(obs1_images, obs1_lang))


def test_staleness_guard_names_the_chunk():
    """FAILS IF: the error does not say which chunk failed.

    In a 200-chunk rollout an unlocated error is much harder to act on, and the
    guard exists to be acted on.
    """
    rollout = HeldMemoryRollout()
    with pytest.raises(StalePrefixKVError, match="chunk 7"):
        rollout.check_prefix_is_fresh(7, {"obs_fingerprint": ("a",)}, ("b",))


def test_rollout_never_stores_a_kv_object():
    """FAILS IF: the rollout's cache accepts a KV payload.

    Only memory token IDs may persist across chunks; a KV would carry the
    observation with it, which is the thing section 4.5 forbids.
    """

    class DynamicCache:  # name matches the HF class
        pass

    rollout = HeldMemoryRollout()
    with pytest.raises(MemoryCacheError, match="token ids only"):
        rollout.commit_planner_output(DynamicCache())


# ---------------------------------------------------------------------------
# Documented degradation (design section 7)
# ---------------------------------------------------------------------------


def test_empty_generation_keeps_the_previous_memory():
    """REQUIRED TEST. FAILS IF: an empty Planner generation clears or overwrites
    the held memory.

    Design section 7: `if len(new) > 0: held = new`. Clearing it would leave the
    next Fast Action tick unconditioned, and the failure would show up as
    degraded actions rather than as an empty generation.
    """
    rollout = HeldMemoryRollout(stride=2)
    rollout.begin_chunk(0)
    rollout.commit_planner_output([7, 8, 9], text="Memory: first")

    rollout.begin_chunk(2)
    for empty in ([], np.asarray([], dtype=np.int32), torch.empty(0, dtype=torch.long), None):
        changed = rollout.commit_planner_output(empty)
        assert changed is False, f"{type(empty).__name__} was treated as a real generation"
        tokens, _ = rollout.cache.get()
        assert list(tokens) == [7, 8, 9], "empty generation did not keep the previous memory"
    assert rollout.cache.text == "Memory: first"
    assert rollout.planner_calls == 1
    assert rollout.planner_fallbacks == 4


def test_commit_reports_whether_the_memory_changed():
    """FAILS IF: the caller cannot tell a real update from a fallback.

    Returning None/void would force the rollout log to infer it, and a silent
    fallback rate is exactly what hides a broken Planner.
    """
    rollout = HeldMemoryRollout(stride=1)
    rollout.begin_chunk(0)
    assert rollout.commit_planner_output([1]) is True
    assert rollout.commit_planner_output([]) is False


def test_empty_generation_on_the_very_first_tick_raises():
    """FAILS IF: an empty first generation is silently tolerated.

    There is no previous memory to fall back on at chunk 0, so 'keep the previous
    tokens' is not available and pretending otherwise would run the whole episode
    unconditioned.
    """
    rollout = HeldMemoryRollout(stride=5)
    rollout.begin_chunk(0)
    with pytest.raises(MemoryConditioningError, match="nothing to hold over"):
        rollout.commit_planner_output([])


def test_planner_timeout_reuses_previous_memory():
    """FAILS IF: the timeout path cannot be expressed, or does not require a
    reason. Design section 7: on timeout, execute on the previous memory."""
    rollout = HeldMemoryRollout(stride=3)
    rollout.begin_chunk(0)
    rollout.commit_planner_output([5, 6])

    rollout.begin_chunk(3)
    rollout.skip_planner_update("planner call timed out")
    assert list(rollout.action_tick(3).current_memory_tokens) == [5, 6]
    assert rollout.planner_fallbacks == 1
    assert rollout.planner_calls == 1

    with pytest.raises(MemoryConditioningError, match="non-empty reason"):
        rollout.skip_planner_update("")


def test_timeout_on_the_first_tick_raises():
    """FAILS IF: a first-tick timeout silently produces an unconditioned episode."""
    rollout = HeldMemoryRollout(stride=5)
    rollout.begin_chunk(0)
    with pytest.raises(MemoryConditioningError, match="no previous memory"):
        rollout.skip_planner_update("planner call timed out")


def test_no_event_triggered_refresh_exists():
    """FAILS IF: event-triggered refresh appears in this module.

    It is design P2 and explicitly out of scope for item 3. This test is here so
    that adding it is a deliberate act rather than scope drift.
    """
    import openpi.policies.memory_rollout as rollout_mod

    names = dir(rollout_mod)
    for forbidden in ("trigger", "event_trigger", "refresh_on_event", "EventTrigger"):
        assert not any(forbidden in n for n in names), f"{forbidden} leaked into item 3's scope"


# ---------------------------------------------------------------------------
# Episode boundaries and reporting
# ---------------------------------------------------------------------------


def test_end_episode_drops_held_memory():
    """FAILS IF: memory survives an episode boundary. It is stale by
    construction there and nothing downstream would report it."""
    rollout = HeldMemoryRollout(stride=5)
    rollout.begin_chunk(0)
    rollout.commit_planner_output([1, 2])
    rollout.end_episode()
    assert not rollout.cache.is_populated
    assert rollout.cache.invalidated_reason == "episode boundary"
    with pytest.raises(MemoryConditioningError, match="no memory is held"):
        rollout.begin_chunk(1)


def test_stats_report_the_fields_section_64_asks_for():
    """FAILS IF: the counters design section 6.4 reports go missing.

    planner calls per episode is a primary metric of the K sweep; if it is not
    emitted, the sweep cannot be evaluated on cost at all.
    """
    rollout = HeldMemoryRollout(stride=2)
    for i in range(6):
        tick = rollout.begin_chunk(i)
        if tick.is_planner_tick:
            rollout.commit_planner_output([i])
    stats = rollout.stats()
    assert stats == {
        "planner_stride": 2,
        "planner_calls": 3,
        "planner_fallbacks": 0,
        "memory_generation": 3,
        "chunks_seen": 6,
    }


def test_rollout_holds_no_label_or_token_scheme_knowledge():
    """FAILS IF: the rollout module starts spelling label text or assuming token
    counts.

    The plain-text vs reserved-slot decision is still open upstream, and item 3
    must not have to be revisited when it is settled.
    """
    from pathlib import Path

    src = Path(__import__("openpi.policies.memory_rollout", fromlist=["x"]).__file__).read_text()
    for label in ("Memory:", "Primitive:", "Next skill:", "Next primitive:", "<unused"):
        assert label not in src, f"rollout module hardcodes the label {label!r}"


def test_rollout_can_share_an_externally_owned_cache():
    """FAILS IF: the rollout cannot be pointed at the Policy's existing cache.

    Two caches for one held memory would drift, and the Policy already owns one.
    """
    shared = MemoryTokenCache()
    rollout = HeldMemoryRollout(stride=5, cache=shared)
    rollout.begin_chunk(0)
    rollout.commit_planner_output([4, 5, 6])
    assert shared.is_populated
    assert list(shared.get()[0]) == [4, 5, 6]


# --------------------------------------------------------------------------
# planner_stride wiring: config field -> rollout
# --------------------------------------------------------------------------
# Before this, planner_stride had no reader anywhere: the four mentions in the
# worktree were the definition, a telemetry label in stats(), tests, and docs.
# Setting it changed nothing, so the K=1/2/5/10 sweep could not actually run --
# and every arm would have reported a different configured K while using 5.

def test_from_model_config_takes_the_stride_from_the_config():
    import dataclasses

    from openpi.models.pi05_subtask_config import Pi05SubtaskConfig
    from openpi.policies.memory_rollout import HeldMemoryRollout

    cfg = Pi05SubtaskConfig()
    assert HeldMemoryRollout.from_model_config(cfg).stride == cfg.planner_stride

    for k in (1, 2, 5, 10):                       # the design's four arms
        r = HeldMemoryRollout.from_model_config(dataclasses.replace(cfg, planner_stride=k))
        assert r.stride == k, f"arm K={k} did not reach the rollout"
        assert r.stats()["planner_stride"] == k


def test_a_config_without_the_field_raises_rather_than_defaulting_to_five():
    """A silent fallback would make "every arm used K=5" look like a real sweep."""
    from openpi.policies.memory_rollout import (
        DEFAULT_PLANNER_STRIDE,
        HeldMemoryRollout,
        PlannerScheduleError,
    )

    class ConfigWithoutTheField:
        pass

    with pytest.raises(PlannerScheduleError, match="no planner_stride field"):
        HeldMemoryRollout.from_model_config(ConfigWithoutTheField())
    # the independent module constant still exists but must no longer be reachable
    # by accident from a config that lacks the field
    assert DEFAULT_PLANNER_STRIDE == 5


def test_arms_are_distinguishable_from_each_other():
    """Positive control: if from_model_config ignored its argument, every arm
    above would still pass its own assertion while all being identical."""
    import dataclasses

    from openpi.models.pi05_subtask_config import Pi05SubtaskConfig
    from openpi.policies.memory_rollout import HeldMemoryRollout

    cfg = Pi05SubtaskConfig()
    strides = {
        HeldMemoryRollout.from_model_config(dataclasses.replace(cfg, planner_stride=k)).stride
        for k in (1, 2, 5, 10)
    }
    assert strides == {1, 2, 5, 10}, f"arms collapsed to {strides}"
