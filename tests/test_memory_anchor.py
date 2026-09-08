"""Tests for Planner anchor placement and the chunk-consumption guards.

Every guard here protects a value that is currently correct only because each
entry point happens to set it. The failures they catch all look like "the model
is a bit worse at using its memory", not like errors.
"""

from __future__ import annotations

import pytest

from openpi.training.memory_anchor import (
    DEFAULT_FRAMES_PER_CHUNK,
    REQUIRED_CONTROL_MODE,
    AnchorSchedule,
    AnchorScheduleError,
    MixedStrideSelector,
    check_chunk_consumption_assumptions,
    mixed_stride_spec,
)


# --------------------------------------------------------------------------
# measured constants
# --------------------------------------------------------------------------

def test_frames_per_chunk_default_is_the_measured_32():
    assert DEFAULT_FRAMES_PER_CHUNK == 32
    assert REQUIRED_CONTROL_MODE == "receeding_horizon"


def test_the_quantity_is_not_named_action_horizon():
    """`action_horizon` means three different things on this path.

    Model chunk length 32, wrapper ensemble depth 5, wrapper max_len 32 -- reading
    the wrong one yields 5. So our own quantity must not carry that bare name.

    Checked structurally, via dataclass fields and signature parameters, rather
    than by searching the source text. A substring check here fails on
    `model_action_horizon`, which is a deliberately qualified name referring to
    someone else's quantity -- and two earlier text-matching assertions in this
    change set already misfired the same way, one of them matching its own
    explanatory comment.
    """
    import dataclasses
    import inspect

    import openpi.training.memory_anchor as m

    field_names = {f.name for f in dataclasses.fields(AnchorSchedule)}
    assert "action_horizon" not in field_names, f"AnchorSchedule fields: {field_names}"
    assert "frames_per_chunk" in field_names

    params = set(inspect.signature(m.check_chunk_consumption_assumptions).parameters)
    assert "action_horizon" not in params, (
        f"a bare action_horizon parameter is ambiguous; params were {params}"
    )
    assert "frames_per_chunk" in params
    # the qualified names are fine and expected: they name whose quantity it is
    assert "model_action_horizon" in params
    assert "wrapper_max_len" in params


# --------------------------------------------------------------------------
# guard 1: control_mode
# --------------------------------------------------------------------------

def _ok(**over):
    kw = dict(control_mode=REQUIRED_CONTROL_MODE, wrapper_max_len=32,
              model_action_horizon=32, frames_per_chunk=32)
    kw.update(over)
    return kw


def test_missing_control_mode_raises():
    with pytest.raises(AnchorScheduleError, match="control_mode was not passed"):
        check_chunk_consumption_assumptions(**_ok(control_mode=None))


def test_temporal_ensemble_is_rejected_because_h_becomes_one_and_blended():
    with pytest.raises(AnchorScheduleError, match="consumes 1 action"):
        check_chunk_consumption_assumptions(**_ok(control_mode="temporal_ensemble"))


def test_receeding_temporal_is_rejected_because_h_is_five():
    with pytest.raises(AnchorScheduleError, match="consumes 5 action"):
        check_chunk_consumption_assumptions(**_ok(control_mode="receeding_temporal"))


def test_the_measured_configuration_passes():
    """Positive control: guards that only ever reject carry no information."""
    check_chunk_consumption_assumptions(**_ok())


# --------------------------------------------------------------------------
# guard 2: max_len vs the model's chunk length
# --------------------------------------------------------------------------

def test_model_emitting_50_while_max_len_stays_32_raises():
    """The class default action_horizon is 50; max_len is set to 32 by hand.

    The serving queue is filled with target_joint_positions[:max_len], so this
    mismatch silently discards 18 actions per chunk.
    """
    with pytest.raises(AnchorScheduleError, match="silently discards 18"):
        check_chunk_consumption_assumptions(**_ok(model_action_horizon=50))


def test_max_len_larger_than_the_model_output_also_raises():
    with pytest.raises(AnchorScheduleError, match="silently discards"):
        check_chunk_consumption_assumptions(**_ok(wrapper_max_len=40, frames_per_chunk=40))


def test_frames_per_chunk_disagreeing_with_serving_raises():
    with pytest.raises(AnchorScheduleError, match="anchors would be placed"):
        check_chunk_consumption_assumptions(**_ok(frames_per_chunk=16))


# --------------------------------------------------------------------------
# schedule construction guards
# --------------------------------------------------------------------------

def test_bool_stride_is_rejected():
    """True == 1 would silently select the K=1 baseline arm."""
    with pytest.raises(AnchorScheduleError, match="got bool"):
        AnchorSchedule(stride=True)


def test_non_positive_stride_is_rejected():
    for bad in (0, -1, -5):
        with pytest.raises(AnchorScheduleError, match="must be >= 1"):
            AnchorSchedule(stride=bad)


def test_negative_frame_index_is_rejected_rather_than_wrapping():
    """-1 % 5 == 4, so a negative index yields a plausible wrong schedule."""
    s = AnchorSchedule(stride=5)
    with pytest.raises(AnchorScheduleError, match="must be >= 0"):
        s.chunk_index(-1)
    with pytest.raises(AnchorScheduleError, match="got bool"):
        s.chunk_index(True)


def test_frame_before_origin_is_rejected():
    s = AnchorSchedule(stride=5, origin=100)
    with pytest.raises(AnchorScheduleError, match="before origin"):
        s.chunk_index(50)
    assert s.chunk_index(100) == 0


# --------------------------------------------------------------------------
# anchor placement
# --------------------------------------------------------------------------

def test_periods_are_measured_in_frames_not_chunks():
    s = AnchorSchedule(stride=5, frames_per_chunk=32)
    assert s.frames_per_planner_period == 160


def test_k_equals_one_degrades_exactly_to_per_chunk_regeneration():
    """Design §6.4 arm A. The baseline must be the implementation's limit case,
    which is the strongest self-check available for the scheduler."""
    s = AnchorSchedule(stride=1, frames_per_chunk=32)
    for c in range(12):
        frame = c * 32
        assert s.is_periodic_anchor(frame), f"chunk {c} must be an anchor when K=1"
        assert s.anchor_frame_for(frame) == frame
    # every chunk boundary is an anchor, i.e. memory is regenerated every chunk
    assert [s.anchor_frame_for(c * 32) for c in range(6)] == [0, 32, 64, 96, 128, 160]


def test_non_anchor_frames_use_the_most_recent_anchor_memory():
    """§4.1: a non-anchor sample must carry stale memory, matching deployment."""
    s = AnchorSchedule(stride=5, frames_per_chunk=32)   # anchors every 160 frames
    assert s.anchor_frame_for(0) == 0
    assert s.anchor_frame_for(31) == 0        # same chunk
    assert s.anchor_frame_for(32) == 0        # chunk 1, still holds memory_0
    assert s.anchor_frame_for(159) == 0       # last frame before the next anchor
    assert s.anchor_frame_for(160) == 160     # chunk 5 -> new anchor
    assert s.anchor_frame_for(161) == 160


def test_stale_depth_never_exceeds_k_minus_one_chunks():
    s = AnchorSchedule(stride=5, frames_per_chunk=32)
    for frame in range(0, 1000):
        anchor = s.anchor_frame_for(frame)
        staleness_chunks = s.chunk_index(frame) - s.chunk_index(anchor)
        assert 0 <= staleness_chunks <= 4, (frame, anchor, staleness_chunks)


def test_ground_truth_transition_refreshes_memory_early():
    s = AnchorSchedule(stride=5, frames_per_chunk=32)
    # a transition at frame 100 sits between periodic anchors 0 and 160
    assert s.anchor_frame_for(120, transition_starts=[100]) == 100
    assert s.is_anchor(100, transition_starts=[100])
    # a transition before the current periodic anchor must not win
    assert s.anchor_frame_for(200, transition_starts=[100]) == 160
    # a transition after the frame must not win either
    assert s.anchor_frame_for(120, transition_starts=[130]) == 0


def test_origin_shifts_the_whole_schedule():
    """3,946 of 10,000 episodes do not start at frame 0."""
    s = AnchorSchedule(stride=5, frames_per_chunk=32, origin=100)
    assert s.chunk_index(100) == 0
    assert s.chunk_index(131) == 0
    assert s.chunk_index(132) == 1
    assert s.anchor_frame_for(100) == 100
    assert s.anchor_frame_for(259) == 100     # 100 + 160 - 1
    assert s.anchor_frame_for(260) == 260


def test_mixed_stride_is_stable_per_episode_local_action_chunk():
    selector = MixedStrideSelector(((1, 0.4), (2, 0.3), (5, 0.2), (10, 0.1)), seed=42)
    first = selector.select(episode_index=10, chunk_index=7)
    assert first in {1, 2, 5, 10}
    assert selector.select(episode_index=10, chunk_index=7) == first
    # Same action chunk, different frames: one fixed K and one fixed anchor.
    d0 = selector.decision_for(episode_index=10, frame_idx=224, origin=0, frames_per_chunk=32)
    d1 = selector.decision_for(episode_index=10, frame_idx=255, origin=0, frames_per_chunk=32)
    assert (d0.selected_stride, d0.anchor_frame, d0.chunk_lag) == (
        d1.selected_stride,
        d1.anchor_frame,
        d1.chunk_lag,
    )
    assert (d0.frame_lag, d1.frame_lag) == (32, 63)


def test_mixed_stride_golden_vectors_cover_k5_and_k10():
    selector = MixedStrideSelector(((1, 0.4), (2, 0.3), (5, 0.2), (10, 0.1)), seed=42)
    assert selector.select(episode_index=0, chunk_index=12) == 5
    assert selector.select(episode_index=0, chunk_index=3) == 10
    assert selector.decision_for(
        episode_index=0, frame_idx=384, origin=0, frames_per_chunk=32
    ).anchor_frame == 320
    assert selector.decision_for(
        episode_index=0, frame_idx=96, origin=0, frames_per_chunk=32
    ).anchor_frame == 0


def test_mixed_stride_spec_is_complete_and_canonical():
    assert mixed_stride_spec(((1, 0.4), (10, 0.6)), 42, 32) == {
        "schema_version": 1,
        "algorithm": "blake2b_u64_weighted_cdf",
        "digest": "blake2b-64-big-endian",
        "key_fields": ["seed", "episode_index", "episode_local_chunk_index"],
        "weights": [[1, 0.4], [10, 0.6]],
        "seed": 42,
        "frames_per_chunk": 32,
    }


def test_mixed_stride_uses_episode_local_origin_and_separates_initial():
    selector = MixedStrideSelector(((1, 1.0),), seed=9)
    initial = selector.decision_for(episode_index=4, frame_idx=100, origin=100, frames_per_chunk=32)
    assert initial.anchor_kind == "initial"
    assert initial.selected_stride == 0
    assert initial.chunk_lag == 0 and initial.frame_lag == 0
    periodic = selector.decision_for(episode_index=4, frame_idx=132, origin=100, frames_per_chunk=32)
    assert periodic.anchor_kind == "periodic"
    assert periodic.target_chunk_index == 1
    assert periodic.anchor_frame == 100
    assert periodic.chunk_lag == 1 and periodic.frame_lag == 32


def test_manifest_description_records_the_dependency_not_just_the_number():
    d = AnchorSchedule(stride=5).describe()
    assert d["planner_stride_chunks"] == 5
    assert d["frames_per_chunk"] == 32
    assert d["frames_per_planner_period"] == 160
    # 32 is only true while every entry point sets control_mode; a manifest that
    # records the bare number would read as a property of the system.
    assert REQUIRED_CONTROL_MODE in d["frames_per_chunk_note"]
    assert "control_mode" in d["frames_per_chunk_note"]
