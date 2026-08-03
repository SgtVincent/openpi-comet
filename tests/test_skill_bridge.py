"""Tests for the skill bridge baseline (Phase 1 — data-only minimal implementation).

Covers:
  * Core logic: normal bridging, single-skill, empty, boundary edges, 3+ skills,
    gap/overlap rejection, min-pre/post filtering, padded tails.
  * Config: default-off behavior, field defaults, DataConfig integration.
  * Integration helper: query-frame-aware path, pad-mask path, fallback behavior.
  * Audit functions: segment integrity, bridge cross-validation, stats, contiguity.
  * Off-mode compatibility: bridge disabled = byte-identical to baseline.

All tests use pure Python fake data — no real dataset, no torch, no HF dependency.
"""

from __future__ import annotations

import dataclasses
import pytest

from openpi.training.skill_bridge_core import (
    compute_bridge_info,
    find_segment_at_frame,
)
from openpi.training.skill_bridge_integration import get_bridge_subtask_text
from openpi.training.skill_bridge_audit import (
    audit_episode_segments,
    audit_bridge_validity,
    audit_chunk_crossing_stats,
    audit_segment_contiguity,
)
from openpi.training.skill_bridge_config import SkillBridgeConfig

# DataConfig integration requires the full training config stack (models,
# transforms, etc.) which is not available in this sparse test environment.
# We test SkillBridgeConfig independently and verify the integration contract.
try:
    from openpi.training.data_config import DataConfig
    _HAS_DATA_CONFIG = True
except ImportError:
    _HAS_DATA_CONFIG = False


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def two_skill_contiguous():
    """Two contiguous skills, each 100 frames: [0..99] skill_a, [100..199] skill_b."""
    return [
        (0, 99, "move to the radio"),
        (100, 199, "pick up the radio"),
    ]


@pytest.fixture
def three_skill_contiguous():
    """Three contiguous skills."""
    return [
        (0, 49, "skill_a"),
        (50, 99, "skill_b"),
        (100, 149, "skill_c"),
    ]


@pytest.fixture
def two_skill_with_gap():
    """Two skills with a 2-frame gap between them."""
    return [
        (0, 99, "skill_a"),
        (102, 199, "skill_b"),  # gap of 2 frames: 100, 101
    ]


@pytest.fixture
def two_skill_with_overlap():
    """Two skills that overlap by 3 frames."""
    return [
        (0, 102, "skill_a"),
        (100, 199, "skill_b"),  # overlap frames 100, 101, 102
    ]


# ===========================================================================
# find_segment_at_frame
# ===========================================================================

class TestFindSegmentAtFrame:
    def test_empty_segments(self):
        assert find_segment_at_frame([], 5) is None

    def test_single_segment_hit(self):
        segs = [(0, 99, "a")]
        assert find_segment_at_frame(segs, 0) == 0
        assert find_segment_at_frame(segs, 50) == 0
        assert find_segment_at_frame(segs, 99) == 0

    def test_single_segment_miss(self):
        segs = [(10, 20, "a")]
        assert find_segment_at_frame(segs, 5) is None
        assert find_segment_at_frame(segs, 25) is None

    def test_multi_segment_middle(self):
        segs = [(0, 49, "a"), (50, 99, "b"), (100, 149, "c")]
        assert find_segment_at_frame(segs, 75) == 1

    def test_boundary_frames(self):
        segs = [(0, 49, "a"), (50, 99, "b")]
        # frame 49 is in segment 0, frame 50 is in segment 1
        assert find_segment_at_frame(segs, 49) == 0
        assert find_segment_at_frame(segs, 50) == 1


# ===========================================================================
# compute_bridge_info — valid bridge cases
# ===========================================================================

class TestValidBridge:
    def test_middle_crossing(self, two_skill_contiguous):
        """Chunk crosses boundary in the middle → valid bridge."""
        result = compute_bridge_info(two_skill_contiguous, 85, 32)
        assert result["bridge_valid"] is True
        assert result["current_phrase"] == "move to the radio"
        assert result["successor_phrase"] == "pick up the radio"
        assert result["boundary_step"] == 15  # 100 - 85 = 15
        assert result["combined_phrase"] == "move to the radio then pick up the radio"
        assert result["crossing_count"] == 1
        assert result["rejection_reason"] is None

    def test_min_boundary_at_threshold(self, two_skill_contiguous):
        """Boundary exactly at min_pre and min_post thresholds → valid."""
        # min_pre=5, boundary at step 5 → exactly at threshold
        result = compute_bridge_info(
            two_skill_contiguous, 95, 10,
            min_pre_boundary_steps=5, min_post_boundary_steps=5,
        )
        assert result["bridge_valid"] is True
        assert result["boundary_step"] == 5  # 100 - 95 = 5

    def test_episode_end_exact(self, two_skill_contiguous):
        """Chunk ends exactly at episode end → still valid (no padding)."""
        result = compute_bridge_info(
            two_skill_contiguous, 85, 115,
            episode_end_frame=199,
        )
        assert result["bridge_valid"] is True
        # chunk covers frames 85..199 = 115 frames, boundary at step 15
        assert result["boundary_step"] == 15


# ===========================================================================
# compute_bridge_info — single skill (no crossing)
# ===========================================================================

class TestSingleSkill:
    def test_entirely_in_first_skill(self, two_skill_contiguous):
        result = compute_bridge_info(two_skill_contiguous, 10, 32)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "single_skill"
        assert result["crossing_count"] == 0
        assert result["current_phrase"] == "move to the radio"
        assert result["combined_phrase"] == "move to the radio"
        assert result["successor_phrase"] == ""
        assert result["boundary_step"] == 0

    def test_entirely_in_second_skill(self, two_skill_contiguous):
        result = compute_bridge_info(two_skill_contiguous, 150, 32)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "single_skill"
        assert result["current_phrase"] == "pick up the radio"


# ===========================================================================
# compute_bridge_info — multiple crossings
# ===========================================================================

class TestMultipleCrossings:
    def test_three_skills(self, three_skill_contiguous):
        """Chunk spans 3 skills → multiple_crossings rejection."""
        result = compute_bridge_info(three_skill_contiguous, 25, 100)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "multiple_crossings"
        assert result["crossing_count"] == 2
        assert result["successor_phrase"] == "skill_b"
        assert result["boundary_step"] == 25  # 50 - 25 = 25


# ===========================================================================
# compute_bridge_info — gap / overlap
# ===========================================================================

class TestGapOverlap:
    def test_gap_rejected(self, two_skill_with_gap):
        result = compute_bridge_info(two_skill_with_gap, 85, 32)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "gap_between_skills"
        assert result["crossing_count"] == 1

    def test_overlap_rejected(self, two_skill_with_overlap):
        result = compute_bridge_info(two_skill_with_overlap, 85, 32)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "overlap_between_skills"
        assert result["crossing_count"] == 1


# ===========================================================================
# compute_bridge_info — min_pre / min_post filtering
# ===========================================================================

class TestMinBoundarySteps:
    def test_insufficient_pre(self, two_skill_contiguous):
        """Boundary at step 2 with min_pre=5 → rejected."""
        result = compute_bridge_info(
            two_skill_contiguous, 98, 32,
            min_pre_boundary_steps=5,
        )
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "insufficient_pre_steps"
        assert result["boundary_step"] == 2  # 100 - 98 = 2

    def test_insufficient_post(self, two_skill_contiguous):
        """Only 5 post-boundary steps with min_post=10 → rejected."""
        result = compute_bridge_info(
            two_skill_contiguous, 90, 15,
            min_post_boundary_steps=10,
        )
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "insufficient_post_steps"
        # chunk: 90..104 = 15 steps, boundary at 100 - 90 = 10
        # post_steps = 15 - 10 = 5 < 10

    def test_zero_min_pre_allowed(self, two_skill_contiguous):
        """With min_pre=0, boundary at step 0 is allowed when anchor is in current skill.

        Note: this requires the anchor to be in the *current* skill and the successor
        to start at anchor + 0, which only happens when the current skill end = anchor - 1
        and the successor starts at anchor. That means the anchor frame is the first
        frame of the successor skill — but then the anchor is IN the successor, not
        the current skill. A boundary at step 0 means the entire chunk is successor
        skill, which is "single_skill" (successor), not a bridge.

        The min_pre_boundary_steps=0 case is relevant when chunk_size is very large
        relative to skill length, allowing the boundary to be exactly at step 0 while
        the anchor is still in the current skill — impossible since step 0 = anchor frame
        which must be in the current skill for it to be "current".

        So we verify: min_pre=0 doesn't break anything, and the valid case has
        boundary_step >= 1 when anchor is in the current skill.
        """
        # With min_pre=0, a chunk that starts at frame 99 (last frame of skill_a)
        # has boundary at step 1 (100 - 99 = 1), which is >= 0 → valid
        result = compute_bridge_info(
            two_skill_contiguous, 99, 32,
            min_pre_boundary_steps=0,
        )
        assert result["bridge_valid"] is True
        assert result["boundary_step"] == 1  # 100 - 99 = 1


# ===========================================================================
# compute_bridge_info — padded tail
# ===========================================================================

class TestPaddedTail:
    def test_padded_tail_rejected(self, two_skill_contiguous):
        """Chunk extends past episode end → padded_tail rejection for bridges."""
        result = compute_bridge_info(
            two_skill_contiguous, 85, 130,  # covers 85..214
            episode_end_frame=199,
        )
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "padded_tail"
        assert result["crossing_count"] == 1

    def test_single_skill_with_padding_not_padded_tail(self, two_skill_contiguous):
        """Single-skill chunk with padding → single_skill (not padded_tail)."""
        result = compute_bridge_info(
            two_skill_contiguous, 180, 32,  # covers 180..211
            episode_end_frame=199,
        )
        # Chunk is entirely within skill_b (100..199), with padding past 199
        # But since there's only one skill overlapped, it's single_skill
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "single_skill"


# ===========================================================================
# compute_bridge_info — edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_segments(self):
        result = compute_bridge_info([], 0, 32)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "no_segments"
        assert result["current_phrase"] == ""
        assert result["combined_phrase"] == ""

    def test_anchor_outside_segment(self, two_skill_contiguous):
        result = compute_bridge_info(two_skill_contiguous, -1, 32)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "anchor_outside_any_segment"

    def test_anchor_outside_past_end(self, two_skill_contiguous):
        result = compute_bridge_info(two_skill_contiguous, 250, 32)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "anchor_outside_any_segment"

    def test_chunk_size_zero(self, two_skill_contiguous):
        result = compute_bridge_info(two_skill_contiguous, 50, 0)
        # chunk_size=0: chunk_end = 50 + 0 - 1 = 49
        # overlaps only first skill (0..99)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "single_skill"

    def test_chunk_size_one(self, two_skill_contiguous):
        result = compute_bridge_info(two_skill_contiguous, 50, 1)
        assert result["bridge_valid"] is False
        assert result["rejection_reason"] == "single_skill"

    def test_anchor_at_skill_start(self, two_skill_contiguous):
        result = compute_bridge_info(two_skill_contiguous, 100, 32)
        # Anchor is at start of skill_b, chunk entirely within skill_b
        assert result["rejection_reason"] == "single_skill"
        assert result["current_phrase"] == "pick up the radio"


# ===========================================================================
# Config tests
# ===========================================================================

class TestSkillBridgeConfig:
    def test_defaults_disabled(self):
        cfg = SkillBridgeConfig()
        assert cfg.enabled is False
        assert cfg.min_pre_boundary_steps == 1
        assert cfg.min_post_boundary_steps == 1

    def test_frozen(self):
        cfg = SkillBridgeConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = True

    @pytest.mark.skipif(not _HAS_DATA_CONFIG, reason="DataConfig not available in sparse test env")
    def test_data_config_default_has_bridge(self):
        cfg = DataConfig()
        assert hasattr(cfg, "skill_bridge")
        assert cfg.skill_bridge.enabled is False
        assert isinstance(cfg.skill_bridge, SkillBridgeConfig)

    @pytest.mark.skipif(not _HAS_DATA_CONFIG, reason="DataConfig not available in sparse test env")
    def test_data_config_with_bridge_enabled(self):
        bridge_cfg = SkillBridgeConfig(enabled=True, min_pre_boundary_steps=2, min_post_boundary_steps=3)
        cfg = DataConfig(skill_bridge=bridge_cfg)
        assert cfg.skill_bridge.enabled is True
        assert cfg.skill_bridge.min_pre_boundary_steps == 2
        assert cfg.skill_bridge.min_post_boundary_steps == 3

    def test_skill_bridge_config_follows_dataclass_contract(self):
        """Verify SkillBridgeConfig matches the expected contract for DataConfig integration."""
        cfg = SkillBridgeConfig()
        # Must have exactly these 3 fields (no extra fields in Phase 1)
        field_names = {f.name for f in dataclasses.fields(cfg)}
        assert field_names == {"enabled", "min_pre_boundary_steps", "min_post_boundary_steps"}
        # Must be frozen (immutable)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = True
        # Default factory pattern works with dataclasses.field(default_factory=...)
        import copy
        cfg2 = copy.deepcopy(cfg)
        assert cfg2 == cfg


# ===========================================================================
# Integration helper — get_bridge_subtask_text
# ===========================================================================

class TestGetBridgeSubtaskText:
    def test_valid_bridge_returns_combined(self, two_skill_contiguous):
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            chunk_size=32,
            fallback_phrase="move to the radio",
        )
        assert text == "move to the radio then pick up the radio"

    def test_single_skill_returns_fallback(self, two_skill_contiguous):
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=10,
            chunk_size=32,
            fallback_phrase="move to the radio",
        )
        assert text == "move to the radio"

    def test_empty_segments_returns_fallback(self):
        text = get_bridge_subtask_text(
            [],
            anchor_frame=0,
            chunk_size=32,
            fallback_phrase="some task",
        )
        assert text == "some task"

    def test_empty_fallback_returns_empty(self, two_skill_contiguous):
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            chunk_size=32,
            fallback_phrase="",
        )
        assert text == ""

    def test_with_query_frames_valid(self, two_skill_contiguous):
        """Actual query frames with no padding → valid bridge."""
        query_frames = list(range(85, 117))  # 32 frames: 85..116
        pad_mask = [False] * 32
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            action_query_frames=query_frames,
            action_is_pad=pad_mask,
            fallback_phrase="move to the radio",
        )
        assert text == "move to the radio then pick up the radio"

    def test_with_query_frames_padded(self, two_skill_contiguous):
        """Query frames with any padding → bridge rejected entirely, fallback.

        We do NOT trim to the valid prefix and bridge there — any padded step
        means the action chunk is incomplete, so the bridge phrase would be
        misleading training signal.
        """
        query_frames = list(range(85, 117))  # 32 frames
        pad_mask = [False] * 27 + [True] * 5  # last 5 are padded
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            action_query_frames=query_frames,
            action_is_pad=pad_mask,
            fallback_phrase="move to the radio",
        )
        # Any padding → reject bridge → fallback to anchor phrase
        assert text == "move to the radio"

    def test_with_query_frames_boundary_in_pad(self, two_skill_contiguous):
        """Boundary falls within padded region → still rejected (any pad = no bridge)."""
        query_frames = list(range(190, 222))  # 32 steps: 190..221
        # skill_b is 100..199, so 190..199 are valid (10 steps), 200..221 are pad
        pad_mask = [False] * 10 + [True] * 22
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=190,
            action_query_frames=query_frames,
            action_is_pad=pad_mask,
            fallback_phrase="pick up the radio",
        )
        # Any padding → reject bridge → fallback
        assert text == "pick up the radio"

    def test_single_padded_step_rejects_bridge(self, two_skill_contiguous):
        """Even a single padded step at the very end rejects the bridge."""
        query_frames = list(range(85, 117))  # 32 frames
        pad_mask = [False] * 31 + [True]  # only last step padded
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            action_query_frames=query_frames,
            action_is_pad=pad_mask,
            fallback_phrase="move to the radio",
        )
        assert text == "move to the radio"

    def test_all_padded_returns_fallback(self, two_skill_contiguous):
        query_frames = [500, 501, 502]
        pad_mask = [True, True, True]
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=500,
            action_query_frames=query_frames,
            action_is_pad=pad_mask,
            fallback_phrase="anchor_phrase",
        )
        assert text == "anchor_phrase"

    def test_annotations_primitive_no_bridge_phase1(self, two_skill_contiguous):
        """Phase 1: annotations_primitive source does NOT get bridging."""
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            chunk_size=32,
            fallback_phrase="move to the radio",
            subtask_source="annotations_primitive",
            enabled=True,
        )
        # Even with valid crossing and enabled, primitive source = no bridge
        assert text == "move to the radio"

    def test_orchestrator_no_bridge(self, two_skill_contiguous):
        """orchestrator source does NOT get bridging."""
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            chunk_size=32,
            fallback_phrase="some_orch_task",
            subtask_source="orchestrator",
            enabled=True,
        )
        assert text == "some_orch_task"

    def test_annotations_skill_bridge_works(self, two_skill_contiguous):
        """annotations_skill source with valid crossing → bridge works."""
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            chunk_size=32,
            fallback_phrase="move to the radio",
            subtask_source="annotations_skill",
            enabled=True,
        )
        assert text == "move to the radio then pick up the radio"

    def test_enabled_false_no_bridge(self, two_skill_contiguous):
        """enabled=False → no bridge regardless of source."""
        text = get_bridge_subtask_text(
            two_skill_contiguous,
            anchor_frame=85,
            chunk_size=32,
            fallback_phrase="move to the radio",
            subtask_source="annotations_skill",
            enabled=False,
        )
        assert text == "move to the radio"


# ===========================================================================
# Off-mode compatibility
# ===========================================================================

class TestOffModeCompatibility:
    """Verify that when bridge is disabled, behavior is identical to baseline."""

    def test_disabled_config_produces_same_subtask_text(self, two_skill_contiguous):
        """With default disabled config, subtask text equals anchor frame phrase."""
        cfg = SkillBridgeConfig(enabled=False)

        # Simulate: middle of first skill
        result = compute_bridge_info(two_skill_contiguous, 50, 32)
        # When disabled, the dataset doesn't call bridge logic at all,
        # so subtask_text = anchor phrase. We verify the fallback path works.
        text = get_bridge_subtask_text(
            two_skill_contiguous, 50, chunk_size=32,
            fallback_phrase=result["current_phrase"],
        )
        # Both should equal the anchor phrase
        assert text == result["current_phrase"]
        assert text == "move to the radio"

    def test_enabled_is_different_from_disabled(self, two_skill_contiguous):
        """Sanity check: enabled bridge produces different text for crossing chunks."""
        # Disabled path: only anchor phrase
        disabled_text = "move to the radio"

        # Enabled path: combined phrase for valid crossings
        enabled_result = compute_bridge_info(two_skill_contiguous, 85, 32)
        enabled_text = enabled_result["combined_phrase"] if enabled_result["bridge_valid"] else disabled_text

        assert enabled_text != disabled_text
        assert "then" in enabled_text
        assert "then" not in disabled_text

    def test_non_crossing_same_in_both_modes(self, two_skill_contiguous):
        """For non-crossing chunks, both modes produce same text."""
        disabled_text = "move to the radio"
        result = compute_bridge_info(two_skill_contiguous, 10, 32)
        assert result["bridge_valid"] is False
        # Combined phrase equals current phrase when invalid
        assert result["combined_phrase"] == disabled_text


# ===========================================================================
# Audit — episode segments
# ===========================================================================

class TestAuditEpisodeSegments:
    def test_clean_segments_pass(self, two_skill_contiguous):
        result = audit_episode_segments(two_skill_contiguous, 0, 199)
        assert result["passed"] is True
        assert result["errors"] == []
        assert result["warnings"] == []
        assert result["stats"]["num_segments"] == 2
        assert result["stats"]["overlap_count"] == 0

    def test_overlap_detected(self, two_skill_with_overlap):
        result = audit_episode_segments(two_skill_with_overlap)
        assert result["passed"] is False
        assert result["stats"]["overlap_count"] == 1
        assert any("overlap" in e for e in result["errors"])

    def test_gap_warned(self, two_skill_with_gap):
        result = audit_episode_segments(two_skill_with_gap)
        # Gap of 2 (>1) is a warning, not error
        assert result["passed"] is True  # gaps are warnings
        assert result["stats"]["num_gaps"] == 1
        assert any("gap" in w.lower() for w in result["warnings"])

    def test_single_frame_gap_no_warning(self):
        segs = [(0, 99, "a"), (101, 199, "b")]  # gap of 1
        result = audit_episode_segments(segs)
        assert result["passed"] is True
        assert result["stats"]["num_gaps"] == 1
        # single-frame gaps are not warned (gap > 1 triggers warning)
        assert not any("gap" in w.lower() for w in result["warnings"])

    def test_negative_duration(self):
        segs = [(100, 50, "bad")]
        result = audit_episode_segments(segs)
        assert result["passed"] is False
        assert any("start_frame" in e and ">" in e for e in result["errors"])

    def test_out_of_order(self):
        segs = [(100, 199, "b"), (0, 99, "a")]  # reversed
        result = audit_episode_segments(segs)
        assert result["passed"] is False
        assert any("sorted" in e for e in result["errors"])

    def test_empty_phrase(self):
        segs = [(0, 99, ""), (100, 199, "b")]
        result = audit_episode_segments(segs)
        assert result["passed"] is False
        assert any("phrase" in e for e in result["errors"])

    def test_out_of_episode_bounds(self, two_skill_contiguous):
        result = audit_episode_segments(two_skill_contiguous, episode_start=10, episode_end=150)
        assert result["passed"] is False
        # Both segments partially out of bounds
        assert any("< episode_start" in e for e in result["errors"])
        assert any("> episode_end" in e for e in result["errors"])

    def test_empty_segments(self):
        result = audit_episode_segments([])
        assert result["passed"] is True
        assert result["stats"]["num_segments"] == 0


# ===========================================================================
# Audit — bridge validity cross-check
# ===========================================================================

class TestAuditBridgeValidity:
    def test_valid_bridge_matches(self, two_skill_contiguous):
        bridge_result = compute_bridge_info(two_skill_contiguous, 85, 32)
        audit = audit_bridge_validity(bridge_result, two_skill_contiguous, 85, 32)
        assert audit["passed"] is True
        assert audit["errors"] == []

    def test_single_skill_matches(self, two_skill_contiguous):
        bridge_result = compute_bridge_info(two_skill_contiguous, 10, 32)
        audit = audit_bridge_validity(bridge_result, two_skill_contiguous, 10, 32)
        assert audit["passed"] is True

    def test_injected_error_detected(self, two_skill_contiguous):
        bridge_result = compute_bridge_info(two_skill_contiguous, 85, 32)
        # Corrupt the result
        bad_result = dict(bridge_result)
        bad_result["boundary_step"] = 999  # wrong
        audit = audit_bridge_validity(bad_result, two_skill_contiguous, 85, 32)
        assert audit["passed"] is False
        assert any("boundary_step" in e for e in audit["errors"])

    def test_all_fields_checked(self, two_skill_contiguous):
        """Verify every field in the result is cross-checked."""
        bridge_result = compute_bridge_info(two_skill_contiguous, 85, 32)
        fields = ["bridge_valid", "current_phrase", "successor_phrase",
                  "boundary_step", "combined_phrase", "crossing_count", "rejection_reason"]
        for field in fields:
            bad = dict(bridge_result)
            if isinstance(bad[field], bool):
                bad[field] = not bad[field]
            elif isinstance(bad[field], int):
                bad[field] = bad[field] + 1000
            elif isinstance(bad[field], str):
                bad[field] = "CORRUPTED_" + bad[field] if bad[field] else "CORRUPTED"
            else:
                bad[field] = "SENTINEL"
            audit = audit_bridge_validity(bad, two_skill_contiguous, 85, 32)
            assert audit["passed"] is False, f"field {field} corruption not detected"


# ===========================================================================
# Audit — chunk crossing stats
# ===========================================================================

class TestAuditChunkCrossingStats:
    def test_basic_stats(self, two_skill_contiguous):
        result = audit_chunk_crossing_stats(
            two_skill_contiguous, chunk_size=32, num_samples=200, seed=42,
        )
        assert result["passed"] is True
        stats = result["stats"]
        assert stats["total_samples"] == 200
        assert stats["valid_bridge_count"] + stats["single_skill_count"] <= 200
        assert 0.0 <= stats["valid_bridge_ratio"] <= 1.0

    def test_deterministic_seed(self, two_skill_contiguous):
        """Same seed produces identical results."""
        r1 = audit_chunk_crossing_stats(two_skill_contiguous, 32, 100, seed=0)
        r2 = audit_chunk_crossing_stats(two_skill_contiguous, 32, 100, seed=0)
        assert r1["stats"]["valid_bridge_count"] == r2["stats"]["valid_bridge_count"]

    def test_empty_segments_warns(self):
        result = audit_chunk_crossing_stats([], 32, 100)
        assert result["passed"] is True
        assert len(result["warnings"]) >= 1
        assert result["stats"]["total_samples"] == 0

    def test_three_skills_stats(self, three_skill_contiguous):
        result = audit_chunk_crossing_stats(
            three_skill_contiguous, chunk_size=80, num_samples=500, seed=0,
        )
        stats = result["stats"]
        # With 3 skills and 80-frame chunks, some chunks will span 3
        rejection_counts = stats["rejection_counts"]
        assert "multiple_crossings" in rejection_counts or stats["valid_bridge_count"] > 0


# ===========================================================================
# Audit — segment contiguity
# ===========================================================================

class TestAuditSegmentContiguity:
    def test_perfectly_contiguous(self, two_skill_contiguous):
        result = audit_segment_contiguity(two_skill_contiguous)
        assert result["passed"] is True
        assert result["stats"]["gap_locations"] == []
        assert result["stats"]["overlap_locations"] == []

    def test_gap_detected_as_error(self, two_skill_with_gap):
        result = audit_segment_contiguity(two_skill_with_gap)
        assert result["passed"] is False
        assert len(result["stats"]["gap_locations"]) == 1
        idx, size = result["stats"]["gap_locations"][0]
        assert idx == 0
        assert size == 2

    def test_overlap_detected(self, two_skill_with_overlap):
        result = audit_segment_contiguity(two_skill_with_overlap)
        assert result["passed"] is False
        assert len(result["stats"]["overlap_locations"]) == 1
        idx, size = result["stats"]["overlap_locations"][0]
        assert idx == 0
        assert size == 3  # frames 100, 101, 102

    def test_three_contiguous(self, three_skill_contiguous):
        result = audit_segment_contiguity(three_skill_contiguous)
        assert result["passed"] is True

    def test_single_segment(self):
        segs = [(0, 99, "only")]
        result = audit_segment_contiguity(segs)
        assert result["passed"] is True


# ===========================================================================
# Determinism tests
# ===========================================================================

class TestDeterminism:
    def test_compute_bridge_deterministic(self, two_skill_contiguous):
        r1 = compute_bridge_info(two_skill_contiguous, 85, 32)
        r2 = compute_bridge_info(two_skill_contiguous, 85, 32)
        assert r1 == r2

    def test_audit_stats_deterministic(self, two_skill_contiguous):
        r1 = audit_chunk_crossing_stats(two_skill_contiguous, 32, 100, seed=123)
        r2 = audit_chunk_crossing_stats(two_skill_contiguous, 32, 100, seed=123)
        assert r1["stats"] == r2["stats"]



# ===========================================================================
# Integration tests: _get_bridge_subtask_text on dataset instance
# ===========================================================================

class TestDatasetBridgeIntegration:
    """Integration tests for dataset-level bridge wiring.

    Validates the full logic of _get_bridge_subtask_text, including the
    critical global→local frame index conversion (HF dataset indices are
    global across episodes, but skill segments are per-episode local).

    The mock dataset simulates a non-first episode (global start = 5000)
    to catch the global→local bug.
    """

    @pytest.fixture
    def mock_dataset(self, two_skill_contiguous):
        """Mock dataset with correct global→local index conversion logic.

        Simulates episode at global offset 5000 (realistic: not the first
        episode in the dataset).  Global frame 5000 = local frame 0.
        """
        import types

        ds = types.SimpleNamespace()
        ds.subtask_source = "annotations_skill"
        ds._skill_bridge_config = SkillBridgeConfig(enabled=True)
        ds._subtask_segments = {7: two_skill_contiguous}
        ds._subtask_segment_ends = {7: [99, 199]}
        ds.delta_indices = {"action": list(range(32))}
        ds.fps = 30
        ds.EP_START = 5000
        ds.EP_LEN = 200

        # episode_data_index_pos maps episode_index → position in data_index
        ds.episode_data_index_pos = {7: 0}
        # Simulate episode_data_index: dict of tensor-like arrays, subscriptable by position
        # (matches real LeRobot: episode_data_index["from"][ep_pos].item())
        class _IdxArray:
            def __init__(self, vals):
                self._vals = vals
            def __getitem__(self, pos):
                class _V:
                    def __init__(v_self, v):
                        v_self._v = v
                    def item(v_self):
                        return v_self._v
                return _V(self._vals[pos])
        class _EpiData:
            def __getitem__(self, key):
                if key == "from":
                    return _IdxArray([ds.EP_START])
                elif key == "to":
                    return _IdxArray([ds.EP_START + ds.EP_LEN])
                raise KeyError(key)
        ds.episode_data_index = _EpiData()

        # Base subtask text: same logic as _get_subtask_text for annotations_skill
        def _mock_get_subtask_text(item):
            import bisect
            ep_idx = item["episode_index"]
            frame = round(item["timestamp"] * ds.fps)
            segs = ds._subtask_segments.get(ep_idx, [])
            ends = ds._subtask_segment_ends.get(ep_idx, [])
            if not segs:
                return "fallback_task"
            i = bisect.bisect_left(ends, frame)
            if 0 <= i < len(segs):
                s, e, t = segs[i]
                if s <= frame <= e:
                    return t
            return "fallback_task"

        ds._get_subtask_text = _mock_get_subtask_text

        def _mock_build_subtask_segments(ep_idx):
            return (
                ds._subtask_segments.get(ep_idx, []),
                ds._subtask_segment_ends.get(ep_idx, []),
            )

        ds._build_subtask_segments_for_episode = _mock_build_subtask_segments

        def _mock_action_horizon():
            if ds.delta_indices is not None:
                for key, deltas in ds.delta_indices.items():
                    if "action" in key:
                        return len(deltas)
            return 32

        ds._action_horizon = _mock_action_horizon

        # Method logic mirroring dataset.py _get_bridge_subtask_text
        # including global→local frame conversion (the critical bug fix)
        def _get_bridge_subtask_text(
            item,
            query_indices=None,
            padding=None,
        ):
            base_phrase = ds._get_subtask_text(item)
            if base_phrase is None:
                return None

            sb_cfg = ds._skill_bridge_config
            if sb_cfg is None or not getattr(sb_cfg, "enabled", False):
                return base_phrase
            if ds.subtask_source != "annotations_skill":
                return base_phrase

            ep_idx = item["episode_index"]
            anchor_frame = round(item["timestamp"] * ds.fps)

            if ep_idx not in ds._subtask_segments:
                segs, ends = ds._build_subtask_segments_for_episode(ep_idx)
                ds._subtask_segments[ep_idx] = segs
                ds._subtask_segment_ends[ep_idx] = ends
            segs = ds._subtask_segments[ep_idx]
            if not segs:
                return base_phrase

            # --- Compute action frames (episode-local) and pad mask -------
            action_is_pad = None
            local_action_frames = None
            episode_end_frame = None

            # 1. Try explicit padding from caller (streaming path)
            if padding is not None:
                for key in padding:
                    if "action_is_pad" in key:
                        action_is_pad = list(padding[key])
                        break

            # 2. Try padding from item dict (non-streaming path)
            if action_is_pad is None:
                for key in item:
                    if "action_is_pad" in key:
                        action_is_pad = list(item[key])
                        break

            # 3. Compute local action frame indices
            if query_indices is not None:
                action_global = None
                for key in query_indices:
                    if key.endswith("action") or key.startswith("action"):
                        action_global = query_indices[key]
                        break
                if action_global is not None:
                    ep_pos = ds.episode_data_index_pos[ep_idx]
                    ep_start = int(ds.episode_data_index["from"][ep_pos].item())
                    local_action_frames = [f - ep_start for f in action_global]
            elif action_is_pad is not None:
                # Non-streaming path: derive from anchor + horizon
                horizon = len(action_is_pad)
                local_action_frames = [anchor_frame + d for d in range(horizon)]

            # 4. Episode end (for padded-tail detection when no pad mask)
            if ds.episode_data_index is not None and ep_idx in ds.episode_data_index_pos:
                ep_pos = ds.episode_data_index_pos[ep_idx]
                ep_end_global = int(ds.episode_data_index["to"][ep_pos].item())
                ep_start = int(ds.episode_data_index["from"][ep_pos].item())
                episode_end_frame = ep_end_global - ep_start - 1

            return get_bridge_subtask_text(
                segs,
                anchor_frame,
                chunk_size=ds._action_horizon(),
                action_query_frames=local_action_frames,
                action_is_pad=action_is_pad,
                episode_end_frame=episode_end_frame,
                min_pre_boundary_steps=getattr(sb_cfg, "min_pre_boundary_steps", 1),
                min_post_boundary_steps=getattr(sb_cfg, "min_post_boundary_steps", 1),
                fallback_phrase=base_phrase,
                subtask_source=ds.subtask_source,
                enabled=True,
            )

        ds._get_bridge_subtask_text = _get_bridge_subtask_text
        return ds

    def test_global_indices_convert_to_local(self, mock_dataset):
        """CRITICAL: global HF query indices must be converted to episode-local.

        If this fails, the bridge is completely wrong for all non-first
        episodes — global frame numbers (e.g. 5085) are compared against
        episode-local segment boundaries (0..199).
        """
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,  # local frame 85
        }
        # Global: episode starts at 5000, so local 85 = global 5085
        query_indices = {"action": list(range(5085, 5085 + 32))}
        padding = {"action_is_pad": [False] * 32}
        result = mock_dataset._get_bridge_subtask_text(
            item, query_indices=query_indices, padding=padding
        )
        # Local frame 85 is in skill_a, boundary at local frame 100
        # chunk 85..116 → valid crossing → bridge
        assert result == "move to the radio then pick up the radio"

    def test_meta_global_indices_without_conversion_would_fail(self, mock_dataset):
        """Meta-test: proves global→local conversion is necessary.

        Without conversion, global indices (5085+) would fall way past
        segment end (199) and return fallback, which is the bug.
        """
        segs = mock_dataset._subtask_segments[7]
        # Call helper DIRECTLY with global indices (the bug scenario)
        result = get_bridge_subtask_text(
            segs,
            anchor_frame=85,  # local anchor
            action_query_frames=list(range(5085, 5117)),  # GLOBAL (wrong)
            action_is_pad=[False] * 32,
            fallback_phrase="move to the radio",
            subtask_source="annotations_skill",
            enabled=True,
        )
        # anchor_frame (85) doesn't match query start (5085)
        # The helper would compute bridge on frames 5085..5116
        # but segments are 0..199 → anchor outside → fallback
        # This proves the bug — raw global indices give wrong answer
        assert result == "move to the radio"

    def test_bridge_enabled_valid_crossing_no_query(self, mock_dataset):
        """Enabled + annotations_skill + valid crossing (no query indices)."""
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "move to the radio then pick up the radio"

    def test_bridge_disabled_fallback(self, mock_dataset):
        """Disabled → same as _get_subtask_text."""
        mock_dataset._skill_bridge_config = SkillBridgeConfig(enabled=False)
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "move to the radio"

    def test_bridge_none_config_fallback(self, mock_dataset):
        """Config is None → same as _get_subtask_text."""
        mock_dataset._skill_bridge_config = None
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "move to the radio"

    def test_orchestrator_source_no_bridge(self, mock_dataset):
        """orchestrator source → no bridge even when enabled."""
        mock_dataset.subtask_source = "orchestrator"
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "move to the radio"

    def test_annotations_primitive_no_bridge(self, mock_dataset):
        """annotations_primitive source → no bridge in Phase 1."""
        mock_dataset.subtask_source = "annotations_primitive"
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "move to the radio"

    def test_single_skill_chunk_fallback(self, mock_dataset):
        """Chunk entirely in one skill → fallback to anchor phrase."""
        item = {
            "episode_index": 7,
            "timestamp": 10 / 30.0,  # frame 10, deep in skill_a
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "move to the radio"

    def test_streaming_valid_bridge(self, mock_dataset):
        """Streaming path: global query indices → valid bridge after conversion."""
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
        }
        query_indices = {"action": list(range(5085, 5085 + 32))}  # global
        padding = {"action_is_pad": [False] * 32}
        result = mock_dataset._get_bridge_subtask_text(
            item, query_indices=query_indices, padding=padding
        )
        assert result == "move to the radio then pick up the radio"

    def test_streaming_any_pad_rejects(self, mock_dataset):
        """Streaming path: any padded step → no bridge."""
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
        }
        query_indices = {"action": list(range(5085, 5085 + 32))}
        padding = {"action_is_pad": [False] * 30 + [True, True]}
        result = mock_dataset._get_bridge_subtask_text(
            item, query_indices=query_indices, padding=padding
        )
        assert result == "move to the radio"

    def test_non_streaming_item_level_pad(self, mock_dataset):
        """Non-streaming: action_is_pad in item dict (no explicit query_indices)."""
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
            "action_is_pad": [False] * 32,
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        # 32 steps from frame 85, boundary at 100 (step 15) → valid bridge
        assert result == "move to the radio then pick up the radio"

    def test_non_streaming_padded_tail_rejects(self, mock_dataset):
        """Non-streaming: pad mask with tail padding → no bridge."""
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
            "action_is_pad": [False] * 30 + [True, True],
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "move to the radio"

    def test_episode_end_used_when_no_query_or_pad(self, mock_dataset):
        """When neither query_indices nor pad mask available, episode_end rejects tail."""
        item = {
            "episode_index": 7,
            "timestamp": 190 / 30.0,  # local frame 190
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        # 32 steps from 190 → extends to 221, past episode end (199)
        # padded_tail rejection → fallback to anchor frame skill
        assert result == "pick up the radio"

    def test_off_mode_equality_across_episode(self, mock_dataset):
        """Off-mode: bridge output equals baseline for 10 representative frames
        including near skill boundaries and episode edges."""
        mock_dataset._skill_bridge_config = SkillBridgeConfig(enabled=False)

        test_frames = [0, 1, 5, 50, 98, 99, 100, 101, 150, 199]
        for frame in test_frames:
            item = {
                "episode_index": 7,
                "timestamp": frame / 30.0,
            }
            bridge_result = mock_dataset._get_bridge_subtask_text(item)
            baseline_result = mock_dataset._get_subtask_text(item)
            assert bridge_result == baseline_result, (
                f"Off-mode mismatch at frame {frame}: "
                f"bridge='{bridge_result}' vs baseline='{baseline_result}'"
            )

    def test_off_mode_with_query_indices_equals_baseline(self, mock_dataset):
        """Off-mode with query indices + padding still equals baseline."""
        mock_dataset._skill_bridge_config = SkillBridgeConfig(enabled=False)

        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,
        }
        query_indices = {"action": list(range(5085, 5117))}
        padding = {"action_is_pad": [False] * 30 + [True, True]}
        bridge_result = mock_dataset._get_bridge_subtask_text(
            item, query_indices=query_indices, padding=padding
        )
        baseline_result = mock_dataset._get_subtask_text(item)
        assert bridge_result == baseline_result

    def test_episode_without_segments_fallback(self, mock_dataset):
        """Episode with no subtask segments → fallback."""
        item = {
            "episode_index": 99,
            "timestamp": 5.0,
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "fallback_task"

    def test_min_pre_threshold_wired(self, mock_dataset):
        """min_pre_boundary_steps is actually used from config."""
        mock_dataset._skill_bridge_config = SkillBridgeConfig(
            enabled=True,
            min_pre_boundary_steps=20,
            min_post_boundary_steps=1,
        )
        item = {
            "episode_index": 7,
            "timestamp": 85 / 30.0,  # boundary at step 15 → insufficient pre
        }
        result = mock_dataset._get_bridge_subtask_text(item)
        assert result == "move to the radio"
