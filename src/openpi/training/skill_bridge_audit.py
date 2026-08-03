"""Standalone read-only audit functions for skill segment data and bridge computation results.

All functions are pure (no side effects) and return a consistent structured dict with
``passed``, ``errors``, ``warnings``, and ``stats`` keys.

Segment format used throughout: ``(start_frame, end_frame, phrase)`` tuples where both
start_frame and end_frame are inclusive.
"""

from __future__ import annotations

import random
import bisect


# ---------------------------------------------------------------------------
# Internal helpers — independent reimplementation of bridge logic
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    """Return the canonical empty audit result dict."""
    return {
        "passed": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }


def _find_segment_index(
    segments: list[tuple[int, int, str]],
    frame: int,
) -> int | None:
    """Return the index of the segment that contains *frame*, or ``None``.

    Uses binary search on segment start_frame for efficiency.  Segments are
    assumed to be sorted by start_frame and non-overlapping.
    """
    if not segments:
        return None

    start_frames = [seg[0] for seg in segments]
    idx = bisect.bisect_right(start_frames, frame) - 1
    if idx < 0:
        return None

    start, end, _ = segments[idx]
    if start <= frame <= end:
        return idx
    return None


def _compute_bridge_info(
    segments: list[tuple[int, int, str]],
    start_frame: int,
    chunk_size: int,
    *,
    min_pre_boundary_steps: int = 1,
    min_post_boundary_steps: int = 1,
    episode_end_frame: int | None = None,
) -> dict:
    """Independently compute bridge info from raw segments.

    This is the reference implementation that ``audit_bridge_validity`` uses to
    cross-check an externally supplied ``bridge_result``.

    Mirrors the semantics of ``skill_bridge_core.compute_bridge_info``:
      - Default phrase fields are empty strings, boundary_step defaults to 0.
      - combined_phrase is ``"{current} then {successor}"`` when valid.
      - Rejection reasons: ``no_segments``, ``anchor_outside_any_segment``,
        ``single_skill``, ``multiple_crossings``, ``overlap_between_skills``,
        ``gap_between_skills``, ``insufficient_pre_steps``,
        ``insufficient_post_steps``, ``padded_tail``.
    """
    result = {
        "bridge_valid": False,
        "current_phrase": "",
        "successor_phrase": "",
        "boundary_step": 0,
        "combined_phrase": "",
        "crossing_count": 0,
        "rejection_reason": None,
    }

    if not segments:
        result["rejection_reason"] = "no_segments"
        return result

    # Find anchor segment
    anchor_idx = _find_segment_index(segments, start_frame)
    if anchor_idx is None:
        result["rejection_reason"] = "anchor_outside_any_segment"
        return result

    current_start, current_end, current_phrase = segments[anchor_idx]
    result["current_phrase"] = current_phrase
    result["combined_phrase"] = current_phrase  # default: same as current

    chunk_end = start_frame + chunk_size - 1  # inclusive

    padded_tail = False
    if episode_end_frame is not None and chunk_end > episode_end_frame:
        padded_tail = True

    # Find all segments that overlap the chunk
    overlapped_indices = [anchor_idx]
    i = anchor_idx + 1
    while i < len(segments) and segments[i][0] <= chunk_end:
        if segments[i][1] >= start_frame:
            overlapped_indices.append(i)
        i += 1

    num_overlapped = len(overlapped_indices)
    crossing_count = num_overlapped - 1
    result["crossing_count"] = crossing_count

    # Single segment
    if num_overlapped == 1:
        result["rejection_reason"] = "single_skill"
        return result

    # 3+ segments (multiple crossings)
    if num_overlapped >= 3:
        result["rejection_reason"] = "multiple_crossings"
        result["successor_phrase"] = segments[overlapped_indices[1]][2]
        succ_start = segments[overlapped_indices[1]][0]
        result["boundary_step"] = succ_start - start_frame
        return result

    # Exactly 2 segments — one boundary
    succ_idx = overlapped_indices[1]
    succ_start, succ_end, succ_phrase = segments[succ_idx]
    result["successor_phrase"] = succ_phrase

    boundary_step = succ_start - start_frame
    result["boundary_step"] = boundary_step

    # Contiguity checks
    if succ_start <= current_end:
        result["rejection_reason"] = "overlap_between_skills"
        return result
    if succ_start != current_end + 1:
        result["rejection_reason"] = "gap_between_skills"
        return result

    # Min pre-boundary steps
    if boundary_step < min_pre_boundary_steps:
        result["rejection_reason"] = "insufficient_pre_steps"
        return result

    # Min post-boundary steps
    post_steps = chunk_size - boundary_step
    if post_steps < min_post_boundary_steps:
        result["rejection_reason"] = "insufficient_post_steps"
        return result

    # Padded tail
    if padded_tail:
        result["rejection_reason"] = "padded_tail"
        return result

    # Valid bridge
    result["bridge_valid"] = True
    result["combined_phrase"] = f"{current_phrase} then {succ_phrase}"
    result["rejection_reason"] = None
    return result


# ---------------------------------------------------------------------------
# Public audit functions
# ---------------------------------------------------------------------------

def audit_episode_segments(
    segments: list[tuple[int, int, str]],
    episode_start: int | None = None,
    episode_end: int | None = None,
) -> dict:
    """Validate per-episode skill segment data integrity.

    Checks performed:
      * Segments are sorted by ``start_frame`` (strictly increasing).
      * No overlaps (``segments[i].end < segments[i+1].start``).
      * No gaps of more than 1 frame (reported as warnings, not errors).
      * All segments lie within ``[episode_start, episode_end]`` when provided.
      * ``start_frame <= end_frame`` for every segment.
      * No empty / whitespace-only phrases.

    Args:
        segments: List of ``(start_frame, end_frame, phrase)`` tuples.
        episode_start: If given, all segments must start at or after this frame.
        episode_end: If given, all segments must end at or before this frame.

    Returns:
        Dict with ``passed`` (bool), ``errors`` (list[str]), ``warnings`` (list[str]),
        and ``stats`` (dict with ``num_segments``, ``num_gaps``, ``total_gap_frames``,
        ``overlap_count``).
    """
    result = _empty_result()
    stats = {
        "num_segments": len(segments),
        "num_gaps": 0,
        "total_gap_frames": 0,
        "overlap_count": 0,
    }
    result["stats"] = stats

    # Per-segment validity
    for i, (s, e, phrase) in enumerate(segments):
        if s > e:
            result["errors"].append(
                f"segment[{i}]: start_frame ({s}) > end_frame ({e})"
            )
            result["passed"] = False
        if not phrase or not phrase.strip():
            result["errors"].append(
                f"segment[{i}]: empty or whitespace-only phrase"
            )
            result["passed"] = False

    # Episode bounds
    if episode_start is not None:
        for i, (s, e, _) in enumerate(segments):
            if s < episode_start:
                result["errors"].append(
                    f"segment[{i}]: start_frame {s} < episode_start {episode_start}"
                )
                result["passed"] = False
    if episode_end is not None:
        for i, (s, e, _) in enumerate(segments):
            if e > episode_end:
                result["errors"].append(
                    f"segment[{i}]: end_frame {e} > episode_end {episode_end}"
                )
                result["passed"] = False

    # Sorting, overlaps, gaps
    for i in range(1, len(segments)):
        prev_s, prev_e, _ = segments[i - 1]
        cur_s, cur_e, _ = segments[i]

        if cur_s < prev_s:
            result["errors"].append(
                f"segments not sorted by start_frame: "
                f"segment[{i}].start ({cur_s}) < segment[{i-1}].start ({prev_s})"
            )
            result["passed"] = False
            continue

        # Overlap: since segments are inclusive [s, e], overlap if cur_s <= prev_e
        if cur_s <= prev_e:
            overlap = prev_e - cur_s + 1
            stats["overlap_count"] += 1
            result["errors"].append(
                f"overlap between segment[{i-1}] and segment[{i}]: "
                f"{overlap} frame(s) (prev.end={prev_e}, cur.start={cur_s})"
            )
            result["passed"] = False
        else:
            # Gap — frames between prev_e and cur_s, exclusive on both sides
            gap = cur_s - prev_e - 1
            if gap > 0:
                stats["num_gaps"] += 1
                stats["total_gap_frames"] += gap
                if gap > 1:
                    result["warnings"].append(
                        f"gap of {gap} frame(s) between segment[{i-1}] "
                        f"(end={prev_e}) and segment[{i}] (start={cur_s})"
                    )

    return result


def audit_bridge_validity(
    bridge_result: dict,
    segments: list[tuple[int, int, str]],
    start_frame: int,
    chunk_size: int,
    *,
    min_pre_boundary_steps: int = 1,
    min_post_boundary_steps: int = 1,
    episode_end_frame: int | None = None,
) -> dict:
    """Independently verify that a ``bridge_result`` from ``compute_bridge_info`` is correct.

    Recomputes the expected bridge info from the raw segments using an independent
    implementation and compares field by field.  Any mismatch is recorded as an error.

    Args:
        bridge_result: The bridge result dict to audit (from ``compute_bridge_info``).
        segments: Raw skill segments used as ground truth.
        start_frame: First frame of the chunk window.
        chunk_size: Number of frames in the chunk.
        min_pre_boundary_steps: Minimum frames before boundary required for validity.
        min_post_boundary_steps: Minimum frames after boundary required for validity.
        episode_end_frame: If set, chunks extending past this frame are rejected.

    Returns:
        Dict with ``passed`` (bool — True only when all fields match), ``errors``,
        ``warnings``, and ``stats`` (computed vs expected values for key fields).
    """
    result = _empty_result()

    expected = _compute_bridge_info(
        segments,
        start_frame,
        chunk_size,
        min_pre_boundary_steps=min_pre_boundary_steps,
        min_post_boundary_steps=min_post_boundary_steps,
        episode_end_frame=episode_end_frame,
    )

    stats = {
        "computed_valid": bridge_result.get("bridge_valid"),
        "expected_valid": expected["bridge_valid"],
        "computed_boundary_step": bridge_result.get("boundary_step"),
        "expected_boundary_step": expected["boundary_step"],
    }
    result["stats"] = stats

    # Compare each field
    fields_to_check = [
        "bridge_valid",
        "boundary_step",
        "current_phrase",
        "successor_phrase",
        "combined_phrase",
        "crossing_count",
        "rejection_reason",
    ]

    for field in fields_to_check:
        computed = bridge_result.get(field)
        exp = expected[field]
        if computed != exp:
            result["errors"].append(
                f"{field} mismatch: computed={computed!r}, expected={exp!r}"
            )
            result["passed"] = False

    return result


def audit_chunk_crossing_stats(
    segments: list[tuple[int, int, str]],
    chunk_size: int,
    num_samples: int = 100,
    *,
    seed: int = 0,
    min_pre_boundary_steps: int = 1,
    min_post_boundary_steps: int = 1,
    episode_end_frame: int | None = None,
) -> dict:
    """Compute bridge-outcome statistics by sampling random chunk positions.

    Randomly samples ``num_samples`` start-frame positions within the segment
    range, computes the bridge result for each, and aggregates statistics.

    This is an informative audit — ``passed`` is always True since the function
    simply computes statistics.  Warnings are raised for unusual distributions
    (e.g. very low valid-bridge ratio).

    Args:
        segments: Skill segments to sample from.
        chunk_size: Size of the chunk window in frames.
        num_samples: Number of random start positions to sample.
        seed: Random seed for reproducibility.
        min_pre_boundary_steps: Forwarded to the bridge computation.
        min_post_boundary_steps: Forwarded to the bridge computation.
        episode_end_frame: If set, chunks extending past this frame are rejected.

    Returns:
        Dict with ``passed`` (always True), ``errors`` (empty list), ``warnings``,
        and ``stats`` containing:
          - ``total_samples``
          - ``single_skill_count`` (chunks entirely within one segment)
          - ``valid_bridge_count`` (chunks with bridge_valid == True)
          - ``rejection_counts`` (dict mapping rejection_reason -> count)
          - ``valid_bridge_ratio`` (valid_bridge_count / total_samples)
          - ``avg_boundary_step`` (mean boundary_step across valid bridges)
    """
    result = _empty_result()

    rejection_counts: dict[str, int] = {}
    single_skill_count = 0
    valid_bridge_count = 0
    boundary_step_sum = 0
    valid_boundary_samples = 0

    if not segments:
        result["warnings"].append("no segments provided — all stats are zero")
        result["stats"] = {
            "total_samples": 0,
            "single_skill_count": 0,
            "valid_bridge_count": 0,
            "rejection_counts": {},
            "valid_bridge_ratio": 0.0,
            "avg_boundary_step": 0.0,
        }
        return result

    rng = random.Random(seed)

    # Determine valid start_frame range: sample from first segment start
    # to last segment end minus chunk_size (clamped).
    first_start = segments[0][0]
    last_end = segments[-1][1]
    if episode_end_frame is not None:
        last_end = min(last_end, episode_end_frame)

    max_start = max(first_start, last_end - chunk_size + 1)

    for _ in range(num_samples):
        start = rng.randint(first_start, max_start)
        info = _compute_bridge_info(
            segments,
            start,
            chunk_size,
            min_pre_boundary_steps=min_pre_boundary_steps,
            min_post_boundary_steps=min_post_boundary_steps,
            episode_end_frame=episode_end_frame,
        )

        if info["bridge_valid"]:
            valid_bridge_count += 1
            boundary_step_sum += info["boundary_step"]
            valid_boundary_samples += 1
        elif info["rejection_reason"] == "single_skill":
            single_skill_count += 1

        reason = info["rejection_reason"]
        if reason is not None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    valid_ratio = valid_bridge_count / num_samples if num_samples > 0 else 0.0
    avg_boundary = (
        boundary_step_sum / valid_boundary_samples
        if valid_boundary_samples > 0
        else 0.0
    )

    if valid_ratio < 0.05 and num_samples >= 20:
        result["warnings"].append(
            f"valid_bridge_ratio is very low ({valid_ratio:.3f}); "
            f"chunk_size may be too small or segments too long relative to chunk size"
        )

    result["stats"] = {
        "total_samples": num_samples,
        "single_skill_count": single_skill_count,
        "valid_bridge_count": valid_bridge_count,
        "rejection_counts": rejection_counts,
        "valid_bridge_ratio": valid_ratio,
        "avg_boundary_step": avg_boundary,
    }

    return result


def audit_segment_contiguity(
    segments: list[tuple[int, int, str]],
) -> dict:
    """Check whether segments form a perfectly contiguous sequence.

    A perfectly contiguous sequence has:
      * No overlaps between adjacent segments.
      * No gaps between adjacent segments (i.e.
        ``segments[i+1].start == segments[i].end + 1``).
      * Segments sorted by start_frame.
      * Every segment has ``start_frame <= end_frame``.

    This is a stricter variant of :func:`audit_episode_segments` focused
    exclusively on contiguity — even single-frame gaps are treated as errors.

    Args:
        segments: List of ``(start_frame, end_frame, phrase)`` tuples.

    Returns:
        Dict with ``passed``, ``errors``, ``warnings``, and ``stats``.
        The ``stats`` dict contains:
          - ``num_segments``
          - ``gap_locations`` — list of ``(index, gap_size)`` tuples
          - ``overlap_locations`` — list of ``(index, overlap_size)`` tuples
    """
    result = _empty_result()
    gap_locations: list[tuple[int, int]] = []
    overlap_locations: list[tuple[int, int]] = []

    stats = {
        "num_segments": len(segments),
        "gap_locations": gap_locations,
        "overlap_locations": overlap_locations,
    }
    result["stats"] = stats

    # Per-segment basic checks
    for i, (s, e, _phrase) in enumerate(segments):
        if s > e:
            result["errors"].append(
                f"segment[{i}]: negative duration (start={s} > end={e})"
            )
            result["passed"] = False

    # Sorting + adjacency checks
    for i in range(1, len(segments)):
        prev_s, prev_e, _ = segments[i - 1]
        cur_s, cur_e, _ = segments[i]

        if cur_s < prev_s:
            result["errors"].append(
                f"segments not sorted: segment[{i}].start ({cur_s}) "
                f"< segment[{i-1}].start ({prev_s})"
            )
            result["passed"] = False
            continue

        expected_next_start = prev_e + 1
        if cur_s < expected_next_start:
            # Overlap
            overlap = prev_e - cur_s + 1
            overlap_locations.append((i - 1, overlap))
            result["errors"].append(
                f"overlap at segment[{i-1}]/segment[{i}]: "
                f"{overlap} frame(s) (prev.end={prev_e}, cur.start={cur_s})"
            )
            result["passed"] = False
        elif cur_s > expected_next_start:
            # Gap
            gap = cur_s - prev_e - 1
            gap_locations.append((i - 1, gap))
            result["errors"].append(
                f"gap at segment[{i-1}]/segment[{i}]: "
                f"{gap} frame(s) (prev.end={prev_e}, cur.start={cur_s})"
            )
            result["passed"] = False
        # else cur_s == expected_next_start: perfectly contiguous, no issue

    return result
