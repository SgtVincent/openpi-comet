"""
Skill bridge core logic module.

Detect when an action chunk crosses a skill boundary, and if the crossing is
valid (exactly one boundary, contiguous segments, enough steps on each side),
produce a combined subtask_text of "{current} then {successor}".

All other cases keep the original single-skill subtask_text and optionally
report why it was rejected.

This module is pure Python with only stdlib dependencies (bisect).
"""

from __future__ import annotations

import bisect
from typing import Optional


def find_segment_at_frame(
    segments: list[tuple[int, int, str]],
    frame: int,
) -> Optional[int]:
    """Return the index of the segment containing *frame* (inclusive), or None.

    Uses binary search on segment start_frame for efficiency.  Segments are
    assumed to be sorted by start_frame and non-overlapping.
    """
    if not segments:
        return None

    # bisect on start_frame to find the rightmost segment whose start <= frame
    start_frames = [seg[0] for seg in segments]
    idx = bisect.bisect_right(start_frames, frame) - 1
    if idx < 0:
        return None

    start, end, _ = segments[idx]
    if start <= frame <= end:
        return idx
    return None


def compute_bridge_info(
    segments: list[tuple[int, int, str]],
    start_frame: int,
    chunk_size: int,
    *,
    min_pre_boundary_steps: int = 1,
    min_post_boundary_steps: int = 1,
    episode_end_frame: Optional[int] = None,
) -> dict:
    """Compute bridge information for an action chunk.

    Parameters
    ----------
    segments : list of (start_frame, end_frame, phrase)
        Sorted by start_frame, non-overlapping.  Each segment is inclusive on
        both ends.
    start_frame : int
        First frame index of the action chunk (anchor frame).
    chunk_size : int
        Number of frames in the action chunk (e.g. 32).
    min_pre_boundary_steps : int
        Minimum steps required BEFORE the boundary in the chunk for a valid
        bridge.  The boundary must be at step index >= min_pre_boundary_steps.
    min_post_boundary_steps : int
        Minimum steps required AFTER the boundary in the chunk for a valid
        bridge.  The chunk must contain at least min_post_boundary_steps of
        the successor skill.
    episode_end_frame : int or None
        If provided, the last valid frame index of the episode (inclusive).
        Any chunk frame > episode_end_frame is considered padding and makes
        the bridge invalid.

    Returns
    -------
    dict with keys:
        bridge_valid : bool
            True only if exactly one valid crossing with enough steps on each
            side.
        current_phrase : str
            Phrase of the skill at start_frame.  Empty string if no segment
            matches.
        successor_phrase : str
            Phrase of the next skill after the boundary.  Empty string if no
            successor.
        boundary_step : int
            Step index [0, chunk_size] where the boundary falls.  Step
            boundary_step is the first step of the successor skill.  0 if no
            crossing.
        combined_phrase : str
            "{current_phrase} then {successor_phrase}" if valid, else same as
            current_phrase.
        crossing_count : int
            Number of skill boundaries the chunk crosses (0 = single skill,
            1 = one boundary, 2+ = multiple).
        rejection_reason : str or None
            Why bridge is invalid, or None when valid.
    """
    # --- Result default fields -------------------------------------------------
    result = {
        "bridge_valid": False,
        "current_phrase": "",
        "successor_phrase": "",
        "boundary_step": 0,
        "combined_phrase": "",
        "crossing_count": 0,
        "rejection_reason": None,
    }

    # --- Step 1: no segments ---------------------------------------------------
    if not segments:
        result["rejection_reason"] = "no_segments"
        return result

    # --- Step 2: find anchor segment -------------------------------------------
    anchor_idx = find_segment_at_frame(segments, start_frame)
    if anchor_idx is None:
        result["rejection_reason"] = "anchor_outside_any_segment"
        return result

    current_start, current_end, current_phrase = segments[anchor_idx]
    result["current_phrase"] = current_phrase
    result["combined_phrase"] = current_phrase  # default: same as current

    # --- Step 3: chunk frame range ---------------------------------------------
    chunk_end = start_frame + chunk_size - 1  # inclusive

    # --- Step 4 / Step 11: padded tail check -----------------------------------
    # Record padded_tail flag; the actual rejection is applied at the
    # appropriate branch (single-skill chunks with padding are still
    # "single_skill", not "padded_tail", per the semantics that padded_tail
    # matters for bridge validity specifically).
    padded_tail = False
    if episode_end_frame is not None and chunk_end > episode_end_frame:
        padded_tail = True

    # --- Step 5: find all segments that overlap the chunk ----------------------
    # A segment [s, e] overlaps chunk [start_frame, chunk_end] iff
    # s <= chunk_end and e >= start_frame.
    # Walk forward from anchor_idx (we already know anchor overlaps).
    overlapped_indices = [anchor_idx]

    i = anchor_idx + 1
    while i < len(segments) and segments[i][0] <= chunk_end:
        # segment starts within or before chunk_end; check it overlaps
        if segments[i][1] >= start_frame:
            overlapped_indices.append(i)
        i += 1

    num_overlapped = len(overlapped_indices)
    crossing_count = num_overlapped - 1
    result["crossing_count"] = crossing_count

    # --- Step 6: single segment ------------------------------------------------
    if num_overlapped == 1:
        result["rejection_reason"] = "single_skill"
        return result

    # --- Step 13: 3+ segments (multiple crossings) -----------------------------
    if num_overlapped >= 3:
        result["rejection_reason"] = "multiple_crossings"
        # Populate successor_phrase with the 2nd skill for diagnostics
        result["successor_phrase"] = segments[overlapped_indices[1]][2]
        # boundary_step of first boundary
        succ_start = segments[overlapped_indices[1]][0]
        result["boundary_step"] = succ_start - start_frame
        return result

    # --- Step 7: exactly 2 segments — one boundary -----------------------------
    # overlapped_indices[0] is the current skill, overlapped_indices[1] is successor
    succ_idx = overlapped_indices[1]
    succ_start, succ_end, succ_phrase = segments[succ_idx]
    result["successor_phrase"] = succ_phrase

    # boundary_step: step index where successor skill starts
    # step 0 is start_frame, step k is start_frame + k
    # successor starts at succ_start, so boundary_step = succ_start - start_frame
    boundary_step = succ_start - start_frame
    result["boundary_step"] = boundary_step

    # Contiguity checks
    # current.end + 1 should == successor.start
    if succ_start <= current_end:
        result["rejection_reason"] = "overlap_between_skills"
        return result
    if succ_start != current_end + 1:
        result["rejection_reason"] = "gap_between_skills"
        return result

    # --- Step 9: min_pre_boundary_steps ----------------------------------------
    if boundary_step < min_pre_boundary_steps:
        result["rejection_reason"] = "insufficient_pre_steps"
        return result

    # --- Step 10: min_post_boundary_steps --------------------------------------
    post_steps = chunk_size - boundary_step
    if post_steps < min_post_boundary_steps:
        result["rejection_reason"] = "insufficient_post_steps"
        return result

    # --- Step 11: padded tail --------------------------------------------------
    if padded_tail:
        result["rejection_reason"] = "padded_tail"
        return result

    # --- Step 12: valid bridge -------------------------------------------------
    result["bridge_valid"] = True
    result["combined_phrase"] = f"{current_phrase} then {succ_phrase}"
    result["rejection_reason"] = None
    return result
