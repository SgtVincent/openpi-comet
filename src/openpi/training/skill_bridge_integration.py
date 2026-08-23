"""Integration helpers for skill bridge in BehaviorLeRobotDataset.

This module provides the bridge-aware subtask text computation that plugs into
``BehaviorLeRobotDataset._get_subtask_text``.  When the skill bridge config is
enabled and the dataset uses ``subtask_source="annotations_skill"`` (Phase 1),
we detect whether the action chunk crosses exactly one contiguous skill boundary
and, if so, emit a combined subtask text of
``"{current_skill} then {successor_skill}"``.

All other cases — single-skill chunks, multiple crossings, gaps, overlaps,
padded tails, non-annotations_skill sources — fall back to the original
anchor-frame skill phrase, preserving existing behavior exactly.

Design rules (Phase 1 minimal baseline):
  * Deterministic: same inputs always produce same output.
  * Default-off: ``skill_bridge.enabled=False`` is byte-identical to the
    original ``_get_subtask_text`` path.
  * Uses actual query indices / padding semantics when available; falls back to
    ``start_frame + chunk_size`` when only the anchor frame is known.
  * Any padded action step = bridge rejected (no trim-to-valid).
  * Phase 1: only ``annotations_skill`` source is valid for bridging.
  * No per-sample logging or counters — auditors (``skill_bridge_audit``)
    should be run separately for dataset-level statistics.
  * Pure function: no mutation of dataset state.
"""

from __future__ import annotations

from typing import Optional

from openpi.training.skill_bridge_core import compute_bridge_info


# Phase 1: only annotations_skill supports bridging
VALID_BRIDGE_SOURCES = frozenset({"annotations_skill"})


def is_bridge_source(subtask_source: str) -> bool:
    """Return True if *subtask_source* supports skill bridging (Phase 1)."""
    return subtask_source in VALID_BRIDGE_SOURCES


# ---------------------------------------------------------------------------
# Public helper — call from BehaviorLeRobotDataset.__getitem__
# ---------------------------------------------------------------------------

def get_bridge_subtask_text(
    segments: list[tuple[int, int, str]],
    anchor_frame: int,
    *,
    chunk_size: Optional[int] = None,
    action_query_frames: Optional[list[int]] = None,
    action_is_pad: Optional[list[bool]] = None,
    episode_end_frame: Optional[int] = None,
    min_pre_boundary_steps: int = 1,
    min_post_boundary_steps: int = 1,
    fallback_phrase: str = "",
    subtask_source: str = "annotations_skill",
    enabled: bool = True,
) -> str:
    """Return the subtask text for a sample, with optional skill-bridge enhancement.

    When bridging is enabled, the source supports bridging (Phase 1:
    ``annotations_skill`` only), and the chunk crosses exactly one contiguous
    skill boundary with no padding and enough steps on each side, returns
    ``"{current} then {successor}"``.

    Otherwise returns *fallback_phrase* (the original anchor-frame skill phrase).

    Parameters
    ----------
    segments : list of (start_frame, end_frame, phrase)
        Per-episode skill segments, sorted by start_frame, inclusive on both ends.
    anchor_frame : int
        Frame index of the anchor / observation frame.
    chunk_size : int, optional
        Number of frames in the action chunk.  Required when
        *action_query_frames* is not provided.
    action_query_frames : list of int, optional
        Actual per-step frame indices for the action chunk, as returned by
        ``_get_query_indices``.  If supplied together with *action_is_pad*,
        any padded step causes immediate rejection.
    action_is_pad : list of bool, optional
        Per-step padding mask (True = padded / out-of-episode).  If any step
        is padded, the bridge is rejected outright — we do NOT trim to the
        valid prefix and bridge there.
    episode_end_frame : int, optional
        Last valid frame index of the episode (inclusive).  Used for
        padded-tail rejection when *action_query_frames* is not given.
    min_pre_boundary_steps : int
        Minimum steps before the boundary required for a valid bridge.
    min_post_boundary_steps : int
        Minimum steps after the boundary required for a valid bridge.
    fallback_phrase : str
        Phrase to return when bridge is not valid.  This should be the
        original anchor-frame skill phrase from ``_get_subtask_text``.
    subtask_source : str
        Source of subtask annotations.  Phase 1: only ``annotations_skill``
        supports bridging; all other sources return fallback.
    enabled : bool
        Master enable flag.  When False, returns *fallback_phrase* immediately.

    Returns
    -------
    str
        Combined bridge phrase when valid, *fallback_phrase* otherwise.
    """
    # --- Fast path: disabled, wrong source, empty segments, no fallback ------
    if not enabled or not is_bridge_source(subtask_source):
        return fallback_phrase
    if not segments or fallback_phrase is None or fallback_phrase == "":
        return fallback_phrase

    # --- Determine effective chunk_size and episode_end -----------------------
    if action_query_frames is not None and len(action_query_frames) > 0:
        # Use actual query indices — more accurate than start_frame + chunk_size
        # because it accounts for clamping to episode boundaries.
        effective_start = action_query_frames[0]
        effective_chunk_size = len(action_query_frames)

        # Padding: any padded step = reject bridge entirely.
        # We do NOT trim to the valid prefix — that would give the model a
        # bridge phrase for a chunk whose last action steps are pad zeros,
        # which would be misleading training signal.
        if action_is_pad is not None and len(action_is_pad) == effective_chunk_size:
            if any(action_is_pad):
                return fallback_phrase

        start = effective_start
        chunk_sz = effective_chunk_size
        ep_end = episode_end_frame
    else:
        # Fallback: use anchor_frame + chunk_size
        if chunk_size is None or chunk_size <= 0:
            return fallback_phrase
        start = anchor_frame
        chunk_sz = chunk_size
        ep_end = episode_end_frame

    # --- Compute bridge info --------------------------------------------------
    bridge_result = compute_bridge_info(
        segments,
        start,
        chunk_sz,
        min_pre_boundary_steps=min_pre_boundary_steps,
        min_post_boundary_steps=min_post_boundary_steps,
        episode_end_frame=ep_end,
    )

    if bridge_result["bridge_valid"]:
        return bridge_result["combined_phrase"]
    else:
        return fallback_phrase
