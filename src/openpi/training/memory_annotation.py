"""Frame -> fixed-compact-Memory lookup for the MoMA-VLA training data.

Design doc: MoMA-VLA §3.4.3 (dataloader lookup) and §3.4.4 (training constraints).

Why this is a separate module rather than an extension of the existing
``annotations_skill`` lookup in ``behavior/learning/datas/dataset.py``:

The existing lookup (``dataset.py`` ``_get_subtask_text``) resolves a frame with
``bisect_left`` over segment *ends* plus a **closed** containment test
``s <= f <= e``.  The fixed-compact-Memory dataset declares the opposite
convention in its own schema -- ``memory_interval_convention == "half-open
[start,end)"``.  Running the closed lookup over half-open intervals shifts every
interval one frame late: for adjacent rows ``[a,b) [b,c)`` the boundary frame
``b`` resolves to the *earlier* row, and a one-frame bridge interval ``[b,b+1)``
becomes reachable only at frame ``b+1``.  16,915 of the 261,353 intervals are
exactly one frame long, so that shift lands squarely on the bridge samples the
design exists to model.  Reusing the old lookup would therefore be silently
wrong rather than merely approximate, which is why this module reimplements it
instead of parameterising the old one.

Three properties of the real data drive the guards below; each was measured over
all 10,000 episodes rather than assumed:

1. **Intervals tile their range exactly.**  0 gaps and 0 overlaps across 251,353
   adjacent pairs, ``memory_idx`` contiguous from 0, no inversions.  So inside
   the covered range a lookup can never legitimately miss, and a miss is a bug
   worth raising on.

2. **The range does not start at frame 0.**  ``first_start != 0`` in 3,946 of
   10,000 episodes (max 1,857).  A bare ``bisect_right(starts, f) - 1`` yields
   ``-1`` for a frame before the range, and ``rows[-1]`` silently returns the
   *last* interval -- attaching the end-of-episode Memory to opening frames with
   no error.  The doc's pseudocode guards this with ``if i < 0: raise``; this
   module keeps that guard and additionally refuses negative frame indices.

3. **Annotations cover a sub-range of the video, and 83 episodes claim frames
   the video does not have.**  Annotated frames are 117,747,883 of 119,094,660
   (98.87%); the head is un-annotated in 3,946 episodes and the tail in 7,590.
   In 83 episodes ``last_end`` exceeds the LeRobot episode length (worst case by
   784 frames).  Hence :meth:`MemoryIntervalIndex.sampling_range` intersects the
   annotated range with the real episode length: sampling from the intersection
   is what makes "a miss is a hard error" compatible with the data, instead of
   crashing on the 1.13% of frames that are legitimately unannotated.
"""

from __future__ import annotations

import bisect
import dataclasses
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "b1k_fixed_compact_memory_v1"
INTERVAL_CONVENTION = "half-open [start,end)"

#: Field order of the planner CE target, design doc §3.4.4.  The supervision
#: order is fixed; it is not a presentation choice.
PLANNER_TARGET_FIELDS: tuple[tuple[str, str], ...] = (
    ("Memory", "fixed_compact_memory"),
    ("Primitive", "current_primitive"),
    ("Skill", "current_skill"),
    ("Next skill", "next_skill"),
    ("Next primitive", "next_primitive"),
)

#: Trailing line of ``model_target_text``.  It is a sequence marker for
#: end-to-end checks, NOT text to be encoded a second time (design doc §3.4.3).
ACTION_QUERY_LINE = "Action Query:"


class MemoryAnnotationError(ValueError):
    """Raised instead of degrading silently.

    Every failure mode this module can hit has a plausible-looking silent
    counterpart -- a wrapped ``rows[-1]``, a clamped frame, a fabricated
    all-zero conditioning segment.  Those produce a trainable batch and a
    loss curve, so they surface as "the model is bad at next-skill
    prediction" rather than as a defect.  Raising is the whole point.
    """


def _require_frame_index(frame_idx: Any, *, what: str = "frame_idx") -> int:
    """Reject bools and negatives before they become wrong answers.

    ``bool`` is an ``int`` subclass and ``True == 1``, so a stray boolean would
    silently address frame 1.  A negative index is worse: it is a *valid* Python
    list index, so it returns the wrong interval rather than failing.
    """
    if isinstance(frame_idx, bool):
        raise MemoryAnnotationError(f"{what} must be an int, got bool ({frame_idx!r})")
    if not isinstance(frame_idx, (int,)):
        raise MemoryAnnotationError(f"{what} must be an int, got {type(frame_idx).__name__}")
    if frame_idx < 0:
        raise MemoryAnnotationError(f"{what} must be >= 0, got {frame_idx}")
    return int(frame_idx)


def planner_target_text(row: Mapping[str, Any]) -> str:
    """Build the 5-line planner CE target from the row's content fields.

    Deliberately built from the individual fields rather than by stripping the
    last line off ``model_target_text``: the field-wise form cannot silently
    inherit a schema change in the composed string.  :func:`check_row_consistency`
    then cross-checks the two against each other, so a drift is caught rather
    than picked.
    """
    lines = []
    for label, key in PLANNER_TARGET_FIELDS:
        value = row.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise MemoryAnnotationError(
                f"planner target field {key!r} (label {label!r}) is empty or missing; "
                "an empty field would train the model to emit an empty section"
            )
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def check_row_consistency(row: Mapping[str, Any]) -> None:
    """Assert the field-wise target agrees with the stored ``model_target_text``.

    ``model_target_text`` is 6 lines, the 6th being the literal
    ``Action Query:`` marker (verified: 261,353/261,353 rows, no trailing
    content).  Dropping that line must reproduce the field-wise target exactly.
    A mismatch means the dataset schema and this reader disagree, which is
    precisely the condition that must not be resolved by preferring one of them.
    """
    stored = row.get("model_target_text")
    if stored is None:
        return  # field is optional for synthetic rows in tests
    head, _, last = str(stored).rpartition("\n")
    if last.strip() != ACTION_QUERY_LINE:
        raise MemoryAnnotationError(
            f"model_target_text must end with the literal {ACTION_QUERY_LINE!r} line, "
            f"got {last!r}. Refusing to guess which lines are the CE target."
        )
    expected = planner_target_text(row)
    if head != expected:
        raise MemoryAnnotationError(
            "model_target_text minus its Action Query line does not match the "
            "field-wise planner target.\n"
            f"  from fields: {expected!r}\n"
            f"  from stored: {head!r}"
        )


@dataclasses.dataclass(frozen=True)
class MemoryRow:
    """One resolved memory interval, carrying only what training consumes."""

    memory_idx: int
    start: int
    end: int
    previous_memory_text: str
    planner_target_text: str
    transition_type: str

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise MemoryAnnotationError(
                f"interval {self.memory_idx} is empty or inverted: [{self.start}, {self.end})"
            )


class MemoryIntervalIndex:
    """Half-open interval index over one episode's ``memory_annotation``.

    Built once per episode and cached by the caller; the design doc calls this
    out explicitly because rebuilding it per sampled frame would dominate the
    data pipeline cost.
    """

    __slots__ = ("_rows", "_starts", "_first_start", "_last_end", "_episode_id")

    def __init__(
        self,
        intervals: Sequence[Mapping[str, Any]],
        *,
        episode_id: str = "<unknown>",
        validate_rows: bool = True,
    ) -> None:
        if not intervals:
            raise MemoryAnnotationError(f"episode {episode_id} has an empty memory_annotation")
        self._episode_id = episode_id
        rows: list[MemoryRow] = []
        prev_end: int | None = None
        for position, raw in enumerate(intervals):
            duration = raw.get("frame_duration")
            if not (isinstance(duration, (list, tuple)) and len(duration) == 2):
                raise MemoryAnnotationError(
                    f"episode {episode_id} interval at position {position}: "
                    f"frame_duration must be a 2-element [start, end), got {duration!r}"
                )
            start, end = int(duration[0]), int(duration[1])
            memory_idx = int(raw.get("memory_idx", position))
            if memory_idx != position:
                raise MemoryAnnotationError(
                    f"episode {episode_id}: memory_idx {memory_idx} is not contiguous "
                    f"from 0 (expected {position}). The index assumes positional order."
                )
            if prev_end is not None and start != prev_end:
                # Measured: 0 gaps and 0 overlaps over 251,353 adjacent pairs.
                # So this is not a tolerated data shape -- it is a regression.
                kind = "gap" if start > prev_end else "overlap"
                raise MemoryAnnotationError(
                    f"episode {episode_id}: {kind} between interval {position - 1} "
                    f"(ends {prev_end}) and interval {position} (starts {start}). "
                    "Intervals must tile their range exactly."
                )
            previous_memory = raw.get("previous_fixed_compact_memory")
            if previous_memory is None or not str(previous_memory).strip():
                raise MemoryAnnotationError(
                    f"episode {episode_id} interval {position}: "
                    "previous_fixed_compact_memory is empty; the prefix would carry no memory"
                )
            if validate_rows:
                check_row_consistency(raw)
            rows.append(
                MemoryRow(
                    memory_idx=memory_idx,
                    start=start,
                    end=end,
                    previous_memory_text=str(previous_memory),
                    planner_target_text=planner_target_text(raw),
                    transition_type=str(raw.get("transition_type", "unknown")),
                )
            )
            prev_end = end
        self._rows = tuple(rows)
        self._starts = tuple(r.start for r in rows)
        self._first_start = rows[0].start
        self._last_end = rows[-1].end

    # -- geometry ---------------------------------------------------------

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def first_start(self) -> int:
        return self._first_start

    @property
    def last_end(self) -> int:
        return self._last_end

    def __len__(self) -> int:
        return len(self._rows)

    def sampling_range(self, episode_length: int | None = None) -> tuple[int, int]:
        """Half-open ``[lo, hi)`` of frames that are safe to sample.

        Intersects the annotated range with the real episode length.  Without
        the intersection, 83 episodes would hand out frames beyond the end of
        the video (worst case 784 frames past it), and uniform sampling over the
        full episode would miss the annotated range for 1.13% of frames -- which
        under a hard-error policy means the run simply dies.
        """
        lo, hi = self._first_start, self._last_end
        if episode_length is not None:
            episode_length = _require_frame_index(episode_length, what="episode_length")
            hi = min(hi, episode_length)
        if hi <= lo:
            raise MemoryAnnotationError(
                f"episode {self._episode_id}: no sampleable frames -- annotated range "
                f"[{self._first_start}, {self._last_end}) intersected with episode length "
                f"{episode_length} is empty"
            )
        return lo, hi

    # -- lookup -----------------------------------------------------------

    def lookup(self, frame_idx: int) -> MemoryRow:
        """Resolve a frame to its interval. Raises rather than guessing.

        ``bisect_right(starts, f) - 1`` is the doc's §3.4.3 form.  The ``i < 0``
        branch is not defensive boilerplate: it is reachable for real data in
        3,946 episodes, and without it ``self._rows[-1]`` would return the final
        interval for an opening frame.
        """
        frame_idx = _require_frame_index(frame_idx)
        position = bisect.bisect_right(self._starts, frame_idx) - 1
        if position < 0:
            raise MemoryAnnotationError(
                f"episode {self._episode_id}: frame {frame_idx} is before the annotated "
                f"range [{self._first_start}, {self._last_end}). Sample from "
                f"sampling_range() instead of the raw episode length."
            )
        row = self._rows[position]
        if not (row.start <= frame_idx < row.end):
            raise MemoryAnnotationError(
                f"episode {self._episode_id}: frame {frame_idx} is not covered by "
                f"interval {row.memory_idx} [{row.start}, {row.end}); annotated range is "
                f"[{self._first_start}, {self._last_end}). Half-open containment failed."
            )
        return row


def build_index_from_episode(
    episode_json: Mapping[str, Any],
    *,
    episode_id: str = "<unknown>",
    require_schema: bool = True,
    validate_rows: bool = True,
) -> MemoryIntervalIndex:
    """Build an index from a parsed episode JSON, checking the schema tag first.

    The schema version is checked rather than trusted because the dataset is
    being regenerated concurrently (phrase-template backfill).  Depending on
    field *shape* alone would let a convention change -- half-open to closed,
    say -- pass through as a plausible off-by-one.
    """
    if require_schema:
        version = episode_json.get("memory_schema_version")
        if version != SCHEMA_VERSION:
            raise MemoryAnnotationError(
                f"episode {episode_id}: memory_schema_version is {version!r}, "
                f"expected {SCHEMA_VERSION!r}"
            )
        convention = episode_json.get("memory_interval_convention")
        if convention != INTERVAL_CONVENTION:
            raise MemoryAnnotationError(
                f"episode {episode_id}: memory_interval_convention is {convention!r}, "
                f"expected {INTERVAL_CONVENTION!r}. This module's lookup is half-open; "
                "a closed-interval dataset would be shifted one frame late."
            )
    return MemoryIntervalIndex(
        episode_json.get("memory_annotation") or (),
        episode_id=episode_id,
        validate_rows=validate_rows,
    )
