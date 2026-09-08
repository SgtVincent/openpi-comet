"""Planner anchors and held-memory alignment for MoMA-VLA training.

Design doc §4.1: Planner anchors are the periodic anchors plus the ground-truth
transition anchors, and a non-anchor action sample must be conditioned on the
memory from the most recent anchor -- not on its own frame's memory. That is what
reproduces the deployment condition, where the Action Expert runs on memory that
is up to K-1 chunks stale (§2.3, §7).

Training in frame space, deployment in chunk space
-------------------------------------------------
K is a number of action chunks, but training samples frames, so placing anchors
requires knowing how many frames one chunk covers. Measured, not assumed:

- The model's ``action_horizon`` is 32 in all seven ``Pi05SubtaskConfig``
  instantiations, which override the class default of 50.
- ``behavior_dataset.py`` builds ``delta_timestamps`` as ``[t / 30.0 for t in
  range(action_horizon)]``, so at 30 fps a chunk spans exactly
  ``action_horizon`` consecutive frames. Two independent paths, same 32.
- On the serving side ``eval_b1k_wrapper.py`` pops from a queue filled with
  ``target_joint_positions[: max_len]`` and only calls the policy again once the
  queue empties, with ``max_len = 32``. So 32 env steps per inference.

Hence 32 -- but it is a *configured* value here, never a literal in the logic,
because it is held in place only by every entry point remembering to set
``control_mode``. See :func:`check_chunk_consumption_assumptions`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Iterable

#: Frames consumed per policy call. Measured as 32 on both the training and the
#: serving side; see the module docstring. Deliberately NOT named
#: ``action_horizon``: that name means three different things on this path (model
#: chunk length 32, wrapper ensemble depth 5, wrapper ``max_len`` 32) and reading
#: the wrong one yields 5.
DEFAULT_FRAMES_PER_CHUNK = 32

#: The only ``control_mode`` under which frames-per-chunk equals ``max_len``.
REQUIRED_CONTROL_MODE = "receeding_horizon"

#: Modes whose consumption is NOT a fixed stride, so anchor alignment is invalid.
CONTROL_MODE_CONSUMPTION = {
    "temporal_ensemble": 1,   # re-infers every step and blends up to 5 chunks
    "receeding_horizon": None,  # == max_len
    "receeding_temporal": 5,  # blends up to 3 chunks
}


class AnchorScheduleError(ValueError):
    """Raised rather than silently producing a shifted anchor schedule."""


def _require_positive_int(value, name: str) -> int:
    """Reject bool and non-positive values.

    ``bool`` is an ``int`` subclass and ``True == 1``, so ``planner_stride=True``
    would silently select the K=1 arm -- the baseline -- while the config reads as
    if an experiment were configured.
    """
    if isinstance(value, bool):
        raise AnchorScheduleError(f"{name} must be an int, got bool ({value!r})")
    if not isinstance(value, int):
        raise AnchorScheduleError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise AnchorScheduleError(f"{name} must be >= 1, got {value}")
    return value


def _require_non_negative_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise AnchorScheduleError(f"{name} must be an int, got bool ({value!r})")
    if not isinstance(value, int):
        raise AnchorScheduleError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        # -1 % 5 == 4 in Python, so a negative index does not fail: it produces a
        # plausible-looking but wrong position in the schedule.
        raise AnchorScheduleError(f"{name} must be >= 0, got {value}")
    return value


def check_chunk_consumption_assumptions(
    *,
    control_mode: str | None,
    wrapper_max_len: int | None,
    model_action_horizon: int | None,
    frames_per_chunk: int,
) -> None:
    """Fail loudly on the four ways frames-per-chunk silently stops being true.

    Each of these currently holds only because every entry point sets it, so none
    of them is guaranteed by the code. Anchor alignment is built on all four.
    """
    # 1. control_mode must be explicit and must be the fixed-stride mode. The
    #    class default is temporal_ensemble, which re-infers every step and blends
    #    overlapping chunks -- consumption stops being an integer stride at all.
    if control_mode is None:
        raise AnchorScheduleError(
            "control_mode was not passed. The wrapper's class default is "
            f"{'temporal_ensemble'!r}, under which the policy is re-invoked every step "
            "and each action blends several overlapping chunks, so anchor alignment "
            f"is meaningless. Pass control_mode={REQUIRED_CONTROL_MODE!r} explicitly."
        )
    if control_mode != REQUIRED_CONTROL_MODE:
        implied = CONTROL_MODE_CONSUMPTION.get(control_mode, "unknown")
        raise AnchorScheduleError(
            f"control_mode={control_mode!r} consumes {implied} action(s) per inference, "
            f"not frames_per_chunk={frames_per_chunk}. Anchor alignment requires "
            f"{REQUIRED_CONTROL_MODE!r}."
        )
    # 2. max_len and the model's chunk length are independent knobs that merely
    #    happen to both be 32. If the model emits 50 and max_len stays 32, 18
    #    actions are dropped by a slice with no warning anywhere.
    if wrapper_max_len is not None and model_action_horizon is not None:
        if wrapper_max_len != model_action_horizon:
            raise AnchorScheduleError(
                f"wrapper max_len={wrapper_max_len} != model action_horizon="
                f"{model_action_horizon}. The serving queue is filled with "
                "target_joint_positions[:max_len], so a mismatch silently discards "
                f"{abs(model_action_horizon - wrapper_max_len)} action(s) per chunk."
            )
    # 3. frames_per_chunk must agree with what serving actually consumes.
    effective = wrapper_max_len if wrapper_max_len is not None else model_action_horizon
    if effective is not None and effective != frames_per_chunk:
        raise AnchorScheduleError(
            f"frames_per_chunk={frames_per_chunk} but serving consumes {effective} "
            "actions per inference; anchors would be placed at the wrong spacing."
        )
    # 4. The client-side skip_intermediate_obs_in_chunk flag is intentionally not
    #    consulted: it only suppresses network round-trips and rendering for
    #    intermediate steps. Inference count is decided by the server-side queue,
    #    so folding it in here would double-count.


@dataclasses.dataclass(frozen=True)
class AnchorSchedule:
    """Which frames are Planner anchors, and which memory a frame must use.

    ``stride`` is K in chunks; ``frames_per_chunk`` converts to frame space.
    ``origin`` is the first annotated frame, because 3,946 of 10,000 episodes do
    not start at frame 0 and anchors must be counted from where supervision
    begins rather than from zero.
    """

    stride: int
    frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK
    origin: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(self.stride, "planner_stride")
        _require_positive_int(self.frames_per_chunk, "frames_per_chunk")
        _require_non_negative_int(self.origin, "origin")

    @property
    def frames_per_planner_period(self) -> int:
        return self.stride * self.frames_per_chunk

    def chunk_index(self, frame_idx: int) -> int:
        """Chunk ordinal of a frame, counted from ``origin``."""
        _require_non_negative_int(frame_idx, "frame_idx")
        if frame_idx < self.origin:
            raise AnchorScheduleError(
                f"frame {frame_idx} is before origin {self.origin}; a negative chunk "
                "index would wrap under % and give a plausible but wrong schedule"
            )
        return (frame_idx - self.origin) // self.frames_per_chunk

    def is_periodic_anchor(self, frame_idx: int) -> bool:
        return self.chunk_index(frame_idx) % self.stride == 0

    def anchor_frame_for(self, frame_idx: int, transition_starts: Iterable[int] = ()) -> int:
        """First frame of the memory this sample must be conditioned on.

        The later of (a) the most recent periodic anchor and (b) the most recent
        ground-truth transition at or before this frame. Taking the later of the
        two is what makes K=1 degrade exactly to per-chunk regeneration while
        still letting a real transition refresh memory early (§4.1).
        """
        _require_non_negative_int(frame_idx, "frame_idx")
        c = self.chunk_index(frame_idx)
        periodic = self.origin + (c - c % self.stride) * self.frames_per_chunk
        best = periodic
        for start in transition_starts:
            s = _require_non_negative_int(start, "transition_start")
            if periodic <= s <= frame_idx:
                best = max(best, s)
        return best

    def is_anchor(self, frame_idx: int, transition_starts: Iterable[int] = ()) -> bool:
        return self.anchor_frame_for(frame_idx, transition_starts) == frame_idx

    def describe(self) -> dict:
        """Values for the run manifest.

        The note is carried in the artefact on purpose: 32 is only true while
        every entry point sets control_mode, and a manifest that merely records
        "32" would read as a property of the system.
        """
        return {
            "planner_stride_chunks": self.stride,
            "frames_per_chunk": self.frames_per_chunk,
            "frames_per_planner_period": self.frames_per_planner_period,
            "anchor_origin": self.origin,
            "frames_per_chunk_note": (
                "measured as 32 on both the training side (action_horizon override "
                "plus delta_timestamps construction) and the serving side "
                "(eval_b1k_wrapper max_len); holds only while "
                f"control_mode == {REQUIRED_CONTROL_MODE!r}"
            ),
        }

@dataclasses.dataclass(frozen=True)
class MixedAnchorDecision:
    anchor_kind: str
    selected_stride: int
    target_chunk_index: int
    anchor_chunk_index: int
    anchor_frame: int
    chunk_lag: int
    frame_lag: int


@dataclasses.dataclass(frozen=True)
class MixedStrideSelector:
    """Stable per-action-chunk MIX-C selection.

    The key is episode-local ``chunk_index`` rather than frame or call order, so
    every frame in one action chunk selects the same K across workers, retries,
    and Python hash seeds.
    """

    weights: tuple[tuple[int, float], ...]
    seed: int

    def __post_init__(self) -> None:
        validate_planner_stride_spec(1, self.weights)
        _require_non_negative_int(self.seed, "mixed_stride_seed")

    def select(self, *, episode_index: int, chunk_index: int) -> int:
        _require_non_negative_int(episode_index, "episode_index")
        _require_non_negative_int(chunk_index, "chunk_index")
        payload = json.dumps(
            [self.seed, episode_index, chunk_index], separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        draw = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64
        total = math.fsum(float(weight) for _, weight in self.weights)
        threshold = draw * total
        cumulative = 0.0
        for stride, weight in self.weights:
            cumulative += float(weight)
            if threshold < cumulative:
                return int(stride)
        return int(self.weights[-1][0])

    def decision_for(
        self,
        *,
        episode_index: int,
        frame_idx: int,
        origin: int,
        frames_per_chunk: int,
    ) -> MixedAnchorDecision:
        schedule = AnchorSchedule(stride=1, frames_per_chunk=frames_per_chunk, origin=origin)
        target_chunk = schedule.chunk_index(frame_idx)
        if target_chunk == 0:
            return MixedAnchorDecision("initial", 0, 0, -1, origin, 0, 0)
        stride = self.select(episode_index=episode_index, chunk_index=target_chunk)
        # Option 1b pairs consecutive periodic planner ticks: target tick n is
        # conditioned on the committed Memory from the previous tick. This makes
        # K=1 one action chunk stale rather than a current-frame shortcut.
        anchor_chunk = ((target_chunk - 1) // stride) * stride
        anchor_frame = origin + anchor_chunk * frames_per_chunk
        return MixedAnchorDecision(
            "periodic",
            stride,
            target_chunk,
            anchor_chunk,
            anchor_frame,
            target_chunk - anchor_chunk,
            frame_idx - anchor_frame,
        )

    def describe(self) -> dict:
        return {
            "mode": "per_action_chunk_stable_blake2b",
            "seed": self.seed,
            "weights": [[int(k), float(w)] for k, w in self.weights],
            "key_fields": ["seed", "episode_index", "episode_local_chunk_index"],
        }


def validate_planner_stride_spec(
    planner_stride: int,
    planner_stride_weights: "tuple[tuple[int, float], ...] | None" = None,
) -> None:
    """Validate a planner-stride spec at CONFIG BUILD time.

    Two failure modes are specifically excluded, because both produce a wrong
    schedule instead of an error:

    * ``bool``: Python has ``True == 1``, so ``planner_stride=True`` would
      silently select the K=1 experiment arm.
    * ``<= 0``: ``-1 % 5 == 4``, so a negative stride yields a plausible-looking
      anchor schedule rather than a crash.

    ``planner_stride_weights`` configures the mixed-K arm. Selection itself is
    owned by :class:`MixedStrideSelector`; this function validates the shared
    config without silently normalising or dropping entries.
    """
    if isinstance(planner_stride, bool):
        raise TypeError(
            f"planner_stride must be an int, got bool ({planner_stride!r}). "
            "Python treats True as 1, so this would silently select the K=1 arm."
        )
    if not isinstance(planner_stride, int):
        raise TypeError(f"planner_stride must be an int, got {type(planner_stride).__name__}")
    if planner_stride <= 0:
        raise ValueError(
            f"planner_stride must be >= 1, got {planner_stride}. Negative strides do not "
            "raise later: -1 % 5 == 4, which produces a wrong-but-plausible schedule."
        )
    if planner_stride_weights is None:
        return

    if not planner_stride_weights:
        raise ValueError("planner_stride_weights was provided but empty; pass None to disable it")
    seen = set()
    total = 0.0
    for entry in planner_stride_weights:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise TypeError(f"planner_stride_weights entries must be (K, weight) pairs, got {entry!r}")
        k, w = entry
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"mixed-K keys must be ints, got {k!r}")
        if k <= 0:
            raise ValueError(f"mixed-K keys must be >= 1, got {k}")
        if k in seen:
            raise ValueError(f"duplicate K in planner_stride_weights: {k}")
        seen.add(k)
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            raise TypeError(f"mixed-K weights must be numbers, got {w!r}")
        if w < 0:
            raise ValueError(f"mixed-K weights must be non-negative, got {w}")
        total += float(w)
    if total <= 0:
        raise ValueError("planner_stride_weights sum to 0, which selects nothing")
    if not math.isfinite(total):
        raise ValueError(f"planner_stride_weights sum must be finite, got {total}")
