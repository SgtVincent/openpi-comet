"""Tests for the frame -> fixed-compact-Memory lookup.

Each test targets a failure mode whose silent counterpart produces a trainable
batch, so "no exception" is never sufficient evidence here.
"""

from __future__ import annotations

import json
import os
import random

import pytest

from openpi.training import memory_annotation as ma


def _row(idx, start, end, *, mem="m", prev="p", prim="pick up the radio",
         skill="move to the radio", nxt_skill="grasp", nxt_prim="press the radio",
         transition="normal", with_target=True):
    r = {
        "memory_idx": idx,
        "frame_duration": [start, end],
        "previous_fixed_compact_memory": prev,
        "fixed_compact_memory": mem,
        "current_primitive": prim,
        "current_skill": skill,
        "next_skill": nxt_skill,
        "next_primitive": nxt_prim,
        "transition_type": transition,
    }
    if with_target:
        r["model_target_text"] = (
            f"Memory: {mem}\nPrimitive: {prim}\nSkill: {skill}\n"
            f"Next skill: {nxt_skill}\nNext primitive: {nxt_prim}\n{ma.ACTION_QUERY_LINE}"
        )
    return r


def _episode(rows, *, schema=ma.SCHEMA_VERSION, convention=ma.INTERVAL_CONVENTION):
    return {
        "memory_schema_version": schema,
        "memory_interval_convention": convention,
        "memory_annotation": rows,
    }


# --------------------------------------------------------------------------
# half-open semantics -- the defect that motivated a separate module
# --------------------------------------------------------------------------

def test_boundary_frame_belongs_to_the_later_interval():
    """[0,10) [10,11) [11,20): frame 10 must resolve to the SECOND interval.

    The pre-existing closed-interval lookup returns the first interval here.
    That one-frame shift is the whole reason this module exists, so assert the
    exact frame, not merely that some interval came back.
    """
    idx = ma.MemoryIntervalIndex([_row(0, 0, 10, mem="M0"),
                                  _row(1, 10, 11, mem="M1", transition="inter_primitive_bridge"),
                                  _row(2, 11, 20, mem="M2")])
    assert idx.lookup(9).memory_idx == 0
    assert idx.lookup(10).memory_idx == 1, "boundary frame leaked to the earlier interval"
    assert idx.lookup(11).memory_idx == 2
    assert idx.lookup(19).memory_idx == 2


def test_one_frame_bridge_is_reachable_at_exactly_one_frame():
    """A [b, b+1) bridge must be hit by frame b and by nothing else.

    16,915 real intervals are one frame long, 13,384 of them intra-primitive
    skill bridges. Under the closed-interval lookup this interval is reachable
    only at b+1, i.e. paired with the wrong observation.
    """
    idx = ma.MemoryIntervalIndex([_row(0, 0, 10), _row(1, 10, 11), _row(2, 11, 20)])
    hits = [f for f in range(0, 20) if idx.lookup(f).memory_idx == 1]
    assert hits == [10]


def test_frame_at_last_end_is_out_of_range_not_clamped():
    idx = ma.MemoryIntervalIndex([_row(0, 0, 10)])
    with pytest.raises(ma.MemoryAnnotationError, match="not covered"):
        idx.lookup(10)


# --------------------------------------------------------------------------
# shifted origin -- the silent wrap-around
# --------------------------------------------------------------------------

def test_frame_before_shifted_origin_raises_instead_of_wrapping():
    """3,946 real episodes start at first_start != 0.

    A bare ``bisect_right(starts, f) - 1`` gives -1 there, and ``rows[-1]``
    silently returns the LAST interval -- the end-of-episode Memory attached to
    an opening frame. Assert both that it raises and that it does not return the
    final row.
    """
    idx = ma.MemoryIntervalIndex([_row(0, 100, 200, mem="FIRST"),
                                  _row(1, 200, 300, mem="LAST")])
    with pytest.raises(ma.MemoryAnnotationError, match="before the annotated range"):
        idx.lookup(50)
    # positive control: the same index does resolve a frame inside the range
    assert idx.lookup(100).memory_idx == 0
    assert idx.lookup(250).memory_idx == 1


def test_negative_and_bool_frame_indices_are_rejected():
    idx = ma.MemoryIntervalIndex([_row(0, 0, 10)])
    with pytest.raises(ma.MemoryAnnotationError, match="must be >= 0"):
        idx.lookup(-1)          # a valid Python index -> would return a real row
    with pytest.raises(ma.MemoryAnnotationError, match="got bool"):
        idx.lookup(True)        # True == 1, would silently address frame 1


# --------------------------------------------------------------------------
# sampling range -- intersection with the real video length
# --------------------------------------------------------------------------

def test_sampling_range_is_the_annotated_range_when_video_is_longer():
    idx = ma.MemoryIntervalIndex([_row(0, 100, 200)])
    assert idx.sampling_range(episode_length=5000) == (100, 200)


def test_sampling_range_clips_annotation_that_overruns_the_video():
    """83 real episodes have last_end beyond the LeRobot episode length."""
    idx = ma.MemoryIntervalIndex([_row(0, 0, 1000)])
    assert idx.sampling_range(episode_length=800) == (0, 800)


def test_sampling_range_raises_when_intersection_is_empty():
    idx = ma.MemoryIntervalIndex([_row(0, 900, 1000)])
    with pytest.raises(ma.MemoryAnnotationError, match="no sampleable frames"):
        idx.sampling_range(episode_length=500)


def test_every_frame_in_sampling_range_resolves():
    """Given 0 gaps / 0 overlaps, the range must be fully covered.

    This is the property that makes 'a miss is a hard error' safe.
    """
    idx = ma.MemoryIntervalIndex([_row(0, 7, 9), _row(1, 9, 10), _row(2, 10, 25)])
    lo, hi = idx.sampling_range(episode_length=1000)
    assert [idx.lookup(f).memory_idx for f in range(lo, hi)] == [0, 0, 1] + [2] * 15


# --------------------------------------------------------------------------
# structural invariants
# --------------------------------------------------------------------------

def test_gap_between_intervals_is_rejected():
    with pytest.raises(ma.MemoryAnnotationError, match="gap between"):
        ma.MemoryIntervalIndex([_row(0, 0, 10), _row(1, 12, 20)])


def test_overlap_between_intervals_is_rejected():
    with pytest.raises(ma.MemoryAnnotationError, match="overlap between"):
        ma.MemoryIntervalIndex([_row(0, 0, 10), _row(1, 8, 20)])


def test_non_contiguous_memory_idx_is_rejected():
    with pytest.raises(ma.MemoryAnnotationError, match="not contiguous"):
        ma.MemoryIntervalIndex([_row(0, 0, 10), _row(5, 10, 20)])


def test_inverted_and_empty_intervals_are_rejected():
    with pytest.raises(ma.MemoryAnnotationError, match="empty or inverted"):
        ma.MemoryIntervalIndex([_row(0, 10, 5)])
    with pytest.raises(ma.MemoryAnnotationError, match="empty or inverted"):
        ma.MemoryIntervalIndex([_row(0, 10, 10)])


def test_empty_memory_annotation_is_rejected():
    with pytest.raises(ma.MemoryAnnotationError, match="empty memory_annotation"):
        ma.MemoryIntervalIndex([])


def test_wrong_schema_version_and_convention_are_rejected():
    rows = [_row(0, 0, 10)]
    with pytest.raises(ma.MemoryAnnotationError, match="memory_schema_version"):
        ma.build_index_from_episode(_episode(rows, schema="something_else"))
    with pytest.raises(ma.MemoryAnnotationError, match="half-open"):
        ma.build_index_from_episode(_episode(rows, convention="closed [start,end]"))
    # positive control: the correct tags build fine
    assert len(ma.build_index_from_episode(_episode(rows))) == 1


# --------------------------------------------------------------------------
# planner target text
# --------------------------------------------------------------------------

def test_planner_target_has_five_lines_in_the_fixed_order():
    text = ma.planner_target_text(_row(0, 0, 10))
    lines = text.split("\n")
    assert [l.split(":")[0] for l in lines] == [
        "Memory", "Primitive", "Skill", "Next skill", "Next primitive"]


def test_action_query_is_never_part_of_the_ce_target():
    """§3.4.3: the Action Query marker must not be encoded as text again."""
    text = ma.planner_target_text(_row(0, 0, 10))
    assert "Action Query" not in text
    row = ma.MemoryIntervalIndex([_row(0, 0, 10)]).lookup(0)
    assert "Action Query" not in row.planner_target_text


def test_empty_content_field_is_rejected():
    bad = _row(0, 0, 10, with_target=False)
    bad["next_skill"] = "   "
    with pytest.raises(ma.MemoryAnnotationError, match="next_skill"):
        ma.planner_target_text(bad)


def test_consistency_check_catches_drift_between_fields_and_stored_text():
    row = _row(0, 0, 10)
    row["model_target_text"] = row["model_target_text"].replace("Next skill:", "Nxt skill:")
    with pytest.raises(ma.MemoryAnnotationError, match="does not match"):
        ma.check_row_consistency(row)


def test_consistency_check_requires_the_action_query_terminator():
    row = _row(0, 0, 10)
    row["model_target_text"] = row["model_target_text"].rsplit("\n", 1)[0]
    with pytest.raises(ma.MemoryAnnotationError, match="must end with"):
        ma.check_row_consistency(row)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------

DATA_ROOT = ("/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos"
             "/derived/fixed_compact_memory_annotations")


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT), reason="dataset not mounted")
def test_real_episodes_across_all_tasks_build_and_resolve():
    """Cross-offset sample: these files are task/episode ordered, so head -N
    would draw everything from one task (a previous audit got a false zero that
    way). One episode from every task directory, at a task-dependent offset."""
    tasks = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith("task-"))
    assert len(tasks) == 50, f"expected 50 task dirs, found {len(tasks)}"
    rng = random.Random(20260907)
    checked = 0
    for t in tasks:
        eps = sorted(f for f in os.listdir(os.path.join(DATA_ROOT, t)) if f.endswith(".json"))
        path = os.path.join(DATA_ROOT, t, rng.choice(eps))
        with open(path) as f:
            episode = json.load(f)
        idx = ma.build_index_from_episode(episode, episode_id=path)
        lo, hi = idx.sampling_range(episode_length=None)
        # sample interior frames plus every interval boundary
        probes = {lo, hi - 1}
        probes.update(min(r.start, hi - 1) for r in idx._rows)
        probes.update(rng.randrange(lo, hi) for _ in range(20))
        for f in sorted(probes):
            row = idx.lookup(f)
            assert row.start <= f < row.end
            assert "Action Query" not in row.planner_target_text
            assert row.previous_memory_text
        checked += 1
    assert checked == 50


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT), reason="dataset not mounted")
def test_real_data_first_start_is_sometimes_nonzero():
    """Guards the guard: if this ever became 0 everywhere, the wrap-around test
    above would still pass while protecting nothing. Verified globally as
    3,946/10,000, so a 50-task sample must contain at least one."""
    tasks = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith("task-"))
    shifted = 0
    for t in tasks:
        eps = sorted(f for f in os.listdir(os.path.join(DATA_ROOT, t)) if f.endswith(".json"))
        for fn in eps[:4]:
            with open(os.path.join(DATA_ROOT, t, fn)) as f:
                episode = json.load(f)
            if episode["memory_annotation"][0]["frame_duration"][0] != 0:
                shifted += 1
    assert shifted > 0, "no shifted-origin episode in sample; the i<0 guard is untested on real data"


# --------------------------------------------------------------------------
# BehaviorLeRobotDataset producer integration
# --------------------------------------------------------------------------


def _memory_dataset_shell(indices, lengths):
    """Build only the state needed by the production chunk filter."""
    from types import SimpleNamespace

    from behavior.learning.datas.dataset import BehaviorLeRobotDataset
    from behavior.learning.datas.dataset import MEMORY_SUBTASK_SOURCE

    dataset = object.__new__(BehaviorLeRobotDataset)
    dataset.subtask_source = MEMORY_SUBTASK_SOURCE
    dataset.episodes = list(lengths)
    dataset.meta = SimpleNamespace(episodes={ep: {"length": length} for ep, length in lengths.items()})
    dataset._memory_index_for_episode = lambda ep: indices[ep]
    return dataset


def test_dataset_memory_chunk_filter_preserves_global_and_local_coordinates():
    """Dead chunks are dropped and straddling chunks are clipped before sharding.

    Episode 10 has a shifted annotation origin and its first 250-frame chunk is
    entirely dead. Episode 20's annotation overruns its 200-frame video, so the
    sampling range must be clipped to the video while the returned tuple keeps a
    global start/end and an episode-local anchor.
    """
    indices = {
        10: ma.MemoryIntervalIndex([_row(0, 260, 280)]),
        20: ma.MemoryIntervalIndex([_row(0, 50, 300)]),
    }
    dataset = _memory_dataset_shell(indices, {10: 300, 20: 200})

    chunks = dataset._get_keyframe_chunk_indices(chunk_size=250)

    assert chunks == [
        (260, 280, 260),  # episode 10: shifted origin, first raw chunk dropped
        (350, 500, 50),   # episode 20: base=300, local [50, 200)
    ]
    assert dataset.memory_chunk_stats() == {
        "memory_chunks_kept": 2,
        "memory_dead_chunks_dropped": 1,
        "memory_chunks_clipped": 2,
    }


def test_dataset_memory_chunk_filter_reuses_one_bisect_key(monkeypatch):
    """The 10k-episode bisect key must not be rebuilt for every chunk.

    Full data has about 481k chunks. Rebuilding a 10k-element offsets list in
    that loop performs about 4.8 billion element visits during Dataset startup.
    Recording object identity makes this regression fail deterministically,
    without relying on a timing threshold.
    """
    import behavior.learning.datas.dataset as dataset_module

    indices = {
        10: ma.MemoryIntervalIndex([_row(0, 0, 300)]),
        20: ma.MemoryIntervalIndex([_row(0, 0, 200)]),
    }
    dataset = _memory_dataset_shell(indices, {10: 300, 20: 200})
    original = dataset_module.bisect.bisect_right
    seen_keys = []

    def recording_bisect_right(keys, value, *args, **kwargs):
        seen_keys.append(keys)
        return original(keys, value, *args, **kwargs)

    monkeypatch.setattr(dataset_module.bisect, "bisect_right", recording_bisect_right)
    dataset._get_keyframe_chunk_indices(chunk_size=125)

    assert len(seen_keys) == 5
    assert all(keys is seen_keys[0] for keys in seen_keys), (
        "episode offsets were rebuilt inside the chunk loop"
    )


def test_dataset_memory_chunk_filter_fails_if_every_chunk_is_dead():
    index = ma.MemoryIntervalIndex([_row(0, 100, 200)])
    dataset = _memory_dataset_shell({10: index}, {10: 250})

    with pytest.raises(RuntimeError, match="every one of the 1 chunks was dropped"):
        dataset._restrict_chunks_to_memory_coverage([(0, 50, 0)], [250])
