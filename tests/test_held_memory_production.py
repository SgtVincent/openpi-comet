from __future__ import annotations

import copy

import numpy as np
import pytest

from openpi.models.memory_text import INITIAL_PREVIOUS_MEMORY
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper


class _Model:
    def __init__(self):
        self.cleared_sessions = []

    def set_active_session(self, _session_id):
        pass

    def reset_streaming_state(self, _session_id):
        pass

    def clear_session(self, session_id):
        self.cleared_sessions.append(session_id)


class _MemoryPolicy:
    def __init__(self):
        self._model = _Model()
        self.calls = []
        self.planner_calls = 0

    def infer_memory_chunk(
        self,
        obs,
        *,
        planner_tick,
        held_memory_tokens,
        previous_memory_text,
        chunk_index,
        noise=None,
    ):
        self.calls.append(
            {
                "obs": copy.deepcopy(obs),
                "planner_tick": planner_tick,
                "held": None if held_memory_tokens is None else np.asarray(held_memory_tokens).copy(),
                "previous": previous_memory_text,
                "chunk": chunk_index,
            }
        )
        tokens = None
        text = None
        if planner_tick:
            self.planner_calls += 1
            tokens = np.asarray([100 + chunk_index, 1], dtype=np.int32)
            text = f"Memory: chunk {chunk_index}"
        actions = np.stack(
            [np.full((23,), chunk_index * 100 + i, dtype=np.float32) for i in range(32)]
        )
        return {
            "actions": actions,
            "held_memory_tokens": tokens,
            "held_memory_text": text,
        }


def _wrapper(*, k=5):
    policy = _MemoryPolicy()
    wrapper = B1KPolicyWrapper(
        policy,
        task_name="turning_on_radio",
        control_mode="receeding_horizon",
        max_len=32,
        fine_grained_level=0,
        held_memory_enabled=True,
        planner_stride=k,
    )
    wrapper.process_obs = lambda obs: {
        "observation": np.full((1, 3, 2, 2, 3), obs.get("pixel", 0), dtype=np.uint8),
        "proprio": np.zeros((1, 23), dtype=np.float32),
        "env_step": obs["env_step"],
    }
    return wrapper, policy


def test_missing_or_invalid_absolute_env_step_fails_closed():
    w, _ = _wrapper()
    for obs in ({}, {"env_step": True}, {"env_step": np.asarray(0)}, {"env_step": -1}, {"env_step": 1.5}):
        with pytest.raises(ValueError, match="absolute env_step"):
            w.act(obs)


def test_k5_uses_absolute_chunks_and_holds_memory_on_fast_chunks():
    w, p = _wrapper(k=5)
    for step in (0, 32, 64, 96, 128, 160, 320):
        w.act({"env_step": step, "pixel": step // 32})
    assert [c["chunk"] for c in p.calls] == [0, 1, 2, 3, 4, 5, 10]
    assert [c["chunk"] for c in p.calls if c["planner_tick"]] == [0, 5, 10]
    assert p.planner_calls == 3
    assert p.calls[0]["previous"] == INITIAL_PREVIOUS_MEMORY
    assert np.array_equal(p.calls[1]["held"], np.asarray([100, 1], dtype=np.int32))
    assert p.calls[5]["previous"] == "Memory: chunk 0"
    assert w.last_memory_telemetry["planner_calls"] == 3


@pytest.mark.parametrize(("k", "expected"), [(1, 6), (2, 3), (5, 2)])
def test_changing_k_changes_planner_calls(k, expected):
    w, p = _wrapper(k=k)
    for chunk in range(6):
        w.act({"env_step": chunk * 32})
    assert p.planner_calls == expected


def test_same_step_retry_is_idempotent_and_does_not_call_planner():
    w, p = _wrapper()
    a = w.act({"env_step": 0})
    b = w.act({"env_step": 0, "pixel": 99})
    assert len(p.calls) == 1
    assert p.planner_calls == 1
    assert np.array_equal(a, b)
    assert w.last_memory_telemetry["retry"] is True


def test_new_chunk_uses_latest_observation_and_never_queue_as_clock():
    w, p = _wrapper()
    w.act({"env_step": 0, "pixel": 1})
    # Deliberately leave the old 31-action queue populated. Absolute env_step is
    # the clock, so a request for step 32 builds chunk 1 from pixel=2 anyway.
    w.act({"env_step": 32, "pixel": 2})
    assert len(p.calls) == 2
    assert p.calls[0]["obs"]["observation/egocentric_camera"][0, 0, 0] == 1
    assert p.calls[1]["obs"]["observation/egocentric_camera"][0, 0, 0] == 2


def test_reset_and_spawn_clear_but_rotate_preserves_episode_memory():
    w, p = _wrapper()
    w.act({"env_step": 0})
    tokens = w._held_memory_tokens.copy()
    calls = w._memory_rollout.planner_calls

    first = w._last_step_action.copy()
    calls_before_rotate = len(p.calls)
    w.rotate_session()
    assert np.array_equal(w._held_memory_tokens, tokens)
    assert w._memory_rollout.planner_calls == calls
    retry = w.act({"env_step": 0})
    assert np.array_equal(retry, first)
    assert len(p.calls) == calls_before_rotate

    spawned = w.spawn_session()
    assert spawned._held_memory_tokens is None
    assert spawned._last_env_step is None
    assert spawned._memory_rollout.planner_calls == 0

    w.reset()
    assert w._held_memory_tokens is None
    assert w._last_env_step is None
    assert w._memory_rollout.planner_calls == 0


def test_arbitrary_non_decreasing_steps_use_absolute_chunk_and_only_backwards_refused():
    w, p = _wrapper()
    w.act({"env_step": 0})
    assert w.last_policy_inferred is True
    a33 = w.act({"env_step": 33})
    assert w.last_policy_inferred is True  # new absolute chunk
    assert w.last_action_chunk is not None
    # Same absolute chunk, different step: no second model/planner call and no
    # fresh action plan / action_chunk is advertised to the websocket client.
    a34 = w.act({"env_step": 34})
    assert w.last_policy_inferred is False
    assert w.last_action_chunk is None
    assert w.last_memory_telemetry["cached_action"] is True
    assert [c["chunk"] for c in p.calls] == [0, 1]
    assert p.planner_calls == 1
    assert a33[0, 0].item() == 101
    assert a34[0, 0].item() == 102
    w.act({"env_step": 159})  # chunk 4 offset 31
    w.act({"env_step": 160})  # chunk 5 planner
    w.act({"env_step": 320})  # jump to chunk 10 planner
    assert [c["chunk"] for c in p.calls if c["planner_tick"]] == [0, 5, 10]
    with pytest.raises(ValueError, match="backwards"):
        w.act({"env_step": 319})


def test_planner_failure_is_transactional_and_retry_does_not_double_commit():
    w, p = _wrapper()
    original = p.infer_memory_chunk

    def short(*args, **kwargs):
        out = original(*args, **kwargs)
        out["actions"] = out["actions"][:3]
        return out

    p.infer_memory_chunk = short
    for _ in range(2):
        with pytest.raises(ValueError, match="shaped"):
            w.act({"env_step": 0})
        assert w._memory_rollout.planner_calls == 0
        assert w._memory_rollout.cache.generation == 0
        assert w._memory_rollout.stats()["chunks_seen"] == 0
        assert w._held_memory_tokens is None
        assert w._last_env_step is None
    p.infer_memory_chunk = original
    w.act({"env_step": 0})
    assert w._memory_rollout.planner_calls == 1
    assert w._memory_rollout.cache.generation == 1


def test_moma_plus_explicit_gt_subtask_is_rejected():
    w, _ = _wrapper()
    with pytest.raises(ValueError, match="explicit subtask_text"):
        w.act({"env_step": 0, "subtask_text": "oracle plan"})


def test_legacy_wrapper_is_byte_equivalent_and_does_not_require_env_step(monkeypatch):
    class LegacyPolicy:
        def __init__(self):
            self._model = _Model()
            self.calls = 0

        def infer(self, _obs):
            self.calls += 1
            return {"actions": np.arange(5 * 23, dtype=np.float32).reshape(5, 23)}

    p = LegacyPolicy()
    w = B1KPolicyWrapper(
        p,
        task_name="turning_on_radio",
        control_mode="receeding_horizon",
        max_len=3,
        held_memory_enabled=False,
    )
    monkeypatch.setattr(
        w,
        "process_obs",
        lambda _obs: {
            "observation": np.zeros((1, 3, 2, 2, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 23), dtype=np.float32),
        },
    )
    a0, a1, a2 = w.act({}), w.act({}), w.act({})
    assert p.calls == 1
    expected = np.arange(5 * 23, dtype=np.float32).reshape(5, 23)
    assert np.array_equal(a0, expected[0:1])
    assert np.array_equal(a1, expected[1:2])
    assert np.array_equal(a2, expected[2:3])


def test_close_session_is_idempotent_and_does_not_rotate():
    w, _ = _wrapper()
    w.act({"env_step": 0})
    sid = w._session_id
    w.close_session()
    w.close_session()
    assert w._session_id == sid
    assert w.policy._model.cleared_sessions == [sid]
    assert w._held_memory_tokens is None
    assert w._last_env_step is None
    assert not w.action_queue


def test_close_session_without_model_hook_still_clears_wrapper_state():
    w, _ = _wrapper()
    w.act({"env_step": 0})
    w.policy._model = object()
    w.close_session()
    assert w._held_memory_tokens is None
    assert w._session_closed is True


def test_closing_one_session_does_not_change_another():
    root, _ = _wrapper()
    a, b = root.spawn_session(), root.spawn_session()
    a.act({"env_step": 0})
    b.act({"env_step": 0})
    before = b._held_memory_tokens.copy()
    a.close_session()
    assert np.array_equal(b._held_memory_tokens, before)
    assert b._session_closed is False


def test_two_sessions_are_isolated():
    root, _ = _wrapper()
    a = root.spawn_session()
    b = root.spawn_session()
    a.act({"env_step": 0})
    b.act({"env_step": 0})
    a.act({"env_step": 32})
    assert a._last_env_step == 32
    assert b._last_env_step == 0
    a.reset()
    assert a._held_memory_tokens is None
    assert b._held_memory_tokens is not None
