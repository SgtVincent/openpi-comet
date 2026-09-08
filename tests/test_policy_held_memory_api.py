from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from openpi.policies.policy import Policy


class _Tokenizer:
    def tokenize_prompt(self, prompt, state, previous_memory=None):
        # Encode the observed state and previous text into the prompt token, so
        # tests can prove the latest observation / previous text reached Planner.
        marker = int(np.asarray(state).reshape(-1)[0])
        prev = 1 if previous_memory else 0
        return np.asarray([marker, prev], np.int32), np.asarray([True, True])


class _Observation:
    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, v)


class _Model:
    def __init__(self):
        self.config = types.SimpleNamespace(max_token_len=8, subtask_max_len=8)
        self.predict_calls = 0
        self.build_calls = []
        self.sample_calls = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def predict_subtask_tokens(self, observation):
        self.predict_calls += 1
        # The marker proves latest state/prompt observation reached the Planner.
        marker = int(observation.tokenized_prompt[0, 0].item())
        return torch.tensor([[100 + marker, 1]], dtype=torch.int32)

    def decode_subtask_tokens(self, tokens):
        return [f"Memory: {int(tokens[0, 0])}"]

    def build_hierarchical_observation(self, observation, tokens):
        self.build_calls.append(tokens.detach().clone())
        observation.subtask_tokens = tokens
        observation.subtask_mask = torch.ones_like(tokens, dtype=torch.bool)
        return observation

    def sample_actions(self, _device, observation, noise=None, **_kwargs):
        self.sample_calls.append(observation.subtask_tokens.detach().clone())
        value = float(observation.subtask_tokens[0, 0].item())
        return torch.full((1, 32, 23), value, dtype=torch.float32)


def _policy(monkeypatch):
    import openpi.policies.policy as mod

    monkeypatch.setattr(mod._model.Observation, "from_dict", staticmethod(lambda data: _Observation(data)))
    monkeypatch.setattr(mod._tokenizer, "SubtaskTokenizer", lambda **_kwargs: _Tokenizer())
    model = _Model()
    policy = Policy(
        model,
        transforms=[],
        memory_post_transforms=[],
        output_transforms=[],
        is_pytorch=True,
        pytorch_device="cpu",
        held_memory_enabled=True,
    )
    return policy, model


def _obs(state=0):
    return {
        "prompt": "task",
        "state": np.asarray([state], np.float32),
        "image": {"base": np.zeros((2, 2, 3), np.uint8)},
        "image_mask": {"base": np.asarray(True)},
    }


def test_planner_returns_raw_ids_and_action_uses_same_current_ids(monkeypatch):
    policy, model = _policy(monkeypatch)
    out = policy.infer_memory_chunk(
        _obs(7), planner_tick=True, held_memory_tokens=None,
        previous_memory_text="previous", chunk_index=0,
    )
    assert model.predict_calls == 1
    assert np.array_equal(out["held_memory_tokens"], np.asarray([107, 1], np.int32))
    assert out["held_memory_text"] == "Memory: 107"
    assert torch.equal(model.build_calls[-1], torch.tensor([[107, 1]], dtype=torch.int32))
    assert torch.equal(model.sample_calls[-1], torch.tensor([[107, 1]], dtype=torch.int32))
    assert np.all(out["actions"] == 107)


def test_fast_tick_never_calls_planner_and_injects_held_ids(monkeypatch):
    policy, model = _policy(monkeypatch)
    held = np.asarray([55, 1], np.int32)
    out = policy.infer_memory_chunk(
        _obs(9), planner_tick=False, held_memory_tokens=held,
        previous_memory_text="old", chunk_index=1,
    )
    assert model.predict_calls == 0
    assert out["held_memory_tokens"] is None
    assert torch.equal(model.build_calls[-1], torch.tensor([[55, 1]], dtype=torch.int32))
    assert np.all(out["actions"] == 55)


def test_all_false_action_memory_mask_fails_closed(monkeypatch):
    policy, model = _policy(monkeypatch)

    def broken(observation, tokens):
        observation.subtask_tokens = tokens
        observation.subtask_mask = torch.zeros_like(tokens, dtype=torch.bool)
        return observation

    model.build_hierarchical_observation = broken
    with pytest.raises(RuntimeError, match="all false"):
        policy.infer_memory_chunk(
            _obs(), planner_tick=False, held_memory_tokens=np.asarray([1], np.int32),
            previous_memory_text="p", chunk_index=1,
        )


def test_gt_override_combination_is_rejected_before_model_calls(monkeypatch):
    policy, model = _policy(monkeypatch)
    with pytest.raises(ValueError, match="explicit subtask_text"):
        policy.infer_memory_chunk(
            {**_obs(), "subtask_text": "oracle"}, planner_tick=True,
            held_memory_tokens=None, previous_memory_text="p", chunk_index=0,
        )
    assert model.predict_calls == 0



def test_real_build_hierarchical_observation_and_action_conditioning_forward():
    """Tiny real PI05 APIs: raw IDs enter a live mask and action conditioning.

    The full SigLIP tower remains large even with dummy Gemma, so this smoke uses
    the real PI05SubtaskPytorch builder and its real conditioned-action routing,
    while replacing only the expensive action expert forward with a shape-valid
    deterministic function.  The separate memory_cache tests exercise the real
    prefix fingerprint and stale-KV rejection.
    """
    import torch.nn as nn
    from openpi.models.pi05_subtask_config import Pi05SubtaskConfig
    from openpi.models_pytorch.pi05_subtask import PI05SubtaskPytorch

    model = PI05SubtaskPytorch.__new__(PI05SubtaskPytorch)
    nn.Module.__init__(model)
    model.config = Pi05SubtaskConfig(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_dim=8, action_horizon=4, max_token_len=16, subtask_max_len=8,
    )
    model._text_tokenizer = types.SimpleNamespace(bos_id=lambda: 2, eos_id=lambda: 1)
    model._last_predicted_subtasks = []
    from openpi.models.model import Observation
    base = Observation(
        images={}, image_masks={}, state=torch.zeros(1, 8),
        subtask_tokens=None, subtask_mask=None,
        subtask_loss_mask=None, subtask_ar_mask=None,
    )
    raw = torch.tensor([[17, 18, 1]], dtype=torch.int32)
    conditioned = model.build_hierarchical_observation(base, raw)
    assert torch.equal(conditioned.subtask_tokens[0, :4], torch.tensor([2, 17, 18, 1], dtype=torch.int32))
    assert conditioned.subtask_mask[0, :4].all()
    assert not conditioned.subtask_mask[0, 4:].any()

    seen = {}
    def fake_conditioned(device, observation, noise=None, num_steps=10):
        seen["tokens"] = observation.subtask_tokens.clone()
        return torch.zeros((1, 4, 8), dtype=torch.float32)
    model._sample_actions_with_conditioning = fake_conditioned
    out = model.sample_actions("cpu", conditioned, num_steps=1)
    assert out.shape == (1, 4, 8)
    assert torch.equal(seen["tokens"], conditioned.subtask_tokens)
