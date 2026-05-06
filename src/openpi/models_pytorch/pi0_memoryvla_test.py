from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.memory_baselines.memoryvla_memory import MemoryVLAModule
from openpi.models_pytorch.pi0_memoryvla import Pi05WithMemoryVLA
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class _DummyForwardModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(_attn_implementation=None)


class _DummyPaliGemmaExpert:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.paligemma = SimpleNamespace(
            language_model=_DummyForwardModel(),
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=8)),
        )

    def forward(self, *, inputs_embeds, **kwargs):
        self.forward_calls += 1
        prefix = inputs_embeds[0]
        return prefix + 1.0, None


@pytest.fixture()
def minimal_memoryvla_model(monkeypatch: pytest.MonkeyPatch) -> Pi05WithMemoryVLA:
    def _fake_embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        batch_size = lang_tokens.shape[0]
        prefix_embs = torch.randn(batch_size, 5, 8)
        prefix_pad_masks = torch.ones(batch_size, 5, dtype=torch.bool)
        prefix_att_masks = torch.zeros(batch_size, 5, dtype=torch.bool)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    monkeypatch.setattr(PI0Pytorch, "embed_prefix", _fake_embed_prefix)

    model = Pi05WithMemoryVLA.__new__(Pi05WithMemoryVLA)
    torch.nn.Module.__init__(model)
    model.training = False
    model.prefix_summary_proj = torch.nn.Identity()
    model.memoryvla = MemoryVLAModule(feature_dim=8, bank_capacity=2, similarity_threshold=0.7)
    model.memory_to_prefix_proj = torch.nn.Identity()
    model.paligemma_with_expert = _DummyPaliGemmaExpert()
    model._active_session_id = None
    model._session_memory_state = {}
    model._last_memory_gate = None
    return model


def test_pi0_memoryvla_embed_prefix_appends_memory_tokens(minimal_memoryvla_model: Pi05WithMemoryVLA) -> None:
    lang = torch.ones(2, 3, dtype=torch.long)
    mask = torch.ones(2, 3, dtype=torch.bool)
    embs, pad_masks, att_masks = minimal_memoryvla_model.embed_prefix([], [], lang, mask)
    assert embs.shape == (2, 6, 8)
    assert pad_masks.shape == (2, 6)
    assert att_masks.shape == (2, 6)
    assert minimal_memoryvla_model.memoryvla.get_memory_stats()["memory_count"] == 1
    assert minimal_memoryvla_model.paligemma_with_expert.forward_calls == 0


def test_pi0_memoryvla_session_switch_restores_state(minimal_memoryvla_model: Pi05WithMemoryVLA) -> None:
    lang = torch.ones(1, 3, dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.bool)
    minimal_memoryvla_model.set_active_session(1)
    _ = minimal_memoryvla_model.embed_prefix([], [], lang, mask)
    assert minimal_memoryvla_model.memoryvla.get_memory_stats()["memory_count"] == 1
    minimal_memoryvla_model.set_active_session(2)
    assert minimal_memoryvla_model.memoryvla.get_memory_stats()["memory_count"] == 0
    minimal_memoryvla_model.set_active_session(1)
    assert minimal_memoryvla_model.memoryvla.get_memory_stats()["memory_count"] == 1
