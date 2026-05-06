from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.memory_baselines.hamlet_memory import HamletMemoryAdapter
from openpi.models_pytorch.memory_baselines.hamlet_memory import MomentTokenPool
from openpi.models_pytorch.pi0_hamlet import Pi05WithHamlet
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
def minimal_hamlet_model(monkeypatch: pytest.MonkeyPatch) -> Pi05WithHamlet:
    def _fake_embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        batch_size = lang_tokens.shape[0]
        prefix_embs = torch.randn(batch_size, 5, 8)
        prefix_pad_masks = torch.ones(batch_size, 5, dtype=torch.bool)
        prefix_att_masks = torch.zeros(batch_size, 5, dtype=torch.bool)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    monkeypatch.setattr(PI0Pytorch, "embed_prefix", _fake_embed_prefix)

    model = Pi05WithHamlet.__new__(Pi05WithHamlet)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hamlet_num_moment_tokens=4)
    model.training = False
    model.moment_token_pool = MomentTokenPool(num_tokens=4, feature_dim=8)
    model.prefix_summary_proj = torch.nn.Identity()
    model.hamlet_memory = HamletMemoryAdapter(feature_dim=8, num_moment_tokens=4, history_length=2, num_heads=2, num_layers=1)
    model.memory_to_prefix_proj = torch.nn.Identity()
    model.paligemma_with_expert = _DummyPaliGemmaExpert()
    model._active_session_id = None
    model._session_memory_state = {}
    return model


def test_pi0_hamlet_embed_prefix_appends_moment_and_memory_tokens(minimal_hamlet_model: Pi05WithHamlet) -> None:
    lang = torch.ones(2, 3, dtype=torch.long)
    mask = torch.ones(2, 3, dtype=torch.bool)
    embs, pad_masks, att_masks = minimal_hamlet_model.embed_prefix([], [], lang, mask)

    assert embs.shape == (2, 13, 8)
    assert pad_masks.shape == (2, 13)
    assert att_masks.shape == (2, 13)
    assert minimal_hamlet_model.hamlet_memory.get_memory_stats()["history_count"] == 1
    assert minimal_hamlet_model.paligemma_with_expert.forward_calls == 0


def test_pi0_hamlet_session_switch_restores_runtime_state(minimal_hamlet_model: Pi05WithHamlet) -> None:
    lang = torch.ones(1, 3, dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.bool)

    minimal_hamlet_model.set_active_session(1)
    _ = minimal_hamlet_model.embed_prefix([], [], lang, mask)
    assert minimal_hamlet_model.hamlet_memory.get_memory_stats()["history_count"] == 1

    minimal_hamlet_model.set_active_session(2)
    assert minimal_hamlet_model.hamlet_memory.get_memory_stats()["history_count"] == 0

    minimal_hamlet_model.set_active_session(1)
    assert minimal_hamlet_model.hamlet_memory.get_memory_stats()["history_count"] == 1
