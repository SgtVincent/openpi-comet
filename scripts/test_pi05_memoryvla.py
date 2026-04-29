"""Lightweight smoke test for Pi05WithMemoryVLA without instantiating full 2B weights."""

from types import SimpleNamespace

import torch

from openpi.models.memoryvla_config import MemoryVLAConfig
from openpi.models_pytorch.memory_baselines.memoryvla_memory import MemoryVLAModule
from openpi.models_pytorch.pi0_memoryvla import Pi05WithMemoryVLA
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import train_config as _train_config


class _DummyForwardModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(_attn_implementation=None)


class _DummyPaliGemmaExpert:
    def __init__(self) -> None:
        self.paligemma = SimpleNamespace(
            language_model=_DummyForwardModel(),
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=8)),
        )

    def forward(self, *, inputs_embeds, **kwargs):
        prefix = inputs_embeds[0]
        if prefix is None:
            return [None, inputs_embeds[1]], None
        return prefix + 1.0, None


class _DummyActionExpert:
    def encode_prefix(self, *, model, images, img_masks, lang_tokens, lang_masks):
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        return {
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
            "past_key_values": None,
        }

    def compute_velocity_train(self, *, model, images, img_masks, lang_tokens, lang_masks, state, x_t, time):
        _ = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        return torch.zeros_like(x_t)

    def compute_velocity_infer(self, *, model, prefix_ctx, state, x_t, time):
        return torch.zeros_like(x_t)


def _fake_embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    batch_size = lang_tokens.shape[0]
    prefix_embs = torch.randn(batch_size, 5, 8)
    prefix_pad_masks = torch.ones(batch_size, 5, dtype=torch.bool)
    prefix_att_masks = torch.zeros(batch_size, 5, dtype=torch.bool)
    return prefix_embs, prefix_pad_masks, prefix_att_masks


def _build_smoke_model() -> Pi05WithMemoryVLA:
    model = Pi05WithMemoryVLA.__new__(Pi05WithMemoryVLA)
    torch.nn.Module.__init__(model)
    model.training = False
    model.prefix_summary_proj = torch.nn.Identity()
    model.memoryvla = MemoryVLAModule(feature_dim=8, bank_capacity=2, similarity_threshold=0.7)
    model.memory_to_prefix_proj = torch.nn.Identity()
    model.paligemma_with_expert = _DummyPaliGemmaExpert()
    model.action_expert = _DummyActionExpert()
    model._active_session_id = None
    model._session_memory_state = {}
    model._last_memory_gate = None
    model.config = SimpleNamespace(action_horizon=32, action_dim=32)
    return model


def main() -> None:
    cfg = _train_config.get_config("pi05_memoryvla_test")
    assert cfg.pytorch_model_name == "pi0_memoryvla"
    assert isinstance(cfg.model, MemoryVLAConfig)

    PI0Pytorch.embed_prefix = _fake_embed_prefix
    model = _build_smoke_model()

    lang = torch.ones(2, 4, dtype=torch.long)
    lang_mask = torch.ones(2, 4, dtype=torch.bool)
    state = torch.randn(2, 32)
    actions = torch.randn(2, 32, 32)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix([], [], lang, lang_mask)
    print("prefix_embs", tuple(prefix_embs.shape))
    print("prefix_pad_masks", tuple(prefix_pad_masks.shape))
    print("prefix_att_masks", tuple(prefix_att_masks.shape))

    model.set_active_session(321)
    _ = model.embed_prefix([], [], lang[:1], lang_mask[:1])
    print("memory_count", model.memoryvla.get_memory_stats()["memory_count"])

    model._preprocess_observation = lambda observation, train: ([], [], lang, lang_mask, state)
    loss = model.forward(SimpleNamespace(state=state), actions)
    sampled_actions = model.sample_actions(torch.device("cpu"), SimpleNamespace(state=state), noise=torch.zeros_like(actions), num_steps=2)
    print("loss", tuple(loss.shape))
    print("sampled_actions", tuple(sampled_actions.shape))


if __name__ == "__main__":
    main()
