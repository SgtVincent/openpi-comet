from typing import Any

import torch
from torch import nn

from openpi.models_pytorch.memory_baselines.hamlet_memory import HamletMemoryAdapter
from openpi.models_pytorch.memory_baselines.hamlet_memory import MomentTokenPool
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class Pi05WithHamlet(PI0Pytorch):
    """Pi0.5 backbone with HAMLET-style moment-token history memory."""

    def __init__(self, config) -> None:
        super().__init__(config)
        feature_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.moment_token_pool = MomentTokenPool(config.hamlet_num_moment_tokens, feature_dim)
        self.hamlet_memory = HamletMemoryAdapter(
            feature_dim=feature_dim,
            num_moment_tokens=config.hamlet_num_moment_tokens,
            history_length=config.hamlet_history_length,
            num_heads=config.hamlet_num_heads,
            num_layers=config.hamlet_num_layers,
            dropout=config.hamlet_dropout,
            gate_init=config.hamlet_gate_init,
        )
        self.memory_to_prefix_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.eye_(self.memory_to_prefix_proj.weight)
        nn.init.zeros_(self.memory_to_prefix_proj.bias)

        self._active_session_id: int | None = None
        self._session_memory_state: dict[int, dict[str, Any]] = {}

    def set_active_session(self, session_id: int | None) -> None:
        if session_id == self._active_session_id:
            return
        if self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.hamlet_memory.get_runtime_state()
        self._active_session_id = session_id
        if session_id is None:
            self.hamlet_memory.reset_runtime_state()
            return
        self.hamlet_memory.set_runtime_state(self._session_memory_state.get(session_id))

    def reset_streaming_state(self, session_id: int | None = None) -> None:
        if session_id is None:
            session_id = self._active_session_id
        if session_id is None:
            self.hamlet_memory.reset_runtime_state()
            return
        self._session_memory_state.pop(session_id, None)
        if session_id == self._active_session_id:
            self.hamlet_memory.reset_runtime_state()

    def clear_session(self, session_id: int) -> None:
        self._session_memory_state.pop(session_id, None)
        if session_id == self._active_session_id:
            self._active_session_id = None
            self.hamlet_memory.reset_runtime_state()

    def _encode_current_moment_tokens(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prefix_embs.shape[0]
        moment_tokens = self.moment_token_pool(batch_size).to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        moment_pad_masks = torch.ones(
            batch_size,
            self.config.hamlet_num_moment_tokens,
            dtype=prefix_pad_masks.dtype,
            device=prefix_pad_masks.device,
        )
        moment_att_masks = torch.zeros(
            batch_size,
            self.config.hamlet_num_moment_tokens,
            dtype=prefix_att_masks.dtype,
            device=prefix_att_masks.device,
        )
        prefix_with_moment_embs = torch.cat([prefix_embs, moment_tokens], dim=1)
        prefix_with_moment_pad_masks = torch.cat([prefix_pad_masks, moment_pad_masks], dim=1)
        prefix_with_moment_att_masks = torch.cat([prefix_att_masks, moment_att_masks], dim=1)
        prefix_att_2d_masks = self.make_att_2d_masks(prefix_with_moment_pad_masks, prefix_with_moment_att_masks)
        prefix_position_ids = torch.cumsum(prefix_with_moment_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_with_moment_embs, None],
            use_cache=False,
        )
        assert outputs is not None
        if isinstance(outputs, list):
            outputs = outputs[0]
        return outputs[:, -self.config.hamlet_num_moment_tokens :, :]

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks)
        current_moment_tokens = self._encode_current_moment_tokens(prefix_embs, prefix_pad_masks, prefix_att_masks)
        memory_tokens = self.hamlet_memory(current_moment_tokens, update_memory=not self.training)
        if self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.hamlet_memory.get_runtime_state()
        memory_tokens = self.memory_to_prefix_proj(memory_tokens).to(dtype=prefix_embs.dtype)

        batch_size = prefix_embs.shape[0]
        moment_input_tokens = self.moment_token_pool(batch_size).to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        extra_pad_masks = torch.ones(
            batch_size,
            self.config.hamlet_num_moment_tokens * 2,
            dtype=prefix_pad_masks.dtype,
            device=prefix_pad_masks.device,
        )
        extra_att_masks = torch.zeros(
            batch_size,
            self.config.hamlet_num_moment_tokens * 2,
            dtype=prefix_att_masks.dtype,
            device=prefix_att_masks.device,
        )

        prefix_embs = torch.cat([prefix_embs, moment_input_tokens, memory_tokens], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, extra_pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, extra_att_masks], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks
