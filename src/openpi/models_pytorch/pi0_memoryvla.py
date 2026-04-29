from typing import Any

import torch
from torch import nn

from openpi.models_pytorch.memory_baselines.memoryvla_memory import MemoryVLAModule
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class Pi05WithMemoryVLA(PI0Pytorch):
    """Pi0.5 backbone with a single-stream MemoryVLA-style external memory bank."""

    def __init__(
        self,
        config,
        *,
        action_expert_name: str = "gemma_token",
        action_expert_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, action_expert_name=action_expert_name, action_expert_kwargs=action_expert_kwargs)
        feature_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.memoryvla = MemoryVLAModule(
            feature_dim=feature_dim,
            bank_capacity=config.memoryvla_bank_capacity,
            similarity_threshold=config.memoryvla_similarity_threshold,
            gate_init=config.memoryvla_gate_init,
        )
        self.memory_to_prefix_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.eye_(self.memory_to_prefix_proj.weight)
        nn.init.zeros_(self.memory_to_prefix_proj.bias)
        # Kept for checkpoint compatibility; used to project the current-summary token.
        self.prefix_summary_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.eye_(self.prefix_summary_proj.weight)
        nn.init.zeros_(self.prefix_summary_proj.bias)

        self._active_session_id: int | None = None
        self._session_memory_state: dict[int, dict[str, Any]] = {}
        self._last_memory_gate: torch.Tensor | None = None

    def set_active_session(self, session_id: int | None) -> None:
        if session_id == self._active_session_id:
            return
        if self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.memoryvla.get_runtime_state()
        self._active_session_id = session_id
        if session_id is None:
            self.memoryvla.reset_runtime_state()
            return
        self.memoryvla.set_runtime_state(self._session_memory_state.get(session_id))

    def reset_streaming_state(self, session_id: int | None = None) -> None:
        if session_id is None:
            session_id = self._active_session_id
        if session_id is None:
            self.memoryvla.reset_runtime_state()
            return
        self._session_memory_state.pop(session_id, None)
        if session_id == self._active_session_id:
            self.memoryvla.reset_runtime_state()

    def clear_session(self, session_id: int) -> None:
        self._session_memory_state.pop(session_id, None)
        if session_id == self._active_session_id:
            self._active_session_id = None
            self.memoryvla.reset_runtime_state()

    def _encode_prefix_hidden(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
    ) -> torch.Tensor:
        prefix_att_2d_masks = self.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        assert outputs is not None
        if isinstance(outputs, list):
            outputs = outputs[0]
        return outputs

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_hidden = self._encode_prefix_hidden(prefix_embs, prefix_pad_masks, prefix_att_masks)
        current_summary = prefix_hidden.mean(dim=1)
        # The MemoryVLA-specific modules are loaded/stored in float32, while the
        # backbone often runs in bfloat16 during eval. Run the memory branch in
        # the module dtype and cast back before concatenating with the prefix.
        memory_dtype = self.prefix_summary_proj.weight.dtype
        current_tokens = self.prefix_summary_proj(current_summary.to(dtype=memory_dtype)).unsqueeze(1)
        memory_tokens, gate = self.memoryvla(current_tokens, update_memory=not self.training)
        self._last_memory_gate = gate
        if self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.memoryvla.get_runtime_state()
        memory_tokens = self.memory_to_prefix_proj(memory_tokens).to(dtype=prefix_embs.dtype)

        extra_pad_masks = torch.ones(
            prefix_embs.shape[0],
            memory_tokens.shape[1],
            dtype=prefix_pad_masks.dtype,
            device=prefix_pad_masks.device,
        )
        extra_att_masks = torch.zeros(
            prefix_embs.shape[0],
            memory_tokens.shape[1],
            dtype=prefix_att_masks.dtype,
            device=prefix_att_masks.device,
        )
        prefix_embs = torch.cat([prefix_embs, memory_tokens], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, extra_pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, extra_att_masks], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks
