from typing import Any

import torch
from torch import nn

from openpi.models_pytorch.memory_baselines.hamlet_memory import HamletMemoryAdapter
from openpi.models_pytorch.memory_baselines.hamlet_memory import MomentTokenPool
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class Pi05WithHamlet(PI0Pytorch):
    """Pi0.5 backbone with HAMLET-style moment-token history memory."""

    def __init__(
        self,
        config,
        *,
        action_expert_name: str = "gemma_token",
        action_expert_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, action_expert_name=action_expert_name, action_expert_kwargs=action_expert_kwargs)
        feature_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.moment_token_pool = MomentTokenPool(config.hamlet_num_moment_tokens, feature_dim)
        self.prefix_summary_proj = nn.Linear(feature_dim, feature_dim)
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
        # Kept for checkpoint compatibility with older/newer MemoryVLA/HAMLET variants.
        self.prefix_summary_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.eye_(self.prefix_summary_proj.weight)
        nn.init.zeros_(self.prefix_summary_proj.bias)

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

    def _masked_prefix_summary(self, prefix_embs: torch.Tensor, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        weights = prefix_pad_masks.to(dtype=prefix_embs.dtype).unsqueeze(-1)
        denom = torch.clamp(weights.sum(dim=1), min=1.0)
        return (prefix_embs * weights).sum(dim=1) / denom

    def _encode_current_moment_tokens(self, prefix_embs: torch.Tensor, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        batch_size = prefix_embs.shape[0]
        base_moment_tokens = self.moment_token_pool(batch_size).to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        prefix_summary = self._masked_prefix_summary(prefix_embs, prefix_pad_masks)
        proj_dtype = self.prefix_summary_proj.weight.dtype
        summary_context = self.prefix_summary_proj(prefix_summary.to(dtype=proj_dtype)).to(dtype=prefix_embs.dtype).unsqueeze(1)
        return base_moment_tokens + summary_context

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks)
        current_moment_tokens = self._encode_current_moment_tokens(prefix_embs, prefix_pad_masks)
        # Keep the HAMLET memory branch in its module dtype to avoid bfloat16 /
        # float32 mismatches with the main backbone during eval.
        memory_dtype = self.memory_to_prefix_proj.weight.dtype
        memory_tokens = self.hamlet_memory(
            current_moment_tokens.to(dtype=memory_dtype),
            update_memory=not self.training,
        )
        if not self.training and self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.hamlet_memory.get_runtime_state()
        memory_tokens = self.memory_to_prefix_proj(memory_tokens).to(dtype=prefix_embs.dtype)

        batch_size = prefix_embs.shape[0]
        moment_input_tokens = current_moment_tokens.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
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
