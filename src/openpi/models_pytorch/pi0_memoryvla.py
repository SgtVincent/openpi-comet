from typing import Any

import torch
from torch import nn
from torch.nn import functional

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
        self.prefix_summary_proj = nn.Linear(feature_dim, feature_dim)
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

    def _masked_prefix_summary(self, prefix_embs: torch.Tensor, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        prefix_embs = prefix_embs.float()
        weights = prefix_pad_masks.to(dtype=torch.float32).unsqueeze(-1)
        denom = torch.clamp(weights.sum(dim=1), min=1.0)
        return (prefix_embs * weights).sum(dim=1) / denom

    @staticmethod
    def _linear_fp32(layer: nn.Module, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(layer, nn.Linear):
            return layer(value.float())
        return functional.linear(
            value.float(),
            layer.weight.float(),
            layer.bias.float() if layer.bias is not None else None,
        )

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_summary = self._masked_prefix_summary(prefix_embs, prefix_pad_masks)
        # Run the numerically sensitive memory branch in fp32, then cast back
        # before concatenating with the backbone prefix tokens.
        with torch.autocast(device_type=prefix_embs.device.type, enabled=False):
            current_tokens = self._linear_fp32(self.prefix_summary_proj, prefix_summary).unsqueeze(1)
            memory_tokens, gate = self.memoryvla(current_tokens, update_memory=not self.training)
            memory_tokens = self._linear_fp32(self.memory_to_prefix_proj, memory_tokens)
        self._last_memory_gate = gate
        if not self.training and self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.memoryvla.get_runtime_state()
        memory_tokens = memory_tokens.to(dtype=prefix_embs.dtype)

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
