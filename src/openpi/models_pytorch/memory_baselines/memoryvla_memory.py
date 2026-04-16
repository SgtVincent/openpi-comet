from typing import Any

import torch
from torch import nn


class SingleStreamMemoryBank(nn.Module):
    """Fixed-capacity single-stream memory bank with similarity-aware replacement."""

    def __init__(self, feature_dim: int, bank_capacity: int, similarity_threshold: float) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bank_capacity = bank_capacity
        self.similarity_threshold = similarity_threshold
        self.register_buffer("memory_bank", None, persistent=False)
        self.register_buffer("memory_count", torch.tensor(0), persistent=False)

    def reset_runtime_state(self) -> None:
        self.memory_bank = None
        self.memory_count = torch.tensor(0)

    def get_runtime_state(self) -> dict[str, Any]:
        return {
            "memory_bank": self.memory_bank,
            "memory_count": self.memory_count,
        }

    def set_runtime_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            self.reset_runtime_state()
            return
        self.memory_bank = state.get("memory_bank")
        self.memory_count = state.get("memory_count", torch.tensor(0))

    def _cosine_similarity(self, query: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        query = torch.nn.functional.normalize(query, dim=-1)
        memory_bank = torch.nn.functional.normalize(memory_bank, dim=-1)
        return torch.einsum("bd,bkd->bk", query, memory_bank)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor | None:
        if self.memory_bank is None or int(self.memory_count.item()) == 0:
            return None
        valid_bank = self.memory_bank[:, : int(self.memory_count.item())]
        sim = self._cosine_similarity(query, valid_bank)
        weights = torch.softmax(sim, dim=-1)
        return torch.einsum("bk,bkd->bd", weights, valid_bank)

    def update(self, item: torch.Tensor) -> None:
        item = item.detach()
        if self.memory_bank is None:
            self.memory_bank = item.unsqueeze(1)
            self.memory_count = torch.tensor(1, device=item.device)
            return
        if self.memory_bank.shape[0] != item.shape[0]:
            self.reset_runtime_state()
            self.memory_bank = item.unsqueeze(1)
            self.memory_count = torch.tensor(1, device=item.device)
            return

        count = int(self.memory_count.item())
        if count < self.bank_capacity:
            self.memory_bank = torch.cat([self.memory_bank[:, :count], item.unsqueeze(1)], dim=1)
            self.memory_count = torch.tensor(count + 1, device=item.device)
            return

        sim = self._cosine_similarity(item, self.memory_bank)
        best_idx = torch.argmax(sim, dim=-1)
        if torch.mean(torch.gather(sim, 1, best_idx.unsqueeze(1))).item() >= self.similarity_threshold:
            updated = self.memory_bank.clone()
            for batch_idx in range(updated.shape[0]):
                idx = int(best_idx[batch_idx].item())
                updated[batch_idx, idx] = 0.5 * (updated[batch_idx, idx] + item[batch_idx])
            self.memory_bank = updated
        else:
            self.memory_bank = torch.cat([self.memory_bank[:, 1:], item.unsqueeze(1)], dim=1)

    def get_memory_stats(self) -> dict[str, int]:
        return {
            "memory_count": int(self.memory_count.item()),
            "bank_capacity": self.bank_capacity,
        }


class GatedMemoryFusion(nn.Module):
    def __init__(self, feature_dim: int, gate_init: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.gate_bias = nn.Parameter(torch.full((feature_dim,), gate_init))

    def forward(self, current_tokens: torch.Tensor, retrieved_summary: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if retrieved_summary is None:
            gate = torch.zeros_like(current_tokens)
            return current_tokens, gate
        retrieved_tokens = retrieved_summary[:, None, :].expand_as(current_tokens)
        gate_logits = self.mlp(torch.cat([current_tokens, retrieved_tokens], dim=-1)) + self.gate_bias
        gate = torch.sigmoid(gate_logits)
        fused = gate * retrieved_tokens + (1.0 - gate) * current_tokens
        return fused, gate


class MemoryVLAModule(nn.Module):
    """Single-stream MemoryVLA-inspired external memory module."""

    def __init__(
        self,
        feature_dim: int,
        bank_capacity: int = 16,
        similarity_threshold: float = 0.7,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bank = SingleStreamMemoryBank(feature_dim, bank_capacity, similarity_threshold)
        self.fusion = GatedMemoryFusion(feature_dim, gate_init=gate_init)

    def reset_runtime_state(self) -> None:
        self.bank.reset_runtime_state()

    def get_runtime_state(self) -> dict[str, Any]:
        return self.bank.get_runtime_state()

    def set_runtime_state(self, state: dict[str, Any] | None) -> None:
        self.bank.set_runtime_state(state)

    def get_memory_stats(self) -> dict[str, int]:
        return self.bank.get_memory_stats()

    def encode_memory_item(self, prefix_hidden: torch.Tensor, lang_hidden: torch.Tensor | None = None) -> torch.Tensor:
        del lang_hidden
        return prefix_hidden.mean(dim=1)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor | None:
        return self.bank.retrieve(query)

    def forward(
        self,
        current_tokens: torch.Tensor,
        text_query: torch.Tensor | None = None,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del text_query
        current_summary = self.encode_memory_item(current_tokens)
        retrieved = self.retrieve(current_summary)
        fused_tokens, gate = self.fusion(current_tokens, retrieved)
        if update_memory:
            self.bank.update(current_summary)
        return fused_tokens, gate
