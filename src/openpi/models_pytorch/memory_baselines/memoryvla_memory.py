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
        self.register_buffer("memory_write_index", torch.tensor(0), persistent=False)

    def reset_runtime_state(self) -> None:
        self.memory_bank = None
        self.memory_count = torch.tensor(0)
        self.memory_write_index = torch.tensor(0)

    def get_runtime_state(self) -> dict[str, Any]:
        return {
            "memory_bank": self.memory_bank,
            "memory_count": self.memory_count,
            "memory_write_index": self.memory_write_index,
        }

    def set_runtime_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            self.reset_runtime_state()
            return
        self.memory_bank = state.get("memory_bank")
        self.memory_count = state.get("memory_count", torch.tensor(0))
        self.memory_write_index = state.get("memory_write_index", torch.tensor(0))

    def _ensure_memory_bank(self, item: torch.Tensor) -> None:
        batch_size, feature_dim = item.shape
        if (
            self.memory_bank is not None
            and self.memory_bank.shape[0] == batch_size
            and self.memory_bank.shape[1] == self.bank_capacity
            and self.memory_bank.shape[2] == feature_dim
            and self.memory_bank.device == item.device
            and self.memory_bank.dtype == item.dtype
        ):
            return
        self.memory_bank = torch.zeros(
            batch_size,
            self.bank_capacity,
            feature_dim,
            device=item.device,
            dtype=item.dtype,
        )
        self.memory_count = torch.tensor(0, device=item.device)
        self.memory_write_index = torch.tensor(0, device=item.device)

    def _cosine_similarity(self, query: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        query = torch.nn.functional.normalize(query, dim=-1)
        memory_bank = torch.nn.functional.normalize(memory_bank, dim=-1)
        return torch.einsum("bd,bkd->bk", query, memory_bank)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor | None:
        if self.memory_bank is None or int(self.memory_count.item()) == 0:
            return None
        # Snapshot the mutable bank before update(); autograd may need these values during backward.
        valid_bank = self.memory_bank[:, : int(self.memory_count.item())].detach().clone()
        sim = self._cosine_similarity(query, valid_bank)
        weights = torch.softmax(sim, dim=-1)
        return torch.einsum("bk,bkd->bd", weights, valid_bank)

    def update(self, item: torch.Tensor) -> None:
        item = item.detach()
        self._ensure_memory_bank(item)
        assert self.memory_bank is not None

        count = int(self.memory_count.item())
        if count < self.bank_capacity:
            self.memory_bank[:, count].copy_(item)
            self.memory_count = torch.tensor(count + 1, device=item.device)
            self.memory_write_index = torch.tensor((count + 1) % self.bank_capacity, device=item.device)
            return

        sim = self._cosine_similarity(item, self.memory_bank)
        best_idx = torch.argmax(sim, dim=-1)
        if torch.mean(torch.gather(sim, 1, best_idx.unsqueeze(1))).item() >= self.similarity_threshold:
            for batch_idx in range(self.memory_bank.shape[0]):
                idx = int(best_idx[batch_idx].item())
                self.memory_bank[batch_idx, idx].mul_(0.5).add_(item[batch_idx], alpha=0.5)
        else:
            write_index = int(self.memory_write_index.item())
            self.memory_bank[:, write_index].copy_(item)
            self.memory_write_index = torch.tensor((write_index + 1) % self.bank_capacity, device=item.device)

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
            zero_retrieved = torch.zeros_like(current_tokens)
            # Keep the same parameter-usage pattern across steps so DDP static_graph mode remains valid.
            gate_logits = self.mlp(torch.cat([current_tokens, zero_retrieved], dim=-1)) + self.gate_bias
            gate = torch.zeros_like(gate_logits)
            return current_tokens + gate_logits * 0.0, gate
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
